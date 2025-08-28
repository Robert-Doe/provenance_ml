#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extractor for provenance-ml
-----------------------------------
Reads weak + gold JSONL, merges labels (gold overrides), and emits:
  - data/features/X.npz          (scipy.sparse CSR matrix)
  - data/features/y.npy          (int labels: developer=0, user=1, unknown=2)
  - data/features/is_gold.npy    (bool mask)
  - data/features/ids.csv        (page_id,candidate_id)
  - data/features/feature_names.json
  - data/features/meta.json

Design goals:
  * Deterministic builds (fixed seeds, stable sorting)
  * Sparse representations (fast + memory efficient)
  * Clear metadata for reproducibility (label_map, seeds, checksums)
  * Easy to extend: add features inside the builders below

Usage:
  python tools/feature_extractor.py \
    --weak data/labeled/weak.jsonl \
    --gold data/labeled/gold.jsonl \
    --raw-root data/raw \
    --out-dir data/features \
    --hash-buckets-text 4096 \
    --hash-buckets-attr 2048 \
    --hash-buckets-char 1024 \
    --hash-seed 42
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp

# ------------------------------
# Constants & label taxonomy
# ------------------------------

LABELS = ["developer", "user", "unknown"]  # labels-v1 (3-class)
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Legacy coercions (read-time only; never write legacy values)
LEGACY_LABEL_COERCIONS = {
    "third_party": "unknown",
    "thirdparty": "unknown",
    "3p": "unknown",
}

# Regexes (compiled once) for counts/flags
RE_URL = re.compile(r"https?://[^\s'\"]+", re.IGNORECASE)
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_TIME_PHRASE = re.compile(
    r"\b(\d+\s*(min|mins|minute|minutes|hour|hours|hr|hrs|day|days)|yesterday|today|just now)\b",
    re.IGNORECASE,
)
RE_WEEKDAY = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)\w*\b", re.IGNORECASE)
RE_INLINE_HANDLER = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)
RE_SCRIPT_TAG = re.compile(r"<\s*script\b", re.IGNORECASE)
# Emojis are broad; a decent proxy uses Unicode ranges:
RE_EMOJI = re.compile(
    "["  # basic ranges; not exhaustive but good signal
    "\U0001F300-\U0001F6FF"  # Misc Symbols & Pictographs
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u27BF"          # Misc symbols
    "]"
)

# Attribute (class/id) heuristics
RE_USERISH = re.compile(r"(comment|reply|user|ugc|review|avatar|profile)", re.IGNORECASE)
RE_DEVISH = re.compile(r"(header|nav|navbar|footer|hero|admin|dashboard|template)", re.IGNORECASE)

# Tags we one-hot (keep small to avoid width blowup; others go to "other")
TAG_VOCAB = ["div", "section", "article", "p", "li", "span", "pre", "code", "button", "a", "img"]
TAG_INDEX = {t: i for i, t in enumerate(TAG_VOCAB)}
TAG_OTHER_INDEX = len(TAG_VOCAB)  # "other" bucket


# ------------------------------
# I/O utils
# ------------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")


# ------------------------------
# Tokenization & hashing
# ------------------------------

def tokenize(text: str, keep_min_len: int = 2) -> List[str]:
    """
    Deterministic tokenizer:
      * lowercases
      * splits on non-alphanumerics AND splits hyphen/underscore boundaries
      * drops tokens shorter than keep_min_len
    """
    if not text:
        return []
    s = text.lower().replace("-", " ").replace("_", " ")
    toks = re.split(r"[^a-z0-9]+", s)
    return [t for t in toks if len(t) >= keep_min_len]


def normalize_attr_token(s: str) -> List[str]:
    """
    Normalize class/id tokens:
      * lowercase
      * collapse digit runs to '#'
      * split on hyphen/underscore and non-alnum
    """
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"\d+", "#", s)
    s = s.replace("-", " ").replace("_", " ")
    toks = re.split(r"[^a-z#]+", s)
    return [t for t in toks if len(t) >= 2]


def char_ngrams(s: str, n_low: int = 3, n_high: int = 5) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = s.replace(" ", "")
    grams = []
    for n in range(n_low, n_high + 1):
        if len(s) < n:
            continue
        grams.extend([s[i : i + n] for i in range(len(s) - n + 1)])
    return grams


def murmurhash3_32(key: str, seed: int) -> int:
    """
    Deterministic 32-bit hash. For simplicity and portability, use Python's hashlib.md5 seeded,
    which is slower than a real murmur but acceptable for offline builds. We mix seed into key.
    """
    m = hashlib.md5()
    m.update(str(seed).encode("utf-8"))
    m.update(key.encode("utf-8"))
    # Convert first 4 bytes to unsigned int
    return int.from_bytes(m.digest()[:4], byteorder="little", signed=False)


def hashed_bucket(token: str, buckets: int, seed: int) -> int:
    return murmurhash3_32(token, seed) % buckets


# ------------------------------
# Safe scalar counters & flags
# ------------------------------

def clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def count_urls(s: str) -> int:
    return len(RE_URL.findall(s or ""))


def count_emails(s: str) -> int:
    return len(RE_EMAIL.findall(s or ""))


def count_mentions(s: str) -> int:
    return len(RE_MENTION.findall(s or ""))


def count_hashtags(s: str) -> int:
    return len(RE_HASHTAG.findall(s or ""))


def count_emojis(s: str) -> int:
    return len(RE_EMOJI.findall(s or ""))


def has_time_phrase(s: str) -> int:
    return int(bool(RE_TIME_PHRASE.search(s or "")))


def has_weekday(s: str) -> int:
    return int(bool(RE_WEEKDAY.search(s or "")))


def upper_ratio(s: str) -> float:
    if not s:
        return 0.0
    upp = sum(1 for ch in s if ch.isupper())
    return upp / max(1, len(s))


def digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    dig = sum(1 for ch in s if ch.isdigit())
    return dig / max(1, len(s))


def char_entropy(s: str) -> float:
    """Shannon entropy over characters; proxy for boilerplate vs UGC."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    # clip to a reasonable range
    return clip(ent, 0.0, 8.0)


# ------------------------------
# Merging weak + gold labels
# ------------------------------

@dataclass(frozen=True)
class Key:
    page_id: str
    candidate_id: str


@dataclass
class Row:
    key: Key
    text: str
    html_excerpt: str
    tag: Optional[str]  # best-effort; extractor can accept None
    classes: str        # space-separated class string, optional
    elem_id: str        # id attribute, optional
    dom_path: str
    # structural hints (optional, else defaults)
    depth: Optional[int] = None
    child_count: Optional[int] = None
    sibling_index: Optional[int] = None
    # runtime/dynamic (optional)
    inserted_via_api: Optional[bool] = None
    time_since_load_ms: Optional[float] = None
    mutation_count: Optional[int] = None
    # labels
    weak_label: Optional[str] = None
    weak_conf: Optional[float] = None
    gold_label: Optional[str] = None
    # resolved
    label: Optional[str] = None
    is_gold: bool = False


def coerce_label(lbl: Optional[str]) -> Optional[str]:
    if lbl is None:
        return None
    lbl = lbl.strip().lower()
    lbl = LEGACY_LABEL_COERCIONS.get(lbl, lbl)
    if lbl not in LABEL2ID:
        return None
    return lbl


def load_and_merge(weak_path: Path, gold_path: Optional[Path]) -> List[Row]:
    weak_map: Dict[Key, Row] = {}
    # Load weak
    for r in iter_jsonl(weak_path):
        key = Key(r.get("page_id", ""), r.get("candidate_id", ""))
        if not key.page_id or not key.candidate_id:
            continue
        row = Row(
            key=key,
            text=r.get("text", "") or "",
            html_excerpt=r.get("html_excerpt", "") or "",
            tag=(r.get("tag") or r.get("tag_name") or "").lower() or None,
            classes=r.get("class", r.get("classes", "")) or "",
            elem_id=r.get("id", r.get("elem_id", "")) or "",
            dom_path=r.get("dom_path", "") or "",
            depth=r.get("features", {}).get("depth") if isinstance(r.get("features"), dict) else r.get("depth"),
            child_count=r.get("features", {}).get("child_count") if isinstance(r.get("features"), dict) else r.get("child_count"),
            sibling_index=r.get("features", {}).get("sibling_index") if isinstance(r.get("features"), dict) else r.get("sibling_index"),
            inserted_via_api=r.get("features", {}).get("inserted_via_api") if isinstance(r.get("features"), dict) else r.get("inserted_via_api"),
            time_since_load_ms=r.get("features", {}).get("time_since_load_ms") if isinstance(r.get("features"), dict) else r.get("time_since_load_ms"),
            mutation_count=r.get("features", {}).get("mutation_count") if isinstance(r.get("features"), dict) else r.get("mutation_count"),
            weak_label=coerce_label(r.get("weak_label")),
            weak_conf=r.get("weak_conf"),
        )
        weak_map[key] = row

    # Overlay gold
    gold_map: Dict[Key, str] = {}
    if gold_path and gold_path.exists():
        for g in iter_jsonl(gold_path):
            key = Key(g.get("page_id", ""), g.get("candidate_id", ""))
            lbl = coerce_label(g.get("gold_label"))
            if key.page_id and key.candidate_id and lbl in LABEL2ID:
                gold_map[key] = lbl

    # Resolve labels
    rows: List[Row] = []
    for key, row in weak_map.items():
        if key in gold_map:
            row.label = gold_map[key]
            row.is_gold = True
        elif row.weak_label in LABEL2ID:
            row.label = row.weak_label
            row.is_gold = False
        else:
            # drop unlabeled
            continue
        rows.append(row)

    # Stable order for determinism
    rows.sort(key=lambda r: (r.key.page_id, r.key.candidate_id))
    return rows


# ------------------------------
# Feature builders
# ------------------------------

def build_structural(rows: List[Row]) -> Tuple[sp.csr_matrix, List[str]]:
    names: List[str] = []
    cols: List[dict] = []

    # scalar columns
    scalar_names = [
        "struct.depth",
        "struct.child_count",
        "struct.sibling_index",
        "struct.link_density",  # placeholder: can compute from html/text links
    ]
    names.extend(scalar_names)

    # tag one-hot (|TAG_VOCAB| + 1 for "other")
    tag_names = [f"struct.tag[{t}]" for t in TAG_VOCAB] + ["struct.tag[other]"]
    names.extend(tag_names)
    total_cols = len(names)

    data, indices, indptr = [], [], [0]
    for r in rows:
        row_map: Dict[int, float] = {}

        depth = 0 if r.depth is None else int(r.depth)
        child_count = 0 if r.child_count is None else int(r.child_count)
        sibling_index = 0 if r.sibling_index is None else int(r.sibling_index)

        # compute a naive link density from text (URLs / tokens)
        t = r.text or ""
        tok_n = max(1, len(tokenize(t)))
        link_density = count_urls(t) / tok_n

        vals = [
            clip(depth, 0, 50),
            clip(child_count, 0, 100),
            clip(sibling_index, 0, 200),
            clip(link_density, 0.0, 1.0),
        ]
        for i, v in enumerate(vals):
            if v != 0:
                row_map[i] = float(v)

        # tag one-hot
        tag_idx = TAG_OTHER_INDEX
        if r.tag and r.tag in TAG_INDEX:
            tag_idx = TAG_INDEX[r.tag]
        tag_col = len(scalar_names) + tag_idx
        row_map[tag_col] = row_map.get(tag_col, 0.0) + 1.0

        # finalize
        if row_map:
            cols_sorted = sorted(row_map.items())
            indices.extend([c for c, _ in cols_sorted])
            data.extend([v for _, v in cols_sorted])
        indptr.append(len(indices))

    X = sp.csr_matrix((data, indices, indptr), shape=(len(rows), total_cols), dtype=np.float32)
    return X, names


def build_textual(rows: List[Row], K: int, seed: int) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Word-level hashed BOW (+ scalar counts/flags).
    """
    scalar_names = [
        "text.char_len",
        "text.token_len",
        "text.avg_token_len",
        "text.num_urls",
        "text.num_emails",
        "text.num_mentions",
        "text.num_hashtags",
        "text.num_emojis",
        "text.num_exclamations",
        "text.upper_ratio",
        "text.digit_ratio",
        "text.has_time_phrase",
        "text.has_weekday",
        "text.char_entropy",
        "text.has_inline_handler",
        "text.contains_script_tag",
    ]
    hash_names = [f"text.hash[{i}]" for i in range(K)]
    names = scalar_names + hash_names
    offset = len(scalar_names)
    total_cols = len(names)

    data, indices, indptr = [], [], [0]
    for r in rows:
        s = (r.text or "").strip()
        toks = tokenize(s)
        tok_len = len(toks)

        avg_tok = (sum(len(t) for t in toks) / tok_len) if tok_len > 0 else 0.0

        row_map: Dict[int, float] = {
            0: float(clip(len(s), 0, 4000)),
            1: float(clip(tok_len, 0, 1000)),
            2: float(clip(avg_tok, 0, 30)),
            3: float(clip(count_urls(s), 0, 50)),
            4: float(clip(count_emails(s), 0, 50)),
            5: float(clip(count_mentions(s), 0, 200)),
            6: float(clip(count_hashtags(s), 0, 200)),
            7: float(clip(count_emojis(s), 0, 200)),
            8: float(clip(s.count("!"), 0, 50)),
            9: float(clip(upper_ratio(s), 0.0, 1.0)),
            10: float(clip(digit_ratio(s), 0.0, 1.0)),
            11: float(has_time_phrase(s)),
            12: float(has_weekday(s)),
            13: float(clip(char_entropy(s), 0.0, 8.0)),
            14: float(int(bool(RE_INLINE_HANDLER.search(r.html_excerpt or "")))),
            15: float(int(bool(RE_SCRIPT_TAG.search(r.html_excerpt or "")))),
        }

        # Hashed bag-of-words with signed hashing
        bucket_vals: Dict[int, float] = defaultdict(float)
        for tok in toks:
            h = murmurhash3_32(tok, seed)
            j = h % K
            signed = 1.0 if (h & 1) else -1.0
            bucket_vals[j] += signed

        for j, v in bucket_vals.items():
            row_map[offset + j] = row_map.get(offset + j, 0.0) + v

        # finalize
        if row_map:
            cols_sorted = sorted(row_map.items())
            indices.extend([c for c, _ in cols_sorted])
            data.extend([v for _, v in cols_sorted])
        indptr.append(len(indices))

    X = sp.csr_matrix((data, indices, indptr), shape=(len(rows), total_cols), dtype=np.float32)
    return X, names


def build_attributes(rows: List[Row], K_attr: int, seed_attr: int, K_char: int, seed_char: int) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Attribute-based features: regex flags + hashed tokens from class/id + char n-grams.
    Separate namespaces for word tokens and char-grams.
    """
    scalar_names = [
        "attr.has_userish_class",
        "attr.has_devish_class",
        "attr.class_count",
        "attr.has_id",
    ]
    hash_names = [f"attr.hash_tok[{i}]" for i in range(K_attr)]
    char_names = [f"attr.hash_char[{i}]" for i in range(K_char)]
    names = scalar_names + hash_names + char_names
    off_tok = len(scalar_names)
    off_char = off_tok + len(hash_names)
    total_cols = len(names)

    data, indices, indptr = [], [], [0]
    for r in rows:
        cls = r.classes or ""
        eid = r.elem_id or ""
        cls_tokens = []
        if cls:
            # Split class list and normalize each token further
            for token in cls.split():
                cls_tokens.extend(normalize_attr_token(token))
        id_tokens = normalize_attr_token(eid)

        # Scalar flags
        row_map: Dict[int, float] = {
            0: float(int(bool(RE_USERISH.search(cls)))),
            1: float(int(bool(RE_DEVISH.search(cls)))),
            2: float(clip(len(cls.split()), 0, 50)),
            3: float(int(bool(eid))),
        }

        # Hashed tokens (classes + id)
        bucket_vals: Dict[int, float] = defaultdict(float)
        for tok in cls_tokens + id_tokens:
            h = murmurhash3_32(tok, seed_attr)
            j = h % K_attr
            signed = 1.0 if (h & 1) else -1.0
            bucket_vals[j] += signed
        for j, v in bucket_vals.items():
            row_map[off_tok + j] = row_map.get(off_tok + j, 0.0) + v

        # Char n-grams hashed (for robustness to hyphens/plurals)
        grams = []
        if cls:
            grams.extend(char_ngrams(cls))
        if eid:
            grams.extend(char_ngrams(eid))
        char_vals: Dict[int, float] = defaultdict(float)
        for g in grams:
            h = murmurhash3_32(g, seed_char)
            j = h % K_char
            signed = 1.0 if (h & 1) else -1.0
            char_vals[j] += signed
        for j, v in char_vals.items():
            row_map[off_char + j] = row_map.get(off_char + j, 0.0) + v

        # finalize
        if row_map:
            cols_sorted = sorted(row_map.items())
            indices.extend([c for c, _ in cols_sorted])
            data.extend([v for _, v in cols_sorted])
        indptr.append(len(indices))

    X = sp.csr_matrix((data, indices, indptr), shape=(len(rows), total_cols), dtype=np.float32)
    return X, names


def build_dynamic(rows: List[Row]) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Optional runtime/dynamic features if present in weak outputs/snapshots.
    If absent, zeros. You can extend this to parse logs under raw-root if needed.
    """
    names = [
        "dyn.inserted_via_api",
        "dyn.time_since_load_bin[0]",  # 0..4 bins
        "dyn.time_since_load_bin[1]",
        "dyn.time_since_load_bin[2]",
        "dyn.time_since_load_bin[3]",
        "dyn.time_since_load_bin[4]",
        "dyn.mutation_count_capped",
    ]
    total_cols = len(names)

    data, indices, indptr = [], [], [0]
    for r in rows:
        row_map: Dict[int, float] = {}
        # inserted flag
        row_map[0] = float(int(bool(r.inserted_via_api)))

        # time bins (0..4 over 0..10s, >10s)
        ts = r.time_since_load_ms if r.time_since_load_ms is not None else -1
        if ts < 0:
            bin_idx = None
        else:
            # bins: [0-500], (500-1500], (1500-3000], (3000-10000], >10000
            if ts <= 500:
                bin_idx = 1
            elif ts <= 1500:
                bin_idx = 2
            elif ts <= 3000:
                bin_idx = 3
            elif ts <= 10000:
                bin_idx = 4
            else:
                bin_idx = 5  # will be mapped to index 5 but we only have up to 4 -> cap
        if bin_idx is not None:
            row_map[min(bin_idx, 5)] = 1.0  # we only have indices 1..5; cap to 5, but names only 1..4

        # mutation count
        mc = 0 if r.mutation_count is None else int(r.mutation_count)
        row_map[6] = float(clip(mc, 0, 100))

        # finalize
        cols = sorted(k for k, v in row_map.items() if v != 0)
        indices.extend(cols)
        data.extend([row_map[c] for c in cols])
        indptr.append(len(indices))

    # Fix shape: only 7 columns (indices 0..6)
    X = sp.csr_matrix((data, indices, indptr), shape=(len(rows), total_cols), dtype=np.float32)
    return X, names


# ------------------------------
# Assembly & saving
# ------------------------------

def encode_labels(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    y = np.empty(len(rows), dtype=np.int64)
    mask = np.zeros(len(rows), dtype=np.bool_)
    counts: Dict[str, int] = {l: 0 for l in LABELS}
    for i, r in enumerate(rows):
        lbl = r.label
        if lbl not in LABEL2ID:
            raise ValueError(f"Unexpected missing/invalid label for {r.key}")
        y[i] = LABEL2ID[lbl]
        mask[i] = bool(r.is_gold)
        counts[lbl] += 1
    return y, mask, counts


def hstack_safe(blocks: List[sp.csr_matrix]) -> sp.csr_matrix:
    blocks = [b for b in blocks if b is not None and b.shape[1] > 0]
    if not blocks:
        raise RuntimeError("No feature blocks to stack")
    X = sp.hstack(blocks, format="csr", dtype=np.float32)
    X.eliminate_zeros()
    return X


def save_ids_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "candidate_id"])
        for r in rows:
            w.writerow([r.key.page_id, r.key.candidate_id])


def build_meta(
    feature_names: List[str],
    counts: Dict[str, int],
    inputs_info: Dict[str, str],
    args: argparse.Namespace,
) -> dict:
    n_samples = sum(counts.values())
    meta = {
        "schema_version": "features-v1",
        "feature_extractor_version": "0.1.0",
        "build_time_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "hash_seed": int(args.hash_seed),
        "hash_buckets": {
            "text": int(args.hash_buckets_text),
            "attr": int(args.hash_buckets_attr),
            "char": int(args.hash_buckets_char),
        },
        "label_map": {
            "to_id": LABEL2ID,
            "to_name": {str(i): ID2LABEL[i] for i in sorted(ID2LABEL)},
            "taxonomy_version": "labels-v1",
        },
        "counts": {
            "n_samples": n_samples,
            "by_label_str": counts,
        },
        "inputs": inputs_info,
        "feature_names": feature_names,
    }
    return meta


# ------------------------------
# Main build
# ------------------------------

def build_features(
    weak_jsonl: Path,
    gold_jsonl: Optional[Path],
    raw_root: Optional[Path],
    out_dir: Path,
    hash_buckets_text: int = 4096,
    hash_buckets_attr: int = 2048,
    hash_buckets_char: int = 1024,
    hash_seed: int = 42,
) -> Dict[str, str]:
    # Load & merge
    rows = load_and_merge(weak_jsonl, gold_jsonl)

    # Encode labels
    y, is_gold, counts = encode_labels(rows)

    # Build feature blocks
    X_struct, names_struct = build_structural(rows)
    X_text, names_text = build_textual(rows, K=hash_buckets_text, seed=hash_seed + 101)
    X_attr, names_attr = build_attributes(
        rows,
        K_attr=hash_buckets_attr,
        seed_attr=hash_seed + 202,
        K_char=hash_buckets_char,
        seed_char=hash_seed + 303,
    )
    X_dyn, names_dyn = build_dynamic(rows)

    # Assemble
    X = hstack_safe([X_struct, X_text, X_attr, X_dyn])
    feature_names = names_struct + names_text + names_attr + names_dyn

    # Sanity checks
    assert X.shape[0] == len(rows) == y.shape[0] == is_gold.shape[0], "Row count mismatch"
    if X.nnz == 0:
        raise RuntimeError("Feature matrix is empty (nnz=0)")
    if not np.isfinite(X.data).all():
        raise RuntimeError("Found non-finite values in feature matrix")
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise RuntimeError("Labels outside expected set {0,1,2}")

    # Prepare output paths
    out_dir.mkdir(parents=True, exist_ok=True)
    pX = out_dir / "X.npz"
    py = out_dir / "y.npy"
    pg = out_dir / "is_gold.npy"
    pids = out_dir / "ids.csv"
    pnames = out_dir / "feature_names.json"
    pmeta = out_dir / "meta.json"

    # Save artifacts
    sp.save_npz(pX, X)
    np.save(py, y)
    np.save(pg, is_gold)
    save_ids_csv(pids, rows)
    write_json(pnames, feature_names)

    # Input checksums (for reproducibility)
    inputs_info = {
        "weak_jsonl": str(weak_jsonl),
        "weak_jsonl_sha256": sha256_of_file(weak_jsonl),
    }
    if gold_jsonl and gold_jsonl.exists():
        inputs_info.update({
            "gold_jsonl": str(gold_jsonl),
            "gold_jsonl_sha256": sha256_of_file(gold_jsonl),
        })

    # Write meta.json
    meta = build_meta(feature_names, counts, inputs_info, argparse.Namespace(
        hash_seed=hash_seed,
        hash_buckets_text=hash_buckets_text,
        hash_buckets_attr=hash_buckets_attr,
        hash_buckets_char=hash_buckets_char,
    ))
    write_json(pmeta, meta)

    return {
        "X": str(pX),
        "y": str(py),
        "is_gold": str(pg),
        "ids": str(pids),
        "feature_names": str(pnames),
        "meta": str(pmeta),
        "n_samples": str(X.shape[0]),
        "n_features": str(X.shape[1]),
        "nnz": str(X.nnz),
    }


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build sparse features for provenance classification")
    ap.add_argument("--weak", type=Path, required=True, help="Path to weak.jsonl")
    ap.add_argument("--gold", type=Path, required=False, help="Path to gold.jsonl (optional)")
    ap.add_argument("--raw-root", type=Path, required=False, help="Root of data/raw (optional; reserved)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (e.g., data/features)")
    ap.add_argument("--hash-buckets-text", type=int, default=4096, help="Buckets for text hashed BOW")
    ap.add_argument("--hash-buckets-attr", type=int, default=2048, help="Buckets for class/id tokens")
    ap.add_argument("--hash-buckets-char", type=int, default=1024, help="Buckets for char n-grams (attrs)")
    ap.add_argument("--hash-seed", type=int, default=42, help="Base hash seed (namespaces add offsets)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    outputs = build_features(
        weak_jsonl=args.weak,
        gold_jsonl=args.gold,
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        hash_buckets_text=args.hash_buckets_text,
        hash_buckets_attr=args.hash_buckets_attr,
        hash_buckets_char=args.hash_buckets_char,
        hash_seed=args.hash_seed,
    )
    # Minimal console summary
    print(json.dumps({
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
