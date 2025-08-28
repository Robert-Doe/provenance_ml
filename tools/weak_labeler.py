#!/usr/bin/env python3
"""
Assign weak (heuristic) provenance labels to DOM candidates and write a JSONL file.

Usage:
  python tools/weak_labeler.py --in data/raw/synth --out data/labeled/synth_weak.jsonl
  # Optional flags:
  #   --p-user-hi 0.7   threshold above which we call it 'user'
  #   --p-user-lo 0.3   threshold below which we call it 'developer'
  #   --min-text  8     minimum visible text length to consider text-based cues
  #   --explain         include per-candidate feature contributions in the JSONL
"""
import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Any, List, Tuple

# ---------------------------
# Heuristic patterns (tweak here)
# ---------------------------

# Signals that push a block toward USER provenance
USER_CLASS_ID_PAT = re.compile(
    r"(comment|reply|review|rating|thread|post|message|ugc|user|author)s?",
    re.I,
)
REL_TIME_PAT      = re.compile(r"\b\d+\s+(minutes?|hours?|days?)\s+ago\b", re.I)
HAS_URL_PAT       = re.compile(r"https?://", re.I)

# Tightened mention/hashtag patterns (avoid matching emails or things like C#)
HAS_MENTION_PAT   = re.compile(r"(^|[^@\w])@([A-Za-z0-9_]{3,})\b", re.I)
HAS_HASHTAG_PAT   = re.compile(r"(^|[^#\w])#([A-Za-z0-9_]{3,})\b", re.I)

# Improved emoji coverage: include common astral-plane emoji + BMP dingbats
EMOJI_PAT         = re.compile(
    r"[\U0001F1E6-\U0001F1FF"   # flags
    r"\U0001F300-\U0001F5FF"    # symbols & pictographs
    r"\U0001F600-\U0001F64F"    # emoticons
    r"\U0001F680-\U0001F6FF"    # transport & map
    r"\U0001F700-\U0001F77F"    # alchemical
    r"\u2600-\u27BF]"           # miscellaneous symbols, dingbats
)

# Signals that push a block toward DEVELOPER provenance
DEV_CLASS_ID_PAT  = re.compile(
    r"(header|nav|footer|sidebar|menu|breadcrumbs|cookie|consent|legal|hero|promo)",
    re.I,
)

# Extended ARIA roles beyond the original three
DEV_ROLE_PAT      = re.compile(
    r'role\s*=\s*"(navigation|banner|contentinfo|complementary|search|menu|toolbar)"',
    re.I,
)

# API endpoints likely to return UGC
UGC_ENDPOINT_PAT  = re.compile(r"/api/(comments?|reviews?|search|posts?)", re.I)

# ---------------------------
# Feature extraction
# ---------------------------

def user_feature_contribs(
    cand: Dict[str, Any],
    recent_urls: List[str],
    recent_inserts: List[str],
    min_text: int,
) -> List[Tuple[str, float]]:
    """Return a list of (feature_name, weight) for USER-ish evidence."""
    feats: List[Tuple[str, float]] = []
    clsid = f"{cand.get('class','')} #{cand.get('id','')}"
    text  = (cand.get('text') or "")

    # Class/ID semantics
    if USER_CLASS_ID_PAT.search(clsid):
        feats.append(("class_id_semantics", 0.30))

    # Text cues (only if non-trivial)
    if len(text) >= min_text:
        if REL_TIME_PAT.search(text):    feats.append(("relative_time", 0.15))
        if HAS_URL_PAT.search(text):     feats.append(("has_url", 0.10))
        if HAS_MENTION_PAT.search(text): feats.append(("mention", 0.05))
        if HAS_HASHTAG_PAT.search(text): feats.append(("hashtag", 0.05))
        if EMOJI_PAT.search(text):       feats.append(("emoji", 0.05))

    # SPA insertion + matching UGC endpoint shortly before
    if any(UGC_ENDPOINT_PAT.search(u) for u in recent_urls) and \
       any(v in ("innerHTML", "appendChild") for v in recent_inserts):
        feats.append(("spa_ugc_insert", 0.25))

    # Small leaf-ish blocks with content lean user-ish (note: childCount = number of children)
    child_count = cand.get("childCount", 0)
    if child_count <= 3 and len(text) >= min_text:
        feats.append(("small_leafish", 0.05))

    return feats


def dev_feature_contribs(cand: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return a list of (feature_name, weight) for DEVELOPER-ish evidence."""
    feats: List[Tuple[str, float]] = []
    clsid = f"{cand.get('class','')} #{cand.get('id','')}"

    if DEV_CLASS_ID_PAT.search(clsid):
        feats.append(("class_id_chrome", 0.30))

    tag = (cand.get("tag") or "").upper()
    if tag in ("HEADER", "NAV", "FOOTER", "ASIDE"):
        feats.append(("chrome_tag", 0.25))

    # Very container-ish but with little text → often layout/template
    text = (cand.get("text") or "")
    if cand.get("childCount", 0) >= 8 and len(text) < 40:
        feats.append(("containerish", 0.10))

    # Raw attribute string if present in outer_html to catch ARIA roles
    outer = cand.get("outer_html") or ""
    if DEV_ROLE_PAT.search(outer):
        feats.append(("aria_role", 0.10))

    return feats


# ---------------------------
# Scoring & classification
# ---------------------------

def combine_scores(s_user: float, s_dev: float) -> float:
    """
    Turn (s_user, s_dev) into a pseudo-probability p_user in [0,1].
    Simple, monotonic mapping; can later be replaced by a learned calibrator.
    """
    p_user = 0.5 + (s_user - s_dev)
    return 0.0 if p_user < 0.0 else 1.0 if p_user > 1.0 else p_user


def classify(p_user: float, hi: float, lo: float) -> str:
    if p_user >= hi: return "user"
    if p_user <= lo: return "developer"
    return "unknown"


# ---------------------------
# I/O and CLI
# ---------------------------

def read_bundles(in_dir: pathlib.Path) -> list[dict]:
    files = sorted(in_dir.glob("*.json"))
    bundles = []
    for fp in files:
        text = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = fp.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            print(f"[warn] skipping {fp}: could not decode with utf-8/utf-8-sig/cp1252/latin-1")
            continue
        try:
            bundles.append(json.loads(text))
        except Exception as e:
            print(f"[warn] skipping {fp}: JSON parse error: {e}")
    return bundles



def main():
    ap = argparse.ArgumentParser(description="Assign weak provenance labels to DOM candidates.")
    ap.add_argument("--in",  dest="in_dir", required=True, help="Input bundle directory, e.g., data/raw/synth")
    ap.add_argument("--out", required=True, help="Output JSONL path, e.g., data/labeled/synth_weak.jsonl")
    ap.add_argument("--p-user-hi", type=float, default=0.7, help="Threshold above which we output 'user'")
    ap.add_argument("--p-user-lo", type=float, default=0.3, help="Threshold below which we output 'developer'")
    ap.add_argument("--min-text",  type=int,   default=8,   help="Minimum text length before using text cues")
    ap.add_argument("--explain", action="store_true", help="Emit per-candidate feature contributions")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_fp = pathlib.Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    bundles = read_bundles(in_dir)
    n_written = 0

    with out_fp.open("w", encoding="utf-8") as out:
        for bundle in bundles:
            site_id   = bundle.get("site_id")
            url       = bundle.get("url")
            net_log   = bundle.get("net_log", [])
            inserts   = bundle.get("runtime_inserts", [])
            recent_urls    = [e.get("url", "") for e in net_log[-25:]]
            recent_inserts = [e.get("via", "") for e in inserts[-50:]]

            for cand in bundle.get("candidates", []):
                # Extract features and scores
                u_feats = user_feature_contribs(cand, recent_urls, recent_inserts, min_text=args.min_text)
                d_feats = dev_feature_contribs(cand)
                s_u = sum(w for _, w in u_feats)
                s_d = sum(w for _, w in d_feats)

                # Combine and classify
                p_u = combine_scores(s_u, s_d)
                wl  = classify(p_u, hi=args.p_user_hi, lo=args.p_user_lo)

                rec = {
                    "site_id": site_id,
                    "url": url,
                    **cand,
                    "weak_label": wl,
                    "weak_conf": round(float(p_u if wl == "user" else (1.0 - p_u) if wl == "developer" else 0.5), 3),
                }

                if args.explain:
                    rec["explain"] = {
                        "user_feats": u_feats,
                        "dev_feats": d_feats,
                        "s_user": round(s_u, 3),
                        "s_dev": round(s_d, 3),
                        "p_user": round(p_u, 3),
                    }

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[ok] wrote {n_written} candidates with weak labels → {out_fp}")

if __name__ == "__main__":
    main()
