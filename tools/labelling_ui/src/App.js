import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Provenance Labeling UI — Per-annotator (all-time), no session confusion
 * ----------------------------------------------------------------------
 * - Progress is per annotator, counts everything you've labeled (ever) with this ID
 * - Clear status for current candidate: Labeled by you | Not labeled
 * - Progress bars: current filter (site or "All sites") + all sites
 * - Jump to next unlabeled (button + hotkey J)
 * - JSONL or JSON array loader with tolerant field mapping
 * - Keyboard: 1/2/3 labels, S save, N next, P prev, K skip, J next unlabeled, [ ] confidence
 */

const LABELS = ["developer", "user", "unknown"];

// ------------------ utils ------------------
function smallHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) { h = (h << 5) - h + str.charCodeAt(i); h |= 0; }
  return (h >>> 0).toString(16);
}
function formatUTC(d = new Date()) {
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().replace(/\.\d{3}Z$/, "Z");
}
function downloadText(filename, text) {
  const blob = new Blob([text], { type: "application/jsonl;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

// ------------------ parsing & mapping ------------------
function coerceLegacyLabel(lbl) {
  if (!lbl) return undefined;
  const x = String(lbl).toLowerCase().trim();
  if (x === "third_party" || x === "thirdparty" || x === "3p") return "unknown"; // collapsed
  return LABELS.includes(x) ? x : undefined;
}
function parseJSONLOrArray(text) {
  const raw = text.trim();
  if (raw.startsWith("[") && raw.endsWith("]")) {
    try { const arr = JSON.parse(raw); return Array.isArray(arr) ? arr : []; } catch { /* fallthrough */ }
  }
  const out = [];
  for (const line of raw.split(/\r?\n/)) {
    const t = line.trim(); if (!t) continue;
    try { out.push(JSON.parse(t)); } catch { /* skip bad line */ }
  }
  return out;
}
function deepGet(obj, keys) {
  for (const k of keys) {
    if (!obj) continue;
    if (Object.prototype.hasOwnProperty.call(obj, k)) return obj[k];
    if (k.includes(".")) {
      const parts = k.split("."); let v = obj;
      for (const p of parts) { if (!v || !Object.prototype.hasOwnProperty.call(v, p)) { v = undefined; break; } v = v[p]; }
      if (v !== undefined) return v;
    }
  }
  return undefined;
}
function makePageId(r) {
  const pid = deepGet(r, ["page_id","pageId","pageID","page","source_page_id"]) ?? "";
  const url = deepGet(r, ["url","page_url","pageUrl","page.url","request.url"]) ?? "";
  const site = r.site_id || r.site || "";
  if (pid && typeof pid === "string") return pid;
  if (typeof url === "string" && url) {
    try {
      const u = new URL(url);
      const domainPath = `${u.hostname}_${u.pathname || "/"}`.replace(/[?#].*$/, "");
      return site ? `${site}__${domainPath}` : domainPath;
    } catch {}
  }
  if (site) return String(site);
  const basis = (r.text || r.html || r.outer_html || r.dom_path || r.path || "").slice(0, 256);
  return `auto_page_${smallHash(basis || JSON.stringify(r).slice(0, 256))}`;
}
function makeCandId(r) {
  const cid = deepGet(r, ["candidate_id","candidateId","cand_id","node_id","nodeId","dom_path","path","xpath","id"]);
  if (typeof cid === "string" && cid.trim()) return cid.trim();
  const tag = (r.tag || r.tag_name || "").toLowerCase();
  const id = (r.id || "").trim();
  const cls = (r.class || r.className || "").trim().replace(/\s+/g, ".");
  if (tag && id) return `${tag}#${id}`;
  if (tag && cls) return `${tag}.${cls}`;
  if (tag) return `${tag}@d${r.depth ?? ""}_c${r.childCount ?? r.child_count ?? ""}`;
  const basis = (r.outer_html || r.html || r.text || "").slice(0, 128);
  return `cand_${smallHash(basis)}`;
}

// ------------------ styles ------------------
const S = {
  page: { fontFamily: "Inter, system-ui, sans-serif", background: "#f7f7f7", color: "#111", minHeight: "100vh" },
  wrap: { maxWidth: 1100, margin: "0 auto", padding: 16 },
  header: { display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", marginBottom: 12 },
  h1: { fontSize: 22, fontWeight: 600, margin: 0 },
  card: { background: "#fff", border: "1px solid #e5e7eb", borderRadius: 16, boxShadow: "0 1px 2px rgba(0,0,0,0.04)" },
  cardPad: { padding: 12 },
  btn: { padding: "10px 12px", borderRadius: 12, border: "1px solid #ddd", background: "#fff", cursor: "pointer" },
  btnPri: { padding: "10px 12px", borderRadius: 12, border: "1px solid #059669", background: "#059669", color: "#fff", cursor: "pointer" },
  tag: { padding: "4px 8px", background: "#f3f4f6", borderRadius: 8, fontSize: 12 },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  grid3: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 },
  small: { fontSize: 12, color: "#555" },
  mono: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace" },
  progressOuter: { height: 8, width: "100%", background: "#e5e7eb", borderRadius: 999, overflow: "hidden" },
  progressInner: (pct, color = "#10b981") => ({ height: "100%", width: `${pct}%`, background: color }),
  pill: (bg, fg) => ({ padding: "4px 8px", borderRadius: 999, fontSize: 12, background: bg, color: fg, border: "1px solid rgba(0,0,0,0.06)" }),
};
function LabelBtn({ value, selected, onClick, hotkey }) {
  const on = selected ? { background: "#059669", color: "#fff", borderColor: "#059669" } : { background: "#f6f6f6", color: "#111", borderColor: "#ddd" };
  return (<button onClick={onClick} style={{ ...S.btn, ...on }}>{value}<div style={{ fontSize: 11, opacity: 0.7, marginTop: 4 }}>{hotkey}</div></button>);
}

// ------------------ main ------------------
export default function App() {
  const [weakRows, setWeakRows] = useState([]);
  const [queue, setQueue] = useState([]); // indices into weakRows (respects filter/sampling)
  const [idx, setIdx] = useState(0);

  const [annotator, setAnnotator] = useState("");
  const storageKey = useMemo(() => `goldRows:${annotator || "anon"}`, [annotator]);
  const [goldRows, setGoldRows] = useState([]);

  const [notes, setNotes] = useState("");
  const [conf, setConf] = useState(0.9);
  const [label, setLabel] = useState(null);
  const [sampling, setSampling] = useState("sequential");
  const [siteFilter, setSiteFilter] = useState("all");
  const [uiVersion] = useState("0.6.0");
  const iframeRef = useRef(null);
  const [toast, setToast] = useState("");

  // toasts
  useEffect(() => { if (!toast) return; const t = setTimeout(() => setToast(""), 2000); return () => clearTimeout(t); }, [toast]);

  // load/persist per-annotator labels
  useEffect(() => {
    try { const saved = localStorage.getItem(storageKey); setGoldRows(saved ? JSON.parse(saved) : []); }
    catch { setGoldRows([]); }
  }, [storageKey]);
  useEffect(() => { localStorage.setItem(storageKey, JSON.stringify(goldRows)); }, [goldRows, storageKey]);

  // keyboard
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "1") setLabel("developer");
      if (e.key === "2") setLabel("user");
      if (e.key === "3") setLabel("unknown");
      if (e.key.toLowerCase() === "s") handleSave();
      if (e.key.toLowerCase() === "n") nextItem();
      if (e.key.toLowerCase() === "p") prevItem();
      if (e.key.toLowerCase() === "k") skipItem();
      if (e.key.toLowerCase() === "j") jumpToNextUnlabeled();
      if (e.key === "]") setConf((c) => Math.min(1, Math.round((c + 0.05) * 100) / 100));
      if (e.key === "[") setConf((c) => Math.max(0.5, Math.round((c - 0.05) * 100) / 100));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [label, conf, idx, queue, weakRows, annotator, notes, goldRows]);

  // build queue (filter + sampling)
  useEffect(() => {
    if (!weakRows.length) { setQueue([]); setIdx(0); return; }
    let indices = weakRows.map((_, i) => i);
    if (siteFilter !== "all") indices = indices.filter(i => weakRows[i]._siteKey === siteFilter);
    if (sampling === "uncertainty") {
      indices.sort((a, b) =>
        Math.abs((weakRows[a].weak_conf ?? 0.5) - 0.5) -
        Math.abs((weakRows[b].weak_conf ?? 0.5) - 0.5));
    }
    setQueue(indices);
    setIdx(0);
    setLabel(null);
    setNotes("");
  }, [weakRows, sampling, siteFilter]);

  const current = useMemo(() => {
    if (!queue.length) return null;
    const i = queue[Math.min(idx, queue.length - 1)];
    return weakRows[i] || null;
  }, [queue, idx, weakRows]);

  // sites list
  const sites = useMemo(() => ["all", ...Array.from(new Set(weakRows.map(r => r._siteKey))).sort()], [weakRows]);

  // iframe preview
  useEffect(() => {
    if (!current || !iframeRef.current) return;
    const doc = iframeRef.current.contentDocument; if (!doc) return;
    doc.open();
    const style = `
      <style>
        body { font-family: system-ui, sans-serif; padding: 12px; line-height: 1.45; color: #111; }
        .hl { outline: 3px solid #10b981; border-radius: 8px; padding: 6px; }
      </style>`;
    const safe = (current.html_excerpt || "")
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/\son[a-z]+\s*=/gi, " ");
    doc.write(`<!doctype html><html><head>${style}</head><body><div class="hl">${safe || "<i>(empty)</i>"}</div></body></html>`);
    doc.close();
  }, [current]);

  // gold keys (by you, all-time)
  const goldKeys = useMemo(() => new Set(goldRows.map(r => r.page_id + "\u0000" + r.candidate_id)), [goldRows]);

  // progress (by you, all-time)
  const filterProgress = useMemo(() => {
    const total = queue.length;
    let done = 0;
    for (const i of queue) {
      const k = weakRows[i].page_id + "\u0000" + weakRows[i].candidate_id;
      if (goldKeys.has(k)) done++;
    }
    return { done, total, pct: total ? Math.round((done / total) * 100) : 0 };
  }, [queue, weakRows, goldKeys]);

  const allProgress = useMemo(() => {
    const total = weakRows.length;
    let done = 0;
    for (const r of weakRows) {
      const k = r.page_id + "\u0000" + r.candidate_id;
      if (goldKeys.has(k)) done++;
    }
    return { done, total, pct: total ? Math.round((done / total) * 100) : 0 };
  }, [weakRows, goldKeys]);

  const allDone = useMemo(() => filterProgress.total > 0 && filterProgress.done === filterProgress.total, [filterProgress]);

  // ------------------ actions ------------------
  function normalizeRows(rows) {
    let kept = 0, dropped = 0;
    const norm = rows.map(r => {
      const page_id = makePageId(r);
      const candidate_id = makeCandId(r);
      let siteKey = "";
      try { const url = deepGet(r, ["url","page_url","pageUrl","page.url"]); if (url) siteKey = new URL(url).hostname; } catch {}
      if (!siteKey) siteKey = (r.site_id || r.site || "").toString() || page_id.split("__")[0];
      const obj = {
        page_id, candidate_id, _siteKey: siteKey,
        text: r.text || "",
        html_excerpt: r.html || r.outer_html || "",
        weak_label: coerceLegacyLabel(r.weak_label),
        weak_conf: typeof r.weak_conf === "number" ? r.weak_conf : (typeof r.confidence === "number" ? r.confidence : undefined),
        dom_path: r.path || r.dom_path || r.xpath || "",
        tag: (r.tag || r.tag_name || "").toLowerCase(),
        classes: r.class || r.classes || "",
        id: r.id || r.elem_id || "",
        depth: r.depth ?? r.features?.depth,
        childCount: r.childCount ?? r.child_count ?? r.features?.child_count,
      };
      const ok = Boolean(obj.page_id && obj.candidate_id);
      ok ? kept++ : dropped++;
      return ok ? obj : null;
    }).filter(Boolean);
    console.log(`Loaded ${rows.length} records; kept ${kept}, dropped ${dropped}.`);
    if (kept === 0) { console.warn("Example raw row:", rows[0]); alert("Loaded the file but normalized 0 rows.\nCheck console for the first raw row to adjust field mappings."); }
    else { console.log("Sample normalized rows:", norm.slice(0, 2)); }
    return norm;
  }

  function handleWeakUpload(file) {
    const reader = new FileReader();
    reader.onerror = () => { alert("Could not read file."); };
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : new TextDecoder("utf-8").decode(reader.result);
      const rows = parseJSONLOrArray(text);
      const norm = normalizeRows(rows);
      setWeakRows(norm);
      setIdx(0); setLabel(null); setNotes(""); setToast("");
    };
    reader.readAsText(file);
  }

  function handleSave() {
    if (!current) return;
    if (!annotator) { alert("Please set your Annotator ID first."); return; }
    if (!label) { alert("Pick a label (1/2/3) before saving."); return; }
    const row = {
      page_id: current.page_id,
      candidate_id: current.candidate_id,
      gold_label: label,
      gold_conf: Math.max(0.5, Math.min(1.0, Number(conf))),
      annotator_id: annotator,
      notes: notes || undefined,
      created_at: formatUTC(),
      ui_version: uiVersion,
    };
    setGoldRows((prev) => {
      const key = row.page_id + "\u0000" + row.candidate_id;
      const map = new Map(prev.map((r) => [r.page_id + "\u0000" + r.candidate_id, r]));
      map.set(key, row);
      return Array.from(map.values());
    });
    if (allDone) setToast("All candidates in this filter are labeled. You can download gold.jsonl.");
    nextItem();
  }

  function nextItem() {
    if (idx >= Math.max(0, queue.length - 1)) {
      setToast(allDone ? "All candidates in this filter are labeled. You can download gold.jsonl." : "End of list.");
      return;
    }
    setIdx((i) => Math.min(i + 1, Math.max(0, queue.length - 1)));
    setLabel(null); setNotes("");
  }
  function prevItem() {
    setIdx((i) => Math.max(0, i - 1));
    setLabel(null); setNotes("");
  }
  function skipItem() { nextItem(); }

  function jumpToNextUnlabeled() {
    if (!queue.length) return;
    const start = idx;
    for (let step = 1; step <= queue.length; step++) {
      const j = (start + step) % queue.length;
      const r = weakRows[queue[j]];
      const key = r.page_id + "\u0000" + r.candidate_id;
      if (!goldKeys.has(key)) {
        setIdx(j); setLabel(null); setNotes(""); return;
      }
    }
    setToast("No unlabeled items in this filter.");
  }

  function handleDownload() {
    if (!goldRows.length) { alert("No labels yet."); return; }
    const jsonl = goldRows.map((r) => JSON.stringify(r)).join("\n") + "\n";
    const fname = `gold_${annotator || "anon"}_${formatUTC().replace(/[:T-]/g, "").slice(0, 14)}.jsonl`;
    downloadText(fname, jsonl);
  }

  // current status (by you, all-time)
  const currentKey = current ? current.page_id + "\u0000" + current.candidate_id : "";
  const labeledByYou = current ? goldKeys.has(currentKey) : false;

  // ------------------ render ------------------
  return (
    <div style={S.page}>
      <div style={S.wrap}>
        {/* Header */}
        <div style={S.header}>
          <div>
            <h1 style={S.h1}>Provenance Labeling UI</h1>
            <div style={S.small}>
              Labels: <b>developer</b>, <b>user</b>, <b>unknown</b>. Hotkeys: <b>1/2/3</b>, <b>S</b>=save, <b>N</b>=next, <b>P</b>=prev, <b>K</b>=skip, <b>J</b>=next unlabeled, <b>[</b>/<b>]</b> confidence.
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ ...S.btn, background: "#fff", borderStyle: "dashed" }}>
              <input type="file" accept=".jsonl,.json,.txt" hidden onChange={(e) => e.target.files?.[0] && handleWeakUpload(e.target.files[0])} />
              Load weak.jsonl
            </label>
            <input placeholder="Annotator ID" value={annotator} onChange={(e) => setAnnotator(e.target.value)} style={{ ...S.btn, width: 160 }} />
            <select value={sampling} onChange={(e) => setSampling(e.target.value)} style={S.btn}>
              <option value="sequential">Sampling: Sequential</option>
              <option value="uncertainty">Sampling: Uncertainty first</option>
            </select>
            <select value={siteFilter} onChange={(e) => setSiteFilter(e.target.value)} style={S.btn}>
              {sites.map((s) => <option key={s} value={s}>{s === "all" ? "All sites" : `Site: ${s}`}</option>)}
            </select>
            <button style={S.btnPri} onClick={handleDownload}>Download gold.jsonl</button>
            {weakRows.length > 0 && (<span style={S.tag}>Loaded: <b>{weakRows.length}</b></span>)}
          </div>
        </div>

        {/* Toast */}
        {toast && (
          <div style={{ ...S.card, marginBottom: 12, borderColor: "#bfdbfe" }}>
            <div style={{ ...S.cardPad, background: "#eff6ff", borderRadius: 16 }}>
              <div style={S.small}>{toast}</div>
            </div>
          </div>
        )}

        {/* Progress (Current Filter) */}
        <div style={{ ...S.card, marginBottom: 8 }}>
          <div style={S.cardPad}>
            <div style={{ marginBottom: 6, fontWeight: 600 }}>
              Progress — {siteFilter === "all" ? "All sites (filter)" : `Site: ${siteFilter}`} <span style={{ marginLeft: 8, ...S.small }}>(by you)</span>
            </div>
            <div style={S.progressOuter}><div style={S.progressInner(filterProgress.pct, "#10b981")} /></div>
            <div style={{ ...S.small, marginTop: 6 }}>
              {filterProgress.done} / {filterProgress.total} ({filterProgress.pct}%)
            </div>
          </div>
        </div>

        {/* Progress (All Sites) */}
        <div style={{ ...S.card, marginBottom: 12 }}>
          <div style={S.cardPad}>
            <div style={{ marginBottom: 6, fontWeight: 600 }}>
              Progress — All sites <span style={{ marginLeft: 8, ...S.small }}>(by you)</span>
            </div>
            <div style={S.progressOuter}><div style={S.progressInner(allProgress.pct, "#3b82f6")} /></div>
            <div style={{ ...S.small, marginTop: 6 }}>
              {allProgress.done} / {allProgress.total} ({allProgress.pct}%)
            </div>
          </div>
        </div>

        {/* All done banner for current filter */}
        {filterProgress.total > 0 && filterProgress.done === filterProgress.total && (
          <div style={{ ...S.card, marginBottom: 12, borderColor: "#bbf7d0" }}>
            <div style={{ ...S.cardPad, background: "#ecfdf5", borderRadius: 16 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>All done for this filter 🎉</div>
              <div style={S.small}>
                You’ve labeled all {filterProgress.total} candidates in <b>{siteFilter === "all" ? "all sites" : `site: ${siteFilter}`}</b>. You can download your labels now.
              </div>
              <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button style={S.btnPri} onClick={handleDownload}>Download gold.jsonl</button>
                <button style={S.btn} onClick={jumpToNextUnlabeled}>Jump to next unlabeled</button>
              </div>
            </div>
          </div>
        )}

        {/* Main */}
        {current ? (
          <div style={{ ...S.grid2 }}>
            {/* Excerpt card */}
            <div style={{ ...S.card }}>
              <div style={S.cardPad}>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8, alignItems: "center" }}>
                  <span style={S.tag}>page_id: <b style={S.mono}>{current.page_id}</b></span>
                  <span style={S.tag}>candidate_id: <b style={S.mono}>{current.candidate_id}</b></span>
                  {current.weak_label && (
                    <span style={{ ...S.tag, background: "#FEF3C7" }}>
                      weak: <b>{current.weak_label}</b> ({(current.weak_conf ?? 0.5).toFixed(2)})
                    </span>
                  )}
                  {labeledByYou
                    ? <span style={S.pill("#d1fae5", "#065f46")}>Status: Labeled by you</span>
                    : <span style={S.pill("#f3f4f6", "#111")}>Status: Not labeled</span>}
                </div>
                <iframe
                  ref={iframeRef}
                  title="excerpt"
                  style={{ width: "100%", height: 320, background: "#fff", border: "1px solid #e5e7eb", borderRadius: 8 }}
                  sandbox="allow-same-origin"
                />
                <div style={{ ...S.small, marginTop: 8 }}>
                  <div><b>text:</b> <span>{current.text || <i style={{ color: "#888" }}>(empty)</i>}</span></div>
                  <div style={{ marginTop: 4 }}>
                    <b>tag:</b> {current.tag || <i style={{ color: "#888" }}>n/a</i>}{" "}
                    <b style={{ marginLeft: 8 }}>class:</b>{" "}
                    <span style={S.mono}>{current.classes || <i style={{ color: "#888" }}>(none)</i>}</span>{" "}
                    <b style={{ marginLeft: 8 }}>id:</b>{" "}
                    <span style={S.mono}>{current.id || <i style={{ color: "#888" }}>(none)</i>}</span>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <b>dom_path:</b> <span style={{ ...S.mono, wordBreak: "break-all" }}>{current.dom_path || current.candidate_id}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Label panel */}
            <div style={{ ...S.card }}>
              <div style={{ ...S.cardPad, display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={S.grid3}>
                  {LABELS.map((L, i) => (
                    <LabelBtn key={L} value={L} hotkey={i + 1} selected={label === L} onClick={() => setLabel(L)} />
                  ))}
                </div>

                <div>
                  <div style={S.small}>Confidence: <b>{conf.toFixed(2)}</b> (use [ and ] keys)</div>
                  <input type="range" min="0.5" max="1" step="0.01" value={conf} onChange={(e) => setConf(parseFloat(e.target.value))} style={{ width: "100%" }} />
                </div>

                <div>
                  <div style={S.small}>Notes (optional)</div>
                  <textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Reasoning / edge case notes" style={{ width: "100%", height: 90, padding: 8, borderRadius: 8, border: "1px solid #ddd" }} />
                </div>

                <div style={{ display: "flex", gap: 8, marginTop: 4, flexWrap: "wrap" }}>
                  <button onClick={prevItem} style={S.btn}>Prev (P)</button>
                  <button onClick={skipItem} style={S.btn}>Skip (K)</button>
                  <button onClick={jumpToNextUnlabeled} style={S.btn}>Next unlabeled (J)</button>

                  <button
                    onClick={handleSave}
                    style={{ ...S.btnPri, background: labeledByYou ? "#2563eb" : "#059669", borderColor: labeledByYou ? "#2563eb" : "#059669" }}
                    title={labeledByYou ? "Update (relabel) your existing label" : "Save new label"}
                  >
                    {labeledByYou ? "Update (relabel)" : "Save (S)"}
                  </button>

                  <button onClick={nextItem} style={S.btn}>Next (N)</button>
                </div>

                <div style={S.small}>
                  Selected label: <b>{label || "—"}</b> · {labeledByYou ? "Already labeled by you" : "Not yet labeled by you"} · UI {uiVersion} · Session{" "}
                  <span style={S.mono}>{smallHash((annotator || "anon") + "|" + (weakRows[0]?.page_id || ""))}</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ ...S.card }}>
            <div style={{ ...S.cardPad, color: "#374151" }}>
              <p style={{ marginTop: 0 }}>Load your <code style={{ ...S.mono, background: "#f3f4f6", padding: "0 4px", borderRadius: 4 }}>weak.jsonl</code> to begin.</p>
              <ul style={{ marginLeft: 18 }}>
                <li>File can be JSONL (one JSON per line) or a JSON <i>array</i>.</li>
                <li>Fields recognized: <code style={S.mono}>site_id</code>, <code style={S.mono}>url</code>, <code style={S.mono}>path</code>, <code style={S.mono}>outer_html</code>, <code style={S.mono}>text</code>, <code style={S.mono}>tag</code>, <code style={S.mono}>class</code>, <code style={S.mono}>id</code>, <code style={S.mono}>weak_label</code>, <code style={S.mono}>weak_conf</code>.</li>
                <li>Click <b>Download gold.jsonl</b> anytime; append to <code style={S.mono}>data/labeled/gold.jsonl</code>.</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
