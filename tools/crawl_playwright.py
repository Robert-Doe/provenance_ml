#!/usr/bin/env python3
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import argparse, json, time, pathlib, logging, os

HOOKS_JS = r"""(function(){
  const _fetch = window.fetch;
  window.__net = [];
  window.fetch = async function(input, init){
    const url = (typeof input === 'string') ? input : input.url;
    window.__net.push({t: Date.now(), type:'fetch', url});
    return _fetch.apply(this, arguments);
  };
  const open = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function(method, url){
    (window.__net = window.__net || []).push({t: Date.now(), type:'xhr', url});
    return open.apply(this, arguments);
  };
  window.__inserts = [];
  const setHTML = Object.getOwnPropertyDescriptor(Element.prototype,'innerHTML').set;
  Object.defineProperty(Element.prototype,'innerHTML', {
    set(v){
      try { window.__inserts.push({t: Date.now(), via:'innerHTML', preview: (''+v).slice(0,200)}); } catch(e){}
      return setHTML.call(this, v);
    },
    get: Object.getOwnPropertyDescriptor(Element.prototype,'innerHTML').get
  });
  const ap = Element.prototype.appendChild;
  Element.prototype.appendChild = function(n){
    try { window.__inserts.push({t: Date.now(), via:'appendChild', node: n.tagName || n.nodeName}); } catch(e){}
    return ap.call(this, n);
  };
})();"""

CANDIDATE_SELECTOR = "div,section,article,li,aside,p"

def setup_logging(verbose: bool):
  logging.basicConfig(
      level=logging.DEBUG if verbose else logging.INFO,
      format="%(asctime)s | %(levelname)-7s | %(message)s",
      datefmt="%H:%M:%S",
  )

def parse_args():
  ap = argparse.ArgumentParser(description="Playwright crawler → raw JSON bundles")
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("--url", help="Single URL to crawl")
  g.add_argument("--urls", help="Path to a file with one URL per line")
  ap.add_argument("--site_id", required=True)
  ap.add_argument("--out", default="data/raw")
  ap.add_argument("--scrolls", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=30000)
  ap.add_argument("--headful", action="store_true")
  ap.add_argument("--screenshot", action="store_true")
  ap.add_argument("--trace", action="store_true")
  ap.add_argument("--verbose", action="store_true")
  return ap.parse_args()

def main():
  args = parse_args()
  setup_logging(args.verbose)

  # URLs
  urls = [args.url.strip()] if args.url else [
    ln.strip() for ln in open(args.urls, "r", encoding="utf-8")
    if ln.strip() and not ln.strip().startswith("#")
  ]
  logging.info(f"Loaded {len(urls)} URL(s)")

  # Output dir
  out_dir = pathlib.Path(args.out, args.site_id)
  out_dir.mkdir(parents=True, exist_ok=True)
  logging.info(f"Output dir: {out_dir.resolve()}")

  with sync_playwright() as p:
    logging.info("Launching Chromium…")
    browser = p.chromium.launch(headless=not args.headful, args=["--disable-dev-shm-usage"])
    ctx = browser.new_context()
    if args.trace:
      logging.info("Tracing enabled")
      ctx.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = ctx.new_page()

    # Debug hooks
    page.on("console", lambda m: logging.debug(f"[console:{m.type}] {m.text}"))  # <-- m.text (no ())
    page.on("pageerror", lambda e: logging.warning(f"[pageerror] {e}"))
    page.on("requestfailed", lambda r: logging.warning(f"[requestfailed] {r.url} {r.failure().error_text if r.failure() else ''}"))

    for i, url in enumerate(urls, 1):
      logging.info(f"[{i}/{len(urls)}] {url}")
      ts = int(time.time()*1000)
      out_fp = out_dir / f"{ts}.json"
      bundle = {
        "site_id": args.site_id, "url": url, "ts": ts,
        "rendered_html": "", "net_log": [], "runtime_inserts": [], "candidates": [],
        "status": None, "hook_injected": False, "error": None
      }
      try:
        page.add_init_script(HOOKS_JS)
        resp = page.goto(url, wait_until="domcontentloaded", timeout=args.timeout)
        bundle["status"] = resp.status if resp else None

        # Verify hooks exist
        try:
          injected_ok = page.evaluate("() => typeof window.__net !== 'undefined' && typeof window.__inserts !== 'undefined'")
          bundle["hook_injected"] = bool(injected_ok)
        except Exception as e:
          logging.debug(f"Hook check failed: {e}")

        # Scroll via JS (avoid Mouse.wheel quirks on some Windows setups)
        for _ in range(args.scrolls):
          page.evaluate("() => window.scrollBy(0, 1200)")
          page.wait_for_timeout(400)

        # Best-effort network idle
        try:
          page.wait_for_load_state("networkidle", timeout=6000)
        except Exception:
          pass

        # Capture artifacts (each safe-guarded)
        try: bundle["rendered_html"] = page.evaluate("() => document.documentElement.outerHTML")
        except Exception as e: logging.debug(f"html capture failed: {e}")
        try: bundle["net_log"] = page.evaluate("() => window.__net || []")
        except Exception as e: logging.debug(f"net_log capture failed: {e}")
        try: bundle["runtime_inserts"] = page.evaluate("() => window.__inserts || []")
        except Exception as e: logging.debug(f"inserts capture failed: {e}")

        # Extract candidates: inline getPath helper inside evaluate (no extra args!)
        try:
          cands = page.evaluate(f"""
            (sel) => {{
              function cssPath(e){{
                if (!(e instanceof Element)) return "";
                const path = [];
                while (e && e.nodeType === Node.ELEMENT_NODE) {{
                  let selector = e.nodeName;
                  if (e.id) {{ selector += "#" + e.id; path.unshift(selector); break; }}
                  let sib = e, nth = 1;
                  while ((sib = sib.previousElementSibling)) {{
                    if (sib.nodeName === e.nodeName) nth++;
                  }}
                  selector += ":nth-of-type(" + nth + ")";
                  path.unshift(selector);
                  e = e.parentElement;
                }}
                return path.join(">");
              }}
              const arr = [];
              const els = document.querySelectorAll(sel);
              for (const el of els) {{
                const text = (el.innerText || "").trim();
                const longP = (el.tagName === 'P' && text.length < 40) ? false : true;
                if (el.tagName !== 'P' || longP) {{
                  const path = cssPath(el);
                  const depth = path ? path.split('>').length : 0;
                  arr.push({{
                    tag: el.tagName,
                    class: el.className || "",
                    id: el.id || "",
                    depth: depth,
                    path: path,
                    text: text.slice(0, 1000),
                    outer_html: el.outerHTML.slice(0, 4000),
                    childCount: el.childElementCount
                  }});
                }}
              }}
              return arr;
            }}
          """, CANDIDATE_SELECTOR)
          bundle["candidates"] = cands
        except Exception as e:
          logging.debug(f"candidate extraction failed: {e}")

        out_fp.write_text(json.dumps(bundle, ensure_ascii=False))
        logging.info(f"✔ Wrote {out_fp} (cands={len(bundle['candidates'])})")

        if args.screenshot:
          shot = out_dir / f"{ts}.png"
          page.screenshot(path=str(shot), full_page=True)
          logging.info(f"📸 Screenshot → {shot}")

      except PWTimeout as e:
        bundle["error"] = f"timeout: {e}"
        out_fp.write_text(json.dumps(bundle, ensure_ascii=False))
        logging.error(f"Timeout; wrote error bundle → {out_fp}")
      except Exception as e:
        bundle["error"] = repr(e)
        out_fp.write_text(json.dumps(bundle, ensure_ascii=False))
        logging.exception(f"Error; wrote error bundle → {out_fp}")

    if args.trace:
      trace_path = out_dir / "trace.zip"
      ctx.tracing.stop(path=str(trace_path))
      logging.info(f"Trace saved → {trace_path}")

    browser.close()
    logging.info("Done.")

if __name__ == "__main__":
  main()