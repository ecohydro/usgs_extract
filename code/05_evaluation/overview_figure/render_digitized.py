"""Render Reducto's output for one page as a typeset page image for the methods overview figure.

Text blocks are drawn as plain text, table blocks as ruled tables with a tinted header, so the image
shows what digitization produced (surrounding text + tables) rather than the JSON itself.

Run: <python with playwright> code/05_evaluation/overview_figure/render_digitized.py
Output: code/05_evaluation/overview_figure/assets/digitized_553_p267.png
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/annaboser/optimal-sampling/.venv/lib/python3.13/site-packages")
from playwright.sync_api import sync_playwright  # noqa: E402

JSON = Path("/home/waves/data/usgs_extract/ReductJson/553/553_page_267.json")
OUT = Path(__file__).resolve().parent / "assets" / "digitized_553_p267.png"
WIDTH, HEIGHT = 778, 1252          # same width and crop as the scanned page shown beside it
ACCENT, ACCENT_FILL, INK, MUTED = "#1f5f8b", "#e6eef5", "#141413", "#52514e"

blocks = [b for c in json.loads(JSON.read_text())["result"]["chunks"] for b in c["blocks"]]
parts = []
for b in blocks:
    if b["type"] == "Table":
        parts.append(f'<div class="tbl">{b["content"]}</div>')
    elif b["type"] in ("Title", "Section Header"):
        parts.append(f'<p class="title">{html.escape(b["content"])}</p>')
    elif b["type"] == "Header":
        parts.append(f'<p class="header">{html.escape(b["content"])}</p>')
    else:
        parts.append(f'<p class="text">{html.escape(b["content"])}</p>')

page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
* {{ margin:0; box-sizing:border-box }}
body {{ width:{WIDTH}px; height:{HEIGHT}px; overflow:hidden; background:#ffffff; font-family:Arial, sans-serif;
        color:{INK}; padding:22px 26px; }}
.header {{ font-size:12px; color:{MUTED}; text-align:center; }}
.title {{ font-size:15px; font-weight:bold; text-align:center; margin:4px 0; }}
.text {{ font-size:12px; line-height:1.3; margin:4px 0; color:{INK}; }}
.tbl {{ margin:6px 0 8px; }}
table {{ border-collapse:collapse; width:100%; font-size:10.5px; border:1.5px solid {ACCENT}; }}
th {{ background:{ACCENT_FILL}; color:{INK}; font-weight:bold; border:1px solid {ACCENT}; padding:1px 3px; }}
td {{ border:1px solid #c9d6e2; padding:0 3px; text-align:right; }}
td:first-child {{ text-align:left; }}
</style></head><body>{''.join(parts)}</body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--host-resolver-rules=MAP * ^NOTFOUND"])
    ctx = browser.new_context(java_script_enabled=False, viewport={"width": WIDTH, "height": HEIGHT},
                              device_scale_factor=2)
    pg = ctx.new_page()
    pg.route("**/*", lambda r: r.abort("aborted"))
    pg.set_content(page)
    pg.screenshot(path=str(OUT))
    browser.close()
print("saved", OUT)
