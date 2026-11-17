"""Typeset the metadata the pipeline returned for one table, for the methods pipeline-example figure.

Reads the full metadata run (data/analysis/evaluation/metadata/full_run/out/*.jsonl) for report 1264,
PDF page 31, table 1. Until that page has been processed it falls back to PROVISIONAL rows written by
hand from the printed page, and says so.

Run: /home/annaboser/optimal-sampling/.venv/bin/python code/01_digitization/methods_figures/render_metadata_example.py
Output: code/01_digitization/methods_figures/assets/metadata_1264_p31_t1.png
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/annaboser/optimal-sampling/.venv/lib/python3.13/site-packages")
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/analysis/evaluation/metadata/full_run/out"
DOC, PAGE, TABLE = 1264, 31, 1
OUT = Path(__file__).resolve().parent / "assets" / f"metadata_{DOC}_p{PAGE}_t{TABLE}.png"
ACCENT, ACCENT_FILL, INK = "#1f5f8b", "#e6eef5", "#141413"

PROVISIONAL = [  # written by hand from the printed page; replaced by the model's answer once it exists
    {"station_name": s, "data_type": "Stream Discharge", "date_start": "1950-11", "date_end": "1966-12",
     "temporal_resolution": "discrete", "variables": ["discharge"]}
    for s in ("Kern River near Kernville", "South Fork Kern River near Onyx", "Tule River near Springville",
              "Kaweah River near Three Rivers")
]

rows, source = PROVISIONAL, "PROVISIONAL (hand-written)"
for f in sorted(RUN.glob("*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r["doc_id"] == DOC and r["page_number"] == PAGE and r.get("parsed"):
            t = [t for t in r["parsed"] if t["table"] == TABLE]
            if t:
                rows, source = t[0]["locations"], f"model output ({f.name})"

COLS = [("Location", lambda r: r.get("station_name") or r.get("place_name") or r.get("coordinates") or ""),
        ("Data type", lambda r: r.get("data_type", "")),
        ("Start", lambda r: r.get("date_start", "")),
        ("End", lambda r: r.get("date_end", "")),
        ("Resolution", lambda r: r.get("temporal_resolution", "")),
        ("Variables", lambda r: ", ".join(r.get("variables", [])))]
head = "".join(f"<th>{c}</th>" for c, _ in COLS)
body = "".join("<tr>" + "".join(f"<td>{html.escape(str(fn(r)))}</td>" for _, fn in COLS) + "</tr>" for r in rows)
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
* {{ margin:0; box-sizing:border-box }}
body {{ background:#ffffff; font-family:Arial, sans-serif; color:{INK}; padding:6px; display:inline-block }}
table {{ border-collapse:collapse; font-size:20px; border:2px solid {INK}; }}
th {{ background:{INK}; color:#ffffff; font-weight:bold; padding:6px 12px; text-align:left; }}
td {{ border-top:1px solid #c9ccd1; padding:5px 12px; white-space:nowrap }}
</style></head><body><table><tr>{head}</tr>{body}</table></body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--host-resolver-rules=MAP * ^NOTFOUND"])
    ctx = browser.new_context(java_script_enabled=False, viewport={"width": 1400, "height": 800},
                              device_scale_factor=2)
    pg = ctx.new_page()
    pg.route("**/*", lambda r: r.abort("aborted"))
    pg.set_content(page)
    pg.locator("body").screenshot(path=str(OUT))
    browser.close()
print("saved", OUT, "from", source, len(rows), "rows")
