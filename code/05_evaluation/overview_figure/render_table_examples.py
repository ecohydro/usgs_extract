"""Crop example tables from the scanned pages and typeset Reducto's output for each, for the methods
digitization figure. One pair of PNGs per table: <name>_scan.png (crop of the page at Reducto's own
table bounding box) and <name>_digitized.png (Reducto's HTML table, unedited, rendered).

Run: /home/annaboser/optimal-sampling/.venv/bin/python code/05_evaluation/overview_figure/render_table_examples.py
Output: code/05_evaluation/overview_figure/assets/tables/
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, "/home/annaboser/optimal-sampling/.venv/lib/python3.13/site-packages")
from playwright.sync_api import sync_playwright  # noqa: E402

DATA = Path("/home/waves/data/usgs_extract")
OUT = Path(__file__).resolve().parent / "assets" / "tables"
PAD = 0.008                      # extra margin around Reducto's box, share of page size
ACCENT, ACCENT_FILL, INK = "#1f5f8b", "#e6eef5", "#141413"

# (name, report, page, table number on the page counted from 1)
TABLES = [
    ("peaks_1264_p31", "1264", 31, 1),
    ("reservoir_733_p29", "733", 29, 1),
    ("budget_1258_p109", "1258", 109, 1),
    ("pumping_1128_p26", "1128", 26, 1),
    ("runoff_15_p124", "15", 124, 3),
    ("levels_2060_p79", "2060", 79, 1),
    ("quality_1084_p227", "1084", 227, 1),
    ("springs_2508_p24", "2508", 24, 2),
]


def page_image(doc, page):
    for batch in ("UpdatedDataDec", "UpdatedDataJan"):
        f = DATA / batch / doc / f"{doc}_page_{page}.png"
        if f.exists():
            return Image.open(f).convert("L")
    raise FileNotFoundError(f"{doc} p{page}")


def table_blocks(doc, page):
    res = json.loads((DATA / "ReductJson" / doc / f"{doc}_page_{page}.json").read_text())["result"]
    return [b for c in res["chunks"] for b in c["blocks"] if b["type"] == "Table"]


CSS = f"""* {{ margin:0; box-sizing:border-box }}
body {{ background:#ffffff; font-family:Arial, sans-serif; color:{INK}; padding:6px; display:inline-block }}
table {{ border-collapse:collapse; font-size:15px; border:1.5px solid {ACCENT}; }}
th {{ background:{ACCENT_FILL}; font-weight:bold; border:1px solid {ACCENT}; padding:3px 6px; }}
td {{ border:1px solid #c9d6e2; padding:2px 6px; text-align:right; white-space:nowrap }}
td:first-child {{ text-align:left; }}"""

OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--host-resolver-rules=MAP * ^NOTFOUND"])
    ctx = browser.new_context(java_script_enabled=False, viewport={"width": 1600, "height": 1200},
                              device_scale_factor=2)
    pg = ctx.new_page()
    pg.route("**/*", lambda r: r.abort("aborted"))
    for name, doc, page, k in TABLES:
        b = table_blocks(doc, page)[k - 1]
        im = page_image(doc, page)
        W, H = im.size
        bb = b["bbox"]
        box = (max(0, int((bb["left"] - PAD) * W)), max(0, int((bb["top"] - PAD) * H)),
               min(W, int((bb["left"] + bb["width"] + PAD) * W)), min(H, int((bb["top"] + bb["height"] + PAD) * H)))
        im.crop(box).save(OUT / f"{name}_scan.png", optimize=True)
        pg.set_content(f"<!doctype html><html><head><style>{CSS}</style></head><body>{b['content']}</body></html>")
        pg.locator("body").screenshot(path=str(OUT / f"{name}_digitized.png"))
        print(name, "crop", box)
    browser.close()
