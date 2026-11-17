"""Draw the table detector's own boxes on one example page, for the methods table-detection figure.

Boxes and scores are the saved production output of 01_detect.py (cluster csvs/cali_1980_results.csv),
drawn on the same page image the detector read.

Run: /home/annaboser/optimal-sampling/.venv/bin/python code/05_evaluation/overview_figure/render_detection_example.py
Output: code/05_evaluation/overview_figure/assets/detection_1264_p31.png
"""
import ast
import base64
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, "/home/annaboser/optimal-sampling/.venv/lib/python3.13/site-packages")
from playwright.sync_api import sync_playwright  # noqa: E402

DATA = Path("/home/waves/data/usgs_extract")
DOC, PAGE = "1264", 31
IMAGE = DATA / "UpdatedDataJan" / DOC / f"{DOC}_page_{PAGE}.png"
OUT = Path(__file__).resolve().parent / "assets" / f"detection_{DOC}_p{PAGE}.png"
ACCENT = "#1f5f8b"

scores = pd.concat(pd.read_csv(DATA / "csvs" / f"cali_{y}_results.csv") for y in (1970, 1980))
row = scores[scores.file_path.str.endswith(f"/{DOC}_page_{PAGE}.png")].iloc[0]
dets = [d for d in ast.literal_eval(row.score) if d["label"].startswith("table")]

W, H = Image.open(IMAGE).size
img = base64.b64encode(IMAGE.read_bytes()).decode()
boxes = "".join(
    f'<div class="box" style="left:{x0}px; top:{y0}px; width:{x1 - x0}px; height:{y1 - y0}px">'
    f'<span class="tag">Table, score {d["score"]:.2f}</span></div>'
    for d in dets for x0, y0, x1, y1 in [[round(v) for v in d["bbox"]]]
)
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
* {{ margin:0; box-sizing:border-box }}
body {{ width:{W}px; height:{H}px; position:relative; font-family:Arial, sans-serif; }}
img {{ position:absolute; left:0; top:0; width:{W}px; height:{H}px; }}
.box {{ position:absolute; border:16px solid {ACCENT}; background:rgba(31,95,139,0.10); }}
.tag {{ position:absolute; right:-16px; top:100%; background:{ACCENT}; color:#ffffff; font-size:56px;
        font-weight:bold; padding:4px 14px; border-radius:0 0 6px 6px; }}
</style></head><body><img src="data:image/png;base64,{img}">{boxes}</body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--host-resolver-rules=MAP * ^NOTFOUND"])
    ctx = browser.new_context(java_script_enabled=False, viewport={"width": W, "height": H})
    pg = ctx.new_page()
    pg.route("http*://**", lambda r: r.abort("aborted"))
    pg.set_content(page)
    pg.screenshot(path=str(OUT))
    browser.close()
print("saved", OUT, [(round(d["score"], 2)) for d in dets])
