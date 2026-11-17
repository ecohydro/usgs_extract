"""Render the two methods figures from the slide files in project/slides/ to manuscript/figures/.

    methods_overview.png/.pdf   Figure M1, the pipeline flowchart          (project/slides/overview.html)
    methods_example.png/.pdf    Figure M2, one table through the pipeline  (project/slides/example.html)

The slides are a mirror of the Claude Slides deck (see README.md). Each slide is drawn on its
1920 x 1080 canvas with the deck's stylesheet, speaker notes removed, and trimmed to its content.
Arial is not installed here, so Arimo (same letter widths, Apache licence, assets/fonts/) stands in.

Run: /home/annaboser/optimal-sampling/.venv/bin/python code/05_evaluation/overview_figure/render_slides.py
"""
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "manuscript" / "figures"
SCALE = 2      # pixels per slide unit
PAD = 40       # white margin kept around the content, in output pixels
IMAGES = {     # image id in the deck -> local file
    "79645ae9f79beff0d82fb4f45aa423ff": "assets/detection_1264_p31.png",
    "2e5c57dbe5344cd62d465bc006ff039e": "assets/tables/peaks_1264_p31_digitized.png",
    "c01e6dc3e57e4137396f1163612b7a8e": "assets/metadata_1264_p31_t1.png",
}
FIGURES = {"overview": "methods_overview", "example": "methods_example"}

css = (HERE / "project/deck.css").read_text()
font = (HERE / "assets/fonts/Arimo.ttf").as_uri()
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=SCALE)
    for slide, name in FIGURES.items():
        html = re.sub(r"<aside>.*?</aside>", "", (HERE / f"project/slides/{slide}.html").read_text(), flags=re.S)
        for blob, path in IMAGES.items():
            html = html.replace(f"/_blob/{blob}", (HERE / path).as_uri())
        doc = f"""<html><head><style>
@font-face {{ font-family: Arial; src: url({font}); font-weight: 100 900 }}
* {{ box-sizing: border-box }} body, p {{ margin: 0 }} {css}
section {{ position: relative; width: 1920px; height: 1080px; overflow: hidden; background: #fff !important }}
</style></head><body>{html}</body></html>"""
        with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=True) as f:   # local files load only from a file page
            f.write(doc)
            f.flush()
            page.goto(Path(f.name).as_uri(), wait_until="networkidle")
        full = OUT / f"{name}.png"
        page.screenshot(path=str(full))
        im = Image.open(full).convert("RGB")
        box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
        box = (max(box[0] - PAD, 0), max(box[1] - PAD, 0), min(box[2] + PAD, im.width), min(box[3] + PAD, im.height))
        im = im.crop(box)
        im.save(full)
        im.save(OUT / f"{name}.pdf", resolution=72 * SCALE * 2)
        print(name, im.size)
    browser.close()
