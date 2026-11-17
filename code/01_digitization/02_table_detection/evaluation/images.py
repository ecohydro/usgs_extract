"""Shared helpers: where a page image comes from, and contact sheets of four pages for a blind look."""
import pymupdf
from PIL import Image, ImageDraw
import defaults as D

BATCHES = {"batch_one": ("csvs/cali_1970_results.csv", "UpdatedDataDec"),     # detector scores, page images
           "batch_two": ("csvs/cali_1980_results.csv", "UpdatedDataJan")}
PER, W = 4, 700


def validation_image(report, q, docs):
    """Image of PDF page q of a listed report: the 2024 image if it exists, otherwise rendered from the PDF
    at RENDER_DPI in grayscale (as 03_preprocess.py does) and saved under NEW_IMAGES. docs caches open PDFs."""
    old = D.VAL_IMAGES / f"{report}_page_{q}.png"
    if old.exists():
        return Image.open(old), "images_2024"
    new = D.NEW_IMAGES / f"{report}_page_{q}.png"
    if not new.exists():
        if report not in docs:
            docs[report] = pymupdf.open(D.VAL_PDFS / f"{report}.pdf")
        if not 1 <= q <= len(docs[report]):
            return None, None
        pix = docs[report][q - 1].get_pixmap(dpi=D.RENDER_DPI, colorspace=pymupdf.csGRAY)
        D.NEW_IMAGES.mkdir(parents=True, exist_ok=True)
        Image.frombytes("L", (pix.width, pix.height), pix.samples).save(new)
    return Image.open(new), "rendered_now"


def corpus_image(page):
    """Image of a corpus page ('<report>_page_<n>') from either production batch; None if there is none."""
    doc = page.split("_page_")[0]
    for _, folder in BATCHES.values():
        p = D.CLUSTER / folder / doc / f"{page}.png"
        if p.exists():
            return Image.open(p)
    return None


def sheets(tiles, folder, prefix):
    """tiles: list of (caption, image or None). Four pages side by side per sheet, <prefix>_NNN.png."""
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for caption, im in tiles:
        im = Image.new("L", (W, 900), 255) if im is None else im.convert("L")
        im = im.resize((W, int(W * im.height / im.width)))
        t = Image.new("L", (W, im.height + 24), 255)
        ImageDraw.Draw(t).text((4, 4), caption, fill=0)
        t.paste(im, (0, 24)); made.append(t)
    for k in range(0, len(made), PER):
        ts = made[k:k + PER]
        sheet = Image.new("L", (W * len(ts), max(t.height for t in ts)), 255)
        for i, t in enumerate(ts):
            sheet.paste(t, (W * i, 0))
        sheet.save(folder / f"{prefix}_{k // PER:03d}.png")
