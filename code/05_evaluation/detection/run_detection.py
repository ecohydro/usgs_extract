"""Runs the repo's table detector (01_detect.py, unchanged) on the listed pages of the validation reports.

Produces OUT/detections.csv: one row per scored page with every detection the model returns
(no threshold applied; thresholds are applied when scoring). Resumable: pages already in the
file are skipped. Run: python run_detection.py   (GPU, about 15 minutes for ~4,000 pages)
Pages scored: every mapped listed page, plus MARGIN pages either side where the page map was
not read off the page itself (so a later correction of the map needs no new run).
Images: the 2024 page images where they exist; otherwise the page is rendered from the PDF
at RENDER_DPI in grayscale (as 03_preprocess.py does) and saved under NEW_IMAGES.
"""
import csv, json, importlib.util, os
import pandas as pd, pymupdf, torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForObjectDetection, TableTransformerConfig
import defaults as D

spec = importlib.util.spec_from_file_location("detect", D.DETECT_SCRIPT)
detect = importlib.util.module_from_spec(spec); spec.loader.exec_module(detect)
MODEL, REVISION = "microsoft/table-transformer-detection", "no_timm"     # as in 01_detect.py


def load_model(device):
    # transformers 5.x rejects the null fields in this model's config.json; drop them, weights unchanged
    cfg = json.load(open(hf_hub_download(MODEL, "config.json", revision=REVISION)))
    cfg = TableTransformerConfig(**{k: v for k, v in cfg.items() if v is not None})
    return AutoModelForObjectDetection.from_pretrained(MODEL, revision=REVISION, config=cfg).to(device).eval()


def pages_to_score():
    pm = pd.concat([pd.read_csv(D.OUT / f, dtype={"report": str}) for f in ("page_map.csv", "page_map_final.csv")]).dropna(subset=["pdf_page"])
    want = set()
    for r in pm.itertuples():
        lo = hi = int(r.pdf_page)
        if not pd.isna(r.pdf_page_alt):
            lo, hi = min(lo, int(r.pdf_page_alt)), max(hi, int(r.pdf_page_alt))
        m = D.NEIGHBOUR if r.method in ("printed_on_page", "read_off_scan") else D.MARGIN
        want.update((r.report, q) for q in range(lo - m, hi + m + 1))
    return sorted(want)


def image_for(report, q, docs):
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


def main():
    out = D.OUT / "detections.csv"
    done = set()
    if out.exists():
        done = {(r["report"], int(r["pdf_page"])) for r in csv.DictReader(open(out))}
    todo = [x for x in pages_to_score() if x not in done]
    print(len(done), "done,", len(todo), "to score")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    id2label = dict(model.config.id2label); id2label[len(id2label)] = "no object"
    docs = {}
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["report", "pdf_page", "image_source", "best_score", "detections"])
        for i, (report, q) in enumerate(todo):
            im, src = image_for(report, q, docs)
            if im is None:
                continue
            im = im.convert("RGB")
            with torch.no_grad():
                o = model(detect.detection_transform(im).unsqueeze(0).to(device))
            dets = detect.outputs_to_objects(o, im.size, id2label, 0.0)
            w.writerow([report, q, src, max((d["score"] for d in dets), default=0.0), json.dumps(dets)])
            if i % 200 == 0:
                f.flush(); print(i, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
