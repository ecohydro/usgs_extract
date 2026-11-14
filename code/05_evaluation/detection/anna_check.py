"""One web page for Anna to verify Claude's calls: OUT/for_anna/check.html (+ img/).
Sections: (1) PAGE_CHECK_SAMPLE random listed pages with the printed number the page map expects,
(2) every borderline page, (3) ANNA_LABEL_SAMPLE random table / no-table calls from the list check and
from the false-alarm check. Run: python anna_check.py"""
import html
import pandas as pd
from PIL import Image
import defaults as D
from run_detection import image_for
from reducto_false_alarms import BATCHES

OUT = D.OUT / "for_anna"


def save(im, name, width=900):
    im = im.convert("L")
    im = im.resize((width, int(width * im.height / im.width)))
    im.save(OUT / "img" / name, quality=70)
    return f"img/{name}"


def corpus_image(page):
    doc = page.split("_page_")[0]
    for _, folder in BATCHES.values():
        p = D.CLUSTER / folder / doc / f"{page}.png"
        if p.exists():
            return Image.open(p)
    return None


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    docs, parts = {}, ["<html><body style='font-family:sans-serif;max-width:950px;margin:auto'>"]

    def block(title, caption, src):
        parts.append(f"<h3>{html.escape(title)}</h3><p>{html.escape(caption)}</p><img src='{src}' style='width:100%;border:1px solid #999'>")

    pm = pd.read_csv(D.OUT / "page_map_final.csv", dtype={"report": str}).dropna(subset=["pdf_page"])
    parts.append("<h1>1. Page numbers</h1><p>Is the printed page number on the scan the one stated? (Fold-out tables have no number of their own.)</p>")
    for r in pm.sample(D.PAGE_CHECK_SAMPLE, random_state=D.SEED).itertuples():
        im, _ = image_for(r.report, int(r.pdf_page), docs)
        block(f"Report {r.report}: list says printed page {r.printed_page}", f"PDF page {int(r.pdf_page)}; found by: {r.method.replace('_', ' ')}",
              save(im, f"map_{r.report}_{int(r.pdf_page)}.jpg"))

    lc = pd.read_csv(D.OUT / "list_check/pages_labeled.csv", dtype={"report": str})
    fa = pd.concat([pd.read_csv(D.OUT / "false_alarm_check" / f) for f in ("sample_labeled.csv", "sample_extra_labeled.csv")])
    fa = fa[~fa.note.str.contains("no image", case=False, na=False)]
    parts.append("<h1>2. Borderline pages</h1><p>Table or not? Your ruling settles each kind.</p>")
    for r in lc[lc.label == "borderline"].itertuples():
        im, _ = image_for(r.report, int(r.pdf_page), docs)
        block(f"Listed page: report {r.report}, PDF page {int(r.pdf_page)}", f"Claude: borderline. {r.note}", save(im, f"b_{r.report}_{int(r.pdf_page)}.jpg"))
    for r in fa[fa.label == "borderline"].itertuples():
        block(f"Corpus page {r.page}", f"Claude: borderline. {r.note}", save(corpus_image(r.page), f"b_{r.page}.jpg"))
    parts.append("<h1>3. A sample of table / no-table calls</h1><p>Do you agree with each call?</p>")
    for r in lc[lc.label != "borderline"].sample(D.ANNA_LABEL_SAMPLE, random_state=D.SEED).itertuples():
        im, _ = image_for(r.report, int(r.pdf_page), docs)
        block(f"Listed page: report {r.report}, PDF page {int(r.pdf_page)}", f"Claude: {r.label.replace('_', ' ')}. {r.note}", save(im, f"s_{r.report}_{int(r.pdf_page)}.jpg"))
    for r in fa[fa.label != "borderline"].sample(D.ANNA_LABEL_SAMPLE, random_state=D.SEED).itertuples():
        block(f"Corpus page {r.page}", f"Claude: {r.label.replace('_', ' ')}. {r.note}", save(corpus_image(r.page), f"s_{r.page}.jpg"))
    parts.append("</body></html>")
    (OUT / "check.html").write_text("\n".join(parts))
    print(OUT / "check.html")


if __name__ == "__main__":
    main()
