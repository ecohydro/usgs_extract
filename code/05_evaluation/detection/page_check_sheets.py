"""Contact sheets for checking the page map by eye: the top and bottom strip of each chosen PDF page,
labelled with the report, the PDF page and the printed page the map expects there.

Run: python page_check_sheets.py uncertain   (up to 3 pages per report and method where the map is extrapolated, ambiguous or missing)
     python page_check_sheets.py sample      (PAGE_CHECK_SAMPLE random listed pages, for Anna's spot check)
Writes OUT/page_check/<mode>_NN.png and <mode>_pages.csv.
"""
import sys, random
import pandas as pd, pymupdf
from PIL import Image, ImageDraw
import defaults as D

PER_SHEET = 8


def strips(report, pdf_page, label):
    doc = pymupdf.open(D.VAL_PDFS / f"{report}.pdf")
    if not 1 <= pdf_page <= len(doc):
        return None
    pix = doc[pdf_page - 1].get_pixmap(dpi=D.CHECK_DPI, colorspace=pymupdf.csGRAY)
    im = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    if im.width > im.height:                       # landscape scan: show whole page small
        parts = [im.resize((1000, int(1000 * im.height / im.width)))]
    else:
        im = im.resize((1000, int(1000 * im.height / im.width)))
        parts = [im.crop((0, 0, 1000, int(im.height * D.CHECK_TOP))),
                 im.crop((0, int(im.height * (1 - D.CHECK_BOTTOM)), 1000, im.height))]
    h = sum(p.height for p in parts) + 22
    out = Image.new("L", (1000, h), 255)
    ImageDraw.Draw(out).text((4, 4), label, fill=0)
    y = 22
    for p in parts:
        out.paste(p, (0, y)); y += p.height
    ImageDraw.Draw(out).line((0, h - 1, 1000, h - 1), fill=0, width=2)
    return out


def main(mode):
    pm = pd.read_csv(D.OUT / ("page_map.csv" if mode == "uncertain" else "page_map_final.csv"), dtype={"report": str})
    rng = random.Random(D.SEED)
    rows = []
    if mode == "uncertain":
        for rep, g in pm[~pm.method.isin(["printed_on_page", "between_agreeing_pages"])].groupby("report"):
            for meth, gg in g.groupby("method"):
                gg = gg.sort_values("printed_page")
                for i in sorted({0, len(gg) // 2, len(gg) - 1}):
                    r = gg.iloc[i]
                    if pd.isna(r.pdf_page):
                        rows.append((rep, int(r.printed_page), int(r.printed_page), meth + " (showing pdf page = printed)"))
                        continue
                    rows.append((rep, int(r.printed_page), int(r.pdf_page), meth))
                    if not pd.isna(r.pdf_page_alt):
                        for q in range(int(min(r.pdf_page, r.pdf_page_alt)) + 1, int(max(r.pdf_page, r.pdf_page_alt)) + 1):
                            rows.append((rep, int(r.printed_page), q, meth + " alt"))
    else:
        ok = pm.dropna(subset=["pdf_page"])
        for i in rng.sample(range(len(ok)), D.PAGE_CHECK_SAMPLE):
            r = ok.iloc[i]
            rows.append((r.report, int(r.printed_page), int(r.pdf_page), r.method))
    out = D.OUT / "page_check"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["report", "expected_printed", "pdf_page", "method"]).to_csv(out / f"{mode}_pages.csv", index=False)
    tiles = []
    for rep, p, q, meth in rows:
        t = strips(rep, q, f"report {rep} | PDF page {q} | expect printed {p} | {meth}")
        if t is not None:
            tiles.append(t)
    for k in range(0, len(tiles), PER_SHEET):
        ts = tiles[k:k + PER_SHEET]
        sheet = Image.new("L", (1000, sum(t.height for t in ts)), 255)
        y = 0
        for t in ts:
            sheet.paste(t, (0, y)); y += t.height
        sheet.save(out / f"{mode}_{k // PER_SHEET:02d}.png")
    print(len(tiles), "pages on", (len(tiles) + PER_SHEET - 1) // PER_SHEET, "sheets in", out)


if __name__ == "__main__":
    main(sys.argv[1])
