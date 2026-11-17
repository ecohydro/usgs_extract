"""Visual check that listed pages really hold a table (Anna: every listed page scoring below MISS_BELOW,
plus N_HIT_SAMPLE random pages at or above it), done blind to the score.

python list_check.py draw        -> OUT/list_check/pages.csv (n, report, pdf_page; shuffled, score hidden),
                                    key.csv (with score and group) and sheets sheet_NNN.png (4 pages each)
python list_check.py rest        -> (Anna 2026-11-16: check them all) every listed page not yet labelled:
                                    key_rest.csv, pages_rest.csv, sheets rest_NNN.png; labels go in
                                    pages_rest_labeled.csv; neighbours / finish then use both label files
python list_check.py neighbours  -> for pages labelled no_table in pages_labeled.csv: sheets nb_NNN.png showing
                                    the NEIGHBOUR pages either side (an unnumbered fold-out table 'facing page N'
                                    sits next to the page the list names) and neighbours.csv to label
python list_check.py finish      -> OUT/listed_page_labels.csv (report, pdf_page, label) and
                                    OUT/page_map_final.csv updated: a no_table page with a table on a neighbour
                                    moves to that neighbour (method 'moved_to_adjacent_table')
Labels are entered in pages_labeled.csv / neighbours_labeled.csv (columns n, label, note).
"""
import sys
import pandas as pd
from PIL import Image, ImageDraw
import defaults as D
from run_detection import image_for

FOLDER = D.OUT / "list_check"
PER, W = 4, 700


def sheets(rows, prefix):
    """rows: DataFrame with n, report, pdf_page. Four page images side by side per sheet."""
    docs = {}
    for k in range(0, len(rows), PER):
        if (FOLDER / f"{prefix}_{k // PER:03d}.png").exists():
            continue                                   # resumable
        tiles = []
        for r in rows.iloc[k:k + PER].itertuples():
            im, _ = image_for(r.report, int(r.pdf_page), docs)
            im = Image.new("L", (W, 900), 255) if im is None else im.convert("L")
            im = im.resize((W, int(W * im.height / im.width)))
            t = Image.new("L", (W, im.height + 24), 255)
            ImageDraw.Draw(t).text((4, 4), f"#{r.n}  report {r.report} PDF page {int(r.pdf_page)}", fill=0)
            t.paste(im, (0, 24)); tiles.append(t)
        sheet = Image.new("L", (W * len(tiles), max(t.height for t in tiles)), 255)
        for i, t in enumerate(tiles):
            sheet.paste(t, (W * i, 0))
        sheet.save(FOLDER / f"{prefix}_{k // PER:03d}.png")


def draw():
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / "key.csv").exists():
        sys.exit("already drawn; delete OUT/list_check to redraw")
    pm = pd.read_csv(D.OUT / "page_map_final.csv", dtype={"report": str}).dropna(subset=["pdf_page"])
    det = pd.read_csv(D.OUT / "detections.csv", dtype={"report": str})[["report", "pdf_page", "best_score"]]
    df = pm.merge(det, on=["report", "pdf_page"], how="inner").drop_duplicates(["report", "pdf_page"])
    low = df[df.best_score < D.MISS_BELOW].assign(group="scores_low")
    high = df[df.best_score >= D.MISS_BELOW].sample(D.N_HIT_SAMPLE, random_state=D.SEED).assign(group="scores_high_sample")
    key = pd.concat([low, high]).sample(frac=1, random_state=D.SEED).reset_index(drop=True)
    key.insert(0, "n", range(1, len(key) + 1))
    key.to_csv(FOLDER / "key.csv", index=False)
    key[["n", "report", "pdf_page"]].assign(label="", note="").to_csv(FOLDER / "pages.csv", index=False)
    sheets(key, "sheet")
    print(key.group.value_counts().to_string())


def labels():
    """Labels of the first draw plus, when present, of the rest."""
    files = [f for f in ("pages_labeled.csv", "pages_rest_labeled.csv") if (FOLDER / f).exists()]
    return pd.concat([pd.read_csv(FOLDER / f, dtype={"report": str}) for f in files], ignore_index=True)


def rest():
    if (FOLDER / "key_rest.csv").exists():             # interrupted draw: finish the missing sheets
        return sheets(pd.read_csv(FOLDER / "key_rest.csv", dtype={"report": str}), "rest")
    df = pd.read_csv(D.OUT / "listed_pages_scored.csv", dtype={"report": str}).dropna(subset=["best_score"])
    key = df[df.label.isna()].drop_duplicates(["report", "pdf_page"]).sample(frac=1, random_state=D.SEED)
    key = key[["report", "pdf_page", "best_score"]].reset_index(drop=True).assign(group="rest")
    first = pd.read_csv(FOLDER / "key.csv").n.max()
    key.insert(0, "n", range(first + 1, first + 1 + len(key)))
    key.to_csv(FOLDER / "key_rest.csv", index=False)
    key[["n", "report", "pdf_page"]].assign(label="", note="").to_csv(FOLDER / "pages_rest.csv", index=False)
    sheets(key, "rest")
    print(len(key), "pages")


def neighbours():
    lab = labels()
    if (FOLDER / "neighbours.csv").exists():           # first round done: only the pages not yet handled
        lab = lab[~lab.n.isin(pd.read_csv(FOLDER / "neighbours.csv").listed_n)]
        out, prefix = "neighbours_rest.csv", "nbrest"
    else:
        out, prefix = "neighbours.csv", "nb"
    listed = pd.read_csv(D.OUT / "page_map_final.csv", dtype={"report": str}).dropna(subset=["pdf_page"])
    taken = set(zip(listed.report, listed.pdf_page.astype(int)))
    rows = []
    for r in lab[lab.label == "no_table"].itertuples():
        for q in range(int(r.pdf_page) - D.NEIGHBOUR, int(r.pdf_page) + D.NEIGHBOUR + 1):
            if q != r.pdf_page and q >= 1 and (r.report, q) not in taken:
                rows.append(dict(listed_n=r.n, report=r.report, listed_pdf_page=int(r.pdf_page), pdf_page=q))
    nb = pd.DataFrame(rows)
    nb.insert(0, "n", range(1, len(nb) + 1))
    nb.assign(label="", note="").to_csv(FOLDER / out, index=False)
    sheets(nb, prefix)
    print(len(nb), "neighbour pages of", lab.label.eq("no_table").sum(), "no_table pages")


def finish():
    lab = labels()
    pm = pd.read_csv(D.OUT / "page_map_final.csv", dtype={"report": str})
    moved = 0
    nbs = [f for f in ("neighbours_labeled.csv", "neighbours_rest_labeled.csv") if (FOLDER / f).exists()]
    if nbs:
        nb = pd.concat([pd.read_csv(FOLDER / f, dtype={"report": str}) for f in nbs], ignore_index=True)
        for n, g in nb[nb.label == "table"].groupby("listed_n"):
            g = g.assign(d=(g.pdf_page - g.listed_pdf_page).abs()).sort_values(["d", "pdf_page"], ascending=[True, False])
            r = g.iloc[0]                              # nearest table; the page after wins a tie (fold-outs follow)
            m = (pm.report == r.report) & (pm.pdf_page == r.listed_pdf_page)
            pm.loc[m, ["pdf_page", "method"]] = [r.pdf_page, "moved_to_adjacent_table"]
            lab.loc[lab.n == n, ["pdf_page", "label"]] = [r.pdf_page, "table"]
            moved += 1                                 # counts earlier moves too (already in the map)
    pm.to_csv(D.OUT / "page_map_final.csv", index=False)
    lab[["report", "pdf_page", "label", "note"]].to_csv(D.OUT / "listed_page_labels.csv", index=False)
    print("moved to an adjacent table:", moved); print(lab.label.value_counts().to_string())


if __name__ == "__main__":
    {"draw": draw, "rest": rest, "neighbours": neighbours, "finish": finish}[sys.argv[1]]()
