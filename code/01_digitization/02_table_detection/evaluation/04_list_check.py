"""Visual check, blind to the detector score, that every listed page really holds a table.

python 04_list_check.py draw        -> OUT/list_check/pages.csv (n, report, pdf_page; shuffled) and sheets
                                       sheet_NNN.png of four pages; labels (table / borderline / no_table)
                                       are entered in pages_labeled.csv
python 04_list_check.py neighbours  -> for pages labelled no_table: the NEIGHBOUR unlisted pages either side
                                       (a fold-out table 'facing page N' sits next to the page the list names):
                                       neighbours.csv and sheets nb_NNN.png; labels in neighbours_labeled.csv
python 04_list_check.py finish      -> OUT/page_map_final.csv: a no_table page with a table on a neighbour
                                       moves to the nearest one (method 'moved_to_adjacent_table');
                                       OUT/listed_page_labels.csv: the label of every listed page after the moves

The labels on file were collected in two rounds (column 'round' of pages_labeled.csv): first every page
scoring below 0.8 and a random 150 of the others (scores_low, scores_high_sample; 2026-11-14), then all the
rest (2026-11-16, Anna: "check them all"). second_look_labeled.csv is a second blind look at 128 pages of the
rest (all its no_table pages and 96 others); the 4 no_table pages it called borderline are borderline in
pages_labeled.csv.
"""
import sys
import pandas as pd
import defaults as D
import images

FOLDER = D.OUT / "list_check"


def sheet_tiles(rows):
    docs = {}
    return [(f"#{r.n}  report {r.report} PDF page {int(r.pdf_page)}", images.validation_image(r.report, int(r.pdf_page), docs)[0])
            for r in rows.itertuples()]


def draw():
    if (FOLDER / "pages_labeled.csv").exists():
        sys.exit("labels exist; not redrawn")
    pm = pd.read_csv(D.OUT / "page_map_before_list_check.csv", dtype={"report": str}).dropna(subset=["pdf_page"])
    det = pd.read_csv(D.OUT / "detections.csv", dtype={"report": str})[["report", "pdf_page"]]
    key = pm.merge(det, on=["report", "pdf_page"]).drop_duplicates(["report", "pdf_page"])[["report", "pdf_page"]]
    key = key.sample(frac=1, random_state=D.SEED).reset_index(drop=True)
    key.insert(0, "n", range(1, len(key) + 1))
    FOLDER.mkdir(exist_ok=True)
    key.assign(label="", note="").to_csv(FOLDER / "pages.csv", index=False)
    images.sheets(sheet_tiles(key), FOLDER, "sheet")


def neighbours():
    if (FOLDER / "neighbours_labeled.csv").exists():
        sys.exit("labels exist; not redrawn")
    lab = pd.read_csv(FOLDER / "pages_labeled.csv", dtype={"report": str})
    listed = pd.read_csv(D.OUT / "page_map_before_list_check.csv", dtype={"report": str}).dropna(subset=["pdf_page"])
    taken = set(zip(listed.report, listed.pdf_page.astype(int)))
    rows = [dict(listed_n=r.n, report=r.report, listed_pdf_page=int(r.pdf_page), pdf_page=q)
            for r in lab[lab.label == "no_table"].itertuples()
            for q in range(int(r.pdf_page) - D.NEIGHBOUR, int(r.pdf_page) + D.NEIGHBOUR + 1)
            if q != r.pdf_page and q >= 1 and (r.report, q) not in taken]
    nb = pd.DataFrame(rows)
    nb.insert(0, "n", range(1, len(nb) + 1))
    nb.assign(label="", note="").to_csv(FOLDER / "neighbours.csv", index=False)
    images.sheets(sheet_tiles(nb), FOLDER, "nb")


def finish():
    lab = pd.read_csv(FOLDER / "pages_labeled.csv", dtype={"report": str})
    nb = pd.read_csv(FOLDER / "neighbours_labeled.csv", dtype={"report": str})
    pm = pd.read_csv(D.OUT / "page_map_before_list_check.csv", dtype={"report": str})
    for n, g in nb[nb.label == "table"].groupby("listed_n"):
        g = g.assign(d=(g.pdf_page - g.listed_pdf_page).abs()).sort_values(["d", "pdf_page"], ascending=[True, False])
        r = g.iloc[0]                                  # nearest table; the page after wins a tie (fold-outs follow)
        m = (pm.report == r.report) & (pm.pdf_page == r.listed_pdf_page)
        pm.loc[m, ["pdf_page", "method"]] = [r.pdf_page, "moved_to_adjacent_table"]
        lab.loc[lab.n == n, ["pdf_page", "label"]] = [r.pdf_page, "table"]
    pm.to_csv(D.OUT / "page_map_final.csv", index=False)
    lab[["report", "pdf_page", "label", "note"]].to_csv(D.OUT / "listed_page_labels.csv", index=False)
    print("moved to an adjacent table:", int((pm.method == "moved_to_adjacent_table").sum()))
    print(lab.label.value_counts().to_string())


if __name__ == "__main__":
    {"draw": draw, "neighbours": neighbours, "finish": finish}[sys.argv[1]]()
