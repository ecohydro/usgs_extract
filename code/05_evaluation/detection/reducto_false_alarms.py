"""False alarms in the real corpus, estimated from what Reducto returned for the flagged pages.

A flagged page (sent to Reducto) with no table CSV back is a candidate false alarm.
Produces OUT/corpus_flagged_pages.csv (one row per page sent to Reducto: batch, best detector score,
reducto_table yes/no), OUT/reducto_no_table_by_score.csv, and the blind check sample:
OUT/false_alarm_check/sample.csv (page ids in random order, group hidden), key.csv (the groups) and
sheets of the page images. Run: python reducto_false_alarms.py
"""
import ast, csv, os, re, random
import pandas as pd
from PIL import Image, ImageDraw
import defaults as D

csv.field_size_limit(10**9)
BATCHES = {"batch_one": ("csvs/cali_1970_results.csv", "UpdatedDataDec"),
           "batch_two": ("csvs/cali_1980_results.csv", "UpdatedDataJan")}
BANDS = [0, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0001]


def names_in(folder, strip_table=False):
    out = set()
    for sub in os.listdir(folder):
        p = os.path.join(folder, sub)
        if os.path.isdir(p):
            for f in os.listdir(p):
                n = f.rsplit(".", 1)[0]
                out.add(re.sub(r"_table\d+$", "", n) if strip_table else n)
    return out


def flagged_pages():
    sent = names_in(D.CLUSTER / "ReductJson")
    with_table = names_in(D.CLUSTER / "ReductCSVs", strip_table=True)
    rows, seen = [], set()
    for batch, (scores, _) in BATCHES.items():
        for r in csv.DictReader(open(D.CLUSTER / scores)):
            page = os.path.basename(r["file_path"])[:-4]
            if page not in sent or page in seen:
                continue
            seen.add(page)
            best = max((d["score"] for d in ast.literal_eval(r["score"])), default=0.0)
            rows.append(dict(page=page, batch=batch, best_score=best, reducto_table=page in with_table))
    for page in sent - seen:
        rows.append(dict(page=page, batch="no_saved_score", best_score=None, reducto_table=page in with_table))
    return pd.DataFrame(rows)


def sheets(sample, folder):
    per, w = 4, 700
    for k in range(0, len(sample), per):
        tiles = []
        for r in sample.iloc[k:k + per].itertuples():
            doc = r.page.split("_page_")[0]
            src = [D.CLUSTER / b / doc / f"{r.page}.png" for _, b in BATCHES.values()]
            src = [s for s in src if s.exists()]
            im = Image.open(src[0]).convert("L") if src else Image.new("L", (w, 900), 255)
            im = im.resize((w, int(w * im.height / im.width)))
            t = Image.new("L", (w, im.height + 24), 255)
            ImageDraw.Draw(t).text((4, 4), f"#{r.n}  {r.page}" + ("" if src else "  (no image)"), fill=0)
            t.paste(im, (0, 24)); tiles.append(t)
        h = max(t.height for t in tiles)
        sheet = Image.new("L", (w * len(tiles), h), 255)
        for i, t in enumerate(tiles):
            sheet.paste(t, (w * i, 0))
        sheet.save(folder / f"sheet_{k // per:02d}.png")


def extend(pool, folder):
    """Second draw, Reducto-table pages only: that group is 90% of flagged pages, so its error rate
    drives the corrected estimate and 60 pages leave it too uncertain."""
    if (folder / "key_extra.csv").exists():
        return
    first = pd.read_csv(folder / "key.csv")
    rest = pool[pool.reducto_table & ~pool.page.isin(first.page)]
    pick = rest.sample(D.N_REDUCTO_EXTRA, random_state=D.SEED + 1).reset_index(drop=True)
    pick.insert(0, "n", range(1001, 1001 + len(pick)))
    pick.to_csv(folder / "key_extra.csv", index=False)
    pick[["n", "page"]].assign(label="", note="").to_csv(folder / "sample_extra.csv", index=False)
    sub = folder / "extra"; sub.mkdir(exist_ok=True)
    sheets(pick, sub)


def main():
    df = flagged_pages()
    df.to_csv(D.OUT / "corpus_flagged_pages.csv", index=False)
    s = df.dropna(subset=["best_score"]).copy()
    s["score_band"] = pd.cut(s.best_score, BANDS, right=False)
    tab = s.groupby(["batch", "score_band"], observed=True).agg(pages=("page", "size"),
                                                             no_table=("reducto_table", lambda x: int((~x).sum())))
    tab["share_no_table"] = tab.no_table / tab.pages
    tab.to_csv(D.OUT / "reducto_no_table_by_score.csv")
    print(df.batch.value_counts().to_string()); print(tab.round(3).to_string())
    folder = D.OUT / "false_alarm_check"; folder.mkdir(exist_ok=True)
    pool = s[s.best_score >= min(D.THRESHOLDS_USED)]
    if (folder / "key.csv").exists():
        print("sample already drawn; not redrawn")
        extend(pool, folder); return
    pick = pd.concat([pool[pool.reducto_table].sample(D.N_REDUCTO_SAMPLE, random_state=D.SEED),
                      pool[~pool.reducto_table].sample(D.N_REDUCTO_SAMPLE, random_state=D.SEED)])
    pick = pick.sample(frac=1, random_state=D.SEED).reset_index(drop=True)
    pick.insert(0, "n", range(1, len(pick) + 1))
    pick.to_csv(folder / "key.csv", index=False)
    pick[["n", "page"]].assign(label="", note="").to_csv(folder / "sample.csv", index=False)
    sheets(pick, folder)


if __name__ == "__main__":
    main()
