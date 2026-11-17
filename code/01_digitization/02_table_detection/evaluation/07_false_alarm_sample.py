"""False alarms in the real corpus: what Reducto returned for the flagged pages, and the blind check sample.

A flagged page (sent to Reducto) with no table CSV back is a candidate false alarm.
Produces OUT/corpus_flagged_pages.csv (one row per page sent to Reducto: batch, best detector score,
reducto_table yes/no), OUT/corpus_counts.csv (per production batch: reports and pages scored by the detector,
pages sent to Reducto) and, under OUT/false_alarm_check/, three draws from the flagged pages scoring at or
above the lower threshold used, each with a key (n, page, group) and sheets of page images to label blind:
  key.csv          N_REDUCTO_SAMPLE pages with a Reducto table and as many without  -> sample_labeled.csv
  key_extra.csv    N_REDUCTO_EXTRA more with a Reducto table (90% of flagged pages) -> sample_extra_labeled.csv
  key_replace.csv  one replacement for each drawn page that has no image            -> sample_replace_labeled.csv
A draw whose key exists is not redrawn. Run: python 07_false_alarm_sample.py
"""
import ast, csv, os, re
import pandas as pd
import defaults as D
import images

csv.field_size_limit(10**9)
FOLDER = D.OUT / "false_alarm_check"


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
    """(pages sent to Reducto, counts per batch). A page in both score files counts in the first batch."""
    sent = names_in(D.CLUSTER / "ReductJson")
    with_table = names_in(D.CLUSTER / "ReductCSVs", strip_table=True)
    rows, counts, seen, scored = [], [], set(), set()
    for batch, (scores, _) in images.BATCHES.items():
        new = set()
        for r in csv.DictReader(open(D.CLUSTER / scores)):
            page = os.path.basename(r["file_path"])[:-4]
            new.add(page)
            if page not in sent or page in seen:
                continue
            seen.add(page)
            best = max((d["score"] for d in ast.literal_eval(r["score"])), default=0.0)
            rows.append(dict(page=page, batch=batch, best_score=best, reducto_table=page in with_table))
        new -= scored; scored |= new
        counts.append(dict(batch=batch, reports_scored=len({p.split("_page_")[0] for p in new}), pages_scored=len(new),
                           pages_sent_to_reducto=sum(r["batch"] == batch for r in rows)))
    for page in sent - seen:
        rows.append(dict(page=page, batch="no_saved_score", best_score=None, reducto_table=page in with_table))
    c = pd.DataFrame(counts)
    c.loc[len(c)] = ["both", len({p.split("_page_")[0] for p in scored}), len(scored), int(c.pages_sent_to_reducto.sum())]
    return pd.DataFrame(rows), c


def save_draw(pick, name, first_n, sub=None):
    pick = pick.reset_index(drop=True)
    pick.insert(0, "n", range(first_n, first_n + len(pick)))
    pick.to_csv(FOLDER / f"key{name}.csv", index=False)
    pick[["n", "page"]].assign(label="", note="").to_csv(FOLDER / f"sample{name}.csv", index=False)
    images.sheets([(f"#{r.n}  {r.page}", images.corpus_image(r.page)) for r in pick.itertuples()],
                  FOLDER / sub if sub else FOLDER, "sheet")


def draw_first(pool):
    pick = pd.concat([pool[pool.reducto_table].sample(D.N_REDUCTO_SAMPLE, random_state=D.SEED),
                      pool[~pool.reducto_table].sample(D.N_REDUCTO_SAMPLE, random_state=D.SEED)])
    save_draw(pick.sample(frac=1, random_state=D.SEED), "", 1)


def draw_extra(pool):
    first = pd.read_csv(FOLDER / "key.csv")
    rest = pool[pool.reducto_table & ~pool.page.isin(first.page)]
    save_draw(rest.sample(D.N_REDUCTO_EXTRA, random_state=D.SEED + 1), "_extra", 1001, "extra")


def draw_replacements(pool):
    """Same pool and group as the page replaced, pages not drawn before; the first page of the shuffled
    group that has an image is taken."""
    drawn = pd.concat([pd.read_csv(FOLDER / "key.csv"), pd.read_csv(FOLDER / "key_extra.csv")])
    missing = drawn[drawn.page.map(lambda p: images.corpus_image(p) is None)]
    picks = []
    for r in missing.itertuples():
        rest = pool[(pool.reducto_table == r.reducto_table) & ~pool.page.isin(drawn.page) &
                    ~pool.page.isin([p.page for p in picks])].sample(frac=1, random_state=D.SEED + 2)
        picks.append(next(x for x in rest.itertuples(index=False) if images.corpus_image(x.page) is not None))
    pick = pd.DataFrame(picks).assign(replaces=list(missing.page))
    save_draw(pick.sample(frac=1, random_state=D.SEED), "_replace", 2001, "replace")


def main():
    df, counts = flagged_pages()
    df.to_csv(D.OUT / "corpus_flagged_pages.csv", index=False)
    counts.to_csv(D.OUT / "corpus_counts.csv", index=False)
    print(counts.to_string(index=False))
    FOLDER.mkdir(exist_ok=True)
    pool = df[df.best_score >= min(D.THRESHOLDS_USED)]
    for name, draw in (("", draw_first), ("_extra", draw_extra), ("_replace", draw_replacements)):
        if not (FOLDER / f"key{name}.csv").exists():
            draw(pool)


if __name__ == "__main__":
    main()
