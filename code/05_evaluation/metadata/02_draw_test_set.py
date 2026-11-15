"""Draw the metadata test set: published tables, stratified by water type group.

Writes data/analysis/evaluation/metadata/: test_set_tables.csv (one row per sampled table),
test_set_frame_counts.csv (what each rule removed), pages/ (the page scans).
Pages already used for development or tuning (defaults.DEVELOPMENT_PAGE_LISTS) are excluded.
Tables on pages withheld for tribal data sovereignty (tribal.py, Annette's rule) are replaced in
place by the next table in their stratum's random order, so the other sample numbers never change.
Run from the repo root:  python code/05_evaluation/metadata/02_draw_test_set.py [test|validation]
A set also excludes every page of the sets defined before it in defaults.SETS.
"""
import re
import shutil
import sys

import numpy as np
import pandas as pd

import defaults as D
import tribal


def page_image(doc, page):
    for root in D.PAGE_IMAGES:
        f = root / str(doc) / f"{doc}_page_{page}.png"
        if f.exists():
            return f
    return None


def development_pages():
    pages = set()
    for rel, dcol, pcol in D.DEVELOPMENT_PAGE_LISTS:
        f = D.REPO / rel
        if not f.exists():
            print("  (missing, skipped)", rel)
            continue
        d = pd.read_csv(f, low_memory=False, encoding="utf-8", encoding_errors="replace")
        if dcol not in d or pcol not in d:
            print("  (columns not found, skipped)", rel)
            continue
        d = d.dropna(subset=[dcol, pcol])
        pages |= set(zip(d[dcol].astype(int), d[pcol].astype(int)))
    for f in D.DEVELOPMENT_JSON_DIR.glob("*.json"):
        m = re.match(r"(\d+)_page_(\d+)", f.name)
        pages.add((int(m[1]), int(m[2])))
    return pages


def main(set_name="test"):
    cfg = D.SETS[set_name]
    strata, pages_dir = cfg["strata"], cfg["pages"]
    D.OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(D.PUBLISHED_TABLES, low_memory=False)
    t["stratum"] = t.water_type_final.where(t.water_type_final.isin(list(strata)), "Small types")
    counts = [("published tables", len(t), t.doc_id.nunique())]
    dev = development_pages()
    for other in D.SETS:                      # a set never shares a page with the sets defined before it
        if other == set_name:
            break
        ts = pd.read_csv(D.SETS[other]["tables"])
        dev |= set(zip(ts.doc_id.astype(int), ts.page_number.astype(int)))
    key = list(zip(t.doc_id.astype(int), t.page_number.astype(int)))
    t = t[[k not in dev for k in key]]
    counts.append((f"... not on a development page ({len(dev)} pages set aside)", len(t), t.doc_id.nunique()))
    weights = t.stratum.value_counts()
    for s in strata:
        counts.append((f"    {s}", int(weights[s]), t[t.stratum == s].doc_id.nunique()))
    pd.DataFrame(counts, columns=["step", "tables", "reports"]).to_csv(D.OUT / f"{set_name}_set_frame_counts.csv", index=False)
    print(pd.DataFrame(counts, columns=["step", "tables", "reports"]).to_string(index=False))

    rng = np.random.default_rng(cfg["seed"])
    picks, pages_used, per_report, queue = [], set(), {}, {}

    def take(s, r):
        """Return a pick for candidate row r of stratum s, or None if a rule turns it down."""
        doc, page = int(r.doc_id), int(r.page_number)
        if (doc, page) in D.EXCLUDED_PAGES:
            return None
        if D.ONE_TABLE_PER_PAGE and (doc, page) in pages_used:
            return None
        if per_report.get(doc, 0) >= D.MAX_TABLES_PER_REPORT:
            return None
        img = page_image(doc, page)
        if img is None:
            return None
        pages_used.add((doc, page))
        per_report[doc] = per_report.get(doc, 0) + 1
        return dict(stratum=s, doc_id=doc, page_number=page, table_number=int(r.table_number),
                    stratum_tables=int(weights[s]), page_image=str(img), replaces="")

    # smallest stratum first, so the per-report cap cannot starve it
    for s in sorted(strata, key=lambda k: weights[k]):
        pool = t[t.stratum == s].sort_values(["doc_id", "page_number", "table_number"]).reset_index(drop=True)
        queue[s] = [pool.iloc[i] for i in rng.permutation(len(pool))]
        n = 0
        while queue[s] and n < strata[s]:
            p = take(s, queue[s].pop(0))
            if p:
                picks.append(p)
                n += 1
    # shuffle so any prefix of the list is a balanced mix of strata (grading can stop early)
    picks = [picks[i] for i in rng.permutation(len(picks))]
    # Replacement pass: a pick on a withheld (tribal lands) page is replaced in place, so every other
    # sample number stays what it was. Reasons stay in the git-ignored tribal/ folder.
    log = []
    for i, p in enumerate(picks):
        held, why = tribal.withheld(p["doc_id"], p["page_number"])
        while held:
            log.append(dict(sample_no=i + 1, doc_id=p["doc_id"], page_number=p["page_number"], evidence=why))
            new = None
            while queue[p["stratum"]] and new is None:
                new = take(p["stratum"], queue[p["stratum"]].pop(0))
            new["replaces"] = "a table on a page withheld for tribal data sovereignty"
            p = picks[i] = new
            held, why = tribal.withheld(p["doc_id"], p["page_number"])
    pd.DataFrame(log, columns=["sample_no", "doc_id", "page_number", "evidence"]).to_csv(
        tribal.DIR / f"replaced_in_{set_name}_set.csv", index=False)
    s = pd.DataFrame(picks)
    s.insert(0, "sample_no", range(1, len(s) + 1))
    s["weight"] = s.stratum_tables / s.stratum.map(s.stratum.value_counts())
    keep = {f"{r.doc_id}_page_{r.page_number}.png" for r in s.itertuples()}
    if pages_dir.exists():
        for old in pages_dir.glob("*.png"):   # a replaced page's scan leaves the folder
            if old.name not in keep:
                old.unlink()
    pages_dir.mkdir(exist_ok=True)
    for r in s.itertuples():
        dst = pages_dir / f"{r.doc_id}_page_{r.page_number}.png"
        if not dst.exists():
            shutil.copy(r.page_image, dst)
    s.to_csv(cfg["tables"], index=False)
    print(f"drawn {len(s)} tables from {s.doc_id.nunique()} reports; "
          f"{len(log)} replaced after the tribal lands screen (sample numbers: {sorted({l['sample_no'] for l in log})})")
    print(s.stratum.value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
