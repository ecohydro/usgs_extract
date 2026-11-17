"""Draw the truth-set pages: random pages within each kind of table (kind = kind of the page's largest table), one page per report,
from pages that have Reducto output, PaddleOCR output and a page image.

Produces  data/analysis/evaluation/digitization/sample_frame_counts.csv  (per-step counts)
          data/analysis/evaluation/digitization/sample_pages.csv         (the drawn pages)
          data/analysis/evaluation/digitization/pages/{doc}_page_{N}.png (copies of page images)
Run:      ~/miniconda3/envs/geofm-vlm/bin/python 01_draw_sample.py
"""
import re
import shutil
import numpy as np
import pandas as pd
import defaults as D


def paddle_pages():
    """One row per page with PaddleOCR output: number of tables, any rotated."""
    rows = []
    for name in sorted(p.name for p in D.PADDLE.iterdir()):
        m = re.match(r"(\d+)_page_(\d+)_table_(rotated_)?(\d+)$", name)
        if m:
            rows.append((int(m[1]), int(m[2]), bool(m[3])))
    p = pd.DataFrame(rows, columns=["doc_id", "page_number", "rotated"])
    return p.groupby(["doc_id", "page_number"]).agg(
        n_paddle_tables=("rotated", "size"), paddle_rotated=("rotated", "any")).reset_index()


def page_image(doc, page):
    for root in D.PAGE_IMAGES:
        f = root / str(doc) / f"{doc}_page_{page}.png"
        if f.exists():
            return f
    return None


def page_frame():
    """One row per page: its kind = the kind of its largest Reducto table (most printed rows).
    Pages mixing kinds (e.g. a daily grid above its monthly summary) stay in, flagged by n_kinds."""
    t = pd.read_csv(D.TABLE_RESOLUTION, low_memory=False)
    t = t[t.published == True]
    big = (t.sort_values("number_rows", ascending=False)
            .drop_duplicates(["doc_id", "page_number"])
            .set_index(["doc_id", "page_number"])["resolution"].rename("kind"))
    g = t.groupby(["doc_id", "page_number"])
    f = g.agg(n_reducto_tables=("table_number", "size"),
              n_kinds=("resolution", "nunique"),
              kinds_on_page=("resolution", lambda s: "; ".join(sorted(set(s)))),
              water_types=("water_type_final", lambda s: "; ".join(sorted(set(s)))),
              n_rows=("number_rows", "sum"),
              n_values=("n_values", "sum"))
    return f.join(big).reset_index()


def main():
    D.OUT.mkdir(parents=True, exist_ok=True)
    f = page_frame()
    counts = [("pages with published Reducto tables", len(f), f.doc_id.nunique())]
    f = f.merge(paddle_pages(), on=["doc_id", "page_number"])
    counts.append(("... that also have PaddleOCR output", len(f), f.doc_id.nunique()))
    counts.append(("    of which mix kinds of table (kept)", int((f.n_kinds > 1).sum()),
                   f[f.n_kinds > 1].doc_id.nunique()))
    f = f[f.kind.isin(D.KINDS)]
    counts.append(("... of a kind in the design (not unclassified)", len(f), f.doc_id.nunique()))
    for k in D.KINDS:
        s = f[f.kind == k]
        counts.append((f"    {k}", len(s), s.doc_id.nunique()))
    pd.DataFrame(counts, columns=["step", "pages", "reports"]).to_csv(
        D.OUT / "sample_frame_counts.csv", index=False)
    print(pd.DataFrame(counts, columns=["step", "pages", "reports"]).to_string(index=False))

    rng = np.random.default_rng(D.SEED)
    used, picks, queue = set(), [], {}
    # scarcest kind first, so the one-page-per-report rule cannot starve it
    order = sorted(D.KINDS, key=lambda k: f[f.kind == k].doc_id.nunique())
    for k in order:
        pool = f[f.kind == k].sort_values(["doc_id", "page_number"]).reset_index(drop=True)
        queue[k] = [pool.iloc[i] for i in rng.permutation(len(pool))]
        n = 0
        while queue[k] and n < D.PAGES_PER_KIND:
            r = queue[k].pop(0)
            if D.ONE_PAGE_PER_REPORT and r.doc_id in used:
                continue
            img = page_image(r.doc_id, r.page_number)
            if img is None:
                continue
            used.add(r.doc_id)
            picks.append({**r.to_dict(), "page_image": str(img), "replaces": ""})
            n += 1
    # Replacements: an excluded page is dropped after the full draw and replaced by the next page in
    # its kind's random order, so the other pages of the draw do not change.
    for (doc, page), why in D.EXCLUDED_PAGES.items():
        hit = [p for p in picks if p["doc_id"] == doc and p["page_number"] == page]
        if not hit:
            continue
        picks.remove(hit[0])
        k = hit[0]["kind"]
        while queue[k]:
            r = queue[k].pop(0)
            img = page_image(r.doc_id, r.page_number)
            if r.doc_id in used or (r.doc_id, r.page_number) in D.EXCLUDED_PAGES or img is None:
                continue
            used.add(r.doc_id)
            picks.append({**r.to_dict(), "page_image": str(img), "replaces": f"{doc} p{page}: {why}"})
            break
    s = pd.DataFrame(picks)
    s["kind"] = pd.Categorical(s.kind, D.KINDS)
    s = s.sort_values(["kind", "doc_id"]).reset_index(drop=True)
    s.insert(0, "sample_no", range(1, len(s) + 1))
    (D.OUT / "pages").mkdir(exist_ok=True)
    keep = {f"{r.doc_id}_page_{r.page_number}.png" for r in s.itertuples()}
    for old in (D.OUT / "pages").glob("*.png"):   # an excluded page's image leaves the folder
        if old.name not in keep:
            old.unlink()
    for r in s.itertuples():
        shutil.copy(r.page_image, D.OUT / "pages" / f"{r.doc_id}_page_{r.page_number}.png")
    s.to_csv(D.OUT / "sample_pages.csv", index=False)
    print(s.drop(columns="page_image").to_string(index=False))


if __name__ == "__main__":
    main()
