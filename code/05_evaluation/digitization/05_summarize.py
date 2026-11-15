"""Summaries of the scores, for every output and variant in the store.

Produces (data/analysis/evaluation/digitization/summary/):
    by_page.csv      share of printed numbers in each outcome, per page
    by_kind.csv      mean over pages within each kind of table
    overall.csv      corpus-wide figures: kinds weighted by their share of published pages, and
                     by their share of recorded values (kind of a page = kind of its largest
                     table), each with a 95% range from resampling pages within kind
    kind_weights.csv the weights
    tables.csv       printed tables found / split / merged / missed; GriTS-style means
    extras.csv       output numbers that match nothing printed
    marks.csv        printed marks kept
Run:  ~/miniconda3/envs/geofm-vlm/bin/python code/05_evaluation/digitization/05_summarize.py
"""
import importlib
import numpy as np
import pandas as pd

import defaults as D
import scoring

draw = importlib.import_module("01_draw_sample")
KEY = ["variant", "system"]


def kind_weights():
    """Share of each kind in the published corpus, two ways: by pages (what a random draw of
    pages would hold) and by recorded values (what a random draw of numbers would hold).
    Pages whose largest table is unclassified (8% of pages) have no stratum and are left out."""
    f = draw.page_frame()
    f = f[f.kind.isin(D.KINDS)]
    pages = f.kind.value_counts().reindex(list(D.KINDS))
    values = f.groupby("kind").n_values.sum().reindex(list(D.KINDS))
    return pd.DataFrame({"pages": pages / pages.sum(), "values": values / values.sum(),
                         "n_pages": pages, "n_values": values})


def main():
    out = D.OUT / "summary"
    out.mkdir(exist_ok=True)
    c = pd.read_csv(D.OUT / "scores" / "cells.csv", keep_default_na=False)
    t = pd.read_csv(D.OUT / "scores" / "tables.csv")
    e = pd.read_csv(D.OUT / "scores" / "extras.csv")
    page = ["sample_no", "doc_id", "page_number", "kind"]
    bp = (c.groupby(KEY + page).outcome.value_counts(normalize=True).unstack(fill_value=0.0)
           .reindex(columns=scoring.RANK, fill_value=0.0))
    bp["number_right_labels_aside"] = bp[scoring.RANK[:4]].sum(axis=1)
    bp["n_numbers"] = c.groupby(KEY + page).size()
    bp = bp.reset_index()
    bp.to_csv(out / "by_page.csv", index=False)
    cols = scoring.RANK + ["number_right_labels_aside"]
    bk = bp.groupby(KEY + ["kind"])[cols].mean()
    bk["n_pages"] = bp.groupby(KEY + ["kind"]).size()
    bk.reset_index().to_csv(out / "by_kind.csv", index=False)

    W = kind_weights()
    W.reset_index().rename(columns={"index": "kind"}).to_csv(out / "kind_weights.csv", index=False)
    rng = np.random.default_rng(D.SEED)
    rows = []
    for basis in ("pages", "values"):
        w = W[basis]
        for (variant, system), g in bp.groupby(KEY):
            est = {m: sum(w[k] * g[g.kind == k][m].mean() for k in D.KINDS) for m in cols}
            boots = {m: [] for m in ("right_findable", "number_right_labels_aside")}
            per_kind = {k: g[g.kind == k] for k in D.KINDS}
            for _ in range(D.BOOTSTRAP_DRAWS):
                samp = {k: v.iloc[rng.integers(0, len(v), len(v))] for k, v in per_kind.items()}
                for m in boots:
                    boots[m].append(sum(w[k] * samp[k][m].mean() for k in D.KINDS))
            r = {"weighted_by": basis, "variant": variant, "system": system, **est}
            for m in boots:
                r[m + "_lo"], r[m + "_hi"] = np.percentile(boots[m], [2.5, 97.5])
            r["unweighted_mean_of_pages"] = g.right_findable.mean()
            r["pooled_right_findable"] = (c[(c.variant == variant) & (c.system == system)].outcome == "right_findable").mean()
            rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(out / "overall.csv", index=False)

    ts = t.groupby(KEY).status.value_counts().unstack(fill_value=0)
    g = t.groupby(KEY)[["layout_recall", "layout_precision", "content_recall", "content_precision"]].mean()
    for k in ("layout", "content"):
        g[k + "_f"] = 2 * g[k + "_recall"] * g[k + "_precision"] / (g[k + "_recall"] + g[k + "_precision"])
    ts.join(g).reset_index().to_csv(out / "tables.csv", index=False)
    ex = e.groupby(KEY)[["n_output_numbers", "n_not_printed"]].sum()
    ex["share_not_printed"] = ex.n_not_printed / ex.n_output_numbers
    ex.reset_index().to_csv(out / "extras.csv", index=False)
    m = c[c.has_mark.astype(str) == "True"].assign(kept=lambda d: d.mark_kept.astype(str) == "True")
    m.groupby(KEY).kept.agg(["size", "mean"]).rename(columns={"size": "n_marked", "mean": "share_kept"}).reset_index().to_csv(out / "marks.csv", index=False)

    pd.set_option("display.width", 200)
    print(W.round(3).to_string())
    print(ov[["weighted_by", "variant", "system", "right_findable", "right_findable_lo", "right_findable_hi",
              "number_right_labels_aside", "pooled_right_findable"]].round(3).to_string(index=False))
    d = bk.reset_index()
    print(d[d.variant == "default"].pivot(index="kind", columns="system", values="right_findable").reindex(list(D.KINDS)).round(2).to_string())


if __name__ == "__main__":
    main()
