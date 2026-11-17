"""Summaries of the scores: every figure the paper reports about table digitization.

Produces (data/analysis/evaluation/digitization/summary/), for the default (lenient) and strict runs:
    by_page.csv         share of a page's printed numbers in each outcome, and how many it has
    by_kind_pooled.csv  share of a kind's printed numbers in each outcome (its test pages pooled)
    kind_weights.csv    share of published pages by kind of table
    overall.csv         corpus figure with a 95% range (see overall())
    tables.csv          printed tables found / split / merged / missed; GriTS-style means
    extras.csv          output numbers that match nothing printed
    marks.csv           printed marks kept
Run:  ~/miniconda3/envs/geofm-vlm/bin/python 05_summarize.py   (from this folder)
"""
import importlib
import numpy as np
import pandas as pd

import defaults as D
import scoring

draw = importlib.import_module("01_draw_sample")
KEY = ["variant", "system"]
COLS = scoring.RANK + ["number_right_labels_aside"]


def kind_weights():
    """Share of published pages by kind (kind of a page = kind of its largest table). Pages whose
    largest table is unclassified (8% of pages) have no stratum and are left out."""
    f = draw.page_frame()
    n = f[f.kind.isin(D.KINDS)].kind.value_counts().reindex(list(D.KINDS))
    return pd.DataFrame({"pages": n / n.sum(), "n_pages": n})


def corpus_share(per_kind, outcome, w):
    """Share of printed numbers with this outcome, expected over the corpus. Every printed number
    counts equally; a kind stands for its share of published pages (w) times the mean number of
    printed numbers on one of its test pages."""
    num = sum(w[k] * (per_kind[k][outcome] * per_kind[k].n_numbers).mean() for k in D.KINDS)
    return num / sum(w[k] * per_kind[k].n_numbers.mean() for k in D.KINDS)


def overall(bp, w):
    """corpus_share of every outcome, per run and output; 95% range = 2.5 to 97.5 percentiles over
    BOOTSTRAP_DRAWS resamples of the test pages within each kind (same resamples for every output)."""
    rows = []
    for (variant, system), g in bp.groupby(KEY):
        rng = np.random.default_rng(D.SEED)
        per_kind = {k: g[g.kind == k].sort_values("sample_no") for k in D.KINDS}
        r = {"variant": variant, "system": system, **{m: corpus_share(per_kind, m, w) for m in COLS}}
        ranged = ("right_findable", "number_right_labels_aside")
        boots = {m: [] for m in ranged}
        for _ in range(D.BOOTSTRAP_DRAWS):
            samp = {k: v.iloc[rng.integers(0, len(v), len(v))] for k, v in per_kind.items()}
            for m in ranged:
                boots[m].append(corpus_share(samp, m, w))
        for m in ranged:
            r[m + "_lo"], r[m + "_hi"] = np.percentile(boots[m], [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


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

    counts = bp[COLS].mul(bp.n_numbers, axis=0).join(bp[KEY + ["kind", "n_numbers"]])
    pk = counts.groupby(KEY + ["kind"])[COLS + ["n_numbers"]].sum()
    pk[COLS] = pk[COLS].div(pk.n_numbers, axis=0)
    pk["n_pages"] = bp.groupby(KEY + ["kind"]).size()
    pk.reset_index().to_csv(out / "by_kind_pooled.csv", index=False)

    W = kind_weights()
    W.reset_index().rename(columns={"index": "kind"}).to_csv(out / "kind_weights.csv", index=False)
    ov = overall(bp, W.pages)
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
    (m.groupby(KEY).kept.agg(["size", "mean"]).rename(columns={"size": "n_marked", "mean": "share_kept"})
      .reset_index().to_csv(out / "marks.csv", index=False))

    pd.set_option("display.width", 200)
    print(W.round(3).to_string())
    print((ov.set_index(KEY) * 100).round(1).T.to_string())
    d = pk.reset_index()
    print(d[d.variant == "default"].pivot(index="kind", columns="system", values="right_findable")
          .reindex(list(D.KINDS)).round(3).to_string())


if __name__ == "__main__":
    main()
