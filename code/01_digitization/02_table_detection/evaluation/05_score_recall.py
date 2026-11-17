"""Page-level recall of the table detector on the listed pages, across thresholds.

A listed page is 'found' at threshold t when any detection on it scores >= t.
Produces OUT/recall_curve.csv (one row per list version x threshold, with a range from resampling reports),
OUT/recall_by_report.csv and OUT/listed_pages_scored.csv (one row per listed page: position, score, label).
Run: python 05_score_recall.py
List versions: 'cleaned' (the result of record) = every listed page with a score, minus pages labelled
no_table in OUT/listed_page_labels.csv (borderline pages stay in); 'as_listed' = none removed.
"""
import numpy as np, pandas as pd
import defaults as D


def scored_pages():
    pm = pd.read_csv(D.OUT / "page_map_final.csv", dtype={"report": str})
    det = pd.read_csv(D.OUT / "detections.csv", dtype={"report": str})[["report", "pdf_page", "image_source", "best_score"]]
    df = pm.merge(det, on=["report", "pdf_page"], how="left")
    labels = D.OUT / "listed_page_labels.csv"
    if labels.exists():
        lab = pd.read_csv(labels, dtype={"report": str})[["report", "pdf_page", "label"]]
        df = df.merge(lab, on=["report", "pdf_page"], how="left")
    else:
        df["label"] = np.nan
    return df


def recall_with_range(df, t, rng):
    """Recall at t and the 2.5-97.5% range over N_BOOTSTRAP resamples of whole reports."""
    hit = (df.best_score >= t).to_numpy()
    codes, _ = pd.factorize(df.report)
    k = codes.max() + 1
    hits = np.bincount(codes, weights=hit, minlength=k)
    ns = np.bincount(codes, minlength=k)
    draws = rng.integers(0, k, size=(D.N_BOOTSTRAP, k))
    boot = hits[draws].sum(1) / ns[draws].sum(1)
    return hit.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    df = scored_pages()
    df.to_csv(D.OUT / "listed_pages_scored.csv", index=False)
    counts = dict(listed_pages=len(df), no_pdf_position=int(df.pdf_page.isna().sum()),
                  no_score=int((df.pdf_page.notna() & df.best_score.isna()).sum()))
    print(counts)
    base = df.dropna(subset=["best_score"])
    versions = {"as_listed": base, "cleaned": base[base.label != "no_table"]}
    rows = []
    for name, d in versions.items():
        rng = np.random.default_rng(D.SEED)
        for t in sorted(set(D.CURVE_THRESHOLDS) | set(D.THRESHOLDS_USED)):
            r, lo, hi = recall_with_range(d, t, rng)
            rows.append(dict(list_version=name, threshold=t, recall=r, low=lo, high=hi,
                             pages=len(d), found=int((d.best_score >= t).sum()), reports=d.report.nunique()))
    curve = pd.DataFrame(rows)
    curve.to_csv(D.OUT / "recall_curve.csv", index=False)
    by = base.groupby("report").agg(pages=("best_score", "size"),
                                    **{f"found_{t}": ("best_score", lambda s, t=t: int((s >= t).sum())) for t in D.THRESHOLDS_USED})
    by.to_csv(D.OUT / "recall_by_report.csv")
    print(curve[curve.threshold.isin(D.THRESHOLDS_USED)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
