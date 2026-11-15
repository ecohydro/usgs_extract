"""Score every typed page against the three outputs, for the default settings and each variant.

Produces (data/analysis/evaluation/digitization/scores/, every row carries system + variant):
    cells.csv    one row per printed number: outcome, what the output held, why
    tables.csv   one row per printed table: found / missed / split / merged, GriTS-style scores
    extras.csv   one row per output table: numbers that match nothing printed
Reads everything in the truth folder on every run (pages without typed tables are listed, not scored).
Run:  ~/miniconda3/envs/geofm-vlm/bin/python code/05_evaluation/digitization/02_score.py
"""
import pandas as pd

import defaults as D
import loaders
import scoring


def params(variant):
    P = {k: getattr(D, k) for k in ("LABEL_MATCH_MIN_RATIO", "REPAIR_NUMBER_FORMAT", "MARK_MAX_CHARS", "IGNORE_LABEL_FOOTNOTES",
                                    "LINK_MIN_EQUAL_CELLS", "LINK_MIN_EQUAL_SHARE", "SPLIT_MERGE_MIN_SHARE")}
    P.update(D.VARIANTS[variant])
    return P


def main():
    sample = pd.read_csv(D.SAMPLE)
    cells, tables, extras, skipped = [], [], [], []
    for s in sample.itertuples():
        if not (D.TRUTH / f"{s.doc_id}_page_{s.page_number}_tables.csv").exists():
            skipped.append(f"{s.doc_id} p{s.page_number}")
            continue
        truth = loaders.truth_tables(s.doc_id, s.page_number)
        for system in D.SYSTEMS:
            outputs = loaders.output_tables(system, s.doc_id, s.page_number)
            for variant in D.VARIANTS:
                c, t, e = scoring.score_page(truth, outputs, params(variant))
                key = {"sample_no": s.sample_no, "doc_id": s.doc_id, "page_number": s.page_number,
                       "kind": s.kind, "system": system, "variant": variant or "default"}
                cells += [{**key, **x} for x in c]
                tables += [{**key, **x} for x in t]
                extras += [{**key, **x} for x in e]
    out = D.OUT / "scores"
    out.mkdir(exist_ok=True)
    cells, tables, extras = map(pd.DataFrame, (cells, tables, extras))
    cells.to_csv(out / "cells.csv", index=False)
    tables.to_csv(out / "tables.csv", index=False)
    extras.to_csv(out / "extras.csv", index=False)
    print("pages scored:", cells[["doc_id", "page_number"]].drop_duplicates().shape[0],
          "| not typed yet:", ", ".join(skipped) or "none")
    d = cells[cells.variant == "default"]
    print(pd.crosstab([d.doc_id, d.page_number], [d.system, d.outcome]).T.to_string())


if __name__ == "__main__":
    main()
