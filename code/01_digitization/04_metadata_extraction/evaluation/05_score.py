"""Score one source against the answer key of a set.

  python code/01_digitization/04_metadata_extraction/evaluation/05_score.py validation pipeline     the pipeline (model answers of 04_model_reader.py)
  python code/01_digitization/04_metadata_extraction/evaluation/05_score.py test pipeline --final   the test set (scored once, 2026-11-15)

Writes scores/{set}_{source}_long.csv (every verdict) and _summary.csv (per column, overall and per
water type group, weighted to the published tables). Only tables already in the key are scored.
"""
import sys

import pandas as pd

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # shared modules are in the pipeline folder
import defaults as D
import scoring


def main(set_name, source, *extra):
    cfg = D.SETS[set_name]
    key = pd.read_csv(cfg["key_dir"] / "answer_key.csv", dtype=str)
    tables = pd.read_csv(cfg["tables"])
    import pipeline
    pred = pipeline.rows(set_name, *extra)
    tag = source + ("_" + extra[0] if extra else "")
    if set_name == "test" and "--final" not in sys.argv:
        sys.exit("The test set is scored once, after Anna says go: add --final.")
    D.SCORES.mkdir(parents=True, exist_ok=True)
    pred.to_csv(D.SCORES / f"{set_name}_{tag}_predicted.csv", index=False)
    long = scoring.score(key, pred, tables)
    summary = scoring.summarize(long)
    long.to_csv(D.SCORES / f"{set_name}_{tag}_long.csv", index=False)
    summary.to_csv(D.SCORES / f"{set_name}_{tag}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{set_name} / {tag}: {long.sample_no.nunique()} tables scored")
    print(summary[summary.group == "all"].drop(columns="group").round(2).to_string(index=False))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if a != '--final'])
