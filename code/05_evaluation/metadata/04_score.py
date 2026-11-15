"""Score one source against the answer key of a set.

  python code/05_evaluation/metadata/04_score.py validation current
  python code/05_evaluation/metadata/04_score.py validation model [prompt_version]
  python code/05_evaluation/metadata/04_score.py validation pipeline

Writes scores/{set}_{source}_long.csv (every verdict) and _summary.csv (per column, overall and per
water type group, weighted to the published tables). Only tables already in the key are scored.
"""
import sys

import pandas as pd

import defaults as D
import predictions
import scoring


def main(set_name, source, *extra):
    cfg = D.SETS[set_name]
    key = pd.read_csv(cfg["key_dir"] / "answer_key.csv", dtype=str)
    tables = pd.read_csv(cfg["tables"])
    if source == "current":
        pred = predictions.current_rows(set_name)
    elif source == "model":
        pred = predictions.model_rows(set_name, *extra)
    else:
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
