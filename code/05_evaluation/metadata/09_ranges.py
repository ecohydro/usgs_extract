"""95% ranges for the scores of one scored run, from its stored verdicts (nothing is scored again).

  python code/05_evaluation/metadata/09_ranges.py test_pipeline_v5

Reads scores/{run}_long.csv and writes scores/{run}_ranges.csv: per column the share found (all tables,
weighted) with its 95% range, plus the mean of the eight graded columns without construction_date.
Method and settings: scoring.ranges, defaults.RANGE_*.
"""
import sys

import pandas as pd

import defaults as D
import scoring

run = sys.argv[1]
r = scoring.ranges(pd.read_csv(D.SCORES / f"{run}_long.csv", dtype={"key": str, "pred": str}))
r.to_csv(D.SCORES / f"{run}_ranges.csv", index=False)
print((r.set_index("field") * 100).round(1).to_string())
