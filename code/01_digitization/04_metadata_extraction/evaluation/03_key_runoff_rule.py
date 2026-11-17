"""Apply one written key rule (KEY_FORMAT.md) uniformly to the answer keys, by script, without reading them.

Rule: runoff printed beside discharge in a daily or monthly table is a total or conversion of the discharge values
(a summary value), so it is not listed in `variables`. The test key and the first validation key were drafted before
the rule was written down; the second validation key already follows it.
Each key is backed up once as answer_key_before_runoff_rule.csv; prints only how many rows changed.
"""
import re
import shutil

import pandas as pd

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # shared modules are in the pipeline folder
import defaults as D

for name, s in D.SETS.items():
    f = s["key_dir"] / "answer_key.csv"
    bak = s["key_dir"] / "answer_key_before_runoff_rule.csv"
    if not bak.exists():
        shutil.copy(f, bak)
    k = pd.read_csv(f, dtype=str, keep_default_na=False)
    n = 0
    for i, r in k.iterrows():
        v = [x.strip() for x in re.split(r"[;,]", r["variables"]) if x.strip()]
        if r["temporal_resolution"] in ("daily", "monthly") and "discharge" in v and "runoff" in v:
            v.remove("runoff")
            k.at[i, "variables"] = ";".join(v)
            n += 1
    k.to_csv(f, index=False)
    print(f"{name}: {n} of {len(k)} key rows changed, in {k.sample_no.nunique()} tables")
