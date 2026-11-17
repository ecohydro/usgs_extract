"""Every number of the metadata evaluation that the paper cites, collected from the output files.

  python code/01_digitization/04_metadata_extraction/evaluation/08_paper_numbers.py   -> paper_numbers.csv (number, value, read from)

Nothing is computed here beyond counting rows and dividing; each line names the file it was read from and
README.md names the script that wrote that file.
"""
import pandas as pd

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # shared modules are in the pipeline folder
import defaults as D

out = []
add = lambda what, value, source: out.append(dict(number=what, value=value, read_from=source))
pct = lambda x: f"{100 * x:.1f}%"

t = pd.read_csv(D.SETS["test"]["tables"])
add("test set: tables", len(t), "test_set_tables.csv")
add("test set: reports", t.doc_id.nunique(), "test_set_tables.csv")
for s, n in D.STRATA.items():
    add(f"test set: tables drawn, {s}", int((t.stratum == s).sum()), "test_set_tables.csv")
add("test set: most tables from one report", int(t.groupby("doc_id").size().max()), "test_set_tables.csv")
frame = t.groupby("stratum").stratum_tables.first()
for s in D.STRATA:
    add(f"weight of group in the overall score, {s}", pct(frame[s] / frame.sum()), "test_set_tables.csv (stratum_tables)")
k = pd.read_csv(D.SETS["test"]["key_dir"] / "answer_key.csv", dtype=str)
add("answer key: rows (locations)", len(k), "key/answer_key.csv")
add("answer key: tables checked by an author", int(pd.read_csv(D.SETS["test"]["key_dir"] / "checked.csv").sample_no.nunique()), "key/checked.csv")
for name in ("validation", "validation2"):
    add(f"{name} set: tables", len(pd.read_csv(D.SETS[name]["tables"])), f"{name}_set_tables.csv")

r = pd.read_csv(D.SCORES / "test_pipeline_v5_ranges.csv").set_index("field")
for f in r.index:
    add(f"test, share found, {f}", f"{pct(r.found[f])} ({pct(r.low[f])} to {pct(r.high[f])})", "scores/test_pipeline_v5_ranges.csv")
s = pd.read_csv(D.SCORES / "test_pipeline_v5_summary.csv")
for row in s.itertuples():
    if row.n_page_gives:
        add(f"test, {row.group}, {row.field}", f"{pct(row.found)} (rows where the page gives it: {row.n_page_gives})",
            "scores/test_pipeline_v5_summary.csv")
o = pd.read_csv(D.OUT / "on_page_check.csv").set_index("set")
add("location values dropped by the found-on-page check, both validation sets",
    f"{pct(o.share_dropped['both'])} ({o.dropped_not_on_page['both']} of {o.location_values_given['both']})", "on_page_check.csv")
m = pd.read_csv(D.OUT / "full_run" / "metadata_run_summary.csv")
for row in m.itertuples():
    add(f"full collection: {row.what}", int(row.n), "full_run/metadata_run_summary.csv")
pd.DataFrame(out).to_csv(D.OUT / "paper_numbers.csv", index=False)
print(pd.DataFrame(out).head(40).to_string(index=False))
