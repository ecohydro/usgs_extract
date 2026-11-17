"""How much the found-on-page check removes: of the location values the model gave for the tables of the two
validation sets, the share whose characters are not on the page (pipeline.on_page) and are therefore dropped.

  python code/01_digitization/04_metadata_extraction/evaluation/07_on_page_check.py
  -> on_page_check.csv            one row per set, and both
  -> on_page_check_by_column.csv  both sets, per kind of location evidence
"""
import json

import pandas as pd

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # shared modules are in the pipeline folder
import defaults as D
import page_input
import pipeline
import schema as S

rows = []
for name in ("validation", "validation2"):
    f = D.MODEL_RUNS / f"{name}_{D.PROMPT_VERSION}" / "answers.jsonl"
    ans = {(r["doc_id"], r["page_number"]): r for r in map(json.loads, open(f, encoding="utf-8"))}
    for t in pd.read_csv(D.SETS[name]["tables"]).itertuples():
        doc, page, k = int(t.doc_id), int(t.page_number), int(t.table_number)
        norm = pipeline.page_norm(page_input.page_text(doc, page)[0])
        for loc in pipeline.table_locations((ans.get((doc, page)) or {}).get("parsed"), k):
            if not isinstance(loc, dict):
                continue
            for col in S.EVIDENCE + S.EVIDENCE_EXTRA:
                v = S.clean(loc.get(col))
                if v:
                    rows.append(dict(set=name, column=col, dropped=not pipeline.on_page(v, norm)))
d = pd.DataFrame(rows)


def table(by):
    g = pd.concat([d, d.assign(set="both")]) if by == "set" else d
    out = g.groupby(by, sort=False).dropped.agg(location_values_given="size", dropped_not_on_page="sum").reset_index()
    out["share_dropped"] = out.dropped_not_on_page / out.location_values_given
    return out


table("set").to_csv(D.OUT / "on_page_check.csv", index=False)
table("column").to_csv(D.OUT / "on_page_check_by_column.csv", index=False)
print(table("set").to_string(index=False), table("column").to_string(index=False), sep="\n\n")
