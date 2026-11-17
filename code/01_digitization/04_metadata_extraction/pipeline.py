"""The pipeline, locked 2026-11-15: the open model's answers per table -> found-on-page check -> two tidy rules ->
location hierarchy. One row per location per table, schema.KEY_COLUMNS.

table_rows(doc, page, k, locs, page_norm)  one table (used by the evaluation sets and by the full run)
rows(set_name, version=None)               every table of an evaluation set
"""
import json
import re

import pandas as pd

import defaults as D
import page_input
import schema as S

_norm = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())


def on_page(value, page_norm):
    """True if the value's characters appear on the page; long values pass when most of their words do."""
    v = _norm(value)
    if not v or v in page_norm:
        return True
    words = [w for w in re.findall(r"[A-Za-z0-9]+", str(value)) if len(w) >= 3]
    return bool(words) and sum(_norm(w) in page_norm for w in words) / len(words) >= 0.8




GENERIC_SOURCE = {"groundwater", "well", "wells", "spring", "springs", "stream", "river", "reservoir", "do", "do."}


def tidy(r):
    """The two rules kept after the ablations (NOTES.md). Runoff in acre-feet printed beside discharge is a total
    of the discharge values, not a separate measurement. A generic word is not a water source name."""
    v = [x.strip() for x in re.split(r"[;,]", r["variables"]) if x.strip()]
    if "runoff_beside_discharge" in D.TIDY_RULES and r["temporal_resolution"] in ("daily", "monthly") \
            and "discharge" in v and "runoff" in v:
        v.remove("runoff")
    r["variables"] = ";".join(v)
    if "generic_water_source" in D.TIDY_RULES and r["water_source"].lower() in GENERIC_SOURCE:
        r["water_source"] = ""            # "Groundwater", "Well" is not a name
    return r


def table_rows(doc, page, k, locs, page_norm):
    """The locked pipeline for one table: the model's locations -> final rows (found-on-page check, tidy, hierarchy).
    locs: the model's location objects for this table (may be empty: one blank row)."""
    locs = [l for l in (locs or []) if isinstance(l, dict)] or [{}]
    out = []
    for i, loc in enumerate(locs, start=1):
        r = {col: "" for col in S.KEY_COLUMNS}
        r.update(doc_id=doc, page_number=page, table_number=k, location_number=i, n_locations_in_table=len(locs))
        for col in S.EVIDENCE + S.EVIDENCE_EXTRA:
            v = S.clean(loc.get(col))
            r[col] = v if (not D.REQUIRE_ON_PAGE or on_page(v, page_norm)) else ""
        for col in ("evidence_place", "data_type", "temporal_resolution", "water_source", "construction_date",
                    "date_start", "date_end"):
            r[col] = S.clean(loc.get(col))
        v = loc.get("variables")
        r["variables"] = ";".join(map(str, v)) if isinstance(v, list) else S.clean(v)
        if r["data_type"] == "Not Water Related":
            for col in ("temporal_resolution", "variables", "date_start", "date_end", "water_source"):
                r[col] = ""
        r = tidy(r)
        r["location"], r["location_type"] = S.best_location(r)
        out.append(r)
    return out


def table_locations(parsed, k):
    """The model's location objects for table k of a parsed page answer."""
    mine = [x for x in (parsed or []) if isinstance(x, dict) and str(x.get("table")) == str(k)]
    return (mine[0].get("locations") or []) if mine else []


def page_norm(text):
    return _norm(page_input.plain(text))


def rows(set_name, version=None):
    version = version or D.PROMPT_VERSION
    f = D.MODEL_RUNS / f"{set_name}_{version}" / "answers.jsonl"
    ans = {(r["doc_id"], r["page_number"]): r for r in map(json.loads, open(f, encoding="utf-8"))}
    out = []
    for t in pd.read_csv(D.SETS[set_name]["tables"]).itertuples():
        doc, page, k = int(t.doc_id), int(t.page_number), int(t.table_number)
        a = ans.get((doc, page)) or {}
        out += table_rows(doc, page, k, table_locations(a.get("parsed"), k), page_norm(page_input.page_text(doc, page)[0]))
    return pd.DataFrame(out, columns=S.KEY_COLUMNS)
