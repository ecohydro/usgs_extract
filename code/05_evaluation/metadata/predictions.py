"""Build predicted metadata tables (schema.py columns) from each source, for one set of tables.

model_rows(set_name, version)  -> the open model's answers, one row per location per table
current_rows(set_name)         -> what Annette's published per-table files say today, in the new columns
"""
import json
import re

import pandas as pd

import defaults as D
import schema as S

METHOD_TO_TYPE = [   # Annette's coord_method -> location_type
    ("printed", "coordinates"), ("plss_wellno", "well_number"), ("wellno_group", "well_number"),
    ("nwis_well", "well_number"), ("nwis_station", "station_number"), ("nwis_name", "station_name"),
    ("plss", "township_range_section"), ("gnis", "named_place"), ("b118", "basin_or_area")]
RESOLUTION = {"daily": "daily", "monthly": "monthly", "annual": "annual", "discrete": "discrete",
              "no_time_series": "no_time_series"}


def _tables(set_name):
    return pd.read_csv(D.SETS[set_name]["tables"])


def model_rows(set_name, version=None):
    version = version or D.PROMPT_VERSION
    f = D.MODEL_RUNS / f"{set_name}_{version}" / "answers.jsonl"
    ans = {(r["doc_id"], r["page_number"]): r for r in map(json.loads, open(f, encoding="utf-8"))}
    rows = []
    for t in _tables(set_name).itertuples():
        a = ans.get((int(t.doc_id), int(t.page_number)))
        tabs = (a or {}).get("parsed") or []
        mine = [x for x in tabs if isinstance(x, dict) and str(x.get("table")) == str(t.table_number)]
        locs = (mine[0].get("locations") or []) if mine else []
        for i, loc in enumerate([l for l in locs if isinstance(l, dict)], start=1):
            row = {c: "" for c in S.KEY_COLUMNS}
            row.update(doc_id=t.doc_id, page_number=t.page_number, table_number=t.table_number, location_number=i)
            for c in S.EVIDENCE + S.EVIDENCE_EXTRA + ["evidence_place", "water_source", "data_type", "date_start",
                                                    "date_end", "temporal_resolution", "construction_date"]:
                row[c] = S.clean(loc.get(c))
            v = loc.get("variables")
            row["variables"] = ";".join(v) if isinstance(v, list) else S.clean(v)
            row["location"], row["location_type"] = S.best_location(row)
            rows.append(row)
    return pd.DataFrame(rows, columns=S.KEY_COLUMNS)


def model_tables(set_name, version=None):
    """{(doc, page, table): the model's table-level answer} (measured_columns, summary_rows, measured_rows, rows_are_locations)."""
    version = version or D.PROMPT_VERSION
    f = D.MODEL_RUNS / f"{set_name}_{version}" / "answers.jsonl"
    out = {}
    for r in map(json.loads, open(f, encoding="utf-8")):
        for x in r.get("parsed") or []:
            if isinstance(x, dict) and str(x.get("table", "")).isdigit():
                out[(int(r["doc_id"]), int(r["page_number"]), int(x["table"]))] = x
    return out


def current_rows(set_name):
    """One row per table from table_level_metadata_published.csv and table_resolution.csv.
    The published files hold a point, not the printed location words, so `location` stays blank."""
    ids = ["doc_id", "page_number", "table_number"]
    t = _tables(set_name)
    pub = pd.read_csv(D.PUBLISHED_TABLES, low_memory=False)
    res = pd.read_csv(D.TABLE_RESOLUTION, low_memory=False)
    m = t.merge(pub, on=ids, how="left").merge(res[ids + ["resolution", "parameters", "n_values", "n_summary_values"]], on=ids, how="left")
    rows = []
    for r in m.itertuples():
        row = {c: "" for c in S.KEY_COLUMNS}
        row.update(doc_id=r.doc_id, page_number=r.page_number, table_number=r.table_number, location_number=1)
        method = str(r.coord_method) if isinstance(r.coord_method, str) else ""
        row["location_type"] = next((typ for pre, typ in METHOD_TO_TYPE if method.startswith(pre)), "none")
        row["water_source"] = S.clean(r.watersource_name)
        row["data_type"] = S.clean(r.water_type_final)
        fallback = str(r.date_status) in ("period_of_record", "construction_year", "undated")
        if not fallback and pd.notna(r.year_start):
            row["date_start"], row["date_end"] = str(int(r.year_start)), str(int(r.year_end)) if pd.notna(r.year_end) else ""
        if pd.notna(r.construction_start):
            row["construction_date"] = str(int(r.construction_start))
        row["temporal_resolution"] = RESOLUTION.get(str(r.resolution), "")
        row["variables"] = S.clean(r.parameters).replace(" ", "") if isinstance(r.parameters, str) else ""
        if pd.notna(r.n_values):
            row["n_measurements"] = str(int(r.n_values))
            row["n_variables"] = str(int(r.n_values + (r.n_summary_values if pd.notna(r.n_summary_values) else 0)))
        rows.append(row)
    return pd.DataFrame(rows, columns=S.KEY_COLUMNS)
