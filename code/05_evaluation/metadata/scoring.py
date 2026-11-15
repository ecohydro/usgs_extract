"""Compare a predicted metadata table with the answer key, column by column.

score(key, pred, weights) -> long table: one row per key location x graded column, with a verdict:
  correct | wrong | missed (the page gives it, the output is blank) | extra (output gives a value the
  page does not) | both_blank
Key rows of one table are matched to predicted rows by their identifiers (well / station number, names),
else by position. Every table counts once: its rows share the table's sampling weight.
"""
import re

import pandas as pd

import defaults as D
import schema as S

COUNT_TOLERANCE = 0.05      # a count is right within 5% (or 2), proposal
COUNT_ABS = 2


def norm(v):
    return re.sub(r"[^a-z0-9]", "", S.clean(v).lower())


def same_text(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return a == b
    return a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a)


def same_location(a, b):
    """Printed coordinates are compared by their digits (so "La 34°03'49" and "Lat 34°03'49" agree); other
    location words by text."""
    da, db = re.sub(r"\D", "", S.clean(a)), re.sub(r"\D", "", S.clean(b))
    if re.search(r"lat|long|°", S.clean(a), re.I) and len(da) >= 8:
        return da == db
    return same_text(a, b)


def year(v):
    m = re.match(r"(\d{4})", S.clean(v))
    return m.group(1) if m else ""


def same_date(key, pred):
    """Right if the two agree as far as the coarser one goes (1952-10 and 1952-10-01 are the same date here);
    a year-only output against a finer key is scored by the *_year reading, not this one."""
    key, pred = S.clean(key), S.clean(pred)
    return bool(key) and bool(pred) and len(pred) >= 7 and (pred.startswith(key) or key.startswith(pred)) or (key == pred and key != "")


def _same_date_strict(key, pred):
    """Right if the output agrees with the key at the key's own granularity (key 1916-10: 1916-10-01 is right)."""
    key, pred = S.clean(key), S.clean(pred)
    return bool(key) and pred.startswith(key)


def var_set(v):
    v = S.clean(v)
    same = {"gage_height": "stage", "elevation": "stage"}   # stream gage height and lake elevation: one notion (stage)
    return {same.get(x.strip(), x.strip()) for x in re.split(r"[;,]", v) if x.strip()}


def same_count(key, pred):
    try:
        k, p = float(key), float(pred)
    except (TypeError, ValueError):
        return False
    return abs(k - p) <= max(COUNT_ABS, COUNT_TOLERANCE * k)


COMPARE = {
    "location": same_location, "location_type": lambda a, b: S.clean(a) == S.clean(b),
    "water_source": same_text, "data_type": lambda a, b: S.clean(a) == S.clean(b),
    "date_start": same_date, "date_end": same_date,
    "temporal_resolution": lambda a, b: S.clean(a) == S.clean(b),
    "variables": lambda a, b: var_set(a) == var_set(b),
    "construction_date": lambda a, b: year(a) == year(b) and year(a) != "",
}
EXTRA = {   # secondary, more lenient readings reported next to the strict one
    "date_start_year": ("date_start", lambda a, b: year(a) == year(b) and year(a) != ""),
    "date_end_year": ("date_end", lambda a, b: year(a) == year(b) and year(a) != ""),
    "variables_overlap": ("variables", lambda a, b: bool(var_set(a) & var_set(b))),
}
ID_FIELDS = ["well_number", "station_number", "station_name", "water_source", "place_name", "coordinates"]


def blankish(field, v):
    v = S.clean(v)
    if field == "location_type":
        return v in ("", "none")
    if field in ("n_measurements", "n_variables"):
        return v in ("", "0", "0.0")
    return v == "" or v == "cannot tell"


def match_rows(krows, prows):
    """Greedy match of key rows to predicted rows: shared identifier text first, then position."""
    pairs, used = {}, set()
    for i, k in enumerate(krows):
        best, best_score = None, 0
        for j, p in enumerate(prows):
            if j in used:
                continue
            sc = sum(2 if f in ("well_number", "station_number") else 1
                     for f in ID_FIELDS if norm(k.get(f)) and same_text(k.get(f), p.get(f)))
            if sc > best_score:
                best, best_score = j, sc
        if best is not None:
            pairs[i] = best
            used.add(best)
    for i, k in enumerate(krows):          # position fallback
        if i in pairs:
            continue
        try:
            pos = int(float(k.get("location_number") or i + 1)) - 1
        except ValueError:
            pos = i
        if 0 <= pos < len(prows) and pos not in used:
            pairs[i] = pos
            used.add(pos)
        elif len(krows) == 1 and len(prows) >= 1 and 0 not in used:
            pairs[i] = 0
            used.add(0)
    return pairs


def score(key, pred, tables):
    """key, pred: DataFrames of strings with doc_id, page_number, table_number + schema columns.
    tables: the set's table list (sample_no, stratum, weight)."""
    key, pred = key.fillna("").astype(str), pred.fillna("").astype(str)
    ids = ["doc_id", "page_number", "table_number"]
    pg = {k: g.to_dict("records") for k, g in pred.groupby(ids)}
    out = []
    for r in tables.itertuples():
        tid = (str(r.doc_id), str(r.page_number), str(r.table_number))
        krows = key[(key.doc_id == tid[0]) & (key.page_number == tid[1]) & (key.table_number == tid[2])].to_dict("records")
        if not krows:
            continue
        prows = pg.get(tid, [])
        pairs = match_rows(krows, prows)
        n_key = S.clean(krows[0].get("n_locations_in_table")) or str(len(krows))
        base = dict(sample_no=r.sample_no, stratum=r.stratum, weight=r.weight / len(krows))
        out.append({**base, "weight": r.weight, "field": "n_locations", "key": n_key, "pred": str(len(prows)),
                    "verdict": "correct" if n_key == str(len(prows)) else ("missed" if not prows else "wrong")})
        for i, k in enumerate(krows):
            p = prows[pairs[i]] if i in pairs else {}
            tests = [(f, f, fn) for f, fn in COMPARE.items()] + [(name, f, fn) for name, (f, fn) in EXTRA.items()]
            for name, f, fn in tests:
                kv, pv = S.clean(k.get(f)), S.clean(p.get(f))
                kb, pb = blankish(f, kv), blankish(f, pv)
                if kb and pb:
                    v = "both_blank"
                elif kb:
                    v = "extra"
                elif pb:
                    v = "missed"
                else:
                    v = "correct" if fn(kv, pv) else "wrong"
                out.append({**base, "field": name, "key": kv, "pred": pv, "verdict": v})
    return pd.DataFrame(out)


def summarize(long):
    """Per field, weighted: of the tables where the page gives the value, the share extracted correctly
    (found), and of the values the output gives, the share correct (right_when_given)."""
    rows = []
    for (grp, f), d in [((g, f), d) for g in ["all"] + sorted(long.stratum.unique())
                        for f, d in (long if g == "all" else long[long.stratum == g]).groupby("field", sort=False)]:
        w = d.groupby("verdict").weight.sum()
        g = lambda k: float(w.get(k, 0.0))
        has, given = g("correct") + g("wrong") + g("missed"), g("correct") + g("wrong") + g("extra")
        rows.append(dict(group=grp, field=f, n_rows=len(d), n_page_gives=int((d.verdict.isin(["correct", "wrong", "missed"])).sum()),
                         found=g("correct") / has if has else float("nan"),
                         right_when_given=g("correct") / given if given else float("nan"),
                         share_missed=g("missed") / has if has else float("nan"),
                         share_wrong=g("wrong") / has if has else float("nan"),
                         share_extra=g("extra") / (g("extra") + g("both_blank")) if (g("extra") + g("both_blank")) else float("nan")))
    return pd.DataFrame(rows)
