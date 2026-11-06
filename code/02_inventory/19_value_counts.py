"""
Per-table count of recorded VALUES, a better measurement estimate than CSV row counts.

Why: number_rows (one per printed line) undercounts measurements. A daily USGS table has one
row per day and one column per month, so a row holds up to 12 values. A 3,000-table sample
(2026-11-04) found ~8 numeric values per row.

Definition of a value: a numeric token in a DATA cell of a DATA row.
  - The first column (day / year / station / well label) is never counted.
  - Columns whose header marks them as site description or time labels are excluded
    (EXCLUDE_HEADER: date, year, station, well no., owner, depth of well, altitude of land
    surface, township/range/section, ...). "Depth to water" and "gage height" are kept.
  - Columns where >= 80% of the numbers are years (1800-1990) are excluded (date columns).
  - Summary rows are excluded (first cell starts with Total, Mean, Max, Min, Ac-ft, Runoff,
    Water [year], Peak, ...). In day-by-month grids only rows labeled with day numbers count.
  - A cell with several numbers (Reducto merged rows, e.g. "7 8" / ".5 .4") counts each one.
  - A lone "o"/"O" counts (zero in old USGS type). Footnote marks ("e", "*", "a") are allowed.
This is still an estimate: it can't tell a measurement from, e.g., a computed column it doesn't
recognize, and it misses cells Reducto dropped.

Calendar check (automatic validation, daily tables only):
  A day-by-month table (first column = days 1-31, >= 6 month headers) should hold one value per
  day for each month with record. For each month column with any value, expected = days in
  that month (Feb = 28 or 29, either accepted). Reported per table as cal_expected /
  cal_found. Partial-record months (station began mid-month) legitimately fall short, so
  the share of month columns that match exactly is the main accuracy figure.

Outputs (cached; delete or pass --force to re-scan):
  data/analysis/table_value_counts.csv            one row per table CSV
  data/analysis/table_value_counts_summary.csv    totals: all / published / catalog, by water type
  data/analysis/table_value_counts_calendar.csv   calendar-check summary
"""
import csv
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_resolution import classify, DAY_NUM, DAY_HEAD  # noqa: E402

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
DIGITIZED = ROOT / 'data/digitized'
OUT = A / 'table_value_counts.csv'
FILE_RE = re.compile(r'^(\d+)_page_(\d+)_table(\d+)\.csv$')

NUM = re.compile(r'^[-+(]?[*#e]?\$?(?:\d+\.?\d*|\.\d+)\)?[a-zA-Z*#]?$|^[Oo]$')
DAY = re.compile(r'(?:[1-9]|[12]\d|3[01])')
YEAR = re.compile(r'^(18\d\d|19[0-8]\d|1990)$')
SUMMARY = re.compile(r'(?i)^\W*(total|mean|max|min|monthly|month|ac\.?-?\s*ft|acre|runoff|cfsm|'
                     r'in\.|inches|calendar|water\b|average|sum|yearly|annual|peak|extremes|period)')
EXCLUDE_HEADER = re.compile(
    r'(?i)\b(date|dates|year|yr|years?\s+drilled|station|site|well\s*no|well\s*number|number|no\.|'
    r'owner|depth\s+of\s+well|well\s+depth|diameter|casing|drilled|altitude|elevation\s+of\s+land|'
    r'land\s+surface|measuring\s+point|township|range|section|sec\.|lat|long|map|page|'
    r'hour|time|location|name|remarks?|use|type|period|record|side|'
    r'difference|differ-?\s*ence|distance|apart|ratio|deviation|regression|coefficient|equation)\b'
    r'|^\W*totals?\W*$')  # (hand check 2026-11-04: derived / site-description columns)
KEEP_HEADER = re.compile(r'(?i)depth\s+to\s+water|water\s+level|gage\s+height|discharge|stage|'
                         r'adsorption|sodium')  # "sodium adsorption ratio" is a measurement
# a cell that is a date ("Nov. 17", "9, 1961", "1929-36") holds no measurement
MONTH_WORD = re.compile(r'(?i)\b(?:jan|feb|mar|apr|may|june?|july?|aug|sept?|oct|nov|dec)[a-z]*\b\.?')
YEAR_RANGE = re.compile(r'(?<!\d)1[89]\d\d\s*-\s*(?:\d{2}|1[89]\d\d)?(?![\d.])')
YEAR_IN_CELL = re.compile(r'(?<![\d.])1[89]\d\d(?![\d.])')


def is_date_cell(c):
    c = c.strip()
    if not c:
        return False
    if MONTH_WORD.search(c) or YEAR_RANGE.search(c):
        return True
    return bool(YEAR_IN_CELL.search(c)) and len(tokens(c)) >= 2  # "9, 1961"

MONTHS = [('oct', 31), ('nov', 30), ('dec', 31), ('jan', 31), ('feb', 28), ('mar', 31),
          ('apr', 30), ('may', 31), ('june', 30), ('jun', 30), ('july', 31), ('jul', 31),
          ('aug', 31), ('sept', 30), ('sep', 30)]


def tokens(cell):
    return [x for x in cell.replace(',', '').split() if NUM.match(x)]


def month_days(header):
    h = header.strip().lower().rstrip('.')
    for m, d in MONTHS:
        if h == m or h.startswith(m + ' ') or h == m + 'ember' or (m == 'sept' and h.startswith('sept')):
            return m, d
    return None


def metrics(path: Path, caption: str = '') -> dict:
    try:
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            rows = list(csv.reader(f))
    except Exception:
        rows = []
    header = rows[0] if rows else []
    data = rows[1:]
    ncol = max((len(r) for r in rows), default=0)
    hdr = [(header[j] if j < len(header) else '') for j in range(ncol)]
    resolution, evidence = classify(rows, caption)

    summary = [bool(r) and bool(SUMMARY.match(r[0].strip())) for r in data]
    body = [r for r, s in zip(data, summary) if not s]
    summary_rows = [r for r, s in zip(data, summary) if s]
    # day-by-month grid: only rows labeled with day numbers are data (footer lines are not)
    months = {j: month_days(hdr[j]) for j in range(1, ncol)}
    months = {j: v for j, v in months.items() if v}
    day_rows = [r for r in body if r and r[0].split() and all(DAY.fullmatch(t)
                                                              for t in r[0].split())]
    is_daily = len(months) >= 6 and len(day_rows) >= 25   # standard grid (calendar check)
    if is_daily:
        body = day_rows
    elif resolution == 'daily':
        # other daily layouts (split grids "Day | Feb | Mar | Day.1 ...", day labels "1.0" /
        # "1..", two-level headers): rows labeled with a day number are data, other non-summary
        # rows (repeated header rows, notes) are not counted (2026-11-05)
        body = [r for r in body if r and DAY_NUM.match(r[0])]

    # column decisions
    excluded = set()
    for j in range(1, ncol):
        h = re.sub(r'(?i)unnamed:\s*\d+', '', hdr[j])
        if DAY_HEAD.match(h):
            excluded.add(j)  # repeated day-label column of a split daily grid ("Day.1")
            continue
        if EXCLUDE_HEADER.search(h) and not KEEP_HEADER.search(h):
            excluded.add(j)
            continue
        toks = [t for r in body if j < len(r) for t in tokens(r[j])]
        if len(toks) >= 3 and sum(bool(YEAR.match(t)) for t in toks) / len(toks) >= 0.8:
            excluded.add(j)
            continue
        filled = [r[j] for r in body if j < len(r) and r[j].strip()]
        if not is_daily and filled and sum(map(is_date_cell, filled)) >= 0.5 * len(filled):
            excluded.add(j)  # date column without a date header (incl. its bare day numbers)

    n_values = n_merged = 0
    col_counts = {}
    for r in body:
        for j in range(1, min(len(r), ncol)):
            if j in excluded:
                continue
            k = 0 if (not is_daily and is_date_cell(r[j])) else len(tokens(r[j]))
            n_values += k
            n_merged += k > 1
            col_counts[j] = col_counts.get(j, 0) + k
    # derived values (Total / Mean / Max / Min / Runoff rows): reported, never counted as values
    n_summary_values = sum(len(tokens(r[j])) for r in summary_rows
                           for j in range(1, min(len(r), ncol)) if j not in excluded)

    # calendar check for day x month tables
    rec = {}
    if is_daily:
        exp = found = exact = cols = 0
        for j, (m, d) in months.items():
            c = col_counts.get(j, 0)
            if c == 0:
                continue
            cols += 1
            exp += d
            found += c
            exact += c == d or (m == 'feb' and c == 29)
        rec = {'cal_month_cols': cols, 'cal_exact_cols': exact, 'cal_expected': exp, 'cal_found': found}

    return {'n_data_rows': len(data), 'n_summary_rows': sum(summary), 'n_cols': ncol,
            'n_excluded_cols': len(excluded), 'n_values': n_values, 'n_merged_cells': n_merged,
            'is_daily_grid': is_daily, 'resolution': resolution, 'resolution_evidence': evidence,
            'n_summary_values': n_summary_values, **rec}


def scan():
    files = list(DIGITIZED.rglob('*_table*.csv'))
    print(f'{len(files):,} table CSVs', flush=True)
    td = pd.read_csv(A / 'table_dates.csv', usecols=['id', 'page_number', 'table_number', 'caption'],
                     low_memory=False)
    captions = {(int(a), int(b), int(c)): d for a, b, c, d in td.itertuples(index=False)
                if isinstance(d, str)}
    recs = []
    for i, p in enumerate(files):
        if i % 20000 == 0:
            print(f'  {i:,}', flush=True)
        m = FILE_RE.match(p.name)
        if m:
            key = (int(m[1]), int(m[2]), int(m[3]))
            recs.append({'doc_id': key[0], 'page_number': key[1], 'table_number': key[2],
                         **metrics(p, captions.get(key, ''))})
    v = pd.DataFrame(recs).sort_values(['doc_id', 'page_number', 'table_number'])
    v.to_csv(OUT, index=False)
    return v


def summarize(v):
    key = ['doc_id', 'page_number', 'table_number']
    pub = pd.read_csv(A / 'table_level_metadata_published.csv', usecols=key + ['number_rows', 'water_type_final'])
    trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv')
    trib = trib[trib.withhold.astype(bool)].rename(columns={'id': 'doc_id'})[['doc_id', 'page_number']]
    p = pub.merge(v, on=key, how='left')
    p['withheld'] = p.set_index(['doc_id', 'page_number']).index.isin(
        trib.set_index(['doc_id', 'page_number']).index)

    rows = []
    for scope, d in [('published', p), ('catalog (published minus tribal)', p[~p.withheld]),
                     ('tribal withheld', p[p.withheld])]:
        for wt, g in [('ALL', d)] + list(d.groupby('water_type_final')):
            rows.append({'scope': scope, 'water_type': wt, 'tables': len(g),
                         'rows (number_rows)': int(g.number_rows.sum()),
                         'values': int(g.n_values.sum()),
                         'values_per_row': round(g.n_values.sum() / max(g.number_rows.sum(), 1), 2),
                         'daily_grid_tables': int(g.is_daily_grid.fillna(False).astype(bool).sum())})
    s = pd.DataFrame(rows)
    s.to_csv(A / 'table_value_counts_summary.csv', index=False)
    print(s[s.water_type.eq('ALL')].to_string(index=False))
    print(s[s.scope.eq('published')].to_string(index=False))

    dg = v[v.is_daily_grid.fillna(False).astype(bool)]
    cal = {'daily grid tables (all CSVs)': len(dg),
           'month columns with record': int(dg.cal_month_cols.sum()),
           'month columns exactly one value per day': int(dg.cal_exact_cols.sum()),
           'share exact': round(dg.cal_exact_cols.sum() / max(dg.cal_month_cols.sum(), 1), 3),
           'values found / expected': round(dg.cal_found.sum() / max(dg.cal_expected.sum(), 1), 3),
           'tables within 2% of expected': round(((dg.cal_found - dg.cal_expected).abs()
                                                  <= 0.02 * dg.cal_expected).mean(), 3)}
    c = pd.Series(cal, name='value')
    c.to_csv(A / 'table_value_counts_calendar.csv', header=True)
    print(c.to_string())


def main():
    v = pd.read_csv(OUT) if OUT.exists() and '--force' not in sys.argv else scan()
    summarize(v)


if __name__ == '__main__':
    main()
