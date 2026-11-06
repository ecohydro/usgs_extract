"""
Temporal resolution of ONE digitized table, read from the table itself (2026-11-05).

Replaces the page-level LLM field `temporal_resolution` (one guess per page, free text) for
counting. Each table gets one PRIMARY class describing its observations:

  daily          day-of-month rows (1-31) with month context: the classic USGS day x month
                 grid, split grids (Day | Feb | Mar | Day.1 | Feb.1 ...), sediment/temperature
                 day tables, two-level headers (Day | OCTOBER ...)
  monthly        Month rows (October ... September) x Max/Min/Mean/Runoff, or year rows x
                 month columns (Water year | Oct | Nov ...)
  annual         one row per year (annual peaks, yearly totals/means), no month columns
  discrete       individual dated measurements at irregular times: field discharge
                 measurements, water levels on given dates, water-quality samples, peaks above
                 a base ("Date | Gage height | Discharge", "Feb. 6, 1961 | 14.6")
  no_time_series no time dimension in the table: well logs, well inventories, rating tables,
                 analyses without dates
  unclassified   none of the above could be decided from the table or its caption

Summary rows (Total / Mean / Max / Min / Runoff / "The year") are DERIVED and never define
the class; 19_value_counts.py counts their values separately. A monthly table printed beside
the daily table it was computed from is flagged derived in 20_temporal_resolution.py.

`classify(rows, caption)` -> (class, evidence). `rows` = CSV rows incl. the header row.
"""
import re

MONTH = r'(?:jan|feb|mar|apr|may|june?|july?|aug|sept?|oct|nov|dec)[a-z]*'
MONTH_CELL = re.compile(rf'(?i)^\W*{MONTH}\.?(?:\.\d+)?\W*$')          # "Feb.", "OCTOBER.1"
MONTH_START = re.compile(rf'(?i)^\W*{MONTH}\b\.?(?:\s+(?:1[89]\d\d))?\W*$')  # "October 1917"
DAY_NUM = re.compile(r'^\s*([1-9]|[12]\d|3[01])(?:\.0|\.+)?\s*$')  # "1", "1.0", "1.." 
DAY_HEAD = re.compile(r'(?i)^\W*day\W*(?:\.\d+)?$')
YEAR_CELL = re.compile(r'^\s*(1[89]\d\d)(?:\.0)?(?:\s*-\s*(?:\d{1,4}))?\W*$')  # 1951, 1877-78
# a date with a day: "Feb. 20", "Oct. 10, 1950", "6, 1961", "3 - 6 -51", "8/5/52"
DATE_DAY = re.compile(rf'(?i)\b{MONTH}\.?\s*\d{{1,2}}(?!\d)|^\s*\d{{1,2}},\s*1[89]\d\d|'
                      r'\b\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2}(?![\dA-Za-z])')  # not well no. "11/4-30N2" 
DATE_HEAD = re.compile(r'(?i)^\W*(date|dates|date of (?:collection|measurement|sampling)|'
                       r'date\s+and\s+time)\W*(?:\.\d+)?$')
SUMMARY = re.compile(r'(?i)^\W*(total|mean|max|min|monthly|ac\.?-?\s*ft|acre|runoff|cfsm|'
                     r'in\.|inches|calendar|water\b|average|sum|yearly|annual|peak|extremes|'
                     r'period|the year|year\b)')
NO_TS = re.compile(r'(?i)thickness|rating\s+table|\blog of\b|drillers?\W+log|material|'
                   r'character of|formation|constituents?\b|specific capacity')
RATING = re.compile(r'(?i)(?:rating|capacity)\s+tables?\b')   # lookup tables, not records
MONTH_HEAD = re.compile(r'(?i)^\W*months?\W*$')
WELL_INV = re.compile(r'(?i)owner|cost\s+of|method\s+of\s+lift|class\s+of\s+well|'
                      r'depth\s+of\s+well|type\s+of\s+well|diameter|casing|year\s+com')
CAP = [('daily', re.compile(r'(?i)\bdaily\b|\bday\b(?!s)')),
       ('monthly', re.compile(r'(?i)\bmonthly\b')),
       ('annual', re.compile(r'(?i)\bannual\b|\byearly\b|annual\s+peak')),
       ('discrete', re.compile(r'(?i)discharge\s+measurements|measurements\s+of|analys[ie]s|'
                               r'water\s+levels?|peak\s+discharges?\s+above|miscellaneous'))]


def _clean(c):
    c = (c or '').strip()
    return '' if re.match(r'(?i)^unnamed:', c) else c


def classify(rows, caption=''):
    rows = [[_clean(c) for c in r] for r in rows if r]
    if not rows:
        return 'unclassified', 'empty'
    ncol = max(len(r) for r in rows)
    grid = [r + [''] * (ncol - len(r)) for r in rows]
    head = grid[:4]                       # CSV header + repeated multi-level header rows
    head_text = ' '.join(c for r in head for c in r)
    cap = caption or ''
    if RATING.search(cap[-300:]):
        return 'no_time_series', 'rating table caption'  # gage height -> discharge lookup

    # ---- daily: a column of day numbers 1..31 counting up, with month context ----
    months_in_head = {re.sub(r'\W|\d', '', c.lower())[:3] for r in head for c in r if MONTH_CELL.match(c)}
    for j in range(ncol):
        col = [r[j] for r in grid]
        days = [int(DAY_NUM.match(c).group(1)) for c in col if DAY_NUM.match(c)]
        if len(days) < 8:
            continue
        steps = sum(b == a + 1 for a, b in zip(days, days[1:]))
        if steps < 0.7 * (len(days) - 1):
            continue
        labeled = any(DAY_HEAD.match(r[j]) for r in head)
        if labeled or len(months_in_head) >= 2 or CAP[0][1].search(cap):
            return 'daily', 'day column' + (' (labeled)' if labeled else '')

    # ---- monthly: month rows (not followed by a day/date), or year rows x month columns ----
    first = [r[0] for r in grid]
    mrows = [i for i, c in enumerate(first) if MONTH_START.match(c)]
    if len(mrows) >= 6:
        nxt = [grid[i][1] if ncol > 1 else '' for i in mrows]
        dated = sum(bool(DAY_NUM.match(c) or DATE_DAY.search(c)) for c in nxt)
        if dated < 0.3 * len(mrows):
            return 'monthly', 'month rows'
    year_rows = sum(bool(YEAR_CELL.match(c)) for c in first[1:])
    if len(months_in_head) >= 6 and year_rows >= 2:
        return 'monthly', 'year rows x month columns'

    # ---- annual: first column mostly years, no month columns, not a well inventory ----
    nonempty = [c for c in first[1:] if c and not SUMMARY.match(c)]
    if year_rows >= 3 and year_rows >= 0.6 * max(len(nonempty), 1) and len(months_in_head) < 3 \
            and not WELL_INV.search(head_text):
        return 'annual', 'year rows'

    # ---- discrete: a Date column or cells holding dates with a day ----
    if any(DATE_HEAD.match(c) for r in head for c in r):
        return 'discrete', 'date column'
    n_date_cells = sum(bool(DATE_DAY.search(c)) for r in grid for c in r if len(c) < 30)
    if n_date_cells >= 3:
        return 'discrete', 'dated cells'

    # ---- no time dimension ----
    if NO_TS.search(head_text) or NO_TS.search(cap[-200:]) or WELL_INV.search(head_text):
        return 'no_time_series', 'log / inventory / rating / analyses headers'

    # ---- a "Month" label column (monthly summaries with < 6 month rows) ----
    if any(MONTH_HEAD.match(r[0]) for r in head):
        return 'monthly', 'month column'

    # ---- caption as last table-level evidence ----
    for cls, rx in CAP:
        if rx.search(cap[-300:]):
            return cls, 'caption'
    return 'unclassified', ''
