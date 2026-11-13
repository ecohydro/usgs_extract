"""
Temporal resolution of ONE digitized table, read from the table itself (2026-11-05).

Replaces the page-level LLM field `temporal_resolution` (one guess per page, free text) for
counting. Each table gets one PRIMARY class describing its observations:

  daily          day-of-month rows (1-31) with month context: the classic USGS day x month
                 grid, split grids (Day | Feb | Mar | Day.1 | Feb.1 ...), sediment/temperature
                 day tables, two-level headers (Day | OCTOBER ...)
  monthly        Month rows (October ... September) x Max/Min/Mean/Runoff, or year rows x
                 month columns (Water year | Oct | Nov ...)
  annual         one row per year (annual peaks, yearly totals/means), no month columns;
                 also annual max/min per consecutive water year, peaks above a base and annual
                 maximum (caption), and stream RATING tables (one per year / period of use)
  discrete       individual dated measurements at irregular times: field discharge
                 measurements, water levels on given dates, water-quality samples
                 ("Date | Gage height | Discharge", "Feb. 6, 1961 | 14.6"); also selected
                 years with gaps plus a Date column (e.g. lists of revised peaks)
                 EXCEPTION: dated samples taken about once a month (months in sequence) are
                 MONTHLY, because the record was designed as a monthly series
  no_time_series no time dimension in the table: well logs, well inventories, reservoir
                 capacity tables, analyses without dates

Rating-table, peak, and monthly-sample rules follow Annette's grading decisions (2026-11-13).
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
DATE_HEAD = re.compile(r'(?i)^\W*(date|dates|date of(?:\s+(?:collection|measurement|sampling))?|'
                       r'date\s+and\s+time)\W*(?:\.\d+)?$')
SUMMARY = re.compile(r'(?i)^\W*(total|mean|max|min|monthly|ac\.?-?\s*ft|acre|runoff|cfsm|'
                     r'in\.|inches|calendar|water\b|average|sum|yearly|annual|peak|extremes|'
                     r'period|the year|year\b)')
NO_TS = re.compile(r'(?i)thickness|capacity\s+table|\blog of\b|drillers?\W+log|material|'
                   r'character of|formation|constituents?\b|specific capacity|aquifers?\b|'
                   r'probabilit|exceed(?:ed|ance)|recurrence|storage\s+capacity')
RATING = re.compile(r'(?i)rating\s+tables?\b')        # stream rating: one per year / period of use
CAPACITY = re.compile(r'(?i)capacity\s+tables?\b')    # reservoir capacity: lookup, no time
PEAK_ANNUAL = re.compile(r'(?i)peak\s+discharges?\s+above\s+(?:a\s+)?base|annual\s+maximum|'
                         r'(?:maximum|peak)\s+and\s+minimum\s+discharges?\s+for\s+the\s+water\s+years')
MONTH_IDX = {m: i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                                          'sep', 'oct', 'nov', 'dec'])}
MONTH_TOKEN = re.compile(rf'(?i)\b({MONTH})\b')
MONTH_END = re.compile(rf'(?i)^\W*{MONTH}\.?\s*(2[89]|3[01])\W*$')   # "Oct. 31", "Sept 30"
YEAR_HEAD = re.compile(r'(?i)\byear\b')
MONTH_HEAD = re.compile(r'(?i)^\W*months?\W*$')
WELL_INV = re.compile(r'(?i)owner|cost\s+of|method\s+of\s+lift|class\s+of\s+well|'
                      r'depth\s+of\s+well|type\s+of\s+well|diameter|casing|year\s+com')
CAP = [('daily', re.compile(r'(?i)\bdaily\b|\bday\b(?!s)')),
       ('monthly', re.compile(r'(?i)\bmonthly\b')),
       ('annual', re.compile(r'(?i)\bannual\b|\byearly\b|annual\s+peak')),
       ('discrete', re.compile(r'(?i)discharge\s+measurements|measurements\s+of|analys[ie]s|'
                               r'water\s+levels?|miscellaneous'))]


def _monthly_sequence(cols):
    """True if the date columns hold about one sample per month: the month names advance one
    month at a time (>= 6 names, >= 70% of steps exactly +1, wrapping Dec -> Jan) AND >= 80% of
    the sample cells name their own month. The second test rejects weekly field measurements
    printed as "Oct. 4 17 31" followed by bare days "27", "9" (month named once per month).
    Split month | day columns (JAN. | 4, 1966) are handled by skipping columns with no month
    names; pure year cells ("1926") and "-Con." labels are not samples."""
    cols = [[c for c in col if c and not YEAR_CELL.match(c) and not DATE_HEAD.match(c)
             and not re.search(r'(?i)con\.?\W*$', c)] for col in cols]
    cols = [col for col in cols if any(MONTH_TOKEN.search(c) for c in col)]
    cells = [c for col in cols for c in col]
    if not cells or sum(bool(MONTH_TOKEN.search(c)) for c in cells) < 0.8 * len(cells):
        return False
    ms = [MONTH_IDX.get(m.group(1).lower()[:3]) for c in cells for m in MONTH_TOKEN.finditer(c)]
    ms = [m for m in ms if m is not None]
    if len(ms) < 6:
        return False
    steps = sum((b - a) % 12 == 1 for a, b in zip(ms, ms[1:]))
    return steps >= 0.7 * (len(ms) - 1)


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
    date_cols = [j for j in range(ncol) if any(DATE_HEAD.match(r[j]) for r in head)]
    # rating table: gage height -> discharge lookup. Not if the table has a Date column (the
    # caption block can hold the rating-table note of the station above a measurements table)
    if CAPACITY.search(cap[-300:]) and not date_cols:
        return 'no_time_series', 'capacity table caption'
    if (RATING.search(cap[-300:]) or RATING.search(head_text)) and not date_cols:
        return 'annual', 'rating table'

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
    # transposed daily grid: a header row of day numbers 1, 2, 3 ... with month rows
    month_rows_any = sum(bool(MONTH_START.match(r[0])) for r in grid)
    for r in head:
        days = [int(DAY_NUM.match(c).group(1)) for c in r[1:] if DAY_NUM.match(c)]
        steps = sum(b == a + 1 for a, b in zip(days, days[1:]))
        if len(days) >= 8 and steps >= 0.7 * (len(days) - 1) and month_rows_any >= 3:
            return 'daily', 'day header row x month rows'

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
        # selected years with gaps + a Date column = individual events (e.g. revised peaks)
        ys = [int(YEAR_CELL.match(c).group(1)) for c in first[1:] if YEAR_CELL.match(c)]
        consec = sum(b - a == 1 for a, b in zip(ys, ys[1:]))
        if date_cols and consec <= 0.5 * (len(ys) - 1):
            return 'discrete', 'selected years with dates'
        return 'annual', 'year rows'
    # a "(Water) year" column that is not the first (e.g. Station | Area | Water year | ...)
    if len(months_in_head) < 3 and not WELL_INV.search(head_text) and not date_cols:
        for j in range(1, ncol):
            if not any(YEAR_HEAD.search(r[j]) for r in head):
                continue
            col = [r[j] for r in grid[1:]]
            ys = sum(bool(YEAR_CELL.match(c)) for c in col)
            if ys >= 3 and ys >= 0.6 * max(sum(bool(c) for c in col), 1):
                return 'annual', 'year column'

    # ---- peaks above a base / annual maximum (caption) = an annual peak series ----
    if PEAK_ANNUAL.search(cap[-300:]):
        return 'annual', 'peaks above base / annual maximum caption'

    # ---- discrete: a Date column or cells holding dates with a day ----
    if date_cols:
        # end-of-month dates (Sept. 30, Oct. 31, Nov. 30 ...) = monthly reservoir/well readings
        cells = [r[j] for j in date_cols for r in grid if r[j] and not DATE_HEAD.match(r[j])]
        dated = [c for c in cells if DATE_DAY.search(c)]
        ends = sum(bool(MONTH_END.match(c)) for c in dated)
        if ends >= 6 and ends >= 0.7 * len(dated):
            return 'monthly', 'end-of-month dates'
        # about one sample a month, months in sequence = designed monthly series
        if _monthly_sequence([[r[j] for r in grid[1:]] for j in date_cols]):
            return 'monthly', 'one sample per month'
        return 'discrete', 'date column'
    n_date_cells = sum(bool(DATE_DAY.search(c)) for r in grid for c in r if len(c) < 30)
    if n_date_cells >= 3:
        return 'discrete', 'dated cells'

    # ---- no time dimension ----
    if NO_TS.search(head_text) or NO_TS.search(cap[-200:]) or WELL_INV.search(head_text):
        return 'no_time_series', 'log / inventory / capacity / analyses headers'

    # ---- a "Month" label column (monthly summaries with < 6 month rows) ----
    if any(MONTH_HEAD.match(r[0]) for r in head):
        return 'monthly', 'month column'

    # ---- caption as last table-level evidence ----
    for cls, rx in CAP:
        if rx.search(cap[-300:]):
            return cls, 'caption'
    return 'unclassified', ''
