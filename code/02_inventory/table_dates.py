"""
Table-level date extraction from Reducto table blocks + page text.

Protocol (agreed with Annette, 2026-10-29). For each table, dates are taken
from the FIRST tier that yields a valid year:

  A  date_column   a column whose header (or label cell) names a date/year
                   ("Date", "Date of measurement", "Water year", "Year",
                   "Calendar year"), excluding drilled/completed/recurrence
                   columns. Bare years in such columns count (USGS tables put
                   the year as a divider row: "1918 -Continued.", "1930").
  B  table_text    dates written anywhere else in the table: column headers
                   ("Water year October 1969 to September 1970", "Date (1951)")
                   and body cells with an explicit date form (month + year,
                   numeric date, water-year range "1941-42").
  C  caption       text blocks just above the table, back to the previous
                   table ("Discharge, in second-feet, 1941-42").
  D  page_text     the rest of the page's text (backup; "better than nothing").

In every tier, clauses about OTHER dates are removed first: period of record
("Records available", "period of record", "since"), well construction
("drilled", "completed", "dug"), extremes ("highest", "maximum ... known"),
datums, publication/report references, recurrence intervals. Period-of-record
years are kept separately (por_start / por_end) because they show a station
existed, not that its data were restored.

Valid years: YEAR_MIN <= year <= document publication year (same rule as
date_parsing.py). Water years: "1941-42" -> 1941..1942; "year ending
September 30, 1942" -> 1941..1942.
"""
import re

from html_table import parse_table, column_headers

YEAR_MIN = 1800
YEAR_MAX = 1980

MONTHS = (r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|'
          r'sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)')
Y4 = r'(1[89]\d{2})'

# ---- clauses that describe something other than the table's data dates ----
POR = re.compile(r'records?\s+(?:are\s+)?avail|period\s+of\s+record|periods?\s+of\s+records|'
                 r'records?\s+(?:began|begin|start)|record\s+since|\bsince\b|\bbegan\b|'
                 r'to\s+(?:the\s+)?current\s+year|to\s+date\b', re.I)
# a clause naming the table's own water / calendar year is data, even if it also mentions
# e.g. "land-surface datum" ("WATER LEVEL, IN FEET BELOW LAND-SURFACE DATUM, WATER YEAR ...")
DATA_YEAR = re.compile(r'water\s+year|year\s+ending|calendar\s+year', re.I)
EXCL = re.compile(
    r'drill|complet|construct|\bdug\b|\bbored\b|deepen|install|\bbuilt\b|finished|prior\s+to|'
    r'maximum|minimum|time\s+basis|no\s+flow|'
    r'highest|lowest|\b\w?[ir][gc]hest\b|below\s+(?:lsd|land[- ]?surface)|above\s+lsd|'  # OCR "Fighest"
    r'\bextremes?\b|known|recurrence|interval|\bdatum\b|adjustment|'
    r'revis|water[- ]supply\s+paper|\bpaper\b|bulletin|\breport|publish|\bprinted\b|'
    r'established|discontinued|abandoned|destroyed|average|\bmean\s+for\b|'
    r'copyright|census|reprint|\bsurvey\s+of\b|\bact\s+of\b|decree|adjudicat|\bpage\b',
    re.I)
DATE_HEAD = re.compile(r'\bdate|\byear|\bwtr\.?\s*yr|calend|calen?d[ae]r|\bmonth\b|\btime\s+of\b|\bperiod\b', re.I)
HEAD_EXCL = re.compile(r'drill|complet|\bcom-\W*$|construct|\bdug\b|deepen|recurrence|interval|\byears?\s*\)|'
                       r'number\s+of\s+years|\bage\b|install|abandon|reported|\bperiod\s+of\s+record|'
                       r'record\s+of\s+period|\bknown\b|avail', re.I)
HYPHEN_BREAK = re.compile(r'([a-z])-\s+[:;|]?\s*([a-z])')  # "com- pleted", "Sep- tember", "min- imum"


MONTH_SPLIT = re.compile(r'\b(jan|feb|sep|sept|oct|nov|dec|novem|decem|septem)\s+'
                         r'(uary|ruary|tember|ember|ber|ober)\b', re.I)  # OCR "Sep tember"


def dehyphen(text):
    return MONTH_SPLIT.sub(r'\1\2', HYPHEN_BREAK.sub(r'\1\2', text or ''))

WELL_DESC_HEAD = re.compile(r'depth|diameter|altitude|\buse\b|casing|owner', re.I)
# Well-inventory tables (one row per well). An UNLABELED column of bare years in such a table
# is the year completed: Reducto often drops that header ("Year com- pleted."), e.g. WSP 60
# doc 3133, doc 2294, doc 2749 pumping plants (2026-11-05 review of pre-1890 GW dates).
WELL_INVENTORY = re.compile(r'owner|cost\s+of|method\s+of\s+lift|class\s+of\s+well|depth\s+of\s+well|'
                            r'type\s+of\s+well|diameter|casing', re.I)

_ABBR_END = re.compile(r'(?:\b' + MONTHS + r'|\bcalif|\bco|\bno|\bsec|\bft|\bmi|\bt|\br|\bvol|\bp|\bpl|\b[a-z])$', re.I)


def split_clauses(text):
    """Split text on ';' and sentence ends, but not after 'Oct.', 'Calif.', 'T.' etc."""
    out = []
    for part in re.split(r';|\n', text):
        start = 0
        for m in re.finditer(r'\.\s+(?=[A-Z0-9])', part):
            before = part[start:m.start()]
            if _ABBR_END.search(before.rstrip()):
                continue
            nxt = part[m.end():m.end() + 1]
            if nxt.isdigit() and re.search(r'\b[A-Za-z]{1,4}$', before):
                continue  # OCR'd month abbreviation ("Har. 1, 1944", "AUR. 2, 1949")
            out.append(part[start:m.end()])
            start = m.end()
        out.append(part[start:])
    return [c.strip() for c in out if c.strip()]


# ---- year patterns ----
P_WY_END = re.compile(r'year\s+ending\s+' + MONTHS + r'\.?\s*\d{0,2},?\s*' + Y4, re.I)
P_RANGE = re.compile(r'(?<![\d,.])' + Y4 + r'\s*(?:-|–|to|and)\s*(\d{2}|1[89]\d{2})(?![\d,])', re.I)
P_MONTH_YEAR = re.compile(r'\b' + MONTHS + r'\.?\s*(?:\d{1,2}\s*(?:-\s*\d{1,2})?,?\s*)?' + Y4 + r'(?!\d)', re.I)
# numeric dates must not run into a letter: "6/11-13J1" is a state well number, not 6/11/13
# Both separators must be the same: "4/28-1802" and "6/11-13J1" are well numbers (T/R-section).
P_NUM_DATE4 = re.compile(r'(?<![\d/])(?:0?[1-9]|1[0-2])\s*(?P<s>[-/])\s*(?:0?[1-9]|[12]\d|3[01])\s*(?P=s)\s*'
                         + Y4 + r'(?![\dA-Za-z])')
P_ISO = re.compile(r'(?<!\d)' + Y4 + r'-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![\dA-Za-z])')
P_NUM_DATE2 = re.compile(r'(?<![\d.,/])(?:0?[1-9]|1[0-2])\s*(?P<s>[-/])\s*(?:0?[1-9]|[12]\d|3[01])\s*(?P=s)\s*'
                         r'(\d{2})(?![\d.,A-Za-z])')
P_BARE = re.compile(r'(?<![\d,.$])(?<!\d-)(?<!\d/)(?<![A-Za-z]-)' + Y4 + r'(?![\d,]|\.\d)')  # not "4/28-1802" (well ID)


def _ok(y, cap):
    return YEAR_MIN <= y <= cap


def years_in(text, cap, allow_bare=False, allow_two_digit=False):
    """Return list of years (ints) for date expressions in text."""
    ys = []
    s = text
    for m in P_WY_END.finditer(s):
        y = int(m.group(1))
        ys += [y - 1, y]
    s = P_WY_END.sub(' ', s)
    for m in P_RANGE.finditer(s):
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else (a // 100) * 100 + int(b)
        if a <= b <= a + 120:
            ys += [a, b]
    s = P_RANGE.sub(' ', s)
    for pat in (P_MONTH_YEAR, P_NUM_DATE4, P_ISO):
        for m in pat.finditer(s):
            ys.append(int(m.groups()[-1]))
        s = pat.sub(' ', s)
    if allow_two_digit:
        for m in P_NUM_DATE2.finditer(s):
            ys.append(1900 + int(m.groups()[-1]))
        s = P_NUM_DATE2.sub(' ', s)
    if allow_bare:
        ys += [int(m.group(1)) for m in P_BARE.finditer(s)]
    return [y for y in ys if _ok(y, cap)]


# USGS station descriptions are organised in headed sections ("Records available.-",
# "PERIOD OF RECORD.--", "EXTREMES.--", "REMARKS.--"). A whole section inherits the
# meaning of its heading, so "Water temperatures: July 1961 ..." under PERIOD OF RECORD
# is period of record even though the sentence itself does not say so.
HEADING = re.compile(r"(?:(?<=\s)|^)([A-Z][A-Za-z ,'()/-]{2,40}?)\s*\.\s*-{1,2}\s*")
SECT_POR = re.compile(r'records?\s+available|period\s+of\s+record', re.I)
SECT_EXCL = re.compile(r'extremes?|remarks|revisions?|\bgage\b|average|drainage|location|cooperation|'
                       r'accuracy|diversions?|regulation|maximum|minimum|established|history|datum|'
                       r'water[- ]quality|explanation|footnote', re.I)
CONSTRUCTION = re.compile(r'drill|complet|\bcom-\W*$|construct|\bdug\b|\bbored\b|deepen|install|\bbuilt\b|finished', re.I)


def _sections(text):
    marks = list(HEADING.finditer(text))
    if not marks:
        return [('', text)]
    out = [('', text[:marks[0].start()])]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), text[m.end():end]))
    return out


def clean_text(text):
    """Split into clauses; return (kept_clauses, por_clauses, excluded_clauses)."""
    keep, por, excl = [], [], []
    # sections never run past a block boundary (blocks are joined with newlines)
    secs = [s for line in dehyphen(text).split('\n') for s in _sections(line)]
    for head, body in secs:
        clauses = split_clauses(body)
        if head and SECT_POR.search(head):
            por += clauses
            continue
        if head and SECT_EXCL.search(head):
            excl += clauses
            continue
        for c in clauses:
            if POR.search(c):
                por.append(c)
            elif EXCL.search(c) and not DATA_YEAR.search(c):
                excl.append(c)
            else:
                keep.append(c)
    return keep, por, excl


def _years_from_clauses(clauses, cap, **kw):
    ys, quotes = [], []
    for c in clauses:
        y = years_in(c, cap, **kw)
        if y:
            ys += y
            quotes.append(c[:120])
    return ys, quotes


def date_columns(cols, body):
    """Indices of date/year columns. Also detects a label in the first body rows."""
    idx = {}
    well_desc = any(WELL_DESC_HEAD.search(h or '') for h in cols)
    for i, h in enumerate(cols):
        h = h or ''
        if DATE_HEAD.search(h) and not HEAD_EXCL.search(h):
            if len(h.split()) > 8 and not re.match(r'\W*(date|year|water\s+year|calendar)', h, re.I):
                continue  # a table title spread over the columns ("DISCHARGE ..., WATER YEAR ...")
            bare_year = re.fullmatch(r'\W*years?\W*', h, re.I) is not None
            if bare_year and well_desc:
                continue  # "Year" in a well-description table = year completed
            idx[i] = 0
    for ri, row in enumerate(body[:3]):
        for i, cell in enumerate(row):
            if i not in idx and re.match(r'\s*date\b', cell or '', re.I) and not HEAD_EXCL.search(cell) \
                    and len(cell) < 40:
                idx[i] = ri + 1  # date column starts below this label row
    return idx


def construction_columns(cols):
    """Columns holding year drilled / completed (incl. a bare 'Year' in well tables)."""
    well_desc = any(WELL_DESC_HEAD.search(h or '') for h in cols)
    out = set()
    for i, h in enumerate(cols):
        h = h or ''
        if CONSTRUCTION.search(h) and DATE_HEAD.search(h):
            out.add(i)
        elif well_desc and re.fullmatch(r'\W*years?\W*', h, re.I):
            out.add(i)
    return out


def _label_cells(row):
    """Short label cells at the start of a row (rotated tables put 'Date' in cell 0-2)."""
    return [(i, c) for i, c in enumerate(row[:3]) if c and len(c) < 40]


def date_rows(body):
    """Rows of a ROTATED table whose label says Date / Year (values run across the row).
    Returns {row_index: label_cell_index}; also the set of rows labeled period of record /
    recurrence / known floods (excluded from tier B)."""
    drows, xrows = {}, set()
    for ri, row in enumerate(body):
        if CONSTRUCTION.search(dehyphen(' '.join(c for _, c in _label_cells(row)))):
            continue  # rotated "Year | com- pleted" row = year completed, not dates
        for i, c in _label_cells(row):
            c2 = dehyphen(c)
            if HEAD_EXCL.search(c2) or POR.search(c2):
                if DATE_HEAD.search(c2) or POR.search(c2) or re.search(r'recurrence|interval', c2, re.I):
                    xrows.add(ri)
                break
            if re.fullmatch(r'\W*(?:dates?|years?|water\s+years?|date\s+of\s+\w+)\W*', c2, re.I):
                vals = [v for v in row[i + 1:] if v and v.strip() and v.strip().lower() != c2.strip().lower()]
                with_year = [v for v in vals if re.search(r'(?<!\d)1[89]\d{2}(?!\d)', v)]
                if len(vals) >= 2 and len(with_year) >= 0.5 * len(vals):  # not a summary row of flows
                    drows[ri] = i
                break
    return drows, xrows


BARE_YEAR_CELL = re.compile(r'\W*\??\s*(1[89]\d{2})\W*')
QUANTITY_HEAD = re.compile(r'feet|cfs|second|acre|discharge|gage|depth|level|ft\b|inches|elev|altitude', re.I)


def year_columns(cols, body, cap):
    """Unlabeled columns whose cells are nearly all bare years (continuation pages without
    headers). Returns {col: True if the row also describes well construction}."""
    out = {}
    width = max([len(r) for r in body], default=0)
    # data grids (day-by-month, max/min/mean) hold values that can look like years (1850 cfs)
    top = ' '.join(cols + [c for r in body[:2] for c in r])
    if re.search(r'\b(?:day|max|min|mean|' + MONTHS + r')\b', top, re.I):
        return out
    for ci in range(width):
        h = cols[ci] if ci < len(cols) else ''
        if QUANTITY_HEAD.search(h or ''):
            continue
        vals = [r[ci] for r in body if ci < len(r) and r[ci] and r[ci].strip()]
        ys = [BARE_YEAR_CELL.fullmatch(v) for v in vals]
        good = [int(m.group(1)) for m in ys if m and _ok(int(m.group(1)), cap)]
        if len(set(good)) < 0.4 * len(good):
            continue  # many repeats: measured values, not a list of years
        if len(good) >= 3 and len(good) >= 0.8 * len(vals):
            row_text = ' '.join(' '.join(r) for r in body[:10] + [cols])
            inventory = WELL_INVENTORY.search(dehyphen(' '.join(cols + [c for r in body[:2] for c in r])))
            out[ci] = bool(CONSTRUCTION.search(row_text) or inventory)
    return out


def drop_isolated(ys, quotes):
    """Drop a stray year > 30 years from every other year (OCR / ID fragments such as
    '4/28-1802'), when >= 2 other distinct years exist. Years written with a month name
    ('Dec. 1861' flood peaks) are never dropped."""
    distinct = sorted(set(ys))
    if len(distinct) < 3:
        return ys
    named = {int(m.group(1)) for q in quotes for m in P_MONTH_YEAR.finditer(q)}
    bad = {y for y in distinct if y not in named and min(abs(y - o) for o in distinct if o != y) > 30}
    return [y for y in ys if y not in bad]


EARLY_YEAR = 1850  # before this, a year needs corroboration (see plausible_years)


def _runs(years):
    """Years that sit in a run of >= 3 consecutive distinct years."""
    s, out = set(years), set()
    for y in s:
        if {y - 1, y + 1} <= s or {y + 1, y + 2} <= s or {y - 2, y - 1} <= s:
            out.add(y)
    return out


def plausible_years(ys, quotes):
    """Drop OCR century errors and uncorroborated very early years (added 2026-11-04 after
    pre-1850 dates appeared in Fig 7: '1834' printed for 1934, well IDs '35W-1823', a bare
    '1800' value in a data column).
    1. Century mirror: if the table has an 18xx block and a 19xx block >= 30 years apart, and
       at least half of the 18xx years have their +100 year in the 19xx block, the 18xx
       block is a misread of the 19xx block and is dropped.
    2. A year before EARLY_YEAR is kept only if written with a month name, as a water-year
       range ("1849-50"), or in a run of >= 3 consecutive years."""
    distinct = sorted(set(ys))
    early = [y for y in distinct if y < 1900]
    late = [y for y in distinct if y >= 1900]
    bad = set()
    if early and late and min(late) - max(early) >= 30 and \
            sum((y + 100) in late for y in early) >= 0.5 * len(early):
        bad |= set(early)
    named = {int(m.group(1)) for q in quotes for m in P_MONTH_YEAR.finditer(q)}
    ranged = {int(m.group(1)) for q in quotes for m in P_RANGE.finditer(q)}
    runs = _runs(distinct)
    bad |= {y for y in distinct if y < EARLY_YEAR and y not in named | ranged | runs}
    # 3. century mirror for very early years only (e.g. 1803/1804/1808 in a 1879-1913 table)
    vearly = [y for y in distinct if y < EARLY_YEAR]
    if vearly and sum((y + 100) in distinct for y in vearly) >= 0.5 * len(vearly):
        bad |= set(vearly)
    # 4. a very early year > 30 years from every other year is an OCR error even with a month
    #    name ('Aug. 9, 1808' among 1898-1916 dates)
    for y in vearly:
        others = [o for o in distinct if o != y]
        if others and min(abs(y - o) for o in others) > 30:
            bad.add(y)
    # 5. round values: a column of numbers like 1800, 1810, 1850, 1870 (elevations, rating
    #    values) is not a list of years. If >= 4 bare (not month-named, not range) years are
    #    mostly multiples of 10, drop the bare multiples of 10 before 1900.
    bare = [y for y in distinct if y not in named | ranged]
    tens = [y for y in bare if y % 10 == 0]
    if len(bare) >= 4 and len(tens) >= 0.6 * len(bare):
        bad |= {y for y in tens if y < 1900}
    return [y for y in ys if y not in bad]


def table_dates(html, caption, page_text, cap):
    """Dates for one table. Returns dict with year_start/end, tier, quotes, period of record."""
    cap = min(cap or YEAR_MAX, YEAR_MAX)
    head, body = parse_table(html)
    cols = [dehyphen(c) for c in column_headers(head)]
    width = max([len(cols)] + [len(r) for r in body]) if (cols or body) else 0
    dcols = date_columns(cols, body)
    cons_cols = construction_columns(cols)
    drows, xrows = date_rows(body)
    ycols = year_columns(cols, body, cap) if not dcols and not drows else {}
    cons_cols |= {c for c, is_cons in ycols.items() if is_cons}
    por_clauses, excl_clauses = [], []
    res = {'year_start': None, 'year_end': None, 'date_tier': None, 'date_quote': None,
           'n_date_values': 0, 'date_columns': '; '.join(cols[i] for i in dcols if i < len(cols))[:200]}

    def is_merged(row):
        vals = [c for c in row if c]
        return len(vals) > 1 and len(set(vals)) == 1

    def finish(ys, quotes, tier):
        ys = plausible_years(drop_isolated(ys, quotes), quotes)
        if ys:
            res.update(year_start=min(ys), year_end=max(ys), date_tier=tier,
                       date_quote=' | '.join(dict.fromkeys(quotes))[:300], n_date_values=len(ys))
        return ys

    empty = not body or width == 0
    # ---- tier A: date columns (running year down each column) ----
    ys, quotes, row_years = [], [], {}
    for ci in [c for c, is_cons in ycols.items() if not is_cons]:
        dcols[ci] = 0  # unlabeled column of bare years, not a well-construction table
    for ci, start in dcols.items():
        last_year = None  # running year from divider rows ("1919.", "1918 -Continued.")
        for ri, row in enumerate(body):
            if ri < start or ci >= len(row) or is_merged(row):
                continue
            keep, por, excl = clean_text(row[ci])
            por_clauses += por
            excl_clauses += excl
            y, q = _years_from_clauses(keep, cap, allow_bare=True, allow_two_digit=True)
            if y:
                ys += y
                quotes += q
                row_years.setdefault(ri, []).extend(y)
                last_year = max(y)
            elif last_year and re.search(MONTHS + r'\.?\s*\d{1,2}|\d{1,2}\s*[-/]\s*\d{1,2}', row[ci], re.I):
                row_years.setdefault(ri, []).append(last_year)  # "Jan. 25" under a 1919 divider
    for ci in dcols:
        if ci < len(cols):
            keep, por, excl = clean_text(cols[ci])
            y, q = _years_from_clauses(keep, cap, allow_bare=True)
            ys += y
            quotes += q
    # rotated tables: a row labeled "Date" holds the dates across the row
    for ri, li in drows.items():
        for c in body[ri][li + 1:]:
            keep, por, excl = clean_text(c)
            y, q = _years_from_clauses(keep, cap, allow_bare=True, allow_two_digit=True)
            ys += y
            quotes += q
    if ys:
        ys = finish(ys, quotes, 'A_date_column')
        if not ys:
            row_years = {}

    # ---- tier B: other table text ----
    if not ys and not empty:
        q_all = []
        for hi, h in enumerate(cols):
            if hi in cons_cols:
                continue  # construction-year column (on continuation pages the 'header' is a data row)
            keep, por, excl = clean_text(h)
            por_clauses += por
            short = len(h.split()) <= 8 or re.search(r'year|' + MONTHS, h, re.I)
            y, q = _years_from_clauses(keep, cap, allow_bare=bool(short))
            ys += y
            q_all += q
        seen_merged = set()
        y2, q2 = [], []
        for ri, row in enumerate(body):
            if ri in xrows:  # rotated-table row labeled period of record / known floods
                (por_clauses if POR.search(' '.join(row)) else excl_clauses).extend(c for c in row if c)
                continue
            if is_merged(row):
                cells = [row[0]] if row[0] not in seen_merged else []
                seen_merged.add(row[0])
            else:
                cells = [c for i, c in enumerate(row) if i not in cons_cols]
            for cell in cells:
                keep, por, excl = clean_text(cell)
                por_clauses += por
                excl_clauses += excl
                y, q = _years_from_clauses(keep, cap)  # explicit date forms only
                ys += y
                q_all += q
                for c in keep:  # numeric dates with 2-digit years ("4-19-66")
                    yy = [int(m.groups()[-1]) + 1900 for m in P_NUM_DATE2.finditer(c)]
                    yy = [v for v in yy if _ok(v, cap)]
                    if yy:
                        y2 += yy
                        q2.append(c[:60])
        if len(y2) >= 2:  # only trusted when the pattern repeats in the table
            ys += y2
            q_all += q2
        if ys:
            ys = finish(ys, q_all, 'B_table_text')

    # ---- tier C: caption -- the NEAREST text block above the table that has a date wins
    # (the table title), so notes further up ("storage began 1955", flood remarks) don't widen it
    cap_lines = [ln for ln in (caption or '').split('\n') if ln.strip()]
    for ln in cap_lines:
        keep, por, excl = clean_text(ln)
        por_clauses += por
        excl_clauses += excl
    if not ys and not empty:
        for ln in reversed(cap_lines):
            keep, por, excl = clean_text(ln)
            y, q = _years_from_clauses(keep, cap, allow_bare=True)
            if y:
                ys = finish(y, q, 'C_caption')
                if ys:
                    break

    # ---- tier D: rest of the page ----
    keep, por, excl = clean_text(page_text or '')
    por_clauses += por
    if not ys and not empty:
        ctx = [c for c in keep if re.search(MONTHS + r'|\byear|\bdate', c, re.I)]
        y1, q1 = _years_from_clauses(keep, cap)                      # explicit forms anywhere
        y2, q2 = _years_from_clauses(ctx, cap, allow_bare=True)      # bare years near date words
        ys = finish(y1 + y2, q1 + q2, 'D_page_text') if (y1 or y2) else []

    # ---- period of record (kept separately) ----
    py = []
    for c in dict.fromkeys(por_clauses):
        py += years_in(c, cap, allow_bare=True)
    res['por_start'] = min(py) if py else None
    res['por_end'] = max(py) if py else None
    res['por_quote'] = ' | '.join(dict.fromkeys(por_clauses))[:300] or None

    # ---- construction years (well drilled / completed), kept separately ----
    cy = []
    for ci in cons_cols:
        if ci in ycols and ci < len(cols):  # headerless continuation page: header row is data
            cy += years_in(cols[ci], cap, allow_bare=True)
        for row in body:
            if ci < len(row) and not is_merged(row):
                cy += years_in(row[ci], cap, allow_bare=True)
    for c in dict.fromkeys(excl_clauses):
        if CONSTRUCTION.search(c):
            cy += years_in(c, cap, allow_bare=True)
    res['construction_start'] = min(cy) if cy else None
    res['construction_end'] = max(cy) if cy else None

    # ---- last-resort fallbacks (labeled; see protocol): the table has no data date at all.
    # E: the station's period of record ("Records available: 1941-53")
    # F: years the wells were drilled / completed (well tables that print nothing else)
    if not ys and not empty:
        if res['por_start'] is not None:
            res.update(year_start=res['por_start'], year_end=res['por_end'],
                       date_tier='E_period_of_record', date_quote=res['por_quote'])
        elif res['construction_start'] is not None:
            res.update(year_start=res['construction_start'], year_end=res['construction_end'],
                       date_tier='F_construction_year', date_quote='construction years')
    res['row_years'] = row_years
    res['n_body_rows'] = len(body)
    res['n_cols'] = width
    return res
