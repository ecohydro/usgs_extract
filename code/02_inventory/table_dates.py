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
POR = re.compile(r'records?\s+(?:are\s+)?available|period\s+of\s+record|periods?\s+of\s+records|'
                 r'records?\s+(?:began|begin|start)|record\s+since|\bsince\b', re.I)
EXCL = re.compile(
    r'drill|complet|construct|\bdug\b|\bbored\b|deepen|install|\bbuilt\b|finished|prior\s+to|'
    r'maximum|minimum|time\s+basis|no\s+flow|'
    r'highest|lowest|\bextremes?\b|known|recurrence|interval|\bdatum\b|adjustment|'
    r'revis|water[- ]supply\s+paper|\bpaper\b|bulletin|\breport|publish|\bprinted\b|'
    r'established|discontinued|abandoned|destroyed|average|\bmean\s+for\b|'
    r'copyright|census|reprint|\bsurvey\s+of\b|\bact\s+of\b|decree|adjudicat|\bpage\b',
    re.I)
DATE_HEAD = re.compile(r'\bdate|\byear|\bwtr\.?\s*yr|calend|calen?d[ae]r|\bmonth\b|\btime\s+of\b|\bperiod\b', re.I)
HEAD_EXCL = re.compile(r'drill|complet|construct|\bdug\b|deepen|recurrence|interval|\byears?\s*\)|'
                       r'number\s+of\s+years|\bage\b|install|abandon|reported|\bperiod\s+of\s+record', re.I)
WELL_DESC_HEAD = re.compile(r'depth|diameter|altitude|\buse\b|casing|owner', re.I)

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
            out.append(part[start:m.end()])
            start = m.end()
        out.append(part[start:])
    return [c.strip() for c in out if c.strip()]


# ---- year patterns ----
P_WY_END = re.compile(r'year\s+ending\s+' + MONTHS + r'\.?\s*\d{0,2},?\s*' + Y4, re.I)
P_RANGE = re.compile(r'(?<![\d,.])' + Y4 + r'\s*(?:-|–|to|and)\s*(\d{2}|1[89]\d{2})(?![\d,])', re.I)
P_MONTH_YEAR = re.compile(r'\b' + MONTHS + r'\.?\s*(?:\d{1,2}\s*(?:-\s*\d{1,2})?,?\s*)?' + Y4 + r'(?!\d)', re.I)
P_NUM_DATE4 = re.compile(r'(?<!\d)(?:0?[1-9]|1[0-2])\s*[-/]\s*(?:0?[1-9]|[12]\d|3[01])\s*[-/]\s*' + Y4 + r'(?!\d)')
P_ISO = re.compile(r'(?<!\d)' + Y4 + r'-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)')
P_NUM_DATE2 = re.compile(r'(?<![\d.,])(?:0?[1-9]|1[0-2])\s*[-/]\s*(?:0?[1-9]|[12]\d|3[01])\s*[-/]\s*(\d{2})(?![\d.,])')
P_BARE = re.compile(r'(?<![\d,.$])' + Y4 + r'(?![\d,]|\.\d)')


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
            ys.append(int(m.group(1)))
        s = pat.sub(' ', s)
    if allow_two_digit:
        for m in P_NUM_DATE2.finditer(s):
            ys.append(1900 + int(m.group(1)))
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
CONSTRUCTION = re.compile(r'drill|complet|construct|\bdug\b|\bbored\b|deepen|install|\bbuilt\b|finished', re.I)


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
    secs = [s for line in (text or '').split('\n') for s in _sections(line)]
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
            elif EXCL.search(c):
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


def table_dates(html, caption, page_text, cap):
    """Dates for one table. Returns dict with year_start/end, tier, quotes, period of record."""
    cap = min(cap or YEAR_MAX, YEAR_MAX)
    head, body = parse_table(html)
    cols = column_headers(head)
    width = max([len(cols)] + [len(r) for r in body]) if (cols or body) else 0
    dcols = date_columns(cols, body)
    cons_cols = construction_columns(cols)
    por_clauses, excl_clauses = [], []
    res = {'year_start': None, 'year_end': None, 'date_tier': None, 'date_quote': None,
           'n_date_values': 0, 'date_columns': '; '.join(cols[i] for i in dcols if i < len(cols))[:200]}

    # wide merged rows (one cell repeated across the table) are description rows
    def is_merged(row):
        vals = [c for c in row if c]
        return len(vals) > 1 and len(set(vals)) == 1

    # ---- tier A: date columns (running year down each column) ----
    ys, quotes, row_years = [], [], {}
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
    # header text of the date column itself ("Date (1951)", "Water year 1934-35")
    for ci in dcols:
        if ci < len(cols):
            keep, por, excl = clean_text(cols[ci])
            y, q = _years_from_clauses(keep, cap, allow_bare=True)
            ys += y
            quotes += q
    if ys:
        res.update(year_start=min(ys), year_end=max(ys), date_tier='A_date_column',
                   date_quote=' | '.join(dict.fromkeys(quotes))[:300], n_date_values=len(ys))

    # ---- tier B: other table text ----
    if not ys:
        q_all = []
        for h in cols:
            keep, por, excl = clean_text(h)
            por_clauses += por
            short = len(h.split()) <= 8 or re.search(r'year|' + MONTHS, h, re.I)
            y, q = _years_from_clauses(keep, cap, allow_bare=bool(short))
            ys += y
            q_all += q
        seen_merged = set()
        y2, q2 = [], []
        for row in body:
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
                    yy = [int(m.group(1)) + 1900 for m in P_NUM_DATE2.finditer(c)]
                    yy = [v for v in yy if _ok(v, cap)]
                    if yy:
                        y2 += yy
                        q2.append(c[:60])
        if len(y2) >= 2:  # only trusted when the pattern repeats in the table
            ys += y2
            q_all += q2
        if ys:
            res.update(year_start=min(ys), year_end=max(ys), date_tier='B_table_text',
                       date_quote=' | '.join(dict.fromkeys(q_all))[:300], n_date_values=len(ys))

    # ---- tier C: caption ----
    keep, por, excl = clean_text(caption or '')
    por_clauses += por
    excl_clauses += excl
    if not ys:
        ys, q = _years_from_clauses(keep, cap, allow_bare=True)
        if ys:
            res.update(year_start=min(ys), year_end=max(ys), date_tier='C_caption',
                       date_quote=' | '.join(q)[:300], n_date_values=len(ys))

    # ---- tier D: rest of the page ----
    keep, por, excl = clean_text(page_text or '')
    por_clauses += por
    if not ys:
        ctx = [c for c in keep if re.search(MONTHS + r'|\byear|\bdate', c, re.I)]
        y1, q1 = _years_from_clauses(keep, cap)                      # explicit forms anywhere
        y2, q2 = _years_from_clauses(ctx, cap, allow_bare=True)      # bare years near date words
        ys = y1 + y2
        if ys:
            res.update(year_start=min(ys), year_end=max(ys), date_tier='D_page_text',
                       date_quote=' | '.join(dict.fromkeys(q1 + q2))[:300], n_date_values=len(ys))

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
        for row in body:
            if ci < len(row) and not is_merged(row):
                cy += years_in(row[ci], cap, allow_bare=True)
    for c in dict.fromkeys(excl_clauses):
        if CONSTRUCTION.search(c):
            cy += years_in(c, cap, allow_bare=True)
    res['construction_start'] = min(cy) if cy else None
    res['construction_end'] = max(cy) if cy else None
    res['row_years'] = row_years
    res['n_body_rows'] = len(body)
    res['n_cols'] = width
    return res
