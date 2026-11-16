"""
Figure 5 (groundwater example): parse the water-level readings of USGS Water-Supply Paper 1464 (Thomasson,
Olmsted and LeRoux, 1960, "Geology, water resources and usable ground-water storage capacity of part of
Solano County, California"; project doc 2809), Appendix B "Water-level measurements" (PDF pages 518-659).

Why a dedicated parser (2026-11-16): each page prints the "Date | Water level" header once; Reducto splits
every well into its own table, so most tables have no header (catalog primary_parameter 'unknown' for 83
of 103) and the pipeline's record-level matching does not tie readings to wells. Here each reading is
tied to the well heading printed above it on the page (same-page evidence):
  - Text block starting with a state well number ("8/1E-15B1 (UC 392). F. E. Russell. About ...")
    = a new well; its description gives land-surface altitude and measuring point.
  - A full-width table row "5/3W-26F2 (Clark-14)-Continued" = the well continued from the previous page.
    The same kind of row inside a table marks a well change when Reducto merged several wells.
  - Every other row holds (date, water level) pairs, 3 per printed row.
Values are depth to water below land-surface datum, feet (Appendix B, p. 500 of the report). A number
before the value is a footnote (e.g. "3 90.78" = footnote 3; footnotes listed on printed pp. 640-641;
14-66 are on PDF page 659, 1-13 are on a page that was never digitized). "(57)" = footnote only, no value.

Only pages the pipeline digitized are used (48 of the 142 section pages; the other 94 were missed by table
detection and never sent to Reducto).

Output: data/analysis/gw_example/solano_wsp1464_readings.csv  (one row per reading)
        data/analysis/gw_example/solano_wsp1464_wells.csv     (one row per well: heading, description,
                                                               altitude, n readings, years, pages)
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
from html_table import parse_table  # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'gw_example'
DOC, P0, P1 = '2809', 518, 659

WELL = r'(\d{1,2})/(\d{1,2})([EW])\s*-\s*(\d{1,2})([A-R])(\d{1,2})'
HEAD_RE = re.compile(r'(?:^|level\s+|Water\s+)' + WELL + r'\s*(\([^)]*\))?\s*[.,-]', re.I)
CONT_RE = re.compile(WELL + r'.{0,60}?Continued', re.I)
MONTH = r'(Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sept?|Oct|Nov|Dec)[a-z]*'
DATE_RE = re.compile(r'^\W*' + MONTH + r'\.?\s*(\d{1,2})\s*[,.]?\s*(\d{4})?\s*$', re.I)
MON = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                                         'oct', 'nov', 'dec'])}
ALT_RE = re.compile(r'land-?\s*surface datum,? which is ([\d,.]+)\s*feet (above|below) (?:mean )?sea level', re.I)
ALT2_RE = re.compile(r'Land-?\s*surface datum is ([\d,.]+)\s*feet (above|below) (?:mean )?sea level', re.I)
AREA_RE = re.compile(r'(PUTAH|SUISUN-FAIRFIELD)\s+AREA|TEST WELLS', re.I)


def well_id(m):
    t, r, ew, sec, tr, seq = m[:6]
    return f'{int(t)}/{int(r)}{ew.upper()}-{int(sec)}{tr.upper()}{int(seq)}'


def parse_value(raw):
    """'25. 0' -> 25.0; '3 90.78' -> (90.78, fn 3); '(57)' -> (None, 57); '24, 1' -> 24.1."""
    s = raw.strip()
    if not s:
        return None, None, ''
    m = re.fullmatch(r'\((\d{1,2})\)', s)
    if m:
        return None, int(m.group(1)), 'footnote only'
    s = re.sub(r'(\d)\s*[.,]\s+(\d)', r'\1.\2', s)          # "25. 0", "24, 1"
    s = re.sub(r'(\d)\s+\.(\d)', r'\1.\2', s)
    fn = None
    m = re.fullmatch(r'(\d{1,2}(?:\s+\d{1,2})?)\s+(-?\d+(?:\.\d*)?|-?\.\d+)\s*\+?', s)
    if m:
        fn, s = m.group(1), m.group(2)
    m = re.fullmatch(r'(-?\d+(?:\.\d*)?|-?\.\d+)\s*(\+)?', s)
    if not m:
        return None, fn, 'not a number'
    note = 'trailing period (digits may be lost)' if s.endswith('.') else ''
    if m.group(2):
        note = 'plus sign'
    return float(m.group(1).rstrip('.') or 'nan'), fn, note


pb = pd.read_parquet(A / 'page_blocks.parquet')
pb = pb[(pb.id.astype(str) == DOC) & pb.page_number.between(P0, P1)].sort_values(['page_number', 'block_order'])
print(f'digitized section pages: {pb.page_number.nunique()} of {P1 - P0 + 1}')

wells, rows = {}, []
cur, area = None, None
for blk in pb.itertuples(index=False):
    txt = str(blk.content)
    a = AREA_RE.search(txt[:200]) if blk.block_type in ('Title', 'Text', 'Header') else None
    if a:
        area = 'Test wells' if 'TEST' in a.group(0).upper() else a.group(1).title() + ' area'
    if blk.block_type != 'Table':
        for h in HEAD_RE.finditer(txt):
            w = well_id(h.groups())
            cur = w
            desc = txt[h.start():]
            nxt = HEAD_RE.search(desc, 5)
            desc = desc[:nxt.start()] if nxt else desc
            d = wells.setdefault(w, {'well': w, 'alias': (h.group(7) or '').strip('() '), 'area': area,
                                     'heading_page': blk.page_number, 'description': desc[:600]})
            m = ALT_RE.search(desc) or ALT2_RE.search(desc)
            if m:
                d['lsd_ft_above_msl'] = float(m.group(1).replace(',', '')) * (1 if m.group(2) == 'above' else -1)
        continue
    head, body = parse_table(txt)
    seg = 0
    grid = []                                  # rows of this table after the current heading
    def flush():
        # column-major reading order inside a segment; carry the year down each column
        if not grid:
            return
        ncol = max(len(r) for r in grid)
        for ci in range(ncol):
            year = None
            for ri, r in enumerate(grid):
                if ci >= len(r):
                    continue
                dm = DATE_RE.match(r[ci])
                if not dm:
                    continue
                vcol = next((cj for cj in range(ci + 1, len(r)) if r[cj].strip() and not DATE_RE.match(r[cj])), None)
                if vcol is None or (vcol > ci + 2):
                    continue
                mon = MON[dm.group(1)[:3].lower()]
                year = int(dm.group(3)) if dm.group(3) else year
                if year is None:
                    continue
                val, fn, note = parse_value(r[vcol])
                rows.append({'well': gw_cur, 'page': blk.page_number, 'table_idx': blk.table_idx,
                             'segment': gseg, 'date': f'{year}-{mon:02d}-{int(dm.group(2)):02d}',
                             'depth_ft': val, 'footnote': fn, 'raw_date': r[ci], 'raw_value': r[vcol],
                             'note': note})
        grid.clear()
    gw_cur, gseg = cur, seg
    for r in head + body:
        joined = ' '.join(dict.fromkeys(c for c in r if c))
        if len(set(c for c in r if c)) == 1 and re.search(WELL, joined):
            flush()
            m = re.search(WELL, joined)
            cur = well_id(m.groups())
            wells.setdefault(cur, {'well': cur, 'alias': '', 'area': area, 'heading_page': None,
                                   'description': ''})
            seg += 1
            gw_cur, gseg = cur, seg
            continue
        if re.fullmatch(r'(Date|Water|level|Water level|\s)*', joined, re.I):
            continue
        grid.append(r)
    flush()

rd = pd.DataFrame(rows)
rd = rd[rd.well.notna()]
rd['year'] = rd.date.str[:4].astype(int)
rd.to_csv(OUT / 'solano_wsp1464_readings.csv', index=False)
wl = pd.DataFrame(wells.values())
agg = rd.groupby('well').agg(n_readings=('date', 'size'), n_values=('depth_ft', lambda x: x.notna().sum()),
                             first=('year', 'min'), last=('year', 'max'),
                             years=('year', lambda x: ' '.join(map(str, sorted(set(x))))),
                             pages=('page', lambda x: ' '.join(map(str, sorted(set(x))))))
wl = wl.merge(agg, on='well', how='left')
wl.to_csv(OUT / 'solano_wsp1464_wells.csv', index=False)
print(f'wells {len(wl)} (with readings {wl.n_readings.notna().sum()}); readings {len(rd):,}; '
      f'values {rd.depth_ft.notna().sum():,}; footnoted {rd.footnote.notna().sum():,}; '
      f'years {rd.year.min()}-{rd.year.max()}')
print(rd.note.value_counts().to_string())
print(wl.sort_values('n_readings', ascending=False)[['well', 'alias', 'area', 'n_readings', 'first', 'last',
                                                     'pages']].head(15).to_string(index=False))
