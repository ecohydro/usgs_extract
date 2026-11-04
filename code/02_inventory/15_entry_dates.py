"""
Dates for each LLM metadata entry (row of main_metadata.csv), from the table-level
dates in table_dates.csv / table_row_dates.parquet (14_table_dates.py).

The LLM often returned several entries per page (one per well / station). Each entry
takes its dates from the most specific evidence available, on its own page only:
  row         the entry's well number / site ID (or >=2 owner-name words) matches the
              site label of one or more dated table rows -> range over those rows
  table       the entry's distinctive name words all appear in one table's caption
              (multi-table pages), or the page has only one table
  page        union of all tables on the page (entry could not be tied to one table)
  none        no table on the page has a date
The old LLM date string is kept as year_start_llm / year_end_llm / date_status_llm.

Output: data/analysis/entry_dates.csv (key: meta_row = row position in main_metadata.csv)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from date_parsing import parse_dates  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

md = pd.read_csv(ROOT / 'data/metadata/main_metadata.csv', low_memory=False,
                 usecols=['id', 'page_number', 'watersource_name', 'location', 'dates_of_recording',
                          'Year Published'], dtype={'id': str})
md = md.rename(columns={'Year Published': 'pub_year'})
md['meta_row'] = np.arange(len(md))
md['id'] = md['id'].str.replace(r'\.0$', '', regex=True)
md['page_number'] = pd.to_numeric(md['page_number'], errors='coerce').astype('Int64')

td = pd.read_csv(ROOT / 'data/analysis/table_dates.csv', dtype={'id': str})
rd = pd.read_parquet(ROOT / 'data/analysis/table_row_dates.parquet')
rd['id'] = rd['id'].astype(str)

STOP = set('''river creek near at above below calif california fork north south east west middle
the and of in on for from well wells spring springs canal ditch lake reservoir mile miles county
station gage gaging water mouth main branch little big upper lower slough drain lateral tributary
co company ranch city town valley basin area total near'''.split())
ID_TOKEN = re.compile(r'\d+[A-Z]+\d*[A-Z]?|\d+/\d+[A-Z]?-\d+[A-Z]\d*')


def ids(s):
    s = str(s or '').upper().replace(' ', '')
    return set(ID_TOKEN.findall(s))


def words(s):
    return {w for w in re.findall(r'[a-z]{3,}', str(s or '').lower()) if w not in STOP}


tables_by_page = {k: g for k, g in td.groupby(['id', 'page_number'])}
rows_by_page = {k: g for k, g in rd.groupby(['id', 'page_number'])}

out = []
for e in md.itertuples():
    key = (e.id, int(e.page_number)) if pd.notna(e.page_number) else None
    T = tables_by_page.get(key)
    rec = {'meta_row': e.meta_row, 'date_source': 'none', 'year_start': None, 'year_end': None,
           'date_tier': 'none', 'date_quote': None, 'matched_tables': None,
           'por_start': None, 'por_end': None, 'construction_start': None, 'construction_end': None}
    y0, y1, st = parse_dates(e.dates_of_recording, e.pub_year)
    rec.update(year_start_llm=y0, year_end_llm=y1, date_status_llm=st)
    if T is None:
        out.append(rec)
        continue
    text = f'{e.watersource_name} {e.location}'
    chosen, source, row_range = None, None, None

    # 1. row match
    R = rows_by_page.get(key)
    if R is not None and len(R):
        e_ids, e_words = ids(text), words(text)
        hit = []
        for r in R.itertuples():
            l_ids = ids(r.site_label)
            if (l_ids and e_ids & l_ids) or (not l_ids and len(words(r.site_label) & e_words) >= 2):
                hit.append(r)
        if hit:
            row_range = (min(h.year_min for h in hit), max(h.year_max for h in hit))
            chosen = T[T.table_number.isin({h.table_number for h in hit})]
            source = 'row'
    # 2. caption match / single table
    if chosen is None:
        if len(T) == 1:
            chosen, source = T, 'table'
        else:
            w = words(e.watersource_name)
            if w:
                m = T[T.caption.fillna('').str.lower().apply(lambda c: all(x in c for x in w))]
                if 0 < len(m) < len(T):
                    chosen, source = m, 'table'
    # 3. page union
    if chosen is None:
        chosen, source = T, 'page'

    dated = chosen[chosen.year_start.notna()]
    if row_range:
        rec.update(year_start=row_range[0], year_end=row_range[1], date_tier='A_date_column')
    elif len(dated):
        rec.update(year_start=dated.year_start.min(), year_end=dated.year_end.max(),
                   date_tier=sorted(dated.date_tier)[0])  # best (lowest-letter) tier used
    if rec['year_start'] is not None:
        rec['date_source'] = source
        rec['date_quote'] = ' || '.join(dated.date_quote.dropna().astype(str))[:300] or None
    rec['matched_tables'] = ','.join(str(t) for t in chosen.table_number)
    for c in ['por_start', 'construction_start']:
        v = chosen[c].min()
        rec[c] = v if pd.notna(v) else None
    for c in ['por_end', 'construction_end']:
        v = chosen[c].max()
        rec[c] = v if pd.notna(v) else None
    out.append(rec)

df = pd.DataFrame(out)
for c in ['year_start', 'year_end', 'year_start_llm', 'year_end_llm', 'por_start', 'por_end',
          'construction_start', 'construction_end']:
    df[c] = pd.to_numeric(df[c]).astype('Int64')
df.to_csv(ROOT / 'data/analysis/entry_dates.csv', index=False)
print(f'entries {len(df):,}; dated {df.year_start.notna().mean():.1%} (LLM string: {df.year_start_llm.notna().mean():.1%})')
print(df.date_source.value_counts().to_string())
print(df.date_tier.value_counts().to_string())
both = df.year_start.notna() & df.year_start_llm.notna()
print(f'both dated: {both.sum():,}; same start year {(df.year_start[both] == df.year_start_llm[both]).mean():.1%}')
