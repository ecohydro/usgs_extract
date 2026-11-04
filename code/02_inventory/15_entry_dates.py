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
  llm_grounded  no table date: the LLM's date is kept only if both its start and end
              years are printed on the same page (date_tier G_llm_grounded)
  none        no table on the page has a date
Period-of-record (E) and construction-year (F) fallback tiers of a table are used only
when none of the entry's tables has a data date (tiers A-D). Added 2026-11-04 after
Annette's grading of entry_date_review_sample.csv.
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
from table_dates import years_in, YEAR_MAX  # noqa: E402

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


FALLBACK_TIERS = {'E_period_of_record', 'F_construction_year'}

# Years written anywhere on each page (all Reducto blocks, incl. tables), for the grounded
# LLM fallback: an entry with no table date keeps the LLM's date only if BOTH its start and
# end year are printed on the same page (tier G_llm_grounded).
PB = pd.read_parquet(ROOT / 'data/analysis/page_blocks.parquet', columns=['id', 'page_number', 'content'])
PB['id'] = PB['id'].astype(str)
page_years = {k: set(years_in(' '.join(g.content.fillna('').astype(str)), YEAR_MAX, allow_bare=True))
              for k, g in PB.groupby(['id', 'page_number'])}
del PB


pages_with_data_table = set(td.loc[td.n_body_rows.fillna(0) > 0, ['id', 'page_number']]
                            .itertuples(index=False, name=None))


def is_grounded(rec, key):
    y0, y1 = rec['year_start_llm'], rec['year_end_llm']
    ys = page_years.get(key, set())
    return y0 is not None and y0 in ys and (y1 is None or y1 == y0 or y1 in ys)


def grounded_llm(rec, key):
    """Only on pages with a non-empty table (reference / index pages print citation years)."""
    y0, y1 = rec['year_start_llm'], rec['year_end_llm']
    rec['llm_grounded'] = key is not None and is_grounded(rec, key)
    if rec['year_start'] is None and rec['llm_grounded'] and key in pages_with_data_table:
        ys = page_years.get(key, set())
        if y0 in ys and (y1 is None or y1 == y0 or y1 in ys):
            rec.update(year_start=y0, year_end=y1 if y1 is not None else y0,
                       date_tier='G_llm_grounded', date_source='llm_grounded',
                       date_quote='LLM date; start and end years printed on the page')
    return rec
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
        out.append(grounded_llm(rec, key))
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
    data_dated = dated[~dated.date_tier.isin(FALLBACK_TIERS)]
    if len(data_dated):  # period-of-record / construction fallbacks only if no table has data dates
        dated = data_dated
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
    out.append(grounded_llm(rec, key))

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
