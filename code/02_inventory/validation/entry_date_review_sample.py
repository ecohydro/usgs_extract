"""
Targeted grading sample for entry_dates.csv (15_entry_dates.py).

Every entry whose new start year differs from the LLM start year is put in one category:
  llm=period_of_record  LLM start = the table's period-of-record start (known LLM error)
  llm=year_drilled      LLM start = construction / year-drilled (known LLM error)
  off_by_1              water-year convention (Oct-Sep spans two calendar years)
  new_contains_llm      new range is wider than the LLM range (stray years widening it?)
  llm_contains_new      LLM range is wider than the new range
  overlap / no_overlap  partial or no overlap (real conflicts)
The explained categories (period of record, year drilled, off by 1) get only a few rows to
confirm; the uncertain ones get most of the sample. Page images are copied next to the CSV.

Output: data/analysis/date_parsing/entry_date_review/entry_date_review_sample.csv (+ pngs/)
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/analysis/date_parsing/entry_date_review'
(OUT / 'pngs').mkdir(parents=True, exist_ok=True)
N_PER_CAT = {'new_contains_llm': 10, 'llm_contains_new': 5, 'no_overlap': 8, 'overlap': 3,
             'llm=period_of_record': 2, 'off_by_1': 2}

e = pd.read_csv(ROOT / 'data/analysis/entry_dates.csv')
m = pd.read_csv(ROOT / 'data/metadata/main_metadata.csv', low_memory=False, encoding_errors='replace',
                usecols=['id', 'page_number', 'watersource_name', 'water_type', 'dates_of_recording'],
                dtype={'id': str})
e = e.join(m)
e['id'] = e['id'].str.replace(r'\.0$', '', regex=True)
b = e[e.year_start.notna() & e.year_start_llm.notna() & (e.year_start != e.year_start_llm)].copy()
s, t, ls, lt = b.year_start, b.year_end, b.year_start_llm, b.year_end_llm
b['category'] = np.select(
    [ls.eq(b.por_start) & b.por_start.notna(), ls.eq(b.construction_start) & b.construction_start.notna(),
     (s - ls).abs() == 1, (s <= ls) & (t >= lt), (ls <= s) & (lt >= t), (s <= lt) & (ls <= t)],
    ['llm=period_of_record', 'llm=year_drilled', 'off_by_1', 'new_contains_llm', 'llm_contains_new',
     'overlap'], 'no_overlap')


def png(r):
    return ROOT / f'data/digitized/{r.id}/page_{int(r.page_number)}/{r.id}_page_{int(r.page_number)}.png'


b = b[[png(r).exists() for r in b.itertuples()]]
rng = np.random.default_rng(20261103)
picks = []
for cat, n in N_PER_CAT.items():
    g = b[b.category == cat]
    # half from page-union entries (least specific), rest from any source; one entry per page
    g = g.sample(frac=1, random_state=rng.integers(1e9)).drop_duplicates(['id', 'page_number'])
    pg = g[g.date_source == 'page'].head(n // 2)
    picks.append(pd.concat([pg, g.drop(pg.index).head(n - len(pg))]))
S = pd.concat(picks).reset_index(drop=True)
S.insert(0, 'review_no', np.arange(1, len(S) + 1))
S['local_png'] = [f'pngs/{r.review_no:03d}_{r.id}_page_{int(r.page_number)}.png' for r in S.itertuples()]
for r in S.itertuples():
    shutil.copy2(png(r), OUT / r.local_png)
cols = ['review_no', 'category', 'date_source', 'id', 'page_number', 'water_type', 'watersource_name',
        'dates_of_recording', 'year_start_llm', 'year_end_llm', 'year_start', 'year_end', 'date_tier',
        'date_quote', 'por_start', 'por_end', 'meta_row', 'local_png']
S = S[cols].rename(columns={'dates_of_recording': 'llm_date_string', 'year_start': 'new_start',
                            'year_end': 'new_end', 'date_quote': 'new_date_evidence'})
for c in ['grader_true_start', 'grader_true_end', 'grader_which_right (new/llm/both/neither)',
          'grader_notes']:
    S[c] = ''
S.to_csv(OUT / 'entry_date_review_sample.csv', index=False)
print(f'{len(S)} rows ->', OUT)
print(S.groupby(['category', 'date_source']).size().to_string())
