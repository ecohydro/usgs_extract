"""
Blind hand-check sample for the value counts of NON-daily tables (19_value_counts.py).
Daily day-by-month tables are validated automatically by the calendar check; this covers the rest.

20 published, non-withheld tables with 20-120 counted values (a few minutes each to count),
stratified by water type. The grader counts the measurement values on the page image and fills
grader_true_values. The script's count is kept in a separate key file so grading stays blind.

Outputs (data/analysis/value_counts/review/):
  value_count_review_sample.csv   grading sheet (+ pngs/, git-ignored)
  value_count_review_key.csv      script counts, joined only after grading
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / 'data/analysis'
OUT = A / 'value_counts/review'
STRATA = {'Stream Discharge': 6, 'Groundwater': 5, 'Reservoir': 3, 'Springs': 2, 'Irrigation': 2,
          'Water Quality': 1, 'Precipitation': 1}
KEY = ['doc_id', 'page_number', 'table_number']

v = pd.read_csv(A / 'table_value_counts.csv')
pub = pd.read_csv(A / 'table_level_metadata_published.csv', usecols=KEY + ['water_type_final', 'watersource_name'])
trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv')
trib = trib[trib.withhold.astype(bool)].rename(columns={'id': 'doc_id'})
t = pub.merge(v, on=KEY)
t = t[~t.set_index(['doc_id', 'page_number']).index.isin(trib.set_index(['doc_id', 'page_number']).index)]
t = t[(t.is_daily_grid != True) & t.n_values.between(20, 120)]
s = pd.concat([t[t.water_type_final == wt].drop_duplicates('doc_id').sample(n, random_state=11)
               for wt, n in STRATA.items()]).reset_index(drop=True)
s.insert(0, 'review_no', range(1, len(s) + 1))

(OUT / 'pngs').mkdir(parents=True, exist_ok=True)
for r in s.itertuples():
    folder = ROOT / f'data/digitized/{r.doc_id}/page_{r.page_number}'
    png = folder / f'{r.doc_id}_page_{r.page_number}.png'
    if png.exists():
        shutil.copy(png, OUT / f'pngs/{r.review_no:02d}_{png.name}')
s['local_png'] = [f'pngs/{r.review_no:02d}_{r.doc_id}_page_{r.page_number}.png' for r in s.itertuples()]
s['table_csv'] = [f'data/digitized/{r.doc_id}/page_{r.page_number}/{r.doc_id}_page_{r.page_number}_table{r.table_number}.csv'
                  for r in s.itertuples()]
sheet = s[['review_no'] + KEY + ['water_type_final', 'watersource_name', 'local_png', 'table_csv']].copy()
sheet['grader_true_values'] = ''
sheet['grader_notes'] = ''
sheet.to_csv(OUT / 'value_count_review_sample.csv', index=False)
s[['review_no'] + KEY + ['n_values', 'n_data_rows', 'n_excluded_cols']].to_csv(OUT / 'value_count_review_key.csv', index=False)
print(sheet[['review_no', 'doc_id', 'page_number', 'table_number', 'water_type_final']].to_string(index=False))
