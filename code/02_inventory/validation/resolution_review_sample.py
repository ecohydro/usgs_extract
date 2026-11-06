"""
Blind grading sample for the per-table temporal resolution (20_temporal_resolution.py).

Stratified by water type x assigned class so every class is checked, with extra weight on
groundwater. Catalog tables only (published, not withheld tribal pages), page image required.
The grader sees the page and table number, NOT the assigned class (kept in _key.csv).

Outputs: data/analysis/temporal_resolution/review/
  resolution_review_sample.csv   for grading (blank columns: true_resolution, derived_summary, note)
  resolution_review_key.csv      assigned class per sample_no (do not open before grading)
  pngs/{sample_no}_{doc}_page_{N}.png
  README.txt                     grading rules
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / 'data/analysis'
OUT = A / 'temporal_resolution/review'
SEED = 11
PLAN = {'Stream Discharge': {'daily': 6, 'monthly': 6, 'annual': 5, 'discrete': 5,
                             'no_time_series': 4, 'unclassified': 4},
        'Groundwater': {'daily': 3, 'monthly': 3, 'annual': 4, 'discrete': 6,
                        'no_time_series': 4, 'unclassified': 5}}
N_OTHER = 10

t = pd.read_csv(A / 'table_resolution.csv')
t = t[t.published]
trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv')
trib = trib[trib.withhold.astype(bool)]
t = t[~pd.Series(list(zip(t.doc_id, t.page_number))).isin(set(zip(trib.id, trib.page_number))).values]
td = pd.read_csv(A / 'table_dates.csv', usecols=['id', 'page_number', 'table_number', 'caption'],
                 low_memory=False).rename(columns={'id': 'doc_id'})
t = t.merge(td, on=['doc_id', 'page_number', 'table_number'], how='left')
t['png'] = [ROOT / f'data/digitized/{d}/page_{p}/{d}_page_{p}.png' for d, p in zip(t.doc_id, t.page_number)]
t = t[[p.exists() for p in t.png]]

picks = []
for wt, plan in PLAN.items():
    for cls, n in plan.items():
        g = t[(t.water_type_final == wt) & (t.resolution == cls)]
        if cls == 'monthly':  # half derived summaries, half stand-alone
            picks += [g[g.derived_summary].sample(min(n // 2, g.derived_summary.sum()), random_state=SEED),
                      g[~g.derived_summary].sample(min(n - n // 2, (~g.derived_summary).sum()), random_state=SEED)]
        else:
            picks.append(g.sample(min(n, len(g)), random_state=SEED))
picks.append(t[~t.water_type_final.isin(PLAN)].sample(N_OTHER, random_state=SEED))
s = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
s.insert(0, 'sample_no', range(1, len(s) + 1))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'pngs').mkdir(exist_ok=True)
for no, d, p, src in zip(s.sample_no, s.doc_id, s.page_number, s.png):
    shutil.copy(src, OUT / 'pngs' / f'{no:02d}_{d}_page_{p}.png')

grade = s[['sample_no', 'doc_id', 'page_number', 'table_number', 'water_type_final', 'caption']].copy()
grade['caption'] = grade.caption.fillna('').str[-150:]
grade['png'] = [f'pngs/{n:02d}_{d}_page_{p}.png' for n, d, p in zip(s.sample_no, s.doc_id, s.page_number)]
for c in ['true_resolution', 'derived_summary (y/n)', 'true_parameter', 'note']:
    grade[c] = ''
grade.to_csv(OUT / 'resolution_review_sample.csv', index=False)
s[['sample_no', 'doc_id', 'page_number', 'table_number', 'water_type_final', 'resolution',
   'resolution_source', 'resolution_evidence', 'derived_summary', 'primary_parameter', 'parameters', 'llm_resolution',
   'n_values_primary']].to_csv(OUT / 'resolution_review_key.csv', index=False)

(OUT / 'README.txt').write_text("""Temporal resolution grading (blind)

For each row, open the png and find the table (table_number = its order on the page, top to
bottom; the caption is the text just above it). Fill in:

true_resolution  what ONE table's own observations are:
  daily           one value per day (day 1-31 rows x months, or a day-by-day list)
  monthly         one value per month (month rows, or year rows x month columns)
  annual          one value per year (annual peaks, yearly totals or means)
  discrete        individual measurements on given dates (field discharge measurements,
                  water levels on dates, water-quality samples, peaks above a base)
  no_time_series  no time dimension (well log, well inventory, rating/capacity table,
                  undated analyses)
  unsure / not a table   if the table can't be read or isn't on the page
derived_summary  y if the table is a summary computed from another table on this or the
                 adjacent page (e.g. "Monthly discharge" printed with the daily table), else n
true_parameter   main quantity in the table: discharge, gage_height, contents, water_level,
                 temperature, sediment, water_chemistry, precipitation, evaporation, other
note             anything useful (e.g. "daily but water temperature, not discharge")

Summary rows inside a table (Mean / Total / Max / Min) don't change the class: a daily table
with monthly means at the bottom is daily.
""", encoding='utf-8')
print(f'{len(s)} rows -> {OUT}')
print(s.groupby(['water_type_final', 'resolution']).size().to_string())
