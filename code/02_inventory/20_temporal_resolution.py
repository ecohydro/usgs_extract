"""
Temporal resolution per table and measurement counts by water type x resolution (2026-11-05).

Goal (Annette): tell researchers what kind of record they will find (daily / monthly / annual
/ discrete measurements) and point them to it, WITHOUT double counting. Each table gets one
primary class (temporal_resolution.py, run inside 19_value_counts.py). Derived numbers are
secondary and never added to the primary totals:
  - summary rows inside a table (Total / Mean / Max / Min / Runoff): n_summary_values
  - a monthly or annual table printed next to the daily table it was computed from (same
    page, or the adjacent page of the same document, same water type): derived_summary=True

resolution_source:
  table      decided from the table's structure (day column, month rows, year rows, dates)
  caption    decided from the table title (structure inconclusive)
  llm        (off, USE_LLM_FALLBACK) page-level LLM temporal_resolution field
             names one resolution (daily / monthly / annual) -- weakest evidence
  none       unclassified

Inputs:  data/analysis/table_value_counts.csv (19_value_counts.py, with resolution columns)
         data/analysis/table_level_metadata_published.csv, processed_metadata.parquet,
         data/analysis/tribal_lands/tribal_screen_pages.csv
Outputs: data/analysis/table_resolution.csv           one row per table CSV
         data/analysis/table_resolution_summary.csv   scope x water type x resolution
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
KEY = ['doc_id', 'page_number', 'table_number']
ORDER = ['daily', 'monthly', 'annual', 'discrete', 'no_time_series', 'unclassified']

v = pd.read_csv(A / 'table_value_counts.csv')
pub = pd.read_csv(A / 'table_level_metadata_published.csv',
                  usecols=KEY + ['water_type_final', 'number_rows', 'header_text'])
t = v.merge(pub, on=KEY, how='left')
t['published'] = t.water_type_final.notna()
td = pd.read_csv(A / 'table_dates.csv', usecols=['id', 'page_number', 'table_number', 'caption'],
                 low_memory=False).rename(columns={'id': 'doc_id'})
t = t.merge(td, on=KEY, how='left')

# ---- parameter(s) measured: table title first, then column headers ----
# A "Stream Discharge" table can be daily discharge, gage height, water temperature, sediment
# or conductance at a stream site; researchers filtering for daily discharge need to know which.
PARAMS = [
    ('discharge', r'discharge|second[- ]*f(?:ee|oo)t|sec\.?[- ]*ft|\bcfs\b|ft3/s|cubic\s+feet\s+per\s+sec|'
                  r'runoff|\bflow\b(?!ing)'),
    ('gage_height', r'gage[- ]*height|\bstage\b'),
    ('contents', r'contents|storage|capacity'),
    ('water_level', r'water[- ]*level|depth\s+to\s+water|below\s+land[- ]*surface|\blsd\b'),
    ('temperature', r'temper-?\s*ature|\bdeg\.?\s*[cf]\b|°'),
    ('sediment', r'sediment|suspended'),
    ('water_chemistry', r'conductance|dissolved|chemical|analys[ie]s|\bppm\b|mg/l|hardness|silica|'
                        r'chloride|sodium|calcium|\bph\b'),
    ('precipitation', r'rainfall|precipitation'),
    ('evaporation', r'evaporation'),
]
PARAMS = [(n, re.compile(rx, re.I)) for n, rx in PARAMS]


def params_in(text):
    """Parameters named in text, in order of first appearance."""
    hits = [(m.start(), n) for n, rx in PARAMS for m in [rx.search(text or '')] if m]
    return [n for _, n in sorted(hits)]


cap_tail = t.caption.fillna('').str[-200:]   # the title just above the table (not station notes)
p_cap = cap_tail.map(params_in)
p_head = t.header_text.fillna('').map(params_in)
t['parameters'] = [';'.join(dict.fromkeys(a + b)) for a, b in zip(p_cap, p_head)]
t['primary_parameter'] = [(a or b or ['unknown'])[0] for a, b in zip(p_cap, p_head)]
t['parameter_source'] = ['caption' if a else 'headers' if b else 'none' for a, b in zip(p_cap, p_head)]

# ---- source of the class ----
t['resolution_source'] = 'table'
t.loc[t.resolution_evidence.eq('caption'), 'resolution_source'] = 'caption'
t.loc[t.resolution.eq('unclassified'), 'resolution_source'] = 'none'

# ---- LLM fallback for unclassified tables (page-level free text) ----
md = pd.read_parquet(A / 'processed_metadata.parquet', columns=['id', 'page_number', 'temporal_resolution'])
md['doc_id'] = pd.to_numeric(md.id, errors='coerce')
md = md.dropna(subset=['doc_id', 'temporal_resolution'])


def llm_class(texts):
    s = ' '.join(texts).lower()
    hits = [c for c, rx in [('daily', r'\bdaily\b'), ('monthly', r'\bmonthly\b'),
                            ('annual', r'\bannual|\byearly\b')] if re.search(rx, s)]
    return hits[0] if len(hits) == 1 else None   # only when the LLM names exactly one


llm = (md.groupby(['doc_id', 'page_number']).temporal_resolution
         .agg(lambda s: llm_class(s.astype(str))).dropna().rename('llm_resolution').reset_index())
llm['doc_id'] = llm.doc_id.astype(int)
llm['page_number'] = llm.page_number.astype(int)
t = t.merge(llm, on=['doc_id', 'page_number'], how='left')
# NOT applied as a class: a 2026-11-05 spot check found only ~2 of 6 LLM-"daily" fallbacks were
# daily (others: monthly summaries, a capacity table, a rotated grid). Kept as a hint column.
USE_LLM_FALLBACK = False
if USE_LLM_FALLBACK:
    fb = t.resolution.eq('unclassified') & t.llm_resolution.notna()
    t.loc[fb, 'resolution'] = t.loc[fb, 'llm_resolution']
    t.loc[fb, 'resolution_source'] = 'llm'

# ---- derived summaries: monthly/annual tables beside a daily table ----
# same water type AND same primary parameter (a monthly discharge summary is not derived from
# a daily temperature table on the same page)
DCOLS = ['doc_id', 'page_number', 'water_type_final', 'primary_parameter']
daily_pages = t[t.resolution.eq('daily')][DCOLS].drop_duplicates()
near = pd.concat([daily_pages.assign(page_number=daily_pages.page_number + d) for d in (-1, 0, 1)])
near = near.drop_duplicates().assign(near_daily=True)
t = t.merge(near, on=DCOLS, how='left')
t['derived_summary'] = t.resolution.isin(['monthly', 'annual']) & t.near_daily.fillna(False).astype(bool)
t['n_values_primary'] = t.n_values.where(~t.derived_summary, 0)
t['n_values_secondary'] = t.n_summary_values.fillna(0) + t.n_values.where(t.derived_summary, 0)

out_cols = KEY + ['water_type_final', 'published', 'resolution', 'resolution_source', 'resolution_evidence',
                  'primary_parameter', 'parameters', 'parameter_source',
                  'derived_summary', 'n_values', 'n_values_primary', 'n_summary_values', 'n_values_secondary',
                  'number_rows', 'llm_resolution']
t[out_cols].to_csv(A / 'table_resolution.csv', index=False)

# ---- summary: published and catalog (published minus withheld tribal pages) ----
trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv')
trib = set(zip(trib[trib.withhold.astype(bool)].id.astype(int), trib[trib.withhold.astype(bool)].page_number.astype(int)))
p = t[t.published].copy()
p['withheld'] = [(a, b) in trib for a, b in zip(p.doc_id, p.page_number)]

rows = []
for scope, d in [('published', p), ('catalog', p[~p.withheld])]:
    for wt, g in [('ALL', d)] + list(d.groupby('water_type_final')):
        tot = g.n_values_primary.sum()
        for res in ORDER:
            h = g[g.resolution.eq(res)]
            rows.append({'scope': scope, 'water_type': wt, 'resolution': res,
                         'tables': len(h), 'tables_derived_summary': int(h.derived_summary.sum()),
                         'values_primary': int(h.n_values_primary.sum()),
                         'pct_values_primary': round(100 * h.n_values_primary.sum() / max(tot, 1), 1),
                         'values_secondary': int(h.n_values_secondary.sum()),
                         'table_rows': int(h.number_rows.sum()),
                         'tables_by_llm_fallback': int(h.resolution_source.eq('llm').sum())})
s = pd.DataFrame(rows)
s.to_csv(A / 'table_resolution_summary.csv', index=False)

pd.set_option('display.width', 200)
for wt in ['ALL', 'Stream Discharge', 'Groundwater']:
    print(f'\n=== catalog: {wt} ===')
    print(s[(s.scope == 'catalog') & (s.water_type == wt)].drop(columns=['scope', 'water_type']).to_string(index=False))
# values by resolution x primary parameter (catalog), stream discharge + groundwater
c = p[~p.withheld]
pp = (c[c.water_type_final.isin(['Stream Discharge', 'Groundwater'])]
      .groupby(['water_type_final', 'resolution', 'primary_parameter'])
      .agg(tables=('n_values_primary', 'size'), values_primary=('n_values_primary', 'sum')).reset_index())
pp.to_csv(A / 'table_resolution_by_parameter.csv', index=False)
for wt in ['Stream Discharge', 'Groundwater']:
    print(f'\n=== catalog {wt}: primary values by primary parameter x resolution ===')
    x = pp[pp.water_type_final == wt].pivot_table(index='primary_parameter', columns='resolution',
                                                  values='values_primary', fill_value=0, aggfunc='sum')
    print(x.reindex(columns=[o for o in ORDER if o in x.columns]).astype(int).to_string())
print('\nresolution_source (published):')
print(p.resolution_source.value_counts().to_string())
