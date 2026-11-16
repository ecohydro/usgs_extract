"""
Blind grading sample: which well does a single-well water-level table belong to? (2026-11-15)

Context: 12_table_owner_wells_scope.py found that the current well assignment (07_well_numbers.py,
a well number printed on the SAME page) and a candidate "table owner" rule (the last well
description before the table, possibly on an earlier page) disagree for 2,132 records. This sample
lets a person decide which is right, by group, to choose a rule.

Groups (from data/analysis/location_validation/table_owner_scope_records.csv):
  G1 disagree, owner described on the PREVIOUS page
  G2 disagree, owner described 2+ pages earlier
  G3 disagree, owner described on the SAME page
  G4 agree (control)
Max 2 items per document per group; doc 2060 left out (already checked by hand: p203).
Candidates shown as A / B in random order (blind); the key is in a separate file.

Outputs in data/analysis/manual_review/well_owner_check/:
  well_owner_review_sample.csv  (grade this)   well_owner_review_key.csv  (do not open while grading)
  pngs/  page images, {sample_no}_page.png and {sample_no}_prev_page.png (git-ignored)
"""
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
A = ROOT / 'data/analysis'
DIG = ROOT / 'data/digitized'
OUT = A / 'manual_review/well_owner_check'
PNG = OUT / 'pngs'
PNG.mkdir(parents=True, exist_ok=True)
N = {'G1': 12, 'G2': 12, 'G3': 12, 'G4': 6}
rng = random.Random(20261115)

w = pd.read_csv(A / 'location_validation/table_owner_scope_records.csv', low_memory=False, dtype={'id': str})
t = pd.read_csv(A / 'location_validation/table_owner_scope_tables.csv', low_memory=False, dtype={'id': str})
t = t[t.single_well]
w = w[(w.id != '2060') & ~w.owner_wells.str.contains(';', na=False)]   # one owner -> one clear question


def group(r):
    if r.verdict == 'agree':
        return 'G4'
    return {'previous page': 'G1', 'earlier page': 'G2', 'page': 'G3'}.get(r.owner_scopes)


w['group'] = w.apply(group, axis=1)
w = w[w.group.notna()]


def pick_table(r):
    ts = [int(float(x)) for x in str(r.matched_tables).split(',') if x not in ('nan', '')]
    cand = t[(t.id == r.id) & (t.page_number == r.page_number) & (t.owner_well == r.owner_wells)]
    if ts:
        cand = cand[cand.table_idx.isin(ts)] if cand.table_idx.isin(ts).any() else cand
    return cand.iloc[0] if len(cand) else None


items = []
for g, n in N.items():
    pool = w[w.group == g].sample(frac=1, random_state=rng.randint(0, 10**6))
    per_doc = {}
    for r in pool.itertuples(index=False):
        if per_doc.get(r.id, 0) >= 2 or len([i for i in items if i['group'] == g]) >= n:
            continue
        tb = pick_table(r)
        if tb is None:
            continue
        # same page+table only once
        if any(i['id'] == r.id and i['page_number'] == r.page_number and i['table_idx'] == tb.table_idx for i in items):
            continue
        per_doc[r.id] = per_doc.get(r.id, 0) + 1
        items.append(dict(group=g, meta_row=r.meta_row, id=r.id, page_number=int(r.page_number),
                          table_idx=int(tb.table_idx), current_well=r.pipe_well, rule_well=r.owner_wells,
                          owner_page=int(tb.owner_page), match_rule=r.match_rule))

rng.shuffle(items)
rows, keys = [], []
for k, it in enumerate(items, 1):
    cands = [it['current_well'], it['rule_well']]
    if cands[0] != cands[1]:
        rng.shuffle(cands)
    a, b = cands
    page = DIG / it['id'] / f'page_{it["page_number"]}'
    csvs = sorted(page.glob(f'*_table{it["table_idx"]}.csv'))
    first = ''
    if csvs:
        lines = csvs[0].read_text(encoding='utf-8', errors='replace').splitlines()
        first = ' / '.join(lines[:3])[:250]
    src = page / f'{it["id"]}_page_{it["page_number"]}.png'
    if src.exists():
        shutil.copy(src, PNG / f'{k:02d}_page.png')
    prev = DIG / it['id'] / f'page_{it["page_number"] - 1}' / f'{it["id"]}_page_{it["page_number"] - 1}.png'
    if prev.exists():
        shutil.copy(prev, PNG / f'{k:02d}_prev_page.png')
    n_tables = len(list(page.glob('*_table*.csv')))
    rows.append(dict(sample_no=k, doc=it['id'], scan_page=it['page_number'],
                     table_on_page=f'{it["table_idx"]} of {n_tables}', table_first_rows=first,
                     candidate_A=a, candidate_B=b, png=f'{k:02d}_page.png',
                     prev_png=f'{k:02d}_prev_page.png' if prev.exists() else '',
                     answer='', true_well_if_other='', notes=''))
    keys.append(dict(sample_no=k, **it, A_is=('current' if a == it['current_well'] else 'rule')
                     if a != b else 'both'))

pd.DataFrame(rows).to_csv(OUT / 'well_owner_review_sample.csv', index=False)
pd.DataFrame(keys).to_csv(OUT / 'well_owner_review_key.csv', index=False)
print(f'{len(rows)} items:', pd.Series([i['group'] for i in items]).value_counts().sort_index().to_dict(),
      f'from {len({i["id"] for i in items})} documents')
