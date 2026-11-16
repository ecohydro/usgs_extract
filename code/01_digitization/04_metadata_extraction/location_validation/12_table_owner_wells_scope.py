"""
SCOPING ONLY (2026-11-15): how often would a "table owner" rule change well assignments statewide?

Problem found in doc 2060 (WSP 1068) p203: water-level reports print each well's DESCRIPTION
("7/34-25D1. City of Santa Barbara, well 9-D ...") ABOVE its table, and tables run onto the next page.
07_well_numbers.py links a row to a well number printed on the SAME page ("only well on page",
"well number in LLM text", "owner name in well heading"), so a table continued from the previous
page gets the NEXT well's number (p203: the 1930-40 table belongs to 7/34-24N2 described on p202,
not to 7/34-25D1 described below the table).

Candidate rule (NOT applied anywhere yet): a single-well water-level table belongs to the last well
description before it in reading order (Reducto block order, across pages).
  description block  = non-table block whose text STARTS with a well number (first 12 characters)
  single-well table  = a 'water level' title between the description and the table, or a table
                       header with 'date' and 'level'; tables whose header has 'owner' are
                       inventories (many wells) and are excluded

Outputs: data/analysis/location_validation/table_owner_scope_tables.csv (one row per table)
         data/analysis/location_validation/table_owner_scope_records.csv (tier2b rows vs owner)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
A = ROOT / 'data/analysis'
LV = A / 'location_validation'

# same pattern as 07_well_numbers.py RE_WELL (copied for this scoping script; move to a shared module
# if the rule is adopted)
TRACT = r'[A-HJ-NP-R8]'
RE_WELL = re.compile(r'(?<![\d/.])(\d{1,2})\s?([NS5])?\s?/\s?(\d{1,2})\s?([EWF])?\s?[-–—]\s?'
                     r'(\d{1,2})\s?(' + TRACT + r')\s?(\d{1,3})(?:\s?([MSH])(?![a-z]))?')
RE_WL_TITLE = re.compile(r'water[\s-]*levels?', re.I)


def key(t, r, sec, tract, seq):
    return f'{int(t)}/{int(r)}-{int(sec)}{"B" if tract == "8" else tract}{int(seq)}'


def lead_well(text):
    m = RE_WELL.search(text[:40])
    if not m or m.start() > 12:
        return None
    t, _, r, _, sec, tract, seq, _ = m.groups()
    if not (1 <= int(t) <= 60 and 1 <= int(r) <= 60 and 1 <= int(sec) <= 36):
        return None
    return key(t, r, sec, tract, seq)


b = pd.read_parquet(A / 'page_blocks.parquet')
b['id'] = b['id'].astype(str)
b = b.sort_values(['id', 'page_number', 'block_order'])
docs_with_desc = set()
rows = []
for doc, d in b.groupby('id', sort=False):
    owner = None            # (well, page)
    wl_title = False
    for r in d.itertuples(index=False):
        c = str(r.content)
        if r.block_type != 'Table':
            w = lead_well(c)
            if w:
                owner, wl_title = (w, r.page_number), False
                docs_with_desc.add(doc)
            elif RE_WL_TITLE.search(c[:200]):
                wl_title = True
            continue
        head = c[:600].lower()
        inventory = 'owner' in head
        wl_header = 'date' in head and 'level' in head
        rows.append(dict(id=doc, page_number=r.page_number, table_idx=r.table_idx,
                         owner_well=owner[0] if owner else None,
                         owner_page=owner[1] if owner else None,
                         single_well=bool(owner) and not inventory and (wl_title or wl_header)))

t = pd.DataFrame(rows)
t = t[t.id.isin(docs_with_desc)]
t['owner_scope'] = t.apply(lambda x: None if pd.isna(x.owner_page) else
                           'page' if x.owner_page == x.page_number else
                           'previous page' if x.owner_page == x.page_number - 1 else 'earlier page', axis=1)
# wells described on each page (to see whether the page itself prints a different well)
desc = b[b.block_type != 'Table'].assign(w=lambda x: x.content.astype(str).map(lead_well)).dropna(subset=['w'])
page_wells = desc.groupby(['id', 'page_number']).w.apply(set)
t['page_wells'] = [page_wells.get((i, p), set()) for i, p in zip(t.id, t.page_number)]
t['page_prints_other_well'] = [bool(pw) and o not in pw for pw, o in zip(t.page_wells, t.owner_well)]
sw = t[t.single_well]
print(f'documents with well-description blocks: {len(docs_with_desc)}')
print(f'tables in those documents: {len(t)}; single-well water-level tables with an owner: {len(sw)} '
      f'in {sw.id.nunique()} documents')
print('owner described on:', sw.owner_scope.value_counts().to_dict())
print(f'single-well tables whose owner is on an earlier page while the page prints another well: '
      f'{(sw.owner_scope != "page").sum()} / {(sw.page_prints_other_well & (sw.owner_scope != "page")).sum()}')
t.drop(columns='page_wells').to_csv(LV / 'table_owner_scope_tables.csv', index=False)

# ------------------------------------------------------------------ compare with current assignments
wn = pd.read_csv(LV / 'tier2b_well_numbers.csv', low_memory=False)
wn = wn[wn.well.notna() & wn.sec.notna()].copy()
wn['id'] = wn['id'].astype(str)
wn['pipe_well'] = [key(a, b_, c, d_, e) for a, b_, c, d_, e in zip(wn.t, wn.r, wn.sec, wn.tract, wn.seq)]
ent = pd.read_csv(A / 'entry_dates.csv', usecols=['meta_row', 'matched_tables'], low_memory=False)
wn = wn.merge(ent, on='meta_row', how='left')
swi = sw.set_index(['id', 'page_number', 'table_idx'])


def owners(r):
    ts = [int(float(x)) for x in str(r.matched_tables).split(',') if x not in ('nan', '')]
    if not ts:
        ts = list(sw[(sw.id == r.id) & (sw.page_number == r.page_number)].table_idx)
    o = [swi.loc[(r.id, r.page_number, x)] for x in ts if (r.id, r.page_number, x) in swi.index]
    return pd.Series({'owner_wells': ';'.join(sorted({x.owner_well for x in o})),
                      'owner_scopes': ';'.join(sorted({x.owner_scope for x in o}))})


wn = wn[wn.set_index(['id', 'page_number']).index.isin(sw.set_index(['id', 'page_number']).index)]
wn = pd.concat([wn, wn.apply(owners, axis=1)], axis=1)
wn = wn[wn.owner_wells != '']
wn['verdict'] = ['agree' if p in o.split(';') else 'differs' for p, o in zip(wn.pipe_well, wn.owner_wells)]
print(f'\ncurrent well-number records on single-well water-level tables: {len(wn)}')
print(wn.groupby(['verdict', 'match_rule']).size().unstack(fill_value=0).to_string())
print('differs, by owner scope:', wn[wn.verdict == 'differs'].owner_scopes.value_counts().to_dict())
print(f'documents with a differing record: {wn[wn.verdict == "differs"].id.nunique()}; '
      f'differing records that carry an NWIS link: {wn[(wn.verdict == "differs") & wn.nwis_site_no.notna()].shape[0]}')
wn.drop(columns=[c for c in wn.columns if c in ('group_wells',)]).to_csv(LV / 'table_owner_scope_records.csv', index=False)
