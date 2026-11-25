"""Review folder for tables that got metadata in Anna's Nov 20 run but have no table CSV on disk.

Input:  data/analysis/evaluation/metadata/full_run/metadata_per_location.csv, data/analysis/metadata_base.csv
        (the table set before that run), data/analysis/table_value_counts.csv, data/digitized/{doc}/page_{N}/
Output: data/analysis/manual_review/new_tables_no_csv/
        {doc}_page_{N}/  copy of every file in the digitized page folder (png, json, csv) + metadata rows
        new_tables_no_csv.csv  one row per table: which files exist, Reducto Table blocks in the JSON
ONE-OFF (run 2026-11-24 against the pre-rebuild metadata_base.csv, archived in
archive/2026-11-24_before_full_metadata/). The 105 empty tables were then removed (excluded_empty_tables.csv,
read by 00a_metadata_base.py) and their page folders deleted. Refuses to run again.
"""
import json, shutil
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[3]
A = R / 'data/analysis'
OUT = A / 'manual_review/new_tables_no_csv'
K = ['doc_id', 'page_number', 'table_number']

new = pd.read_csv(A / 'evaluation/metadata/full_run/metadata_per_location.csv', dtype=str, keep_default_na=False)
for c in K:
    new[c] = new[c].astype(float).astype(int)
old = pd.read_csv(A / 'metadata_base.csv', usecols=K)
vc = pd.read_csv(A / 'table_value_counts.csv', usecols=K)
t = new[K].drop_duplicates().merge(old.drop_duplicates(), on=K, how='left', indicator=True)
t = t[t._merge == 'left_only'][K].merge(vc, on=K, how='left', indicator=True)
t = t[t._merge == 'left_only'][K].reset_index(drop=True)

def table_blocks(js):
    try:
        d = json.load(open(js, encoding='utf-8'))
    except Exception:
        return None, None
    blocks = []
    def walk(o):
        if isinstance(o, dict):
            if o.get('type') == 'Table':
                blocks.append(o.get('content', ''))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return len(blocks), blocks

if (OUT / 'new_tables_no_csv.csv').exists():
    raise SystemExit('already built (one-off); see docstring')
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for (doc, page), g in t.groupby(['doc_id', 'page_number']):
    src = R / f'data/digitized/{doc}/page_{page}'
    dst = OUT / f'{doc}_page_{page}'
    dst.mkdir(exist_ok=True)
    files = sorted(p.name for p in src.iterdir()) if src.exists() else []
    for f in files:
        shutil.copy2(src / f, dst / f)
    js = src / f'{doc}_page_{page}.json'
    n_tab, blocks = table_blocks(js) if js.exists() else (None, None)
    meta = new[(new.doc_id == doc) & (new.page_number == page)]
    meta.to_csv(dst / f'{doc}_page_{page}_new_metadata.csv', index=False)
    for k in g.table_number:
        b = blocks[k - 1] if blocks and k <= len(blocks) else ''
        m = meta[meta.table_number == k]
        rows.append(dict(doc_id=doc, page_number=page, table_number=k,
                         page_folder_exists=src.exists(), has_png=any(f.endswith('.png') for f in files),
                         has_json=js.exists(), n_csv_on_page=sum(f.endswith('.csv') and '_table' in f for f in files),
                         n_table_blocks_in_json=n_tab, table_block_chars=len(b), table_block_empty=(str(b).replace(' ', '') in ('', '<table></table>')),
                         table_block_start=' '.join(str(b).split())[:150],
                         n_metadata_rows=len(m), data_type=';'.join(sorted(set(m.data_type)))))
s = pd.DataFrame(rows)
s.to_csv(OUT / 'new_tables_no_csv.csv', index=False)
print(len(s), 'tables on', s[['doc_id', 'page_number']].drop_duplicates().shape[0], 'pages')
for c in ['page_folder_exists', 'has_png', 'has_json']:
    print(c, s[c].value_counts().to_dict())
print('JSON table block empty (<table></table>):', int(s.table_block_empty.sum()), '| has content:', int((~s.table_block_empty).sum()))
print(s[~s.table_block_empty].groupby('data_type').size().to_dict())
print('data_type:', s.data_type.value_counts().to_dict())
