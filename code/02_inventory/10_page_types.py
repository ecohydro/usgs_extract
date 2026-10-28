"""
Flag pages that are not measurement data (agreed 2026-10-28, option 3 of the groundwater
location work): index / contents pages and narrative pages with no table.

page_type
  index_contents  index, table of contents, list of illustrations/tables (e.g. doc 2252 p944,
                  whose LLM date "1850-1899" became a false "earliest stream record")
  narrative       no table at all: no table CSV and no <table> in the Reducto output
  no_page_text    no Reducto JSON for the page; cannot tell, so kept (not excluded)
  data            everything else (incl. pages with a Reducto table but no CSV, so possible
                  data is not dropped)

Rows on index_contents / narrative pages are kept in the metadata but excluded from maps
and statistics (00_data_prep joins page_type; 07/08 filter page_type == 'data').

Inputs:  data/metadata/main_metadata.csv, data/digitized/{id}/page_{n}/
Output:  data/analysis/page_types.csv  (one row per metadata page)
"""

import json
import re
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[2]
RE_INDEX = re.compile(r'^[\s#\d.|-]*(index|contents|table of contents|illustrations|list of '
                      r'(tables|illustrations|figures|plates))\b', re.I)
RE_LEADERS = re.compile(r'(\.\s?){6,}\s*\d+')          # "Introduction ........ 5"

meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', usecols=['id', 'page_number'], low_memory=False)
pages = meta.dropna().astype(int).drop_duplicates()

recs = []
for doc, page in pages.itertuples(index=False):
    folder = root / f'data/digitized/{doc}/page_{page}'
    n_csv = len(list(folder.glob('*_table*.csv'))) if folder.exists() else 0
    jp = folder / f'{doc}_page_{page}.json'
    text, has_html_table = '', False
    if jp.exists():
        text = ' '.join(c.get('content', '') for c in json.loads(jp.read_text(encoding='utf-8'))['result']['chunks'])
        has_html_table = '<table' in text
    plain = re.sub(r'<[^>]+>', ' ', text)
    head = re.sub(r'\s+', ' ', plain)[:160]
    is_index = bool(RE_INDEX.search(head)) or len(RE_LEADERS.findall(plain)) >= 8
    if not jp.exists():
        ptype = 'no_page_text'           # cannot tell; kept (not excluded)
    elif is_index:
        ptype = 'index_contents'
    elif n_csv == 0 and not has_html_table:
        ptype = 'narrative'
    else:
        ptype = 'data'
    recs.append({'id': doc, 'page_number': page, 'page_type': ptype, 'n_table_csv': n_csv,
                 'has_reducto_table': has_html_table, 'has_json': jp.exists(), 'page_start': head[:100]})

pt = pd.DataFrame(recs)
pt.to_csv(root / 'data/analysis/page_types.csv', index=False)
print(pt['page_type'].value_counts().to_string())
print('\nno JSON:', (~pt['has_json']).sum())
for t in ['index_contents', 'narrative']:
    print(f'\n--- sample {t}')
    print(pt[pt['page_type'] == t].sample(min(8, (pt['page_type'] == t).sum()), random_state=1)[
        ['id', 'page_number', 'n_table_csv', 'page_start']].to_string(index=False))
