"""
Flag pages that are not measurement data (agreed 2026-10-28, option 3 of the groundwater
location work): index / contents pages and narrative pages with no table.

page_type
  index_contents  index, table of contents, list of illustrations/tables (e.g. doc 2252 p944,
                  whose LLM date "1850-1899" became a false "earliest stream record")
  report_list     list of earlier reports with the period each covers ("Reports containing
                  compilations of records of discharge ...", "State | Period | Report | Issued by",
                  "Water-Supply Paper | Title"), printed in the front of Surface Water Supply
                  volumes. Not measurements; its "Period" column (e.g. Massachusetts 1845-1915)
                  had been read as table dates (added 2026-11-14)
  narrative       no table at all: no table CSV and no <table> in the Reducto output
  no_page_text    no Reducto JSON for the page; cannot tell, so kept (not excluded)
  data            everything else (incl. pages with a Reducto table but no CSV, so possible
                  data is not dropped)

Rows on index_contents / report_list / narrative pages are kept in the metadata but excluded from maps
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
RE_REPORT_CAPTION = re.compile(r'titles of these reports|reports containing compilations', re.I)
RE_REPORT_TABLE = re.compile(r'Issued\s+by|Passaic flood of 19\d\d|titles of these reports'
                             r'|Destructive floods in the United States|drainage basin and title'
                             r'|Character of data'
                             r'|^\W*(Water.Supply\s+)?(Title\s+)?Paper\W+Title', re.I)
RE_REPORT_WORD = re.compile(r'\bRep[eo]rt\b', re.I)          # "Report", OCR "Repert"
RE_WSP = re.compile(r'W\.?\s?S\.?\s?P|Water.Supply', re.I)
RE_YEAR_WSP = re.compile(r'Year[\W\d]*W\.?S\.?P.*Year[\W\d]*W\.?S\.?P', re.I)   # "Year,WSP,Year.1,WSP.1,..."


def is_report_table(head):
    """First ~300 characters of a table CSV look like a list of reports."""
    if RE_REPORT_TABLE.search(head):
        return True
    first = head.split('\n', 1)[0]
    if RE_YEAR_WSP.search(first):
        return True
    return bool(re.search(r'\bPeriod\b', first, re.I) and RE_REPORT_WORD.search(first)
                and RE_WSP.search(first))


RE_LEADERS = re.compile(r'(\.\s?){6,}\s*\d+')          # "Introduction ........ 5"

meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', usecols=['id', 'page_number'], low_memory=False)
pages = meta.dropna().astype(int).drop_duplicates()

recs = []
for doc, page in pages.itertuples(index=False):
    folder = root / f'data/digitized/{doc}/page_{page}'
    csvs = sorted(folder.glob('*_table*.csv')) if folder.exists() else []
    n_csv = len(csvs)
    n_report = sum(is_report_table(c.read_text(encoding='utf-8', errors='replace')[:300]) for c in csvs)
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
    elif n_csv and (n_report == n_csv or (n_report and RE_REPORT_CAPTION.search(plain))):
        ptype = 'report_list'
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
for t in ['index_contents', 'report_list', 'narrative']:
    print(f'\n--- sample {t}')
    print(pt[pt['page_type'] == t].sample(min(8, (pt['page_type'] == t).sum()), random_state=1)[
        ['id', 'page_number', 'n_table_csv', 'page_start']].to_string(index=False))
