"""
Base metadata table built from the new per-location metadata (Anna's pipeline; decision Annette
2026-11-18: the Phi-4 metadata is no longer used). One row per location per table.

  python 00a_metadata_base.py   -> data/analysis/metadata_base.csv  (+ metadata_base_summary.csv)

Input:  data/analysis/evaluation/metadata/full_run/metadata_per_location.csv (Qwen3.6-27B, prompt v5;
        code/01_digitization/04_metadata_extraction/), publication_list.csv (USGS document fields).

What this adds (nothing else in Anna's columns is changed; her columns keep their names):
  meta_row          row number in this file; the key the location scripts use
  id                = doc_id (name used by the older scripts)
  water_type        Anna's data_type, made to be one of the nine categories:
                      reviewer   Annette's grade (manual_review/new_metadata_gaps/blank_water_type_review)
                      page_type  blank data_type, and every other water-related location on the page has
                                 ONE type -> that type (Annette 2026-11-18)
                      evaporation  'Evaporation' -> Other, or Reservoir when measured at a reservoir
                                 (Annette 2026-11-18; Vail Lake, doc 38962 p18, read from the page)
                      model      as given
  water_type_source which of the above
  year_start, year_end, date_status   from date_start / date_end:
                      'YYYY', 'YYYY-MM', 'YYYY-MM-DD' -> parsed; 'YY-MM-DD' -> 19YY (two_digit);
                      valid years 1800 <= year <= publication year (date protocol 2026-10-28):
                      one end outside -> trimmed (that end replaced by the other); both outside ->
                      implausible (undated). No year -> missing / unparseable.
  construction_year first 4-digit year of construction_date ('1890?' -> 1890), wells only
  pub_year, title   from the publication list (not from any language model)

Dated rows: date_status in DATED (= year_start not null).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
SRC = A / 'evaluation/metadata/full_run/metadata_per_location.csv'
PUBS = ROOT / 'data/digitization_intermediates/01_download_and_preprocess/publication_list.csv'
GRADES = A / 'manual_review/new_metadata_gaps/blank_water_type_review/blank_water_type_review.csv'
OUT = A / 'metadata_base.csv'

NINE = ['Stream Discharge', 'Groundwater', 'Reservoir', 'Irrigation', 'Springs', 'Precipitation',
        'Water Quality', 'Not Water Related', 'Other']
DATED = ('parsed', 'two_digit', 'trimmed')
FULL = re.compile(r'^\s*(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\s*$')
TWO = re.compile(r'^\s*(\d{2})-(\d{2})-(\d{2})\s*$')
MDY = re.compile(r'^\s*(?:1[0-2]|0?[1-9])\s*-\s*(?:(?:3[01]|[12]\d|0?[1-9])\s*-\s*)?(\d{2})\s*$')
YEAR = re.compile(r'(?<!\d)(1[6-9]\d\d|20\d\d)(?!\d)')
# Reservoir evaporation stations (read from the page image; doc 38962 p18 table 2, 2026-11-18)
EVAP_RESERVOIR = {(38962, 18, 1, '33�30\' 116�59\'')}


def grade_to_type(g):
    g = str(g).lower()
    if 'groundwater' in g:
        return 'Groundwater'
    if 'stream' in g:
        return 'Stream Discharge'
    return None


def water_types(m):
    m['water_type'] = m['data_type']
    m['water_type_source'] = np.where(m['data_type'].isin(NINE), 'model', '')
    key = list(zip(m.doc_id, m.page_number, m.table_number))

    # reviewer grades for the tables the page rule cannot settle (Annette 2026-11-18)
    g = pd.read_csv(GRADES)
    g['t'] = g['your_water_type'].map(grade_to_type)
    # ungraded rows: 2850 p34 t3 is the same well table as t2 (Groundwater); 4538 p19 lists the same
    # street sites as 1609 p19 (graded "index of stream discharge")
    g.loc[(g.doc_id == 2850) & (g.page_number == 34) & (g.table_number == 3), 't'] = 'Groundwater'
    g.loc[(g.doc_id == 4538) & (g.page_number == 19), 't'] = 'Stream Discharge'
    graded = dict(zip(zip(g.doc_id, g.page_number, g.table_number), g['t']))
    blank = m['data_type'].eq('') & m['status'].eq('ok')

    # page rule: one water type among the page's other water-related locations
    typed = m[m['data_type'].isin(NINE) & m['data_type'].ne('Not Water Related')]
    one = typed.groupby(['doc_id', 'page_number'])['data_type'].agg(lambda s: s.iat[0] if s.nunique() == 1 else None)
    page_t = pd.Series(list(zip(m.doc_id, m.page_number)), index=m.index).map(one)
    i = blank & page_t.notna()
    m.loc[i, 'water_type'], m.loc[i, 'water_type_source'] = page_t[i], 'page_type'
    rv = pd.Series(key, index=m.index).map(graded)
    i = blank & rv.notna()
    m.loc[i, 'water_type'], m.loc[i, 'water_type_source'] = rv[i], 'reviewer'

    ev = m['data_type'].eq('Evaporation')
    at_res = pd.Series([(d, p, t, l) in EVAP_RESERVOIR for d, p, t, l in
                        zip(m.doc_id, m.page_number, m.table_number, m.location)], index=m.index)
    m.loc[ev, 'water_type'] = np.where(at_res[ev], 'Reservoir', 'Other')
    m.loc[ev, 'water_type_source'] = 'evaporation'
    m.loc[~m['water_type'].isin(NINE), 'water_type'] = ''
    return m


def year(s):
    """(year, how) from one of Anna's date strings."""
    if not s:
        return None, 'missing'
    f = FULL.match(s)
    if f:
        return int(f.group(1)), 'parsed'
    t = TWO.match(s)
    if t:
        return 1900 + int(t.group(1)), 'two_digit'
    t = MDY.match(s)      # printed month-day-year / month-year: '4 -19 -66', '6 -53'
    if t:
        return 1900 + int(t.group(1)), 'two_digit'
    return None, 'unparseable'


def dates(m):
    ys = m['date_start'].map(year)
    ye = m['date_end'].map(year)
    m['ys_raw'] = [y for y, _ in ys]
    m['ye_raw'] = [y for y, _ in ye]
    how_s = pd.Series([h for _, h in ys], index=m.index)
    how_e = pd.Series([h for _, h in ye], index=m.index)
    cap = m['pub_year'].fillna(1980)
    ok_s = m['ys_raw'].between(1800, cap)
    ok_e = m['ye_raw'].between(1800, cap)
    m['year_start'] = m['ys_raw'].where(ok_s)
    m['year_end'] = m['ye_raw'].where(ok_e)
    # a single date (start only, or end only) stands for both ends
    m['year_end'] = m['year_end'].where(m['ye_raw'].notna(), m['year_start'])
    m['year_start'] = m['year_start'].where(m['ys_raw'].notna(), m['year_end'])
    status = np.where(how_s.eq('two_digit') | how_e.eq('two_digit'), 'two_digit', 'parsed')
    has_any = m['ys_raw'].notna() | m['ye_raw'].notna()
    bad = (m['ys_raw'].notna() & ~ok_s) | (m['ye_raw'].notna() & ~ok_e)
    m['date_status'] = np.select(
        [~has_any & how_s.eq('missing') & how_e.eq('missing'), ~has_any, bad & m['year_start'].isna() & m['year_end'].isna(), bad],
        ['missing', 'unparseable', 'implausible', 'trimmed'], default=status)
    # trimmed: the invalid end is replaced by the valid one
    m['year_start'] = m['year_start'].fillna(m['year_end'])
    m['year_end'] = m['year_end'].fillna(m['year_start'])
    sw = m['year_start'] > m['year_end']
    m.loc[sw, ['year_start', 'year_end']] = m.loc[sw, ['year_end', 'year_start']].values
    m.loc[~m['date_status'].isin(DATED), ['year_start', 'year_end']] = np.nan
    m['decade_start'] = (m['year_start'] // 10) * 10
    cy = m['construction_date'].str.extract(YEAR, expand=False).astype(float)
    m['construction_year'] = cy.where(cy.between(1800, cap))
    return m.drop(columns=['ys_raw', 'ye_raw'])


def main():
    m = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    for c in ('doc_id', 'page_number', 'table_number', 'location_number', 'n_locations_in_table'):
        m[c] = pd.to_numeric(m[c], errors='coerce').astype('Int64')
    p = pd.read_csv(PUBS, usecols=['Publication ID', 'Title', 'Year Published'])
    # 32 IDs are listed twice with identical title and year: keep one
    p = p.rename(columns={'Publication ID': 'doc_id', 'Title': 'title', 'Year Published': 'pub_year'}).drop_duplicates('doc_id')
    m = m.merge(p, on='doc_id', how='left')
    m = water_types(m)
    m = dates(m)
    m.insert(0, 'meta_row', np.arange(len(m)))
    m.insert(1, 'id', m['doc_id'])
    m.to_csv(OUT, index=False)

    s = [('rows', len(m)), ('tables', m.groupby(['doc_id', 'page_number', 'table_number']).ngroups),
         ('documents', m.doc_id.nunique()), ('pages', m.groupby(['doc_id', 'page_number']).ngroups)]
    s += [(f'water_type_source {k}', v) for k, v in m.water_type_source.replace('', '(none)').value_counts().items()]
    s += [(f'water_type {k}', v) for k, v in m.water_type.replace('', '(blank)').value_counts().items()]
    s += [(f'date_status {k}', v) for k, v in m.date_status.value_counts().items()]
    wr = m[m.water_type.ne('Not Water Related') & m.status.eq('ok')]
    s += [('water-related rows dated', int(wr.date_status.isin(DATED).sum())), ('water-related rows', len(wr))]
    s = pd.DataFrame(s, columns=['what', 'n'])
    s.to_csv(A / 'metadata_base_summary.csv', index=False)
    print(s.to_string(index=False))


if __name__ == '__main__':
    main()
