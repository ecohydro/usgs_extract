"""
Entry-level metadata for analyses: one row per location per table (2026-11-18, new metadata).
Replaces the role of 00_data_prep.ipynb, which cleaned the Phi-4 metadata (no longer used).

  python 00_data_prep.py   -> data/analysis/processed_metadata.parquet

Inputs:
  data/analysis/metadata_base.csv              00a_metadata_base.py (Anna's table + water type, dates)
  data/analysis/location_validation/locations.csv   04_build_locations.py (rebuilt locations)
  data/analysis/page_types.csv                 10_page_types.py
  data/analysis/table_value_counts.csv         19_value_counts.py (measurements per table; optional)

Column names kept from the earlier pipeline so the figure scripts need no change:
  water_type_clean, year_start, year_end, decade_start, date_status, lat_combined, lon_combined,
  coord_method, precision_km, precision_class, evidence_scope, location_check, nwis_site_no,
  nwis_site_source, well_number (rebuilt), page_type, is_data_page, watersource_name,
  matched_tables (= the row's own table number), n_measurements, Title, Year Published.
n_measurements = the table's measurement count divided by its number of locations (an even split; the
count is per table, see 19_value_counts.py), so summing over rows gives the table totals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
sys.path.insert(0, str(ROOT / 'code/01_digitization/04_metadata_extraction/location_validation'))
from meta_source import load_meta  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_filters import NON_DATA_PAGE_TYPES, apply_review  # noqa: E402

LOC_COLS = ['meta_row', 'lat', 'lon', 'coord_method', 'precision_km', 'evidence_scope', 'location_check',
            'n_sources', 'max_disagree_km', 'stream_check_km', 'precision_class', 'nwis_site_no',
            'nwis_site_source', 'well_number', 'gnis_place', 'b118_basin', 'row_class', 'row_states',
            'page_state_class', 'page_states', 'continuation_rule']


def main():
    m = load_meta()
    m = m.rename(columns={'water_type': 'water_type_clean', 'title': 'Title', 'pub_year': 'Year Published',
                          'well_number': 'well_number_printed'})
    loc = pd.read_csv(A / 'location_validation/locations.csv', low_memory=False, dtype={'nwis_site_no': str})
    loc = loc[[c for c in LOC_COLS if c in loc.columns]].rename(columns={'lat': 'lat_combined', 'lon': 'lon_combined'})
    m = m.merge(loc, on='meta_row', how='left')
    m['precision_class'] = m['precision_class'].fillna('unlocated')
    m['coord_method'] = m['coord_method'].fillna('unlocated')

    pt = pd.read_csv(A / 'page_types.csv', usecols=['id', 'page_number', 'page_type'])
    m = m.merge(pt, on=['id', 'page_number'], how='left')
    m['page_type'] = m['page_type'].fillna('data')
    m['is_data_page'] = ~m['page_type'].isin(NON_DATA_PAGE_TYPES)
    m['matched_tables'] = m['table_number'].astype('Int64').astype(str)

    vc = A / 'table_value_counts.csv'
    if vc.exists():
        v = pd.read_csv(vc, usecols=['doc_id', 'page_number', 'table_number', 'n_measurements'])
        v = v.rename(columns={'doc_id': 'id', 'n_measurements': 'table_measurements'})
        m = m.merge(v, on=['id', 'page_number', 'table_number'], how='left')
        n_loc = m.groupby(['id', 'page_number', 'table_number'])['meta_row'].transform('size')
        m['n_measurements'] = m['table_measurements'] / n_loc
    # reviewer verdicts (manual_review/earliest_records_review.csv): retype / undate / set_year /
    # construction_year / unlocate fix the published data too; 'drop' rows are kept (drop=False:
    # 'drop' = not usable for a paper claim, the figure scripts apply it themselves)
    m = apply_review(m, id_col='id', page_col='page_number', type_col='water_type_clean', drop=False)
    m = m.sort_values('meta_row').reset_index(drop=True)
    m.to_parquet(A / 'processed_metadata.parquet', index=False)
    print(f'rows {len(m):,}; data-page rows {int(m.is_data_page.sum()):,}')
    print(m['precision_class'].value_counts().to_string())
    print('dated (year_start):', int(m['year_start'].notna().sum()))


if __name__ == '__main__':
    main()
