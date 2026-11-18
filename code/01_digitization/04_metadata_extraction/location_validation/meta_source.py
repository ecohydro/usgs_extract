"""
Metadata rows for the location rebuild (2026-11-18). Since the switch to the new per-location metadata
(Anna's pipeline; the Phi-4 metadata is no longer used) every location script reads its rows here.

One row per location per table, from data/analysis/metadata_base.csv (code/02_inventory/00a_metadata_base.py).
`meta_row` is the row number in that file. The new metadata copies its location words from the page
(a found-on-page check drops anything not printed there), so these fields are pointers into the page,
as the Phi-4 text was before; every method still checks the page itself.

Columns under the names the location scripts used before:
  id, page_number, meta_row, water_type
  watersource_name           station_name if printed, else water_source (river, lake, spring; well owner)
  location                   "<place_relation> <place_name>, <basin_or_area>, <county>, <state>"
  townships_ranges_sections  township_range_section
plus the new evidence columns as they are: coordinates, well_number, station_number, station_name,
water_source, township_range_section, place_name, place_relation, basin_or_area, county, state,
table_number, location_number, n_locations_in_table, evidence_place, status, title, pub_year.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'data/analysis/metadata_base.csv'
EVIDENCE = ['coordinates', 'well_number', 'station_number', 'station_name', 'water_source',
            'township_range_section', 'place_name', 'place_relation', 'basin_or_area', 'county', 'state']


def _join(*parts):
    return ', '.join(p for p in parts if p)


def load_meta(columns=None):
    """All metadata rows (data pages and others; filter with page_types.csv where needed)."""
    m = pd.read_csv(BASE, dtype={c: str for c in EVIDENCE + ['location', 'title', 'evidence_place', 'status']},
                    keep_default_na=False, low_memory=False)
    for c in ('id', 'doc_id', 'page_number', 'table_number', 'location_number', 'meta_row', 'pub_year',
              'n_locations_in_table', 'year_start', 'year_end', 'decade_start', 'construction_year'):
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m['watersource_name'] = m['station_name'].where(m['station_name'].ne(''), m['water_source'])
    place = (m['place_relation'] + ' ' + m['place_name']).str.strip()
    m['new_location'] = m['location']                      # Anna's best-evidence location string
    m['location'] = [_join(a, b, c, d) for a, b, c, d in zip(place, m['basin_or_area'], m['county'], m['state'])]
    m['townships_ranges_sections'] = m['township_range_section']
    # blank strings -> NaN, as in the old metadata (scripts test pd.isna / notna)
    for c in ['watersource_name', 'location', 'townships_ranges_sections'] + EVIDENCE:
        m[c] = m[c].replace('', pd.NA)
    if columns:
        m = m[[c for c in dict.fromkeys(['meta_row'] + list(columns))]]
    return m
