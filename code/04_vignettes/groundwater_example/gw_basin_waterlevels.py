"""
Figure 5 (groundwater example), basin ranking at TABLE level, 2026-11-16.

Site-level ranking (gw_statewide_candidates.py) was misled by groundwater-typed tables that are not water
levels (pumpage 38939 p41, pumped volumes 1147 p114, chemistry 2507 p116) and by unnumbered wells located
only by a town point. Here: catalog tables (published, tribal out, rebuilt locations) that are
  groundwater, primary_parameter == water_level, measurement pages, measurement dates (no fallbacks),
  start <= 1979, inside a DWR B118 (sub)basin (catalog b118_basin_id)
Restored readings = n_values_primary (numeric values in the table, 19_value_counts.py), by start decade.
NWIS readings = water-level field measurements per well and year (nwis_years_gw.csv), wells inside the basin.

Output: data/analysis/gw_example/gw_basin_waterlevels.csv (one row per basin) and
        gw_basin_waterlevels_by_decade.csv (basin x decade: restored vs NWIS readings and wells)
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
from paper_filters import measurement_page_mask, data_page_mask  # noqa: E402
from date_parsing import FALLBACK_STATUSES                       # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'gw_example'
CAP = 1979

c = pd.read_csv(ROOT / 'data/catalog/restored_tables_catalog.csv', low_memory=False)
c = c[(c.water_type == 'Groundwater') & (c.primary_parameter == 'water_level')
      & data_page_mask(c) & measurement_page_mask(c)
      & c.year_start.notna() & (c.year_start <= CAP) & ~c.date_status.isin(FALLBACK_STATUSES)
      & c.b118_basin_id.notna()].copy()
c['decade'] = (c.year_start // 10 * 10).astype(int)
print(f'water-level tables in a B118 basin: {len(c):,}; readings {int(c.n_values_primary.sum()):,}')

b = gpd.read_file(A / 'spatial/sgma/sgma_2019_basin_prioritization.geojson')
b['geometry'] = b.geometry.make_valid().buffer(0)
b = b.to_crs('EPSG:4326')[['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority', 'geometry']]

nw = pd.read_csv(A / 'nwis_sites/ca_groundwater_sites.csv', low_memory=False, dtype={'site_no': str})
nw = nw.dropna(subset=['dec_lat_va']).drop_duplicates('site_no')
nwg = gpd.GeoDataFrame(nw[['site_no']], geometry=gpd.points_from_xy(nw.dec_long_va, nw.dec_lat_va),
                       crs='EPSG:4326')
nwg = gpd.sjoin(nwg, b, predicate='within').drop_duplicates('site_no')[['site_no', 'Basin_Subbasin_Number']]
ny = pd.read_csv(A / 'nwis_sites/nwis_years_gw.csv', dtype={'site_no': str})
ny = ny[ny.year <= CAP].merge(nwg, on='site_no')
ny['decade'] = ny.year // 10 * 10

r = c.groupby(['b118_basin_id', 'decade']).agg(restored_readings=('n_values_primary', 'sum'),
                                                restored_tables=('table_csv', 'size'),
                                                restored_docs=('doc_id', 'nunique')).reset_index()
r = r.rename(columns={'b118_basin_id': 'Basin_Subbasin_Number'})
n = ny.groupby(['Basin_Subbasin_Number', 'decade']).agg(nwis_readings=('n_measurements', 'sum'),
                                                        nwis_wells=('site_no', 'nunique')).reset_index()
d = r.merge(n, on=['Basin_Subbasin_Number', 'decade'], how='outer').fillna(0)
d = d.merge(b.drop(columns='geometry'), on='Basin_Subbasin_Number', how='left').sort_values(
    ['Basin_Subbasin_Number', 'decade'])
d.to_csv(OUT / 'gw_basin_waterlevels_by_decade.csv', index=False)

pre = d[d.decade < 1950]
s = d.groupby(['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority']).agg(
    restored_readings=('restored_readings', 'sum'), restored_tables=('restored_tables', 'sum'),
    nwis_readings=('nwis_readings', 'sum')).reset_index()
p = pre.groupby('Basin_Subbasin_Number').agg(restored_pre1950=('restored_readings', 'sum'),
                                             nwis_pre1950=('nwis_readings', 'sum'),
                                             nwis_wells_pre1950_max=('nwis_wells', 'max'))
s = s.merge(p, left_on='Basin_Subbasin_Number', right_index=True, how='left').fillna(0)
first_r = d[d.restored_readings > 0].groupby('Basin_Subbasin_Number').decade.min().rename('restored_first_decade')
first_n = d[d.nwis_readings > 0].groupby('Basin_Subbasin_Number').decade.min().rename('nwis_first_decade')
s = s.join(first_r, on='Basin_Subbasin_Number').join(first_n, on='Basin_Subbasin_Number')
s['docs'] = s.Basin_Subbasin_Number.map(c.groupby('b118_basin_id').doc_id.nunique())
s = s.sort_values('restored_pre1950', ascending=False)
s.to_csv(OUT / 'gw_basin_waterlevels.csv', index=False)
pd.set_option('display.width', 250)
print(s.head(25).to_string(index=False))
