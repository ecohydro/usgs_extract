"""
Restored groundwater data vs. NWIS, by SGMA groundwater basin.

Question: in which DWR Bulletin 118 groundwater basins/subbasins does the
restored data provide groundwater measurements where NWIS has none, and which
of those are SGMA High/Medium priority (2019 prioritization)?

Per basin, three levels of "new" are reported:
  never_nwis       restored sites present, ZERO NWIS wells of any start year
  no_hist_nwis     restored sites present, NWIS wells exist but none begin <= 1979
                   (restored data is the only pre-1980 record)
  predates_nwis    restored record begins before the basin's first NWIS well
                   (years of extension = nwis_first_yr - restored_first_yr)

Definitions match 07_ca_combined_6panel.py / 08_scope_extension_stats.py:
  - restored sites = unique lat/lon of Groundwater rows, clipped to CA
  - restored tables/measurements = LLM-published table-level file
  - NWIS = ca_groundwater_sites.csv (wells with groundwater-level data,
    active + discontinued, one row per site)

SGMA layer: DWR i08_B118_SGMA_2019_Basin_Prioritization (515 basins/subbasins),
downloaded 2026-10-01 from gis.water.ca.gov MapServer layer 1 (paged, simplified
with maxAllowableOffset=0.001 deg), saved to data/analysis/spatial/sgma/.

Output: data/analysis/sgma_basin_groundwater_coverage.csv (one row per basin)
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

root         = Path(__file__).resolve().parents[2]
parquet_path = root / 'data/analysis/processed_metadata.parquet'
table_path   = root / 'data/analysis/table_level_metadata_published.csv'
nwis_path    = root / 'data/analysis/nwis_sites/ca_groundwater_sites.csv'
spatial_path = root / 'data/analysis/spatial'
out_path     = root / 'data/analysis/sgma_basin_groundwater_coverage.csv'

NWIS_YEAR_CUTOFF = 1979

ca = gpd.read_file(spatial_path / 'ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
basins = gpd.read_file(spatial_path / 'sgma/sgma_2019_basin_prioritization.geojson')
basins = basins[['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority',
                 'Area_SqMiles', 'geometry']].to_crs('EPSG:4326')
basins['geometry'] = basins.geometry.make_valid()
print(f'{len(basins)} SGMA basins/subbasins')


def points(df, lon, lat):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]),
                            crs='EPSG:4326')


def to_basin(g):
    """Attach basin number to points (points outside all basins are dropped)."""
    g = gpd.sjoin(g, ca[['geometry']], how='inner', predicate='within').drop(columns='index_right')
    return gpd.sjoin(g, basins[['Basin_Subbasin_Number', 'geometry']], how='inner',
                     predicate='within').drop(columns='index_right')


# ---- restored sites (same definition as the 6-panel figure)
md = pd.read_parquet(parquet_path)
md = md[(md['water_type_clean'] == 'Groundwater') &
        md['lat_combined'].notna() & md['lon_combined'].notna()]
r_sites = to_basin(points(md.drop_duplicates(['lat_combined', 'lon_combined']),
                          'lon_combined', 'lat_combined'))

# ---- restored tables / measurements / years (published table-level file)
tl = pd.read_csv(table_path, low_memory=False)
tl = tl[(tl['water_type_final'] == 'Groundwater') &
        tl['lat_combined'].notna() & tl['lon_combined'].notna()]
r_tab = to_basin(points(tl, 'lon_combined', 'lat_combined'))

# ---- NWIS wells
nw = pd.read_csv(nwis_path, low_memory=False).drop_duplicates('site_no')
nw = nw[nw['dec_lat_va'].notna() & nw['dec_long_va'].notna()]
n_sites = to_basin(points(nw, 'dec_long_va', 'dec_lat_va'))

key = 'Basin_Subbasin_Number'
agg = pd.DataFrame({
    'restored_sites':      r_sites.groupby(key).size(),
    'restored_tables':     r_tab.groupby(key).size(),
    'restored_meas':       r_tab.groupby(key)['number_rows'].sum(),
    'restored_docs':       r_tab.groupby(key)['doc_id'].nunique(),
    'restored_first_yr':   r_tab.groupby(key)['year_start'].min(),
    'restored_last_yr':    r_tab.groupby(key)['year_end'].max(),
    'nwis_sites':          n_sites.groupby(key).size(),
    'nwis_sites_le1979':   n_sites[n_sites['begin_year'] <= NWIS_YEAR_CUTOFF].groupby(key).size(),
    'nwis_first_yr':       n_sites.groupby(key)['begin_year'].min(),
})
out = basins.drop(columns='geometry').merge(agg, left_on=key, right_index=True, how='left')
cnt = ['restored_sites', 'restored_tables', 'restored_meas', 'restored_docs',
       'nwis_sites', 'nwis_sites_le1979']
out[cnt] = out[cnt].fillna(0).astype(int)

has_r = out['restored_sites'] > 0
out['never_nwis']    = has_r & (out['nwis_sites'] == 0)
out['no_hist_nwis']  = has_r & (out['nwis_sites'] > 0) & (out['nwis_sites_le1979'] == 0)
out['predates_nwis'] = has_r & (out['restored_first_yr'] < out['nwis_first_yr'])
out['yrs_before_nwis'] = (out['nwis_first_yr'] - out['restored_first_yr']).where(out['predates_nwis'])

out = out.sort_values(['Priority', 'restored_meas'], ascending=[True, False])
out.to_csv(out_path, index=False)
print(f'Saved: {out_path}')

# ---- summary
pri_order = ['High', 'Medium', 'Low', 'Very Low']
summ = out.groupby('Priority').agg(
    basins=(key, 'size'), with_restored=('restored_sites', lambda x: (x > 0).sum()),
    never_nwis=('never_nwis', 'sum'), no_hist_nwis=('no_hist_nwis', 'sum'),
    predates_nwis=('predates_nwis', 'sum')).reindex(pri_order)
print('\n', summ)

cols = ['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority', 'restored_sites',
        'restored_meas', 'restored_docs', 'restored_first_yr', 'restored_last_yr',
        'nwis_sites', 'nwis_sites_le1979', 'nwis_first_yr', 'yrs_before_nwis']
hm = out[out['Priority'].isin(['High', 'Medium'])]
pd.set_option('display.width', 250)
for lab, mask in [('NEVER any NWIS well', hm['never_nwis']),
                  ('NWIS present but none before 1980', hm['no_hist_nwis']),
                  ('Restored predates first NWIS well', hm['predates_nwis'])]:
    print(f'\n=== High/Medium priority — {lab}: {mask.sum()} basins ===')
    print(hm[mask].sort_values('restored_meas', ascending=False)[cols].to_string(index=False))
