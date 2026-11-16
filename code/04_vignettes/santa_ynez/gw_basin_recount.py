"""
Santa Ynez groundwater, steps 1 and 3 of the agreed plan (2026-11-15):
  1. recount the Santa Ynez River Valley basin (DWR B118 3-015) groundwater sites under the current
     paper rules (same records and site definition as Figure 3: code/02_inventory/site_activity.py,
     site_B1 = identity + 1 km; measurement pages only; measurement dates only; precise/approximate
     locations; tribal pages out; reviewer decisions applied)
  3. link restored wells to NWIS wells by state well number only (tier2b nwis_site_no), as in Figure 3

Outputs (data/analysis/santa_ynez/):
  gw_basin_sites.csv        one row per restored GW site in the basin (decades, first year, NWIS link)
  gw_basin_nwis_wells.csv   NWIS GW wells in the basin with a water-level record beginning <= 1979
  gw_basin_links.csv        restored site <-> NWIS well pairs: restored decades vs NWIS decades
  gw_basin_by_doc.csv       restored records in the basin by document (to pick candidate wells, step 4)
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
from site_activity import load_records, nwis_table, all_nwis_coords, restored_sites, CAP  # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'santa_ynez'
BASIN = '3-015'

b118 = gpd.read_file(A / 'spatial/sgma/sgma_2019_basin_prioritization.geojson')
b118['geometry'] = b118.geometry.make_valid().buffer(0)
poly = b118[b118.Basin_Subbasin_Number == BASIN].to_crs('EPSG:4326')[['geometry']]
print(f'basin {BASIN}: {len(poly)} polygon(s)')


def inside(df, lat, lon):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs='EPSG:4326')
    return gpd.sjoin(g, poly, how='inner', predicate='within').drop(columns=['index_right', 'geometry'])


# ---------------------------------------------------------------- step 1: restored sites
csv = 'ca_groundwater_sites.csv'
rec = load_records()
gw_rec = rec[rec.water_type == 'Groundwater']
sites = restored_sites(rec, 'Groundwater', all_nwis_coords(csv), 'site_B1').reset_index()
sites = inside(sites, 'lat', 'lon')
rec_in = inside(gw_rec, 'lat_combined', 'lon_combined')
print(f'restored GW records (entries) in basin: {len(rec_in)}; sites (B1): {len(sites)}; '
      f'documents: {rec_in.doc_id.nunique()}')
sites['decades'] = sites.dec.map(lambda d: ' '.join(str(x) for x in sorted(d)))
print('first measured year of restored sites:', sites.first_year.min(), '-', sites.first_year.max())
print('restored sites active per decade:')
print(sites.dec.explode().value_counts().sort_index().to_string())

# ---------------------------------------------------------------- NWIS wells
nw = nwis_table(csv).reset_index()
nw = inside(nw, 'dec_lat_va', 'dec_long_va')
nw['decades_str'] = nw.decades.map(lambda d: ' '.join(str(x) for x in sorted(d)))
print(f'\nNWIS GW wells in basin with a water-level series beginning <= {CAP}: {len(nw)}; '
      f'first year {nw.first_year.min():.0f}')
print('NWIS wells active per decade (actual years with data where downloaded; decades_source):')
print(nw.decades.explode().value_counts().sort_index().to_string())

# ---------------------------------------------------------------- step 3: links by well number
lk = sites[sites.nwis_link.notna()].copy()
lk['nwis_link'] = lk.nwis_link.astype(str).str.split(';').str[0].str.replace(r'\.0$', '', regex=True)
nwd = nw.set_index('site_no').decades
lk['nwis_in_basin_list'] = lk.nwis_link.isin(nwd.index)
lk['nwis_decades'] = lk.nwis_link.map(lambda s: ' '.join(str(x) for x in sorted(nwd.get(s, set()))))
lk['restored_only_decades'] = [' '.join(str(x) for x in sorted(d - nwd.get(s, set())))
                               for d, s in zip(lk.dec, lk.nwis_link)]
print(f'\nrestored sites linked to an NWIS well by well number: {len(lk)} of {len(sites)}; '
      f'linked well in the NWIS <= {CAP} basin list: {lk.nwis_in_basin_list.sum()}')
print(f'linked sites with restored decades NWIS lacks: {(lk.restored_only_decades != "").sum()}')

# ---------------------------------------------------------------- records by document (for step 4)
docs = rec_in.groupby('doc_id').agg(
    n_records=('meta_row', 'size'), n_sites=('site_B1', 'nunique'), n_pages=('page_number', 'nunique'),
    year_min=('year_start', 'min'), year_max=('year_end', 'max'),
    n_linked=('gw_nwis', lambda x: x.notna().sum())).sort_values('n_records', ascending=False)
print('\nrecords by document (top 12):')
print(docs.head(12).to_string())

keep = ['site_B1', 'lat', 'lon', 'first_year', 'decades', 'nwis_link']
sites[keep].to_csv(OUT / 'gw_basin_sites.csv', index=False)
nw[['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'first_year', 'decades_str', 'decades_source']].rename(columns={'decades_str': 'decades'}).to_csv(
    OUT / 'gw_basin_nwis_wells.csv', index=False)
lk[['site_B1', 'nwis_link', 'nwis_in_basin_list', 'first_year', 'decades', 'nwis_decades',
    'restored_only_decades']].to_csv(OUT / 'gw_basin_links.csv', index=False)
docs.to_csv(OUT / 'gw_basin_by_doc.csv')
print('\nsaved gw_basin_*.csv in', OUT)
