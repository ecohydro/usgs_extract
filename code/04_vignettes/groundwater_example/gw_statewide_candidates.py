"""
Figure 5 (groundwater example), statewide search, 2026-11-16 (Annette: not limited to Santa Ynez; show an area
where restored data add wells or fill gaps, then one long record / one well).

Restored records = Figure 3 set (site_activity.load_records: California, precise/approximate, measurement
dates, measurement pages, tribal out, reviewer decisions), groundwater only.
Years = years PRINTED in the rows of the record's tables only (lesson from Santa Ynez, 2026-11-16: ranges
from page text and period-of-record lists were not measurement years). Records without printed years are
left out of the year counts.
NWIS = groundwater sites with a water-level record beginning <= 1979 (site_activity.nwis_table), years
with data from nwis_years_gw.csv. Restored well linked to NWIS by state well number only (as Figure 3).

Outputs (data/analysis/gw_example/):
  gw_sites_statewide.csv   one row per restored GW site: B118 basin, printed years, NWIS link + years,
                           restored-only years, longest run of consecutive printed years
  gw_basins_statewide.csv  one row per B118 basin (sub)basin: restored sites, restored site-years,
                           restored-only site-years (no NWIS link, or years the linked well lacks),
                           NWIS wells <= 1979 and NWIS well-years before 1950
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
from site_activity import load_records, nwis_table, CAP  # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'gw_example'
OUT.mkdir(exist_ok=True)

rec = load_records()
rec = rec[rec.water_type == 'Groundwater'].copy()


def printed(r):
    if not isinstance(r.printed_years, set):
        return set()
    return {y for y in r.printed_years if r.year_start <= y <= min(r.year_end, CAP)}


rec['yrs'] = [printed(r) for r in rec.itertuples(index=False)]
rec['src'] = rec.doc_id.astype(str) + ' p' + rec.page_number.astype(int).astype(str)
s = rec.groupby('site_B1').agg(lat=('lat_combined', 'mean'), lon=('lon_combined', 'mean'),
                               yrs=('yrs', lambda x: set().union(*x)),
                               nwis=('gw_nwis', lambda x: x.dropna().iloc[0] if x.notna().any() else None),
                               docs=('doc_id', lambda x: ' '.join(sorted({str(v) for v in x}))),
                               pages=('src', lambda x: ' '.join(sorted(set(x)))),
                               n_records=('meta_row', 'size'))
s['nwis'] = s.nwis.astype('string').str.split(';').str[0].str.replace(r'\.0$', '', regex=True)

ny = pd.read_csv(A / 'nwis_sites/nwis_years_gw.csv', dtype={'site_no': str})
ny = ny[ny.year <= CAP]
nyrs = ny.groupby('site_no').year.apply(set)
nmeas = ny.groupby('site_no').n_measurements.sum()


def run(ys):
    best = cur = 0
    prev = None
    for y in sorted(ys):
        cur = cur + 1 if prev is not None and y == prev + 1 else 1
        best, prev = max(best, cur), y
    return best


s['nwis_yrs'] = [nyrs.get(n, set()) if isinstance(n, str) else set() for n in s.nwis]
s['only'] = [y - n for y, n in zip(s.yrs, s.nwis_yrs)]
s['n_yrs'] = s.yrs.map(len)
s['n_only'] = s.only.map(len)
s['longest_run'] = s.yrs.map(run)
s['nwis_meas'] = [int(nmeas.get(n, 0)) if isinstance(n, str) else 0 for n in s.nwis]

# B118 (sub)basins
b = gpd.read_file(A / 'spatial/sgma/sgma_2019_basin_prioritization.geojson')
b['geometry'] = b.geometry.make_valid().buffer(0)
b = b.to_crs('EPSG:4326')[['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority', 'geometry']]
g = gpd.GeoDataFrame(s.reset_index(), geometry=gpd.points_from_xy(s.lon, s.lat), crs='EPSG:4326')
g = gpd.sjoin(g, b, how='left', predicate='within').drop_duplicates('site_B1')

nw = nwis_table('ca_groundwater_sites.csv').reset_index()
nwg = gpd.GeoDataFrame(nw, geometry=gpd.points_from_xy(nw.dec_long_va, nw.dec_lat_va), crs='EPSG:4326')
nwg = gpd.sjoin(nwg, b, how='inner', predicate='within').drop_duplicates('site_no')
nwg['pre1950_yrs'] = [len({y for y in nyrs.get(x, set()) if y < 1950}) for x in nwg.site_no]
nb = nwg.groupby('Basin_Subbasin_Number').agg(nwis_wells=('site_no', 'size'),
                                              nwis_well_years_pre1950=('pre1950_yrs', 'sum'))

g['fmt'] = lambda: None
out = g.drop(columns=['geometry', 'index_right', 'fmt'])
for c in ('yrs', 'nwis_yrs', 'only'):
    out[c] = out[c].map(lambda v: ' '.join(map(str, sorted(v))))
out.sort_values('n_only', ascending=False).to_csv(OUT / 'gw_sites_statewide.csv', index=False)

g['pre1950_only'] = g.only.map(lambda v: len({y for y in v if y < 1950}))
bb = g.groupby(['Basin_Subbasin_Number', 'Basin_Subbasin_Name', 'Priority']).agg(
    restored_sites=('site_B1', 'size'), sites_with_printed_years=('n_yrs', lambda x: (x > 0).sum()),
    restored_site_years=('n_yrs', 'sum'), restored_only_site_years=('n_only', 'sum'),
    restored_only_site_years_pre1950=('pre1950_only', 'sum'),
    sites_no_nwis_link=('nwis', lambda x: x.isna().sum()),
    sites_run_ge10=('longest_run', lambda x: (x >= 10).sum()),
    docs=('docs', lambda x: len({d for v in x for d in v.split()}))).reset_index()
bb = bb.merge(nb, left_on='Basin_Subbasin_Number', right_index=True, how='left').fillna(
    {'nwis_wells': 0, 'nwis_well_years_pre1950': 0})
bb = bb.sort_values('restored_only_site_years', ascending=False)
bb.to_csv(OUT / 'gw_basins_statewide.csv', index=False)

pd.set_option('display.width', 250)
print(f'restored GW sites {len(g):,}; with printed years {(g.n_yrs > 0).sum():,}; '
      f'in a B118 basin {g.Basin_Subbasin_Number.notna().sum():,}')
print(bb.head(20).to_string(index=False))
print('\nsites: longest printed runs / most restored-only years')
cols = ['site_B1', 'Basin_Subbasin_Name', 'nwis', 'n_yrs', 'longest_run', 'n_only', 'nwis_meas', 'docs']
print(out.sort_values(['longest_run', 'n_only'], ascending=False)[cols].head(20).to_string(index=False))
