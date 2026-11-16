"""
Temporal and spatial scope extension: restored historical measurements vs.
what is digitally available from USGS NWIS, for California streamflow and
groundwater. Produces the abstract/manuscript statistics.

Two questions answered per water type:
  TEMPORAL  How many sites with data do the restored records add in early decades, compared with
            NWIS? (2026-11-15: replaces "how many years earlier does the restored record begin";
            NWIS actual years reach back to isolated flood peaks in 1854, so a single start-year
            comparison misleads. Same sites and NWIS years as Figure 3; per-decade table in
            data/analysis/scope_decade_comparison.csv.)
  SPATIAL   How many HUC12 subwatersheds ("basins") contain restored sites but
            NO NWIS site with a record beginning <= 1979 (same period as the restored data;
            same sites as Figure 3), and what % increase in monitored basins is that?

Method notes (state these in the manuscript methods):
  - Restored sites are clipped to the CA state boundary. The restored metadata
    includes out-of-state documents (e.g. Ohio/Mississippi River gauges); those
    must be excluded or they contaminate the "earliest CA record" figure. This
    is why the temporal extension is computed on CA-clipped data, matching the
    spatial analysis.
  - "Basin" = HUC12 subwatershed (4,473 statewide). HUC8 is too coarse: NWIS
    already touches nearly every HUC8, so the ungauged-basin signal only appears
    at HUC12. Both restored and NWIS points are assigned to HUC12 by spatial
    join against the same polygon layer (data/analysis/spatial/ca_wbd/).
  - NWIS streamflow uses the EXPANDED site set (daily-value + peak-flow + field
    measurements). Restricting to daily-value gauges only would lower NWIS basin
    coverage and raise the restored spatial gain. Toggle USE_EXPANDED_STREAM.
  - Only DATED restored records are used (date_parsing.DATED_STATUSES; protocol
    agreed 2026-10-28), so spatial and temporal numbers describe the same sites.
  - LOCATIONS (rebuilt 2026-10-28): lat_combined/lon_combined come from the validated
    location hierarchy, not the LLM. All statistics use PRECISE + APPROXIMATE locations
    (Annette, 2026-10-28). Approximate locations (median 3-8 km) can fall in a neighbouring
    HUC12; set PRECISION_CLASSES = {'precise'} for the conservative version.
  - Temporal extension is earliest-restored vs earliest-NWIS. Both extensions are
    backed by 150-270 CA sites predating NWIS, not single records.

WBD HUC layers were downloaded 2026-07-23 from the USGS Watershed Boundary
Dataset MapServer (hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer,
layer 4 = HUC8, layer 6 = HUC12), filtered to states LIKE '%CA%', simplified
with maxAllowableOffset, and saved to data/analysis/spatial/ca_wbd/.

Inputs:
  data/analysis/processed_metadata.parquet
  data/analysis/nwis_sites/ca_stream_sites_expanded.csv  (or ca_stream_sites.csv)
  data/analysis/nwis_sites/ca_groundwater_sites.csv
  data/analysis/spatial/ca_shapefile/ca_poly.shp
  data/analysis/spatial/ca_wbd/ca_huc12.geojson

Output:
  data/analysis/scope_extension_stats.csv  (one row per water type)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from date_parsing import is_dated
from paper_filters import drop_tribal, apply_review, reviewed_keep
from site_activity import load_records, nwis_table, all_nwis_coords, restored_sites as _restored_sites

root         = Path(__file__).resolve().parents[2]
parquet_path = root / 'data/analysis/processed_metadata.parquet'
nwis_path    = root / 'data/analysis/nwis_sites'
spatial_path = root / 'data/analysis/spatial'
out_path     = root / 'data/analysis/scope_extension_stats.csv'
early_path   = root / 'data/analysis/scope_earliest_records_review.csv'  # records predating NWIS, for manual review

USE_EXPANDED_STREAM = True   # True: daily+peak+field NWIS stream sites; False: daily-value only
RECORD_END_ERA      = 1980   # historical era end (restored data ends 1980); used for % record-length
RECORD_END_PRESENT  = 2025
PRECISION_CLASSES   = {'precise', 'approximate'}   # {'precise'} = conservative version   # present-day record end; alternative % framing

# ------------------------------------------------------------------ load
df   = pd.read_parquet(parquet_path)
df   = drop_tribal(df)   # tribal pages withheld from paper statistics (paper_filters.py)
df   = apply_review(df)  # reviewer decisions: drop / retype / undate / fix year (manual_review/)
ca   = gpd.read_file(spatial_path / 'ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
huc  = gpd.read_file(spatial_path / 'ca_wbd/ca_huc12.geojson').to_crs('EPSG:4326')
n_huc_total = huc['huc12'].nunique()
print(f'Loaded {len(df):,} metadata rows | {n_huc_total:,} CA HUC12 subwatersheds')


def restored_ca(water_type):
    """Dated restored sites of a water type, clipped to the CA boundary."""
    s = df[(df['water_type_clean'] == water_type) & is_dated(df, include_fallbacks=False) & df['is_data_page'] &
           df['lat_combined'].notna() & df['lon_combined'].notna()].copy()
    g = gpd.GeoDataFrame(
        s, geometry=gpd.points_from_xy(s['lon_combined'], s['lat_combined']),
        crs='EPSG:4326')
    return gpd.sjoin(g, ca[['geometry']], how='inner', predicate='within').drop(columns='index_right')


def nwis_gdf(csv_name):
    """NWIS sites (one row per site) with coordinates."""
    n = pd.read_csv(nwis_path / csv_name, low_memory=False).drop_duplicates('site_no')
    n = n[n['dec_lat_va'].notna() & n['dec_long_va'].notna()]
    return gpd.GeoDataFrame(
        n, geometry=gpd.points_from_xy(n['dec_long_va'], n['dec_lat_va']),
        crs='EPSG:4326')


def huc12_set(points):
    """Set of HUC12 codes containing at least one of the given points."""
    j = gpd.sjoin(points, huc[['huc12', 'geometry']], how='left', predicate='within')
    return set(j['huc12'].dropna())


stream_csv = 'ca_stream_sites_expanded.csv' if USE_EXPANDED_STREAM else 'ca_stream_sites.csv'
rows = []

MIN_DOCS = 2  # earliest year must be confirmed by >= MIN_DOCS documents (within +-1 year)


def corroborated_earliest(r, water_type):
    """Earliest restored start year confirmed by >= MIN_DOCS different documents starting
    within one year of it, OR by a page a reviewer checked and kept (manual_review/,
    2026-11-05: GW 1883 Hungry Hollow borings, doc 1128 p27). A raw minimum over ~100k
    automatically parsed dates always picks up a stray artifact (2026-11-04 review: 1830 = a
    time-of-day value in a 1977 water-quality table, 1854 = a Tennessee page mislocated in
    CA, 1870 = a depth interval '1870-1930 FEET'). The raw minimum is kept as
    restored_earliest_yr_raw."""
    kept = reviewed_keep(water_type)
    yd = r[['year_start', 'id', 'page_number']].dropna().astype({'year_start': int}).drop_duplicates()
    yd['reviewed'] = [(str(i).removesuffix('.0'), int(p)) in kept for i, p in zip(yd.id, yd.page_number)]
    for y in sorted(yd.year_start.unique()):
        if yd[(yd.year_start - y).abs() <= 1].id.nunique() >= MIN_DOCS or \
                yd[(yd.year_start == y) & yd.reviewed].shape[0] > 0:
            return int(y)
    return int(yd.year_start.min())


def nwis_actual_years(kind):
    f = nwis_path / f'nwis_years_{kind}.csv'
    if not f.exists():
        return None
    y = pd.read_csv(f, dtype={'site_no': str})
    return y[y.year <= 1979]


fig3_rec = load_records()
decade_tables = []
early_review = []
for wt, csv_name in [('Stream Discharge', stream_csv),
                     ('Groundwater', 'ca_groundwater_sites.csv')]:
    r = restored_ca(wt)
    r_any = r
    r = r[r['precision_class'].isin(PRECISION_CLASSES)]
    n = nwis_gdf(csv_name)
    r_sites = r.drop_duplicates(['lat_combined', 'lon_combined'])

    # --- temporal ---
    ry = r['year_start'].dropna().astype(int)
    r_early_raw = int(ry.min())
    r_early = corroborated_earliest(r, wt)
    n_begin = int(n['begin_year'].dropna().min())       # series-catalogue begin date (old basis)
    early_review.append(r[r['year_start'] < n_begin].assign(water_type=wt))
    # decade comparison on the Figure 3 basis (2026-11-15): same restored sites (site_B1) and NWIS
    # ACTUAL years with data. The NWIS series catalogue starts stream records in 1891, but the
    # actual years include isolated historic flood peaks back to 1854 (Sacramento R nr Hamilton City),
    # so "restored 1862 vs NWIS 1891 (+29 yr)" no longer holds; report sites active per period instead.
    nt = nwis_table(csv_name)
    st = _restored_sites(fig3_rec, wt, all_nwis_coords(csv_name), 'site_B1')
    act_years = nwis_actual_years('stream' if 'stream' in csv_name else 'gw')
    n_early_actual = int(act_years.year.min()) if act_years is not None else None
    dec_rows = []
    for d in range(1850, 1980, 10):
        rs = st[st.dec.map(lambda v: d in v)]
        dec_rows.append({'water_type': wt, 'decade': d, 'nwis_active': sum(d in v for v in nt.decades),
                         'restored_active': len(rs), 'restored_active_no_nwis_link': int(rs.nwis_link.isna().sum())})
    decade_tables.append(pd.DataFrame(dec_rows))
    before = {}
    for cut in (1900, 1910):
        rs = st[st.dec.map(lambda v: any(x < cut for x in v))]
        before[cut] = (sum(any(x < cut for x in v) for v in nt.decades), len(rs), int(rs.nwis_link.isna().sum()))

    # --- spatial (HUC12 basins), Figure 3 basis (Annette, 2026-11-15): the restored data end in 1979,
    # so NWIS is compared over the same period: NWIS sites with a record beginning <= 1979
    # (site_activity.nwis_table) and the Figure 3 restored sites (site_B1). The earlier version
    # counted NWIS sites of ANY start year (through today) and was an error (+396 vs +476 stream).
    r_b = huc12_set(gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs='EPSG:4326'))
    n_b = huc12_set(gpd.GeoDataFrame(nt, geometry=gpd.points_from_xy(nt.dec_long_va, nt.dec_lat_va),
                                     crs='EPSG:4326'))
    ungauged = r_b - n_b
    pct_basin_increase = 100 * len(ungauged) / len(n_b)

    rows.append({
        'water_type': wt,
        'nwis_source': csv_name,
        # temporal
        'restored_earliest_yr': r_early,
        'restored_earliest_yr_raw': r_early_raw,
        'nwis_earliest_series_begin_yr': n_begin,
        'nwis_earliest_actual_yr': n_early_actual,
        'nwis_decades_source': ';'.join(f'{k}={v}' for k, v in nt.decades_source.value_counts().items()),
        'nwis_sites_active_pre1900': before[1900][0],
        'restored_sites_active_pre1900': before[1900][1],
        'restored_sites_active_pre1900_no_nwis_link': before[1900][2],
        'nwis_sites_active_pre1910': before[1910][0],
        'restored_sites_active_pre1910': before[1910][1],
        'restored_sites_active_pre1910_no_nwis_link': before[1910][2],
        # spatial
        'restored_sites_ca': len(st),
        'nwis_sites_le1979': len(nt),
        'restored_basins_huc12': len(r_b),
        'nwis_basins_huc12': len(n_b),
        'previously_ungauged_basins': len(ungauged),
        'spatial_pct_basin_increase': round(pct_basin_increase, 1),
        'restored_pct_of_ca_basins': round(100 * len(r_b) / n_huc_total, 1),
        'nwis_pct_of_ca_basins': round(100 * len(n_b) / n_huc_total, 1),
    })

    print(f'\n=== {wt} ===')
    print(f'  TEMPORAL: restored earliest {r_early}; NWIS earliest actual year {n_early_actual} '
          f'(series catalogue {n_begin}); NWIS decades from {dict(nt.decades_source.value_counts())}')
    for cut, (a_, b_, c_) in before.items():
        print(f'    sites active before {cut}: NWIS {a_} | restored {b_} ({c_} with no NWIS link)')
    print(f'  SPATIAL:  restored {len(r_b)} basins vs NWIS {len(n_b)}; '
          f'+{len(ungauged)} previously-ungauged basins (+{pct_basin_increase:.0f}%)')

rev = pd.concat(early_review).sort_values(['water_type', 'year_start'])
rev_cols = ['water_type', 'id', 'page_number', 'meta_row', 'year_start', 'year_end', 'date_status',
            'year_start_llm', 'watersource_name', 'coord_method', 'precision_class']
try:
    rev[[c for c in rev_cols if c in rev.columns]].to_csv(early_path, index=False)
except PermissionError:  # file open in Excel: write a copy instead of failing
    early_path = early_path.with_name(early_path.stem + '_new.csv')
    rev[[c for c in rev_cols if c in rev.columns]].to_csv(early_path, index=False)
print(f'Records predating NWIS (for manual review): {len(rev):,} -> {early_path.name}')

out = pd.DataFrame(rows)
out.to_csv(out_path, index=False)
pd.concat(decade_tables).to_csv(root / 'data/analysis/scope_decade_comparison.csv', index=False)
print('Saved: data/analysis/scope_decade_comparison.csv (sites active per decade, Figure 3 basis)')
print(f'\nSaved: {out_path}')
