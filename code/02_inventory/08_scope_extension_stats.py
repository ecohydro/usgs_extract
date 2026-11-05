"""
Temporal and spatial scope extension: restored historical measurements vs.
what is digitally available from USGS NWIS, for California streamflow and
groundwater. Produces the abstract/manuscript statistics.

Two questions answered per water type:
  TEMPORAL  How many years earlier does the restored record begin than the
            earliest digitally-available (NWIS) record?
  SPATIAL   How many HUC12 subwatersheds ("basins") contain restored sites but
            NO NWIS site (previously ungauged basins), and what % increase in
            monitored basins does that represent?

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
    n_early = int(n['begin_year'].dropna().min())
    early_review.append(r[r['year_start'] < n_early].assign(water_type=wt))
    extension_yrs = n_early - r_early
    n_predate_locs = r[r['year_start'] < n_early].drop_duplicates(
        ['lat_combined', 'lon_combined']).shape[0]
    pct_era     = 100 * extension_yrs / (RECORD_END_ERA - n_early)
    pct_present = 100 * extension_yrs / (RECORD_END_PRESENT - n_early)

    # --- spatial (HUC12 basins) ---
    r_b = huc12_set(r_sites)
    n_b = huc12_set(n)
    ungauged = r_b - n_b
    pct_basin_increase = 100 * len(ungauged) / len(n_b)

    rows.append({
        'water_type': wt,
        'nwis_source': csv_name,
        # temporal
        'restored_earliest_yr': r_early,
        'restored_earliest_yr_raw': r_early_raw,
        'nwis_earliest_yr': n_early,
        'temporal_extension_yrs': extension_yrs,
        'restored_locs_predating_nwis': n_predate_locs,
        'temporal_pct_vs_1980era': round(pct_era, 1),
        'temporal_pct_vs_present': round(pct_present, 1),
        # spatial
        'restored_sites_ca': len(r_sites),
        'restored_sites_ca_any_precision': r_any.drop_duplicates(['lat_combined', 'lon_combined']).shape[0],
        'nwis_sites': len(n),
        'restored_basins_huc12': len(r_b),
        'nwis_basins_huc12': len(n_b),
        'previously_ungauged_basins': len(ungauged),
        'spatial_pct_basin_increase': round(pct_basin_increase, 1),
        'restored_pct_of_ca_basins': round(100 * len(r_b) / n_huc_total, 1),
        'nwis_pct_of_ca_basins': round(100 * len(n_b) / n_huc_total, 1),
    })

    print(f'\n=== {wt} ===')
    print(f'  TEMPORAL: restored {r_early} vs NWIS {n_early} -> +{extension_yrs} yrs '
          f'(+{pct_era:.0f}% vs 1980 era, +{pct_present:.0f}% vs present); '
          f'{n_predate_locs} CA sites predate NWIS')
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
print(f'\nSaved: {out_path}')
