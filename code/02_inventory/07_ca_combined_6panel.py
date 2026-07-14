"""
Draft 6-panel figure: CA statewide restored vs. NWIS sites,
stream discharge (left column) and groundwater (right column).

Row 1: new sites per decade (time series, like dissertation Figs 13/17 Panel A,
       but statewide instead of Santa Barbara County)
Row 2: CA map of restored vs. NWIS site locations (like notebooks 05/06)
Row 3: placeholder (TBD)

Inputs:
- data/analysis/processed_metadata.parquet
- data/analysis/nwis_sites/ca_stream_sites_expanded.csv  (or ca_stream_sites.csv)
- data/analysis/nwis_sites/ca_groundwater_sites.csv
- data/analysis/spatial/ca_shapefile/ca_poly.shp

Output: manuscript/figures/ca_nwis_combined_6panel.png
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

root         = Path(__file__).resolve().parents[2]
parquet_path = root / 'data/analysis/processed_metadata.parquet'
spatial_path = root / 'data/analysis/spatial'
nwis_path    = root / 'data/analysis/nwis_sites'
figures_path = root / 'manuscript/figures'
figures_path.mkdir(parents=True, exist_ok=True)

NWIS_COLOR     = '#4477AA'
RESTORED_COLOR = '#EE6677'

# True: NWIS stream sites include peak-flow-only and discrete measurement sites
# False: daily-value gauges only (ca_stream_sites.csv)
USE_EXPANDED_STREAM = True

NWIS_YEAR_CUTOFF = 1979   # match restored data period (records end 1980)
# last full decade is the 1970s; the 1980 bin only catches records starting
# exactly in 1980 (n=2) and plots as a misleading drop to zero
DECADE_MIN, DECADE_MAX = 1850, 1970

# ---------------------------------------------------------------- load data
df = pd.read_parquet(parquet_path)
print(f'Loaded {len(df):,} metadata rows')

ca = gpd.read_file(spatial_path / 'ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
xmin, ymin, xmax, ymax = ca.total_bounds
buf = 0.3
CA_XMIN, CA_XMAX = xmin - buf, xmax + buf
CA_YMIN, CA_YMAX = ymin - buf, ymax + buf


def restored_sites(water_type):
    """Unique restored locations for a water type, clipped to CA.
    Returns (all unique sites gdf, unique site x decade gdf)."""
    base = df[
        (df['water_type_clean'] == water_type) &
        df['lat_combined'].notna() &
        df['lon_combined'].notna()
    ]
    sites = base.drop_duplicates(subset=['lat_combined', 'lon_combined'])
    g = gpd.GeoDataFrame(
        sites, geometry=gpd.points_from_xy(sites['lon_combined'], sites['lat_combined']),
        crs='EPSG:4326')
    g_ca = gpd.sjoin(g, ca[['geometry']], how='inner', predicate='within')

    by_dec = base[base['decade_start'].notna()].drop_duplicates(
        subset=['lat_combined', 'lon_combined', 'decade_start'])
    gd = gpd.GeoDataFrame(
        by_dec, geometry=gpd.points_from_xy(by_dec['lon_combined'], by_dec['lat_combined']),
        crs='EPSG:4326')
    gd_ca = gpd.sjoin(gd, ca[['geometry']], how='inner', predicate='within')
    return g_ca, gd_ca


def nwis_sites(csv_name):
    """NWIS sites with begin_year <= cutoff, one row per site."""
    nwis = pd.read_csv(nwis_path / csv_name, low_memory=False)
    hist = nwis[
        (nwis['begin_year'] <= NWIS_YEAR_CUTOFF) &
        nwis['dec_lat_va'].notna() &
        nwis['dec_long_va'].notna()
    ].drop_duplicates(subset=['site_no']).copy()
    hist['decade_nwis'] = (hist['begin_year'] // 10 * 10).astype(int)
    g = gpd.GeoDataFrame(
        hist, geometry=gpd.points_from_xy(hist['dec_long_va'], hist['dec_lat_va']),
        crs='EPSG:4326')
    return g


sd_all, sd_dec = restored_sites('Stream Discharge')
gw_all, gw_dec = restored_sites('Groundwater')
print(f'Restored stream discharge sites in CA: {len(sd_all):,}')
print(f'Restored groundwater sites in CA:      {len(gw_all):,}')

stream_csv = 'ca_stream_sites_expanded.csv' if USE_EXPANDED_STREAM else 'ca_stream_sites.csv'
nwis_sd = nwis_sites(stream_csv)
nwis_gw = nwis_sites('ca_groundwater_sites.csv')
print(f'NWIS stream sites (begin <= {NWIS_YEAR_CUTOFF}): {len(nwis_sd):,}  [{stream_csv}]')
print(f'NWIS groundwater sites (begin <= {NWIS_YEAR_CUTOFF}): {len(nwis_gw):,}')


def decade_counts(restored_dec_gdf, nwis_gdf):
    """New sites per decade for restored and NWIS, on a common decade index."""
    r = restored_dec_gdf.groupby('decade_start').size()
    n = nwis_gdf.groupby('decade_nwis').size()
    decades = [d for d in range(DECADE_MIN, DECADE_MAX + 1, 10)]
    out = pd.DataFrame(index=decades)
    out['restored'] = r.reindex(decades).fillna(0).astype(int)
    out['nwis']     = n.reindex(decades).fillna(0).astype(int)
    return out


ts_sd = decade_counts(sd_dec, nwis_sd)
ts_gw = decade_counts(gw_dec, nwis_gw)
print('\nStream discharge per decade:\n', ts_sd)
print('\nGroundwater per decade:\n', ts_gw)

# ---------------------------------------------------------------- figure
map_aspect = (CA_YMAX - CA_YMIN) / (CA_XMAX - CA_XMIN)   # ~1.06

fig = plt.figure(figsize=(11, 14), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 1.0, 0.62], hspace=0.22, wspace=0.25)

ax_ts_sd  = fig.add_subplot(gs[0, 0])
ax_ts_gw  = fig.add_subplot(gs[0, 1])
ax_map_sd = fig.add_subplot(gs[1, 0])
ax_map_gw = fig.add_subplot(gs[1, 1])
ax_e      = fig.add_subplot(gs[2, 0])
ax_f      = fig.add_subplot(gs[2, 1])

# ---- Row 1: time series (new sites per decade)
for ax, ts, label in [(ax_ts_sd, ts_sd, 'stream discharge'),
                      (ax_ts_gw, ts_gw, 'groundwater')]:
    ax.plot(ts.index, ts['nwis'], '-o', color=NWIS_COLOR, markersize=4,
            label='NWIS')
    ax.plot(ts.index, ts['restored'], '-o', color=RESTORED_COLOR, markersize=4,
            label='Restored')
    ax.set_xlabel('Decade')
    ax.set_ylabel('New sites per decade')
    ax.set_xlim(DECADE_MIN - 5, DECADE_MAX + 5)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

ax_ts_sd.set_title('Stream Discharge', fontsize=13, pad=10)
ax_ts_gw.set_title('Groundwater', fontsize=13, pad=10)

# ---- Row 2: CA maps
def draw_map(ax, restored_gdf, nwis_gdf, restored_on_top):
    ca.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.6, zorder=1)
    layers = [
        (nwis_gdf, NWIS_COLOR, f'NWIS (n={len(nwis_gdf):,})'),
        (restored_gdf, RESTORED_COLOR, f'Restored (n={len(restored_gdf):,})'),
    ]
    if not restored_on_top:
        layers.reverse()
    # bottom layer smaller/fainter so the top layer stays readable
    layers[0][0].plot(ax=ax, color=layers[0][1], markersize=3, alpha=0.4,
                      edgecolor='none', zorder=2, label=layers[0][2])
    layers[1][0].plot(ax=ax, color=layers[1][1], markersize=4, alpha=0.6,
                      edgecolor='none', zorder=3, label=layers[1][2])
    ax.set_xlim(CA_XMIN, CA_XMAX)
    ax.set_ylim(CA_YMIN, CA_YMAX)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)
    ax.legend(loc='lower left', markerscale=3.0, frameon=True, fancybox=True,
              framealpha=0.9, fontsize=9)

# stream: NWIS sparse -> plot on top; groundwater: NWIS dense -> restored on top
draw_map(ax_map_sd, sd_all, nwis_sd, restored_on_top=False)
draw_map(ax_map_gw, gw_all, nwis_gw, restored_on_top=True)

# ---- Row 3: placeholders
for ax, lab in [(ax_e, 'E'), (ax_f, 'F')]:
    ax.text(0.5, 0.5, f'Panel {lab} — TBD', ha='center', va='center',
            fontsize=14, color='#999999', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#dddddd')

# panel letters
for ax, lab in [(ax_ts_sd, 'A'), (ax_ts_gw, 'B'), (ax_map_sd, 'C'),
                (ax_map_gw, 'D'), (ax_e, 'E'), (ax_f, 'F')]:
    ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

out_path = figures_path / 'ca_nwis_combined_6panel.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'\nSaved: {out_path}')
