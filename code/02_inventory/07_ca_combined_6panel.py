"""
Draft 6-panel figure: CA statewide restored vs. NWIS sites,
stream discharge (top row, A-C) and groundwater (bottom row, D-F).
Sized for a 2-column journal figure (7.25 in wide).

Col 1: new sites per decade (time series, like dissertation Figs 13/17 Panel A,
       but statewide instead of Santa Barbara County)
Col 2: CA map of restored vs. NWIS site locations (like notebooks 05/06)
Col 3: HUC12 subwatersheds with NWIS sites vs. restored-only (previously
       ungauged) subwatersheds; basins with both count as NWIS, basins with
       no sites are not drawn (see BASIN_NWIS_ALL_YEARS)

Inputs:
- data/analysis/processed_metadata.parquet
- data/analysis/nwis_sites/ca_stream_sites_expanded.csv  (or ca_stream_sites.csv)
- data/analysis/nwis_sites/ca_groundwater_sites.csv
- data/analysis/spatial/ca_shapefile/ca_poly.shp
- data/analysis/spatial/ca_wbd/ca_huc12.geojson

Output: manuscript/figures/ca_nwis_combined_6panel.png
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from date_parsing import is_dated

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

# Row 3 basin classification. True: a basin counts as NWIS-gauged if it has ANY
# NWIS site, regardless of start year. This matches 08_scope_extension_stats.py
# and the abstract "previously ungauged basins" numbers (conservative claim).
# False: only NWIS sites beginning <= NWIS_YEAR_CUTOFF, matching the Row 2 maps.
BASIN_NWIS_ALL_YEARS = True

# Restored-site locations used (rebuilt hierarchy, 2026-10-28). Annette: precise + approximate.
# {'precise'} = conservative version (~1 km; approximate is 3-8 km median error).
PRECISION_CLASSES = {'precise', 'approximate'}

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
    """Unique DATED, LOCATED restored sites for a water type, clipped to CA.
    Undated records are excluded from every panel (date protocol, 2026-10-28); locations
    come from the rebuilt hierarchy, precise + approximate (PRECISION_CLASSES).
    Returns (all unique sites gdf, unique site x decade gdf)."""
    base = df[
        (df['water_type_clean'] == water_type) & is_dated(df) &
        df['precision_class'].isin(PRECISION_CLASSES) & df['is_data_page'] &
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

# ---------------------------------------------------------------- HUC12 basins
huc = gpd.read_file(spatial_path / 'ca_wbd/ca_huc12.geojson').to_crs('EPSG:4326')

# Basins with both restored and NWIS sites are labeled NWIS, so the restored
# class shows only where restored data fills a basin with zero NWIS sites.
# Basins with no sites are not drawn (no legend entry).
BASIN_CLASSES = [   # (key, legend label, color) -- drawn and listed in this order
    ('nwis',     'NWIS',                    NWIS_COLOR),
    ('restored', 'Restored, no NWIS sites', RESTORED_COLOR),
]


def huc12_set(points):
    """Set of HUC12 codes containing at least one of the given points."""
    j = gpd.sjoin(points, huc[['huc12', 'geometry']], how='inner', predicate='within')
    return set(j['huc12'])


def all_nwis(csv_name):
    """All NWIS sites with coordinates, any start year (one row per site)."""
    n = pd.read_csv(nwis_path / csv_name, low_memory=False).drop_duplicates('site_no')
    n = n[n['dec_lat_va'].notna() & n['dec_long_va'].notna()]
    return gpd.GeoDataFrame(
        n, geometry=gpd.points_from_xy(n['dec_long_va'], n['dec_lat_va']),
        crs='EPSG:4326')


def classify_basins(restored_gdf, nwis_gdf):
    """HUC12 polygons labeled restored-only / NWIS (incl. both) / none."""
    r = huc12_set(restored_gdf[['geometry']])
    n = huc12_set(nwis_gdf[['geometry']])
    b = huc.copy()
    b['cls'] = 'none'
    b.loc[b['huc12'].isin(n), 'cls'] = 'nwis'
    b.loc[b['huc12'].isin(r), 'cls'] = 'restored'
    b.loc[b['huc12'].isin(n), 'cls'] = 'nwis'   # NWIS takes precedence over restored
    return b


basin_nwis_sd = all_nwis(stream_csv) if BASIN_NWIS_ALL_YEARS else nwis_sd
basin_nwis_gw = all_nwis('ca_groundwater_sites.csv') if BASIN_NWIS_ALL_YEARS else nwis_gw
basins_sd = classify_basins(sd_all, basin_nwis_sd)
basins_gw = classify_basins(gw_all, basin_nwis_gw)
for lab, b in [('Stream discharge', basins_sd), ('Groundwater', basins_gw)]:
    print(f'\n{lab} HUC12 basins:\n', b['cls'].value_counts())

# ---------------------------------------------------------------- figure
# Built at final print size: journal 2-column width = 7.25 in, max depth 7.8 in.
# Layout: rows = water type (stream, groundwater); columns = time series,
# site map, HUC12 basin coverage map. Myriad (journal caption font) is not
# installed, so Arial is used as the closest available sans-serif.
map_aspect = (CA_YMAX - CA_YMIN) / (CA_XMAX - CA_XMIN)
FIG_W, FIG_H = 7.25, 5.2
FS_BASE   = 7     # axis labels
FS_SMALL  = 6     # tick labels, legends
FS_TITLE  = 8     # column titles, row labels
FS_LETTER = 9     # panel letters

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': FS_BASE,
    'axes.labelsize': FS_BASE,
    'xtick.labelsize': FS_SMALL,
    'ytick.labelsize': FS_SMALL,
    'legend.fontsize': FS_SMALL,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
})

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0],
                      left=0.10, right=0.995, top=0.947, bottom=0.105,
                      wspace=0.12, hspace=0.26)

ax_ts_sd  = fig.add_subplot(gs[0, 0])
ax_map_sd = fig.add_subplot(gs[0, 1])
ax_bas_sd = fig.add_subplot(gs[0, 2])
ax_ts_gw  = fig.add_subplot(gs[1, 0])
ax_map_gw = fig.add_subplot(gs[1, 1])
ax_bas_gw = fig.add_subplot(gs[1, 2])

# map legends sit below the map box, where an x-axis would be
BELOW_KW = dict(loc='upper center', bbox_to_anchor=(0.5, -0.01), frameon=False,
                borderpad=0.2, handlelength=1.2, handletextpad=0.5, labelspacing=0.3)

LEGEND_KW = dict(frameon=True, fancybox=False, framealpha=0.9, edgecolor='#cccccc',
                 borderpad=0.4, handlelength=1.2, handletextpad=0.5, labelspacing=0.3)

# ---- Column 1: time series (new sites per decade)
for ax, ts in [(ax_ts_sd, ts_sd), (ax_ts_gw, ts_gw)]:
    ax.plot(ts.index, ts['nwis'], '-o', color=NWIS_COLOR, markersize=2.5,
            linewidth=1.0, label='NWIS')
    ax.plot(ts.index, ts['restored'], '-o', color=RESTORED_COLOR, markersize=2.5,
            linewidth=1.0, label='Restored')
    ax.set_ylabel('New sites per decade', labelpad=2)
    ax.set_xlim(DECADE_MIN - 5, DECADE_MAX + 5)
    # each point is a decade bin plotted at its start year; the last bin (1970)
    # covers 1970-1979, so the axis ends at 1970 rather than 1980
    ax.set_xticks(range(DECADE_MIN, DECADE_MAX + 1, 20))
    ax.legend(loc='upper left', **LEGEND_KW)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_box_aspect(map_aspect)   # same height as the maps so rows align

# ---- Column 2: CA site maps
def style_map(ax):
    ax.set_xlim(CA_XMIN, CA_XMAX)
    ax.set_ylim(CA_YMIN, CA_YMAX)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.6)


def draw_map(ax, restored_gdf, nwis_gdf, restored_on_top):
    ca.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.4, zorder=1)
    layers = [
        (nwis_gdf, NWIS_COLOR, f'NWIS (n={len(nwis_gdf):,})'),
        (restored_gdf, RESTORED_COLOR, f'Restored (n={len(restored_gdf):,})'),
    ]
    if not restored_on_top:
        layers.reverse()
    # bottom layer smaller/fainter so the top layer stays readable
    layers[0][0].plot(ax=ax, color=layers[0][1], markersize=0.6, alpha=0.4,
                      edgecolor='none', zorder=2, label=layers[0][2])
    layers[1][0].plot(ax=ax, color=layers[1][1], markersize=0.9, alpha=0.6,
                      edgecolor='none', zorder=3, label=layers[1][2])
    style_map(ax)
    ax.legend(markerscale=4.0, **BELOW_KW)

# stream: NWIS sparse -> plot on top; groundwater: NWIS dense -> restored on top
draw_map(ax_map_sd, sd_all, nwis_sd, restored_on_top=False)
draw_map(ax_map_gw, gw_all, nwis_gw, restored_on_top=True)

# ---- Column 3: HUC12 basin coverage maps
from matplotlib.patches import Patch


from matplotlib.lines import Line2D

# HUC8 / HUC12 boundary lines (requested 2026-10-28 for the groundwater map, panel F)
SHOW_HUC_LINES = {'gw'}          # add 'sd' to draw them on the stream panel (C) as well
HUC12_LINE = dict(color='#9a9a9a', linewidth=0.08)
HUC8_LINE  = dict(color='#333333', linewidth=0.35)
huc8 = gpd.read_file(spatial_path / 'ca_wbd/ca_huc8.geojson').to_crs('EPSG:4326')
_ca_valid = ca.set_geometry(ca.geometry.make_valid())
huc8_ca = gpd.clip(huc8.set_geometry(huc8.geometry.make_valid()), _ca_valid)
huc12_ca = gpd.clip(huc.set_geometry(huc.geometry.make_valid()), _ca_valid)


def draw_basins(ax, basins, huc_lines=False):
    counts = basins['cls'].value_counts()
    # clip for display only: some WBD HUC12s straddle the state line
    # (make_valid: the simplified WBD download has a few self-intersections)
    basins = gpd.clip(basins.set_geometry(basins.geometry.make_valid()), _ca_valid)
    handles = []
    for key, label, color in BASIN_CLASSES:
        sub = basins[basins['cls'] == key]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.05, zorder=2)
        handles.append(Patch(facecolor=color, edgecolor='none',
                             label=f'{label} ({counts.get(key, 0):,})'))
    if huc_lines:
        huc12_ca.boundary.plot(ax=ax, zorder=3, **HUC12_LINE)
        huc8_ca.boundary.plot(ax=ax, zorder=4, **HUC8_LINE)
        handles += [Line2D([], [], label='HUC8 boundary', **{**HUC8_LINE, 'linewidth': 0.8}),
                    Line2D([], [], label='HUC12 boundary', **{**HUC12_LINE, 'linewidth': 0.6})]
    ca.boundary.plot(ax=ax, color='black', linewidth=0.4, zorder=5)
    style_map(ax)
    ax.legend(handles=handles, **BELOW_KW)


draw_basins(ax_bas_sd, basins_sd, huc_lines='sd' in SHOW_HUC_LINES)
draw_basins(ax_bas_gw, basins_gw, huc_lines='gw' in SHOW_HUC_LINES)

# ---- column titles and row labels
for ax, title in [(ax_ts_sd, 'Sites Per Decade'),
                  (ax_map_sd, 'Site locations (1850–1980)'),
                  (ax_bas_sd, 'Subwatershed coverage')]:
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=6)

for ax, label in [(ax_ts_sd, 'Stream discharge'), (ax_ts_gw, 'Groundwater')]:
    pos = ax.get_position()
    fig.text(0.012, (pos.y0 + pos.y1) / 2, label, rotation=90, ha='left',
             va='center', fontsize=FS_TITLE, fontweight='bold')

# panel letters: outside the top-left corner so they don't sit on the CA outline
for ax, lab in [(ax_ts_sd, 'A'), (ax_map_sd, 'B'), (ax_bas_sd, 'C'),
                (ax_ts_gw, 'D'), (ax_map_gw, 'E'), (ax_bas_gw, 'F')]:
    pos = ax.get_position()
    fig.text(pos.x0 - 0.008, pos.y1 + 0.006, lab, fontsize=FS_LETTER,
             fontweight='bold', va='bottom', ha='right')

out_path = figures_path / 'ca_nwis_combined_6panel.png'
# no bbox_inches='tight': keep the exact journal width
plt.savefig(out_path, dpi=300)
print(f'\nSaved: {out_path}  ({FIG_W} x {FIG_H} in)')
