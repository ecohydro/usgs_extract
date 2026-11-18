"""
Figure 2: locations of restored California data, one map panel per water category.

Same table set as Figure 1 (01_measurements_by_type.ipynb), plus: the table must have a
location to plot (Annette, 2026-11-14):
  - catalog base (published LLM tables, rebuilt locations and table dates, tribal pages removed)
  - data pages only (no index / report-list / narrative pages)
  - the 7 core water types (no Other / Not Water Related)
  - dated, loose dates allowed (period-of-record / construction-year fallbacks), start year < 1980
  - ANY rebuilt location class: precise, approximate, basin (DWR B118 polygon point) and conflict
    (sources disagree; coordinates kept); point must lie in California
One point per unique location within a panel (duplicate coordinates plotted once).
A more constrained groundwater figure will follow separately.

Output: manuscript/figures/fig2_category_maps.png (+ .pdf),
        data/analysis/fig2_category_counts.csv (panel counts)
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_filters import data_page_mask                 # noqa: E402
from water_palette import CATEGORY_ORDER, get_color_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
FIG = ROOT / 'manuscript/figures'
LOC_CLASSES = ('precise', 'approximate', 'basin', 'conflict')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})

# ---------------------------------------------------------------- data
cat = pd.read_csv(ROOT / 'data/catalog/restored_tables_catalog.csv', low_memory=False)
n0 = len(cat)
cat = cat[data_page_mask(cat) & cat.water_type.isin(CATEGORY_ORDER)
          & cat.year_start.notna() & (cat.year_start < 1980)
          & cat.precision_class.isin(LOC_CLASSES) & cat.latitude.notna()].copy()

ca = gpd.read_file(A / 'spatial/ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
pts = gpd.GeoDataFrame(cat, geometry=gpd.points_from_xy(cat.longitude, cat.latitude), crs='EPSG:4326')
in_ca = gpd.sjoin(pts[['geometry']], ca[['geometry']], how='left', predicate='within').index_right.notna()
in_ca = in_ca.groupby(level=0).any()
pts = pts[in_ca.reindex(pts.index, fill_value=False).values]
print(f'catalog tables {n0:,} -> plotted tables {len(pts):,}')
print(pts.precision_class.value_counts().to_string(), '\n')

pts['xy'] = pts.latitude.round(5).astype(str) + ',' + pts.longitude.round(5).astype(str)
rows = []
for wt in CATEGORY_ORDER:
    g = pts[pts.water_type == wt]
    rows.append({'water_type': wt, 'locations': g.xy.nunique(), 'tables': len(g),
                 'est_measurements': int(g.n_measurements.sum()),
                 **{f'locations_{c}': g[g.precision_class == c].xy.nunique() for c in LOC_CLASSES}})
counts = pd.DataFrame(rows)
counts.to_csv(A / 'fig2_category_counts.csv', index=False)
print(counts.to_string(index=False))


def fmt_meas(v):
    return f'{v / 1e6:.1f} M' if v >= 1e6 else (f'{v / 1e3:.0f} k' if v >= 1e3 else f'{v:,}')


# ---------------------------------------------------------------- figure
# Styling (Annette, 2026-11-15): no table; lower-case bold panel letters; "n =" (estimated measurements)
# in each panel subtitle; points without outline, slightly transparent so overlaps show; thin white
# HUC12 subwatershed lines on every map. Two layouts:
#   fig2_category_maps         7 panels, one per water type
#   fig2_category_maps_3panel  stream discharge | groundwater | all other categories in one panel
Y0, Y1 = int(pts.year_start.min()), int(pts.year_end.fillna(pts.year_start).max())
print(f'years covered (table start - end): {Y0}-{Y1}')
xmin, ymin, xmax, ymax = ca.total_bounds
huc = gpd.read_file(A / 'spatial/ca_wbd/ca_huc12.geojson').to_crs('EPSG:4326')
huc['geometry'] = huc.geometry.make_valid()
huc_lines = gpd.clip(huc, ca).boundary
huc8 = gpd.read_file(A / 'spatial/ca_wbd/ca_huc8.geojson').to_crs('EPSG:4326')
huc8['geometry'] = huc8.geometry.make_valid()
huc8_lines = gpd.clip(huc8, ca).boundary
# 3-panel version (Annette, 2026-11-15): HUC8 instead of HUC12 lines. (The groundwater/precipitation
# colour swap now lives in water_palette.py, so every figure matches.)
SWAP = {}
cidx = counts.set_index('water_type')
OTHERS = [w for w in CATEGORY_ORDER if w not in ('Stream Discharge', 'Groundwater')]
PT = dict(s=4.0, alpha=0.7, linewidths=0, rasterized=True, zorder=3)   # points above the white HUC12 lines


def base_map(ax, lines=None, lw=0.15):
    ca.plot(ax=ax, color='#ececea', edgecolor='none')
    (huc_lines if lines is None else lines).plot(ax=ax, color='white', linewidth=lw)
    ca.boundary.plot(ax=ax, color='#4d4d4d', linewidth=0.4)
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.set_axis_off()


def plot_type(ax, wt, colors=None):
    g = pts[pts.water_type == wt].drop_duplicates('xy')
    ax.scatter(g.longitude, g.latitude, color=(colors or {}).get(wt, get_color_for(wt)), **PT)


def letter(ax, i):
    ax.text(0.0, 1.0, 'abcdefg'[i], transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='left')


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=400)
    plt.close(fig)
    print('saved', FIG / f'{name}.png')


# ---- 7 panels
fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.9))
for i, (ax, wt) in enumerate(zip(axes.flat, CATEGORY_ORDER)):
    base_map(ax)
    plot_type(ax, wt)
    c = cidx.loc[wt]
    ax.set_title(wt + '\n' + f'{c.locations:,} locations · n = {fmt_meas(c.est_measurements)}',
                 fontsize=7, pad=3)
    letter(ax, i)
axes.flat[7].set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.18)
save(fig, 'fig2_category_maps')

# ---- 3 panels: stream | groundwater | all other categories (largest drawn first, so small on top)
# Layout (Annette, 2026-11-15): water-type title larger than the counts line; panel letter on the
# water-type title baseline; less space between panels; "Other" location count = sum over the 5
# categories (1,055; a location holding two categories counts twice).
from matplotlib.transforms import offset_copy  # noqa: E402
from matplotlib.lines import Line2D            # noqa: E402
T_SIZE, N_SIZE, L_SIZE = 10, 7, 11             # points: title, counts line, panel letter
T_PAD = 13                                     # points between map top and title baseline


def titles(ax, i, name, nloc, nmeas):
    ax.set_title(name, fontsize=T_SIZE, pad=T_PAD)
    ax.text(0.5, 1.0, f'{nloc:,} locations · n = {fmt_meas(nmeas)}', fontsize=N_SIZE, ha='center',
            va='baseline', transform=offset_copy(ax.transAxes, fig=ax.figure, y=3, units='points'))
    ax.text(0.0, 1.0, 'abc'[i], fontsize=L_SIZE, fontweight='bold', ha='left', va='baseline',
            transform=offset_copy(ax.transAxes, fig=ax.figure, y=T_PAD, units='points'))


fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.6))
for i, wt in enumerate(['Stream Discharge', 'Groundwater']):
    ax = axes[i]
    base_map(ax, huc8_lines, 0.4)
    plot_type(ax, wt, SWAP)
    c = cidx.loc[wt]
    titles(ax, i, wt, c.locations, c.est_measurements)
ax = axes[2]
base_map(ax, huc8_lines, 0.4)
for wt in sorted(OTHERS, key=lambda w: -cidx.loc[w].locations):
    plot_type(ax, wt, SWAP)
titles(ax, 2, 'Other categories', int(cidx.loc[OTHERS].locations.sum()),
       int(cidx.loc[OTHERS].est_measurements.sum()))
ax.legend(handles=[Line2D([0], [0], marker='o', ls='', markersize=3.5, alpha=0.8,
                          markerfacecolor=get_color_for(w), markeredgewidth=0,
                          label=f'{w} ({cidx.loc[w].locations:,})') for w in OTHERS],
          loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=5.5, frameon=False,
          handletextpad=0.3, borderaxespad=0.2)
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=-0.12)
save(fig, 'fig2_category_maps_3panel')
