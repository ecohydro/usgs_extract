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
                 'est_measurements': int(g.n_values.sum()),
                 **{f'locations_{c}': g[g.precision_class == c].xy.nunique() for c in LOC_CLASSES}})
counts = pd.DataFrame(rows)
counts.to_csv(A / 'fig2_category_counts.csv', index=False)
print(counts.to_string(index=False))


def fmt_meas(v):
    return f'{v / 1e6:.1f} M' if v >= 1e6 else (f'{v / 1e3:.0f} k' if v >= 1e3 else f'{v:,}')


# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.9))
xmin, ymin, xmax, ymax = ca.total_bounds
for ax, wt in zip(axes.flat, CATEGORY_ORDER):
    g = pts[pts.water_type == wt].drop_duplicates('xy')
    ca.plot(ax=ax, color='#ececea', edgecolor='#4d4d4d', linewidth=0.4)
    col = np.array(get_color_for(wt))
    ax.scatter(g.longitude, g.latitude, s=2.0, color=col, alpha=0.85,
               edgecolors=tuple(col * 0.6), linewidths=0.15, rasterized=True)
    c = counts.set_index('water_type').loc[wt]
    ax.set_title(f'{wt}\n{c.locations:,} locations · {fmt_meas(c.est_measurements)} meas.',
                 fontsize=7, pad=3)
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.set_axis_off()

# 8th slot: counts table
ax = axes.flat[7]
ax.set_axis_off()
tab = [[wt, f'{r.locations:,}', f'{r.tables:,}', fmt_meas(r.est_measurements)]
       for wt, r in counts.set_index('water_type').iterrows()]
tot = counts[['locations', 'tables', 'est_measurements']].sum()
tab.append(['Total', f'{tot.locations:,}', f'{tot.tables:,}', fmt_meas(tot.est_measurements)])
t = ax.table(cellText=tab, colLabels=['', 'Locations', 'Tables', 'Meas.'],
             cellLoc='right', colLoc='right', edges='horizontal',
             bbox=[0.0, 0.10, 1.0, 0.76], colWidths=[0.40, 0.20, 0.20, 0.20])
ax.text(1.0, 0.04, 'Meas. = estimated measurements', transform=ax.transAxes, ha='right',
        va='top', fontsize=5.5, color='#555555')
t.auto_set_font_size(False)
t.set_fontsize(6)
for (r, col), cell in t.get_celld().items():
    cell.set_linewidth(0.3)
    if col == 0:
        cell.set_text_props(ha='left')
        cell._loc = 'left'
    if r == len(tab):
        cell.set_text_props(fontweight='bold')

for i, ax in enumerate(axes.flat[:7]):
    ax.text(0.0, 1.0, 'ABCDEFG'[i], transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='left')

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.18)
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'fig2_category_maps.{ext}', dpi=400)
print('saved', FIG / 'fig2_category_maps.png')
