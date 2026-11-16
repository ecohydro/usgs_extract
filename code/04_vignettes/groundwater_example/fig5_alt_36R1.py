"""
Figure 5 alternative (2026-11-16, Annette asked to see option 2 separately): one restored well with a
continuous record that is not in NWIS. Well 8/1W-36R1 (UC 194), Sweeney, about 3.5 miles southeast of
Winters, unused well (WSP 1464 Appendix B, PDF page 598): University of California readings twice a year,
1931-1942. Whole record printed on one digitized page (checked: page 599 does not continue it).

a  Locator: restored WSP 1464 wells (grey) and 8/1W-36R1 (star), with the B118 subbasins.
b  Hydrograph, static readings only (footnoted or tenths-truncated readings shown as grey x).
Output: manuscript/figures/fig5_alt_36R1.png/.pdf
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
A = ROOT / 'data/analysis'
G = A / 'gw_example'
FIG = ROOT / 'manuscript/figures'
RESTORED, STAR = '#EE6677', '#FFD23F'
WELL = '8/1W-36R1'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})

w = pd.read_csv(G / 'solano_wsp1464_wells_located.csv')
rd = pd.read_csv(G / 'solano_wsp1464_readings.csv', parse_dates=['date'])
rd['static'] = rd.footnote.isna() & rd.depth_ft.notna() & rd.note.isna()
r = rd[rd.well == WELL].sort_values('date')
ok, bad = r[r.static], r[~r.static & r.depth_ft.notna()]
p = w[w.well == WELL].iloc[0]

b = gpd.read_file(A / 'spatial/sgma/sgma_2019_basin_prioritization.geojson')
b['geometry'] = b.geometry.make_valid().buffer(0)
b = b.to_crs('EPSG:4326')
wg = gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.lon, w.lat), crs='EPSG:4326')
area = b[b.Basin_Subbasin_Number.isin(gpd.sjoin(wg, b, predicate='within').Basin_Subbasin_Number.unique())]

fig = plt.figure(figsize=(7.25, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.15)
ax = fig.add_subplot(gs[0])
area.plot(ax=ax, color='#f2f2f0', edgecolor='#9e9e9e', linewidth=0.6)
xmin, ymin, xmax, ymax = wg.total_bounds
ax.set_xlim(xmin - 0.05, xmax + 0.05)
ax.set_ylim(ymin - 0.04, ymax + 0.04)
ax.scatter(w.lon, w.lat, s=8, color='#9e9e9e', edgecolors='white', linewidths=0.3, zorder=5)
ax.scatter([p.lon], [p.lat], s=110, marker='*', color=STAR, edgecolor='black', linewidths=0.5, zorder=9)
ax.annotate(WELL, (p.lon, p.lat), xytext=(6, 4), textcoords='offset points', fontsize=6.5,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
ax.set_aspect(1 / np.cos(np.radians(38.4)))
ax.set_xticks([])
ax.set_yticks([])
ax.legend(handles=[Line2D([], [], marker='o', color='#9e9e9e', mec='white', lw=0, ms=3.5,
                          label=f'Restored wells, WSP 1464 ({len(w)})'),
                   Line2D([], [], marker='*', color=STAR, mec='black', lw=0, ms=8, label=WELL)],
          loc='lower right', fontsize=5.8, frameon=True, framealpha=0.9, edgecolor='none')
ax.text(-0.02, 1.02, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right')

axh = fig.add_subplot(gs[1])
axh.plot(ok.date, ok.depth_ft, '-', color=RESTORED, lw=0.8)
axh.scatter(ok.date, ok.depth_ft, s=14, color=RESTORED, edgecolors='white', linewidths=0.4, zorder=4,
            label=f'Restored, static ({len(ok)})')
if len(bad):
    axh.scatter(bad.date, bad.depth_ft, s=14, marker='x', color='#9e9e9e', linewidths=0.7, zorder=4,
                label=f'Footnoted or truncated ({len(bad)})')
axh.invert_yaxis()
axh.set_ylabel('Depth to water (ft below land surface)')
axh.set_title(f'{WELL} (UC 194), unused well SE of Winters; not in NWIS', fontsize=7.5, loc='left')
axh.spines[['top', 'right']].set_visible(False)
axh.grid(axis='y', color='#e6e6e6', lw=0.5)
axh.legend(fontsize=6, frameon=False, loc='lower left')
axh.text(-0.08, 1.02, 'b', transform=axh.transAxes, fontsize=11, fontweight='bold')

fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10)
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'fig5_alt_36R1.{ext}', dpi=400)
print('saved', FIG / 'fig5_alt_36R1.png', '| static', len(ok), 'other', len(bad),
      '| years', r.year.min(), '-', r.year.max(), '| location', p.location_method)
