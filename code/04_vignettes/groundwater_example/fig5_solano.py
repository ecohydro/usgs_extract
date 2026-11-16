"""
Figure 5 (DRAFT, 2026-11-16): restored groundwater levels in Solano County (USGS Water-Supply Paper 1464,
Thomasson and others, 1960, Appendix B; project doc 2809). Same layout and colours as Figure 4.

a  Map: DWR Bulletin 118 subbasins that hold the restored wells (Solano, Yolo, Suisun-Fairfield).
   Restored wells (solano_wsp1464_wells_located.csv: PLSS section centroid / grid / NWIS coordinates) vs
   NWIS groundwater wells in the same subbasins with a water-level reading <= 1979 (nwis_years_gw.csv).
c  Well 8/1E-15B1 (NWIS 383248121505501): NWIS readings 1931-1979 and the restored readings 1931-1951.
   Restored values are depth below land-surface datum as printed + 1.7 ft (constant difference to NWIS at
   all 26 shared dates with complete values, +/-0.04 ft; cause not established).
b  Wells with water-level readings per decade, restored vs NWIS, same subbasins.
c  (above as b)
d  Putah-area network 1931-1951: median and 25-75th percentile of the restored wells' annual mean depth,
   NWIS 15B1 annual means, and wells measured per year (restored vs NWIS in the same area). Replaced the
   single-well panel (5/2W-32A1, record has gaps) on 2026-11-16; one-well alternative = fig5_alt_36R1.py.
Plotted readings: static levels only (restored: no footnote, value not truncated; NWIS: qualifier 'Static').

Output: manuscript/figures/fig5_solano.png/.pdf (draft), data/analysis/gw_example/fig5_decade_counts.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'santa_ynez'))
from fetch_nwis_gw_levels import fetch  # noqa: E402

A = ROOT / 'data/analysis'
G = A / 'gw_example'
FIG = ROOT / 'manuscript/figures'
RESTORED, NWIS_C, STAR = '#EE6677', '#4477AA', '#FFD23F'      # Figure 4 colours
OFFSET = 1.7
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})

w = pd.read_csv(G / 'solano_wsp1464_wells_located.csv')
rd = pd.read_csv(G / 'solano_wsp1464_readings.csv', parse_dates=['date'])
rd['static'] = rd.footnote.isna() & rd.depth_ft.notna() & rd.note.isna()

b = gpd.read_file(A / 'spatial/sgma/sgma_2019_basin_prioritization.geojson')
b['geometry'] = b.geometry.make_valid().buffer(0)
b = b.to_crs('EPSG:4326')
wg = gpd.GeoDataFrame(w, geometry=gpd.points_from_xy(w.lon, w.lat), crs='EPSG:4326')
wg = gpd.sjoin(wg, b[['Basin_Subbasin_Number', 'geometry']], how='left', predicate='within')
area = b[b.Basin_Subbasin_Number.isin(wg.Basin_Subbasin_Number.dropna().unique())]

n = pd.read_csv(A / 'nwis_sites/ca_groundwater_sites.csv', low_memory=False, dtype={'site_no': str})
n = n.drop_duplicates('site_no')
ng = gpd.GeoDataFrame(n, geometry=gpd.points_from_xy(n.dec_long_va, n.dec_lat_va), crs='EPSG:4326')
ng = gpd.sjoin(ng, area[['Basin_Subbasin_Number', 'geometry']], predicate='within')
ny = pd.read_csv(A / 'nwis_sites/nwis_years_gw.csv', dtype={'site_no': str})
ny = ny[ny.site_no.isin(ng.site_no) & (ny.year <= 1979)]
ng = ng[ng.site_no.isin(ny.site_no)]

# decade counts (for the caption)
rr = rd.merge(wg[['well', 'Basin_Subbasin_Number']], on='well')
rr = rr[rr.Basin_Subbasin_Number.notna()]
dec = pd.DataFrame({'restored_wells': rr.groupby(rr.year // 10 * 10).well.nunique(),
                    'restored_readings': rr.groupby(rr.year // 10 * 10).size(),
                    'nwis_wells': ny.groupby(ny.year // 10 * 10).site_no.nunique(),
                    'nwis_readings': ny.groupby(ny.year // 10 * 10).n_measurements.sum()}).fillna(0).astype(int)
dec.index.name = 'decade'
dec.to_csv(G / 'fig5_decade_counts.csv')
print(dec.to_string())

# ---------------------------------------------------------------- figure
import matplotlib.patheffects as pe  # noqa: E402
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
first = ny.groupby('site_no').year.min()
ng['early'] = ng.site_no.map(first) < 1950

fig = plt.figure(figsize=(7.25, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.45, 1.0], hspace=0.28, wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
area.plot(ax=ax, color='#f2f2f0', edgecolor='#9e9e9e', linewidth=0.6)
xmin, ymin, xmax, ymax = wg.total_bounds
ax.set_xlim(xmin - 0.05, xmax + 0.05)
ax.set_ylim(ymin - 0.04, ymax + 0.04)
for name, (x, y) in {'Yolo': (-121.72, 38.62), 'Solano': (-121.80, 38.30),
                     'Suisun-Fairfield Valley': (-122.13, 38.17)}.items():
    if xmin - 0.05 < x < xmax + 0.05 and ymin - 0.04 < y < ymax + 0.04:
        ax.text(x, y, name, fontsize=6, color='#6b6b6b', ha='center', style='italic', path_effects=HALO)
late = ng[~ng.early]
early = ng[ng.early]
ax.scatter(late.dec_long_va, late.dec_lat_va, s=9, marker='s', facecolor='none', edgecolor='#a9bfd9',
           linewidths=0.5, zorder=4)
ax.scatter(early.dec_long_va, early.dec_lat_va, s=18, marker='s', facecolor='none', edgecolor=NWIS_C,
           linewidths=0.9, zorder=5)
ax.scatter(w.lon, w.lat, s=13, color=RESTORED, edgecolors='white', linewidths=0.35, zorder=6)
for well, lab, dx in (('8/1E-15B1', 'c', 6),):
    p = w[w.well == well].iloc[0]
    ax.scatter([p.lon], [p.lat], s=90, marker='*', color=STAR, edgecolor='black', linewidths=0.5, zorder=9)
    ax.annotate(f'{well} ({lab})', (p.lon, p.lat), xytext=(dx, 5), textcoords='offset points', fontsize=6.5,
                path_effects=HALO, zorder=10)
ax.set_aspect(1 / np.cos(np.radians(38.4)))
ax.set_xticks([])
ax.set_yticks([])
ax.legend(handles=[Line2D([], [], marker='o', color=RESTORED, mec='white', lw=0, ms=4,
                          label=f'Restored wells, 1918-1952 ({len(w)})'),
                   Line2D([], [], marker='s', mfc='none', mec=NWIS_C, mew=0.9, lw=0, ms=4,
                          label=f'NWIS wells, reading before 1950 ({len(early)})'),
                   Line2D([], [], marker='s', mfc='none', mec='#a9bfd9', lw=0, ms=3.5,
                          label=f'NWIS wells, first reading 1950-1979 ({len(late)})'),
                   Line2D([], [], marker='*', color=STAR, mec='black', lw=0, ms=8, label='Well in c')],
          loc='lower right', fontsize=5.8, frameon=True, framealpha=0.9, edgecolor='none')
ax.set_title('Solano County basins (DWR Bulletin 118)', fontsize=7.5, loc='left')
ax.text(-0.02, 1.02, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold', ha='right')

# b: wells measured per decade
axd = fig.add_subplot(gs[0, 1])
d = dec.loc[1910:1970]
x = np.arange(len(d))
axd.bar(x - 0.2, d.restored_wells, 0.4, color=RESTORED, edgecolor='#4d4d4d', linewidth=0.3, label='Restored')
axd.bar(x + 0.2, d.nwis_wells, 0.4, color=NWIS_C, edgecolor='#4d4d4d', linewidth=0.3, label='NWIS')
for xi, (rv, nv) in enumerate(zip(d.restored_wells, d.nwis_wells)):
    for off, v in ((-0.2, rv), (0.2, nv)):
        if 0 < v < 40 or (v == 0 and off < 0 and xi < 5):
            axd.text(xi + off, v + 3, f'{v}', ha='center', fontsize=5.5)
axd.set_xticks(x, [f'{i}s' for i in d.index], rotation=45, ha='right')
axd.set_ylabel('Wells with water-level readings')
axd.spines[['top', 'right']].set_visible(False)
axd.grid(axis='y', color='#e6e6e6', lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=6, frameon=False, loc='upper left')
axd.set_title('Wells measured per decade, same basins', fontsize=7.5, loc='left')
axd.text(-0.16, 1.02, 'b', transform=axd.transAxes, fontsize=11, fontweight='bold')


def hydro(ax, title):
    ax.invert_yaxis()
    ax.set_title(title, fontsize=7.5, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e6e6', lw=0.5)


# c: 15B1
axb = fig.add_subplot(gs[1, 0])
nw = fetch('383248121505501')
nw['date'] = pd.to_datetime(nw.time.str[:10])
nw = nw[nw.qualifier.astype(str).str.contains('Static')].sort_values('date')
r15 = rd[(rd.well == '8/1E-15B1') & rd.static].sort_values('date').copy()
r15['adj'] = r15.depth_ft + OFFSET
shared = r15.date.isin(nw.date)
axb.plot(nw.date, nw.value, '-', color=NWIS_C, lw=0.5, alpha=0.6, zorder=2)
axb.scatter(nw.date, nw.value, s=5, color=NWIS_C, zorder=3, label=f'NWIS ({len(nw)})')
axb.scatter(r15.date[~shared], r15.adj[~shared], s=9, color=RESTORED, edgecolors='white', linewidths=0.3,
            zorder=4, label=f'Restored, not in NWIS ({(~shared).sum()})')
axb.scatter(r15.date[shared], r15.adj[shared], s=16, facecolor='none', edgecolor=RESTORED, linewidths=0.6,
            zorder=4, label=f'Restored, also in NWIS ({shared.sum()})')
hydro(axb, '8/1E-15B1, west of Davis (NWIS 383248121505501)')
axb.set_ylabel('Depth to water (ft below land surface)')
axb.legend(fontsize=6, frameon=False, loc='lower left')
axb.text(-0.02, 1.02, 'c', transform=axb.transAxes, fontsize=11, fontweight='bold', ha='right')

# d: the restored network around 15B1 (Annette, 2026-11-16, option 1). Putah-area wells of WSP 1464,
# static readings 1931-1951; per well the annual mean, then per year the median and 25-75th percentile
# across wells measured that year (well set changes between years: UC Davis network 1931-42, USGS
# network 1948-51). NWIS: wells inside the Putah-area well hull (+2 km) with readings 1931-1951.
put = w[w.area == 'Putah area']
pn = rd[rd.well.isin(put.well) & rd.static & rd.year.between(1931, 1951)]
ann = pn.groupby(['well', 'year']).depth_ft.mean().reset_index()
yr = ann.groupby('year').depth_ft.agg(med='median', q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
                                      n='size').reindex(range(1931, 1952))
hull = gpd.GeoSeries(gpd.points_from_xy(put.lon, put.lat), crs='EPSG:4326').to_crs(3310).union_all() \
    .convex_hull.buffer(2000)
n3310 = gpd.GeoSeries(gpd.points_from_xy(n.dec_long_va, n.dec_lat_va), crs='EPSG:4326').to_crs(3310)
nin = set(n.site_no[n3310.within(hull).values])
nyp = pd.read_csv(A / 'nwis_sites/nwis_years_gw.csv', dtype={'site_no': str})
nyp = nyp[nyp.site_no.isin(nin) & nyp.year.between(1931, 1951)]
nwells = nyp.groupby('year').site_no.nunique().reindex(range(1931, 1952), fill_value=0)
nw15 = nw[nw.date.dt.year.between(1931, 1951)].groupby(nw.date.dt.year).value.mean()

sub = gs[1, 1].subgridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
axc = fig.add_subplot(sub[0])
ok = yr.n >= 10                                          # years with few wells (1945-47: 3-8) not summarised
yy = yr[ok]
for seg in (yy.loc[:1942], yy.loc[1948:]):
    axc.fill_between(seg.index, seg.q1, seg.q3, color=RESTORED, alpha=0.25, lw=0)
    axc.plot(seg.index, seg.med, '-o', color=RESTORED, ms=2.5, lw=1.0)
axc.plot(nw15.index, nw15.values, 's', color=NWIS_C, ms=3, zorder=5)
hydro(axc, 'Putah area network, 1931-1951')
axc.set_xlim(1930.3, 1951.7)
axc.tick_params(labelbottom=False)
axc.legend(handles=[Line2D([], [], color=RESTORED, marker='o', ms=2.5, lw=1.0,
                           label=f'Restored wells: median ({pn.well.nunique()} wells)'),
                    plt.Rectangle((0, 0), 1, 1, color=RESTORED, alpha=0.25, label='Restored: 25th-75th percentile'),
                    Line2D([], [], color=NWIS_C, marker='s', ms=3, lw=0, label='NWIS: 8/1E-15B1 (annual mean)')],
           fontsize=5.8, frameon=False, loc='lower left')
axc.text(-0.16, 1.02, 'd', transform=axc.transAxes, fontsize=11, fontweight='bold')

axn = fig.add_subplot(sub[1], sharex=axc)
x = np.arange(1931, 1952)
axn.bar(x - 0.2, yr.n.fillna(0), 0.4, color=RESTORED, lw=0)
axn.bar(x + 0.2, nwells.values, 0.4, color=NWIS_C, lw=0)
axn.set_ylabel('Wells', fontsize=6.5)
axn.spines[['top', 'right']].set_visible(False)
axn.set_xticks(range(1932, 1952, 4))
axn.tick_params(labelsize=6.5)
print('Putah network per year:', yr.n.fillna(0).astype(int).to_dict())
print('NWIS wells per year:', nwells.to_dict())

fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.06)
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'fig5_solano.{ext}', dpi=400)
print('saved', FIG / 'fig5_solano.png')
print(f'15B1: NWIS static {len(nw)}, restored static {len(r15)} (not in NWIS {(~shared).sum()}); '
      f'NWIS early {len(early)} late {len(late)}')
