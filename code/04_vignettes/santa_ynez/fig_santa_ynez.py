"""
Figure 4, Santa Ynez River (2026-11-14): A = watershed map, B = Gibraltar Dam annual flow.
Replaces the dissertation stream-order figure from 02_inventory/04_santa_ynez.ipynb (its timeline
panel is dropped, Annette 2026-11-14). Panel letters lower-case bold, 11 pt (journal; Annette 2026-11-19).

Panel A (same rules as Figure 3, via code/02_inventory/site_activity.py):
  - restored stream discharge SITES (definition B, 1 km), not records; inside the Santa Ynez HUC8;
    colour = decade of the site's first measurement
  - NWIS stream sites with a discharge-related record beginning <= 1979 (Figure 3 set), not the old
    sb_streamflow_sites.csv (7 sites with partial spans, e.g. 11123000 listed as 1973-75)
  - rivers coloured by stream order (chap2map.gdb StreamsandRivers_Clip, StreamOrderEdit); lakes
  - star = Gibraltar Dam gauge (NWIS 11123000), the station in panel b
Panel B: code/04_vignettes/santa_ynez/gibraltar_plot.py (data from gibraltar_annual.py).
Tables 7-9 rebuilt with the same sites -> data/analysis/santa_ynez/table7_*.csv ... table9_*.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
sys.path.insert(0, str(HERE))
from site_activity import load_records, nwis_table, all_nwis_coords, restored_sites, CAP  # noqa: E402
from gibraltar_plot import plot_gibraltar  # noqa: E402
import fig_style as S  # noqa: E402

A = ROOT / 'data/analysis'
SP = A / 'spatial'
OUT = A / 'santa_ynez'
FIG = ROOT / 'manuscript/figures'
GDB = SP / 'lakes/chap2map.gdb'
S.apply()

# ---------------------------------------------------------------- layers
ws = gpd.read_file(SP / 'santaynezhuc8/santaynez.shp').to_crs('EPSG:4326')
layers = [l for l, _ in pyogrio.list_layers(GDB)]
rivers = gpd.read_file(GDB, layer='StreamsandRivers_Clip').to_crs('EPSG:4326')
lake_layer = [l for l in layers if 'lake' in l.lower() or 'wetland' in l.lower()][0]
lakes = gpd.clip(gpd.read_file(GDB, layer=lake_layer).to_crs('EPSG:4326'), ws)
county = gpd.read_file(SP / 'sb_county_shapefile/sb_county_shp.shp').to_crs('EPSG:4326')


def in_ws(df, lat, lon):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs='EPSG:4326')
    return gpd.sjoin(g, ws[['geometry']], how='inner', predicate='within').drop(columns='index_right')


def stream_order(g):
    near = gpd.sjoin_nearest(g.to_crs('EPSG:3310'), rivers[['StreamOrderEdit', 'geometry']].to_crs('EPSG:3310'),
                             how='left', distance_col='dist_m')
    near = near[~near.index.duplicated(keep='first')]
    return near.StreamOrderEdit.astype(int), near.dist_m


# ---------------------------------------------------------------- sites
csv = 'ca_stream_sites_expanded.csv'
rec = load_records()
sites = restored_sites(rec, 'Stream Discharge', all_nwis_coords(csv), 'site_B1').reset_index()
sites = in_ws(sites, 'lat', 'lon')
sites['order'], sites['dist_m'] = stream_order(sites)
sites['first_decade'] = (sites.first_year // 10 * 10).astype(int)
nwis = in_ws(nwis_table(csv).reset_index(), 'dec_lat_va', 'dec_long_va')
nwis['order'], nwis['dist_m'] = stream_order(nwis)
print(f'restored sites in watershed: {len(sites)} (median distance to reach {sites.dist_m.median():.0f} m); '
      f'NWIS sites (record begins <= {CAP}): {len(nwis)}')

# ---------------------------------------------------------------- tables 7-9
nwis['last_year'] = nwis.decades.map(lambda d: min(max(d) + 9, CAP))
t7 = nwis.groupby('order').agg(n_nwis_sites=('site_no', 'size'), first_year=('first_year', 'min'),
                               last_year=('last_year', 'max')).reset_index()
t8 = t7[['order', 'n_nwis_sites']].merge(sites.groupby('order').size().rename('n_restored_sites').reset_index(),
                                         on='order', how='outer').fillna(0).astype(int)
t8['n_restored_sites_no_nwis_link'] = t8.order.map(sites[sites.nwis_link.isna()].groupby('order').size()).fillna(0).astype(int)
dec = sites[['order', 'dec']].explode('dec')
t9 = dec.groupby(['dec', 'order']).size().unstack(fill_value=0)
t9.index.name = 'decade (active restored sites)'
for name, t in [('table7_nwis_by_order', t7), ('table8_nwis_vs_restored_by_order', t8),
                ('table9_restored_active_sites_by_decade_order', t9)]:
    t.to_csv(OUT / f'{name}.csv', index=name.startswith('table9'))
print('\nTable 7\n', t7.to_string(index=False), '\nTable 8\n', t8.to_string(index=False), '\nTable 9\n', t9.to_string())

# ---------------------------------------------------------------- figure
# Figure 4 (Annette, 2026-11-14): restored sites in ONE colour (the decade colouring was hard to read;
# kept as an alternative file). Colours (Annette 2026-11-19): the Figure 3 scheme in all figures
# (fig_style.py): restored blue, NWIS grey, both gold. Streams in a pale blue ramp so the blue
# restored dots stand out; Gibraltar gauge star white (yellow read as the gold 'both' colour).
RESTORED, NWIS_C, STAR = S.RESTORED, S.NWIS_DARK, 'white'


def make_figure(by_decade, name):
    fig = plt.figure(figsize=(7.25, 5.4))
    gsp = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)
    ax = fig.add_subplot(gsp[0])
    county.plot(ax=ax, color='#f2f2f0', edgecolor='#bdbdbd', linewidth=S.LW_MIN)
    orders = sorted(rivers.StreamOrderEdit.dropna().unique())
    blues = plt.cm.Blues(np.linspace(0.18, 0.5, len(orders)))
    for o, c in zip(orders, blues):
        rivers[rivers.StreamOrderEdit == o].plot(ax=ax, color=c, linewidth=0.7 + 0.3 * o, zorder=2)
    if not lakes.empty:
        lakes.plot(ax=ax, color='#9fd8ef', linewidth=0, zorder=3)
    ws.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=4)
    if by_decade:
        decs = sorted(sites.first_decade.unique())
        dcol = dict(zip(decs, plt.cm.viridis(np.linspace(0, 0.95, len(decs)))))
        for d in decs:
            p = sites[sites.first_decade == d]
            ax.scatter(p.lon, p.lat, s=14, color=dcol[d], linewidths=0, zorder=6)
    else:
        ax.scatter(sites.lon, sites.lat, s=14, color=RESTORED, linewidths=0, zorder=6)
    nwis_edge = '#d95f02' if by_decade else NWIS_C
    ax.scatter(nwis.dec_long_va, nwis.dec_lat_va, s=22, marker='s', facecolor='none', edgecolor=nwis_edge,
               linewidths=1.0, zorder=7)
    gib = nwis.set_index('site_no').loc['11123000']
    star = '#d7301f' if by_decade else STAR
    ax.scatter([gib.dec_long_va], [gib.dec_lat_va], s=110, marker='*', color=star, edgecolor='black',
               linewidths=S.LW_MIN, zorder=9)
    ax.annotate('Gibraltar Dam gauge (b)', (gib.dec_long_va, gib.dec_lat_va), xytext=(-20, 34),
                textcoords='offset points', fontsize=S.SMALL, arrowprops=dict(arrowstyle='-', lw=S.LW_MIN),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85), zorder=10)
    minx, miny, maxx, maxy = ws.total_bounds
    ax.set_xlim(minx - 0.03, maxx + 0.03)
    ax.set_ylim(miny - 0.03, maxy + 0.03)
    ax.set_aspect(1 / np.cos(np.radians(34.6)))
    ax.set_xticks([]); ax.set_yticks([])
    S.scalebar(ax, 10, 34.6, loc=(0.04, 0.08))   # journal: scale bar on maps (2026-11-19)
    for sp in ax.spines.values():
        sp.set_linewidth(S.LW_MIN)
    h1 = ([Line2D([], [], color=c, lw=0.7 + 0.3 * o, label=f'Stream order {int(o)}') for o, c in zip(orders, blues)] +
          [Line2D([], [], marker='s', mfc='none', mec=nwis_edge, lw=0, ms=4,
                  label=f'NWIS site, record\nbegins by {CAP} ({len(nwis)})')] +
          ([] if by_decade else [Line2D([], [], marker='o', color=RESTORED, lw=0, ms=4,
                                        label=f'Restored site ({len(sites)})')]) +
          [Line2D([], [], marker='*', color=star, mec='black', lw=0, ms=8, label='Gibraltar Dam gauge')])
    leg1 = ax.legend(handles=h1, fontsize=S.SMALL, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.04))
    if by_decade:
        ax.add_artist(leg1)
        h2 = [Line2D([], [], marker='o', color=dcol[d], lw=0, ms=4, label=f'{d}s') for d in decs]
        ax.legend(handles=h2, fontsize=S.SMALL, frameon=False, loc='upper left', bbox_to_anchor=(1.21, 1.04),
                  title=f'Restored sites ({len(sites)}),\nfirst measured', title_fontsize=S.SMALL, alignment='left')
    S.letter(ax, 'a', x=-0.01, y=1.03)

    axb = fig.add_subplot(gsp[1])
    plot_gibraltar(axb, legend_kw=dict(fontsize=S.SMALL, frameon=False, loc='upper center',
                                       bbox_to_anchor=(0.5, -0.18), ncol=3))
    S.letter(axb, 'b', x=-0.01, y=1.03)
    fig.subplots_adjust(left=0.08, right=0.74 if by_decade else 0.80, top=0.96, bottom=0.16)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=400)
    plt.close(fig)
    print('saved', FIG / f'{name}.png')


make_figure(by_decade=False, name='fig4_santa_ynez')
make_figure(by_decade=True, name='fig4_santa_ynez_by_decade')   # alternative, not the paper figure
