"""
Figure 3: restored sites compared with USGS NWIS sites, California, through 1979.
Rows: stream discharge (A, B), groundwater (C, D).
  A, C  active sites per decade, stacked: NWIS / restored at an NWIS site in decades NWIS has
        no data for it / restored sites with no NWIS link
  B, D  HUC12 subwatersheds with >= 1 NWIS site vs restored sites but no NWIS site
Replaces the former six-panel figure (07_ca_combined_6panel.py, kept for reference).

Decisions (Annette, 2026-11-14):
  - NWIS and restored data capped at the same period: NWIS sites must begin <= 1979 and their
    years are cut at 1979; no NWIS site or year after 1979 is used anywhere in the figure.
  - Restored site = definition B, 1 km (21_site_definitions.py, entry_site_keys.csv site_B1):
    records naming the same NWIS station (number or name on the page) or the same state well
    number are one site where their points lie within 1 km; all others one site per location.
  - Restored records: precise + approximate rebuilt location, measurement dates only (no
    period-of-record / construction fallbacks), data pages, inside CA, tribal pages excluded.
  - Active in a decade = at least one year with data in that decade. Restored: years printed in
    the record's table rows, else year_start..year_end. NWIS: years with actual data
    (site_activity.nwis_table; stream since 2026-11-15, groundwater once its download is done),
    so gaps inside an NWIS record now count as decades without NWIS data.
  - NWIS stream = expanded set (daily values, peak flows, other types); NWIS groundwater = wells
    with water-level data.
  - Link restored -> NWIS by identity only: stream = NWIS station number or name printed on the
    page (nearest restored cluster of that station); groundwater = full state well number that
    matches an NWIS well name. Distance alone never links.

Outputs: manuscript/figures/fig3_nwis_comparison.png/.pdf, data/analysis/fig3_stats.csv,
         data/analysis/fig3_decade_counts.csv; with a tolerance argument of 0.5 or 2 (supplement):
         supp_fig3_nwis_comparison_tol{T}km.png/.pdf, fig3_stats_tol{T}km.csv, fig3_decade_counts_tol{T}km.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
FIG = ROOT / 'manuscript/figures'
# Site-definition tolerance (km): 1 = main figure (Annette, 2026-11-14); 0.5 and 2 = supplement.
# Usage: python fig3_nwis_comparison.py [0.5|1|2]
TOL = sys.argv[1] if len(sys.argv) > 1 else '1'
SITE_KEY = f'site_B{float(TOL):g}'
TAG = '' if float(TOL) == 1 else f'_tol{float(TOL):g}km'
PREFIX = 'fig3' if not TAG else 'supp_fig3'
DECADES = list(range(1850, 1980, 10))
# Colours (Annette, 2026-11-15): NWIS as a neutral grey baseline; restored contributions in colour:
# gold (Tol high-contrast #DDAA33) and blue (Okabe-Ito #0072B2). Every pair separated under normal +
# deuteranopia/protanopia/tritanopia simulation (closest: grey vs gold, OKLab dE 12.7; target >= 8).
NWIS_COLOR, LINK_COLOR, NEW_COLOR = '#9E9E9E', '#DDAA33', '#0072B2'

plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})
TITLE_SIZE, LETTER_SIZE, TITLE_PAD = 9, 11, 6   # points (Annette, 2026-11-15: a little bigger, aligned)


from site_activity import (CAP, hav_km, decades_of, load_records, nwis_table,   # noqa: E402
                           all_nwis_coords, restored_sites as _restored_sites)

rec = load_records()


def restored_sites(wt, nwis_coords):
    return _restored_sites(rec, wt, nwis_coords, SITE_KEY)


def decade_stacks(sites, nwis):
    rows = []
    for d in DECADES:
        n_act = {k for k, v in nwis.decades.items() if d in v}
        act = sites[sites.dec.map(lambda v: d in v)]
        linked = act[act.nwis_link.notna()]
        add_link = set(linked.nwis_link) - n_act         # NWIS station, but no NWIS data this decade
        rows.append({'decade': d, 'nwis': len(n_act), 'restored_at_nwis_site': len(add_link),
                     'restored_no_nwis_link': int(act.nwis_link.isna().sum())})
    return pd.DataFrame(rows).set_index('decade')


ca = gpd.read_file(A / 'spatial/ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
huc = gpd.read_file(A / 'spatial/ca_wbd/ca_huc12.geojson').to_crs('EPSG:4326')
huc8 = gpd.read_file(A / 'spatial/ca_wbd/ca_huc8.geojson').to_crs('EPSG:4326')
for g_ in (ca, huc, huc8):
    g_['geometry'] = g_.geometry.make_valid()
huc_ca = gpd.clip(huc, ca)


def huc_set(lat, lon):
    p = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')
    return set(gpd.sjoin(p, huc[['huc12', 'geometry']], how='inner', predicate='within').huc12)


def huc_of(lat, lon, index):
    """HUC12 code per point (Series on index; NaN outside)."""
    p = gpd.GeoDataFrame({'k': index}, geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')
    j = gpd.sjoin(p, huc[['huc12', 'geometry']], how='left', predicate='within')
    return j.drop_duplicates('k').set_index('k').huc12


def filled_hucs(sites, nwis):
    """HUC12s with NWIS sites where restored data cover >= 1 decade in which NO NWIS site in that
    HUC12 has data (before NWIS starts there, after it ends, or between) -- the map version of the
    yellow bar category, at subwatershed level (any restored site in the HUC12 counts)."""
    nh = huc_of(nwis.dec_lat_va.values, nwis.dec_long_va.values, nwis.index)
    rh = huc_of(sites.lat.values, sites.lon.values, sites.index)
    n_dec = pd.Series(nwis.decades.values, index=nwis.index).groupby(nh).agg(lambda x: set().union(*x))
    r_dec = pd.Series(sites.dec.values, index=sites.index).groupby(rh).agg(lambda x: set().union(*x))
    return {h for h, d in r_dec.items() if h in n_dec.index and (d - n_dec[h])}


results, stats = {}, []
for wt, csv in [('Stream Discharge', 'ca_stream_sites_expanded.csv'),
                ('Groundwater', 'ca_groundwater_sites.csv')]:
    nwis = nwis_table(csv)
    sites = restored_sites(wt, all_nwis_coords(csv))
    ts = decade_stacks(sites, nwis)
    n_h = huc_set(nwis.dec_lat_va.values, nwis.dec_long_va.values)
    r_h = huc_set(sites.lat.values, sites.lon.values)
    f_h = filled_hucs(sites, nwis) & n_h
    results[wt] = (ts, n_h, r_h, f_h)
    stats.append({'water_type': wt, 'nwis_sites_le1979': len(nwis),
                  'restored_sites': len(sites),
                  'restored_sites_linked_to_nwis': int(sites.nwis_link.notna().sum()),
                  'restored_sites_no_link': int(sites.nwis_link.isna().sum()),
                  'huc12_nwis': len(n_h), 'huc12_nwis_filled_by_restored': len(f_h),
                  'huc12_nwis_only': len(n_h - f_h), 'huc12_restored': len(r_h),
                  'huc12_restored_no_nwis': len(r_h - n_h), 'huc12_total': len(huc),
                  'pct_gain_basins': round(100 * len(r_h - n_h) / len(n_h), 1)})
    print(wt, '\n', ts.to_string(), '\n')
st = pd.DataFrame(stats)
st.to_csv(A / f'fig3_stats{TAG}.csv', index=False)
pd.concat({k: v[0] for k, v in results.items()}).to_csv(A / f'fig3_decade_counts{TAG}.csv')
print(st.T.to_string())

# ---------------------------------------------------------------- figure
# Stacked bars for panels A/C (Annette, 2026-11-14; a line version was compared and dropped).
for STYLE in ('bars',):
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.6), gridspec_kw={'width_ratios': [1.35, 1]})
    labels = [('NWIS', NWIS_COLOR), ('Restored, at an NWIS site\n(decades without NWIS data)', LINK_COLOR),
              ('Restored, no NWIS site', NEW_COLOR)]
    for row, (wt, title) in enumerate([('Stream Discharge', 'Stream discharge'), ('Groundwater', 'Groundwater')]):
        ts, n_h, r_h, f_h = results[wt]
        s = st.set_index('water_type').loc[wt]
        ax = axes[row, 0]
        x = np.arange(len(DECADES))
        bottom = np.zeros(len(x))
        for col, (lab, c) in zip(['nwis', 'restored_at_nwis_site', 'restored_no_nwis_link'], labels):
            v = ts[col].values
            ax.bar(x, v, 0.8, bottom=bottom, color=c, label=lab,
                   edgecolor='#4d4d4d', linewidth=0.3)   # thin dark-grey outline (Annette, 2026-11-15)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}s' for d in DECADES], rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('Active sites per decade')
        # no x-axis label (Annette, 2026-11-15): the tick labels already read as decades
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'{title}: active sites', fontsize=TITLE_SIZE, loc='left', pad=TITLE_PAD)
        ax.legend(fontsize=6, frameon=False, loc='upper left')

        ax = axes[row, 1]
        b = huc_ca.copy()
        b['cls'] = np.where(b.huc12.isin(f_h), 'filled',
                            np.where(b.huc12.isin(n_h), 'nwis', np.where(b.huc12.isin(r_h), 'restored', 'none')))
        ca.plot(ax=ax, color='white', edgecolor='#4d4d4d', linewidth=0.4)
        b[b.cls == 'nwis'].plot(ax=ax, color=NWIS_COLOR, linewidth=0)
        b[b.cls == 'filled'].plot(ax=ax, color=LINK_COLOR, linewidth=0)
        b[b.cls == 'restored'].plot(ax=ax, color=NEW_COLOR, linewidth=0)
        # HUC12 and HUC8 grey/dark lines dropped (Annette, 2026-11-15); instead a thin WHITE line between
        # subwatersheds on panel D, which separates the filled patches without adding grey linework
        huc_ca.boundary.plot(ax=ax, color='white', linewidth=0.15)   # panels b and d (Annette, 2026-11-15)
        ca.boundary.plot(ax=ax, color='#4d4d4d', linewidth=0.4)
        ax.set_axis_off()
        ax.set_title(f'{title}: subwatersheds (HUC12)', fontsize=TITLE_SIZE, loc='left', pad=TITLE_PAD)
        from matplotlib.patches import Patch
        h = [Patch(color=NWIS_COLOR, label=f'NWIS site ({int(s.huc12_nwis_only):,})'),
             Patch(color=LINK_COLOR, label=f'NWIS site + restored decades without NWIS data '
                                           f'({int(s.huc12_nwis_filled_by_restored):,})'),
             Patch(color=NEW_COLOR, label=f'Restored, no NWIS site ({int(s.huc12_restored_no_nwis):,}; '
                                          f'+{s.pct_gain_basins:.0f}%)')]
        ax.legend(handles=h, fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.01),
                  ncol=1)

    for ax, letter in zip(axes.flat, 'abcd'):   # lower-case bold panel letters (Annette, 2026-11-15)
        # same baseline as the panel title: title sits TITLE_PAD points above the axes top, baseline-aligned
        ax.text(-0.02, 1.0, letter, fontsize=LETTER_SIZE, fontweight='bold', ha='right', va='baseline',
                transform=offset_copy(ax.transAxes, fig=fig, y=TITLE_PAD, units='points'))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, wspace=0.12, hspace=0.38)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{PREFIX}_nwis_comparison{TAG}.{ext}', dpi=400)
    plt.close(fig)
    print('saved', STYLE)
