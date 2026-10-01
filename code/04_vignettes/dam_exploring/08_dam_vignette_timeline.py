"""
Dam vignette figure (draft): restored vs. NWIS streamflow records on two dammed
rivers, 1880-1980.

One row per river: (left) watershed map with stations numbered; (right) station
timeline -- NWIS period of record (blue bar, capped at 1980) and restored years
(red ticks), with dam completion years as vertical lines.

Stations are identified by NAME, not coordinates: restored tables are matched to
the river by watersource_name, and paired with an NWIS station when the location
part of the name agrees ("... near Mentone" <-> "SANTA ANA R NR MENTONE CA").
This sidesteps the restored-coordinate problem (location accuracy write-up,
2026-10-01). Map positions use NWIS coordinates for paired stations; restored-
only stations use their restored coordinates (hollow markers) and stations with
no named location are not mapped.

Restored years: union over a station's tables of [year_start, year_end] from the
LLM metadata. Multi-year summary tables therefore cover their whole span.
NWIS: sites beginning <= 1979 only (post-1980 NWIS is ignored); canals and
diversions are excluded on both sides.

Inputs:
  data/analysis/table_level_metadata_published.csv
  data/analysis/nwis_sites/ca_stream_sites_expanded.csv
  data/analysis/dams.csv
  data/analysis/spatial/ca_wbd/ca_huc8.geojson
  data/analysis/spatial/nhd/{key}_flowlines.geojson, _waterbodies.geojson
    (fetch_nhd_focal_rivers.py)
Output: manuscript/figures/dam_vignette_timeline.png
"""

import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

root = Path(__file__).resolve().parents[3]
spatial = root / 'data/analysis/spatial'
figures_path = root / 'manuscript/figures'

NWIS_COLOR     = '#4477AA'
RESTORED_COLOR = '#EE6677'
DAM_COLOR      = '#222222'
YEAR_MIN, YEAR_MAX = 1880, 1980
NWIS_YEAR_CUTOFF = 1979
MAX_ROWS = 12
ANNUAL_SPAN = 1   # water year Oct-Sep -> year_end = year_start + 1
EXCLUDE = r'canal|ditch|diversion|spreading|\bcn\b|\bdiv\b|sprd|basin|tributary|trib\b|\bpp\d'

RIVERS = [
    dict(key='santa_ana', title='Santa Ana River', huc8='18070203',
         restored_pat=r'^santa ana river', nwis_prefix='SANTA ANA R ',
         dams=[('Prado', 1941, 33.889999, -117.642111, True)],
         dam_label_offsets=[(0, -18)]),
    dict(key='santa_ynez', title='Santa Ynez River', huc8='18060010',
         restored_pat=r'^santa ynez river', nwis_prefix='SANTA YNEZ R ',
         dams=[('Gibraltar', 1920, 34.5269, -119.6883, False),
               ('Juncal', 1930, 34.4929, -119.5083, False),
               ('Bradbury (Cachuma)', 1953, 34.5872, -119.9815, True)],
         dam_label_offsets=[(-10, 14), (0, -16), (-20, 14)]),
]

ABBR = {'nr': 'near', 'a': 'at', 'bl': 'below', 'ab': 'above', 'blw': 'below',
        'abv': 'above', 'cyn': 'canyon', 'ave': 'avenue', 'hwy': 'highway',
        'st': 'street', 'rd': 'road', 'c': 'creek', 'laurls': 'laureles',
        'snta': 'santa'}
POS = {'near', 'at', 'below', 'above', 'in', 'the', 'of', 'ca', 'calif', 'california',
       'cal', 'and'}


def loc_tokens(s):
    """Significant location words after the river name, normalized."""
    toks = [ABBR.get(t, t) for t in re.findall(r'[a-z0-9]+', s.lower().replace("'", ''))]
    return tuple(t for t in toks if t not in POS)


def years_covered(df):
    """(annual, summary) year sets. Annual = tables spanning <= ANNUAL_SPAN years
    (one water year per table, the usual Water Supply Paper layout); summary =
    years spanned by multi-year compilation tables."""
    annual, summary = set(), set()
    for a, b in zip(df['year_start'], df['year_end'].fillna(df['year_start'])):
        if pd.isna(a):
            continue
        a, b = int(a), int(min(max(a, b), YEAR_MAX))
        (annual if b - a <= ANNUAL_SPAN else summary).update(range(a, b + 1))
    keep = lambda ys: sorted(y for y in ys if YEAR_MIN <= y <= YEAR_MAX)
    return keep(annual), keep(summary - annual)


tl = pd.read_csv(root / 'data/analysis/table_level_metadata_published.csv', low_memory=False)
tl = tl[(tl['water_type_final'] == 'Stream Discharge') & tl['year_start'].notna()]
nw = pd.read_csv(root / 'data/analysis/nwis_sites/ca_stream_sites_expanded.csv',
                 low_memory=False).drop_duplicates('site_no')
nw = nw[nw['begin_year'] <= NWIS_YEAR_CUTOFF]
huc8 = gpd.read_file(spatial / 'ca_wbd/ca_huc8.geojson')


def build_stations(cfg):
    # ---- NWIS stations on this river
    n = nw[nw['station_nm'].str.startswith(cfg['nwis_prefix'])].copy()
    n = n[~n['station_nm'].str.lower().str.contains(EXCLUDE)]
    n['loc'] = n['station_nm'].str[len(cfg['nwis_prefix']):].map(loc_tokens)

    # ---- restored tables on this river, grouped by location words
    w = tl['watersource_name'].fillna('').str.lower().str.strip()
    r = tl[w.str.contains(cfg['restored_pat']) & ~w.str.contains(EXCLUDE)].copy()
    r['loc'] = r['watersource_name'].str.lower().str.replace(
        cfg['restored_pat'].lstrip('^'), '', regex=True).map(loc_tokens)

    rows = []
    used_nwis = set()
    for loc, g in r.groupby('loc'):
        # pair: identical location words, or restored words contained in NWIS words
        match = n[n['loc'].map(lambda t: len(loc) > 0 and (t == loc or set(loc) <= set(t)))]
        if len(match) > 1:   # prefer exact, then earliest
            exact = match[match['loc'] == loc]
            match = exact if len(exact) else match
        m = match.sort_values('begin_year').head(1)
        annual, summ = years_covered(g)
        row = dict(label=' '.join(loc) if loc else '(location not named)',
                   r_annual=annual, r_summary=summ, r_years=sorted(set(annual) | set(summ)),
                   r_tables=len(g),
                   r_meas=int(g['number_rows'].sum()),
                   lat=g['lat_combined'].median(), lon=g['lon_combined'].median(),
                   paired=len(m) > 0, n_begin=np.nan, n_end=np.nan)
        if len(m):
            m = m.iloc[0]
            used_nwis.add(m['site_no'])
            row.update(n_begin=m['begin_year'], n_end=min(m['end_year'], YEAR_MAX),
                       lat=m['dec_lat_va'], lon=m['dec_long_va'], site_no=m['site_no'])
        if not loc:
            row.update(lat=np.nan, lon=np.nan)
        rows.append(row)
    for _, m in n[~n['site_no'].isin(used_nwis)].iterrows():   # NWIS-only stations
        rows.append(dict(label=' '.join(m['loc']), r_years=[], r_annual=[], r_summary=[],
                         r_tables=0, r_meas=0,
                         lat=m['dec_lat_va'], lon=m['dec_long_va'], paired=False,
                         n_begin=m['begin_year'], n_end=min(m['end_year'], YEAR_MAX),
                         site_no=m['site_no']))

    st = pd.DataFrame(rows)
    if 'site_no' not in st:
        st['site_no'] = np.nan
    # merge restored name variants that pair to the same NWIS gauge
    nwis_label = dict(zip(n['site_no'], n['loc'].map(' '.join)))
    st['key'] = st['site_no'].where(st['site_no'].notna(), 'r:' + st['label'])
    st = st.groupby('key', sort=False).agg(
        label=('label', 'first'), r_years=('r_years', lambda x: sorted(set().union(*x))),
        r_annual=('r_annual', lambda x: sorted(set().union(*x))),
        r_summary=('r_summary', lambda x: sorted(set().union(*x))),
        r_tables=('r_tables', 'sum'), r_meas=('r_meas', 'sum'), lat=('lat', 'first'),
        lon=('lon', 'first'), n_begin=('n_begin', 'first'), n_end=('n_end', 'first'),
        site_no=('site_no', 'first')).reset_index(drop=True)
    st.loc[st['site_no'].notna(), 'label'] = st['site_no'].map(nwis_label)
    st['unnamed'] = st['label'] == '(location not named)'

    st['first'] = [min([b] + y[:1]) if pd.notna(b) else (y[0] if y else 9999)
                   for b, y in zip(st['n_begin'], st['r_years'])]
    # keep the stations with the most restored data, plus every NWIS station that existed pre-dam
    main_dam_yr = [d[1] for d in cfg['dams'] if d[4]][0]
    keep = ((st['r_meas'].rank(ascending=False, method='first') <= MAX_ROWS) |
            (st['n_begin'] < main_dam_yr))
    st = st[keep].sort_values(['unnamed', 'first']).reset_index(drop=True)
    st['r_summary'] = [sorted(set(sm) - set(an)) for sm, an in zip(st['r_summary'], st['r_annual'])]
    st['label'] = st['label'].str.title().str.replace('Mwd', 'MWD')
    st.loc[st['unnamed'], 'label'] = 'Station not named in source'
    st['num'] = np.where(st['unnamed'], 0, np.arange(1, len(st) + 1))
    return st, r, n


def draw_map(ax, cfg, st):
    shed = huc8[huc8['huc8'] == cfg['huc8']]
    fl = gpd.read_file(spatial / f"nhd/{cfg['key']}_flowlines.geojson")
    wb = gpd.read_file(spatial / f"nhd/{cfg['key']}_waterbodies.geojson")
    fl = gpd.clip(fl, shed)
    wb = gpd.clip(wb, shed)
    shed.plot(ax=ax, color='#F4F4F4', edgecolor='#888888', linewidth=0.8, zorder=1)
    for order, lw in [(3, 0.3), (4, 0.6), (5, 1.0), (6, 1.5), (7, 2.0)]:
        sub = fl[fl['StreamOrde'] == order] if order < 7 else fl[fl['StreamOrde'] >= 7]
        if len(sub):
            sub.plot(ax=ax, color='#9DB9D3', linewidth=lw, zorder=2)
    main = fl[fl['GNIS_NAME'].fillna('').str.lower() == cfg['title'].lower()]
    main.plot(ax=ax, color='#5B8DB8', linewidth=2.0, zorder=3)
    if len(wb):
        wb.plot(ax=ax, color='#9DB9D3', edgecolor='none', zorder=3)


    xmin, ymin, xmax, ymax = shed.total_bounds
    st = st[~st['unnamed']]
    inside = st[st['lat'].between(ymin - 0.05, ymax + 0.05) & st['lon'].between(xmin - 0.05, xmax + 0.05)]
    for _, s in inside.iterrows():
        has_r, has_n = len(s['r_years']) > 0, pd.notna(s['n_begin'])
        face = '#AA3377' if (has_r and has_n) else (RESTORED_COLOR if has_r else NWIS_COLOR)
        hollow = has_r and not has_n
        ax.plot(s['lon'], s['lat'], 'o', markersize=11, zorder=8,
                markerfacecolor='white' if hollow else face, markeredgecolor=face,
                markeredgewidth=1.6)
        ax.text(s['lon'], s['lat'], str(s['num']), fontsize=6.5, ha='center', va='center',
                color=face if hollow else 'white', fontweight='bold', zorder=9)
    for (name, yr, lat, lon, is_main), off in zip(cfg['dams'], cfg['dam_label_offsets']):
        ax.plot(lon, lat, marker='^', markersize=12 if is_main else 9, color=DAM_COLOR,
                markeredgecolor='white', markeredgewidth=1.0, zorder=10)
        ax.annotate(f'{name} {yr}', (lon, lat), xytext=off, textcoords='offset points',
                    fontsize=7.5, fontweight='bold' if is_main else 'normal', zorder=11,
                    ha='center', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    pad = 0.04
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect(1 / np.cos(np.radians((ymin + ymax) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#cccccc')
    ax.set_title(cfg['title'], fontsize=12, loc='left', fontweight='bold')


def draw_timeline(ax, cfg, st):
    n = len(st)
    for i, s in st.iterrows():
        y = n - 1 - i
        if pd.notna(s['n_begin']):        # NWIS lane (upper)
            ax.barh(y + 0.17, s['n_end'] - s['n_begin'] + 1, left=s['n_begin'], height=0.28,
                    color=NWIS_COLOR, edgecolor='none', zorder=2)
        for yr in s['r_summary']:          # restored lane (lower): multi-year tables, thin
            ax.add_patch(plt.Rectangle((yr, y - 0.2), 1, 0.08, color=RESTORED_COLOR,
                                       alpha=0.55, linewidth=0, zorder=3))
        for yr in s['r_annual']:           # restored lane: annual tables, full height
            ax.add_patch(plt.Rectangle((yr + 0.08, y - 0.31), 0.84, 0.3, color=RESTORED_COLOR,
                                       linewidth=0, zorder=3))
    for name, yr, _, _, is_main in cfg['dams']:
        ax.axvline(yr, color=DAM_COLOR, linewidth=1.6 if is_main else 0.8,
                   linestyle='-' if is_main else '--', zorder=4)
        ax.text(yr + 0.6, n - 0.35, name.split(' ')[0], fontsize=7, rotation=90,
                va='top', ha='left', color=DAM_COLOR, zorder=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s['label'] if s['unnamed'] else f"{s['num']}  {s['label']}"
                        for _, s in st.iloc[::-1].iterrows()],
                       fontsize=7.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(YEAR_MIN, YEAR_MAX + 1)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_xlabel('Year')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


fig = plt.figure(figsize=(14, 5.6 * len(RIVERS)), dpi=300)
gs = fig.add_gridspec(len(RIVERS), 2, width_ratios=[1, 1.45], wspace=0.28, hspace=0.25)
summary = []
for i, cfg in enumerate(RIVERS):
    st, r, n = build_stations(cfg)
    draw_map(fig.add_subplot(gs[i, 0]), cfg, st)
    draw_timeline(fig.add_subplot(gs[i, 1]), cfg, st)
    main_yr = [d[1] for d in cfg['dams'] if d[4]][0]
    r_years_all = sorted(set(y for ys in st['r_years'] for y in ys))
    nwis_first = n['begin_year'].min()
    summary.append(dict(river=cfg['title'], stations_shown=len(st),
                        restored_tables=len(r), restored_meas=int(r['number_rows'].sum()),
                        restored_first=min(r_years_all), nwis_first=nwis_first,
                        restored_years_before_nwis=sum(y < nwis_first for y in r_years_all),
                        restored_predam_years=sum(y < main_yr for y in r_years_all)))
    print(f"\n=== {cfg['title']} ===")
    print(st[['num', 'label', 'r_tables', 'r_meas', 'n_begin', 'n_end']].assign(
        r_first=st['r_years'].map(lambda y: y[0] if y else None),
        r_nyears=st['r_years'].map(len), r_annual_yrs=st['r_annual'].map(len),
        r_annual_first=st['r_annual'].map(lambda y: y[0] if y else None)).to_string(index=False))
    span = (r['year_end'].fillna(r['year_start']) - r['year_start'])
    print(f'  tables: {len(r)}; annual (span<={ANNUAL_SPAN}): {(span <= ANNUAL_SPAN).mean():.0%}')

legend = [Patch(color=NWIS_COLOR, label='NWIS period of record (to 1980)'),
          Patch(color=RESTORED_COLOR, label='Restored: annual table for that year'),
          Patch(color=RESTORED_COLOR, alpha=0.55, label='Restored: within a multi-year table'),
          Line2D([], [], marker='o', color='w', markerfacecolor='#AA3377', markersize=9,
                 label='Station in both'),
          Line2D([], [], marker='o', color='w', markerfacecolor='white',
                 markeredgecolor=RESTORED_COLOR, markeredgewidth=1.6, markersize=9,
                 label='Restored only (approx. location)'),
          Line2D([], [], marker='^', color='w', markerfacecolor=DAM_COLOR, markersize=10,
                 label='Dam (year completed)')]
fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

print('\n', pd.DataFrame(summary).to_string(index=False))
out = figures_path / 'dam_vignette_timeline.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'\nSaved: {out}')
