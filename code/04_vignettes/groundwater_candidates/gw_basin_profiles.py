"""
Groundwater example candidates: one profile per basin (2026-11-13, exploratory).

For each candidate DWR Bulletin 118 basin:
  - map: basin outline, restored groundwater sites (precise + approximate rebuilt locations)
    colored by first measurement decade, NWIS groundwater sites with records beginning <= 1979
  - time series: sites with a record in each year, restored vs NWIS (1860-1980)
  - summary + example tables (printed to CSV for the profile pages)

Restored data: data/catalog/restored_tables_catalog.csv (published, tribal pages excluded),
water_type == Groundwater, b118_basin_id in the basin. A restored "site" = unique rounded
location (4 decimals). Time series use MEASUREMENT dates only (date tiers A-D: no period of
record / construction year / LLM fallbacks). NWIS = data/analysis/nwis_sites/ca_groundwater_sites.csv
(one row per data series; site span = min begin_year .. max end_year), spatially joined to
the basin polygon (DWR 2019 B118 layer, data/analysis/spatial/sgma/).

Outputs: data/analysis/gw_candidates/{key}_profile.png, gw_candidate_summary.csv,
gw_candidate_examples.csv, gw_candidate_by_decade.csv
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/analysis/gw_candidates'
OUT.mkdir(parents=True, exist_ok=True)

BASINS = {   # key -> (label, B118 ids)
    'santa_ynez': ('Santa Ynez River Valley (3-015)', ['3-015']),
    'antelope': ('Antelope Valley (6-044)', ['6-044']),
    'la_coastal': ('Coastal Plain of Los Angeles (4-011, all subbasins)',
                   ['4-011.01', '4-011.02', '4-011.03', '4-011.04']),
}
MEASUREMENT = {'table_date_column', 'table_text', 'table_caption', 'page_text'}
YEARS = np.arange(1860, 1981)

cat = pd.read_csv(ROOT / 'data/catalog/restored_tables_catalog.csv', low_memory=False)
gw = cat[cat.water_type.eq('Groundwater') & cat.page_type.eq('data')].copy()   # no index/contents pages

b118 = gpd.read_file(ROOT / 'data/analysis/spatial/sgma/sgma_2019_basin_prioritization.geojson')
b118['geometry'] = b118.geometry.make_valid().buffer(0)

nw = pd.read_csv(ROOT / 'data/analysis/nwis_sites/ca_groundwater_sites.csv',
                 dtype={'site_no': str}, low_memory=False)
nw = (nw.groupby('site_no')
        .agg(lat=('dec_lat_va', 'first'), lon=('dec_long_va', 'first'),
             begin=('begin_year', 'min'), end=('end_year', 'max'))
        .reset_index())
nw_g = gpd.GeoDataFrame(nw, geometry=gpd.points_from_xy(nw.lon, nw.lat), crs='EPSG:4326')


def active_per_year(spans):
    """spans: DataFrame with start, end -> count of sites active in each year of YEARS."""
    s = spans.start.to_numpy()[:, None]
    e = spans.end.to_numpy()[:, None]
    return ((YEARS >= s) & (YEARS <= e)).sum(axis=0)


summary, examples, decades = [], [], []
for key, (label, ids) in BASINS.items():
    poly = b118[b118.Basin_Subbasin_Number.isin(ids)].to_crs('EPSG:4326')
    shape = poly.union_all()

    t = gw[gw.b118_basin_id.isin(ids)].copy()
    t['measured'] = t.date_status.isin(MEASUREMENT)
    loc = t[t.precision_class.isin(['precise', 'approximate'])].copy()
    loc['site'] = loc.latitude.round(4).astype(str) + ',' + loc.longitude.round(4).astype(str)
    dated = loc[loc.measured & loc.year_start.notna()]
    sites = (dated.groupby('site')
                  .agg(start=('year_start', 'min'), end=('year_end', 'max'),
                       lat=('latitude', 'first'), lon=('longitude', 'first'))
                  .reset_index())
    sites['end'] = sites.end.fillna(sites.start)
    undated_sites = loc[~loc.site.isin(sites.site)].drop_duplicates('site')

    nwb = nw_g[nw_g.within(shape)]
    nw_pre = nwb[nwb.begin <= 1979].rename(columns={'begin': 'start'})
    nw_pre = nw_pre.assign(end=nw_pre.end.clip(upper=1980))

    r_ts, n_ts = active_per_year(sites), active_per_year(nw_pre)
    first_nwis = int(nwb.begin.min()) if len(nwb) else None

    # ---- overlap with NWIS at the same well: tables tied to one NWIS well (number on page,
    # matched well number, or nearest well <= 0.5 km), measurement-dated ----
    span = nw.set_index('site_no')[['begin', 'end']]
    tw = t[t.measured & t.year_start.notna() & t.nwis_site_no.notna()].copy()
    tw['nwis_site_no'] = tw.nwis_site_no.astype(str).str.split(';').str[0].str.replace(r'\.0$', '', regex=True)
    tw = tw.join(span, on='nwis_site_no', how='inner')
    ye = tw.year_end.fillna(tw.year_start)
    tw['overlap'] = np.select([ye < tw.begin, tw.year_start < tw.begin, tw.year_start > tw.end],
                              ['all before NWIS record', 'starts before NWIS record', 'after NWIS record'],
                              'inside NWIS record')
    ov = tw.overlap.value_counts()
    summary.append({
        'basin': label, 'tables': len(t), 'documents': t.doc_id.nunique(),
        'values_primary': int(t.n_values_primary.sum()),
        'tables_measurement_dated': int(t.measured.sum()),
        'tables_located': len(loc),
        'restored_sites_dated': len(sites), 'restored_sites_undated': len(undated_sites),
        'restored_first_year': int(sites.start.min()) if len(sites) else None,
        'restored_last_year': int(sites.end.max()) if len(sites) else None,
        'nwis_sites_all': len(nwb), 'nwis_sites_begin_le_1979': len(nw_pre),
        'nwis_first_year': first_nwis,
        'restored_sites_before_first_nwis': int((sites.start < first_nwis).sum()) if first_nwis else None,
        'tables_tied_to_nwis_well': len(tw),
        **{f'nwis_well_{k.replace(" ", "_")}': int(ov.get(k, 0)) for k in
           ['all before NWIS record', 'starts before NWIS record', 'inside NWIS record', 'after NWIS record']},
        'values_before_nwis_record': int(tw[tw.overlap.isin(['all before NWIS record', 'starts before NWIS record'])].n_values_primary.sum()),
        'peak_restored_active': int(r_ts.max()), 'peak_restored_year': int(YEARS[r_ts.argmax()]),
        'record_types': '; '.join(f'{k} {v}' for k, v in t.data_resolution.value_counts().items()),
        'parameters': '; '.join(f'{k} {v}' for k, v in t.primary_parameter.value_counts().head(4).items()),
    })
    for d in range(1860, 1980, 10):
        decades.append({'basin': label, 'decade': d,
                        'restored_sites_active': int(r_ts[(YEARS >= d) & (YEARS < d + 10)].max()),
                        'nwis_sites_active': int(n_ts[(YEARS >= d) & (YEARS < d + 10)].max()),
                        'restored_tables_starting': int(((t.year_start // 10 * 10) == d)[t.measured].sum())})

    # ---- example tables: earliest measured, most values, longest span ----
    m = t[t.measured & t.year_start.notna()].copy()
    m['span'] = m.year_end.fillna(m.year_start) - m.year_start
    picks = {'earliest': m.sort_values(['year_start', 'n_values_primary'], ascending=[True, False]).head(3),
             'most values': m.sort_values('n_values_primary', ascending=False).head(2),
             'longest span': m.sort_values(['span', 'n_values_primary'], ascending=False).head(2)}
    seen = set()
    for why, df in picks.items():
        for r in df.itertuples():
            k = (r.doc_id, r.page_number, r.table_number)
            if k in seen:
                continue
            seen.add(k)
            examples.append({'basin': label, 'why': why, 'doc_id': r.doc_id, 'page_number': r.page_number,
                             'table_number': r.table_number, 'years': f'{int(r.year_start)}-{int(r.year_end) if pd.notna(r.year_end) else int(r.year_start)}',
                             'date_status': r.date_status, 'record_type': r.data_resolution,
                             'parameter': r.primary_parameter, 'values': r.n_values_primary,
                             'coord_method': r.coord_method, 'precision': r.precision_class,
                             'doc_title': r.doc_title, 'year_published': r.year_published,
                             'caption': str(r.table_caption)[:200], 'table_csv': r.table_csv,
                             'page_png': r.page_png})

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.2), gridspec_kw={'height_ratios': [1.25, 1]})
    poly.boundary.plot(ax=ax1, color='black', lw=0.8)
    ax1.scatter(nw_pre.lon, nw_pre.lat, s=4, c='#4477aa', alpha=0.35, lw=0, label=f'NWIS wells, begin ≤1979 (n={len(nw_pre):,})')
    if len(undated_sites):
        ax1.scatter(undated_sites.longitude, undated_sites.latitude, s=10, facecolors='none',
                    edgecolors='#999999', lw=0.6, label=f'Restored, undated (n={len(undated_sites):,})')
    sc = ax1.scatter(sites.lon, sites.lat, s=14, c=sites.start, cmap='plasma', vmin=1900, vmax=1980,
                     edgecolors='black', lw=0.3, label=f'Restored, dated (n={len(sites):,})')
    cb = fig.colorbar(sc, ax=ax1, shrink=0.8, pad=0.01)
    cb.set_label('First measurement year (restored)')
    x0, y0, x1, y1 = poly.total_bounds
    ax1.set_xlim(x0 - 0.02, x1 + 0.02); ax1.set_ylim(y0 - 0.02, y1 + 0.02)
    ax1.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7, frameon=False)
    ax1.set_title('Well locations', fontsize=10)

    ax2.plot(YEARS, r_ts, color='#ee6677', lw=1.8, label='Restored sites')
    ax2.plot(YEARS, n_ts, color='#4477aa', lw=1.8, label='NWIS sites (records ≤1980)')
    ax2.set_xlim(1895, 1980)
    ax2.set_ylabel('Sites with a record in that year')
    ax2.set_title('Sites with records, by year', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc='upper left')
    fig.suptitle(label, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / f'{key}_profile.png', dpi=200)
    plt.close(fig)

pd.DataFrame(summary).to_csv(OUT / 'gw_candidate_summary.csv', index=False)
pd.DataFrame(examples).to_csv(OUT / 'gw_candidate_examples.csv', index=False)
pd.DataFrame(decades).to_csv(OUT / 'gw_candidate_by_decade.csv', index=False)
print(pd.DataFrame(summary).T.to_string())
