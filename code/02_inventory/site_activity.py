"""
Shared logic for restored and NWIS site activity (import; do not copy). Used by Figure 3
(fig3_nwis_comparison.py) and the Santa Ynez figure (code/04_vignettes/santa_ynez/fig_santa_ynez.py).
Moved here unchanged from fig3_nwis_comparison.py on 2026-11-14.

Restored records = the filtered entry set of 21_site_definitions.py (entry_site_keys.csv): California,
precise/approximate rebuilt location, measurement dates only, pages with measurements, tribal pages
out, reviewer decisions applied. A record's active decades = decades of the years printed in the rows
of its matched tables (table_row_dates.parquet), kept within its year_start..year_end; records whose
tables have no row years (daily tables) use year_start..year_end. Capped at CAP (1979).

NWIS: discharge-related series (stream: dv/uv 00060, uv 00065, peaks, field measurements; groundwater:
water levels), sites with a series beginning <= CAP. Active decades = decades with actual NWIS data
(nwis_years_{stream,gw}.csv, 2026-11-15) where downloaded, else the union of the series' begin..end spans.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
CAP = 1979
STREAM_SERIES = {('dv', '00060'), ('uv', '00060'), ('uv', '00065'), ('pk', None), ('sv', None)}
GW_SERIES = {('gw', '72019'), ('gw', '62610'), ('gw', '62611')}   # same water levels, different units


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def decades_of(y0, y1):
    """Set of decades covered by the years y0..y1 (capped at CAP)."""
    y1 = min(y1, CAP)
    return set(range(int(y0) // 10 * 10, int(y1) // 10 * 10 + 1, 10)) if y0 <= y1 else set()


def record_decades(y0, y1, printed):
    y1 = min(y1, CAP)
    if isinstance(printed, set):
        ys = {y for y in printed if y0 <= y <= y1}
        if ys:
            return {y // 10 * 10 for y in ys}
    return decades_of(y0, y1)


def load_records():
    """Filtered restored records with printed table years and groundwater NWIS well links."""
    keys = pd.read_csv(A / 'site_definitions/entry_site_keys.csv')
    pm = pd.read_parquet(A / 'processed_metadata.parquet',
                         columns=['meta_row', 'year_start', 'year_end', 'lat_combined', 'lon_combined'])
    rec = keys.merge(pm, on='meta_row', how='left')
    rec = rec[rec.year_start.notna() & (rec.year_start <= CAP)].copy()
    rec['year_end'] = rec.year_end.fillna(rec.year_start).clip(lower=rec.year_start)

    rd = pd.read_parquet(A / 'table_row_dates.parquet',
                         columns=['id', 'page_number', 'table_number', 'year_min', 'year_max'])
    rd['id'] = rd['id'].astype(str)
    yrs = pd.concat([rd[['id', 'page_number', 'table_number', 'year_min']].rename(columns={'year_min': 'y'}),
                     rd[['id', 'page_number', 'table_number', 'year_max']].rename(columns={'year_max': 'y'})])
    yrs = yrs[(yrs.y >= 1800) & (yrs.y <= CAP)].drop_duplicates()
    table_years = yrs.groupby(['id', 'page_number', 'table_number']).y.apply(set).to_dict()
    ent = pd.read_csv(A / 'entry_dates.csv', usecols=['meta_row', 'matched_tables', 'date_source'],
                      low_memory=False)
    pid = pd.read_parquet(A / 'processed_metadata.parquet', columns=['meta_row', 'id', 'page_number'])
    ent = ent.merge(pid, on='meta_row', how='left')

    def printed_years(row):
        if pd.isna(row.matched_tables) or row.date_source == 'row' or pd.isna(row.id):
            return None
        doc, page = str(row.id).replace('.0', ''), int(row.page_number)
        out = set()
        for t in str(row.matched_tables).split(','):
            out |= table_years.get((doc, page, int(float(t))), set())
        return out or None

    ent['printed_years'] = [printed_years(r) for r in ent.itertuples(index=False)]
    rec = rec.merge(ent[['meta_row', 'printed_years']], on='meta_row', how='left')
    wn = pd.read_csv(A / 'location_validation/tier2b_well_numbers.csv', low_memory=False,
                     usecols=['meta_row', 'nwis_site_no'], dtype={'nwis_site_no': str})
    wn = wn.dropna().drop_duplicates('meta_row').set_index('meta_row').nwis_site_no
    rec['gw_nwis'] = rec.meta_row.map(wn)
    return rec


def nwis_series_decades(raw_csv, keep):
    r = pd.read_csv(A / 'nwis_sites' / raw_csv, low_memory=False, dtype={'site_no': str, 'parm_cd': str},
                    usecols=['site_no', 'data_type_cd', 'parm_cd', 'begin_date', 'end_date'])
    # missing parameter code (peaks, field measurements) -> None; pandas .where() would leave NaN
    key = [(t.strip(), p if isinstance(p, str) else None) for t, p in zip(r.data_type_cd, r.parm_cd)]
    r = r[[k in keep for k in key]].copy()
    r['b'] = pd.to_datetime(r.begin_date, errors='coerce').dt.year
    r['e'] = pd.to_datetime(r.end_date, errors='coerce').dt.year.fillna(r['b'])
    r = r[r.b.notna() & (r.b <= CAP)]
    r['dec'] = [decades_of(b, e) for b, e in zip(r.b, r.e)]
    return r.groupby('site_no').agg(first_year=('b', 'min'), dec=('dec', lambda x: set().union(*x)))


def nwis_actual_decades(kind):
    """site_no -> decades with at least one year of NWIS data <= CAP, from the downloaded years
    (code/03_nwis_usgs/fetch_nwis_years_with_data.py; stream: peaks + daily discharge + field
    measurements; groundwater: water-level field measurements). None if not downloaded yet."""
    f = A / 'nwis_sites' / f'nwis_years_{kind}.csv'
    if not f.exists():
        return None
    y = pd.read_csv(f, dtype={'site_no': str})
    y = y[y.year <= CAP]
    return y.groupby('site_no').year.apply(lambda s: {int(v) // 10 * 10 for v in s})


def nwis_table(csv):
    """NWIS sites with a relevant record beginning <= CAP. Site list and first year from the series
    catalogue; active decades from the ACTUAL years with data where downloaded (2026-11-15: 154 of
    2,094 stream sites have gap decades inside their begin-end span, e.g. historic flood peaks
    followed by decades without data), else the series spans (decades_source column)."""
    kind = 'stream' if 'stream' in csv else 'gw'
    raw, keep = (('ca_stream_sites_expanded_raw.csv', STREAM_SERIES) if kind == 'stream'
                 else ('ca_groundwater_sites_raw.csv', GW_SERIES))
    n = pd.read_csv(A / 'nwis_sites' / csv, low_memory=False, dtype={'site_no': str})
    n = n[n.dec_lat_va.notna()].drop_duplicates('site_no').set_index('site_no')
    sd = nwis_series_decades(raw, keep)
    n = n.join(sd, how='inner')                       # sites with a relevant record beginning <= CAP
    act = nwis_actual_decades(kind)
    if act is None:
        n['decades'], n['decades_source'] = n['dec'], 'series span'
    else:
        got = n.index.isin(act.index)
        n['decades'] = [act[s] if g else d for s, g, d in zip(n.index, got, n['dec'])]
        n['decades_source'] = np.where(got, 'actual years', 'series span')
    return n


def all_nwis_coords(csv):
    n = pd.read_csv(A / 'nwis_sites' / csv, low_memory=False, dtype={'site_no': str})
    return n.drop_duplicates('site_no').set_index('site_no')[['dec_lat_va', 'dec_long_va']]


def restored_sites(rec, wt, nwis_coords, site_key):
    """One row per restored site: decades active, location, linked NWIS site (or None)."""
    r = rec[rec.water_type == wt].copy()
    r['dec'] = [record_decades(a, b, p) for a, b, p in zip(r.year_start, r.year_end, r.printed_years)]
    s = r.groupby(site_key).agg(lat=('lat_combined', 'mean'), lon=('lon_combined', 'mean'),
                                dec=('dec', lambda x: set().union(*x)),
                                first_year=('year_start', 'min'),
                                gw_nwis=('gw_nwis', lambda x: x.dropna().iloc[0] if x.notna().any() else None))
    s['nwis_link'] = None
    if wt == 'Stream Discharge':
        st = s.index.to_series().str.extract(r'^nwis:([^#]+)#')[0]
        s['station'] = st
        for stn, g in s[st.notna()].groupby('station'):
            if stn in nwis_coords.index:            # nearest cluster of this station is the linked one
                la, lo = nwis_coords.loc[stn, ['dec_lat_va', 'dec_long_va']]
                d = hav_km(g.lat.values, g.lon.values, la, lo)
                s.loc[g.index[int(np.nanargmin(d))], 'nwis_link'] = stn
            else:
                s.loc[g.index[0], 'nwis_link'] = stn
    else:
        s['nwis_link'] = s.gw_nwis
    return s[s.dec.map(len) > 0]
