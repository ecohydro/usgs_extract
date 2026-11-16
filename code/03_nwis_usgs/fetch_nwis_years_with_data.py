"""
Years in which each NWIS California site actually has data, through 1979 (2026-11-14).

Figure 3 had treated each NWIS record as continuous from its first to its last date, so gaps inside
NWIS records were invisible (16% of pre-1980 daily-discharge and peak series cover < 80% of their
span). This script downloads the actual years with data:

  stream  annual peaks       NWIS peak-flow service, one request per station with a peak (pk) series
                             beginning <= 1979  (nwis.waterdata.usgs.gov/nwis/peak, rdb)
  stream  daily discharge    NWIS daily-values service (waterservices.usgs.gov/nwis/dv), days with data
                             per year, 10 stations per request (the statistics service omits older
                             daily data at some sites, e.g. Solvang 11128500)
  stream  field measurements USGS Water Data API, collection field-measurements, parameter 00060
  gw      water levels       USGS Water Data API, field-measurements, parameter 72019 (depth to water)
                             (the old waterservices gwlevels service was retired in fall 2025)
The Water Data API is queried by 0.5-degree tiles over California, datetime <= 1979-12-31, following the
API's 'next' paging links.

Every response is cached under data/analysis/nwis_sites/years_cache/ (re-running resumes).
Outputs: data/analysis/nwis_sites/nwis_years_stream.csv  (site_no, year, source: pk / dv / fm)
         data/analysis/nwis_sites/nwis_years_gw.csv      (site_no, year, n_measurements)
"""
import io
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
NS = ROOT / 'data/analysis/nwis_sites'
CACHE = NS / 'years_cache'
for sub in ('peaks', 'dv', 'fm'):
    (CACHE / sub).mkdir(parents=True, exist_ok=True)
CAP = 1979
UA = {'User-Agent': 'python-requests'}
S = requests.Session()
S.headers.update(UA)


def _api_key():
    """USGS Water Data API key (api.data.gov; raises the hourly rate limit). Read from the
    USGS_API_KEY environment variable, or on Windows from the user environment saved by setx.
    Never written into code or outputs."""
    import os
    k = os.environ.get('USGS_API_KEY')
    if not k and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
                k = winreg.QueryValueEx(h, 'USGS_API_KEY')[0]
        except OSError:
            k = None
    return k


API_KEY = _api_key()
print(f'USGS Water Data API key: {"found" if API_KEY else "none (default rate limits)"}', flush=True)


def get(url, params=None, tries=8):
    for i in range(tries):
        try:
            # key goes in a header (not the URL) so it never appears in logs or error messages
            hdr = {'X-Api-Key': API_KEY} if API_KEY and 'api.waterdata.usgs.gov' in url else None
            r = S.get(url, params=params, headers=hdr, timeout=120)
            if r.status_code == 429:                 # rate limited (Water Data API, no key): back off
                wait = int(r.headers.get('Retry-After', 0)) or 60 * (i + 1)
                print(f'    rate limited; waiting {wait} s', flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def rdb(text):
    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
    if len(lines) < 3:
        return pd.DataFrame()
    d = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[2:])), sep='\t', dtype=str)
    # multi-site responses repeat the header + format lines per time series: keep data rows only
    return d[d.iloc[:, 0].eq('USGS')] if len(d) else d


# ---------------------------------------------------------------- site lists
raw = pd.read_csv(NS / 'ca_stream_sites_expanded_raw.csv', low_memory=False, dtype={'site_no': str, 'parm_cd': str})
raw['b'] = pd.to_datetime(raw.begin_date, errors='coerce').dt.year
raw['t'] = raw.data_type_cd.str.strip()
pk_sites = sorted(set(raw[(raw.t == 'pk') & (raw.b <= CAP)].site_no))
dv_sites = sorted(set(raw[(raw.t == 'dv') & (raw.parm_cd == '00060') & (raw.b <= CAP)].site_no))
print(f'stream: {len(pk_sites)} peak series, {len(dv_sites)} daily-discharge series beginning <= {CAP}', flush=True)

# ---------------------------------------------------------------- 1. peaks
rows, estimated = [], []
for i, s in enumerate(pk_sites):
    f = CACHE / 'peaks' / f'{s}.rdb'
    if not f.exists():
        f.write_text(get('https://nwis.waterdata.usgs.gov/nwis/peak',
                         {'site_no': s, 'agency_cd': 'USGS', 'format': 'rdb'}), encoding='utf-8')
        time.sleep(0.3)
    d = rdb(f.read_text(encoding='utf-8'))
    if len(d):
        # ALL peaks count as NWIS data, including those USGS codes as estimates (2) or historic peaks (7)
        # (Annette, 2026-11-15: an exclusion was tried and dropped; it did not change the take-away).
        # Those are listed in nwis_years_estimated_peaks.csv for reporting only (not used to filter).
        cd = d.peak_cd.fillna('').astype(str) if 'peak_cd' in d else pd.Series('', index=d.index)
        est = cd.str.contains(r'(?:^|,)\s*[27]\s*(?:,|$)')
        yrs = pd.to_numeric(d.peak_dt.str[:4], errors='coerce')
        estimated += [(s, int(y), c) for y, c, e in zip(yrs, cd, est) if e and pd.notna(y) and y <= CAP]
        rows += [(s, int(y), 'pk') for y in yrs.dropna().unique() if y <= CAP]
    if i % 200 == 0:
        print(f'  peaks {i}/{len(pk_sites)}', flush=True)

# ---------------------------------------------------------------- 2. daily discharge (annual stats)
# The daily-values service itself, not the statistics service: statistics are computed from one
# time series and can omit older daily data (11128500 Solvang: statistics start 1951, daily values
# exist from 1928). Cache = days with data per site and year.
for i in range(0, len(dv_sites), 10):
    batch = dv_sites[i:i + 10]
    f = CACHE / 'dv' / f'batch_{i:05d}.csv'
    if not f.exists():
        text = get('https://waterservices.usgs.gov/nwis/dv/',
                   {'format': 'rdb', 'sites': ','.join(batch), 'parameterCd': '00060',
                    'startDT': '1800-01-01', 'endDT': f'{CAP}-12-31'})
        # sites differ in column count, so read lines directly: agency, site_no, date are always first
        recs = [(p[1], p[2][:4]) for p in (l.split('\t') for l in text.splitlines() if l.startswith('USGS\t'))]
        d = pd.DataFrame(recs, columns=['site_no', 'year'])
        d.groupby(['site_no', 'year']).size().rename('n_days').reset_index().to_csv(f, index=False)
        time.sleep(0.3)
    d = pd.read_csv(f, dtype={'site_no': str})
    rows += [(s, int(y), 'dv') for s, y in zip(d.site_no, d.year) if int(y) <= CAP]
    if i % 200 == 0:
        print(f'  daily {i}/{len(dv_sites)}', flush=True)


# ---------------------------------------------------------------- 3. Water Data API field measurements
FM_URL = 'https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items'
FAILED = []                                            # tiles the server would not answer (not cached)


def fetch_bbox(param, lon, lat, size, depth=0, y0=1800, y1=CAP):
    """Records in one bbox and year window (never past CAP = 1979). The server cancels slow queries with
    HTTP 400 'Long running query has been cancelled' (seen 2026-11-14 on tiles -119.5,40.5 and -116,34):
    retry once, then split the YEAR window in two (2026-11-15: the cancellations follow the number of
    records, not the area); only a single year that still fails is split by area."""
    params = {'f': 'json', 'bbox': f'{lon},{lat},{lon + size},{lat + size}',
              'parameter_code': param, 'datetime': f'{y0}-01-01T00:00:00Z/{y1}-12-31T23:59:59Z',
              'limit': 10000, 'properties': 'monitoring_location_id,time'}
    for attempt in range(2):
        try:
            recs, nxt = [], None
            while True:
                j = json.loads(get(nxt) if nxt else get(FM_URL, params))
                recs += [(p['properties']['monitoring_location_id'], p['properties']['time'][:4])
                         for p in j.get('features', [])]
                nxt = next((l['href'] for l in j.get('links', []) if l.get('rel') == 'next'), None)
                if not nxt:
                    return recs
                time.sleep(0.3)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 400:
                raise
            print(f'    server cancelled {params["bbox"]} {y0}-{y1} (attempt {attempt + 1})', flush=True)
            time.sleep(15)
    if y1 > y0:                                       # split the years: 1800-1889 / 1890-1979, ...
        mid = (y0 + y1) // 2
        parts = [fetch_bbox(param, lon, lat, size, depth, y0, mid),
                 fetch_bbox(param, lon, lat, size, depth, mid + 1, y1)]
    elif depth < 3:                                   # one year still too slow: split the area
        h = size / 2
        parts = [fetch_bbox(param, lon + dx, lat + dy, h, depth + 1, y0, y1) for dx in (0, h) for dy in (0, h)]
    else:
        return None
    return None if any(p is None for p in parts) else [r for p in parts for r in p]


def field_measurements(param):
    """All field measurements of `param` <= 1979 in California tiles -> DataFrame(site_no, year)."""
    out = []
    for lon in np.arange(-124.5, -114.0, 0.5):
        # tiles start at 32.5 and the last one is 41.5-42.0 (California's northern border is 42 N;
        # the 42.0-42.5 tiles are Oregon and the server kept cancelling them, 2026-11-15)
        for lat in np.arange(32.5, 41.6, 0.5):
            f = CACHE / 'fm' / f'{param}_{lon:.1f}_{lat:.1f}.csv'
            if not f.exists():
                recs = fetch_bbox(param, float(lon), float(lat), 0.5)
                if recs is None:
                    FAILED.append((param, round(float(lon), 1), round(float(lat), 1)))
                    print(f'    FAILED tile {param} {lon:.1f} {lat:.1f} (not cached; re-run retries it)', flush=True)
                    continue
                pd.DataFrame(recs, columns=['id', 'year']).to_csv(f, index=False)
                time.sleep(2)                          # slower pacing for the Water Data API
            d = pd.read_csv(f, dtype=str)
            if len(d):
                out.append(d)
        print(f'  {param} tiles: lon {lon:.1f} done', flush=True)
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['id', 'year'])
    d['site_no'] = d.id.str.replace('USGS-', '', regex=False)
    d['year'] = pd.to_numeric(d.year, errors='coerce')
    return d[d.year <= CAP]


fm_q = field_measurements('00060')
stream_sites = set(raw.site_no)
fm_q = fm_q[fm_q.site_no.isin(stream_sites)]          # keep the stream sites in our NWIS list
rows += [(s, int(y), 'fm') for s, y in fm_q[['site_no', 'year']].drop_duplicates().itertuples(index=False)]
ys = pd.DataFrame(rows, columns=['site_no', 'year', 'source']).drop_duplicates()
ys.to_csv(NS / 'nwis_years_stream.csv', index=False)
print(f'stream: {ys.site_no.nunique()} sites, {len(ys)} site-year-source rows', flush=True)
pd.DataFrame(estimated, columns=['site_no', 'year', 'peak_cd']).to_csv(NS / 'nwis_years_estimated_peaks.csv', index=False)
print(f'peaks coded estimate / historic (2, 7), kept, listed for reporting: {len(estimated)}', flush=True)
if '--stream-only' in sys.argv:
    sys.exit(0)

gw = field_measurements('72019')
g = gw.groupby(['site_no', 'year']).size().rename('n_measurements').reset_index()
g['year'] = g.year.astype(int)
g.to_csv(NS / 'nwis_years_gw.csv', index=False)
print(f'groundwater: {g.site_no.nunique()} wells, {len(g)} well-years', flush=True)
fl = NS / 'nwis_years_failed_tiles.csv'
pd.DataFrame(FAILED, columns=['param', 'lon', 'lat']).to_csv(fl, index=False)
print(f'failed tiles: {len(FAILED)} (listed in {fl.name}; outputs are INCOMPLETE if > 0 - re-run)', flush=True)
