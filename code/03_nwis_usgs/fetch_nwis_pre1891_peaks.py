"""
NWIS stream records before 1891 (Annette's check, 2026-11-14).

The NWIS site series lists (ca_stream_sites_expanded_raw.csv) show peak-flow (pk) series that begin
before 1891 at a few California stations. This script downloads the actual annual-peak records for
those stations from the NWIS peak-flow service and writes every peak before 1891, with the
qualification codes decoded from the service's own header (e.g. 2 = estimate, 7 = historic peak),
plus the station's first peak after 1890 and its number of peaks before 1980 for context.

Output: data/analysis/nwis_sites/nwis_stream_peaks_pre1891.csv
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
NS = ROOT / 'data/analysis/nwis_sites'
URL = 'https://nwis.waterdata.usgs.gov/nwis/peak'
CUTOFF = 1891

PEAK_CODES = {   # from the RDB header of the peak service ("Peak Streamflow-Qualification Codes")
    '1': 'maximum daily average', '2': 'estimate', '3': 'affected by dam failure',
    '4': 'less than indicated value (minimum recordable discharge)',
    '5': 'affected to unknown degree by regulation or diversion', '6': 'affected by regulation or diversion',
    '7': 'historic peak', '8': 'actually greater than indicated value',
    '9': 'snowmelt, hurricane, ice-jam or debris dam breakup', 'A': 'year of occurrence unknown or not exact',
    'Bd': 'day of occurrence unknown or not exact', 'Bm': 'month of occurrence unknown or not exact',
    'C': 'urbanization, mining, agricultural changes, channelization, or other',
    'F': 'peak supplied by another agency', 'O': 'opportunistic value, not systematic data collection',
    'R': 'revised'}
GH_CODES = {'1': 'backwater', '2': 'not the maximum for the year', '3': 'different site and/or datum',
            '4': 'below minimum recordable elevation', '5': 'estimate', '6': 'datum changed this year',
            '7': 'debris, mud or hyper-concentrated flow', '8': 'tidally affected'}


def decode(codes, table):
    if pd.isna(codes) or not str(codes).strip():
        return ''
    return '; '.join(table.get(c.strip(), c.strip()) for c in str(codes).split(','))


raw = pd.read_csv(NS / 'ca_stream_sites_expanded_raw.csv', low_memory=False, dtype={'site_no': str})
raw['begin'] = pd.to_datetime(raw.begin_date, errors='coerce')
early = raw[(raw.begin.dt.year < CUTOFF) & raw.data_type_cd.str.strip().isin(['pk', 'dv', 'uv', 'sv'])]
sites = early.drop_duplicates('site_no').set_index('site_no').station_nm
print(f'{len(sites)} stations with a discharge-related series beginning before {CUTOFF}')

rows = []
for site, name in sites.items():
    r = requests.get(URL, params={'site_no': site, 'agency_cd': 'USGS', 'format': 'rdb'},
                     headers={'User-Agent': 'python-requests'}, timeout=60)
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if not l.startswith('#')]
    df = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[2:])), sep='\t', dtype=str)
    df['year'] = pd.to_numeric(df.peak_dt.str[:4], errors='coerce')
    pre = df[df.year < CUTOFF].copy()
    after = df[df.year >= CUTOFF].sort_values('peak_dt')
    pre['station_nm'] = name
    pre['first_peak_after_1890'] = after.peak_dt.iloc[0] if len(after) else ''
    pre['n_peaks_before_1980'] = int((df.year < 1980).sum())
    rows.append(pre)
    time.sleep(0.5)

out = pd.concat(rows, ignore_index=True)
out['peak_va_cfs'] = pd.to_numeric(out.peak_va, errors='coerce')
out['peak_cd_meaning'] = out.peak_cd.map(lambda c: decode(c, PEAK_CODES))
out['gage_ht_cd_meaning'] = out.gage_ht_cd.map(lambda c: decode(c, GH_CODES))
cols = ['site_no', 'station_nm', 'peak_dt', 'peak_va_cfs', 'peak_cd', 'peak_cd_meaning', 'gage_ht',
        'gage_ht_cd', 'gage_ht_cd_meaning', 'year_last_pk', 'first_peak_after_1890', 'n_peaks_before_1980']
out = out[cols].sort_values(['peak_dt', 'site_no'])
out.to_csv(NS / 'nwis_stream_peaks_pre1891.csv', index=False)
print(out.to_string(index=False))
