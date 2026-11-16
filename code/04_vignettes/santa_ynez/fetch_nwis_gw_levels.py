"""
NWIS groundwater-level field measurements (parameter 72019, depth to water below land surface, ft) for
individual wells, <= 1979, from the USGS Water Data API. Used to check digitized water levels for the
Figure 5 candidate wells. Cached per well: data/analysis/santa_ynez/nwis_gw_levels/{site_no}.csv

Usage: python fetch_nwis_gw_levels.py SITE_NO [SITE_NO ...]
The API key is read from USGS_API_KEY (environment or Windows user environment) and sent only in the
X-Api-Key header; it is never written to code, URLs or outputs.
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/analysis/santa_ynez/nwis_gw_levels'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items'


def api_key():
    k = os.environ.get('USGS_API_KEY')
    if not k and os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h:
                k = winreg.QueryValueEx(h, 'USGS_API_KEY')[0]
        except OSError:
            k = None
    return k


HDR = {'X-Api-Key': api_key()} if api_key() else {}


def fetch(site):
    f = OUT / f'{site}.csv'
    if f.exists():
        return pd.read_csv(f)
    params = {'f': 'json', 'monitoring_location_id': f'USGS-{site}', 'parameter_code': '72019',
              'datetime': '1800-01-01T00:00:00Z/1979-12-31T23:59:59Z', 'limit': 10000}
    recs, nxt = [], None
    while True:
        r = requests.get(nxt or URL, params=None if nxt else params, headers=HDR, timeout=120)
        r.raise_for_status()
        j = r.json()
        for p in j.get('features', []):
            q = p['properties']
            recs.append({'time': q.get('time'), 'value': q.get('value'), 'qualifier': q.get('qualifier'),
                         'approval': q.get('approval_status'), 'unit': q.get('unit_of_measure')})
        nxt = next((l['href'] for l in j.get('links', []) if l.get('rel') == 'next'), None)
        if not nxt:
            break
        time.sleep(0.5)
    d = pd.DataFrame(recs)
    d.to_csv(f, index=False)
    return d


if __name__ == '__main__':
    for s in sys.argv[1:]:
        d = fetch(s)
        print(s, len(d), 'measurements', d.time.min() if len(d) else '', d.time.max() if len(d) else '')
