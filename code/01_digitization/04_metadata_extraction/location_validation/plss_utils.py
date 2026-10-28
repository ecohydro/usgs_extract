"""
Shared BLM PLSS (CadNSDI) helpers: township and section centroids, cached in
data/analysis/location_validation/ (plss_townships.parquet, plss_sections_cache.parquet).
Same caches as 02_trs_lookup_all.py.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape

root    = Path(__file__).resolve().parents[4]
out_dir = root / 'data/analysis/location_validation'
PLSS = 'https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer'

MERIDIAN_LETTER = {'M': 'Mount Diablo', 'S': 'San Bernardino', 'H': 'Humboldt'}


def blm_query(layer, params):
    url = f'{PLSS}/{layer}/query?' + urllib.parse.urlencode({**params, 'f': 'geojson', 'outSR': 4326})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r).get('features', [])
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'BLM query failed: {params.get("where")}')


def to_centroids(feats, keys):
    rows = []
    for f in feats:
        g = shape(f['geometry'])
        c = g.centroid if g.is_valid else g.representative_point()
        rows.append({**{k: f['properties'][k] for k in keys}, 'lat': c.y, 'lon': c.x})
    return pd.DataFrame(rows)


def townships(states=('CA',)):
    """Whole-township centroids (no fractional townships) with a (t, td, r, rd) key."""
    path = out_dir / 'plss_townships.parquet'
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['STATEABBR'])
    for st in [s for s in states if s not in set(have['STATEABBR'])]:
        feats, offset = [], 0
        while True:
            batch = blm_query(1, {'where': f"STATEABBR='{st}'", 'returnGeometry': 'true',
                                  'maxAllowableOffset': 0.001, 'resultOffset': offset,
                                  'resultRecordCount': 1000,
                                  'outFields': 'PLSSID,STATEABBR,PRINMER,TWNSHPNO,TWNSHPDIR,'
                                               'RANGENO,RANGEDIR,TWNSHPFRAC,RANGEFRAC'})
            feats += batch
            offset += len(batch)
            if len(batch) < 1000:
                break
        have = pd.concat([have, to_centroids(feats, ['PLSSID', 'STATEABBR', 'PRINMER', 'TWNSHPNO',
                                                     'TWNSHPDIR', 'RANGENO', 'RANGEDIR',
                                                     'TWNSHPFRAC', 'RANGEFRAC'])], ignore_index=True)
        have.to_parquet(path, index=False)
    tw = have[have['STATEABBR'].isin(states) & (have['TWNSHPFRAC'] == '0') & (have['RANGEFRAC'] == '0')].copy()
    tw['key'] = list(zip(tw['TWNSHPNO'].astype(int), tw['TWNSHPDIR'],
                         tw['RANGENO'].astype(int), tw['RANGEDIR']))
    return tw


def sections(plssids):
    """Section centroids for the given townships (cached). Index (PLSSID, section no.)."""
    path = out_dir / 'plss_sections_cache.parquet'
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['PLSSID', 'FRSTDIVNO', 'lat', 'lon'])
    need = sorted(set(plssids) - set(have['PLSSID']))
    for i in range(0, len(need), 40):          # 40 townships x 36 sections < 2,000 limit
        ids = ','.join(f"'{p}'" for p in need[i:i + 40])
        feats = blm_query(2, {'where': f"PLSSID IN ({ids}) AND FRSTDIVTYP='SN'",
                              'returnGeometry': 'true', 'maxAllowableOffset': 0.0005,
                              'outFields': 'PLSSID,FRSTDIVNO', 'resultRecordCount': 2000})
        have = pd.concat([have, to_centroids(feats, ['PLSSID', 'FRSTDIVNO'])], ignore_index=True)
        if (i // 40) % 10 == 0:
            print(f'  sections: {min(i + 40, len(need)):,}/{len(need):,} new townships')
            have.to_parquet(path, index=False)
    have.to_parquet(path, index=False)
    have['FRSTDIVNO'] = pd.to_numeric(have['FRSTDIVNO'], errors='coerce')
    return have.groupby(['PLSSID', 'FRSTDIVNO'])[['lat', 'lon']].first()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
