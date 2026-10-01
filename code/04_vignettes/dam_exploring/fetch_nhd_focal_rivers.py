"""
Download NHDPlus flowlines (stream order >= MIN_ORDER) and waterbodies for the
focal dam-vignette watersheds, for map context.

Source: USGS National Map NHD MapServer (hydro.nationalmap.gov/arcgis/rest/
services/nhd/MapServer), layer 4 = Flowline - Small Scale (NHDPlus attributes,
incl. StreamOrde), layer 10 = Waterbody - Small Scale. Queried by watershed
bounding box, simplified with maxAllowableOffset. Downloaded 2026-10-01.

Output: data/analysis/spatial/nhd/{key}_flowlines.geojson, {key}_waterbodies.geojson
"""

import json
import time
import requests
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out  = root / 'data/analysis/spatial/nhd'
out.mkdir(parents=True, exist_ok=True)

URL = 'https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/{}/query'
MIN_ORDER = 3

# xmin, ymin, xmax, ymax (deg); generous boxes around each HUC8 set
BASINS = {
    'santa_ana':  (-118.15, 33.55, -116.55, 34.40),
    'santa_ynez': (-120.70, 34.35, -119.40, 34.85),
}


def query(layer, bbox, where):
    feats, off = [], 0
    while True:
        for attempt in range(3):
            try:
                r = requests.get(URL.format(layer), timeout=180, params=dict(
                    where=where, geometry=','.join(map(str, bbox)),
                    geometryType='esriGeometryEnvelope', inSR=4326, outSR=4326,
                    spatialRel='esriSpatialRelIntersects', maxAllowableOffset=0.0005,
                    outFields='GNIS_NAME,FTYPE' + (',StreamOrde' if layer == 4 else ''),
                    resultOffset=off, resultRecordCount=1000, f='geojson'))
                fs = r.json()['features']
                break
            except Exception as e:
                print('  retry', layer, off, e)
                time.sleep(10)
        else:
            raise RuntimeError(f'NHD layer {layer} query failed at offset {off}')
        feats += fs
        if len(fs) < 1000:
            return feats
        off += 1000


LAYERS = {'flowlines':   (4,  f'StreamOrde >= {MIN_ORDER}'),
          'waterbodies': (10, "FTYPE = 'LakePond' OR FTYPE = 'Reservoir'")}

for key, bbox in BASINS.items():
    for name, (layer, where) in LAYERS.items():
        p = out / f'{key}_{name}.geojson'
        if p.exists():   # resumable: delete a file to re-fetch it
            continue
        feats = query(layer, bbox, where)
        json.dump({'type': 'FeatureCollection', 'features': feats}, open(p, 'w'))
        print(f'{key}: {len(feats)} {name} -> {p.name}')
