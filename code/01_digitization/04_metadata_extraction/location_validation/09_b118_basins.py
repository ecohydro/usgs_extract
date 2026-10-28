"""
Location tier 5 (groundwater only): DWR Bulletin 118 groundwater basin named on the page.

Approved 2026-10-28 (Annette). Many groundwater pages name only a valley / basin / area
("Ground-water inventory for Santa Clara Valley"). These rows get a BASIN-LEVEL location:
a point inside the Bulletin 118 basin (or subbasin) polygon, precision_class = 'basin'.
Basin locations are NOT used for HUC12 statistics or point maps; they support basin-level
summaries (the natural unit for groundwater, cf. 09_sgma_groundwater_basins.py).

Matching (same-page evidence; the LLM text is a pointer only):
  - a B118 basin name appears in the row's LLM water-source name or location, AND in the
    page text; one-word names (e.g. CONEJO) must be followed by Valley / Basin / Area;
  - if the subbasin name (the part after " - ") also appears, the subbasin polygon is used;
  - several different basins matched -> the longest name wins only if it is unique,
    otherwise ambiguous.
Applied to groundwater rows on data pages that no other method located.

Validation: the same matching on PRECISELY located groundwater rows; share whose precise
point falls inside the matched polygon (with a 2 km buffer).

Inputs:  data/analysis/processed_metadata.parquet (water type, page_type, precision_class),
         data/analysis/spatial/sgma/sgma_2019_basin_prioritization.geojson (DWR B118, 515 polygons),
         data/digitized/.../*.json
Outputs: data/analysis/location_validation/tier5_b118_basins.csv, tier5_b118_validation.csv
"""

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'
SUFFIX = r'(?:valley|basin|area|ground[- ]?water basin)'

b = gpd.read_file(root / 'data/analysis/spatial/sgma/sgma_2019_basin_prioritization.geojson').to_crs('EPSG:4326')
b = b.set_geometry(b.geometry.make_valid().buffer(0))     # a few B118 polygons self-intersect
b['sub'] = b['Basin_Subbasin_Name'].str.split(' - ', n=1).str[1]
basins = b.dissolve(by=['Basin_Number', 'Basin_Name'], as_index=False)[['Basin_Number', 'Basin_Name', 'geometry']]
eq = gpd.GeoSeries(basins.geometry, crs='EPSG:4326').to_crs('EPSG:3310')
basins['radius_km'] = np.sqrt(eq.area / np.pi) / 1000
subs = b[b['sub'].notna()].copy()
subs['radius_km'] = np.sqrt(subs.geometry.to_crs('EPSG:3310').area / np.pi) / 1000


def name_rx(name):
    words = re.escape(name.lower()).replace(r'\ ', r'\s+')
    if re.search(r'(valley|basin|area|plain)$', name.lower()):
        return re.compile(rf'\b{words}\b')
    return re.compile(rf'\b{words}\s+{SUFFIX}\b')


basins['rx'] = basins['Basin_Name'].map(name_rx)
subs['rx'] = subs['sub'].map(lambda s: re.compile(rf"\b{re.escape(s.lower()).replace(chr(92) + ' ', chr(92) + 's+')}\b"))


def match(row_text, page_text):
    rt, pt = row_text.lower(), page_text.lower()
    hits = basins[np.array([bool(r.search(rt)) and bool(r.search(pt)) for r in basins['rx']])]
    if hits.empty:
        return None, 'no_basin_name'
    if hits['Basin_Number'].nunique() > 1:
        longest = hits.loc[hits['Basin_Name'].str.len().idxmax()]
        if (hits['Basin_Name'].str.len() == len(longest['Basin_Name'])).sum() > 1:
            return None, 'ambiguous'
        hits = hits[hits['Basin_Number'] == longest['Basin_Number']]
    h = hits.iloc[0]
    sub_hit = np.array([bool(r.search(rt)) and bool(r.search(pt)) for r in subs['rx']])
    sb = subs[(subs['Basin_Number'] == h['Basin_Number']).to_numpy() & sub_hit]
    if len(sb) == 1:
        s = sb.iloc[0]
        return {'method': 'b118_subbasin', 'b118_id': s['Basin_Subbasin_Number'],
                'b118_name': s['Basin_Subbasin_Name'], 'geom': s.geometry, 'radius_km': s['radius_km']}, 'ok'
    return {'method': 'b118_basin', 'b118_id': h['Basin_Number'], 'b118_name': h['Basin_Name'],
            'geom': h.geometry, 'radius_km': h['radius_km']}, 'ok'


d = pd.read_parquet(root / 'data/analysis/processed_metadata.parquet',
                    columns=['meta_row', 'id', 'page_number', 'water_type_clean', 'watersource_name',
                             'location', 'precision_class', 'is_data_page', 'lat_combined', 'lon_combined'])
gw = d[(d['water_type_clean'] == 'Groundwater') & d['is_data_page'] & d['id'].notna() & d['page_number'].notna()].copy()
gw[['id', 'page_number']] = gw[['id', 'page_number']].astype(int)
targets = gw[gw['precision_class'] == 'unlocated']
checks = gw[gw['precision_class'] == 'precise']

_txt = {}


def page_text(doc, page):
    if (doc, page) not in _txt:
        p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
        _txt[(doc, page)] = (' '.join(c.get('content', '') for c in json.loads(p.read_text(encoding='utf-8'))['result']['chunks'])
                             if p.exists() else '')
    return _txt[(doc, page)]


def run(rows):
    out = []
    for r in rows.itertuples():
        res, status = match(f'{r.watersource_name} {r.location}', page_text(r.id, r.page_number))
        rec = {'meta_row': r.meta_row, 'id': r.id, 'page_number': r.page_number, 'status': status}
        if res:
            pt = res['geom'].representative_point()
            rec.update({'method': res['method'], 'b118_id': res['b118_id'], 'b118_name': res['b118_name'],
                        'lat': pt.y, 'lon': pt.x, 'precision_km': round(res['radius_km'], 1),
                        '_geom': res['geom']})
        out.append(rec)
    return pd.DataFrame(out)


res = run(targets)
res.drop(columns=['_geom'], errors='ignore').to_csv(lv / 'tier5_b118_basins.csv', index=False)
print(f'Unlocated groundwater rows on data pages: {len(targets):,}')
print(res['status'].value_counts().to_string())
print(res['method'].value_counts().to_string())
print('basin radius (km):', res['precision_km'].describe().round(1).to_dict())

# validation: does the precise location of already-located rows fall in the matched polygon?
v = run(checks)
v = v[v['status'] == 'ok'].merge(checks[['meta_row', 'lat_combined', 'lon_combined']], on='meta_row')
if len(v):
    pts = gpd.GeoSeries(gpd.points_from_xy(v['lon_combined'], v['lat_combined']), crs='EPSG:4326').to_crs('EPSG:3310')
    polys = gpd.GeoSeries(v['_geom'].values, crs='EPSG:4326').to_crs('EPSG:3310')
    v['inside_2km'] = [pl.buffer(2000).contains(p) for pl, p in zip(polys, pts)]
    val = v.groupby('method')['inside_2km'].agg(n='size', share_inside='mean').round(3)
    val.to_csv(lv / 'tier5_b118_validation.csv')
    print('\nValidation (precisely located GW rows; precise point inside matched polygon +2 km):')
    print(val.to_string())
