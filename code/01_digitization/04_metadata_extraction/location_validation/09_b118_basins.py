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

County check (2026-11-18): see COUNTY_CHECK / NAMESAKE below.

Validation: the same matching on PRECISELY located groundwater rows; share whose precise
point falls inside the matched polygon (with a 2 km buffer).

Inputs:  data/analysis/metadata_base.csv (meta_source.py), location_validation/locations.csv, page_types.csv,
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
COUNTY_CHECK = True

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


# County check (2026-11-18): B118 names collide across the state ("SANTA MARIA VALLEY" is basin 9-011 in San
# Diego County; the Santa Barbara County basin is "SANTA MARIA RIVER VALLEY", 3-012). A match is rejected when
# the row's county field (else the page) names California counties and none of them overlaps the basin.
_co = gpd.read_file(root / 'data/analysis/spatial/ca_counties/tl_2023_us_county.shp')
_co = _co[_co['STATEFP'] == '06'][['NAME', 'geometry']].to_crs('EPSG:3310')
_bx = basins[['Basin_Number', 'geometry']].to_crs('EPSG:3310')
_ov = gpd.overlay(_bx, _co, how='intersection', keep_geom_type=True)
_ov = _ov[_ov.area > 1e6]                                   # > 1 km2 of the basin in the county
BASIN_COUNTIES = _ov.groupby('Basin_Number')['NAME'].apply(set).to_dict()
CA_COUNTIES = sorted(_co['NAME'], key=len, reverse=True)
_CO_KEY = {re.sub(r'\s+', ' ', c.lower()): c for c in CA_COUNTIES}
_CO_RX = re.compile(r'\b(' + '|'.join(re.escape(c).replace(r'\ ', r'\s+') for c in CA_COUNTIES)
                    + r')\s+(?:county|counties|co\.)', re.I)


def counties_in(text):
    """California counties named as '<Name> County' / 'Counties' / 'Co.' in the text."""
    return {_CO_KEY[re.sub(r'\s+', ' ', m.lower())] for m in _CO_RX.findall(text or '')}


# Near-namesakes: names that differ only by River / Creek / Lake / Valley / Basin / Area / Plain
# ("SANTA CLARA VALLEY" 2-009 vs "SANTA CLARA RIVER VALLEY" 4-004). Report text uses the short form for
# both, so a match on such a basin needs a county (row field or page) that overlaps it.
def _core(n):
    return ' '.join(re.sub(r'\b(RIVER|CREEK|VALLEY|BASIN|AREA|PLAIN|LAKE)\b', '', n).split())


_cores = basins.assign(c=basins['Basin_Name'].map(_core)).groupby('c')['Basin_Number'].nunique()
NAMESAKE = set(basins.loc[basins['Basin_Name'].map(_core).map(_cores) > 1, 'Basin_Number'])


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


# rows: the new per-location metadata + the first locations.csv + page types (2026-11-18)
import sys as _s  # noqa: E402
_s.path.insert(0, str(Path(__file__).resolve().parent))
from meta_source import load_meta  # noqa: E402
d = load_meta(['id', 'page_number', 'water_type', 'watersource_name', 'location', 'county']).rename(
    columns={'water_type': 'water_type_clean'})
_l = pd.read_csv(root / 'data/analysis/location_validation/locations.csv',
                 usecols=['meta_row', 'lat', 'lon', 'precision_class']).set_index('meta_row')
d['precision_class'] = d['meta_row'].map(_l['precision_class'])
d['lat_combined'] = d['meta_row'].map(_l['lat'])
d['lon_combined'] = d['meta_row'].map(_l['lon'])
_pt = pd.read_csv(root / 'data/analysis/page_types.csv', usecols=['id', 'page_number', 'page_type'])
d = d.merge(_pt, on=['id', 'page_number'], how='left')
d['is_data_page'] = ~d['page_type'].isin(['index_contents', 'report_list', 'narrative'])
gw = d[(d['water_type_clean'] == 'Groundwater') & d['is_data_page'] & d['id'].notna() & d['page_number'].notna()].copy()
gw[['id', 'page_number']] = gw[['id', 'page_number']].astype(int)
targets = gw[gw['precision_class'].isin(['unlocated', 'basin'])]   # basin = this tier's own earlier output
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
        if res and COUNTY_CHECK:
            ev = counties_in(r.county if isinstance(r.county, str) else '') or counties_in(page_text(r.id, r.page_number))
            bn = str(res['b118_id']).split('.')[0]
            if ev and not (ev & BASIN_COUNTIES.get(bn, set())):
                res, status = None, 'county_conflict'
            elif bn in NAMESAKE and not ev:
                res, status = None, 'namesake_no_county'
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
