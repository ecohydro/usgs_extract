"""
Location validation, step 1: reference coordinates from Township/Range/Section.

For every LLM metadata row in the curated and random 135-page test sets that
quotes a Township/Range (T/R) location, look up the PLSS section (or township)
in the BLM National PLSS CadNSDI service and use its centroid as an INDEPENDENT
reference location. Later steps add NWIS-name and GNIS references; scoring
compares the LLM coordinate to the reference (distance, same HUC12, ...).

Meridian / state choice
-----------------------
Pages almost never name the base meridian, and the same T/R number exists under
several meridians (CA: Humboldt, Mount Diablo, San Bernardino). Every matching
township in the candidate states is looked up, then ONE is chosen using only
evidence from the page itself, never the LLM coordinate (that would be circular):
  1. a meridian named in the page text ("San Bernardino base line and meridian")
  2. state abbreviations in the page text ("Oreg.", "Ariz.") add those states
     to the search (CA is always searched)
  3. the candidate whose county name appears in the page text / LLM county fields
  4. if only one candidate exists, it is used
Otherwise the row is flagged `ambiguous_meridian` and all candidates are kept
for human review.

Precision: section centroid ~0.5 mi (sections are 1 mi square); township-only
centroid ~3 mi (townships are 6 mi square).

Inputs
  data/digitization_intermediates/04_metadata_extraction/validation/
      validation_curated_135.xlsx, validation_random_135.xlsx, jsons/
  data/digitized/{id}/page_{n}/{id}_page_{n}.json   (page-text fallback)
  data/analysis/spatial/ca_counties/tl_2023_us_county.shp
Outputs (data/digitization_intermediates/04_metadata_extraction/validation/location_reference/)
  plss_cache.json        cached BLM query results (delete to re-query)
  trs_reference.csv      one row per test-set metadata row with a T/R
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

root    = Path(__file__).resolve().parents[4]
val_dir = root / 'data/digitization_intermediates/04_metadata_extraction/validation'
out_dir = val_dir / 'location_reference'
out_dir.mkdir(exist_ok=True)
cache_path = out_dir / 'plss_cache.json'

PLSS = 'https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer'

# Out-of-state cues: abbreviations only. Full names collide with CA features
# (Nevada County, Colorado River, Oregon Creek). CA is always searched as well.
STATE_CUES = {   # page-text abbreviation -> state
    'OR': r'\boreg\.',
    'NV': r'\bnev\.',
    'AZ': r'\bariz\.',
    'CO': r'\bcolo\.',
    'UT': r'\butah\b',
    'ID': r'\bidaho\b',
    'NM': r'\bn\. ?mex\.',
}
MERIDIAN_CUES = {
    'Humboldt':       r'humboldt (base|meridian)|h\. ?b\. ?(&|and) ?m',
    'Mount Diablo':   r'(mount|mt\.?) diablo (base|meridian)|m\. ?d\. ?b\. ?(&|and) ?m',
    'San Bernardino': r'san bernardino (base|meridian)|s\. ?b\. ?b\. ?(&|and) ?m',
}

# ------------------------------------------------------------------ test sets
def load_set(fname, sheet, set_name):
    d = pd.read_excel(val_dir / fname, sheet_name=sheet)
    m = d[d['ID'].notna()].copy()
    m['test_set'] = set_name
    m['row_in_file'] = m.index
    m['ID'] = m['ID'].astype(int)
    m['PAGE_NUMBER'] = m['PAGE_NUMBER'].astype(int)
    return m


test_rows = pd.concat([load_set('validation_curated_135.xlsx', 'Sheet1', 'curated'),
                       load_set('validation_random_135.xlsx', 'metadata', 'random')],
                      ignore_index=True)

# The LLM was re-run after the test sets were graded: "actual" coordinates are
# identical, but inferred ones differ between runs. So the PRODUCTION rows
# (data/metadata/main_metadata.csv) for the same test pages are scored too.
prod = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False)
prod = prod.rename(columns={
    'id': 'ID', 'page_number': 'PAGE_NUMBER', 'actual_latitude': 'Actual_Latitude',
    'actual_longitude': 'Actual_Longitude', 'inferred_latitude': 'Inferred_Latitude',
    'inferred_longitude': 'Inferred_Longitude', 'location': 'Location',
    'townships_ranges_sections': 'Townships_Ranges_Sections', 'actual_county': 'Actual_County',
    'inferred_county': 'Inferred_County', 'water_type': 'Water_Type'})
test_pages = test_rows[['ID', 'PAGE_NUMBER']].drop_duplicates()
prod = prod[prod['ID'].notna() & prod['PAGE_NUMBER'].notna()].astype({'ID': int, 'PAGE_NUMBER': int})
prod = prod.merge(test_pages, on=['ID', 'PAGE_NUMBER'])
prod = prod.merge(test_rows[['ID', 'PAGE_NUMBER', 'test_set']].drop_duplicates(['ID', 'PAGE_NUMBER']),
                  on=['ID', 'PAGE_NUMBER'])
prod['test_set'] = 'production_' + prod['test_set']
prod['row_in_file'] = prod.index

rows = pd.concat([test_rows, prod], ignore_index=True)

# ------------------------------------------------------------------ T/R parsing
_T = re.compile(r'\bT(?:p|wp)?\.?\s*(\d{1,2})\s*([NS])\b', re.I)
_R = re.compile(r'\bR\.?\s*(\d{1,2})\s*([EW])\b', re.I)
_S = re.compile(r'\bsec(?:tion|s)?\.?\s*(\d{1,2})\b', re.I)


def parse_trs(text):
    """First township, range and section in text -> (t, tdir, r, rdir, sec) or None."""
    t, r = _T.search(text), _R.search(text)
    if not (t and r):
        return None
    s = _S.search(text)
    sec = int(s.group(1)) if s and 1 <= int(s.group(1)) <= 36 else None
    return int(t.group(1)), t.group(2).upper(), int(r.group(1)), r.group(2).upper(), sec


def trs_text(row):
    # TRS field first; fall back to the Location text, which often repeats it
    parts = [str(row['Townships_Ranges_Sections']), str(row['Location'])]
    return ' | '.join(p for p in parts if p and p != 'nan')


# ------------------------------------------------------------------ page text
def page_text(doc, page):
    for p in [val_dir / f'jsons/{doc}_page_{page}.json',
              root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json']:
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            return ' '.join(c.get('content', '') for c in j['result']['chunks'])
    return ''


# ------------------------------------------------------------------ BLM queries (cached)
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}


def query(layer, where, fields, geometry):
    key = f'{layer}|{where}|{geometry}'
    if key in cache:
        return cache[key]
    params = {'where': where, 'outFields': fields, 'returnGeometry': str(geometry).lower(),
              'outSR': 4326, 'f': 'geojson' if geometry else 'json'}
    url = f'{PLSS}/{layer}/query?' + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                j = json.load(r)
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f'PLSS query failed: {where}')
    feats = j.get('features', [])
    cache[key] = feats
    cache_path.write_text(json.dumps(cache))
    return feats


def candidates(t, tdir, r, rdir, sec, states):
    """All PLSS townships (and the section, if given) matching T/R in the states."""
    st = ','.join(f"'{s}'" for s in states)
    where = (f"STATEABBR IN ({st}) AND TWNSHPNO='{t:03d}' AND TWNSHPDIR='{tdir}' "
             f"AND RANGENO='{r:03d}' AND RANGEDIR='{rdir}' AND TWNSHPFRAC='0' AND RANGEFRAC='0'")
    towns = query(1, where, 'PLSSID,STATEABBR,PRINMER', True)
    out = []
    for f in towns:
        p = f['properties']
        geom, level = shape(f['geometry']), 'township'
        if sec:
            secs = query(2, f"PLSSID='{p['PLSSID']}' AND FRSTDIVNO='{sec:02d}' AND FRSTDIVTYP='SN'",
                         'PLSSID,FRSTDIVNO', True)
            if secs:
                geom, level = shape(secs[0]['geometry']), 'section'
            else:
                level = 'township (section not found)'
        c = geom.representative_point() if not geom.is_valid else geom.centroid
        out.append({'plssid': p['PLSSID'], 'state': p['STATEABBR'], 'meridian': p['PRINMER'],
                    'ref_lat': c.y, 'ref_lon': c.x, 'precision': level})
    return out


# ------------------------------------------------------------------ county lookup
counties = gpd.read_file(root / 'data/analysis/spatial/ca_counties/tl_2023_us_county.shp')
counties = counties.to_crs('EPSG:4326')[['NAME', 'STATEFP', 'geometry']]


def county_of(lat, lon):
    hit = counties[counties.contains(gpd.points_from_xy([lon], [lat])[0])]
    return hit['NAME'].iloc[0] if len(hit) else None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ main
records = []
text_cache = {}
for _, row in rows.iterrows():
    trs = parse_trs(trs_text(row))
    if trs is None:
        continue
    t, tdir, r, rdir, sec = trs
    key = (row['ID'], row['PAGE_NUMBER'])
    if key not in text_cache:
        text_cache[key] = page_text(*key)
    ptext = text_cache[key]
    evidence = ' '.join([ptext, str(row['Location']), str(row['Actual_County']),
                         str(row['Inferred_County'])]).lower()

    states = ['CA'] + [s for s, pat in STATE_CUES.items() if re.search(pat, evidence)]
    cands = candidates(t, tdir, r, rdir, sec, states)

    chosen, rule = None, None
    named = [m for m, pat in MERIDIAN_CUES.items() if re.search(pat, evidence)]
    if len(named) == 1:
        sub = [c for c in cands if named[0].lower() in c['meridian'].lower()]
        if len(sub) == 1:
            chosen, rule = sub[0], 'meridian named on page'
    if chosen is None and len(cands) > 1:
        for c in cands:
            c['county'] = county_of(c['ref_lat'], c['ref_lon'])
        sub = [c for c in cands if c['county'] and re.search(rf"\b{re.escape(c['county'].lower())}\b", evidence)]
        if len(sub) == 1:
            chosen, rule = sub[0], 'county named on page'
    if chosen is None and len(cands) == 1:
        chosen, rule = cands[0], 'only candidate'

    lat_llm = row['Actual_Latitude'] if pd.notna(row['Actual_Latitude']) else row['Inferred_Latitude']
    lon_llm = row['Actual_Longitude'] if pd.notna(row['Actual_Longitude']) else row['Inferred_Longitude']
    rec = {
        'test_set': row['test_set'], 'row_in_file': row['row_in_file'],
        'id': row['ID'], 'page_number': row['PAGE_NUMBER'],
        'water_type': row['Water_Type'], 'location_text': row['Location'],
        'trs_text': row['Townships_Ranges_Sections'],
        'T': f'{t}{tdir}', 'R': f'{r}{rdir}', 'sec': sec,
        'coord_source': 'actual' if pd.notna(row['Actual_Latitude']) else
                        ('inferred' if pd.notna(row['Inferred_Latitude']) else 'none'),
        'llm_lat': lat_llm, 'llm_lon': lon_llm,
        'states_searched': ','.join(states), 'n_candidates': len(cands),
        'candidates': '; '.join(f"{c['meridian']} {c['state']} ({c['ref_lat']:.3f},{c['ref_lon']:.3f})"
                                for c in cands),
    }
    if chosen:
        rec.update({'ref_status': 'ok', 'ref_rule': rule, 'ref_lat': chosen['ref_lat'],
                    'ref_lon': chosen['ref_lon'], 'ref_meridian': chosen['meridian'],
                    'ref_state': chosen['state'], 'ref_precision': chosen['precision']})
    else:
        rec['ref_status'] = 'no_match' if not cands else 'ambiguous_meridian'
    if chosen and pd.notna(lat_llm):
        rec['error_km'] = round(float(haversine_km(lat_llm, lon_llm, chosen['ref_lat'], chosen['ref_lon'])), 2)
    records.append(rec)

ref = pd.DataFrame(records)
ref.to_csv(out_dir / 'trs_reference.csv', index=False)

print(f'Test-set metadata rows: {len(rows):,}; rows with a parseable T/R: {len(ref):,}')
print(ref.groupby(['test_set', 'ref_status']).size().unstack(fill_value=0))
print('\nChoice rule:\n', ref['ref_rule'].value_counts())
ok = ref[ref['ref_status'].eq('ok') & ref['error_km'].notna()].copy()
ok['run'] = np.where(ok['test_set'].str.startswith('production'), 'production', 'test set')
summary = []
for (run, src), g in ok.groupby(['run', 'coord_source']):
    pg = g.groupby(['id', 'page_number'])['error_km'].median()   # one value per page
    summary.append({'run': run, 'coord_source': src, 'rows': len(g), 'pages': len(pg),
                    'row_median_km': g['error_km'].median(),
                    'page_median_km': pg.median(),
                    'pages_le_5km': np.mean(pg <= 5), 'pages_le_25km': np.mean(pg <= 25),
                    'pages_gt_100km': np.mean(pg > 100)})
summary = pd.DataFrame(summary)
summary.to_csv(out_dir / 'trs_reference_summary.csv', index=False)
print('\n', summary.round(2).to_string(index=False))
print(f'\nSaved: {out_dir / "trs_reference.csv"}')
