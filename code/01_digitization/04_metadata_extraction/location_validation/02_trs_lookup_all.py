"""
Location validation, step 2: PLSS (Township/Range/Section) reference for EVERY
production metadata row that quotes a T/R.

The LLM's own T/R parsing was unreliable (that's why the original pipeline told it
NOT to convert T/R to coordinates). So the LLM's T/R is used only as a pointer:
a row gets a PLSS reference only if the same Township AND Range also appear in
the page's own Reducto text. The section is used only if it also appears on the
page next to that T/R; otherwise the township centroid is used.

Meridian/state is chosen from page evidence only (never the LLM coordinate):
  named meridian on the page > single candidate > county named on the page
  (> document meridian, only if USE_DOCUMENT_MERIDIAN).
Out-of-state candidates are searched only when the page has a state abbreviation
("Oreg.", "Nev.", "Ariz.", ...).

Output columns (one row per main_metadata row with a verified T/R):
  meta_row            row position in data/metadata/main_metadata.csv
  ref_lat/ref_lon     PLSS centroid; ref_precision = section | township
  ref_status          ok | ambiguous_meridian | no_match
  coord_source        actual | inferred (the production lat_combined source)
  error_km            distance production coordinate -> PLSS reference

Inputs:  data/metadata/main_metadata.csv, data/digitized/{id}/page_{n}/*.json,
         data/analysis/spatial/ca_counties/tl_2023_us_county.shp, BLM PLSS service
Outputs: data/analysis/location_validation/
             plss_townships.parquet, plss_sections_cache.parquet  (cached downloads)
             trs_lookup_all.csv, trs_lookup_summary.csv
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
out_dir = root / 'data/analysis/location_validation'
out_dir.mkdir(parents=True, exist_ok=True)
# Document-meridian rule borrows evidence from OTHER pages of the same report.
# Off by default: Annette's rule (2026-10-28) is that every location comes from the
# row's own page. True resolved ~1,225 more ambiguous rows.
USE_TITLE_COUNTY = False       # tested 2026-10-28: 0/6 rows agreed with printed coords (median 447 km) -> off
                               # (the same rule works for well numbers: 98% NWIS-confirmed, n=686)
USE_DOCUMENT_MERIDIAN = True   # approved by Annette 2026-10-28 (labeled evidence_scope = document)

PLSS = 'https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer'

# Out-of-state cues: abbreviations only (full names collide with Nevada County,
# Colorado River, ...). CA is always searched.
STATE_CUES = {'OR': r'\boreg\.', 'NV': r'\bnev\.', 'AZ': r'\bariz\.', 'CO': r'\bcolo\.',
              'UT': r'\butah\b', 'ID': r'\bidaho\b', 'NM': r'\bn\. ?mex\.'}
MERIDIAN_CUES = {
    'Humboldt':       r'humboldt (base|meridian)|h\. ?b\. ?(&|and) ?m',
    'Mount Diablo':   r'(mount|mt\.?) diablo (base|meridian)|m\. ?d\. ?b\. ?(&|and) ?m',
    'San Bernardino': r'san bernardino (base|meridian)|s\. ?b\. ?b\. ?(&|and) ?m',
}

# T/R patterns. OCR often writes "R,11 E." -- allow , or . after R.
_T   = r'\bT(?:p|wp)?[.,]?\s*(\d{1,2})\s*([NS])\b'
_R   = r'\bR[.,]?\s*(\d{1,2})\s*([EW])\b'
_SEC = r'\bsecs?[.,]?\s*(\d{1,2})\b'
_SWN = r'\b(\d{1,2})\s*([NS])\s*/\s*(\d{1,2})\s*([EW])\s*-?\s*(\d{1,2})?'   # state well no. 4N/1W-4A1
RE_T, RE_R, RE_SEC, RE_SWN = (re.compile(p, re.I) for p in (_T, _R, _SEC, _SWN))
RE_TR_PAGE = re.compile(_T + r'.{0,40}?' + _R, re.I | re.S)


def llm_trs(text):
    """First T/R(/sec) in the LLM's TRS + Location text -> (t, td, r, rd, sec) or None."""
    t, r = RE_T.search(text), RE_R.search(text)
    if t and r:
        s = RE_SEC.search(text)
        sec = int(s.group(1)) if s and 1 <= int(s.group(1)) <= 36 else None
        return int(t.group(1)), t.group(2).upper(), int(r.group(1)), r.group(2).upper(), sec
    m = RE_SWN.search(text)
    if m:
        sec = int(m.group(5)) if m.group(5) and 1 <= int(m.group(5)) <= 36 else None
        return int(m.group(1)), m.group(2).upper(), int(m.group(3)), m.group(4).upper(), sec
    return None


def page_trs(text):
    """{(t, td, r, rd): set(sections seen next to it)} found in the page's own text."""
    out = {}
    for m in RE_TR_PAGE.finditer(text):
        key = (int(m.group(1)), m.group(2).upper(), int(m.group(3)), m.group(4).upper())
        window = text[max(0, m.start() - 60):m.end() + 10]   # "in NW 1/4 sec. 13, T. 5 N., R. 20 E."
        out.setdefault(key, set()).update(int(s) for s in RE_SEC.findall(window) if 1 <= int(s) <= 36)
    for m in RE_SWN.finditer(text):
        key = (int(m.group(1)), m.group(2).upper(), int(m.group(3)), m.group(4).upper())
        out.setdefault(key, set())
        if m.group(5) and 1 <= int(m.group(5)) <= 36:
            out[key].add(int(m.group(5)))
    return out


def read_page(doc, page):
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding='utf-8'))
    return ' '.join(c.get('content', '') for c in j['result']['chunks'])


# ------------------------------------------------------------------ BLM downloads
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


def townships(states):
    """Township centroids for the given states (cached per state)."""
    path = out_dir / 'plss_townships.parquet'
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['STATEABBR'])
    need = [s for s in states if s not in set(have['STATEABBR'])]
    for st in need:
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
        df = to_centroids(feats, ['PLSSID', 'STATEABBR', 'PRINMER', 'TWNSHPNO', 'TWNSHPDIR',
                                  'RANGENO', 'RANGEDIR', 'TWNSHPFRAC', 'RANGEFRAC'])
        print(f'  downloaded {len(df):,} {st} townships')
        have = pd.concat([have, df], ignore_index=True)
        have.to_parquet(path, index=False)
    return have[have['STATEABBR'].isin(states)]


def sections(plssids):
    """Section centroids for the given townships (cached)."""
    path = out_dir / 'plss_sections_cache.parquet'
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=['PLSSID'])
    done = set(have['PLSSID']) | set(have.attrs.get('empty', []))
    need = sorted(set(plssids) - done)
    for i in range(0, len(need), 40):          # 40 townships x 36 sections < 2,000 limit
        chunk = need[i:i + 40]
        ids = ','.join(f"'{p}'" for p in chunk)
        feats = blm_query(2, {'where': f"PLSSID IN ({ids}) AND FRSTDIVTYP='SN'",
                              'returnGeometry': 'true', 'maxAllowableOffset': 0.0005,
                              'outFields': 'PLSSID,FRSTDIVNO', 'resultRecordCount': 2000})
        df = to_centroids(feats, ['PLSSID', 'FRSTDIVNO'])
        have = pd.concat([have, df], ignore_index=True)
        if (i // 40) % 10 == 0:
            print(f'  sections: {i + len(chunk):,}/{len(need):,} townships')
            have.to_parquet(path, index=False)
    have.to_parquet(path, index=False)
    return have


# ------------------------------------------------------------------ county lookup
counties = gpd.read_file(root / 'data/analysis/spatial/ca_counties/tl_2023_us_county.shp')
counties = counties.to_crs('EPSG:4326')[['NAME', 'geometry']]


def title_counties(title, county_names):
    """CA counties named in a report title as "X County" / "X and Y Counties" (bare names such as
    "San Joaquin Valley" are not counted)."""
    out = set()
    for m in re.finditer(r'([A-Z][\w .,\-]*?)\s+[Cc]ount(?:y|ies)\b', str(title)):
        chunk = m.group(1)
        for c in sorted(county_names, key=len, reverse=True):
            if re.search(rf'\b{re.escape(c)}\b', chunk, re.I):
                out.add(c.lower())
                chunk = re.sub(re.escape(c), '', chunk, flags=re.I)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ================================================================== main
meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False)
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()].copy()
meta['lat_combined'] = meta['actual_latitude'].combine_first(meta['inferred_latitude'])
meta['lon_combined'] = meta['actual_longitude'].combine_first(meta['inferred_longitude'])
meta['coord_source'] = np.where(meta['actual_latitude'].notna(), 'actual',
                                np.where(meta['inferred_latitude'].notna(), 'inferred', 'none'))
txt = (meta['townships_ranges_sections'].fillna('').astype(str) + ' | ' +
       meta['location'].fillna('').astype(str))
meta['llm_trs'] = [llm_trs(s) for s in txt]
meta = meta[meta['llm_trs'].notna()].copy()
print(f'Rows with an LLM T/R: {len(meta):,} on {meta[["id", "page_number"]].drop_duplicates().shape[0]:,} pages')

# ---- verify against page text, collect page evidence
recs = []
for (doc, page), g in meta.groupby(['id', 'page_number']):
    doc, page = int(doc), int(page)
    text = read_page(doc, page)
    ptrs = page_trs(text) if text else {}
    ev = (text or '').lower()
    states = ['CA'] + [s for s, pat in STATE_CUES.items() if re.search(pat, ev)]
    named = [m for m, pat in MERIDIAN_CUES.items() if re.search(pat, ev)]
    for _, row in g.iterrows():
        t, td, r, rd, sec = row['llm_trs']
        key = (t, td, r, rd)
        if text is None:
            status = 'no_page_text'
        elif key not in ptrs:
            status = 'tr_not_on_page'
        else:
            status = 'verified'
        sec_ok = status == 'verified' and sec is not None and sec in ptrs.get(key, set())
        recs.append({'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
                     'T': t, 'TD': td, 'R': r, 'RD': rd, 'sec': sec if sec_ok else None,
                     'llm_sec': sec, 'verify': status, 'states': ','.join(states),
                     'named_meridian': named[0] if len(named) == 1 else None,
                     'page_text_lc': ev if status == 'verified' else '',
                     'location': row['location'], 'water_type': row['water_type'],
                     'coord_source': row['coord_source'],
                     'lat': row['lat_combined'], 'lon': row['lon_combined']})
rec = pd.DataFrame(recs)
print(rec['verify'].value_counts().to_string())

# ---- candidate townships
v = rec[rec['verify'] == 'verified'].copy()
all_states = sorted({s for ss in v['states'] for s in ss.split(',')})
tw = townships(all_states)
tw = tw[(tw['TWNSHPFRAC'] == '0') & (tw['RANGEFRAC'] == '0')].copy()
tw['key'] = list(zip(tw['TWNSHPNO'].astype(int), tw['TWNSHPDIR'], tw['RANGENO'].astype(int), tw['RANGEDIR']))
tw_by_key = tw.groupby('key')

need_ids = set()
for _, row in v.drop_duplicates(['T', 'TD', 'R', 'RD', 'states']).iterrows():
    key = (row['T'], row['TD'], row['R'], row['RD'])
    if key in tw_by_key.groups:
        c = tw_by_key.get_group(key)
        need_ids.update(c.loc[c['STATEABBR'].isin(row['states'].split(',')), 'PLSSID'])
secs = sections(need_ids)
secs['FRSTDIVNO'] = pd.to_numeric(secs['FRSTDIVNO'], errors='coerce')
sec_lookup = secs.groupby(['PLSSID', 'FRSTDIVNO'])[['lat', 'lon']].first()

# county of every candidate township (for the "county named on page" rule)
tw_c = gpd.GeoDataFrame(tw, geometry=gpd.points_from_xy(tw['lon'], tw['lat']), crs='EPSG:4326')
tw_c = gpd.sjoin(tw_c, counties, how='left', predicate='within')
county_of = dict(zip(tw_c['PLSSID'], tw_c['NAME']))

out = []
for _, row in rec.iterrows():
    o = {k: row[k] for k in ['meta_row', 'id', 'page_number', 'T', 'TD', 'R', 'RD', 'sec', 'llm_sec',
                             'verify', 'location', 'water_type', 'coord_source', 'lat', 'lon']}
    if row['verify'] != 'verified':
        o['ref_status'] = 'not_verified'
        out.append(o)
        continue
    key = (row['T'], row['TD'], row['R'], row['RD'])
    cands = (tw_by_key.get_group(key) if key in tw_by_key.groups else tw.iloc[0:0])
    cands = cands[cands['STATEABBR'].isin(row['states'].split(','))]
    chosen, rule = None, None
    if isinstance(row['named_meridian'], str):
        sub = cands[cands['PRINMER'].str.contains(row['named_meridian'], case=False)]
        if len(sub) == 1:
            chosen, rule = sub.iloc[0], 'meridian named on page'
    if chosen is None and len(cands) == 1:
        chosen, rule = cands.iloc[0], 'only candidate'
    if chosen is None and len(cands) > 1:
        named = [p for p in cands['PLSSID']
                 if isinstance(county_of.get(p), str) and re.search(rf'\b{re.escape(county_of[p].lower())}\b', row['page_text_lc'])]
        if len(named) == 1:
            chosen, rule = cands[cands['PLSSID'] == named[0]].iloc[0], 'county named on page'
    o['n_candidates'] = len(cands)
    if chosen is None:
        o['ref_status'] = 'no_match' if len(cands) == 0 else 'ambiguous_meridian'
        out.append(o)
        continue
    lat, lon, prec = chosen['lat'], chosen['lon'], 'township'
    if pd.notna(row['sec']) and (chosen['PLSSID'], row['sec']) in sec_lookup.index:
        lat, lon = sec_lookup.loc[(chosen['PLSSID'], row['sec'])]
        prec = 'section'
    o.update({'ref_status': 'ok', 'ref_rule': rule, 'ref_meridian': chosen['PRINMER'],
              'ref_state': chosen['STATEABBR'], 'plssid': chosen['PLSSID'],
              'ref_lat': lat, 'ref_lon': lon, 'ref_precision': prec})
    if pd.notna(row['lat']):
        o['error_km'] = round(float(haversine_km(row['lat'], row['lon'], lat, lon)), 2)
    out.append(o)

res = pd.DataFrame(out)

# ---- document-meridian rule: pages of one report share a meridian. If every row
# resolved above in a document (>= 3 rows) uses one meridian, apply it to that
# document's ambiguous rows. Still page evidence only, never the LLM coordinate.
resolved = res[res['ref_status'] == 'ok']
doc_mer = resolved.groupby('id')['ref_meridian'].agg(lambda s: s.iloc[0] if s.nunique() == 1 and len(s) >= 3 else None)
doc_mer = doc_mer.dropna().to_dict()
amb = res.index[(res['ref_status'] == 'ambiguous_meridian') & res['id'].isin(doc_mer)] if USE_DOCUMENT_MERIDIAN else []
for i in amb:
    row = res.loc[i]
    key = (row['T'], row['TD'], row['R'], row['RD'])
    cands = tw_by_key.get_group(key)
    cands = cands[cands['PRINMER'] == doc_mer[row['id']]]
    if len(cands) != 1:
        continue
    c = cands.iloc[0]
    lat, lon, prec = c['lat'], c['lon'], 'township'
    if pd.notna(row['sec']) and (c['PLSSID'], row['sec']) in sec_lookup.index:
        lat, lon = sec_lookup.loc[(c['PLSSID'], row['sec'])]
        prec = 'section'
    res.loc[i, ['ref_status', 'ref_rule', 'ref_meridian', 'ref_state', 'plssid',
                'ref_lat', 'ref_lon', 'ref_precision']] = \
        ['ok', 'document meridian', c['PRINMER'], c['STATEABBR'], c['PLSSID'], lat, lon, prec]
    if pd.notna(row['lat']):
        res.loc[i, 'error_km'] = round(float(haversine_km(row['lat'], row['lon'], lat, lon)), 2)
# ---- county named in the report title (document-level, like the document meridian)
if USE_TITLE_COUNTY:
    titles = (pd.read_csv(root / 'data/metadata/main_metadata.csv', usecols=['id', 'Title'], low_memory=False)
                .dropna(subset=['id']).drop_duplicates('id').set_index('id')['Title'])
    names = {n for n in county_of.values() if isinstance(n, str)}
    tc = {i: title_counties(t, names) for i, t in titles.items()}
    for i in res.index[res['ref_status'] == 'ambiguous_meridian']:
        row = res.loc[i]
        want = tc.get(row['id'], set())
        if not want:
            continue
        cands = tw_by_key.get_group((row['T'], row['TD'], row['R'], row['RD']))
        cands = cands[cands['PLSSID'].map(lambda p_: str(county_of.get(p_)).lower() in want)]
        if len(cands) != 1:
            continue
        c = cands.iloc[0]
        lat, lon, prec = c['lat'], c['lon'], 'township'
        if pd.notna(row['sec']) and (c['PLSSID'], row['sec']) in sec_lookup.index:
            lat, lon = sec_lookup.loc[(c['PLSSID'], row['sec'])]
            prec = 'section'
        res.loc[i, ['ref_status', 'ref_rule', 'ref_meridian', 'ref_state', 'plssid',
                    'ref_lat', 'ref_lon', 'ref_precision']] = \
            ['ok', 'county in report title', c['PRINMER'], c['STATEABBR'], c['PLSSID'], lat, lon, prec]
        if pd.notna(row['lat']):
            res.loc[i, 'error_km'] = round(float(haversine_km(row['lat'], row['lon'], lat, lon)), 2)
    print(f'Title-county rule resolved {(res["ref_rule"] == "county in report title").sum():,} ambiguous rows')
print(f'Document-meridian rule resolved {(res["ref_rule"] == "document meridian").sum():,} ambiguous rows')

res.to_csv(out_dir / 'trs_lookup_all.csv', index=False)

# ---- summary: by coordinate source and reference precision
print('\n', res['ref_status'].value_counts().to_string())
ok = res[(res['ref_status'] == 'ok') & res['error_km'].notna()]
rows = []
for (src, prec), g in ok.groupby(['coord_source', 'ref_precision']):
    e = g['error_km']
    rows.append({'coord_source': src, 'ref_precision': prec, 'rows': len(g),
                 'pages': g[['id', 'page_number']].drop_duplicates().shape[0],
                 'median_km': e.median(), 'le_2km': np.mean(e <= 2), 'le_5km': np.mean(e <= 5),
                 'le_10km': np.mean(e <= 10), 'le_25km': np.mean(e <= 25), 'gt_100km': np.mean(e > 100)})
summary = pd.DataFrame(rows)
summary.to_csv(out_dir / 'trs_lookup_summary.csv', index=False)
print('\n', summary.round(3).to_string(index=False))
print(f'\nSaved: {out_dir / "trs_lookup_all.csv"}')
