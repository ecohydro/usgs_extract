"""
Location tier 2b: state well numbers read from the row's OWN page.

The 40-page groundwater scoping check (gw_scoping/, Annette agreed 2026-10-28) found
most unlocated groundwater pages carry a well number that encodes Township/Range/
Section, in forms the earlier steps missed:
  full     "10S/6E-23B1", "29S/27E-34N2 M"   (OCR: "165/17E-17801 S" = 16S/17E-17B01 S)
  short    "5/2W-26Q1", "7/35-36E1", "5/2-25A1"   (N/S, E/W letters omitted)
  section  "4R1", "16Q1" listed under a page heading "T. 10 N., R. 26 W."
(Report-specific compact numbers like "4523Q1" and Nevada "11/24-22dc1" are not handled.)

Missing direction letters / meridian are resolved from the SAME page only:
  1 a full T/R ("T. 10 N., R. 26 W.") or full well number on the page with the same numbers
  2 the meridian letter printed after the well number (M, S, H)
  3 a single PLSS candidate in California
  4 a town named in the well's own heading ("About 4.1 miles west of Lompoc"):
    the candidate within 40 km of that GNIS place
  5 a county named on the page
If the page leaves it open, REPORT-LEVEL evidence is used (approved 2026-10-28; scope =
'document'): the same T/R numbers printed with letters elsewhere in the report, and the
report's single meridian among wells the
pages resolved on their own, and a county named in the report TITLE ("... Fresno and Kings
Counties"). The location itself still comes from the row's own page.
Otherwise the well is ambiguous and not used.

Each metadata row is matched to a well on its page by (a) a well number in the LLM
water-source/location text that is also on the page, (b) an owner name from the LLM
text found in the well's heading, or (c) the page having only one well. Rows covering
several wells get the centroid only if all wells are within 5 km (approximate).

Only rows whose LLM water type is NOT stream discharge / reservoir / precipitation are
matched (first run: stream-gauge rows picked up well numbers printed elsewhere on the
page, e.g. "Clear Lake at Lakeport" placed 180 km away at a well).
The direction combination "T/R on page + county on page" is not used (0/15 NWIS-confirmed).

Location: NWIS groundwater site coordinates when the full well number matches an NWIS
site name (e.g. 006N007W15C003S), else the PLSS section centroid.

Inputs:  data/metadata/main_metadata.csv, data/digitized/.../*.json,
         data/analysis/nwis_sites/ca_groundwater_sites.csv,
         data/analysis/spatial/gnis/CA_Features_20210825.txt,
         data/analysis/spatial/ca_counties/tl_2023_us_county.shp, BLM PLSS (cached)
Outputs: data/analysis/location_validation/tier2b_well_numbers.csv,
         tier2b_well_numbers_validation.csv
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plss_utils import MERIDIAN_LETTER, haversine_km, sections, townships  # noqa: E402

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'

# ------------------------------------------------------------------ patterns
TRACT = r'[A-HJ-NP-R8]'                       # tracts A-R without I, O; OCR 8 -> B
RE_WELL = re.compile(r'(?<![\d/.])(\d{1,2})\s?([NS5])?\s?/\s?(\d{1,2})\s?([EWF])?\s?[-–—]\s?'
                     r'(\d{1,2})\s?(' + TRACT + r')\s?(\d{1,3})(?:\s?([MSH])(?![a-z]))?')
RE_TRHEAD = re.compile(r'\bT\.?\s*(\d{1,2})\s*([NS])\.?\s*,?\s*R\.?\s*(\d{1,2})\s*([EW])\b')
RE_SECID = re.compile(r'(?<![\w/.\-])(\d{1,2})([A-HJ-NP-R])(\d{1,2})(?![\w/])')
RE_PLACE = re.compile(r'\b(?:of|near|in|at|from)\s+([A-Za-z][A-Za-z.\']+(?:\s+[A-Za-z][A-Za-z.\']+){0,2})')
GENERIC = {'WELL', 'WELLS', 'DOMESTIC', 'IRRIGATION', 'OBSERVATION', 'DRILLED', 'CITY', 'COMPANY',
           'WATER', 'TEST', 'STOCK', 'UNUSED', 'ARTESIAN', 'TABLE', 'VALLEY', 'COUNTY', 'BASIN',
           'AREA', 'GROUND', 'GROUNDWATER', 'SUPPLY', 'MUNICIPAL', 'PUBLIC', 'RANCH', 'CALIFORNIA',
           'CALIF', 'NEAR', 'ABOUT', 'MILES', 'MILE', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'INDUSTRIAL'}


def parse_wells(text):
    """Well numbers (full or short) in text: list of dicts with position."""
    out = []
    for m in RE_WELL.finditer(text):
        t, td, r, rd, sec, tract, seq, mer = m.groups()
        t, r, sec, seq = int(t), int(r), int(sec), int(seq)
        if not (1 <= t <= 60 and 1 <= r <= 60 and 1 <= sec <= 36):
            continue
        td = {'5': 'S'}.get(td, td)
        rd = {'F': 'E'}.get(rd, rd)
        out.append({'t': t, 'td': td, 'r': r, 'rd': rd, 'sec': sec,
                    'tract': 'B' if tract == '8' else tract, 'seq': seq, 'mer': mer,
                    'start': m.start(), 'end': m.end(), 'form': 'full' if td and rd else 'short'})
    return out


def parse_section_ids(text, well_spans):
    """Section-tract IDs ("4R1") listed under T/R headings on the page."""
    heads = [(m.start(), (int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))
             for m in RE_TRHEAD.finditer(text)]
    out = []
    for i, (pos, (t, td, r, rd)) in enumerate(heads):
        stop = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        for m in RE_SECID.finditer(text, pos, stop):
            if any(a <= m.start() < b for a, b in well_spans):
                continue
            sec = int(m.group(1))
            if 1 <= sec <= 36:
                out.append({'t': t, 'td': td, 'r': r, 'rd': rd, 'sec': sec, 'tract': m.group(2),
                            'seq': int(m.group(3)), 'mer': None, 'start': m.start(),
                            'end': m.end(), 'form': 'section_under_header'})
    return out


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()


def words(s):
    return {w for w in re.findall(r'[A-Z]{4,}', str(s).upper()) if w not in GENERIC}


# ------------------------------------------------------------------ reference data
tw = townships(('CA',))
tw_by_key = {k: g for k, g in tw.groupby('key')}
counties = gpd.read_file(root / 'data/analysis/spatial/ca_counties/tl_2023_us_county.shp').to_crs('EPSG:4326')
twg = gpd.sjoin(gpd.GeoDataFrame(tw, geometry=gpd.points_from_xy(tw['lon'], tw['lat']), crs='EPSG:4326'),
                counties[['NAME', 'geometry']], how='left', predicate='within')
county_of = dict(zip(twg['PLSSID'], twg['NAME']))

gn = pd.read_csv(root / 'data/analysis/spatial/gnis/CA_Features_20210825.txt', sep='|',
                 encoding='utf-8-sig', low_memory=False)
gn = gn[(gn['STATE_ALPHA'] == 'CA') & gn['FEATURE_CLASS'].isin(['Populated Place', 'Locale', 'Post Office'])
        & (gn['PRIM_LAT_DEC'] != 0)]
places = {k: list(zip(g['PRIM_LAT_DEC'], g['PRIM_LONG_DEC'])) for k, g in gn.groupby(gn['FEATURE_NAME'].map(norm))}

nwis = pd.read_csv(root / 'data/analysis/nwis_sites/ca_groundwater_sites.csv', dtype={'site_no': str},
                   low_memory=False).drop_duplicates('site_no')
nwis['swn'] = nwis['station_nm'].astype(str).str.upper().str.replace(r'[^0-9A-Z]', '', regex=True)
nwis = nwis[nwis['swn'].str.fullmatch(r'\d{3}[NS]\d{3}[EW]\d{2}[A-R]\d{3}[MSH]') & nwis['dec_lat_va'].notna()]
nwis_xy = nwis.drop_duplicates('swn').set_index('swn')[['dec_lat_va', 'dec_long_va', 'site_no']]


COUNTY_NAMES = sorted({n.lower() for n in county_of.values() if isinstance(n, str)})


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


def doc_letters(w, doc):
    """Fill missing N/S, E/W from the rest of the report -> (tds, rds, rule) or None."""
    pairs = doc['pairs'].get((w['t'], w['r']), set())
    if len(pairs) == 1:
        td, rd = next(iter(pairs))
        return [td], [rd], 'document: same T/R'
    # (a 'document: quadrant' rule was tested and dropped: 82% NWIS-confirmed, n=17)
    return None


def resolve(w, page_pairs, window, page_counties, doc=None):
    """Choose one PLSS township for a well -> (row of tw, rule) or (None, reason).
    Page evidence first; `doc` (report-level evidence) only fills what the page leaves open."""
    tds = [w['td']] if w['td'] else ['N', 'S']
    rds = [w['rd']] if w['rd'] else ['E', 'W']
    rule = 'as printed' if w['td'] and w['rd'] else None
    if rule is None:
        on_page = [(td, rd) for (t, td, r, rd) in page_pairs if t == w['t'] and r == w['r']
                   and td in tds and rd in rds]
        if len(set(on_page)) == 1:
            tds, rds = [on_page[0][0]], [on_page[0][1]]
            rule = 'T/R on page'
        elif doc is not None:
            d = doc_letters(w, doc)
            if d:
                tds, rds, rule = d
    cands = [c for td in tds for rd in rds for _, c in
             tw_by_key.get((w['t'], td, w['r'], rd), tw.iloc[0:0]).iterrows()]
    if not cands:
        return None, 'no_plss_township'
    if w['mer'] and w['mer'] in MERIDIAN_LETTER:
        sub = [c for c in cands if MERIDIAN_LETTER[w['mer']] in c['PRINMER']]
        if sub:
            cands = sub
            rule = (rule + ' + ' if rule else '') + 'meridian letter'
    if len(cands) == 1:
        return cands[0], rule or 'only candidate'
    if rule == 'T/R on page':
        # a T/R elsewhere on the page that still leaves several meridians open was wrong
        # whenever a second rule broke the tie (county 0/15, document meridian 0/14)
        return None, 'ambiguous'
    # town named in the well's own heading
    pts = []
    for m in RE_PLACE.finditer(window):
        ws_ = m.group(1).split()
        for n in (3, 2, 1):
            if len(ws_) >= n and norm(' '.join(ws_[:n])) in places:
                pts += places[norm(' '.join(ws_[:n]))]
                break
    if pts:
        near = [c for c in cands if min(haversine_km(c['lat'], c['lon'], la, lo) for la, lo in pts) <= 40]
        if len({c['PLSSID'] for c in near}) == 1:
            return near[0], (rule + ' + ' if rule else '') + 'town in well heading'
    named = [c for c in cands if str(county_of.get(c['PLSSID'])).lower() in page_counties]
    if len({c['PLSSID'] for c in named}) == 1:
        return named[0], (rule + ' + ' if rule else '') + 'county on page'
    if doc is not None and doc['title_counties']:
        named = [c for c in cands if str(county_of.get(c['PLSSID'])).lower() in doc['title_counties']]
        if len({c['PLSSID'] for c in named}) == 1:
            return named[0], (rule + ' + ' if rule else '') + 'document: county in title'
    if doc is not None and len(doc['meridians']) == 1:
        sub = [c for c in cands if c['PRINMER'] == next(iter(doc['meridians']))]
        if len(sub) == 1:
            return sub[0], (rule + ' + ' if rule else '') + 'document: meridian'
    return None, 'ambiguous'


def well_key(w):
    return (w['t'], w['r'], w['sec'], w['tract'], w['seq'])


# ------------------------------------------------------------------ main
meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location', 'townships_ranges_sections',
                            'water_type', 'Title'])
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()]
wt = meta['water_type'].astype(str).str.lower()
meta = meta[wt.str.contains('ground|spring|irrigat|well|quality|other|nan') |
            ~wt.str.contains('stream|reservoir|precip')]

# ---- pass 1: page-level resolution; collect report-level evidence
pages, docs = {}, {}
for (doc, page), g in meta.groupby(['id', 'page_number']):
    doc, page = int(doc), int(page)
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not p.exists():
        continue
    text = ' '.join(c.get('content', '') for c in json.loads(p.read_text(encoding='utf-8'))['result']['chunks'])
    ws = parse_wells(text)
    ws += parse_section_ids(text, [(w['start'], w['end']) for w in ws])
    if not ws:
        continue
    page_pairs = {(int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)) for m in RE_TRHEAD.finditer(text)}
    page_pairs |= {(w['t'], w['td'], w['r'], w['rd']) for w in ws if w['td'] and w['rd']}
    page_lc = text.lower()
    page_counties = {c for c in COUNTY_NAMES if re.search(rf'\b{re.escape(c)}\b', page_lc)}
    uniq = {}
    for w in ws:
        k = well_key(w)
        if k in uniq and uniq[k]['td'] and uniq[k]['rd']:
            continue
        w['window'] = text[w['end']:w['end'] + 300]
        w['heading'] = text[w['end']:w['end'] + 120]
        c, rule = resolve(w, page_pairs, w['window'], page_counties)
        w.update({'dir_rule': rule, 'scope': 'page', 'cand': c})
        uniq[k] = w
    pages[(doc, page)] = (g, uniq, page_pairs, page_counties)
    d = docs.setdefault(doc, {'pairs': {}, 'quadrants': set(), 'meridians': set(),
                              'title_counties': title_counties(g['Title'].iloc[0], COUNTY_NAMES)})
    for (t, td, r, rd) in page_pairs:
        d['pairs'].setdefault((t, r), set()).add((td, rd))
        d['quadrants'].add((td, rd))
    for w in uniq.values():          # meridians settled by page evidence
        if w['cand'] is not None:
            d['meridians'].add(w['cand']['PRINMER'])

# ---- pass 2: report-level evidence for wells the page left ambiguous
for (doc, page), (g, uniq, page_pairs, page_counties) in pages.items():
    for w in uniq.values():
        if w['cand'] is None and w['dir_rule'] == 'ambiguous':
            c, rule = resolve(w, page_pairs, w['window'], page_counties, doc=docs[doc])
            if c is not None:
                w.update({'cand': c, 'dir_rule': rule, 'scope': 'document'})

# ---- match rows to wells
wells_out, row_out = [], []
for (doc, page), (g, uniq, page_pairs, page_counties) in pages.items():
    for w in uniq.values():
        c = w['cand']
        w.update({'plssid': c['PLSSID'] if c is not None else None,
                  'meridian': c['PRINMER'] if c is not None else None,
                  'td_res': c['TWNSHPDIR'] if c is not None else None,
                  'rd_res': c['RANGEDIR'] if c is not None else None})
        wells_out.append({'id': doc, 'page_number': page, **{x: w[x] for x in
                          ['t', 'td', 'r', 'rd', 'sec', 'tract', 'seq', 'mer', 'form', 'dir_rule', 'scope',
                           'plssid', 'meridian', 'td_res', 'rd_res']}})
    resolved = {k: w for k, w in uniq.items() if w['plssid']}

    for _, row in g.iterrows():
        rt = f"{row['watersource_name']} {row['location']} {row['townships_ranges_sections']}"
        rec = {'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
               'n_wells_on_page': len(uniq), 'n_resolved_on_page': len(resolved)}
        pick, how = None, None
        # (a) well number in the LLM text that is also on the page
        rw = parse_wells(rt)
        hits = {well_key(w) for w in rw} & set(uniq)
        if not hits:
            sids = {(int(m.group(1)), m.group(2), int(m.group(3))) for m in RE_SECID.finditer(rt)}
            hits = {k for k in uniq if (k[2], k[3], k[4]) in sids}
        if len(hits) == 1:
            pick, how = uniq[hits.pop()], 'well number in LLM text'
        # (b) owner name in the well heading
        if pick is None:
            rwords = words(row['watersource_name'])
            if rwords:
                score = {k: len(rwords & words(w['heading'])) for k, w in uniq.items()}
                best = max(score.values())
                top = [k for k, v in score.items() if v == best]
                if best > 0 and len(top) == 1:
                    pick, how = uniq[top[0]], 'owner name in well heading'
        # (c) only one well on the page
        if pick is None and len(uniq) == 1:
            pick, how = next(iter(uniq.values())), 'only well on page'
        if pick is not None:
            rec.update({'match_rule': how, 'well': f"{pick['t']}{pick['td_res'] or pick['td'] or '?'}/"
                        f"{pick['r']}{pick['rd_res'] or pick['rd'] or '?'}-{pick['sec']}{pick['tract']}{pick['seq']}",
                        'form': pick['form'], 'dir_rule': pick['dir_rule'], 'scope': pick['scope'],
                        'plssid': pick['plssid'], 'meridian': pick['meridian'], 'sec': pick['sec'],
                        'tract': pick['tract'], 'seq': pick['seq'], 'td_res': pick['td_res'],
                        'rd_res': pick['rd_res'], 't': pick['t'], 'r': pick['r'],
                        'status': 'ok' if pick['plssid'] else pick['dir_rule']})
        elif len(resolved) > 1:
            rec.update({'match_rule': 'group', 'status': 'group',
                        'scope': 'document' if any(w['scope'] == 'document' for w in resolved.values()) else 'page',
                        'group_wells': ';'.join(f"{w['plssid']}|{w['sec']}" for w in resolved.values())})
        else:
            rec['status'] = 'no_row_match'
        row_out.append(rec)

rows = pd.DataFrame(row_out)
wells = pd.DataFrame(wells_out)
print(f'Pages with well numbers: {wells[["id", "page_number"]].drop_duplicates().shape[0]:,}; '
      f'distinct wells: {len(wells):,}')
print(wells['dir_rule'].value_counts().to_string())
print(wells['scope'].value_counts().to_string())

# ------------------------------------------------------------------ coordinates
need = set(wells['plssid'].dropna())
secs = sections(need)


def sec_xy(plssid, sec):
    try:
        return tuple(secs.loc[(plssid, sec)])
    except KeyError:
        return (np.nan, np.nan)


ok = rows['status'] == 'ok'
xy = [sec_xy(p, s) for p, s in zip(rows.loc[ok, 'plssid'], rows.loc[ok, 'sec'])]
rows.loc[ok, 'sec_lat'] = [a for a, _ in xy]
rows.loc[ok, 'sec_lon'] = [b for _, b in xy]
mer_letter = {'Mount Diablo': 'M', 'San Bernardino': 'S', 'Humboldt': 'H'}
rows['nwis_swn'] = None
for i in rows.index[ok]:
    m = next((v for k, v in mer_letter.items() if k in str(rows.at[i, 'meridian'])), None)
    if m:
        rows.at[i, 'nwis_swn'] = (f"{int(rows.at[i, 't']):03d}{rows.at[i, 'td_res']}{int(rows.at[i, 'r']):03d}"
                                  f"{rows.at[i, 'rd_res']}{int(rows.at[i, 'sec']):02d}{rows.at[i, 'tract']}"
                                  f"{int(rows.at[i, 'seq']):03d}{m}")
hit = rows['nwis_swn'].isin(nwis_xy.index)
rows.loc[hit, 'nwis_lat'] = rows.loc[hit, 'nwis_swn'].map(nwis_xy['dec_lat_va'])
rows.loc[hit, 'nwis_lon'] = rows.loc[hit, 'nwis_swn'].map(nwis_xy['dec_long_va'])
rows.loc[hit, 'nwis_site_no'] = rows.loc[hit, 'nwis_swn'].map(nwis_xy['site_no'])

# group rows: centroid only if all wells within 5 km
for i in rows.index[rows['status'] == 'group']:
    pts = [sec_xy(*x.split('|')[:1], int(x.split('|')[1])) for x in rows.at[i, 'group_wells'].split(';')]
    pts = [q for q in pts if not np.isnan(q[0])]
    if pts:
        la, lo = np.mean([q[0] for q in pts]), np.mean([q[1] for q in pts])
        if max(haversine_km(la, lo, a, b) for a, b in pts) <= 5:
            rows.loc[i, ['status', 'sec_lat', 'sec_lon']] = ['ok_group', la, lo]

rows['wn_method'] = np.where(rows['nwis_lat'].notna(), 'nwis_well',
                    np.where(rows['status'] == 'ok_group', 'wellno_group',
                    np.where(rows['status'].eq('ok') & rows['sec_lat'].notna(), 'plss_wellno', None)))
rows['wn_lat'] = rows['nwis_lat'].fillna(rows['sec_lat'])
rows['wn_lon'] = rows['nwis_lon'].fillna(rows['sec_lon'])
rows.loc[rows['wn_method'].isna(), ['wn_lat', 'wn_lon']] = np.nan
rows.to_csv(lv / 'tier2b_well_numbers.csv', index=False)

print('\nRow status:\n', rows['status'].value_counts().to_string())
print('\nMatch rule (located rows):\n', rows.loc[rows['wn_method'].notna(), 'match_rule'].value_counts().to_string())
print('\nMethod:\n', rows['wn_method'].value_counts().to_string())

# ------------------------------------------------------------------ validation
val = []
v = rows[rows['nwis_lat'].notna() & rows['sec_lat'].notna()]
e = haversine_km(v['sec_lat'], v['sec_lon'], v['nwis_lat'], v['nwis_lon'])
for rule, ee in pd.Series(e.values, index=v['dir_rule'].values).groupby(level=0):
    val.append({'check': 'section centroid vs NWIS well', 'dir_rule': rule, 'n': len(ee),
                'median_km': ee.median(), 'le_2km': np.mean(ee <= 2), 'gt_10km': np.mean(ee > 10)})
t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])]
v = rows[rows['wn_method'].notna()].merge(t1[['meta_row', 'actual_latitude', 'actual_longitude']], on='meta_row')
if len(v):
    e = haversine_km(v['wn_lat'], v['wn_lon'], v['actual_latitude'], v['actual_longitude'])
    for m, ee in pd.Series(e.values, index=v['wn_method'].values).groupby(level=0):
        val.append({'check': 'well-number location vs printed coords', 'dir_rule': m, 'n': len(ee),
                    'median_km': ee.median(), 'le_2km': np.mean(ee <= 2), 'gt_10km': np.mean(ee > 10)})
# unbiased direction/meridian check: among inferred (not as-printed) wells for which ANY
# direction/meridian option exists in NWIS, how often is the chosen option the NWIS one?
nwis_names = set(nwis_xy.index)
inf = rows[(rows['status'] == 'ok') & ~rows['dir_rule'].astype(str).str.startswith('as printed')]
for rule, gg in inf.groupby('dir_rule'):
    hits = []
    for x in gg.itertuples():
        opts = {f"{int(x.t):03d}{td}{int(x.r):03d}{rd}{int(x.sec):02d}{x.tract}{int(x.seq):03d}{m}"
                for td in 'NS' for rd in 'EW' for m in 'MSH'} & nwis_names
        if opts:
            hits.append(x.nwis_swn in opts)
    if hits:
        val.append({'check': 'chosen direction/meridian = NWIS option', 'dir_rule': rule, 'n': len(hits),
                    'median_km': np.nan, 'le_2km': np.mean(hits), 'gt_10km': np.nan})
val = pd.DataFrame(val)
val.to_csv(lv / 'tier2b_well_numbers_validation.csv', index=False)
print('\nValidation:\n', val.round(3).to_string(index=False))
loc = pd.read_csv(lv / 'locations.csv', usecols=['meta_row', 'coord_method'])
un = set(loc.loc[loc['coord_method'] == 'unlocated', 'meta_row'])
print(f"\nNew locations for currently unlocated rows: {rows[rows['wn_method'].notna() & rows['meta_row'].isin(un)].shape[0]:,}")
