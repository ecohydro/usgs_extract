"""
Location tier 4: inherit a location along an explicit "Continued" chain.

Approved 2026-10-28 (Annette): many groundwater tables give a well's location only on
the first page of a table that continues for many pages. Labeled evidence_scope =
'continuation' in locations.csv so figures can include or exclude it.

Linking (conservative, explicit evidence only):
  page p is linked back to p-1 only if
    - page p-1 exists in data/digitized (no gaps),
    - page p prints a continuation marker ("Continued", "(cont.)", "-Con."; OCR variants), and
    - p and p-1 share the same table number, or similar table titles (word Jaccard >= 0.5),
      or similar column headers (Jaccard >= 0.6).
  The chain is followed back up to MAX_DEPTH pages.

Inheriting (row level, nearest linked page first):
  a) the row's well number (from its LLM text) matches a located row's well number, or
  b) a distinctive owner/site word in the row's LLM water-source name matches exactly one
     located row on that page, or
  c) every located row on the nearest linked page that has any is within 2 km
     (a single site) and the row names no well number of its own.
  Sources are rows located by PAGE or DOCUMENT evidence only (never by continuation).
  The inherited row keeps the source's coord_method / precision.

Only rows that are not stream discharge / reservoir / precipitation are processed
(same scope as 07_well_numbers.py).

Validation (hold-out): the same procedure is run on rows that ARE located by their own page;
the inherited location is compared with their own location.

Inputs:  data/metadata/main_metadata.csv, data/digitized/{id}/page_{n}/*.json,
         data/analysis/location_validation/locations.csv
Outputs: data/analysis/location_validation/tier4_continuation.csv,
         tier4_continuation_validation.csv
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plss_utils import haversine_km  # noqa: E402

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'
MAX_DEPTH = 15
# STRICT (after hold-out run 1: section-only IDs, 1-word owner matches and the single-site
# rule gave 16-42% errors > 10 km): full well numbers only, >= 2 shared owner words, no
# single-site guessing.
STRICT = True

RE_CONT  = re.compile(r'\bconti[a-z]{2,4}d\b|\(cont\.?\)|-\s?con\.|\bcontd\b', re.I)
RE_TABLE = re.compile(r'\btable\s+(\d{1,3}[a-z]?)\b', re.I)
RE_WELL  = re.compile(r'(?<![\d/.])(\d{1,2})\s?[NS5]?\s?/\s?(\d{1,2})\s?[EWF]?\s?[-–—]\s?'
                      r'(\d{1,2})\s?([A-HJ-NP-R8])\s?(\d{1,3})')
RE_SECID = re.compile(r'(?<![\w/.\-])(\d{1,2})([A-HJ-NP-R])(\d{1,2})(?![\w/])')
GENERIC = {'WELL', 'WELLS', 'DOMESTIC', 'IRRIGATION', 'OBSERVATION', 'DRILLED', 'CITY', 'COMPANY',
           'WATER', 'TEST', 'STOCK', 'UNUSED', 'ARTESIAN', 'TABLE', 'VALLEY', 'COUNTY', 'BASIN',
           'AREA', 'GROUND', 'GROUNDWATER', 'SUPPLY', 'MUNICIPAL', 'PUBLIC', 'RANCH', 'CALIFORNIA',
           'CALIF', 'NEAR', 'ABOUT', 'MILES', 'MILE', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'INDUSTRIAL',
           'SPRING', 'SPRINGS', 'CREEK', 'RIVER', 'UNKNOWN', 'CONTINUED'}


def words(s):
    return {w for w in re.findall(r'[A-Z]{4,}', str(s).upper()) if w not in GENERIC}


def wkeys(s):
    """Well identifiers in text: full/short keys (t, r, sec, tract, seq) and (sec, tract, seq)."""
    s = str(s)
    full = {(int(a), int(b), int(c), 'B' if d == '8' else d, int(e)) for a, b, c, d, e in RE_WELL.findall(s)}
    sec = {(k[2], k[3], k[4]) for k in full}
    sec |= {(int(a), b, int(c)) for a, b, c in RE_SECID.findall(s) if 1 <= int(a) <= 36}
    return full, sec


def page_info(doc, page):
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not p.exists():
        return None
    text = ' '.join(c.get('content', '') for c in json.loads(p.read_text(encoding='utf-8'))['result']['chunks'])
    t = RE_TABLE.search(text)
    head = re.split(r'<table', text, maxsplit=1)[0][:200]
    tr = re.search(r'<tr>(.*?)</tr>', text, re.S)
    clean = lambda x: {w for w in re.findall(r'[a-z]{3,}', x.lower()) if not w.startswith('conti')}
    return {'cont': bool(RE_CONT.search(text)), 'table': t.group(1).lower() if t else None,
            'title': clean(head), 'header': clean(re.sub(r'<[^>]+>', ' ', tr.group(1))) if tr else set()}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


_cache = {}


def info(doc, page):
    if (doc, page) not in _cache:
        _cache[(doc, page)] = page_info(doc, page)
    return _cache[(doc, page)]


def linked(doc, page):
    """True if `page` explicitly continues `page - 1`."""
    a, b = info(doc, page), info(doc, page - 1)
    if a is None or b is None or not a['cont']:
        return False
    return ((a['table'] is not None and a['table'] == b['table']) or
            jac(a['title'], b['title']) >= 0.5 or jac(a['header'], b['header']) >= 0.6)


# ------------------------------------------------------------------ data
meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location', 'water_type'])
meta['meta_row'] = np.arange(len(meta))
loc = pd.read_csv(lv / 'locations.csv', low_memory=False)
meta = meta.merge(loc[['meta_row', 'lat', 'lon', 'coord_method', 'precision_km', 'precision_class',
                       'evidence_scope', 'well_number']], on='meta_row')
meta = meta[meta['id'].notna() & meta['page_number'].notna()]
wt = meta['water_type'].astype(str).str.lower()
meta = meta[wt.str.contains('ground|spring|irrigat|well|quality|other|nan') |
            ~wt.str.contains('stream|reservoir|precip')].copy()
meta[['id', 'page_number']] = meta[['id', 'page_number']].astype(int)
meta['located'] = meta['coord_method'].ne('unlocated') & meta['evidence_scope'].isin(['page', 'document'])
by_page = {k: g for k, g in meta.groupby(['id', 'page_number'])}


def inherit(row):
    """Walk back along the continuation chain -> dict or None."""
    full, sec = wkeys(f"{row['watersource_name']} {row['location']}")
    rwords = words(row['watersource_name'])
    doc, page = row['id'], row['page_number']
    for depth in range(1, MAX_DEPTH + 1):
        if not linked(doc, page - depth + 1):
            return None
        src = by_page.get((doc, page - depth))
        if src is None:
            continue
        src = src[src['located']]
        if src.empty:
            continue
        # a) well number
        if full or sec:
            hit = [r for _, r in src.iterrows()
                   if (full & wkeys(f"{r['well_number']} {r['watersource_name']}")[0]) or
                   (not STRICT and sec & wkeys(f"{r['well_number']} {r['watersource_name']}")[1])]
            if len({(h['lat'], h['lon']) for h in hit}) == 1:
                return {'rule': 'well number', 'depth': depth, 'src': hit[0]}
            continue          # the row names its own well: never fall back to a site guess
        # b) owner / site word
        if rwords:
            score = [(len(rwords & words(r['watersource_name'])), r) for _, r in src.iterrows()]
            best = max(s for s, _ in score)
            top = [r for s, r in score if s == best]
            if best >= (2 if STRICT else 1) and len({(t['lat'], t['lon']) for t in top}) == 1:
                return {'rule': 'owner/site name', 'depth': depth, 'src': top[0]}
        # c) single site on the nearest linked page with located rows
        if STRICT:
            return None
        la, lo = src['lat'].mean(), src['lon'].mean()
        if (haversine_km(la, lo, src['lat'], src['lon']) <= 2).all():
            return {'rule': 'single site on linked page', 'depth': depth, 'src': src.iloc[0]}
        return None
    return None


def run(rows):
    out = []
    for _, row in rows.iterrows():
        res = inherit(row)
        rec = {'meta_row': row['meta_row'], 'id': row['id'], 'page_number': row['page_number'],
               'status': 'ok' if res else 'no_link'}
        if res:
            s = res['src']
            rec.update({'rule': res['rule'], 'link_depth': res['depth'],
                        'source_page': row['page_number'] - res['depth'],
                        'source_meta_row': s['meta_row'], 'source_method': s['coord_method'],
                        'source_precision_km': s['precision_km'],
                        'source_precision_class': s['precision_class'],
                        'lat': s['lat'], 'lon': s['lon']})
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ apply to unlocated rows
targets = meta[~meta['located']]     # not located by page/document evidence (independent of earlier runs)
print(f'Unlocated rows in scope: {len(targets):,}')
res = run(targets)
res.to_csv(lv / 'tier4_continuation.csv', index=False)
print(res['status'].value_counts().to_string())
print(res.loc[res['status'] == 'ok', 'rule'].value_counts().to_string())
print('precision class inherited:\n', res.loc[res['status'] == 'ok', 'source_precision_class'].value_counts().to_string())
print('link depth:\n', res.loc[res['status'] == 'ok', 'link_depth'].describe().round(1).to_string())

# ------------------------------------------------------------------ hold-out validation
hold = meta[meta['located'] & meta['evidence_scope'].eq('page')]
hv = run(hold)
hv = hv[hv['status'] == 'ok'].merge(hold[['meta_row', 'lat', 'lon']].rename(columns={'lat': 'own_lat', 'lon': 'own_lon'}),
                                    on='meta_row')
hv['err_km'] = haversine_km(hv['lat'], hv['lon'], hv['own_lat'], hv['own_lon'])
hv = hv.merge(hold[['meta_row', 'precision_class']].rename(columns={'precision_class': 'own_class'}), on='meta_row')
hv['pair'] = hv['source_precision_class'] + ' source / ' + hv['own_class'] + ' own'
hv.to_csv(lv / 'tier4_continuation_holdout.csv', index=False)
print(hv.groupby(['rule', 'pair'])['err_km'].agg(n='size', median_km='median',
      le_2km=lambda e: np.mean(e <= 2), gt_10km=lambda e: np.mean(e > 10)).round(3).to_string())
val = (hv.groupby('rule')['err_km']
         .agg(n='size', median_km='median', le_2km=lambda e: np.mean(e <= 2),
              le_5km=lambda e: np.mean(e <= 5), gt_10km=lambda e: np.mean(e > 10)))
val.to_csv(lv / 'tier4_continuation_validation.csv')
print(f'\nHold-out validation (rows located by their own page, n={len(hv):,}):')
print(val.round(3).to_string())
