"""
Shared GNIS named-place matching (used by 06_gnis_named_places.py for California pages
and 10_out_of_state.py for pages in other states), so both apply identical rules.
"""
import re

import numpy as np
import pandas as pd

POINT_CLASSES = {'Populated Place', 'Locale', 'Dam', 'Reservoir', 'Lake', 'Spring',
                 'Post Office', 'Crossing', 'Falls', 'Rapids', 'Bridge', 'Mine', 'Building',
                 'Station', 'Tunnel', 'Well', 'Gap'}
SUFFIX_CLASS = {'reservoir': {'Reservoir', 'Lake', 'Dam'}, 'lake': {'Lake', 'Reservoir'},
                'dam': {'Dam'}, 'spring': {'Spring'}, 'springs': {'Spring', 'Populated Place'}}
STOP = r'(?:,|\.|;|\(|\bcalif|\bcalifornia\b|\bcounty\b|\bin\b|\bon\b|\band\b|\bfrom\b|\bat\b|$)'
RE_QUAL = re.compile(r'\b(at|near|nr\.?|above|below|ab\.?|bl\.?)\s+([A-Z][\w\'.-]*(?:\s+[A-Z][\w\'.-]*){0,3}?)\s*' + STOP)
RE_FEAT = re.compile(r'\b([A-Z][\w\'.-]*(?:\s+[A-Z][\w\'.-]*){0,3}\s+(Reservoir|Lake|Dam|Springs?))\b')
RE_PLACE = re.compile(r'^\s*(?:near\s+|nr\.?\s+|vicinity of\s+)?([A-Z][\w\'.-]*(?:\s+[A-Z][\w\'.-]*){0,2})'
                      r'(?:\s+area)?\s*(?:,\s*(?:calif\.?|california|ca)\.?)?\s*$', re.I)
OUT_OF_STATE = re.compile(r'\b(colo|oreg|nev|ariz|wash|wyo|mont)\.|\butah\b|\bidaho\b|\bn\. ?mex\.')
GENERIC_PLACE = {'river', 'creek', 'the', 'station', 'gaging', 'mouth', 'head', 'diversion',
                 'canal', 'ditch', 'intake', 'dam', 'bridge', 'well', 'wells', 'county', 'valley'}


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))



def load_gnis(path, states):
    """Point-like GNIS features for the given state codes, grouped by normalized name."""
    g = pd.read_csv(path, sep='|', encoding='utf-8-sig', low_memory=False) if not isinstance(path, (list, tuple))         else pd.concat([pd.read_csv(p, sep='|', encoding='utf-8-sig', low_memory=False) for p in path])
    g = g[g['STATE_ALPHA'].isin(states) & g['FEATURE_CLASS'].isin(POINT_CLASSES) &
          (g['PRIM_LAT_DEC'] != 0)].copy()
    g['key'] = g['FEATURE_NAME'].map(norm)
    return {k: d[['FEATURE_NAME', 'FEATURE_CLASS', 'COUNTY_NAME', 'STATE_ALPHA', 'PRIM_LAT_DEC', 'PRIM_LONG_DEC']]
            for k, d in g.groupby('key')}


def candidates(gn, name, classes=None):
    c = gn.get(norm(name))
    if c is None:
        return None
    return c[c['FEATURE_CLASS'].isin(classes)] if classes else c


def resolve(c, page_lc):
    """Pick one GNIS point from candidates using page evidence; (lat, lon, rule) or None."""
    if c is None or len(c) == 0:
        return None, 'not_in_gnis'
    lat0, lon0 = c['PRIM_LAT_DEC'].iloc[0], c['PRIM_LONG_DEC'].iloc[0]
    if (hav(lat0, lon0, c['PRIM_LAT_DEC'], c['PRIM_LONG_DEC']) <= 3).all():
        return (c['PRIM_LAT_DEC'].mean(), c['PRIM_LONG_DEC'].mean()), 'unique'
    named = c[c['COUNTY_NAME'].fillna('').map(lambda n: bool(n) and re.search(rf'\b{re.escape(n.lower())}\b', page_lc) is not None)]
    if len(named) and (hav(named['PRIM_LAT_DEC'].iloc[0], named['PRIM_LONG_DEC'].iloc[0],
                           named['PRIM_LAT_DEC'], named['PRIM_LONG_DEC']) <= 3).all():
        return (named['PRIM_LAT_DEC'].mean(), named['PRIM_LONG_DEC'].mean()), 'county named on page'
    return None, 'ambiguous'


def extract(ws, loc):
    """Ordered candidate (name, kind, classes) list from LLM water-source/location text."""
    out = []
    for text in (str(ws), str(loc)):
        if text == 'nan':
            continue
        for m in RE_FEAT.finditer(text):
            suf = m.group(2).lower()
            out.append((m.group(1), 'feature_' + suf.rstrip('s'), SUFFIX_CLASS[suf]))
        for m in RE_QUAL.finditer(text):
            q = m.group(1).lower().rstrip('.')
            q = {'nr': 'near', 'ab': 'above', 'bl': 'below'}.get(q, q)
            if norm(m.group(2)) not in GENERIC_PLACE:
                out.append((m.group(2), 'qualifier_' + q, None))
    m = RE_PLACE.match(str(loc)) if str(loc) != 'nan' else None
    if m and norm(m.group(1)) not in GENERIC_PLACE:
        out.append((m.group(1), 'place', {'Populated Place', 'Locale', 'Post Office'}))
    return out


