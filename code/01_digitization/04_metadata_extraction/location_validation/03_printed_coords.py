"""
Location tier 1: coordinates PRINTED on the page.

Location hierarchy (agreed 2026-10-28; LLM-inferred coordinates are not used):
  1 printed lat/long confirmed in the page text   <- this script
  2 Township/Range/Section -> PLSS centroid        (02_trs_lookup_all.py)
  3 named site (NWIS station name, GNIS place)
  4 coarse area (county)        5 unlocated

An LLM "actual" coordinate is accepted as tier 1 only if BOTH its latitude and its
longitude match a coordinate printed in the page's Reducto text. USGS station
descriptions print them as  lat 37°40'04", long 120°27'26"  (OCR often garbles
the degree sign, e.g. 37�40'04"), sometimes degrees-minutes only (lat 39°52').

Tolerance: 0.005 deg (~0.5 km) when seconds are printed, 0.02 deg (~2 km; one
arc-minute plus rounding) for degrees-minutes. Longitudes are printed without a
sign (west), so absolute values are compared.

Two strictness levels are recorded:
  keyword  the printed coordinate is within 15 characters after "lat"/"long"
  any      any degree-minute(-second) number on the page (looser; used only to
           report how much the keyword rule might miss)
tier 1 uses `keyword`.

Inputs:  data/metadata/main_metadata.csv, data/digitized/{id}/page_{n}/*.json
Outputs: data/analysis/location_validation/
           printed_coords_by_page.csv   every printed lat/long found, per page
           tier1_printed.csv            one row per main_metadata row with an LLM
                                        "actual" coordinate: match status
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

root    = Path(__file__).resolve().parents[4]
out_dir = root / 'data/analysis/location_validation'
out_dir.mkdir(parents=True, exist_ok=True)

TOL_SEC = 0.005   # deg, when seconds are printed
TOL_MIN = 0.02    # deg, degrees-minutes only

# degrees, minutes, optional seconds; separators are OCR-garbled symbols/spaces
_DMS = r'(\d{2,3})\s?[^\w\s,;]{0,3}\s?(\d{1,2})\s?[^\w\s,;]{0,3}\s?(?:(\d{1,2}(?:\.\d+)?)\s?[^\w\s,;]{0,3})?'
RE_LAT = re.compile(r'\blat(?:itude)?\b[.,:\s-]{0,4}' + _DMS, re.I)
RE_LON = re.compile(r'\blong?(?:itude)?\b[.,:\s-]{0,4}' + _DMS, re.I)
RE_ANY = re.compile(r'(?<!\d)' + _DMS.replace(r'(\d{2,3})', r'(\d{2,3})', 1), re.I)


def to_deg(d, m, s):
    d, m = int(d), int(m)
    if m >= 60 or (s and float(s) >= 60):
        return None, None
    val = d + m / 60 + (float(s) / 3600 if s else 0)
    return val, (TOL_SEC if s else TOL_MIN)


def extract(text):
    """Printed latitudes / longitudes: lists of (value, tolerance, source)."""
    lats, lons = [], []
    for rx, out, lo, hi in [(RE_LAT, lats, 24, 50), (RE_LON, lons, 66, 125)]:
        for m in rx.finditer(text):
            v, tol = to_deg(*m.groups())
            if v and lo <= v <= hi:
                out.append((v, tol, 'keyword'))
    for m in RE_ANY.finditer(text):
        v, tol = to_deg(*m.groups())
        if v is None:
            continue
        if 24 <= v <= 50:
            lats.append((v, tol, 'any'))
        elif 66 <= v <= 125:
            lons.append((v, tol, 'any'))
    return lats, lons


def matches(value, printed, level):
    ok = [p for p in printed if (level == 'any' or p[2] == 'keyword')
          and abs(abs(value) - p[0]) <= p[1]]
    return len(ok) > 0


meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False)
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()]

page_rows, recs = [], []
for (doc, page), g in meta.groupby(['id', 'page_number']):
    doc, page = int(doc), int(page)
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    text = None
    if p.exists():
        j = json.loads(p.read_text(encoding='utf-8'))
        text = ' '.join(c.get('content', '') for c in j['result']['chunks'])
    lats, lons = extract(text) if text else ([], [])
    kw_lats = [l for l in lats if l[2] == 'keyword']
    kw_lons = [l for l in lons if l[2] == 'keyword']
    page_rows.append({'id': doc, 'page_number': page, 'has_text': text is not None,
                      'n_printed_lat': len(kw_lats), 'n_printed_lon': len(kw_lons),
                      'printed_lats': ';'.join(f'{v:.5f}' for v, _, _ in kw_lats),
                      'printed_lons': ';'.join(f'{-v:.5f}' for v, _, _ in kw_lons)})
    for _, row in g[g['actual_latitude'].notna()].iterrows():
        la, lo = row['actual_latitude'], row['actual_longitude']
        if text is None:
            status = 'no_page_text'
        elif pd.isna(lo):
            status = 'actual_incomplete'
        elif matches(la, lats, 'keyword') and matches(lo, lons, 'keyword'):
            status = 'printed_match'
        elif matches(la, lats, 'any') and matches(lo, lons, 'any'):
            status = 'printed_match_no_keyword'
        elif kw_lats or kw_lons:
            status = 'printed_other_values'     # page prints coords, but not these
        else:
            status = 'not_printed'
        recs.append({'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
                     'water_type': row['water_type'], 'actual_latitude': la,
                     'actual_longitude': lo, 'tier1_status': status})

pages = pd.DataFrame(page_rows)
pages.to_csv(out_dir / 'printed_coords_by_page.csv', index=False)
t1 = pd.DataFrame(recs)
t1.to_csv(out_dir / 'tier1_printed.csv', index=False)

print(f'Pages scanned: {len(pages):,}  (with page text: {pages["has_text"].sum():,}; '
      f'printing >=1 keyword lat AND lon: {((pages.n_printed_lat > 0) & (pages.n_printed_lon > 0)).sum():,})')
print(f'\nLLM "actual" rows: {len(t1):,}')
print(t1['tier1_status'].value_counts().to_string())
print('\nShare by status:\n', t1['tier1_status'].value_counts(normalize=True).round(3).to_string())
