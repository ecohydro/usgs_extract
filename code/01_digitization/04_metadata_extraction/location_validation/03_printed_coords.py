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

Inputs:  data/analysis/metadata_base.csv (meta_source.py), data/digitized/{id}/page_{n}/*.json
Outputs: data/analysis/location_validation/
           printed_coords_by_page.csv   every printed lat/long found, per page
           tier1_printed.csv            one row per metadata row with a coordinates field: status
"""

import json
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

root    = Path(__file__).resolve().parents[4]
out_dir = root / 'data/analysis/location_validation'
out_dir.mkdir(parents=True, exist_ok=True)

TOL_SEC = 0.005   # deg, when seconds are printed
TOL_MIN = 0.02    # deg, degrees-minutes only

# degrees, minutes, optional seconds; separators are OCR-garbled symbols/spaces.
# Separators are captured so that garbled values with a MISSING component can be rejected.
_DMS = (r'(\d{2,3})\s?([^\w\s,;]{0,3})\s?(\d{1,2})\s?([^\w\s,;]{0,3})\s?'
        r'(?:(\d{1,2}(?:\.\d+)?)\s?[^\w\s,;]{0,3})?')
RE_LAT = re.compile(r'\blat(?:itude)?\b[.,:\s-]{0,4}' + _DMS, re.I)
RE_LON = re.compile(r'\blong?(?:itude)?\b[.,:\s-]{0,4}' + _DMS, re.I)
RE_ANY = re.compile(r'(?<!\d)' + _DMS, re.I)


def garbled(sep1, sep2, s):
    """True if OCR dropped a component (added 2026-11-12, found in the Santa Ynez audit).
    'Lat 38'00"' (true 34°38'00") was read as 38°00': the degrees were lost, and the minutes
    and seconds were taken as degrees and minutes. Two signatures:
      - two numbers, the second followed by a seconds mark (")  -> degrees or minutes missing
        ('38'00"', '39°59"', '33 10"'); the error can be a full degree or more.
      - a degree sign AFTER the second number ('38 3°35'') -> the first number is not degrees."""
    if not s and '"' in sep2:
        return True
    if s and not sep1 and ('\N{DEGREE SIGN}' in sep2 or '�' in sep2):
        return True
    return False


def to_deg(d, sep1, m, sep2, s):
    if garbled(sep1, sep2, s):
        return None, None
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
    """Closest printed value within tolerance of the LLM value, or None.
    Since 2026-11-14 the PRINTED value is stored (printed_latitude/longitude), not the LLM's
    decimal conversion: the LLM converted the same printed text inconsistently across reports
    (e.g. 121°59'35" -> -121.99250 / .99278 / .99306 / .99333), which split one gauge into
    several 'sites'. The LLM value is now only a pointer to which printed coordinate to use."""
    ok = [p for p in printed if (level == 'any' or p[2] == 'keyword')
          and abs(abs(value) - p[0]) <= p[1]]
    if not ok:
        return None
    return min(ok, key=lambda p: abs(abs(value) - p[0]))[0]


# ---- rows: the new per-location metadata (meta_source.py; 2026-11-18). Its `coordinates` field is the
# coordinate text copied from the page (found-on-page check), e.g. 'Lat 37°29'20", long 119°18'10"'.
# Tier 1 = that text parsed with the same rules as before (keyword lat/long, OCR-garbled values rejected).
# The model's text is still only a pointer: the parsed values must also be found among the coordinates
# printed on the page ('printed_match'); a value found only by the looser any-number rule is
# 'printed_match_no_keyword'. actual_latitude/longitude hold the PRINTED values (validation column
# names used by the other tiers).
from meta_source import load_meta  # noqa: E402

meta = load_meta(['id', 'page_number', 'water_type', 'coordinates'])
meta = meta[meta['id'].notna() & meta['page_number'].notna()]


def parse_field(c):
    """(lat, lon) from a coordinates field, keyword form first, else two bare DMS numbers."""
    lats, lons = extract(c)
    kl = [v for v, _, src in lats if src == 'keyword'] or [v for v, _, _ in lats]
    ko = [v for v, _, src in lons if src == 'keyword'] or [v for v, _, _ in lons]
    return (kl[0] if kl else None), (ko[0] if ko else None)


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
    for _, row in g[g['coordinates'].notna()].iterrows():
        la, lo = parse_field(row['coordinates'])
        pla = plo = None
        if la is None or lo is None:
            status = 'field_unparsed'          # garbled / incomplete / not lat-long (e.g. a grid number)
        elif text is None:
            status = 'no_page_text'
        elif (pla := matches(la, lats, 'keyword')) and (plo := matches(lo, lons, 'keyword')):
            status = 'printed_match'
        elif (pla := matches(la, lats, 'any')) and (plo := matches(lo, lons, 'any')):
            status = 'printed_match_no_keyword'
        else:
            status = 'not_on_page'
            pla = plo = None
        recs.append({'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
                     'water_type': row['water_type'], 'coordinates': row['coordinates'],
                     'tier1_status': status, 'printed_latitude': pla,
                     'printed_longitude': -plo if plo else None,
                     'actual_latitude': pla, 'actual_longitude': -plo if plo else None})

pages = pd.DataFrame(page_rows)
pages.to_csv(out_dir / 'printed_coords_by_page.csv', index=False)
t1 = pd.DataFrame(recs)
t1.to_csv(out_dir / 'tier1_printed.csv', index=False)

print(f'Pages scanned: {len(pages):,}  (with page text: {pages["has_text"].sum():,}; '
      f'printing >=1 keyword lat AND lon: {((pages.n_printed_lat > 0) & (pages.n_printed_lon > 0)).sum():,})')
print(f'\nRows with a coordinates field: {len(t1):,}')
print(t1['tier1_status'].value_counts().to_string())
print('\nShare by status:\n', t1['tier1_status'].value_counts(normalize=True).round(3).to_string())
