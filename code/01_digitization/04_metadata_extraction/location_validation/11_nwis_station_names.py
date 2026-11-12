"""
Location tier 3c: NWIS station NAME (no station number on the page).  Added 2026-11-12.

Why: the Santa Ynez audit found ~45 rows like "Santa Ynez River near Santa Barbara" placed on
the GNIS point of the TOWN (tier 3b "near Place"), 8 km outside the watershed and on the other
side of the mountains. USGS names its stations "<stream> <at|near|above|below> <place>", and
the NWIS station list gives the gauge's own coordinates ("SANTA YNEZ R BL GIBRALTAR DAM NR
SANTA BARBARA CA"). Older Water Supply Papers print the station name but no 8-digit number,
so tier 3a (05_nwis_station_numbers.py) cannot use them.

Rule (same-page evidence only; the LLM text is a pointer):
  1. Parse the row's water-source name (plus the location text if it starts with at/near/
     above/below) into  stream + [(qualifier, place), ...].  A stream name alone is not a
     location (no qualifier -> skipped).
  2. The stream name AND every place must be printed on the row's own page.
  3. Candidate NWIS CA stations: same stream name (abbreviations expanded: R -> RIVER,
     C -> CREEK, NF -> NORTH FORK, CYN -> CANYON ...) and EVERY row (qualifier, place) also
     in the station name ("near" and "at" are interchangeable; above/below must agree).
       exact    station name has exactly the row's places              -> that station
       unique   one station contains the row's places (station has more) -> that station
       cluster  several, all within 5 km of each other                  -> centroid
       else     ambiguous (not used)
Out-of-state rows (row_states.csv: oos / either) are skipped: the list is California only.
Run on ALL rows so the method can be validated against tier-1 printed coordinates.

Inputs:  data/metadata/main_metadata.csv, data/analysis/page_blocks.parquet,
         data/analysis/nwis_sites/ca_stream_sites_expanded.csv,
         data/analysis/location_validation/{tier1_printed, row_states}.csv
Outputs: data/analysis/location_validation/tier3c_nwis_name.csv,
         tier3c_nwis_name_validation.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnis_utils import norm, hav  # noqa: E402

root = Path(__file__).resolve().parents[4]
lv = root / 'data/analysis/location_validation'

from nwis_name_utils import EXPAND, QUAL_ROW, QUAL_NWIS, parse, row_text, load_stations  # noqa: E402


# ------------------------------------------------------------------ NWIS
nwis, by_stream = load_stations(root)


def match(stream, segs):
    c = by_stream.get(stream)
    if c is None:
        return None
    hits = [(r, r.segs == segs) for r in c.itertuples() if segs <= r.segs]
    if not hits:
        return None
    exact = [r for r, e in hits if e]
    if len(exact) == 1:
        return 'exact', exact[0].site_no, exact[0].dec_lat_va, exact[0].dec_long_va, exact[0].station_nm
    if len(hits) == 1:
        r = hits[0][0]
        return 'unique', r.site_no, r.dec_lat_va, r.dec_long_va, r.station_nm
    la = np.array([r.dec_lat_va for r, _ in hits]); lo = np.array([r.dec_long_va for r, _ in hits])
    if (hav(la.mean(), lo.mean(), la, lo) <= 5).all():
        return 'cluster', ';'.join(r.site_no for r, _ in hits), la.mean(), lo.mean(), \
            ' | '.join(r.station_nm for r, _ in hits)
    return 'ambiguous', None, None, None, None


# ------------------------------------------------------------------ rows
meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location'])
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()].copy()
rs = pd.read_csv(lv / 'row_states.csv', usecols=['meta_row', 'row_class']).set_index('meta_row')['row_class']
meta['row_class'] = meta['meta_row'].map(rs).fillna('ca')

blocks = pd.read_parquet(root / 'data/analysis/page_blocks.parquet', columns=['id', 'page_number', 'content'])
blocks = blocks[blocks['id'].notna() & blocks['page_number'].notna()]
page_txt = (blocks.groupby([blocks['id'].astype(float).astype(int), blocks['page_number'].astype(int)])['content']
            .apply(lambda x: ' ' + norm(' '.join(map(str, x))) + ' '))

recs = []
for r in meta.itertuples():
    rec = {'meta_row': r.meta_row, 'id': int(r.id), 'page_number': int(r.page_number)}
    if r.row_class != 'ca':
        rec['status'] = 'out_of_state_row'
        recs.append(rec); continue
    text = row_text(r.watersource_name, r.location)
    stream, segs = parse(text, QUAL_ROW)
    if not stream or not segs:
        rec['status'] = 'no_qualified_name'
        recs.append(rec); continue
    page = page_txt.get((int(r.id), int(r.page_number)))
    if page is None:
        rec['status'] = 'no_page_text'
        recs.append(rec); continue
    # same-page evidence: the stream name (as the LLM wrote it, minus the type word) and every place
    left = re.split(r'\b(?:at|near|nr|above|below|ab|bl)\b', text, maxsplit=1, flags=re.I)[0]
    s_core = norm(re.split(r'\b(?:river|creek|r|c|cr|ck)\b', left, maxsplit=1, flags=re.I)[0])
    places = [p for _, p in segs]
    if (s_core and f' {s_core} ' not in page) or any(f' {norm(p)} ' not in page for p in places):
        rec['status'] = 'name_not_on_page'
        recs.append(rec); continue
    m = match(stream, segs)
    if m is None:
        rec['status'] = 'no_nwis_match'
    elif m[0] == 'ambiguous':
        rec['status'] = 'ambiguous'
    else:
        rec.update({'status': 'ok', 'match': m[0], 'site_no': m[1], 'lat': m[2], 'lon': m[3],
                    'station_nm': m[4], 'row_name': text})
    recs.append(rec)

t = pd.DataFrame(recs)
t.to_csv(lv / 'tier3c_nwis_name.csv', index=False)
print(f'Rows: {len(t):,}')
print(t['status'].value_counts().to_string())
print(t.loc[t.status == 'ok', 'match'].value_counts().to_string())

# ------------------------------------------------------------------ validation vs printed coords
t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])]
v = t[t.status == 'ok'].merge(t1[['meta_row', 'actual_latitude', 'actual_longitude']], on='meta_row')
v['err_km'] = hav(v['lat'], v['lon'], v['actual_latitude'], v['actual_longitude'])
val = (v.groupby('match')['err_km']
        .agg(n='size', median_km='median', le_1km=lambda e: np.mean(e <= 1),
             le_5km=lambda e: np.mean(e <= 5), gt_25km=lambda e: np.mean(e > 25)))
val.loc['all'] = [len(v), v.err_km.median(), np.mean(v.err_km <= 1), np.mean(v.err_km <= 5), np.mean(v.err_km > 25)]
val.to_csv(lv / 'tier3c_nwis_name_validation.csv')
print('\nValidation vs tier-1 printed coordinates:')
print(val.round(3).to_string())
