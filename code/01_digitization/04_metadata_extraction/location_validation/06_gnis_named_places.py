"""
Location tier 3b: named place on the page -> GNIS point.

Candidates come from the LLM's water-source name and location text (pointer only),
and each must ALSO appear in the row's own page text:
  qualifier   "<stream> at|near|nr|above|below <Place>"   -> the Place
  feature     a name ending in Reservoir / Lake / Dam / Spring(s)  -> that feature
  place       short location text: "Near Escondido", "Willows area, California"
Only point-like GNIS classes are used (towns, locales, dams, reservoirs, lakes,
springs, post offices, ...). Streams, valleys, ridges, basins are excluded: a
river name alone is not a location (agreed 2026-10-28).

GNIS = 2021 archived full California file (CA_Features_20210825.txt). The current
GNIS dropped locales, dams and post offices, which historical USGS pages rely on.

Disambiguation (page evidence only): one candidate -> ok; several -> the one whose
GNIS county is named on the page; candidates within 3 km of each other (e.g. a town
and its post office) are treated as one place. Otherwise: ambiguous.

Pages that print an out-of-state abbreviation (Colo., Oreg., Nev., Ariz., Utah, Idaho,
N. Mex., Wash., Wyo., Mont.) and no "Calif." are skipped: GNIS here is California-only, so
e.g. "Arkansas River at Poncha Creek" (Colorado) was matched to a California namesake.

Run on ALL rows; validated against tier-1 printed coordinates by sub-method.

Inputs:  data/metadata/main_metadata.csv, data/digitized/.../*.json,
         data/analysis/spatial/gnis/CA_Features_20210825.txt,
         data/analysis/location_validation/tier1_printed.csv
Output:  data/analysis/location_validation/tier3b_gnis.csv, tier3b_gnis_validation.csv
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'

from gnis_utils import (POINT_CLASSES, SUFFIX_CLASS, STOP, RE_QUAL, RE_FEAT, RE_PLACE,  # noqa: E402
                        OUT_OF_STATE, GENERIC_PLACE, norm, hav, resolve, extract, load_gnis)
import gnis_utils  # noqa: E402

# ------------------------------------------------------------------ GNIS
gn = load_gnis(root / 'data/analysis/spatial/gnis/CA_Features_20210825.txt', ['CA'])


def candidates(name, classes=None):
    return gnis_utils.candidates(gn, name, classes)


meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location'])
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()]

recs = []
for (doc, page), grp in meta.groupby(['id', 'page_number']):
    doc, page = int(doc), int(page)
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not p.exists():
        continue
    j = json.loads(p.read_text(encoding='utf-8'))
    raw = ' '.join(c.get('content', '') for c in j['result']['chunks'])
    page_lc, page_n = raw.lower(), ' ' + norm(raw) + ' '
    if OUT_OF_STATE.search(page_lc) and not re.search(r'\bcalif\b|\bcalifornia\b', page_lc):
        for _, row in grp.iterrows():
            recs.append({'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
                         't3b_status': 'out_of_state_page'})
        continue
    for _, row in grp.iterrows():
        rec = {'meta_row': row['meta_row'], 'id': doc, 'page_number': page, 't3b_status': 'no_candidate'}
        for name, kind, classes in extract(row['watersource_name'], row['location']):
            if f' {norm(name)} ' not in page_n:          # must be printed on the row's own page
                rec['t3b_status'] = 'name_not_on_page'
                continue
            res, rule = resolve(candidates(name, classes), page_lc)
            if res is None:
                rec['t3b_status'] = rule
                continue
            rec.update({'t3b_status': 'ok', 'place': name, 'kind': kind, 'rule': rule,
                        'gnis_lat': res[0], 'gnis_lon': res[1]})
            break
        recs.append(rec)

t3 = pd.DataFrame(recs)
t3.to_csv(lv / 'tier3b_gnis.csv', index=False)
print(f'Rows scanned: {len(t3):,}')
print(t3['t3b_status'].value_counts().to_string())

# ---- validation against tier-1 printed coordinates, by kind
t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])]
v = t3[t3['t3b_status'] == 'ok'].merge(t1[['meta_row', 'actual_latitude', 'actual_longitude']], on='meta_row')
v['err_km'] = hav(v['gnis_lat'], v['gnis_lon'], v['actual_latitude'], v['actual_longitude'])
val = (v.groupby('kind')['err_km']
        .agg(n='size', median_km='median',
             le_2km=lambda e: np.mean(e <= 2), le_5km=lambda e: np.mean(e <= 5),
             le_10km=lambda e: np.mean(e <= 10), gt_25km=lambda e: np.mean(e > 25))
        .sort_values('n', ascending=False))
val.to_csv(lv / 'tier3b_gnis_validation.csv')
print(f'\nValidation vs printed coords (n={len(v):,}): median {v.err_km.median():.1f} km; '
      f'<=5 km {np.mean(v.err_km <= 5):.0%}; <=10 km {np.mean(v.err_km <= 10):.0%}; >25 km {np.mean(v.err_km > 25):.1%}')
print(val.round(3).to_string())
loc = pd.read_csv(lv / 'locations.csv')
un = set(loc.loc[loc['coord_method'] == 'unlocated', 'meta_row'])
new = t3[(t3['t3b_status'] == 'ok') & t3['meta_row'].isin(un)]
print(f'\nNew locations for currently unlocated rows: {len(new):,}')
print(new['kind'].value_counts().to_string())
