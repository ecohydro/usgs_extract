"""
Location tier 3a: NWIS stream station number printed on the page.

Later Water Supply Papers print the 8-digit USGS station number in each station
heading ("11433200 RUBICON RIVER NEAR FORESTHILL, CALIF."). A number that is in
the NWIS California site list gives the gauge's NWIS coordinates.

A page can carry several stations, so a row gets a station only if:
  1. the station number is printed on the row's own page, and
  2. the row's LLM water-source name / location shares a distinctive word
     (>= 4 letters, not a generic word like RIVER, CREEK, NEAR) with that
     station's NWIS name, AND at least one shared word comes from the
     water-source name itself (a town in the location text alone is not enough:
     "Chesbro Reservoir" must not match "Pajaro R at Watsonville"). The LLM name is only a pointer; the number must be
     on the page.
If several stations on the page match, the one with the most shared words wins;
ties are left unassigned.

Run on ALL rows (not only unlocated ones) so the method can be validated against
tier-1 printed coordinates.

Inputs:  data/metadata/main_metadata.csv, data/digitized/.../*.json,
         data/analysis/nwis_sites/ca_stream_sites_expanded.csv,
         data/analysis/location_validation/tier1_printed.csv
Output:  data/analysis/location_validation/tier3a_nwis_station.csv
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'

from nwis_utils import ABBREV, GENERIC, tokens  # noqa: E402  (shared with 10_out_of_state.py)


nwis = pd.read_csv(root / 'data/analysis/nwis_sites/ca_stream_sites_expanded.csv',
                   dtype={'site_no': str}, low_memory=False).drop_duplicates('site_no')
nwis = nwis[nwis['dec_lat_va'].notna()].set_index('site_no')
nwis_tok = {s: tokens(n) for s, n in nwis['station_nm'].items()}
RE_NO = re.compile(r'(?<!\d)(1[01]\d{6}|09\d{6})(?!\d)')

meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location'])
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()]

recs = []
for (doc, page), g in meta.groupby(['id', 'page_number']):
    doc, page = int(doc), int(page)
    p = root / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not p.exists():
        continue
    j = json.loads(p.read_text(encoding='utf-8'))
    text = ' '.join(c.get('content', '') for c in j['result']['chunks'])
    stations = [s for s in dict.fromkeys(RE_NO.findall(text)) if s in nwis.index]
    if not stations:
        continue
    for _, row in g.iterrows():
        rt = tokens(f"{row['watersource_name']} {row['location']}")
        wt = tokens(row['watersource_name'])
        scores = {s: (len(rt & nwis_tok[s]) if wt & nwis_tok[s] else 0) for s in stations}
        best = max(scores.values())
        winners = [s for s, v in scores.items() if v == best]
        rec = {'meta_row': row['meta_row'], 'id': doc, 'page_number': page,
               'n_stations_on_page': len(stations), 'watersource_name': row['watersource_name']}
        if best == 0:
            rec['t3a_status'] = 'no_name_match'
        elif len(winners) > 1:
            rec['t3a_status'] = 'tie'
        else:
            s = winners[0]
            rec.update({'t3a_status': 'ok', 'site_no': s, 'station_nm': nwis.at[s, 'station_nm'],
                        'shared_words': ' '.join(sorted(rt & nwis_tok[s])),
                        'nwis_lat': nwis.at[s, 'dec_lat_va'], 'nwis_lon': nwis.at[s, 'dec_long_va']})
        recs.append(rec)

t3 = pd.DataFrame(recs)
t3.to_csv(lv / 'tier3a_nwis_station.csv', index=False)
print(f'Rows on pages printing an NWIS CA station number: {len(t3):,}')
print(t3['t3a_status'].value_counts().to_string())

# ---- validation against tier-1 printed coordinates
t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])]
v = t3[t3['t3a_status'] == 'ok'].merge(t1[['meta_row', 'actual_latitude', 'actual_longitude']], on='meta_row')
if len(v):
    lat1, lon1, lat2, lon2 = map(np.radians, [v['nwis_lat'], v['nwis_lon'],
                                              v['actual_latitude'], v['actual_longitude']])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    e = 6371 * 2 * np.arcsin(np.sqrt(a))
    print(f'\nValidation vs printed coords (n={len(v):,}): median {np.median(e):.2f} km; '
          f'<=1 km {np.mean(e <= 1):.0%}; <=5 km {np.mean(e <= 5):.0%}; >25 km {np.mean(e > 25):.1%}')
loc = pd.read_csv(lv / 'locations.csv')
un = set(loc.loc[loc['coord_method'] == 'unlocated', 'meta_row'])
print(f'New locations for currently unlocated rows: {t3[(t3.t3a_status == "ok") & t3.meta_row.isin(un)].shape[0]:,}')
