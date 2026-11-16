"""
Figure 5: locate the WSP 1464 Appendix B wells (solano_wsp1464_wells.csv) at the centroid of their PLSS
section. Well numbers like 8/1E-15B1 = T.8 N., R.1 E., section 15, tract B, well 1, Mount Diablo meridian
(the report covers part of Solano County; all townships are north of the base line). Section centroid =
same rule as tier 2b of the location rebuild (07_well_numbers.py; section centroid vs NWIS well ~0.75 km).
Wells also present in NWIS (state well no. = NWIS station name) get the NWIS coordinates.

Output: data/analysis/gw_example/solano_wsp1464_wells_located.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/01_digitization/04_metadata_extraction/location_validation'))
from plss_utils import townships, sections, haversine_km  # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'gw_example'

w = pd.read_csv(OUT / 'solano_wsp1464_wells.csv')
p = w.well.str.extract(r'^(\d+)/(\d+)([EW])-(\d+)([A-R])(\d+)$')
w['t'], w['r'], w['rd'], w['sec'] = p[0].astype(int), p[1].astype(int), p[2], p[3].astype(int)
w['nwis_name'] = [f'{t:03d}N{r:03d}{d}{s:02d}{tr}{int(q):03d}M' for t, r, d, s, tr, q in
                  zip(w.t, w.r, w.rd, w.sec, p[4], p[5])]

tw = townships()
tw = tw[tw.PRINMER == 'Mount Diablo Meridian']
key = {(int(a), b, int(c), d): pid for (a, b, c, d), pid in zip(tw.key, tw.PLSSID)}
w['plssid'] = [key.get((t, 'N', r, d)) for t, r, d in zip(w.t, w.r, w.rd)]
sc = sections(w.plssid.dropna().unique())
loc = [sc.loc[(pid, s)] if isinstance(pid, str) and (pid, s) in sc.index else None
       for pid, s in zip(w.plssid, w.sec)]
w['lat'] = [l['lat'] if l is not None else None for l in loc]
w['lon'] = [l['lon'] if l is not None else None for l in loc]
w['location_method'] = w.lat.notna().map({True: 'plss_section', False: None})

# Fallback (2026-11-16): sections not in the BLM layer (Mexican land grants: Rancho Los Putos, Suisun, ...).
# Standard 6x6 section grid (1 mile per section, numbered boustrophedon from the NE corner) placed on the
# township centroid, shifted by the mean offset of that township's surveyed sections. Plain grid vs 2,683
# surveyed sections in T3-10N, R5W-R3E: median 0.10 km, 90% <= 0.64 km, max 4.7 km.
tws = tw.drop_duplicates('PLSSID').set_index('PLSSID')


def grid(pid, sec):
    la, lo = tws.loc[pid, ['lat', 'lon']]
    row, pos = (sec - 1) // 6, (sec - 1) % 6
    col = 5 - pos if row % 2 == 0 else pos
    return la + (2.5 - row) * 1.609 / 111.0, lo + (col - 2.5) * 1.609 / (111.0 * np.cos(np.radians(la)))


miss = w.lat.isna() & w.plssid.notna()
for pid in w.loc[miss, 'plssid'].unique():
    sv = sc.loc[pid] if pid in sc.index.get_level_values(0) else None
    off = (0.0, 0.0)
    if sv is not None and len(sv):
        g = np.array([grid(pid, int(s)) for s in sv.index])
        off = ((sv.lat.values - g[:, 0]).mean(), (sv.lon.values - g[:, 1]).mean())
    for i in w.index[miss & (w.plssid == pid)]:
        la, lo = grid(pid, int(w.at[i, 'sec']))
        w.at[i, 'lat'], w.at[i, 'lon'] = la + off[0], lo + off[1]
        w.at[i, 'location_method'] = 'plss_grid'

n = pd.read_csv(A / 'nwis_sites/ca_groundwater_sites.csv', low_memory=False, dtype={'site_no': str})
n = n.drop_duplicates('site_no')
n['station_nm'] = n.station_nm.str.strip()
nm = n.drop_duplicates('station_nm').set_index('station_nm')   # a few names repeat; first site kept
w['nwis_site_no'] = w.nwis_name.map(nm.site_no)
has = w.nwis_site_no.notna()
d = haversine_km(w.loc[has, 'lat'], w.loc[has, 'lon'], w.loc[has, 'nwis_name'].map(nm.dec_lat_va),
                 w.loc[has, 'nwis_name'].map(nm.dec_long_va))
w.loc[has, 'section_vs_nwis_km'] = d.round(2)
w.loc[has, 'lat'] = w.loc[has, 'nwis_name'].map(nm.dec_lat_va)
w.loc[has, 'lon'] = w.loc[has, 'nwis_name'].map(nm.dec_long_va)
w.loc[has, 'location_method'] = 'nwis_well'
w.to_csv(OUT / 'solano_wsp1464_wells_located.csv', index=False)
print(f'wells {len(w)}; located {w.lat.notna().sum()} '
      f'({w.location_method.value_counts().to_dict()}); in NWIS {has.sum()}')
print(w.loc[has, ['well', 'nwis_site_no', 'section_vs_nwis_km', 'n_readings', 'first', 'last']].to_string(index=False))
print('unlocated:', w.loc[w.lat.isna(), 'well'].tolist())
