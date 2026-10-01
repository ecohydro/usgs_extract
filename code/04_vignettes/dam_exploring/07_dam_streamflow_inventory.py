"""
Inventory of restored streamflow data near dammed California streams, 1849-1980.

For each candidate dam (NID, completed <= 1979), find restored and NWIS stream
discharge sites on the SAME stream near the dam:
  - within RADIUS_KM of the dam (straight-line), AND
  - the site/station name contains the dam's river-name key tokens
    (e.g. NID "Eel River" -> key "eel"; "Nfk Feather R" -> "feather")
Upstream vs. downstream is not resolved (no flowline routing).

Study window matches the 6-panel figure: restored data ends 1980; NWIS sites are
counted only if they begin <= NWIS_YEAR_CUTOFF (post-1980 NWIS is ignored).

Candidate dams: NID storage >= MIN_STORAGE_AF, plus named removal candidates.

Inputs:
  data/analysis/dams.csv                                   (NID, downloaded 2026-05)
  data/analysis/table_level_metadata_published.csv         (restored, one row per table)
  data/analysis/nwis_sites/ca_stream_sites_expanded.csv    (NWIS daily+peak+field)
Output:
  data/analysis/dam_exploring/dam_streamflow_inventory.csv (one row per dam)
  data/analysis/dam_exploring/dam_streamflow_sites.csv     (matched restored sites)

Caveat: restored coordinates are Phi-4 page-level estimates; see the location
accuracy write-up (2026-10-01). Name matching limits, but does not remove, the
effect of mislocated points.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

root     = Path(__file__).resolve().parents[3]
out_dir  = root / 'data/analysis/dam_exploring'

RADIUS_KM        = 30
MIN_STORAGE_AF   = 100_000
DAM_YEAR_MAX     = 1979
NWIS_YEAR_CUTOFF = 1979
RESTORED_YEAR_MAX = 1980
REMOVAL_CANDIDATES = ['Matilija', 'Scott', 'Cape Horn', 'Los Padres', 'Searsville']

STOP = {'river', 'r', 'creek', 'cr', 'c', 'fork', 'fk', 'nfk', 'sfk', 'mfk', 'efk', 'wfk',
        'nf', 'sf', 'mf', 'ef', 'wf', 'north', 'south', 'middle', 'east', 'west', 'of',
        'the', 'tr', 'trib', 'tributary', 'and', 'offstream', 'canyon', 'slough', 'lake'}


def tokens(s):
    return [t for t in re.findall(r'[a-z]+', str(s).lower()) if t not in STOP and len(t) > 1]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# NWIS abbreviates: "EEL R A SCOTIA" -> expand so tokens match restored names
def nwis_name(s):
    return re.sub(r'\bR\b', 'RIVER', re.sub(r'\bC\b', 'CREEK', str(s)))


# ---------------------------------------------------------------- dams
d = pd.read_csv(root / 'data/analysis/dams.csv', low_memory=False)
d['Year Completed'] = pd.to_numeric(d['Year Completed'], errors='coerce')
d['storage'] = pd.to_numeric(d['NID Storage (Acre-Ft)'], errors='coerce')
is_removal = d['Dam Name'].isin(REMOVAL_CANDIDATES) & d['River or Stream Name'].notna()
d = d[(d['Year Completed'] <= DAM_YEAR_MAX) & (d['Is Associated Structure?'] != 'Yes') &
      ((d['storage'] >= MIN_STORAGE_AF) | is_removal) &
      d['Latitude'].notna() & d['River or Stream Name'].notna()].copy()
d['key'] = d['River or Stream Name'].map(tokens)
d = d[d['key'].map(len) > 0]
d['removal_candidate'] = d['Dam Name'].isin(REMOVAL_CANDIDATES)
print(f'Candidate dams: {len(d)}')

# ---------------------------------------------------------------- restored
t = pd.read_csv(root / 'data/analysis/table_level_metadata_published.csv', low_memory=False)
r = t[(t['water_type_final'] == 'Stream Discharge') & t['lat_combined'].notna() &
      t['year_start'].notna() & (t['year_start'] <= RESTORED_YEAR_MAX)].copy()
r['name_tok'] = r['watersource_name'].map(lambda s: set(tokens(s)))

# ---------------------------------------------------------------- NWIS
n = pd.read_csv(root / 'data/analysis/nwis_sites/ca_stream_sites_expanded.csv', low_memory=False)
n = n.drop_duplicates('site_no')
n = n[(n['begin_year'] <= NWIS_YEAR_CUTOFF) & n['dec_lat_va'].notna()].copy()
n['name_tok'] = n['station_nm'].map(lambda s: set(tokens(nwis_name(s))))


def match(df, lat, lon, dam):
    dist = haversine_km(dam['Latitude'], dam['Longitude'], df[lat].values, df[lon].values)
    near = dist <= RADIUS_KM
    named = df['name_tok'].map(lambda s: set(dam['key']) <= s).values
    out = df[near & named].copy()
    out['dist_km'] = dist[near & named]
    return out


rows, site_rows = [], []
for _, dam in d.iterrows():
    dy = int(dam['Year Completed'])
    rm = match(r, 'lat_combined', 'lon_combined', dam)
    nm = match(n, 'dec_lat_va', 'dec_long_va', dam)

    nwis_first = nm['begin_year'].min() if len(nm) else np.nan
    yrs = rm['year_start'].astype(int)
    pre = rm[rm['year_start'] < dy]
    # restored pre-dam years that come before ANY NWIS record on this stream
    first_digital = min(dy, nwis_first) if len(nm) else dy
    new_pre = rm[rm['year_start'] < first_digital]

    rows.append({
        'dam': dam['Dam Name'], 'river': dam['River or Stream Name'],
        'county': dam['County'], 'year_completed': dy,
        'storage_af': dam['storage'], 'removal_candidate': dam['removal_candidate'],
        'lat': dam['Latitude'], 'lon': dam['Longitude'],
        # restored
        'r_sites': rm.assign(la=rm['lat_combined'].round(2), lo=rm['lon_combined'].round(2))
                     .drop_duplicates(['la', 'lo']).shape[0],
        'r_tables': len(rm), 'r_meas': int(rm['number_rows'].sum()),
        'r_first_yr': yrs.min() if len(rm) else np.nan,
        'r_years': yrs.nunique(),
        'r_predam_years': pre['year_start'].nunique(),
        'r_predam_meas': int(pre['number_rows'].sum()),
        'r_postdam_years': rm.loc[rm['year_start'] >= dy, 'year_start'].nunique(),
        # NWIS (<= 1979 starts only)
        'nwis_sites': len(nm),
        'nwis_first_yr': nwis_first,
        'nwis_predam_sites': int((nm['begin_year'] < dy).sum()),
        # what restored adds before any digital record
        'r_years_before_nwis': new_pre['year_start'].nunique(),
        'r_meas_before_nwis': int(new_pre['number_rows'].sum()),
    })
    if len(rm):
        site_rows.append(rm.assign(dam=dam['Dam Name'], dam_year=dy)[
            ['dam', 'dam_year', 'doc_id', 'page_number', 'watersource_name', 'lat_combined',
             'lon_combined', 'dist_km', 'year_start', 'year_end', 'number_rows']])

inv = pd.DataFrame(rows).sort_values(['r_predam_meas', 'r_meas'], ascending=False)
inv.to_csv(out_dir / 'dam_streamflow_inventory.csv', index=False)
pd.concat(site_rows).to_csv(out_dir / 'dam_streamflow_sites.csv', index=False)
print(f'Saved inventory ({len(inv)} dams) and matched sites to {out_dir}')

pd.set_option('display.width', 260)
cols = ['dam', 'river', 'year_completed', 'storage_af', 'r_sites', 'r_meas', 'r_first_yr',
        'r_predam_years', 'r_predam_meas', 'r_postdam_years', 'nwis_sites', 'nwis_first_yr',
        'nwis_predam_sites', 'r_years_before_nwis', 'r_meas_before_nwis']
show = inv[inv['r_tables'] > 0]
print(f'\nDams with matched restored streamflow: {len(show)} of {len(inv)}')
print(show[cols].to_string(index=False))
print('\nRemoval candidates:')
print(inv[inv['removal_candidate']][cols].to_string(index=False))
