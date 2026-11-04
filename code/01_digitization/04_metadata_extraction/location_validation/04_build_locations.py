"""
Combine the location tiers into one table: one row per main_metadata row.

Hierarchy (agreed 2026-10-28; LLM-inferred and unverified "actual" coordinates
are NOT used; every location comes from the row's own page):
  1 printed     LLM "actual" lat/long confirmed printed on the page   (03_printed_coords.py)
  2b wellno     state well number on the page -> NWIS well or section  (07_well_numbers.py)
  2 plss        Township/Range(/Section) on the page -> PLSS centroid (02_trs_lookup_all.py)
  3a nwis_station  NWIS station number printed on the page + name check (05_nwis_station_numbers.py)
  3b gnis_<kind>    named place on the page -> GNIS point               (06_gnis_named_places.py)
  oos           pages about other states (10_out_of_state.py): national NWIS station list and
                GNIS of the page's states; California-only references skipped on those pages
  4 coarse      county                                                (not built yet)
  5 unlocated

coord_method / precision_km:
  printed         ~1 km   (printed to the second or minute)
  plss_section    ~1 km   (sections are 1 mi square; centroid)
  plss_township   ~5 km   (townships are 6 mi square; centroid)
  nwis_station    ~1 km   (NWIS gauge coords; 98% within 5 km of printed coords)
  nwis_well       ~0.1 km (well number matched to an NWIS groundwater site)
  plss_wellno     ~1 km   (section centroid from the well number; ~0.75 km from NWIS wells)
  wellno_group    ~5 km   (row covers several wells within 5 km; centroid)
  gnis_<kind>     median error vs printed coords (tier3b_gnis_validation.csv)

precision_class (validated against tier-1 printed coordinates):
  precise      printed, nwis_well, plss_wellno, plss_section, nwis_station,
               gnis at / dam / reservoir
               (median <= ~2 km, ~90% within 5 km) -> usable for HUC12 claims
  basin        b118_basin / b118_subbasin: groundwater basin named on the page (point inside
               the DWR Bulletin 118 polygon; precision = basin radius). Not for HUC12 or point maps.
  approximate  wellno_group, plss_township, gnis near / place / lake / spring / below / above
               (median 3-8 km) -> county / HUC8-scale claims only
  unlocated

Inputs:  data/metadata/main_metadata.csv, data/analysis/location_validation/
           tier1_printed.csv, tier2b_well_numbers.csv, trs_lookup_all.csv,
           tier3a_nwis_station.csv, tier3b_gnis.csv
evidence_scope: page | document (report-level meridian/direction letters only) |
                continuation (inherited along an explicit "Continued" chain)

Output:  data/analysis/location_validation/locations.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[4]
lv   = root / 'data/analysis/location_validation'

meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'water_type', 'actual_latitude',
                            'actual_longitude', 'inferred_latitude'])
meta['meta_row'] = np.arange(len(meta))
loc = meta[['meta_row', 'id', 'page_number', 'water_type']].copy()
loc['llm_coord_source'] = np.where(meta['actual_latitude'].notna(), 'actual',
                                   np.where(meta['inferred_latitude'].notna(), 'inferred', 'none'))
loc['lat'] = np.nan
loc['lon'] = np.nan
loc['coord_method'] = 'unlocated'
loc['precision_km'] = np.nan
# evidence_scope: page = row's own page only; document = the row's page gives the location,
# other pages of the same report settle only the meridian / direction letters;
# continuation = location inherited along an explicit "Continued" chain (08_continuation.py)
loc['evidence_scope'] = None

# ---- page state (10_out_of_state.py / state_cues.py). On out_of_state pages the
# California-only references (CA GNIS, CA state well numbers, B118 basins, PLSS resolved to
# California, continuation from those) are NOT applied; national references are used instead.
ps_path = lv / 'page_states.csv'
oos_rows = set()
if ps_path.exists():
    ps = pd.read_csv(ps_path, dtype={'id': str})
    key = meta['id'].astype('Int64').astype(str) + '|' + meta['page_number'].astype('Int64').astype(str)
    psk = ps.assign(k=ps['id'] + '|' + ps['page_number'].astype(str)).set_index('k')
    loc['page_state_class'] = key.map(psk['page_state_class']).fillna('ca').values
    loc['page_states'] = key.map(psk['states']).values
rs_path = lv / 'row_states.csv'
either_gnis = set()
if rs_path.exists():
    # row class (10_out_of_state.py): only rows whose own place text names another state
    # (cued on the page) skip the California-only references
    rs = pd.read_csv(rs_path, dtype={'row_states': str}).set_index('meta_row')
    loc['row_class'] = loc['meta_row'].map(rs['row_class']).fillna('ca')
    loc['row_states'] = loc['meta_row'].map(rs['row_states'])
    oos_rows = set(loc.loc[loc['row_class'] == 'oos', 'meta_row'])
    og0 = lv / 'tier_oos_gnis.csv'
    if og0.exists():  # 'either' rows: the CA + page-states lookup replaces the CA-only one
        o = pd.read_csv(og0)
        either_gnis = set(o.loc[o['row_class'] == 'either', 'meta_row'])

# ---- B118 groundwater basin (09_b118_basins.py): basin-level only, lowest of all tiers
b118_path = lv / 'tier5_b118_basins.csv'
if b118_path.exists():
    bt = pd.read_csv(b118_path)
    bt = bt[(bt['status'] == 'ok') & ~bt['meta_row'].isin(oos_rows)].set_index('meta_row')
    i = loc['meta_row'].isin(bt.index)
    loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(bt['lat'])
    loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(bt['lon'])
    loc.loc[i, 'coord_method'] = loc.loc[i, 'meta_row'].map(bt['method'])
    loc.loc[i, 'precision_km'] = loc.loc[i, 'meta_row'].map(bt['precision_km'])
    loc.loc[i, 'evidence_scope'] = 'page'
    loc['b118_basin'] = loc['meta_row'].map(bt['b118_name'])

# ---- continuation (08_continuation.py) is the lowest tier: it only fills rows that no
# page-level method locates, so it is applied first and overwritten by everything else
cont_path = lv / 'tier4_continuation.csv'
if cont_path.exists():
    ct = pd.read_csv(cont_path, low_memory=False)
    ct = ct[(ct['status'] == 'ok') & ~ct['meta_row'].isin(oos_rows)].set_index('meta_row')
    i = loc['meta_row'].isin(ct.index)
    loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(ct['lat'])
    loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(ct['lon'])
    loc.loc[i, 'coord_method'] = loc.loc[i, 'meta_row'].map(ct['source_method'])
    loc.loc[i, 'precision_km'] = loc.loc[i, 'meta_row'].map(ct['source_precision_km'])
    loc.loc[i, 'evidence_scope'] = 'continuation'
    loc['continuation_rule'] = loc['meta_row'].map(ct['rule'])

# ---- lowest tier first; each higher tier overwrites (higher tier wins)
gval = pd.read_csv(lv / 'tier3b_gnis_validation.csv', index_col='kind')
gn = pd.read_csv(lv / 'tier3b_gnis.csv')
gn = gn[(gn['t3b_status'] == 'ok') & ~gn['meta_row'].isin(oos_rows | either_gnis)].set_index('meta_row')
i = loc['meta_row'].isin(gn.index)
kind = loc.loc[i, 'meta_row'].map(gn['kind'])
loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(gn['gnis_lat'])
loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(gn['gnis_lon'])
loc.loc[i, 'coord_method'] = 'gnis_' + kind
loc.loc[i, 'precision_km'] = kind.map(gval['median_km']).round(1)
loc.loc[i, 'evidence_scope'] = 'page'
loc['gnis_place'] = loc['meta_row'].map(gn['place'])

# out-of-state GNIS (10_out_of_state.py): same kinds -> same coord_method / precision
OOS_GNIS_REJECT = {'qualifier_below', 'feature_lake'}
og_path = lv / 'tier_oos_gnis.csv'
if og_path.exists():
    og = pd.read_csv(og_path)
    # Own method names: out-of-state GNIS validates differently from California
    # (tier_oos_validation.csv). 'below' (89% > 25 km) and lakes (median 19 km) are not used;
    # only 'at' is precise (94% <= 5 km); dams/reservoirs/springs/near/above/place approximate.
    ov = pd.read_csv(lv / 'tier_oos_validation.csv').set_index('method')
    og = og[(og['status'] == 'ok') & ~(og['kind'].isin(OOS_GNIS_REJECT) & ~og['in_ca'].astype(bool))].set_index('meta_row')
    i = loc['meta_row'].isin(og.index)
    kind = loc.loc[i, 'meta_row'].map(og['kind'])
    loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(og['lat'])
    loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(og['lon'])
    loc.loc[i, 'coord_method'] = np.where(loc.loc[i, 'meta_row'].map(og['in_ca']).astype(bool),
                                          'gnis_' + kind,        # resolved inside California: CA rules
                                          'gnis_oos_' + kind)
    loc.loc[i, 'precision_km'] = np.where(loc.loc[i, 'coord_method'].str.startswith('gnis_oos_'),
                                          ('gnis_' + kind).map(ov['median_km']).fillna(ov.at['gnis (all)', 'median_km']),
                                          kind.map(gval['median_km'])).round(1)
    loc.loc[i, 'evidence_scope'] = 'page'
    loc.loc[i, 'gnis_place'] = loc.loc[i, 'meta_row'].map(og['place'])

t3 = pd.read_csv(lv / 'tier3a_nwis_station.csv', dtype={'site_no': str})
t3 = t3[t3['t3a_status'] == 'ok'].set_index('meta_row')
i = loc['meta_row'].isin(t3.index)
loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(t3['nwis_lat'])
loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(t3['nwis_lon'])
loc.loc[i, 'coord_method'] = 'nwis_station'
loc.loc[i, 'precision_km'] = 1.0
loc.loc[i, 'evidence_scope'] = 'page'
loc['nwis_site_no'] = loc['meta_row'].map(t3['site_no'])

# out-of-state NWIS station numbers (10_out_of_state.py)
on_path = lv / 'tier_oos_nwis.csv'
if on_path.exists():
    on = pd.read_csv(on_path, dtype={'site_no': str})
    on = on[(on['status'] == 'ok') & ~on['meta_row'].isin(t3.index)].set_index('meta_row')
    i = loc['meta_row'].isin(on.index)
    loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(on['lat'])
    loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(on['lon'])
    loc.loc[i, 'coord_method'] = 'nwis_station'
    loc.loc[i, 'precision_km'] = 1.0
    loc.loc[i, 'evidence_scope'] = 'page'
    loc.loc[i, 'nwis_site_no'] = loc.loc[i, 'meta_row'].map(on['site_no'])

trs = pd.read_csv(lv / 'trs_lookup_all.csv', low_memory=False)
trs = trs[(trs['ref_status'] == 'ok') & ~(trs['meta_row'].isin(oos_rows) & trs['ref_state'].eq('CA'))].set_index('meta_row')
i = loc['meta_row'].isin(trs.index)
loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(trs['ref_lat'])
loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(trs['ref_lon'])
loc.loc[i, 'coord_method'] = 'plss_' + loc.loc[i, 'meta_row'].map(trs['ref_precision'])
loc.loc[i, 'precision_km'] = np.where(loc.loc[i, 'coord_method'] == 'plss_section', 1.0, 5.0)
loc.loc[i, 'evidence_scope'] = np.where(loc.loc[i, 'meta_row'].map(trs['ref_rule']).isin(['document meridian', 'county in report title']),
                                        'document', 'page')

wn = pd.read_csv(lv / 'tier2b_well_numbers.csv', low_memory=False)
wn = wn[wn['wn_method'].notna() & ~wn['meta_row'].isin(oos_rows)].set_index('meta_row')
i = loc['meta_row'].isin(wn.index)
loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(wn['wn_lat'])
loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(wn['wn_lon'])
loc.loc[i, 'coord_method'] = loc.loc[i, 'meta_row'].map(wn['wn_method'])
loc.loc[i, 'precision_km'] = loc.loc[i, 'coord_method'].map({'nwis_well': 0.1, 'plss_wellno': 1.0,
                                                             'wellno_group': 5.0})
loc.loc[i, 'evidence_scope'] = loc.loc[i, 'meta_row'].map(wn['scope']).fillna('page')
loc['well_number'] = loc['meta_row'].map(wn['well'])

t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])].set_index('meta_row')
i = loc['meta_row'].isin(t1.index)
loc.loc[i, 'lat'] = loc.loc[i, 'meta_row'].map(t1['actual_latitude'])
loc.loc[i, 'lon'] = loc.loc[i, 'meta_row'].map(t1['actual_longitude'])
loc.loc[i, 'coord_method'] = 'printed'
loc.loc[i, 'precision_km'] = 1.0
loc.loc[i, 'evidence_scope'] = 'page'

PRECISE = {'printed', 'nwis_well', 'plss_wellno', 'plss_section', 'nwis_station', 'gnis_qualifier_at', 'gnis_feature_dam',
           'gnis_feature_reservoir', 'gnis_oos_qualifier_at'}
loc['precision_class'] = np.where(loc['coord_method'] == 'unlocated', 'unlocated',
                         np.where(loc['coord_method'].isin(PRECISE), 'precise',
                         np.where(loc['coord_method'].str.startswith('b118'), 'basin', 'approximate')))
# continuation by owner/site name: 73% within 2 km in the hold-out -> approximate at best
if 'continuation_rule' in loc:
    loc.loc[loc['evidence_scope'].eq('continuation') & loc['continuation_rule'].ne('well number')
            & loc['precision_class'].eq('precise'), 'precision_class'] = 'approximate'
loc.to_csv(lv / 'locations.csv', index=False)

print(f'Rows: {len(loc):,}')
if 'page_state_class' in loc:
    print('Precision class by row class:')
    print(pd.crosstab(loc.get('row_class', loc['page_state_class']), loc['precision_class']).to_string())
print(loc['coord_method'].value_counts().to_string())
print('\n', loc['precision_class'].value_counts().to_string())
print('\n', pd.crosstab(loc['precision_class'], loc['evidence_scope'].fillna('-')).to_string())
print('\nPrecision class share by water type:')
top = loc['water_type'].value_counts().head(8).index
print(pd.crosstab(loc['water_type'], loc['precision_class'], normalize='index').loc[top].round(3).to_string())
print('\nNew method vs old LLM source:')
print(pd.crosstab(loc['llm_coord_source'], loc['coord_method']).to_string())
print(f'\nSaved: {lv / "locations.csv"}')
