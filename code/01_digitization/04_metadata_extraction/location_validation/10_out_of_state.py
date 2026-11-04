"""
Location tiers for pages about places OUTSIDE California (2026-11-04).

The California tiers use California-only references (CA GNIS, CA NWIS station list,
California state well numbers, Bulletin 118 basins), so pages about Nevada, Utah, Oregon,
North Carolina, ... were left unlocated, or worse, matched to a California namesake
(e.g. "Harpeth River near Kingston, Tenn." placed in California). This step applies the
same same-page rules with references for the states the page itself names.

1. Page state (state_cues.py): states cued on the page right after a comma ("..., Nev.",
   "..., Utah", "AT MESQUITE, NV"). page_state_class = ca | out_of_state | mixed.
   Pages with no cue keep the California pipeline (the corpus is California documents).
2. NWIS station number (same rule as 05_nwis_station_numbers.py): an 8-digit station number
   printed on the page that is an NWIS site in one of the page's states, AND a distinctive
   word shared between the row's water-source name and the NWIS station name.
   Runs on out_of_state and mixed pages (California numbers stay with tier 3a).
3. GNIS named place (same rules as 06_gnis_named_places.py, via gnis_utils.py): a place from
   the LLM water-source / location text that is printed on the page, looked up in the 2021
   GNIS archive of the page's states only; duplicates resolved by a county named on the page.
   Runs on out_of_state pages only.
4. Invalid California-only locations: on out_of_state pages, locations from California-only
   references (CA GNIS, CA well numbers, B118 basins, PLSS resolved to California, and
   continuation from those) are dropped in 04_build_locations.py.
Validation: each method vs tier-1 printed coordinates on out-of-state pages.

Inputs:  data/analysis/page_blocks.parquet, data/metadata/main_metadata.csv,
         data/analysis/location_validation/{tier1_printed, locations}.csv
         downloads (cached): data/analysis/nwis_sites/by_state/{ST}_sites.csv,
         data/analysis/spatial/gnis/{ST}_Features_20210825.txt
Outputs: data/analysis/location_validation/page_states.csv, tier_oos_nwis.csv,
         tier_oos_gnis.csv, tier_oos_validation.csv
"""
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_cues import page_states, classify  # noqa: E402
from nwis_utils import tokens, GENERIC  # noqa: E402
import gnis_utils  # noqa: E402
from gnis_utils import norm, hav, resolve, extract, load_gnis  # noqa: E402

root = Path(__file__).resolve().parents[4]
lv = root / 'data/analysis/location_validation'
NWIS_DIR = root / 'data/analysis/nwis_sites/by_state'
GNIS_DIR = root / 'data/analysis/spatial/gnis'
GNIS_URL = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/Archive/MainDomestic/{st}_Features_20210825.txt'
NWIS_URL = ('https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={st}'
            '&siteType=ST,ST-CA,ST-DCH,ST-TS,LK,SP,ES,WE&siteStatus=all')
RE_NO = re.compile(r'(?<!\d)(0[1-9]\d{6}|1[0-6]\d{6})(?!\d)')
STATE_WORDS = {'NEVADA', 'UTAH', 'IDAHO', 'OREGON', 'OREG', 'ARIZ', 'ARIZONA', 'COLO', 'COLORADO',
               'WYOMING', 'MONT', 'MONTANA', 'WASH', 'TENN', 'TEXAS', 'MEXICO', 'KANS', 'NEBR'}


def fetch(url, path, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
            return True
        except Exception as e:  # network hiccup: retry
            print(f'  retry {k + 1} {url[:80]}: {e}')
            time.sleep(5)
    return False


def nwis_sites(st):
    NWIS_DIR.mkdir(parents=True, exist_ok=True)
    p = NWIS_DIR / f'{st}_sites.csv'
    if not p.exists():
        raw = NWIS_DIR / f'{st}_sites.rdb'
        if not fetch(NWIS_URL.format(st=st), raw):
            return pd.DataFrame()
        lines = [ln for ln in raw.read_text(encoding='utf-8', errors='ignore').splitlines()
                 if ln and not ln.startswith('#')]
        if len(lines) < 3:
            pd.DataFrame(columns=['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va']).to_csv(p, index=False)
        else:
            d = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[2:])), sep='\t', dtype=str)
            d.to_csv(p, index=False)
        raw.unlink(missing_ok=True)
    d = pd.read_csv(p, dtype={'site_no': str})
    d['state'] = st
    return d


def gnis_file(st):
    p = GNIS_DIR / f'{st}_Features_20210825.txt'
    if not p.exists():
        fetch(GNIS_URL.format(st=st), p)
    return p


# ------------------------------------------------------------------ 1. page states
B = pd.read_parquet(root / 'data/analysis/page_blocks.parquet', columns=['id', 'page_number', 'content'])
B['id'] = B['id'].astype(str)
text = B.groupby(['id', 'page_number'])['content'].apply(lambda s: ' '.join(s.fillna('').astype(str)))
del B
ps = []
for (doc, page), t in text.items():
    st, ca = page_states(t)
    ps.append({'id': doc, 'page_number': int(page), 'states': ';'.join(sorted(st)), 'has_ca_cue': ca,
               'page_state_class': classify(st, ca)})
ps = pd.DataFrame(ps)
ps.to_csv(lv / 'page_states.csv', index=False)
print('pages by state class:\n', ps['page_state_class'].value_counts().to_string())

meta = pd.read_csv(root / 'data/metadata/main_metadata.csv', low_memory=False,
                   usecols=['id', 'page_number', 'watersource_name', 'location'])
meta['meta_row'] = np.arange(len(meta))
meta = meta[meta['id'].notna() & meta['page_number'].notna()].copy()
meta['id'] = meta['id'].astype(int).astype(str)
meta['page_number'] = meta['page_number'].astype(int)
meta = meta.merge(ps, on=['id', 'page_number'], how='inner')


# Row class (2026-11-04: a page-level rule wrongly moved 490 California rows -- index and
# station-list pages print one out-of-state entry and no California cue). The row's own LLM
# place text is the pointer; the state must also be cued on the page:
#   oos     row text names a non-CA state that is cued on the page, and not California
#   ca      row text names California, or the page cues no other state
#   either  page cues another state, row text names none -> a place name is used only if it
#           resolves to ONE place across California + the page's states (04_build_locations)
def row_class(r):
    st, ca = page_states(f'{r.watersource_name}, {r.location}')
    pst = set(str(r.states).split(';')) - {''}
    if ca or r.page_state_class == 'ca':
        return 'ca', ''
    hit = st & pst
    if hit:
        return 'oos', ';'.join(sorted(hit))
    return 'either', ''


rc = [row_class(r) for r in meta.itertuples()]
meta['row_class'] = [c for c, _ in rc]
meta['row_states'] = [s for _, s in rc]
meta[['meta_row', 'id', 'page_number', 'page_state_class', 'states', 'row_class', 'row_states']].to_csv(
    lv / 'row_states.csv', index=False)
print('metadata rows by row class:\n', meta['row_class'].value_counts().to_string())
oos = meta[meta['page_state_class'].isin(['out_of_state', 'mixed']) & meta['row_class'].ne('ca')]
print(f'metadata rows on out-of-state pages: {(meta.page_state_class == "out_of_state").sum():,}; '
      f'mixed pages: {(meta.page_state_class == "mixed").sum():,}')
states = sorted({s for v in oos['states'] for s in v.split(';') if s})
print('states cued:', ', '.join(states))

# ------------------------------------------------------------------ 2. NWIS station numbers
nw = pd.concat([nwis_sites(st) for st in states], ignore_index=True)
nw = nw[nw['dec_lat_va'].notna()].drop_duplicates('site_no').set_index('site_no')
nw_tok = {s: tokens(n) - STATE_WORDS for s, n in nw['station_nm'].items()}
recs = []
for (doc, page), g in oos.groupby(['id', 'page_number']):
    pst = set(g['states'].iloc[0].split(';'))
    nums = [s for s in dict.fromkeys(RE_NO.findall(text[(doc, page)]))
            if s in nw.index and nw.at[s, 'state'] in pst]
    if not nums:
        continue
    for row in g.itertuples():
        rt = tokens(f'{row.watersource_name} {row.location}') - STATE_WORDS
        wt = tokens(row.watersource_name) - STATE_WORDS
        scores = {s: (len(rt & nw_tok[s]) if wt & nw_tok[s] else 0) for s in nums}
        best = max(scores.values())
        win = [s for s, v in scores.items() if v == best]
        rec = {'meta_row': row.meta_row, 'id': doc, 'page_number': page, 'n_stations_on_page': len(nums)}
        if best == 0:
            rec['status'] = 'no_name_match'
        elif len(win) > 1:
            rec['status'] = 'tie'
        else:
            s = win[0]
            rec.update(status='ok', site_no=s, station_nm=nw.at[s, 'station_nm'], state=nw.at[s, 'state'],
                       lat=float(nw.at[s, 'dec_lat_va']), lon=float(nw.at[s, 'dec_long_va']))
        recs.append(rec)
tn = pd.DataFrame(recs)
tn.to_csv(lv / 'tier_oos_nwis.csv', index=False)
print('\nNWIS station tier:\n', tn['status'].value_counts().to_string() if len(tn) else 'none')

# ------------------------------------------------------------------ 3. GNIS named places
oo = oos  # rows of class oos / either on out-of-state or mixed pages
gstates = sorted({s for v in oo['states'] for s in v.split(';') if s})
gn_by_state = {}
for st in gstates:
    p = gnis_file(st)
    if p.exists():
        gn_by_state[st] = load_gnis(p, [st])
gn_by_state['CA'] = load_gnis(GNIS_DIR / 'CA_Features_20210825.txt', ['CA'])  # for 'either' rows
recs = []
for (doc, page), grp in oo.groupby(['id', 'page_number']):
    raw = text[(doc, page)]
    page_lc, page_n = raw.lower(), ' ' + norm(raw) + ' '
    page_st = [s for s in grp['states'].iloc[0].split(';') if s in gn_by_state]
    for row in grp.itertuples():
        # oos rows: the page's states named in the row; either rows: CA + the page's states
        pst = [s for s in row.row_states.split(';') if s in gn_by_state] if row.row_class == 'oos'             else page_st + ['CA']
        rec = {'meta_row': row.meta_row, 'id': doc, 'page_number': page, 'row_class': row.row_class,
               'status': 'no_candidate'}
        for name, kind, classes in extract(row.watersource_name, row.location):
            if f' {norm(name)} ' not in page_n:
                rec['status'] = 'name_not_on_page'
                continue
            cands = [gnis_utils.candidates(gn_by_state[s], name, classes) for s in pst]
            cands = [c for c in cands if c is not None and len(c)]
            res, rule = resolve(pd.concat(cands) if cands else None, page_lc)
            if res is None:
                rec['status'] = rule
                continue
            rec.update(status='ok', place=name, kind=kind, rule=rule, lat=res[0], lon=res[1])
            break
        recs.append(rec)
tg = pd.DataFrame(recs)
if len(tg):  # where the resolved place lies (an 'either' row may resolve inside California)
    import geopandas as gpd
    ca = gpd.read_file(root / 'data/analysis/spatial/ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
    ok = tg['status'].eq('ok')
    pts = gpd.GeoDataFrame(tg[ok], geometry=gpd.points_from_xy(tg.loc[ok, 'lon'], tg.loc[ok, 'lat']), crs='EPSG:4326')
    tg['in_ca'] = False
    tg.loc[ok, 'in_ca'] = pts.within(ca.union_all()).values
tg.to_csv(lv / 'tier_oos_gnis.csv', index=False)
print('\nGNIS tier (out-of-state pages):\n', tg['status'].value_counts().to_string() if len(tg) else 'none')

# ------------------------------------------------------------------ validation
t1 = pd.read_csv(lv / 'tier1_printed.csv')
t1 = t1[t1['tier1_status'].isin(['printed_match', 'printed_match_no_keyword'])]
rows = []
for name, t, key in [('nwis_station', tn, None), ('gnis', tg, 'kind')]:
    if not len(t):
        continue
    v = t[t['status'] == 'ok'].merge(t1[['meta_row', 'actual_latitude', 'actual_longitude']], on='meta_row')
    v['err_km'] = hav(v['lat'], v['lon'], v['actual_latitude'], v['actual_longitude'])
    groups = [(name, v)] if key is None else [(f'gnis_{k}', x) for k, x in v.groupby(key)] + [('gnis (all)', v)]
    for k, x in groups:
        rows.append({'method': k, 'n': len(x), 'median_km': x.err_km.median(), 'le_2km': np.mean(x.err_km <= 2),
                     'le_5km': np.mean(x.err_km <= 5), 'gt_25km': np.mean(x.err_km > 25)})
val = pd.DataFrame(rows)
val.to_csv(lv / 'tier_oos_validation.csv', index=False)
print('\nValidation vs printed coordinates on out-of-state / mixed pages:\n', val.round(3).to_string(index=False))

loc = pd.read_csv(lv / 'locations.csv', usecols=['meta_row', 'coord_method'])
un = set(loc.loc[loc['coord_method'] == 'unlocated', 'meta_row'])
for name, t in [('NWIS', tn), ('GNIS', tg)]:
    if len(t):
        print(f'{name}: new locations for currently unlocated rows: '
              f'{t[(t.status == "ok") & t.meta_row.isin(un)].shape[0]:,}')
