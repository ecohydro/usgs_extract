"""
Restored-site definitions for comparing restored sites with NWIS sites (2026-11-14).

Problem: figures counted a restored "site" as a unique exact coordinate pair. One gauge
reprinted in many annual reports can appear at several slightly different coordinates (e.g.
Sacramento R at Butte City, NWIS 11389000: 19 points), and distinct wells can share one PLSS
section centroid. Annette's rule: err on the side of counting unique sites, but be defensible.

Definitions (computed side by side; the paper picks one and reports the others as a range):
  A  coord        one site per unique coordinate (rounded to 1e-5 deg; the old figure rule,
                  which also counted raw float noise; upper bound)
  B  identity     records that NAME the same NWIS station on their page (station number or
                  station name printed on the page) are one site, as NWIS counts them, BUT only
                  where their points also agree within T km (single linkage; B0.5 / B1 / B2);
                  groundwater records with the same state well number likewise.
                  Everything else stays one site per unique coordinate.
                  Distance-only NWIS links (nearest_named, nearest_well, name_cluster) never merge.
  C  identity+name  B (T = MAIN_TOL_KM) plus records without an identity that share a normalized
                  water-source name AND lie within MERGE_KM of each other (lower bound).

Unit = metadata ENTRIES (processed_metadata.parquet), not tables: one table can list many stations.
Site set = the six-panel rule (07_ca_combined_6panel.py): inside the CA polygon, precise or
approximate rebuilt location, measurement-dated (no period-of-record / construction fallback),
data pages, tribal pages excluded, reviewer decisions applied.

Also writes the review sample for split NWIS stations (step 2): stations whose identity group
(B) still holds more than one coordinate, stratified by spread, with the printed location text
of each point, for a manual check of WHY the coordinates differ.

Inputs:  data/analysis/processed_metadata.parquet, data/analysis/nwis_sites/ca_stream_sites_expanded.csv,
         data/analysis/location_validation/locations.csv, data/digitized/*/page_*/*.json
Outputs: data/analysis/site_definitions/site_count_range.csv
         data/analysis/site_definitions/entry_site_keys.csv
         data/analysis/site_definitions/split_station_review.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from date_parsing import FALLBACK_STATUSES          # noqa: E402
from paper_filters import data_page_mask, measurement_page_mask   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
OUT = A / 'site_definitions'
OUT.mkdir(parents=True, exist_ok=True)
MERGE_KM = 0.5
IDENTITY_TOL_KM = (0.5, 1, 2)    # sensitivity
MAIN_TOL_KM = 1                  # site_B (to be confirmed by Annette)
IDENTITY_LINKS = ('station_number_on_page', 'station_name_on_page')
TYPES = ('Stream Discharge', 'Groundwater')


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def norm_name(s):
    s = str(s).lower()
    s = re.sub(r'\b(r|riv)\b\.?', 'river', s)
    s = re.sub(r'\b(cr|ck)\b\.?', 'creek', s)
    s = re.sub(r'\bnr\b\.?', 'near', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def load():
    """Metadata ENTRIES (one per site the LLM described on a page), same filters as the
    six-panel figure (07_ca_combined_6panel.py): tribal pages out, reviewer decisions applied,
    measurement-dated, precise/approximate rebuilt location, data page, inside the CA polygon.
    Entries, not tables: one table can list many stations."""
    import geopandas as gpd
    from paper_filters import drop_tribal, apply_review
    df = pd.read_parquet(A / 'processed_metadata.parquet')
    df = apply_review(drop_tribal(df, verbose=False), verbose=False)
    df = df[df.water_type_clean.isin(TYPES) & df.year_start.notna()
            & ~df.date_status.isin(FALLBACK_STATUSES) & measurement_page_mask(df)
            & df.precision_class.isin(['precise', 'approximate']) & df.lat_combined.notna()].copy()
    ca = gpd.read_file(A / 'spatial/ca_shapefile/ca_poly.shp').to_crs('EPSG:4326')
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_combined, df.lat_combined), crs='EPSG:4326')
    df = pd.DataFrame(gpd.sjoin(g, ca[['geometry']], how='inner', predicate='within').drop(columns='geometry'))
    loc = pd.read_csv(A / 'location_validation/locations.csv', low_memory=False,
                      usecols=['meta_row', 'nwis_site_no', 'nwis_site_source', 'well_number'])
    df = df.drop(columns=['nwis_site_no', 'nwis_site_source', 'well_number'], errors='ignore').merge(loc, on='meta_row', how='left')
    nm = pd.read_csv(A / 'nwis_sites/ca_stream_sites_expanded.csv', dtype={'site_no': str}, low_memory=False,
                     usecols=['site_no', 'station_nm']).drop_duplicates('site_no').set_index('site_no').station_nm
    c = pd.DataFrame({
        'meta_row': df.meta_row, 'doc_id': df.id.astype(int), 'page_number': df.page_number.astype(int),
        'water_type': df.water_type_clean, 'latitude': df.lat_combined, 'longitude': df.lon_combined,
        'year_start': df.year_start, 'coord_method': df.coord_method, 'watersource_name': df.watersource_name,
        'nwis_site_no': df.nwis_site_no.astype(str).str.replace(r'\.0$', '', regex=True).where(df.nwis_site_no.notna()),
        # identity = station NUMBER or NAME printed on the row's page (not name_cluster: a centroid of gauges)
        'nwis_match': df.nwis_site_source.map({'number_on_page': 'station_number_on_page',
                                               'name_on_page': 'station_name_on_page'}),
        'well_number': df.well_number})
    c['nwis_station_name'] = c.nwis_site_no.map(nm)
    c['xy'] = c.latitude.round(5).astype(str) + ',' + c.longitude.round(5).astype(str)
    return c.reset_index(drop=True)


def split_identity(c, tol):
    """Identity key + cluster number: points of one identity group closer than tol km chain together."""
    out = c.site_B.copy()
    grp = c[c.site_B.str.startswith(('nwis:', 'well:'))]
    for key, g in grp.groupby('site_B'):
        pts = g.drop_duplicates('xy')
        if len(pts) == 1:
            continue
        la, lo = pts.latitude.values, pts.longitude.values
        d = hav_km(la[:, None], lo[:, None], la[None, :], lo[None, :]) <= tol
        lab = -np.ones(len(pts), int)
        for i in range(len(pts)):
            if lab[i] < 0:
                stack, lab[i] = [i], i
                while stack:
                    j = stack.pop()
                    for k in np.where(d[j] & (lab < 0))[0]:
                        lab[k] = i
                        stack.append(k)
        m = dict(zip(pts.xy, lab))
        out.loc[g.index] = key + '#' + g.xy.map(m).astype(str)
    return out


def site_keys(c):
    c = c.copy()
    c['site_A'] = 'xy:' + c.xy
    ident = c.nwis_match.isin(IDENTITY_LINKS) & c.nwis_site_no.notna()
    wn = c.water_type.eq('Groundwater') & c.well_number.notna() & ~ident
    c['site_B'] = c.site_A
    c.loc[ident, 'site_B'] = 'nwis:' + c.loc[ident, 'nwis_site_no'].astype(str).str.replace(r'\.0$', '', regex=True)
    c.loc[wn, 'site_B'] = 'well:' + c.loc[wn, 'well_number'].astype(str).str.upper().str.replace(r'\s', '', regex=True)
    c['ident_group'] = c.site_B            # identity alone (used by the review sample)
    # Identity merges only points that also agree in location (review of 40 split stations,
    # 2026-11-14: beyond ~2 km a 'same NWIS station' record is usually a documented different
    # gauge location or another station's coordinates). Single-linkage within each identity
    # group at tolerance T; points farther than T from every other point stay separate sites.
    for tol in IDENTITY_TOL_KM:
        c[f'site_B{tol:g}'] = split_identity(c, tol)
    c['site_B'] = c[f'site_B{MAIN_TOL_KM:g}']
    # C: merge identity-less coordinate sites sharing a normalized name within MERGE_KM (union-find)
    c['site_C'] = c.site_B
    rest = c[c.site_B.str.startswith('xy:')].copy()
    rest['nm'] = rest.watersource_name.map(norm_name)
    pts = rest.groupby(['water_type', 'nm', 'site_B'], as_index=False)[['latitude', 'longitude']].first()
    parent = {k: k for k in pts.site_B}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for _, g in pts[pts.nm.ne('') & pts.nm.ne('nan')].groupby(['water_type', 'nm']):
        if len(g) < 2:
            continue
        la, lo, ks = g.latitude.values, g.longitude.values, g.site_B.values
        d = hav_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        for i, j in zip(*np.where(np.triu(d <= MERGE_KM, 1))):
            parent[find(ks[i])] = find(ks[j])
    m = c.site_B.str.startswith('xy:')
    c.loc[m, 'site_C'] = c.loc[m, 'site_B'].map(lambda k: 'grp:' + find(k)[3:] if k in parent else k)
    return c


def counts(c):
    rows = []
    for wt, g in c.groupby('water_type'):
        dec = (g.year_start // 10 * 10).astype(int)
        for d in ['A'] + [f'B{t:g}' for t in IDENTITY_TOL_KM] + ['C']:
            k = g[f'site_{d}']
            rows.append({'water_type': wt, 'definition': d, 'sites': k.nunique(),
                         'pre_1900': k[dec < 1900].nunique(), 'pre_1940': k[dec < 1940].nunique(),
                         'entries': len(g)})
    return pd.DataFrame(rows)


def location_text(doc, page, key):
    f = ROOT / f'data/digitized/{doc}/page_{page}/{doc}_page_{page}.json'
    if not f.exists():
        return ''
    t = ' '.join(ch.get('content', '') for ch in json.loads(f.read_text(encoding='utf-8'))['result']['chunks'])
    t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', t))
    i = t.upper().find(key.upper()) if key else -1
    m = re.search(r'Location\W{0,5}.{0,260}', t[i:] if i >= 0 else t, re.I)
    return m.group(0) if m else ''


def review_sample(c, n_per=10, seed=14):
    s = c[c.water_type.eq('Stream Discharge') & c.ident_group.str.startswith('nwis:')]
    s = s.assign(site_B=s.ident_group)
    pts = s.groupby(['site_B', 'xy']).agg(lat=('latitude', 'first'), lon=('longitude', 'first'),
                                         tables=('doc_id', 'size'), docs=('doc_id', 'nunique'),
                                         yr_min=('year_start', 'min'), yr_max=('year_start', 'max'),
                                         method=('coord_method', lambda x: x.mode()[0]),
                                         doc=('doc_id', 'first'), page=('page_number', 'first'),
                                         name=('nwis_station_name', 'first')).reset_index()
    spread = pts.groupby('site_B').apply(
        lambda g: hav_km(g.lat.values[:, None], g.lon.values[:, None], g.lat.values[None, :], g.lon.values[None, :]).max(),
        include_groups=False).rename('spread_km')
    npts = pts.groupby('site_B').size().rename('n_points')
    st = pd.concat([spread, npts], axis=1)
    st = st[st.n_points > 1]
    st['stratum'] = pd.cut(st.spread_km, [-0.01, 0.5, 2, 10, 1e9], labels=['<0.5 km', '0.5-2 km', '2-10 km', '>10 km'])
    pick = st.groupby('stratum', observed=True).apply(
        lambda g: g.sample(min(n_per, len(g)), random_state=seed), include_groups=False).reset_index(level=0)
    r = pts[pts.site_B.isin(pick.index)].merge(pick, left_on='site_B', right_index=True)
    r['place_key'] = r.name.astype(str).str.split(r'\s(?:A|AT|NR|NEAR|BL|BELOW|AB|ABOVE)\s', n=1, regex=True).str[-1].str[:12]
    r['location_text'] = [location_text(int(d), int(p), k) for d, p, k in zip(r.doc, r.page, r.place_key)]
    r = r.sort_values(['stratum', 'site_B', 'yr_min'])
    for col in ['claude_class', 'claude_note', 'annette_class', 'annette_note']:
        r[col] = ''
    return st, r.drop(columns=['place_key'])


def main():
    c = site_keys(load())
    keys = ['site_A'] + [f'site_B{t:g}' for t in IDENTITY_TOL_KM] + ['site_B', 'site_C']
    c[['meta_row', 'doc_id', 'page_number', 'water_type'] + keys].to_csv(OUT / 'entry_site_keys.csv', index=False)
    cnt = counts(c)
    cnt.to_csv(OUT / 'site_count_range.csv', index=False)
    print(cnt.to_string(index=False), '\n')
    st, r = review_sample(c)
    print('split NWIS-identity stream stations by stratum (all):')
    print(st.stratum.value_counts().sort_index().to_string())
    f = OUT / 'split_station_review.csv'
    if f.exists():      # graded sample (claude_/annette_ columns): never overwrite
        print(f'\nreview sample kept as is: {f}')
    else:
        r.to_csv(f, index=False)
        print(f'\nreview sample: {r.site_B.nunique()} stations, {len(r)} points -> {f}')


if __name__ == '__main__':
    main()
