"""
Searchable table-level catalog of the restored data (one row per published table CSV).

Rows: the LLM-processed tables in table_level_metadata_published.csv, EXCEPT pages withheld
for tribal data sovereignty (12_tribal_lands_screen.py). Out-of-California tables are kept
and labeled (state, in_california).

Columns (see CATALOG_COLUMNS below for order):
  identity / files   doc_id, page_number, table_number, page_folder, table_csv, page_png,
                     page_json, has_png, has_json (paths relative to the digitized data root)
  document           doc_title, series_title, series_number, index_id, year_published,
                     authors, usgs_url, doi
  content            water_type, water_type_source, table_caption, page_type,
                     n_measurements (CSV rows), watersource_name, location_text,
                     temporal_resolution, units, keyterms (LLM fields from the entry the
                     location came from, else the page's first entry)
  dates              year_start, year_end, date_tier, period_of_record_start/end,
                     construction_start/end, date_status (evidence tier; table_dates.csv via
                     date_parsing.apply_table_dates, validated 2026-11-04),
                     year_start_llm, year_end_llm (old page-level LLM string)
  location           latitude, longitude, coord_method, precision_km, precision_class,
                     evidence_scope, location_match, location_ambiguous, n_site_locations
  search keys        state, in_california, county, huc8, huc8_name, huc12, b118_basin_id,
                     b118_basin_name, nwis_site_no, nwis_station_name, nwis_match_km,
                     nwis_match
  quality            digitization_quality, table_content, frac_ocr_suspect, frac_letter_zero, frac_empty,
                     n_data_rows, n_cols

Search-key rules (a key is left blank when the location is too coarse for it):
  state, county  precise or approximate points
  huc8           precise or approximate points (HUC8s are large, ~1,000s km2)
  huc12          precise points only (the paper's HUC12 claims use the same rule)
  b118 basin     point inside a DWR Bulletin 118 basin/subbasin (precise/approximate), or
                 the basin named on the page for basin-class locations
  nwis           station_number_on_page: an NWIS station number printed on the page and
                   tied to this entry (05_nwis_station_numbers.py / printed tier)
                 nearest_named: precise surface-water point; nearest NWIS stream site within
                   NWIS_STREAM_KM that shares a distinctive name word with watersource_name
                 nearest_well: precise groundwater point; nearest NWIS groundwater site
                   within NWIS_WELL_KM (a candidate, not a confirmed match)

Outputs (data/catalog/, git-ignored):
  restored_tables_catalog.csv
  restored_tables_catalog.gpkg   layer 'tables' (located tables only, EPSG:4326)
  catalog_summary.csv

Tribal-lands sheet (data/withheld_tribal/, git-ignored; NOT published -- for contacting tribes):
  tribal_tables_catalog.csv      same columns + tribal_area, tribal_evidence, area_in_california,
                                 withheld_copy_folder, contact columns (blank). One row per
                                 table PER tribal area (a page on two areas appears under both),
                                 so filtering by tribal_area gives everything for that area.
  tribal_tables_catalog.xlsx     sheets 'areas' (one row per area) and 'tables'
"""
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
SP = A / 'spatial'
OUT = ROOT / 'data/catalog'
NWIS_STREAM_KM = 5.0
NWIS_WELL_KM = 0.5
SURFACE = {'Stream Discharge', 'Reservoir', 'Water Quality', 'Irrigation', 'Springs'}
KEY = ['doc_id', 'page_number']
STOP = set('''river creek near at above below calif california cal fork north south east west
middle the and of in on for from well wells spring springs canal ditch lake reservoir mile miles
county station gage gaging water mouth main branch little big upper lower slough drain lateral
tributary co company ranch city town valley basin area total nr abv blw trib'''.split())


def words(s):
    return {w for w in re.findall(r'[a-z]{3,}', str(s or '').lower()) if w not in STOP}


def load_tables():
    t = pd.read_csv(A / 'table_level_metadata_published.csv', low_memory=False)
    trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv')
    trib = trib[trib.withhold.astype(bool)].rename(columns={'id': 'doc_id'})[KEY]
    n0 = len(t)
    t = t.merge(trib.assign(_w=1), on=KEY, how='left')
    withheld = t[t._w == 1]
    t = t[t._w != 1].drop(columns='_w')
    print(f'published tables {n0:,}; withheld (tribal) {len(withheld):,}; catalog {len(t):,}')
    return t, withheld.drop(columns='_w')


def add_entry_fields(t):
    md = pd.read_csv(ROOT / 'data/metadata/main_metadata.csv', low_memory=False, encoding_errors='replace')
    md['meta_row'] = np.arange(len(md))
    md = md.rename(columns={'id': 'doc_id'})
    ent_cols = {'watersource_name': 'watersource_name', 'location': 'location_text',
                'temporal_resolution': 'temporal_resolution', 'units_of_measurement': 'units',
                'keyterms': 'keyterms'}
    ent = md[['meta_row'] + list(ent_cols)].rename(columns=ent_cols)
    first = md.drop_duplicates(KEY)[KEY + ['meta_row']].rename(columns={'meta_row': 'meta_row_first'})
    t = t.drop(columns=['watersource_name']).merge(first, on=KEY, how='left')
    t['meta_row_entry'] = t.meta_row_location.fillna(t.meta_row_first)
    t = t.merge(ent, left_on='meta_row_entry', right_on='meta_row', how='left').drop(columns='meta_row')
    doc_cols = {'Title': 'doc_title', 'Series title': 'series_title', 'Series number': 'series_number',
                'Index ID': 'index_id', 'Year Published': 'year_published', 'Author(s)': 'authors',
                'URL': 'usgs_url', 'DOI': 'doi'}
    docs = md.drop_duplicates('doc_id')[['doc_id'] + list(doc_cols)].rename(columns=doc_cols)
    return t.merge(docs, on='doc_id', how='left')


def add_dates(t):
    """Table dates already come from table_level_metadata (date_parsing.apply_table_dates);
    only the table title is taken from table_dates.csv."""
    td = pd.read_csv(A / 'table_dates.csv', usecols=['id', 'page_number', 'table_number', 'caption'])
    td = td.rename(columns={'id': 'doc_id', 'caption': 'table_caption'})
    t = t.rename(columns={'por_start': 'period_of_record_start', 'por_end': 'period_of_record_end'})
    return t.merge(td, on=KEY + ['table_number'], how='left')


def add_files(t):
    pt = pd.read_csv(A / 'page_types.csv', usecols=['id', 'page_number', 'page_type', 'has_json'])
    t = t.merge(pt.rename(columns={'id': 'doc_id'}), on=KEY, how='left')
    d, p, n = t.doc_id.astype(int).astype(str), t.page_number.astype(int).astype(str), t.table_number.astype(int).astype(str)
    t['page_folder'] = d + '/page_' + p
    t['table_csv'] = t.page_folder + '/' + d + '_page_' + p + '_table' + n + '.csv'
    t['page_png'] = t.page_folder + '/' + d + '_page_' + p + '.png'
    t['page_json'] = t.page_folder + '/' + d + '_page_' + p + '.json'
    dig = ROOT / 'data/digitized'
    pages = t.drop_duplicates('page_folder')
    has_png = {f: (dig / png).exists() for f, png in zip(pages.page_folder, pages.page_png)}
    t['has_png'] = t.page_folder.map(has_png)
    return t


def first_hit(points, polys, cols):
    j = gpd.sjoin(points, polys[cols + ['geometry']], how='left', predicate='within')
    return j[~j.index.duplicated()][cols]


def add_search_keys(t):
    t['latitude'], t['longitude'] = t.pop('lat_combined'), t.pop('lon_combined')
    loc = t.latitude.notna() & t.precision_class.isin(['precise', 'approximate'])
    pts = gpd.GeoDataFrame(t.loc[loc, []], geometry=gpd.points_from_xy(t.longitude[loc], t.latitude[loc]),
                           crs='EPSG:4326')
    states = gpd.read_file(SP / 'usa_shapefile/CUSA_States.shp').to_crs(4326)
    st = first_hit(pts, states, ['NAME', 'STATEFP'])
    counties = gpd.read_file(SP / 'ca_counties/tl_2023_us_county.shp').to_crs(4326)
    co = first_hit(pts, counties.rename(columns={'NAME': 'county'}), ['county', 'STATEFP'])
    co.loc[co.STATEFP.ne(st.STATEFP.reindex(co.index)), 'county'] = np.nan  # border slivers
    h8 = gpd.read_file(SP / 'ca_wbd/ca_huc8.geojson').to_crs(4326).rename(columns={'name': 'huc8_name'})
    h12 = gpd.read_file(SP / 'ca_wbd/ca_huc12.geojson').to_crs(4326)
    b118 = gpd.read_file(SP / 'sgma/sgma_2019_basin_prioritization.geojson').to_crs(4326).rename(
        columns={'Basin_Subbasin_Number': 'b118_basin_id', 'Basin_Subbasin_Name': 'b118_basin_name'})
    t['state'] = st.NAME
    t['county'] = co.county
    t[['huc8', 'huc8_name']] = first_hit(pts, h8, ['huc8', 'huc8_name'])
    t['huc12'] = first_hit(pts, h12, ['huc12']).huc12
    t.loc[t.precision_class != 'precise', 'huc12'] = np.nan
    t[['b118_basin_id', 'b118_basin_name']] = first_hit(pts, b118, ['b118_basin_id', 'b118_basin_name'])
    # basin-class locations: the B118 basin named on the page (locations.csv)
    L = pd.read_csv(A / 'location_validation/locations.csv', usecols=['meta_row', 'b118_basin', 'nwis_site_no'],
                    dtype={'nwis_site_no': str})
    t = t.merge(L.rename(columns={'meta_row': 'meta_row_location', 'b118_basin': '_b118_named',
                                  'nwis_site_no': '_nwis_page'}), on='meta_row_location', how='left')
    isb = t.precision_class.eq('basin')
    t.loc[isb, 'b118_basin_name'] = t.loc[isb, '_b118_named']
    t['in_california'] = t.state.eq('California').astype('boolean')
    t.loc[t.state.isna(), 'in_california'] = pd.NA
    return t


def _nearest(q, sites, max_km, need_word):
    """Nearest NWIS site to each precise point in q (optionally sharing a name word)."""
    if q.empty:
        return pd.DataFrame(columns=['site_no', 'd'])
    P = gpd.GeoDataFrame(q[['watersource_name']], geometry=gpd.points_from_xy(q.longitude, q.latitude),
                         crs=4326).to_crs(3310)
    S = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.dec_long_va, sites.dec_lat_va),
                         crs=4326).to_crs(3310)
    if not need_word:
        j = gpd.sjoin_nearest(P, S[['site_no', 'geometry']], max_distance=max_km * 1000, distance_col='d')
    else:  # all sites within max_km; keep the nearest one sharing a distinctive name word
        j = gpd.sjoin(P.assign(geometry=P.buffer(max_km * 1000)), S[['site_no', 'station_nm', 'geometry']],
                      predicate='intersects')
        sx = S.set_index('site_no').geometry
        j['d'] = P.geometry.reindex(j.index).distance(sx.reindex(j.site_no).set_axis(j.index))
        j = j[[bool(words(a) & words(b)) for a, b in zip(j.watersource_name, j.station_nm)]]
    j = j.sort_values('d')
    return j[~j.index.duplicated()][['site_no', 'd']]


def add_nwis(t):
    """NWIS station keys; returns (t, validation of the nearest_named rule)."""
    cols = ['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va']
    stream = pd.read_csv(A / 'nwis_sites/ca_stream_sites_expanded.csv', dtype={'site_no': str},
                         usecols=cols).drop_duplicates('site_no')
    gw = pd.read_csv(A / 'nwis_sites/ca_groundwater_sites.csv', dtype={'site_no': str},
                     usecols=cols).drop_duplicates('site_no')
    names = pd.concat([stream, gw]).drop_duplicates('site_no').set_index('site_no').station_nm
    t['nwis_site_no'], t['nwis_match_km'], t['nwis_match'] = None, np.nan, None
    on_page = t._nwis_page.notna()
    t.loc[on_page, 'nwis_site_no'] = t.loc[on_page, '_nwis_page']
    t.loc[on_page, 'nwis_match_km'] = 0.0
    t.loc[on_page, 'nwis_match'] = 'station_number_on_page'
    precise = t.precision_class.eq('precise')
    surface = t.water_type.isin(SURFACE)
    for mask, sites, km, need, label in [(surface, stream, NWIS_STREAM_KM, True, 'nearest_named'),
                                         (t.water_type.eq('Groundwater'), gw, NWIS_WELL_KM, False, 'nearest_well')]:
        j = _nearest(t[mask & precise & ~on_page], sites, km, need)
        t.loc[j.index, 'nwis_site_no'] = j.site_no
        t.loc[j.index, 'nwis_match_km'] = (j.d / 1000).round(2)
        t.loc[j.index, 'nwis_match'] = label
    t['nwis_station_name'] = t.nwis_site_no.map(names)
    # validation: run nearest_named on tables whose station number IS printed on the page
    v = t[on_page & precise & surface]
    j = _nearest(v, stream, NWIS_STREAM_KM, True)
    got = j.site_no.reindex(v.index)
    val = pd.Series({'n': len(v), 'same_station': int((got == v.nwis_site_no).sum()),
                     'other_station': int((got.notna() & (got != v.nwis_site_no)).sum()),
                     'no_match': int(got.isna().sum())})
    return t, val


def add_quality(t):
    q = pd.read_csv(A / 'table_quality.csv')
    cols = ['digitization_quality', 'table_content', 'frac_ocr_suspect', 'frac_letter_zero', 'frac_empty', 'n_data_rows', 'n_cols']
    return t.merge(q[KEY + ['table_number'] + cols], on=KEY + ['table_number'], how='left')


CATALOG_COLUMNS = [
    'doc_id', 'page_number', 'table_number', 'page_folder', 'table_csv', 'page_png', 'page_json',
    'has_png', 'has_json',
    'doc_title', 'series_title', 'series_number', 'index_id', 'year_published', 'authors', 'usgs_url', 'doi',
    'water_type', 'water_type_source', 'table_caption', 'page_type', 'n_measurements', 'watersource_name',
    'location_text', 'temporal_resolution', 'units', 'keyterms',
    'year_start', 'year_end', 'date_status', 'date_tier', 'period_of_record_start', 'period_of_record_end',
    'construction_start', 'construction_end', 'year_start_llm', 'year_end_llm',
    'latitude', 'longitude', 'coord_method', 'precision_km', 'precision_class', 'evidence_scope',
    'location_match', 'location_ambiguous', 'n_site_locations',
    'state', 'in_california', 'county', 'huc8', 'huc8_name', 'huc12', 'b118_basin_id', 'b118_basin_name',
    'nwis_site_no', 'nwis_station_name', 'nwis_match_km', 'nwis_match',
    'digitization_quality', 'table_content', 'frac_ocr_suspect', 'frac_letter_zero', 'frac_empty', 'n_data_rows', 'n_cols',
]


def enrich(t):
    t = t.rename(columns={'water_type_final': 'water_type', 'assignment_source': 'water_type_source',
                          'number_rows': 'n_measurements'})
    t = add_entry_fields(t)
    t = add_dates(t)
    t = add_files(t)
    t = add_search_keys(t)
    t, val = add_nwis(t)
    t = add_quality(t)
    for c in ['huc8', 'huc12']:
        t[c] = t[c].astype('string')
    for c in ['year_start', 'year_end', 'period_of_record_start', 'period_of_record_end',
              'construction_start', 'construction_end', 'year_start_llm', 'year_end_llm',
              'year_published', 'n_measurements', 'doc_id', 'page_number', 'table_number']:
        t[c] = pd.to_numeric(t[c], errors='coerce').astype('Int64')
    return t, val


TRIBAL_OUT = ROOT / 'data/withheld_tribal'
CONTACT_COLUMNS = ['tribe_contact', 'date_contacted', 'date_data_shared', 'contact_notes']


def write_tribal_sheet(w):
    """Tables on withheld pages, one row per table per tribal area (for contacting tribes)."""
    man = pd.read_csv(TRIBAL_OUT / '_withheld_manifest.csv').rename(
        columns={'id': 'doc_id', 'evidence': 'tribal_evidence'})
    man['withheld_copy_folder'] = ('data/withheld_tribal/' + man.folder + '/' + man.doc_id.astype(str)
                                   + '/page_' + man.page_number.astype(str))
    aian = gpd.read_file(SP / 'tribal_lands/aiannh_2023/tl_2023_us_aiannh.shp')
    ca = gpd.read_file(SP / 'usa_shapefile/CUSA_States.shp').to_crs(aian.crs)
    ca_names = set(aian.loc[aian.intersects(ca[ca.NAME == 'California'].union_all()), 'NAMELSAD'])
    man['area_in_california'] = man.tribal_area.isin(ca_names)
    T = w.merge(man[KEY + ['tribal_area', 'tribal_evidence', 'area_in_california', 'withheld_copy_folder']],
                on=KEY, how='left')
    for c in CONTACT_COLUMNS:
        T[c] = ''
    cols = ['tribal_area', 'area_in_california', 'tribal_evidence', 'withheld_copy_folder'] +         CATALOG_COLUMNS + CONTACT_COLUMNS
    T = T[cols].sort_values(['tribal_area', 'doc_id', 'page_number', 'table_number'])
    T.to_csv(TRIBAL_OUT / 'tribal_tables_catalog.csv', index=False)
    A_ = (T.groupby('tribal_area')
          .agg(area_in_california=('area_in_california', 'first'), documents=('doc_id', 'nunique'),
               pages=('page_folder', 'nunique'), tables=('table_number', 'size'),
               measurements=('n_measurements', 'sum'),
               water_types=('water_type', lambda x: '; '.join(x.value_counts().index)),
               counties=('county', lambda x: '; '.join(sorted(x.dropna().unique()))),
               year_start=('year_start', 'min'), year_end=('year_end', 'max'))
          .sort_values('measurements', ascending=False).reset_index())
    for c in CONTACT_COLUMNS:
        A_[c] = ''
    with pd.ExcelWriter(TRIBAL_OUT / 'tribal_tables_catalog.xlsx') as xw:
        A_.to_excel(xw, sheet_name='areas', index=False)
        T.to_excel(xw, sheet_name='tables', index=False)
    print(f'tribal sheet: {len(T):,} table-area rows, {T.drop_duplicates(KEY + ["table_number"]).shape[0]:,} '
          f'tables, {len(A_)} areas -> {TRIBAL_OUT}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t, withheld = load_tables()
    t, val = enrich(t)
    w, _ = enrich(withheld)
    write_tribal_sheet(w)
    cat = t[CATALOG_COLUMNS].sort_values(['doc_id', 'page_number', 'table_number'])
    cat.to_csv(OUT / 'restored_tables_catalog.csv', index=False)
    g = cat[cat.latitude.notna()]
    gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.longitude, g.latitude), crs=4326).to_file(
        OUT / 'restored_tables_catalog.gpkg', layer='tables', driver='GPKG')

    s = {'tables': len(cat), 'measurements': int(cat.n_measurements.sum()),
         'documents': cat.doc_id.nunique(), 'pages': cat.page_folder.nunique(),
         'located tables (precise/approximate/basin)': int(cat.latitude.notna().sum()),
         'precise': int(cat.precision_class.eq('precise').sum()),
         'approximate': int(cat.precision_class.eq('approximate').sum()),
         'basin': int(cat.precision_class.eq('basin').sum()),
         'unlocated': int(cat.precision_class.eq('unlocated').sum()),
         'location_ambiguous': int(cat.location_ambiguous.fillna(False).astype(bool).sum()),
         'in California': int(cat.in_california.eq(True).sum()),
         'outside California': int(cat.in_california.eq(False).sum()),
         'with county': int(cat.county.notna().sum()), 'with huc8': int(cat.huc8.notna().sum()),
         'with huc12': int(cat.huc12.notna().sum()), 'with b118 basin': int(cat.b118_basin_name.notna().sum()),
         'dated (table-level)': int(cat.year_start.notna().sum())}
    for k, v in cat.nwis_match.value_counts().items():
        s[f'nwis {k}'] = int(v)
    for k, v in cat.digitization_quality.value_counts().items():
        s[f'quality {k}'] = int(v)
    if val is not None:
        for k, v in val.items():
            s[f'nwis nearest_named validation: {k}'] = int(v)
    S = pd.DataFrame(list(s.items()), columns=['item', 'value'])
    S.to_csv(OUT / 'catalog_summary.csv', index=False)
    print(S.to_string(index=False))


if __name__ == '__main__':
    main()
