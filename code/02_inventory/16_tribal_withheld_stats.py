"""
Summary statistics for the data withheld because it lies on federally recognized tribal lands
(12_tribal_lands_screen.py), for a manuscript statement such as:
  "We restored N measurements in T tables from P pages on tribal lands; these are
   provided directly to the tribes for data sovereignty and are not featured here."

Counts follow the same rules as the published inventory:
  - tables / measurements: LLM-processed tables only (table_level_metadata_published.csv),
    so they are directly comparable with the published 2,512,776 measurements
  - all digitized tables on the withheld pages are also reported (incl. non-LLM tables)
  - years: table-level dates (table_dates.csv) -- PROVISIONAL until the date fix is final

Outputs (data/analysis/tribal_lands/, git-ignored):
  tribal_withheld_stats.csv            headline numbers
  tribal_withheld_by_water_type.csv    tables + measurements per water type
  tribal_withheld_by_area.csv          pages / tables / measurements per tribal area
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TL = ROOT / 'data/analysis/tribal_lands'

pages = pd.read_csv(TL / 'tribal_screen_pages.csv')
pages = pages[pages.withhold.astype(bool)].copy()
pages['doc_id'] = pages['id'].astype('int64')
pages['page_number'] = pages['page_number'].astype('int64')
key = ['doc_id', 'page_number']

pub = pd.read_csv(ROOT / 'data/analysis/table_level_metadata_published.csv', low_memory=False)
alltab = pd.read_csv(ROOT / 'data/analysis/table_level_metadata.csv', low_memory=False,
                     usecols=key + ['table_number', 'number_rows'])
td = pd.read_csv(ROOT / 'data/analysis/table_dates.csv', usecols=['id', 'page_number', 'table_number',
                                                                   'year_start', 'year_end'])
td = td.rename(columns={'id': 'doc_id'})

wp = pub.merge(pages[key], on=key)
wa = alltab.merge(pages[key], on=key)
wd = td.merge(pages[key], on=key).merge(wp[key + ['table_number']], on=key + ['table_number'])
dated = wd.year_start.notna()

areas = pages.tribal_areas.fillna('').str.split('; ').explode().str.strip()
areas = areas[areas != '']

stats = {
    'pages withheld': len(pages),
    'documents with a withheld page': pages.doc_id.nunique(),
    'tribal areas (federally recognized)': areas.nunique(),
    'pages withheld by text evidence only (no located point)': int((pages.tribal_areas.isna()).sum()),
    'published (LLM-processed) tables on withheld pages': len(wp),
    'published measurements on withheld pages': int(wp.number_rows.sum()),
    'share of published measurements': round(wp.number_rows.sum() / pub.number_rows.sum(), 4),
    'share of published tables': round(len(wp) / len(pub), 4),
    'all digitized tables on withheld pages (incl. non-LLM)': len(wa),
    'all digitized measurements on withheld pages (incl. non-LLM)': int(wa.number_rows.sum()),
    'published tables with a table-level date (provisional)': int(dated.sum()),
    'earliest year (table-level dates, provisional)': int(wd.year_start.min()) if dated.any() else None,
    'latest year (table-level dates, provisional)': int(wd.year_end.max()) if dated.any() else None,
    'median start year (provisional)': int(wd.year_start.median()) if dated.any() else None,
}
# California split: a page counts as California if any of its tribal areas touches CA
aian = gpd.read_file(ROOT / 'data/analysis/spatial/tribal_lands/aiannh_2023/tl_2023_us_aiannh.shp')
ca = gpd.read_file(ROOT / 'data/analysis/spatial/usa_shapefile/CUSA_States.shp').to_crs(aian.crs)
ca_names = set(aian.loc[aian.intersects(ca[ca.NAME == 'California'].union_all()), 'NAMELSAD'])
pages['in_ca'] = pages.tribal_areas.apply(
    lambda x: None if pd.isna(x) else any(n.strip() in ca_names for n in str(x).split(';')))
wpc = wp.merge(pages[key + ['in_ca']], on=key)
for lab, m in [('California', wpc.in_ca.eq(True)), ('other states', wpc.in_ca.eq(False)),
               ('named in page text only', wpc.in_ca.isna())]:
    stats[f'published measurements on withheld pages: {lab}'] = int(wpc.number_rows[m].sum())
    stats[f'withheld pages: {lab}'] = int(pages.in_ca.eq(True).sum() if lab == 'California' else
                                          pages.in_ca.eq(False).sum() if lab == 'other states' else
                                          pages.in_ca.isna().sum())
stats['tribal areas in California'] = len(set(areas) & ca_names)
S = pd.DataFrame(list(stats.items()), columns=['item', 'value'])
S.to_csv(TL / 'tribal_withheld_stats.csv', index=False)

W = (wp.groupby('water_type_final').agg(tables=('table_number', 'size'), measurements=('number_rows', 'sum'))
     .sort_values('measurements', ascending=False))
W['share_of_withheld_measurements'] = (W.measurements / W.measurements.sum()).round(3)
W.to_csv(TL / 'tribal_withheld_by_water_type.csv')

pa = (pages[key].assign(area=pages.tribal_areas.fillna('(named in page text)').str.split('; '))
      .explode('area'))
pg = pa.groupby('area').size().rename('pages')
tm = (pa.merge(wp[key + ['number_rows']], on=key)
      .groupby('area').agg(tables=('number_rows', 'size'), measurements=('number_rows', 'sum')))
A = pd.concat([pg, tm], axis=1).fillna(0).astype(int).sort_values('measurements', ascending=False)
A.to_csv(TL / 'tribal_withheld_by_area.csv')

print(S.to_string(index=False))
print()
print(W.to_string())
print()
print(A.head(15).to_string())
print(f'... {len(A)} areas -> {TL}')
