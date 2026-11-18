"""
Table-level metadata: one row per digitized table (2026-11-18, new metadata).

  python build_table_level_metadata.py   -> data/analysis/table_level_metadata.csv

Built from the entry-level file (00_data_prep.py -> processed_metadata.parquet), which holds one row per
location per table from the new metadata (Anna's pipeline) with our rebuilt locations and dates. The
Phi-4 version of this script (header classifier, page-level LLM types) is in git history and in
archive/2026-11-18_before_new_metadata/.

Rules (Annette 2026-11-18):
  water_type_final   the water type held by most of the table's locations; a tie goes to the first
                     location's type. assignment_source: single_type | majority | tie_first_location |
                     no_metadata (no usable answer; corrupted files, cannot be rerun -- excluded from
                     the published set, Anna 2026-11-18).
  dates              earliest year_start and latest year_end over the table's dated locations
                     (date_status: parsed if any location is dated, else the most common status)
  temporal_resolution  Anna's per-location value, the most common in the table
  variables          union of the locations' variables
  location           the best-ranked located point among the table's own locations
                     (precise > approximate > basin > conflict; then location_number). Also:
                     n_site_locations (distinct located points), site_spread_km (largest distance
                     between them), location_ambiguous (> 5 km), meta_row_location.
  n_measurements     19_value_counts.py (measurement estimate per table), if present
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
OUT = A / 'table_level_metadata.csv'
KEY = ['doc_id', 'page_number', 'table_number']
RANK = {'precise': 4, 'approximate': 3, 'basin': 2, 'conflict': 1}
DATED = ('parsed', 'two_digit', 'trimmed')


def _km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def water_type(g):
    t = g['water_type_clean'].fillna('')
    t = t[t.ne('')]
    if t.empty:
        return '', 'no_metadata'
    vc = t.value_counts()
    if len(vc) == 1:
        return vc.index[0], 'single_type'
    top = vc[vc == vc.iloc[0]].index
    if len(top) == 1:
        return top[0], 'majority'
    first = g.sort_values('location_number')['water_type_clean']
    return next(x for x in first if x in set(top)), 'tie_first_location'


def location(g):
    pts = g[g['precision_class'].isin(RANK) & g['lat_combined'].notna()].copy()
    out = {'location_match': 'none', 'n_site_locations': 0, 'site_spread_km': np.nan,
           'location_ambiguous': False, 'meta_row_location': np.nan}
    if pts.empty:
        return out
    pts['r'] = pts['precision_class'].map(RANK)
    best = pts.sort_values(['r', 'location_number'], ascending=[False, True]).iloc[0]
    xy = pts[['lat_combined', 'lon_combined']].round(5).drop_duplicates().values
    spread = float(_km(xy[:, :1], xy[:, 1:], xy[:, :1].T, xy[:, 1:].T).max()) if len(xy) > 1 else 0.0
    out.update({c: best[c] for c in ['lat_combined', 'lon_combined', 'coord_method', 'precision_km',
                                     'precision_class', 'evidence_scope', 'location_check', 'nwis_site_no',
                                     'nwis_site_source', 'well_number', 'b118_basin']})
    out.update({'location_match': 'entry', 'n_site_locations': len(xy), 'site_spread_km': round(spread, 2),
                'location_ambiguous': spread > 5, 'meta_row_location': best['meta_row']})
    return out


def main():
    m = pd.read_parquet(A / 'processed_metadata.parquet')
    m = m.drop(columns='doc_id').rename(columns={'id': 'doc_id'})
    rows = []
    for k, g in m.groupby(KEY, sort=True):
        wt, src = water_type(g)
        d = g[g['date_status'].isin(DATED)]
        tr = g['temporal_resolution'].replace('', np.nan).dropna()
        var = sorted({v for s in g['variables'].fillna('') for v in str(s).split(';') if v})
        types = sorted({t for t in g['water_type_clean'].fillna('') if t})
        rec = dict(zip(KEY, k))
        rec.update({
            'water_type_final': wt, 'assignment_source': src, 'water_types_in_table': ';'.join(types),
            'n_locations': len(g), 'metadata_status': g['status'].iloc[0],
            'year_start': d['year_start'].min() if len(d) else np.nan,
            'year_end': d['year_end'].max() if len(d) else np.nan,
            'date_status': 'parsed' if len(d) else g['date_status'].mode().iat[0],
            'construction_year_min': g['construction_year'].min(), 'construction_year_max': g['construction_year'].max(),
            'temporal_resolution': tr.mode().iat[0] if len(tr) else '',
            'variables': ';'.join(var),
            'watersource_name': next((x for x in g.sort_values('location_number')['watersource_name'] if isinstance(x, str) and x), ''),
            'page_type': g['page_type'].iloc[0], 'is_data_page': bool(g['is_data_page'].iloc[0]),
            'title': g['Title'].iloc[0], 'pub_year': g['Year Published'].iloc[0],
        })
        rec.update(location(g))
        rows.append(rec)
    t = pd.DataFrame(rows)
    t['decade_start'] = (t['year_start'] // 10) * 10
    t['precision_class'] = t['precision_class'].fillna('unlocated')
    t['coord_method'] = t['coord_method'].fillna('unlocated')
    vc = A / 'table_value_counts.csv'
    if vc.exists():
        v = pd.read_csv(vc, usecols=KEY + ['n_measurements'])
        t = t.merge(v, on=KEY, how='left')
    t.to_csv(OUT, index=False)
    print(f'tables {len(t):,}')
    print(t['assignment_source'].value_counts().to_string())
    print(t['water_type_final'].replace('', '(none)').value_counts().to_string())
    print(t['precision_class'].value_counts().to_string())
    print('dated tables', int(t['year_start'].notna().sum()))


if __name__ == '__main__':
    main()
