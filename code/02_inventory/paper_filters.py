"""
Shared filters for paper figures and statistics (import; do not copy).

- Tribal data sovereignty (Annette, 2026-11-03): pages withheld by the tribal-lands screen
  (12_tribal_lands_screen.py -> data/analysis/tribal_lands/tribal_screen_pages.csv,
  withhold == True) are excluded from paper statistics and figures as well as from
  publication. The paper reports only the aggregate (16_tribal_withheld_stats.py).
- Location classes (2026-10-28): point maps and site statistics use precise + approximate
  rebuilt locations. 'basin' points (B118 polygon centroids) and 'unlocated' rows are never
  plotted as points.

Usage:
    from paper_filters import drop_tribal, MAP_PRECISION, located
    df = drop_tribal(df)                     # metadata rows (id, page_number)
    t  = drop_tribal(t, id_col='doc_id')     # table rows
    pts = df[located(df)]
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRIBAL_PAGES = ROOT / 'data/analysis/tribal_lands/tribal_screen_pages.csv'
MAP_PRECISION = ('precise', 'approximate')


def withheld_pages():
    p = pd.read_csv(TRIBAL_PAGES, usecols=['id', 'page_number', 'withhold'])
    p = p[p.withhold.astype(bool)]
    return set(zip(p.id.astype(str), p.page_number.astype(int)))


def tribal_mask(df, id_col='id', page_col='page_number'):
    """True for rows on a withheld tribal page."""
    w = withheld_pages()
    ids = df[id_col].astype(str).str.replace(r'\.0$', '', regex=True)
    pages = pd.to_numeric(df[page_col], errors='coerce')
    return pd.Series([(i, int(p)) in w if pd.notna(p) else False for i, p in zip(ids, pages)],
                     index=df.index)


def drop_tribal(df, id_col='id', page_col='page_number', verbose=True):
    m = tribal_mask(df, id_col, page_col)
    if verbose:
        print(f'tribal pages excluded: {int(m.sum()):,} of {len(df):,} rows')
    return df[~m].copy()


def located(df, classes=MAP_PRECISION):
    """Rows with a plottable rebuilt point of the given precision classes."""
    return df['precision_class'].isin(classes) & df['lat_combined'].notna() & df['lon_combined'].notna()
