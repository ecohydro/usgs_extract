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


MANUAL_REVIEW = ROOT / 'data/analysis/manual_review/earliest_records_review.csv'


def _review():
    r = pd.read_csv(MANUAL_REVIEW)
    if 'action' not in r.columns:
        r['action'] = r['verdict'].map({'exclude': 'drop'}).fillna('keep')
    r['key'] = list(zip(r.id.astype(str), r.page_number.astype(int), r.water_type))
    return r


def apply_review(df, id_col='id', page_col='page_number', type_col='water_type_clean',
                 verbose=True, drop=True):
    """Apply reviewer decisions (manual_review/earliest_records_review.csv, column `action`)
    to metadata rows of the reviewed page AND water type:
      drop              not usable for this water type (lake, out of state, artifact): removed
      retype            belongs to another water type: type_col set to new_water_type
      undate            no measurement date on the page (e.g. depth "1870-1930 FEET"): undated
      construction_year the year is the year drilled: labeled fallback (not a measurement date)
      set_year          reviewer read the true year: year_start/year_end corrected
      unlocate          the rebuilt location is wrong (e.g. an out-of-state page placed in
                        California): location cleared, row becomes unlocated
      keep              confirmed as is (also lets a single document set the earliest year,
                        see reviewed_keep())
    Several actions can be combined with ';' (e.g. "retype;set_year", "unlocate;undate").
    Run once right after loading, before any is_dated() / water-type filter.
    drop=False (published data: build_table_level_metadata.py, 00_data_prep): water-type and
    date corrections are applied but 'drop' rows are kept -- 'drop' means "not usable for a
    paper claim", not "remove the table". Adds column review_action."""
    if not MANUAL_REVIEW.exists():
        return df
    r = _review().set_index('key')
    df = df.copy()
    ids = df[id_col].astype(str).str.replace(r'\.0$', '', regex=True)
    pages = pd.to_numeric(df[page_col], errors='coerce').fillna(-1).astype(int)
    keys = pd.Series(list(zip(ids, pages, df[type_col])), index=df.index)
    act = keys.map(lambda k: r.action.get(k))
    df['review_action'] = act.where(act.notna(), df.get('review_action'))
    acts = act.fillna('').str.split(';')

    def has(a):
        return acts.map(lambda s: a in s)

    dropped = has('drop') if drop else pd.Series(False, index=df.index)
    year_cols = ['year_start', 'year_end'] + (['decade_start'] if 'decade_start' in df else [])
    for a, status in [('undate', 'undated'), ('construction_year', 'construction_year')]:
        m = has(a)
        df.loc[m, 'date_status'] = status
        if a == 'undate':
            df.loc[m, year_cols] = float('nan')
    m = has('set_year')
    if m.any():
        ys = keys[m].map(lambda k: r.at[k, 'corrected_year_start'])
        ye = keys[m].map(lambda k: r.at[k, 'corrected_year_end'])
        df.loc[m, 'year_start'] = ys
        df.loc[m, 'year_end'] = ye
        if 'decade_start' in df.columns:
            df.loc[m, 'decade_start'] = (ys // 10) * 10
    m = has('unlocate')
    for c in ['lat_combined', 'lon_combined', 'precision_km', 'evidence_scope']:
        if c in df.columns:
            df.loc[m, c] = float('nan')
    for c in ['coord_method', 'coord_source', 'precision_class']:
        if c in df.columns:
            df.loc[m, c] = 'unlocated'
    m = has('retype')  # last: the review key uses the original water type
    if m.any():
        df.loc[m, type_col] = keys[m].map(lambda k: r.at[k, 'new_water_type'])
    if verbose:
        print('  manual review applied: ' + ', '.join(
            f'{a} {int(has(a).sum())}' for a in ['drop', 'retype', 'undate', 'construction_year',
                                                  'set_year', 'unlocate', 'keep']) + ' rows')
    return df[~dropped]


def reviewed_keep(water_type):
    """Set of (id, page) a reviewer confirmed (action keep / set_year) for this water type."""
    if not MANUAL_REVIEW.exists():
        return set()
    r = _review()
    r = r[r.action.isin(['keep', 'set_year']) & (r.water_type == water_type)]
    return set(zip(r.id.astype(str), r.page_number.astype(int)))


def drop_reviewed(df, water_type, id_col='id', page_col='page_number', verbose=True):
    """Drop pages a reviewer marked 'exclude' for this water type (e.g. a lake analysis the
    LLM typed as groundwater, a time-of-day value read as a year). The review file records
    who decided and why (data/analysis/manual_review/earliest_records_review.csv)."""
    if not MANUAL_REVIEW.exists():
        return df
    r = _review()
    r = r[(r.action == 'drop') & (r.water_type == water_type)]
    bad = set(zip(r.id.astype(str), r.page_number.astype(int)))
    ids = df[id_col].astype(str).str.replace(r'\.0$', '', regex=True)
    pages = pd.to_numeric(df[page_col], errors='coerce')
    m = pd.Series([(i, int(p)) in bad if pd.notna(p) else False for i, p in zip(ids, pages)], index=df.index)
    if verbose and m.any():
        print(f'  manual review: {int(m.sum())} {water_type} rows excluded ({len(bad)} pages)')
    return df[~m]
