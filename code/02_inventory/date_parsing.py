"""
Shared date parser for the LLM-extracted `dates_of_recording` field, and (since
2026-11-04) the functions that replace it with TABLE-LEVEL dates:
  apply_entry_dates()  metadata rows  <- data/analysis/entry_dates.csv (15_entry_dates.py)
  apply_table_dates()  table rows     <- data/analysis/table_dates.csv (14_table_dates.py)
The LLM-string dates are kept as year_start_llm / year_end_llm / date_status_llm.
New date_status values = evidence tier (TIER_STATUS below); the last-resort tiers
(FALLBACK_STATUSES) count as dated by default; is_dated(df, include_fallbacks=False)
gives the strict version. Validated against Annette's grades: see
code/02_inventory/validation/score_entry_dates_vs_review.py and score_dates_vs_grades.py.

Single source of truth for year_start / year_end / date_status. Used by
00_data_prep.ipynb, build_table_level_metadata.py, and all figure scripts
(via processed_metadata.parquet). Do not copy this logic elsewhere.

Protocol (agreed 2026-10-28):
- Valid years: YEAR_MIN <= year <= publication year (and <= YEAR_MAX). A record
  cannot extend past the year its document was published, so later years are
  LLM errors (e.g. "01/01/2022-12/31/2022" filled in with the current year).
- The publication year is never used to impute a START year; it only caps the
  end year of open-ended records and rejects impossible years.

date_status values (DATED_STATUSES count as dated in figures and date-based
numbers; date-independent totals keep all rows):
    parsed       4-digit years, all within the valid range
    trimmed      4-digit years, some out of range and dropped, others kept
    open_ended   start year + "current/present/..."; year_end = publication year
    two_digit    2-digit years ("03/58", "10/23/64") expanded to 19xx, only when
                 the result is in range and start <= end
    implausible  years present but none valid (after publication, or < YEAR_MIN),
                 or 2-digit years that fail the checks
    unparseable  digits present but no recognizable year ("06/10-18", "03/02/20XX")
    placeholder  text with no year ("Not specified", "MM/DD/YYYY-MM/DD/YYYY")
    missing      blank / NaN
"""

import re
import pandas as pd

YEAR_MIN = 1800
YEAR_MAX = 1980

OPEN_ENDED = ('present', 'current', 'ongoing', 'to date', 'today')
DATED_STATUSES = ('parsed', 'trimmed', 'open_ended', 'two_digit')

_YEAR4 = re.compile(r'(?<!\d)(1[6-9]\d{2}|20\d{2})(?!\d)')
# 2-digit years are accepted only when the WHOLE string is one clean date or a
# date range of MM/YY or MM/DD/YY parts, e.g. "02/51-05/51", "10/23/64-9/10/64".
# Anything looser ("06/10-18" = June 10-18?) is too ambiguous.
_D2 = r'(?:0?[1-9]|1[0-2])/(?:\d{1,2}/)?(\d{2})'
_RANGE2 = re.compile(r'^\s*' + _D2 + r'(?:\s*(?:-|to)\s*' + _D2 + r')?\s*$')


def parse_dates(date_str, pub_year=None):
    """Return (year_start, year_end, date_status) for one dates_of_recording value."""
    if date_str is None or (isinstance(date_str, float) and pd.isna(date_str)):
        return None, None, 'missing'
    s = str(date_str).strip().lower()
    if s in ('', 'nan', 'none', 'null'):
        return None, None, 'missing'

    cap = YEAR_MAX
    if pub_year is not None and not pd.isna(pub_year):
        cap = min(cap, int(pub_year))
    open_ended = any(t in s for t in OPEN_ENDED)

    years4 = [int(y) for y in _YEAR4.findall(s)]
    if years4:
        valid = [y for y in years4 if YEAR_MIN <= y <= cap]
        if not valid:
            return None, None, 'implausible'
        start = min(valid)
        if open_ended:
            return start, max(cap, max(valid)), 'open_ended'
        status = 'parsed' if len(valid) == len(years4) else 'trimmed'
        return start, max(valid), status

    # 2-digit years: only when no 4-digit year is present
    m = _RANGE2.match(s)
    if m:
        years = [1900 + int(y) for y in m.groups() if y is not None]
        if all(YEAR_MIN <= y <= cap for y in years) and years[0] <= years[-1]:
            if open_ended:
                return min(years), cap, 'two_digit'
            return min(years), max(years), 'two_digit'
        return None, None, 'implausible'
    if re.search(r'\d', s):
        return None, None, 'unparseable'
    return None, None, 'placeholder'


def add_date_columns(df, date_col='dates_of_recording', pub_col='Year Published'):
    """Add year_start, year_end, decade_start (Int64) and date_status to df in place."""
    pub = df[pub_col] if pub_col in df.columns else pd.Series(None, index=df.index)
    out = [parse_dates(d, p) for d, p in zip(df[date_col], pub)]
    df['year_start'] = pd.array([o[0] for o in out], dtype='Int64')
    df['year_end'] = pd.array([o[1] for o in out], dtype='Int64')
    df['decade_start'] = (df['year_start'] // 10 * 10).astype('Int64')
    df['date_status'] = [o[2] for o in out]
    return df


# ---------------------------------------------------------------------------
# Table-level dates (2026-11-04) -- replace the LLM string as the dates of record.
# Tiers come from table_dates.py (per table) and 15_entry_dates.py (per metadata entry).
# date_status then names the evidence the date came from:
TIER_STATUS = {
    'A_date_column': 'table_date_column',     # a Date / Year column (or rotated Date row)
    'B_table_text': 'table_text',             # explicit dates elsewhere in the table
    'C_caption': 'table_caption',             # the table title just above the table
    'D_page_text': 'page_text',               # other text on the same page
    'E_period_of_record': 'period_of_record', # fallback: station's "Records available"
    'F_construction_year': 'construction_year',  # fallback: years wells were drilled
    'G_llm_grounded': 'llm_grounded',         # fallback: LLM date whose years are printed on the page
}
FALLBACK_STATUSES = ('period_of_record', 'construction_year', 'llm_grounded')
TABLE_DATED_STATUSES = tuple(TIER_STATUS.values())
DATED_STATUSES = DATED_STATUSES + TABLE_DATED_STATUSES   # is_dated() works on either kind
_DATE_COLS = ['year_start', 'year_end', 'decade_start', 'date_status']


def _keep_llm_dates(df):
    """Parse the LLM string (if not yet done) and keep it as *_llm columns."""
    if 'date_status' not in df.columns or df['date_status'].isin(TABLE_DATED_STATUSES).any():
        add_date_columns(df)
    for c in _DATE_COLS:
        df[c + '_llm'] = df[c]
    return df.drop(columns=_DATE_COLS)


def _finish(df):
    df['date_status'] = df['date_tier'].map(TIER_STATUS).fillna('undated')
    for c in ['year_start', 'year_end']:
        df[c] = pd.to_numeric(df[c]).astype('Int64')
    df['decade_start'] = (df['year_start'] // 10 * 10).astype('Int64')
    return df


def apply_entry_dates(df, entry_dates_path):
    """Metadata rows (needs meta_row = row position in main_metadata.csv): replace the LLM
    dates with entry dates from 15_entry_dates.py. LLM values stay as *_llm columns."""
    df = _keep_llm_dates(df)
    e = pd.read_csv(entry_dates_path, usecols=[
        'meta_row', 'year_start', 'year_end', 'date_tier', 'date_source', 'matched_tables',
        'por_start', 'por_end', 'construction_start', 'construction_end'])
    df = df.merge(e, on='meta_row', how='left', validate='one_to_one')
    return _finish(df)


def apply_table_dates(df, table_dates_path, keys=('doc_id', 'page_number', 'table_number')):
    """Table-level rows: replace dates with table_dates.csv (14_table_dates.py) per table.
    LLM values (if present) stay as *_llm columns."""
    old = [c for c in _DATE_COLS if c in df.columns]
    for c in old:  # earlier (LLM-string) dates, kept for reference
        df[c + '_llm'] = df[c]
    df = df.drop(columns=old)
    t = pd.read_csv(table_dates_path, dtype={'id': str}, usecols=[
        'id', 'page_number', 'table_number', 'year_start', 'year_end', 'date_tier',
        'por_start', 'por_end', 'construction_start', 'construction_end', 'caption'])
    t = t.rename(columns={'id': keys[0], 'caption': 'table_caption'})
    t[keys[0]] = t[keys[0]].astype(df[keys[0]].dtype)
    df = df.merge(t, on=list(keys), how='left', validate='many_to_one')
    return _finish(df)


def is_dated(df, include_fallbacks=True):
    """Boolean mask: rows that count as dated under the protocol. include_fallbacks=False
    drops the labeled last-resort tiers (period of record, construction year, grounded LLM)."""
    m = df['date_status'].isin(DATED_STATUSES)
    if not include_fallbacks:
        m &= ~df['date_status'].isin(FALLBACK_STATUSES)
    return m
