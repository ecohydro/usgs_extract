"""
Shared date parser for the LLM-extracted `dates_of_recording` field.

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


def is_dated(df):
    """Boolean mask: rows that count as dated under the protocol."""
    return df['date_status'].isin(DATED_STATUSES)
