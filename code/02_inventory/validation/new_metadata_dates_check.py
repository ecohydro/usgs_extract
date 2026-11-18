"""Cross-check of the dates in the new metadata table (Anna, Qwen3.6-27B) against our table-text dates.

  python new_metadata_dates_check.py

Compares, per table, the year range from the new metadata (date_start / date_end over the table's
water-related locations) with our measurement dates (table_dates.csv, tiers A-D; the period-of-record and
construction-year fallbacks E/F are not measurement dates and count as undated here). Also scores both
methods against the dates in Anna's answer keys (test, validation, validation2), aggregated to table level
the same way, so the two methods are judged on the same tables.

Writes data/analysis/date_parsing/new_metadata_check/:
  table_compare.csv         one row per table: both year ranges, agreement class, validity flags
  summary.csv               agreement counts overall and by water type
  key_scores.csv            both methods vs the answer keys (start year, end year, exact and +-1)
  invalid_years.csv         new-metadata rows with a year < 1800 or after the report's publication year
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / 'data/analysis'
NEW = A / 'evaluation/metadata/full_run/metadata_per_location.csv'
KEYS = {'test': A / 'evaluation/metadata/key/answer_key.csv',
        'validation': A / 'evaluation/metadata/key_validation/answer_key.csv',
        'validation2': A / 'evaluation/metadata/key_validation2/answer_key.csv'}
OUT = A / 'date_parsing/new_metadata_check'
K = ['doc_id', 'page_number', 'table_number']
MEASUREMENT_TIERS = ('A', 'B', 'C', 'D')          # table_dates tiers that are measurement dates
YEAR = re.compile(r'(?<!\d)(\d{4})(?!\d)')


def year_of(s):
    m = YEAR.search(str(s))
    return int(m.group(1)) if m else None


def table_years(df):
    """Table-level first/last year from location rows (water-related rows only)."""
    d = df[df.data_type.fillna('') != 'Not Water Related'].copy()
    d['ys'] = d.date_start.map(year_of)
    d['ye'] = d.date_end.map(year_of)
    d['ye'] = d.ye.fillna(d.ys)
    return d.groupby(K).agg(ys=('ys', 'min'), ye=('ye', 'max'), dtype=('data_type', lambda s: s.mode().iat[0] if len(s.dropna()) else '')).reset_index()


def ours():
    t = pd.read_csv(A / 'table_dates.csv', usecols=['id', 'page_number', 'table_number', 'pub_year',
                                                     'year_start', 'year_end', 'date_tier'])
    t = t.rename(columns={'id': 'doc_id'})
    meas = t.date_tier.fillna('').str[0].isin(MEASUREMENT_TIERS)
    t['our_ys'] = t.year_start.where(meas)
    t['our_ye'] = t.year_end.where(meas)
    return t[K + ['pub_year', 'date_tier', 'our_ys', 'our_ye']]


def classify(r):
    a, b = (r.new_ys, r.new_ye), (r.our_ys, r.our_ye)
    if pd.isna(a[0]) and pd.isna(b[0]):
        return 'both undated'
    if pd.isna(a[0]):
        return 'only ours dated'
    if pd.isna(b[0]):
        return 'only new dated'
    if a == b:
        return 'same range'
    if a[0] == b[0]:
        return 'same start'
    if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) <= 1:
        return 'off by one (water year)'
    if a[0] <= b[1] and b[0] <= a[1]:
        return 'overlap'
    return 'no overlap'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    new = pd.read_csv(NEW, dtype={'date_start': str, 'date_end': str}, keep_default_na=False, na_values=[''])
    new = new[new.status == 'ok']
    nt = table_years(new).rename(columns={'ys': 'new_ys', 'ye': 'new_ye', 'dtype': 'data_type'})
    c = nt.merge(ours(), on=K, how='left')
    c['agreement'] = c.apply(classify, axis=1)
    c['start_diff'] = c.new_ys - c.our_ys
    c['new_before_1800'] = c.new_ys < 1800
    c['new_after_pub'] = (c.new_ye > c.pub_year) | (c.new_ys > c.pub_year)
    c.to_csv(OUT / 'table_compare.csv', index=False)

    # invalid years at row level
    r = new.merge(c[K + ['pub_year']], on=K, how='left')
    r['ys'], r['ye'] = r.date_start.map(year_of), r.date_end.map(year_of)
    bad = r[(r.ys < 1800) | (r.ys > r.pub_year) | (r.ye > r.pub_year) | (r.ye < 1800)]
    bad[K + ['location_number', 'location', 'data_type', 'date_start', 'date_end', 'pub_year']].to_csv(
        OUT / 'invalid_years.csv', index=False)

    s = [c.agreement.value_counts().rename('tables').to_frame().assign(group='all')]
    for dt, g in c.groupby('data_type'):
        s.append(g.agreement.value_counts().rename('tables').to_frame().assign(group=dt))
    s = pd.concat(s).reset_index().rename(columns={'index': 'agreement'})
    s['share'] = s.tables / s.groupby('group').tables.transform('sum')
    s.to_csv(OUT / 'summary.csv', index=False)

    rows = []
    for name, f in KEYS.items():
        key = table_years(pd.read_csv(f, dtype={'date_start': str, 'date_end': str}, keep_default_na=False,
                                      na_values=['']))
        key = key.rename(columns={'ys': 'key_ys', 'ye': 'key_ye'}).drop(columns='dtype')
        j = key.merge(c, on=K, how='inner')
        for who in ('new', 'our'):
            for end, kcol in (('ys', 'key_ys'), ('ye', 'key_ye')):
                g = j[j[kcol].notna()]
                pred = g[f'{who}_{end}']
                rows.append(dict(set=name, method='new metadata' if who == 'new' else 'our table dates',
                                 field='start year' if end == 'ys' else 'end year', key_dated=len(g),
                                 exact=(pred == g[kcol]).mean(), within_1=((pred - g[kcol]).abs() <= 1).mean(),
                                 missing=pred.isna().mean()))
            g = j[j.key_ys.isna()]
            rows.append(dict(set=name, method='new metadata' if who == 'new' else 'our table dates',
                             field='dated where key is undated', key_dated=len(g),
                             exact=g[f'{who}_ys'].notna().mean()))
    pd.DataFrame(rows).to_csv(OUT / 'key_scores.csv', index=False)
    print(s[s.group == 'all'].to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('invalid-year rows', len(bad), '| tables new start <1800', int(c.new_before_1800.sum()),
          '| tables new after pub year', int(c.new_after_pub.sum()))


if __name__ == '__main__':
    main()
