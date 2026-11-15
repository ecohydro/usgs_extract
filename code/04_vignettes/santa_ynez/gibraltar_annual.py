"""
Santa Ynez River at Gibraltar Dam: annual mean discharge, water years 1904-1979 (2026-11-14).

Same reach, two NWIS station numbers (checked from the printed station descriptions):
  11122000  "above Gibraltar Dam", pre-dam natural flow, daily values 1904-04 .. 1918-09 (NWIS site type LK)
  11123000  "below Gibraltar Dam", regulated, daily values from 1933-10
Restored (doc 3217, compilation of records through 1950, Santa Ynez River basin):
  p225 table 1  monthly and yearly mean discharge (cfs), "above Gibraltar Dam", water years 1904-1918
  p227 table 1  monthly and yearly mean discharge (cfs), "below Gibraltar Dam", water years 1920-1950
                ("Records show only flow passing Gibraltar Dam"; 1920-48 furnished by the city)
Footnote flags kept per value: * revised; # partly estimated on basis of weather records;
a furnished by city of Santa Barbara.

Checks (written to data/analysis/santa_ynez/):
  1. each restored year: printed yearly mean vs day-weighted mean of its 12 printed monthly means
  2. restored vs NWIS water-year means where both exist (1904-1918 pre-dam; 1934-1950 below dam)
Output: gibraltar_annual.csv, gibraltar_checks.csv, manuscript/figures/santa_ynez_gibraltar_annual.png
"""
import calendar
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIG = ROOT / 'data/digitized/3217'
OUT = ROOT / 'data/analysis/santa_ynez'
FIG = ROOT / 'manuscript/figures'
MONTHS = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
GIBRALTAR_DAM, JUNCAL_DAM = 1920, 1930
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7})


def clean(tok):
    """OCR cell -> (value or NaN, flags, note)."""
    s = str(tok).strip() if pd.notna(tok) else ''
    flags = ''.join(c for c in s[:2] if c in 'a*#')
    s = s.lstrip('a*#').strip()
    if s in ('', '-', '--', 'nan'):
        return np.nan, flags, 'missing'
    if s in ('o', 'O', '0'):
        return 0.0, flags, ''
    s = s.replace('O', '0').replace('o', '0').replace('l', '1').replace(' ', '')
    if s.startswith(','):                      # leading digit lost ("',050'" for 1,050)
        return np.nan, flags, f'unreadable {tok!r}'
    if re.fullmatch(r'\d{1,3}(,\d{3})+(\.\d+)?', s):
        s = s.replace(',', '')                 # thousands separator
    elif ',' in s:
        s = s.replace(',', '.')                # decimal comma ("4,81" -> 4.81)
    try:
        return float(s), flags, ''
    except ValueError:
        return np.nan, flags, f'unreadable {tok!r}'


def read_table(page, table, station):
    raw = pd.read_csv(DIG / f'page_{page}' / f'3217_page_{page}_table{table}.csv', dtype=str)
    raw.columns = ['wy'] + MONTHS + ['year_printed']
    rows = []
    for _, r in raw.iterrows():
        wy = pd.to_numeric(str(r.wy)[:4], errors='coerce')
        if pd.isna(wy):
            continue
        rec = {'water_year': int(wy), 'station': station, 'source': f'doc 3217 p{page} t{table}'}
        notes, flags = [], ''
        for m in MONTHS + ['year_printed']:
            v, f, n = clean(r[m])
            rec[m] = v
            flags += f
            if n and n != 'missing':
                notes.append(f'{m}: {n}')
        rec['flags'] = ''.join(sorted(set(flags)))
        rec['notes'] = '; '.join(notes)
        rows.append(rec)
    return pd.DataFrame(rows)


def days_in(wy, i):
    """Days in month i (0 = Oct of wy-1) of water year wy."""
    m = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9][i]
    return calendar.monthrange(wy - 1 if m >= 10 else wy, m)[1]


def monthly_check(df):
    out = []
    for _, r in df.iterrows():
        vals = r[MONTHS].astype(float).values
        w = np.array([days_in(r.water_year, i) for i in range(12)])
        ok = ~np.isnan(vals)
        mean_m = (vals[ok] * w[ok]).sum() / w[ok].sum() if ok.all() else np.nan
        out.append(mean_m)
    df = df.copy()
    df['year_from_months'] = out
    df['check_diff_pct'] = 100 * (df.year_printed - df.year_from_months) / df.year_from_months.replace(0, np.nan)
    return df


def nwis_wy(site):
    lines = [l.split('\t') for l in (OUT / f'nwis_dv_{site}.rdb').read_text().splitlines() if l.startswith('USGS\t')]
    d = pd.DataFrame({'date': pd.to_datetime([l[2] for l in lines]),
                      'q': pd.to_numeric([l[3] for l in lines], errors='coerce')})
    d['wy'] = d.date.dt.year + (d.date.dt.month >= 10)
    g = d.groupby('wy').agg(mean_cfs=('q', 'mean'), n_days=('q', 'count')).reset_index()
    g['complete'] = g.n_days >= 365
    return g


pre = monthly_check(read_table(225, 1, 'above Gibraltar Dam (pre-dam)'))
post = monthly_check(read_table(227, 1, 'below Gibraltar Dam'))
rest = pd.concat([pre, post], ignore_index=True)
# Yearly value used (checked 2026-11-14): disagreements between the printed yearly mean and its
# months are all lost decimal points. Months were wrong in 1923 (Feb 226 = 22.6), 1924 (Oct 130.0
# = 1.30), 1942 (Mar 445 = 44.5), 1945 (Mar 799 = 79.9): the printed yearly mean is right (1942 and
# 1945 also match NWIS). The yearly mean was wrong in 1934 ("134" for 13.4). Rule: use the printed
# yearly mean, unless it is a x10 decimal shift of the monthly mean (then use the monthly mean).
shift = (rest.year_printed / rest.year_from_months).between(9.5, 10.5) | \
        (rest.year_printed / rest.year_from_months).between(0.095, 0.105)
agree = rest.check_diff_pct.abs() <= 5
rest['annual_cfs'] = np.where(shift, rest.year_from_months, rest.year_printed)
rest['annual_basis'] = np.select([agree, shift, rest.year_from_months.notna()],
                                 ['printed (matches months)', 'from months (printed decimal lost)',
                                  'printed (a month has a lost decimal)'],
                                 'printed (months incomplete)')
n_up, n_dn = nwis_wy('11122000'), nwis_wy('11123000')
cmp = []
for nw, lo, hi in [(n_up, 1904, 1918), (n_dn, 1934, 1950)]:
    j = rest[rest.water_year.between(lo, hi)].merge(nw, left_on='water_year', right_on='wy')
    j = j[j.complete & j.annual_cfs.notna()]
    cmp.append(j[['water_year', 'station', 'annual_cfs', 'annual_basis', 'mean_cfs', 'n_days']])
cmp = pd.concat(cmp)
cmp['diff_pct'] = 100 * (cmp.annual_cfs - cmp.mean_cfs) / cmp.mean_cfs.replace(0, np.nan)
rest.to_csv(OUT / 'gibraltar_restored_monthly.csv', index=False)
cmp.to_csv(OUT / 'gibraltar_checks.csv', index=False)

print('== Check 1: printed yearly mean vs mean of printed months')
print(rest[['water_year', 'station', 'year_printed', 'year_from_months', 'check_diff_pct', 'flags', 'notes']]
      .round(2).to_string(index=False))
print('\n== Check 2: restored vs NWIS (complete water years)')
print(cmp.round(2).to_string(index=False))
print('\n|diff| restored vs NWIS: median %.1f%%, within 1%%: %d of %d, within 5%%: %d'
      % (cmp.diff_pct.abs().median(), (cmp.diff_pct.abs() <= 1).sum(), len(cmp), (cmp.diff_pct.abs() <= 5).sum()))

# ---------------------------------------------------------------- figure: annual mean discharge
series = pd.concat([
    n_up.assign(src='NWIS 11122000 (above dam)'),
    rest[rest.water_year.between(1920, 1933)].rename(columns={'water_year': 'wy', 'annual_cfs': 'mean_cfs'})
        [['wy', 'mean_cfs', 'annual_basis']].assign(src='Restored', complete=True),
    n_dn.assign(src='NWIS 11123000 (below dam)')], ignore_index=True)
series = series[series.mean_cfs.notna() & (series.wy <= 1979)]
series.to_csv(OUT / 'gibraltar_annual.csv', index=False)

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gibraltar_plot import plot_gibraltar  # noqa: E402

fig, ax = plt.subplots(figsize=(7.25, 3.2))
r = plot_gibraltar(ax)
fig.tight_layout()
fig.savefig(FIG / 'santa_ynez_gibraltar_annual.png', dpi=400)
fig.savefig(FIG / 'santa_ynez_gibraltar_annual.pdf')
print('saved figure;', r.wy.min(), '-', r.wy.max(), 'restored years:', len(r))
print(r[['wy', 'mean_cfs', 'annual_basis']].to_string(index=False))
