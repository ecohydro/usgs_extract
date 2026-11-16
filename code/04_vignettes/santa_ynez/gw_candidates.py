"""
Santa Ynez groundwater, step 4 of the agreed plan (2026-11-14): pick candidate wells for Figure 5 panel b.

Restored wells in the Santa Ynez River Valley basin (B118 3-015) linked to an NWIS well by state well number
(gw_basin_links.csv, from gw_basin_recount.py). For each pair, compare YEARS (not decades):
  restored years  = years printed in the rows of the record's tables (table_row_dates) only
  NWIS years      = years with an NWIS water-level field measurement (nwis_years_gw.csv, <= 1979)
A good candidate has restored-only years (the story) AND overlap years with NWIS (to check the digitized
values against NWIS) AND a long NWIS record.

Output: data/analysis/santa_ynez/gw_candidates.csv (supersedes the Nov 15 gw_candidates_doc2060.csv, which
was made before the well-owner rule).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'code/02_inventory'))
from site_activity import load_records, CAP  # noqa: E402

A = ROOT / 'data/analysis'
OUT = A / 'santa_ynez'

links = pd.read_csv(OUT / 'gw_basin_links.csv', dtype={'nwis_link': str})
rec = load_records()
rec = rec[(rec.water_type == 'Groundwater') & rec.site_B1.isin(links.site_B1)].copy()


def years(r):
    """Years printed in the rows of the record's tables only. Records without printed years are left out of
    the ranking: their year_start..year_end often comes from a description ("records available 1930-52")
    while the table holds one year (checked 51920 p56, 2026-11-16)."""
    y0, y1 = int(r.year_start), int(min(r.year_end, CAP))
    if isinstance(r.printed_years, set):
        return {y for y in r.printed_years if y0 <= y <= y1}
    return set()


rec['yrs'] = [years(r) for r in rec.itertuples(index=False)]
rec['src'] = rec.doc_id.astype(str) + ' p' + rec.page_number.astype(int).astype(str)
per = rec.groupby('site_B1').agg(yrs=('yrs', lambda x: set().union(*x)),
                                  pages=('src', lambda x: ' '.join(sorted(set(x)))),
                                  docs=('doc_id', lambda x: ' '.join(sorted({str(v) for v in x}))))

ny = pd.read_csv(A / 'nwis_sites/nwis_years_gw.csv', dtype={'site_no': str})
ny = ny[ny.year <= CAP]
nyrs = ny.groupby('site_no').year.apply(set)
nmeas = ny.groupby('site_no').n_measurements.sum()


def span(s):
    return f'{min(s)}-{max(s)} ({len(s)})' if s else ''


rows = []
for l in links.itertuples(index=False):
    r = per.yrs.get(l.site_B1, set())
    n = nyrs.get(l.nwis_link, set())
    only, both = r - n, r & n
    if not r:
        continue
    rows.append({'site_B1': l.site_B1, 'nwis_site': l.nwis_link, 'docs': per.docs.get(l.site_B1, ''),
                 'restored_years': span(r), 'nwis_years': span(n), 'nwis_meas': int(nmeas.get(l.nwis_link, 0)),
                 'restored_only_years': span(only), 'n_only': len(only), 'n_overlap': len(both),
                 'only_before_nwis': sum(y < min(n) for y in only) if n else len(only),
                 'pages': per.pages.get(l.site_B1, '')})
c = pd.DataFrame(rows)
c['score'] = c.n_only.clip(upper=15) + c.n_overlap.clip(upper=5) * 2 + (c.nwis_meas.clip(upper=100) / 10)
c = c.sort_values('score', ascending=False)
c.to_csv(OUT / 'gw_candidates.csv', index=False)
print(f'linked restored wells: {len(c)}; with restored-only years: {(c.n_only > 0).sum()}; '
      f'with overlap years: {(c.n_overlap > 0).sum()}; both: {((c.n_only > 0) & (c.n_overlap > 0)).sum()}')
pd.set_option('display.width', 250)
print(c.drop(columns=['pages', 'score']).head(20).to_string(index=False))
