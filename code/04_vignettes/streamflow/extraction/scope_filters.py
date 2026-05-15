"""
scope_filters.py — estimate how many pages/tables Claude would have to analyze
for annual and monthly streamflow extraction under various filtering strategies.

Read-only scoping; writes nothing. Run from anywhere.
"""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
META = REPO / 'data' / 'metadata' / 'main_metadata.csv'
COUNTS = REPO / 'data' / 'analysis' / 'csv_row_counts.csv'

# CA bounding box (generous)
CA = dict(lat_lo=32.3, lat_hi=42.1, lon_lo=-124.6, lon_hi=-114.0)


def to_num(s):
    return pd.to_numeric(s, errors='coerce')


def main():
    m = pd.read_csv(META, dtype=str, low_memory=False)
    m = m[m.water_type.str.strip().str.lower() == 'stream discharge'].copy()

    res = m.temporal_resolution.fillna('').str.strip().str.lower()
    m['is_annual'] = res.str.contains('annual') | res.str.contains('year')
    m['is_monthly'] = res.str.contains('month')

    # coordinates: prefer actual, fall back to inferred
    lat = to_num(m.actual_latitude).fillna(to_num(m.inferred_latitude))
    lon = to_num(m.actual_longitude).fillna(to_num(m.inferred_longitude))
    m['has_latlon'] = lat.notna() & lon.notna()
    m['in_ca'] = (
        m['has_latlon']
        & lat.between(CA['lat_lo'], CA['lat_hi'])
        & lon.between(CA['lon_lo'], CA['lon_hi'])
    )

    # table counts per page from csv_row_counts
    c = pd.read_csv(COUNTS, dtype={'doc_id': str, 'page_number': str})
    tbl = c.groupby(['doc_id', 'page_number']).agg(
        n_tables=('table_number', 'nunique'),
        n_singlerow_tables=('number_rows', lambda x: (x == 1).sum()),
    ).reset_index()
    tbl_lookup = tbl.set_index(['doc_id', 'page_number'])

    # metadata rows per page
    mrows = m.groupby(['id', 'page_number']).size().rename('n_meta_rows')

    def page_table(df, label):
        # collapse to one row per (doc,page); a page counts as annual/monthly if ANY
        # metadata row for that page is so flagged
        g = df.groupby(['id', 'page_number']).agg(
            is_annual=('is_annual', 'any'),
            is_monthly=('is_monthly', 'any'),
            has_latlon=('has_latlon', 'any'),
            in_ca=('in_ca', 'any'),
        )
        g = g.join(mrows)
        g = g.join(tbl_lookup, on=['id', 'page_number'])
        g['has_csv'] = g['n_tables'].notna()
        g['n_tables'] = g['n_tables'].fillna(0).astype(int)
        g['n_singlerow_tables'] = g['n_singlerow_tables'].fillna(0).astype(int)
        g['one_meta_row'] = g['n_meta_rows'] == 1
        g['one_table'] = g['n_tables'] == 1
        g['meta_eq_tables'] = (g['n_meta_rows'] == g['n_tables']) & (g['n_tables'] > 0)

        print(f"\n{'='*70}\n{label}\n{'='*70}")
        for kind in ('annual', 'monthly'):
            sub = g[g[f'is_{kind}']]
            print(f"\n--- {kind.upper()}  (base: {len(sub):,} pages) ---")
            scenarios = [
                ('base (all candidate pages)', sub),
                ('+ has lat/lon in metadata', sub[sub.has_latlon]),
                ('+ in California (coords in CA bbox)', sub[sub.in_ca]),
                ('+ has CSV on disk', sub[sub.has_csv]),
                ('+ in CA + has CSV', sub[sub.in_ca & sub.has_csv]),
                ('4a: in CA + CSV + exactly 1 metadata row', sub[sub.in_ca & sub.has_csv & sub.one_meta_row]),
                ('   of which: 1 meta row AND 1 table (unambiguous)',
                 sub[sub.in_ca & sub.has_csv & sub.one_meta_row & sub.one_table]),
                ('   of which: 1 meta row but >1 table (ambiguous)',
                 sub[sub.in_ca & sub.has_csv & sub.one_meta_row & ~sub.one_table]),
                ('4b: in CA + CSV + #meta rows == #tables', sub[sub.in_ca & sub.has_csv & sub.meta_eq_tables]),
                ('4a OR 4b: in CA + CSV + (1 meta row OR meta==tables)',
                 sub[sub.in_ca & sub.has_csv & (sub.one_meta_row | sub.meta_eq_tables)]),
            ]
            print(f"  {'scenario':<52} {'pages':>8} {'tables':>8}")
            for name, s in scenarios:
                print(f"  {name:<52} {len(s):>8,} {int(s.n_tables.sum()):>8,}")

    page_table(m, "PAGES & TABLES BY FILTER")

    # overlap note
    both = m[(m.is_annual) & (m.is_monthly)]
    print(f"\nNote: {both[['id','page_number']].drop_duplicates().shape[0]:,} pages match BOTH "
          f"annual and monthly keywords (e.g. 'monthly and yearly').")


if __name__ == '__main__':
    main()
