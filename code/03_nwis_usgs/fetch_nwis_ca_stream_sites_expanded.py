"""
fetch_nwis_ca_stream_sites_expanded.py
=============================================================================
Extends the daily-values (dv) CA stream site inventory to include sites with
peak-flow records (pk) and site-visit data (sv). Together these three data
types capture any NWIS site that has at least one streamflow-related
observation — a closer analog to the restored historical data than dv-only.

DATA TYPES FETCHED
------------------
  dv  — continuous daily values (already in ca_stream_sites.csv)
  pk  — annual peak-flow records (one value per flood year)

NOTE: 'sv' (site-visit) was tested but those sites carry only water quality
parameter codes (qw) — zero discharge (parm_cd=00060) observations. They are
stream locations but not streamflow measurement sites, so sv is excluded.
'measurements' returns a 400 error from this endpoint and is also excluded.

DESIGN
------
- Each data type is fetched county-by-county into its own scratch subfolder
  so runs are resumable and the dv scratch folder is untouched.
- The existing ca_stream_sites_by_county/ (dv) CSVs are reused as-is.
- All three sets are combined and deduplicated to one row per site_no, keeping
  the series row with the earliest begin_date (preferring parm_cd=00060 rows).
- A `data_types` column lists all data types observed at each site.

OUTPUTS (separate from the dv-only files — originals are NOT overwritten)
-------
  data/analysis/nwis_sites/ca_stream_sites_expanded_by_county/pk/
  data/analysis/nwis_sites/ca_stream_sites_expanded_raw.csv
  data/analysis/nwis_sites/ca_stream_sites_expanded.csv
  data/analysis/nwis_sites/ca_stream_sites_expanded.parquet

Resume: re-running skips counties already cached in each scratch subfolder.
=============================================================================
"""

import io
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO        = Path(__file__).resolve().parents[2]
DV_SCRATCH  = REPO / 'data/analysis/nwis_sites/ca_stream_sites_by_county'
EXP_SCRATCH = REPO / 'data/analysis/nwis_sites/ca_stream_sites_expanded_by_county'
OUT_RAW     = REPO / 'data/analysis/nwis_sites/ca_stream_sites_expanded_raw.csv'
OUT_CSV     = REPO / 'data/analysis/nwis_sites/ca_stream_sites_expanded.csv'
OUT_PARQ    = REPO / 'data/analysis/nwis_sites/ca_stream_sites_expanded.parquet'

for dtype in ('pk',):
    (EXP_SCRATCH / dtype).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NWIS base URL
# ---------------------------------------------------------------------------
NWIS_URL = 'https://waterservices.usgs.gov/nwis/site/'

BASE_PARAMS = {
    'format':              'rdb',
    'siteType':            'ST',
    'seriesCatalogOutput': 'true',
    'siteStatus':          'all',
}

# ---------------------------------------------------------------------------
# California county FIPS codes
# ---------------------------------------------------------------------------
CA_COUNTIES = {
    'Alameda':         '06001',
    'Alpine':          '06003',
    'Amador':          '06005',
    'Butte':           '06007',
    'Calaveras':       '06009',
    'Colusa':          '06011',
    'Contra Costa':    '06013',
    'Del Norte':       '06015',
    'El Dorado':       '06017',
    'Fresno':          '06019',
    'Glenn':           '06021',
    'Humboldt':        '06023',
    'Imperial':        '06025',
    'Inyo':            '06027',
    'Kern':            '06029',
    'Kings':           '06031',
    'Lake':            '06033',
    'Lassen':          '06035',
    'Los Angeles':     '06037',
    'Madera':          '06039',
    'Marin':           '06041',
    'Mariposa':        '06043',
    'Mendocino':       '06045',
    'Merced':          '06047',
    'Modoc':           '06049',
    'Mono':            '06051',
    'Monterey':        '06053',
    'Napa':            '06055',
    'Nevada':          '06057',
    'Orange':          '06059',
    'Placer':          '06061',
    'Plumas':          '06063',
    'Riverside':       '06065',
    'Sacramento':      '06067',
    'San Benito':      '06069',
    'San Bernardino':  '06071',
    'San Diego':       '06073',
    'San Francisco':   '06075',
    'San Joaquin':     '06077',
    'San Luis Obispo': '06079',
    'San Mateo':       '06081',
    'Santa Barbara':   '06083',
    'Santa Clara':     '06085',
    'Santa Cruz':      '06087',
    'Shasta':          '06089',
    'Sierra':          '06091',
    'Siskiyou':        '06093',
    'Solano':          '06095',
    'Sonoma':          '06097',
    'Stanislaus':      '06099',
    'Sutter':          '06101',
    'Tehama':          '06103',
    'Trinity':         '06105',
    'Tulare':          '06107',
    'Tuolumne':        '06109',
    'Ventura':         '06111',
    'Yolo':            '06113',
    'Yuba':            '06115',
}


def parse_rdb(text: str, county_name: str, fips: str) -> pd.DataFrame | None:
    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith('#')]
    if len(data_lines) < 3:
        return None
    header = data_lines[0].split('\t')
    data   = [l.split('\t') for l in data_lines[2:] if l.strip()]
    if not data:
        return None
    df = pd.DataFrame(data, columns=header)
    df['county_name'] = county_name
    df['county_fips'] = fips
    return df


def fetch_county(county_name: str, fips: str, data_type_cd: str,
                 retries: int = 3, backoff: float = 5.0) -> pd.DataFrame | None:
    params = {**BASE_PARAMS, 'countyCd': fips, 'hasDataTypeCd': data_type_cd}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NWIS_URL, params=params, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return parse_rdb(resp.text, county_name, fips)
        except Exception as e:
            if attempt < retries:
                wait = backoff * attempt
                print(f'    attempt {attempt} failed ({e}) — retrying in {wait:.0f}s')
                time.sleep(wait)
            else:
                raise


def fetch_all_counties(data_type_cd: str) -> list[pd.DataFrame]:
    """Fetch all counties for one data type; returns list of DataFrames."""
    scratch = EXP_SCRATCH / data_type_cd
    results = []
    skipped = fetched = 0

    print(f'\n--- Fetching hasDataTypeCd={data_type_cd} ---')
    for i, (county, fips) in enumerate(CA_COUNTIES.items(), 1):
        out_path = scratch / f'{fips}_{county.replace(" ", "_")}.csv'

        if out_path.exists():
            try:
                df = pd.read_csv(out_path, dtype=str)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                results.append(df)
            skipped += 1
            print(f'  [{i:2d}/58] {county:20s} — cached  ({len(df):3d} rows)')
            continue

        try:
            df = fetch_county(county, fips, data_type_cd)
            if df is None or df.empty:
                print(f'  [{i:2d}/58] {county:20s} — no sites')
                pd.DataFrame().to_csv(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
                results.append(df)
                fetched += 1
                print(f'  [{i:2d}/58] {county:20s} — fetched ({len(df):3d} rows)')
        except Exception as e:
            print(f'  [{i:2d}/58] {county:20s} — FAILED: {e}')

        time.sleep(0.5)

    print(f'  Done: {fetched} fetched, {skipped} from cache')
    return results


def main():
    print('Building expanded CA NWIS stream site inventory')
    print('  Data types: dv (existing) + pk')

    # -----------------------------------------------------------------------
    # Load existing dv data
    # -----------------------------------------------------------------------
    print(f'\n--- Loading existing dv data from cache ---')
    dv_results = []
    for path in sorted(DV_SCRATCH.glob('*.csv')):
        df = pd.read_csv(path, dtype=str)
        if not df.empty:
            dv_results.append(df)
    dv_all = pd.concat(dv_results, ignore_index=True) if dv_results else pd.DataFrame()
    print(f'  Loaded {len(dv_all):,} dv rows from {len(dv_results)} county files')

    # -----------------------------------------------------------------------
    # Fetch pk and measurements
    # -----------------------------------------------------------------------
    pk_results = fetch_all_counties('pk')

    pk_all = pd.concat(pk_results, ignore_index=True) if pk_results else pd.DataFrame()
    print(f'\npk rows: {len(pk_all):,}')

    # -----------------------------------------------------------------------
    # Combine dv + pk and save raw
    # -----------------------------------------------------------------------
    all_rows = pd.concat([dv_all, pk_all], ignore_index=True)
    print(f'\nTotal combined rows (pre-dedup): {len(all_rows):,}')
    all_rows.to_csv(OUT_RAW, index=False)
    print(f'Saved raw: {OUT_RAW.relative_to(REPO)}')

    # -----------------------------------------------------------------------
    # Deduplicate to one row per site_no
    # Preference order: parm_cd=00060 row with earliest begin_date;
    # fallback to any row with earliest begin_date.
    # Also record all data_types seen per site.
    # -----------------------------------------------------------------------
    df = all_rows.copy()

    for col in ('begin_date', 'end_date'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ('dec_lat_va', 'dec_long_va', 'drain_area_va', 'count_nu'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Collect all data_types per site
    if 'data_type_cd' in df.columns:
        data_type_map = (
            df.groupby('site_no')['data_type_cd']
            .apply(lambda x: ','.join(sorted(set(x.dropna().str.strip()))))
            .rename('data_types')
        )
    else:
        data_type_map = pd.Series(dtype=str)

    # Primary: parm_cd=00060 rows, earliest begin_date per site
    is_q = df.get('parm_cd', pd.Series(dtype=str)).str.strip() == '00060'
    primary = (df[is_q]
               .sort_values(['site_no', 'begin_date'])
               .drop_duplicates('site_no', keep='first'))

    # Fallback: any row for sites not in primary
    fallback = (df[~df['site_no'].isin(primary['site_no'])]
                .sort_values(['site_no', 'begin_date'])
                .drop_duplicates('site_no', keep='first'))

    sites = pd.concat([primary, fallback], ignore_index=True).sort_values('site_no').reset_index(drop=True)

    # Attach data_types summary column
    if len(data_type_map):
        sites = sites.merge(data_type_map, on='site_no', how='left')

    # Derive year columns
    for col, yr_col in [('begin_date', 'begin_year'), ('end_date', 'end_year')]:
        if col in sites.columns:
            sites[yr_col] = sites[col].dt.year.astype('Int64')

    print(f'\n=== Summary ===')
    print(f'Unique sites (expanded):  {len(sites):,}')
    if 'begin_year' in sites.columns:
        print(f'Earliest begin year:      {sites["begin_year"].min()}')
        print(f'Sites beginning <=1920:   {(sites["begin_year"] <= 1920).sum():,}')
        print(f'Sites beginning <=1945:   {(sites["begin_year"] <= 1945).sum():,}')
        print(f'Sites beginning <=1979:   {(sites["begin_year"] <= 1979).sum():,}')
    if 'data_types' in sites.columns:
        print(f'\nData type combinations:')
        print(sites['data_types'].value_counts().head(15).to_string())

    sites.to_csv(OUT_CSV, index=False)
    sites.to_parquet(OUT_PARQ, index=False)
    print(f'\nSaved:')
    print(f'  {OUT_CSV.relative_to(REPO)}')
    print(f'  {OUT_PARQ.relative_to(REPO)}')


if __name__ == '__main__':
    main()
