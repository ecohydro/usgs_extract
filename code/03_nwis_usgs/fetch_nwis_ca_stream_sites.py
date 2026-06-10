"""
fetch_nwis_ca_stream_sites.py
=============================================================================
Downloads all USGS NWIS stream discharge gauge sites for California,
county by county, with full period-of-record information.

WHY COUNTY-BY-COUNTY
--------------------
A single state-level NWIS query works for site metadata but can silently
truncate when combined with seriesCatalogOutput=true (which adds the
begin_date / end_date / count_nu columns).  Querying each of California's
58 counties individually keeps each request small (~10–200 sites), lets
the script resume from where it left off if interrupted, and produces
per-county intermediate files for inspection.

WHY DIRECT HTTP (not dataretrieval)
------------------------------------
The dataretrieval package's what_sites() wrapper does not expose the
seriesCatalogOutput=true parameter, which is the NWIS RDB flag that adds
period-of-record columns (begin_date, end_date, count_nu, data_type_cd,
parm_cd). We call the NWIS site service URL directly to get those columns.

QUERY PARAMETERS
----------------
  siteType=ST              stream gauging stations only
  hasDataTypeCd=dv         only sites with daily values data
  seriesCatalogOutput=true adds one row per data series per site, with
                           begin_date, end_date, count_nu
  siteStatus=all           active AND discontinued stations
  format=rdb               tab-delimited text with RDB header rows

  Note: parameterCd=00060 (discharge) is NOT set in the URL — setting it
  causes seriesCatalogOutput to dump every parameter series for each site,
  inflating raw rows ~50x. Instead, the merge step filters to parm_cd=00060
  (daily discharge) when deduplicating to one row per site.

One site has multiple raw rows (daily values, peak flow, gage height, etc.).
The final merged file is deduplicated to one row per site_no using the
daily-discharge row (data_type_cd=dv, parm_cd=00060).

OUTPUTS
-------
  data/analysis/nwis_sites/ca_stream_sites_by_county/    one CSV per county
  data/analysis/nwis_sites/ca_stream_sites_raw.csv       all rows, pre-dedup
  data/analysis/nwis_sites/ca_stream_sites.csv           one row per site
  data/analysis/nwis_sites/ca_stream_sites.parquet       same, parquet format

USAGE
-----
  python code/03_nwis_usgs/fetch_nwis_ca_stream_sites.py

Resume: re-running skips counties whose CSV already exists in the scratch
folder. To re-fetch a county, delete its CSV from ca_stream_sites_by_county/.
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
REPO     = Path(__file__).resolve().parents[2]
SCRATCH  = REPO / 'data' / 'analysis' / 'nwis_sites' / 'ca_stream_sites_by_county'
OUT_RAW  = REPO / 'data' / 'analysis' / 'nwis_sites' / 'ca_stream_sites_raw.csv'
OUT_CSV  = REPO / 'data' / 'analysis' / 'nwis_sites' / 'ca_stream_sites.csv'
OUT_PARQ = REPO / 'data' / 'analysis' / 'nwis_sites' / 'ca_stream_sites.parquet'

SCRATCH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NWIS base URL and parameters
# ---------------------------------------------------------------------------
NWIS_URL = 'https://waterservices.usgs.gov/nwis/site/'

BASE_PARAMS = {
    'format':               'rdb',
    'siteType':             'ST',
    'hasDataTypeCd':        'dv',      # sites with daily-values data
    'seriesCatalogOutput':  'true',    # adds begin_date / end_date / count_nu per series
    'siteStatus':           'all',     # active + discontinued
    # NOTE: parameterCd=00060 is intentionally NOT set here. Setting it in the URL
    # causes seriesCatalogOutput to dump all parameter series (including water quality)
    # for every matched site, inflating the row count ~50x. Instead we filter to
    # parm_cd=00060 (discharge) after download in the merge step.
}

# ---------------------------------------------------------------------------
# California county FIPS codes (state 06 + 3-digit county code)
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
    """
    Parse NWIS RDB-format response text into a DataFrame.

    RDB format has:
      - Comment lines starting with '#'
      - A header line (column names, tab-separated)
      - A type-indicator line (e.g. '5s\t15s\t...' — column widths, not data)
      - Data lines

    Returns None if there are no data rows.
    """
    lines = text.splitlines()
    data_lines = [l for l in lines if not l.startswith('#')]

    if len(data_lines) < 3:   # header + type row + at least one data row
        return None

    header = data_lines[0].split('\t')
    # data_lines[1] is the RDB type-indicator row — skip it
    data   = [l.split('\t') for l in data_lines[2:] if l.strip()]

    if not data:
        return None

    df = pd.DataFrame(data, columns=header)
    df['county_name'] = county_name
    df['county_fips'] = fips
    return df


def fetch_county(county_name: str, fips: str,
                 retries: int = 3, backoff: float = 5.0) -> pd.DataFrame | None:
    """Query NWIS for one county; retry on transient errors."""
    params = {**BASE_PARAMS, 'countyCd': fips}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(NWIS_URL, params=params, timeout=60)
            if resp.status_code == 404:
                return None   # NWIS returns 404 when no sites match (e.g. SF county)
            resp.raise_for_status()
            return parse_rdb(resp.text, county_name, fips)
        except Exception as e:
            if attempt < retries:
                wait = backoff * attempt
                print(f'    attempt {attempt} failed ({e}) — retrying in {wait:.0f}s')
                time.sleep(wait)
            else:
                raise


def main():
    print(f'Fetching NWIS stream discharge sites for all {len(CA_COUNTIES)} CA counties')
    print(f'  siteType=ST  hasDataTypeCd=dv  siteStatus=all  (discharge filtered post-download)')
    print(f'  Scratch: {SCRATCH.relative_to(REPO)}')
    print()

    results = []
    skipped = fetched = 0
    failed  = []

    for i, (county, fips) in enumerate(CA_COUNTIES.items(), 1):
        out_path = SCRATCH / f'{fips}_{county.replace(" ", "_")}.csv'

        if out_path.exists():
            df = pd.read_csv(out_path, dtype=str)
            results.append(df)
            skipped += 1
            print(f'  [{i:2d}/58] {county:20s} — cached  ({len(df):3d} rows)')
            continue

        try:
            df = fetch_county(county, fips)
            if df is None or df.empty:
                print(f'  [{i:2d}/58] {county:20s} — no sites')
                # Write empty file so re-runs skip this county
                pd.DataFrame().to_csv(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
                results.append(df)
                fetched += 1
                print(f'  [{i:2d}/58] {county:20s} — fetched ({len(df):3d} rows)')
        except Exception as e:
            failed.append((county, fips, str(e)))
            print(f'  [{i:2d}/58] {county:20s} — FAILED: {e}')

        time.sleep(0.5)   # be polite to the NWIS API

    print()
    print(f'Done: {fetched} fetched, {skipped} loaded from cache, {len(failed)} failed')
    if failed:
        print('\nFailed counties (delete their CSV if exists and re-run to retry):')
        for county, fips, err in failed:
            print(f'  {county} ({fips}): {err}')

    if not results:
        print('No data to merge — exiting.')
        return

    # -----------------------------------------------------------------------
    # Merge all counties
    # -----------------------------------------------------------------------
    print('\nMerging...')
    all_rows = pd.concat(results, ignore_index=True)
    print(f'  Total rows (all series, pre-dedup): {len(all_rows):,}')

    # Save raw (all series rows)
    all_rows.to_csv(OUT_RAW, index=False)
    print(f'  Saved raw: {OUT_RAW.relative_to(REPO)}')

    # -----------------------------------------------------------------------
    # Deduplicate to one row per site_no
    # Filter to daily discharge (data_type_cd=dv, parm_cd=00060) before
    # deduplication so begin_date / end_date reflect the discharge record.
    # Sites with multiple dv/00060 series get the row with the earliest begin.
    # Sites with no dv/00060 series (rare for ST sites) fall back to any dv row.
    # -----------------------------------------------------------------------
    if 'site_no' not in all_rows.columns:
        print('Warning: no site_no column — skipping dedup; saving raw only.')
        return

    df = all_rows.copy()

    # Parse dates
    for col in ('begin_date', 'end_date'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Primary set: daily discharge rows only
    is_dv_q = (
        (df.get('data_type_cd', pd.Series(dtype=str)).str.strip() == 'dv') &
        (df.get('parm_cd', pd.Series(dtype=str)).str.strip() == '00060')
    )
    dv_q = df[is_dv_q].sort_values(['site_no', 'begin_date']).drop_duplicates('site_no', keep='first')

    # Fallback: any dv row for sites missing a 00060 series
    any_dv = (df.get('data_type_cd', pd.Series(dtype=str)).str.strip() == 'dv')
    fallback = (df[any_dv & ~df['site_no'].isin(dv_q['site_no'])]
                .sort_values(['site_no', 'begin_date'])
                .drop_duplicates('site_no', keep='first'))

    sites = pd.concat([dv_q, fallback], ignore_index=True).sort_values('site_no').reset_index(drop=True)
    print(f'  Discharge (dv/00060) sites:   {len(dv_q):,}')
    print(f'  Fallback (dv only, no 00060): {len(fallback):,}')
    print(f'  Total unique sites:           {len(sites):,}')

    # Derive year columns
    for col, yr_col in [('begin_date', 'begin_year'), ('end_date', 'end_year')]:
        if col in sites.columns:
            sites[yr_col] = sites[col].dt.year.astype('Int64')

    # Fix numeric fields that come in as strings
    for col in ('dec_lat_va', 'dec_long_va', 'drain_area_va', 'count_nu'):
        if col in sites.columns:
            sites[col] = pd.to_numeric(sites[col], errors='coerce')

    sites.to_csv(OUT_CSV, index=False)
    sites.to_parquet(OUT_PARQ, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f'\n=== Summary ===')
    print(f'Unique sites:           {len(sites):,}')
    if 'begin_year' in sites.columns:
        has_yr = sites['begin_year'].notna()
        print(f'With begin year:        {has_yr.sum():,}')
        print(f'Earliest begin year:    {sites["begin_year"].min()}')
        print(f'Sites beginning <=1920: {(sites["begin_year"] <= 1920).sum():,}')
        print(f'Sites beginning <=1945: {(sites["begin_year"] <= 1945).sum():,}')
    if 'county_name' in sites.columns:
        print(f'\nTop 15 counties by site count:')
        print(sites['county_name'].value_counts().head(15).to_string())

    print(f'\nSaved:')
    print(f'  {OUT_CSV.relative_to(REPO)}')
    print(f'  {OUT_PARQ.relative_to(REPO)}')
    print(f'  {OUT_RAW.relative_to(REPO)}  (all series rows, pre-dedup)')


if __name__ == '__main__':
    main()
