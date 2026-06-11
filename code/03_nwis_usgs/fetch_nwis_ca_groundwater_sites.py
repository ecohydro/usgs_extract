"""
fetch_nwis_ca_groundwater_sites.py
=============================================================================
Downloads all USGS NWIS groundwater monitoring sites for California,
county by county, with full period-of-record information.

Analogous to fetch_nwis_ca_stream_sites.py but for groundwater wells.

QUERY PARAMETERS
----------------
  siteType=GW              groundwater wells only
  hasDataTypeCd=gw         sites with groundwater level data
  seriesCatalogOutput=true adds begin_date / end_date / count_nu per series
  siteStatus=all           active AND discontinued sites
  format=rdb               tab-delimited RDB format

WHY county-by-county and direct HTTP: same reasoning as the streamflow
script — state-level queries with seriesCatalogOutput can silently truncate,
and the dataretrieval package does not expose seriesCatalogOutput.

DEDUPLICATION
-------------
One site has multiple series rows (different parameters, data types).
Deduplicated to one row per site_no using the earliest begin_date, preferring
parm_cd=72019 (depth to water level below land surface — the most common GW
level parameter) with fallback to any row.

OUTPUTS
-------
  data/analysis/nwis_sites/ca_groundwater_sites_by_county/  one CSV per county
  data/analysis/nwis_sites/ca_groundwater_sites_raw.csv     all rows, pre-dedup
  data/analysis/nwis_sites/ca_groundwater_sites.csv         one row per site
  data/analysis/nwis_sites/ca_groundwater_sites.parquet     same, parquet

Resume: re-running skips counties whose CSV already exists in the scratch
folder. Delete a county CSV to force re-fetch.
=============================================================================
"""

import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = Path(__file__).resolve().parents[2]
SCRATCH  = REPO / 'data/analysis/nwis_sites/ca_groundwater_sites_by_county'
OUT_RAW  = REPO / 'data/analysis/nwis_sites/ca_groundwater_sites_raw.csv'
OUT_CSV  = REPO / 'data/analysis/nwis_sites/ca_groundwater_sites.csv'
OUT_PARQ = REPO / 'data/analysis/nwis_sites/ca_groundwater_sites.parquet'

SCRATCH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NWIS base URL and parameters
# ---------------------------------------------------------------------------
NWIS_URL = 'https://waterservices.usgs.gov/nwis/site/'

BASE_PARAMS = {
    'format':              'rdb',
    'siteType':            'GW',
    'hasDataTypeCd':       'gw',       # sites with groundwater level data
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


def fetch_county(county_name: str, fips: str,
                 retries: int = 3, backoff: float = 5.0) -> pd.DataFrame | None:
    params = {**BASE_PARAMS, 'countyCd': fips}
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


def main():
    print(f'Fetching NWIS groundwater sites for all {len(CA_COUNTIES)} CA counties')
    print(f'  siteType=GW  hasDataTypeCd=gw  siteStatus=all')
    print(f'  Scratch: {SCRATCH.relative_to(REPO)}')
    print()

    results = []
    skipped = fetched = 0
    failed  = []

    for i, (county, fips) in enumerate(CA_COUNTIES.items(), 1):
        out_path = SCRATCH / f'{fips}_{county.replace(" ", "_")}.csv'

        if out_path.exists():
            try:
                df = pd.read_csv(out_path, dtype=str)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                results.append(df)
            skipped += 1
            print(f'  [{i:2d}/58] {county:20s} — cached  ({len(df):4d} rows)')
            continue

        try:
            df = fetch_county(county, fips)
            if df is None or df.empty:
                print(f'  [{i:2d}/58] {county:20s} — no sites')
                pd.DataFrame().to_csv(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
                results.append(df)
                fetched += 1
                print(f'  [{i:2d}/58] {county:20s} — fetched ({len(df):4d} rows)')
        except Exception as e:
            failed.append((county, fips, str(e)))
            print(f'  [{i:2d}/58] {county:20s} — FAILED: {e}')

        time.sleep(0.5)

    print()
    print(f'Done: {fetched} fetched, {skipped} from cache, {len(failed)} failed')
    if failed:
        print('\nFailed counties:')
        for county, fips, err in failed:
            print(f'  {county} ({fips}): {err}')

    if not results:
        print('No data — exiting.')
        return

    # -----------------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------------
    print('\nMerging...')
    all_rows = pd.concat(results, ignore_index=True)
    print(f'  Total rows (all series, pre-dedup): {len(all_rows):,}')

    all_rows.to_csv(OUT_RAW, index=False)
    print(f'  Saved raw: {OUT_RAW.relative_to(REPO)}')

    # -----------------------------------------------------------------------
    # Deduplicate to one row per site_no
    # Prefer parm_cd=72019 (depth to water level) with earliest begin_date.
    # Fallback to any row with earliest begin_date.
    # -----------------------------------------------------------------------
    df = all_rows.copy()

    for col in ('begin_date', 'end_date'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ('dec_lat_va', 'dec_long_va', 'well_depth_va', 'hole_depth_va', 'count_nu'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Primary: parm_cd=72019 (depth to water level below land surface)
    is_wl = df.get('parm_cd', pd.Series(dtype=str)).str.strip() == '72019'
    primary = (df[is_wl]
               .sort_values(['site_no', 'begin_date'])
               .drop_duplicates('site_no', keep='first'))

    # Fallback: any row for sites not in primary
    fallback = (df[~df['site_no'].isin(primary['site_no'])]
                .sort_values(['site_no', 'begin_date'])
                .drop_duplicates('site_no', keep='first'))

    sites = (pd.concat([primary, fallback], ignore_index=True)
             .sort_values('site_no')
             .reset_index(drop=True))

    print(f'  parm_cd=72019 (depth to water level) sites: {len(primary):,}')
    print(f'  Fallback (other parm_cd):                   {len(fallback):,}')
    print(f'  Total unique sites:                         {len(sites):,}')

    # Derive year columns
    for col, yr_col in [('begin_date', 'begin_year'), ('end_date', 'end_year')]:
        if col in sites.columns:
            sites[yr_col] = sites[col].dt.year.astype('Int64')

    sites.to_csv(OUT_CSV, index=False)
    sites.to_parquet(OUT_PARQ, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f'\n=== Summary ===')
    print(f'Unique GW sites:        {len(sites):,}')
    if 'begin_year' in sites.columns:
        has_yr = sites['begin_year'].notna()
        print(f'With begin year:        {has_yr.sum():,}')
        print(f'Earliest begin year:    {sites["begin_year"].min()}')
        print(f'Sites beginning <=1920: {(sites["begin_year"] <= 1920).sum():,}')
        print(f'Sites beginning <=1945: {(sites["begin_year"] <= 1945).sum():,}')
        print(f'Sites beginning <=1979: {(sites["begin_year"] <= 1979).sum():,}')
        print(f'Still active (>=2020):  {(sites["end_year"] >= 2020).sum():,}')
    if 'county_name' in sites.columns:
        print(f'\nTop 10 counties by site count:')
        print(sites['county_name'].value_counts().head(10).to_string())

    print(f'\nSaved:')
    print(f'  {OUT_CSV.relative_to(REPO)}')
    print(f'  {OUT_PARQ.relative_to(REPO)}')
    print(f'  {OUT_RAW.relative_to(REPO)}  (all series rows, pre-dedup)')


if __name__ == '__main__':
    main()
