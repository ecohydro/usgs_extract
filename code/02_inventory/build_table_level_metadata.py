"""
build_table_level_metadata.py
=============================================================================
Produces data/analysis/table_level_metadata.csv — a table-level join of
measurement counts, header-derived water type, and page-level LLM metadata.

PURPOSE
-------
The main LLM metadata file (data/metadata/main_metadata.csv) is not strictly
one row per table.  The LLM (Phi-4) was run per page and sometimes returned
multiple rows for the same page when it detected distinct water sources.
This means the LLM water type cannot be directly joined to individual table
CSVs without double-counting measurements.

This script assigns a water type to each of the 105,920 individual table CSVs
using a two-stage hybrid approach:

    Stage 1 — LLM classification (primary, trusted)
        For pages where the LLM returned exactly one distinct water type, that
        classification is used directly.  The LLM (Phi-4) read the full page
        context — surrounding text, table titles, document section headers —
        and is more reliable than column headers alone for determining the
        primary purpose of a table.  This covers ~81% of pages.

    Stage 2 — Header keyword classifier (resolves LLM ambiguity)
        For the ~19% of pages where the LLM returned more than one distinct
        water type (e.g. one table classified as Stream Discharge and another
        as Reservoir on the same page), the header keyword classifier is used
        to assign each individual table's water type from its column headers.
        If the header classifier is also uncertain, the first LLM type for
        that page is used as a tiebreaker.

    Stage 3 — Header classifier only (no LLM data)
        A small number of tables (~3%) have no LLM metadata at all (the LLM
        was not run on those pages).  For these, a high-confidence header
        keyword match is used if available; otherwise the table is marked
        'uncertain'.

        Assignment source values tracked in output:
          'llm_unambiguous' — Stage 1: single LLM type, used directly
          'header_high'     — Stage 2/3: header resolved ambiguity or filled gap
          'llm_ambiguous'   — Stage 2 tiebreaker: LLM ambiguous, header also
                              uncertain; first LLM type used (flagged)
          'no_data'         — Stage 3: no LLM and no high-confidence header

OUTPUTS
-------
    data/analysis/table_headers.csv         intermediate: raw headers + Stage 1
    data/analysis/table_level_metadata.csv  final: one row per table CSV

COLUMNS IN table_level_metadata.csv
------------------------------------
    doc_id              document ID (matches id column in main_metadata.csv)
    page_number         page number within the document
    table_number        table index on the page (1-based)
    number_rows         estimated measurement count (CSV line count minus 1)
    header_text         first 500 chars of joined header cell values
    water_type_header   Stage 1 classification (or 'uncertain')
    confidence          'high' | 'low' (month signal only) | 'uncertain'
    water_type_llm      LLM water type from main_metadata.csv (page level)
    llm_page_ambiguous  True if the page had >1 distinct LLM water type
    water_type_final    Final assigned water type (used in all figures)
    assignment_source   Which stage/sub-case assigned water_type_final
    year_start          TABLE-LEVEL start year (table_dates.csv, 14_table_dates.py; 2026-11-04)
    year_end            TABLE-LEVEL end year
    decade_start        floor(year_start / 10) * 10
    date_status         evidence tier of the date (date_parsing.TIER_STATUS): table_date_column,
                        table_text, table_caption, page_text; fallbacks period_of_record,
                        construction_year; undated. DATED_STATUSES = dated
    date_tier           same, as the raw tier code (A_date_column ... F_construction_year)
    (table title: see table_dates.csv `caption`; left out here to keep the file small)
    por_start/por_end   station period of record printed on the page (not the data dates)
    construction_start/construction_end   years wells were drilled / completed
    year_start_llm, year_end_llm, decade_start_llm, date_status_llm
                        earlier page-level dates from the LLM string (first entry of the page)
    lat_combined        REBUILT location of the table (see build_table_locations);
    lon_combined          NaN when no entry on the page has a rebuilt location
    coord_method        how the location was found (printed, plss_section, nwis_station, ...)
    precision_km        approximate location uncertainty
    precision_class     precise | approximate | basin | unlocated
    evidence_scope      page | document | continuation (where the location evidence came from)
    location_match      entry  -- location from an entry matched to THIS table
                        page   -- no entry matched to this table; best location on the page
                        none   -- no located entry on the page
    n_site_locations    distinct located sites among the candidate entries (>1 = multi-site table)
    site_spread_km      max distance from the chosen point to the other candidates of the same class
    location_ambiguous  site_spread_km > 5: table covers several sites or could not be tied to one
    meta_row_location   main_metadata.csv row position the location came from
    lat_llm / lon_llm   OLD LLM coordinates (actual, else inferred; first entry on the page) -- not used
    watersource_name    LLM-extracted water source name (page level)

USAGE
-----
    python code/02_inventory/build_table_level_metadata.py

Run from the repo root.  Requires the hydro conda environment.
Re-running overwrites both output files.
=============================================================================
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from date_parsing import add_date_columns, apply_table_dates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO        = Path(__file__).resolve().parents[2]
DIGITIZED   = REPO / 'data' / 'digitized'
COUNTS_PATH = REPO / 'data' / 'analysis' / 'csv_row_counts.csv'
LOC_PATH    = REPO / 'data' / 'analysis' / 'location_validation' / 'locations.csv'
TABLE_DATES_PATH = REPO / 'data' / 'analysis' / 'table_dates.csv'  # 14_table_dates.py
ENTRY_PATH  = REPO / 'data' / 'analysis' / 'entry_dates.csv'   # matched_tables per entry (15_entry_dates.py)
META_PATH   = REPO / 'data' / 'metadata' / 'main_metadata.csv'
OUT_HEADERS = REPO / 'data' / 'analysis' / 'table_headers.csv'
OUT_TABLE   = REPO / 'data' / 'analysis' / 'table_level_metadata.csv'

# ---------------------------------------------------------------------------
# Stage 1 — Header keyword classifier
#
# Rules are checked in order; first match wins.
# Each entry is (water_type, [keywords]).
# Keywords are matched case-insensitively against the full joined header text.
#
# Ordering rationale:
#   1. TOC/index markers first — prevents mis-classifying an index listing
#      "discharge" station names as Stream Discharge data.
#   2. Water Quality before Groundwater — both can contain "temperature";
#      specific chemical parameter terms identify water quality uniquely.
#   3. Groundwater before Reservoir — "water level" / "elevation" appear in
#      both; groundwater-specific terms (depth, static, artesian) take priority.
#   4. Reservoir before Stream Discharge — "storage" and "acre-feet" (in a
#      non-runoff context) indicate reservoir data.
#   5. Stream Discharge last among primary types — broadest keyword set.
#   6. A second Reservoir rule catches plain "storage"/"acre-feet" that appear
#      after stream-discharge keywords are exhausted.
# ---------------------------------------------------------------------------
RULES = [
    ('Not Water Related', [
        'gaging station', 'map no', 'map number', 'w.s.p. no', 'w.s.p no',
        'wsp no', 'period of record', 'station name', 'index no',
        'index number', 'w. s. p.',
    ]),
    ('Water Quality', [
        'ph ', 'alkalinity', 'hardness', 'turbidity', 'specific conductance',
        'suspended sediment', 'dissolved solids', 'biochemical oxygen',
        'coliform', 'nitrate', 'chloride', 'sulfate', 'bicarbonate',
        'aluminum', 'iron (fe', 'lead (pb',
    ]),
    ('Groundwater', [
        'depth to water', 'water-level', 'water level', 'static level',
        'pumping level', 'artesian', 'well depth', 'water table',
    ]),
    ('Reservoir', [
        'reservoir storage', 'end-of-month storage', 'end of month storage',
        'reservoir capacity', 'spillway', 'inflow', 'outflow', 'evaporation',
    ]),
    ('Precipitation', [
        'precipitation', 'rainfall', 'snowfall', 'snow depth',
        'snow pack', 'snow course', 'rain gauge', 'rain gage',
    ]),
    ('Irrigation', [
        'diversion', 'irrigation', 'canal', 'lateral', 'delivery',
        'acres irrigated', 'duty of water', 'water delivered',
    ]),
    ('Springs', [
        'spring discharge', 'spring flow', 'spring temperature',
        'mineral spring',
    ]),
    ('Stream Discharge', [
        'discharge', 'second-feet', 'sec.-ft', 'sec-ft', ' cfs',
        'cubic feet per second', 'gage height', 'gage-height',
        'stream-flow', 'streamflow', 'runoff in acre',
        'maximum day', 'minimum day',
        'runoff',          # annual runoff tables (e.g. "Water year  Runoff")
    ]),
    # Catch-all reservoir signals after stream discharge exhausted
    ('Reservoir', [
        'storage', 'acre-feet', 'contents', 'capacity',
    ]),
]

# Monthly column pattern.
# Presence of month abbreviations = "low" confidence signal (data table, but
# water type is ambiguous — the Water year/Oct/Nov/.../Sept format is shared
# by stream discharge monthly means, reservoir monthly storage, and
# precipitation monthly totals).
MONTH_COLS = re.compile(
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', re.I
)


def classify_headers(header_text: str) -> tuple[str, str]:
    """
    Apply keyword rules to header text.

    Returns
    -------
    (water_type, confidence)
        water_type  : one of the nine USGS categories, or 'uncertain'
        confidence  : 'high'      — matched a keyword rule
                      'low'       — no keyword match but month columns present
                      'uncertain' — no signal at all
    """
    h = header_text.lower()

    for water_type, keywords in RULES:
        for kw in keywords:
            if kw in h:
                return water_type, 'high'

    if MONTH_COLS.search(h):
        return 'uncertain', 'low'

    return 'uncertain', 'uncertain'


# ---------------------------------------------------------------------------
# Step 1 — extract raw headers from every table CSV
# ---------------------------------------------------------------------------

def extract_headers(csv_path: Path, n_header_rows: int = 5) -> str:
    """
    Reads the first n_header_rows rows of a CSV and returns a single string
    of all non-null, non-'Unnamed' cell values joined by spaces.

    n_header_rows=5 captures multi-level Reducto headers (which can span
    3–4 rows before data begins) while staying fast for large files.
    """
    try:
        raw = pd.read_csv(csv_path, header=None, nrows=n_header_rows,
                          dtype=str, encoding='utf-8', encoding_errors='ignore')
    except Exception:
        return ''

    tokens = []
    for _, row in raw.iterrows():
        for cell in row:
            if pd.isna(cell):
                continue
            cell = str(cell).strip()
            if not cell or cell.startswith('Unnamed'):
                continue
            tokens.append(cell)
    return ' '.join(tokens)


FILE_RE = re.compile(r'^(\d+)_page_(\d+)_table(\d+)\.csv$')


def collect_headers() -> pd.DataFrame:
    """
    Walk data/digitized/, read headers from every *_table*.csv, classify each,
    and return a DataFrame with one row per table.
    """
    records = []
    csv_files = list(DIGITIZED.rglob('*_table*.csv'))
    total = len(csv_files)
    print(f'Found {total:,} table CSVs in {DIGITIZED}')

    for i, path in enumerate(csv_files):
        if i % 10000 == 0:
            print(f'  {i:,}/{total:,} ...', flush=True)

        m = FILE_RE.match(path.name)
        if not m:
            continue

        doc_id       = int(m.group(1))
        page_number  = int(m.group(2))
        table_number = int(m.group(3))
        header_text  = extract_headers(path)
        water_type, confidence = classify_headers(header_text)

        records.append({
            'doc_id':            doc_id,
            'page_number':       page_number,
            'table_number':      table_number,
            'header_text':       header_text[:500],
            'water_type_header': water_type,
            'confidence':        confidence,
        })

    df = pd.DataFrame(records).sort_values(['doc_id', 'page_number', 'table_number'])
    return df


# ---------------------------------------------------------------------------
# Step 2 — prepare LLM metadata for fallback join
# ---------------------------------------------------------------------------

# dates: shared parser (code/02_inventory/date_parsing.py) -- do not copy it here


WATER_TYPE_MAP = {
    'stream discharge':  'Stream Discharge',
    'groundwater':       'Groundwater',
    'reservoir':         'Reservoir',
    'irrigation':        'Irrigation',
    'springs':           'Springs',
    'precipitation':     'Precipitation',
    'water quality':     'Water Quality',
    'not water related': 'Not Water Related',
    'other':             'Other',
}

def clean_water_type(wt):
    if pd.isna(wt):
        return 'Other'
    s = str(wt).strip().lower()
    for k, v in WATER_TYPE_MAP.items():
        if k in s:
            return v
    return 'Other'


def build_page_meta() -> pd.DataFrame:
    """
    Build a page-level summary from main_metadata.csv for use as the Stage 2
    fallback.  Returns one row per (doc_id, page_number) with:
        water_type_llm       — first LLM water type for the page
        llm_page_ambiguous   — True if the page had >1 distinct LLM water type
        year_start / year_end / decade_start
        lat_combined / lon_combined
        watersource_name

    The 'first LLM water type' is kept for ambiguous pages because it is the
    most we can reliably say without the header classifier resolving which
    table is which.  The llm_page_ambiguous flag lets downstream users identify
    and audit these cases.

    NOTE: main_metadata.csv is read-only; this function never modifies it.
    """
    meta = pd.read_csv(META_PATH, low_memory=False)
    meta['wt_clean'] = meta['water_type'].map(clean_water_type)

    add_date_columns(meta)   # year_start, year_end, decade_start, date_status

    # OLD LLM coordinates, kept for reference only (rebuilt locations: build_table_locations)
    meta['lat_llm'] = meta['actual_latitude'].combine_first(meta['inferred_latitude'])
    meta['lon_llm'] = meta['actual_longitude'].combine_first(meta['inferred_longitude'])

    # Identify ambiguous pages (>1 distinct LLM water type)
    type_counts = (meta.groupby(['id', 'page_number'])['wt_clean']
                   .nunique()
                   .reset_index(name='n_llm_types'))
    meta = meta.merge(type_counts, on=['id', 'page_number'])
    meta['llm_page_ambiguous'] = meta['n_llm_types'] > 1

    keep = ['id', 'page_number', 'wt_clean', 'llm_page_ambiguous',
            'year_start', 'year_end', 'decade_start', 'date_status',
            'lat_llm', 'lon_llm', 'watersource_name']

    # One row per page — keep first occurrence (consistent ordering)
    page_meta = (meta[keep]
                 .drop_duplicates(subset=['id', 'page_number'], keep='first')
                 .rename(columns={'id': 'doc_id', 'wt_clean': 'water_type_llm'}))
    return page_meta


# ---------------------------------------------------------------------------
# Step 2b — rebuilt table locations
# ---------------------------------------------------------------------------
CLASS_RANK = {'precise': 0, 'approximate': 1, 'basin': 2}
AMBIGUOUS_KM = 5


def _km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_table_locations(tables: pd.DataFrame) -> pd.DataFrame:
    """
    One rebuilt location per table, from the LLM entries (main_metadata rows) on its page.

    Entry -> table links come from 15_entry_dates.py (`matched_tables`): an entry is tied to
    specific tables when its site ID / owner name matches table rows, its name appears in
    one table's caption, or the page has a single table. An entry tied to every table on a
    multi-table page is a page-level candidate only.

    For each table: candidates = located entries tied specifically to it; if none, all
    located entries on the page (location_match = 'page'). Keep the best precision class
    (precise > approximate > basin), then the smallest precision_km (ties: first row).
    Same-page rule: candidates are only entries of the table's own page.
    """
    loc = pd.read_csv(LOC_PATH)
    ent = pd.read_csv(ENTRY_PATH, usecols=['meta_row', 'matched_tables'])
    loc = loc.merge(ent, on='meta_row', how='left')
    loc = loc[loc.precision_class.isin(CLASS_RANK)].copy()
    loc['doc_id'] = loc['id'].astype('int64')
    loc['page_number'] = loc['page_number'].astype('int64')
    loc['rank'] = loc.precision_class.map(CLASS_RANK)
    loc = loc.sort_values(['rank', 'precision_km', 'meta_row'])

    tables = tables.astype('int64')
    tn_by_page = tables.groupby(['doc_id', 'page_number']).table_number.unique()
    out = []
    for (doc, page), g in loc.groupby(['doc_id', 'page_number'], sort=False):
        if (doc, page) not in tn_by_page.index:
            continue
        tns = tn_by_page[(doc, page)]
        n_t = len(tns)
        sets = [set(int(x) for x in str(m).split(',')) if pd.notna(m) and str(m) else set()
                for m in g.matched_tables]
        for tn in tns:
            spec = [i for i, st in enumerate(sets) if tn in st and (len(st) < n_t or n_t == 1)]
            idx, how = (spec, 'entry') if spec else (list(range(len(g))), 'page')
            c = g.iloc[idx]
            c = c[c['rank'] == c['rank'].min()]
            best = c.iloc[0]
            pts = c[['lat', 'lon']].round(4).drop_duplicates()
            spread = float(_km(best.lat, best.lon, pts.lat.values, pts.lon.values).max())
            out.append({'doc_id': doc, 'page_number': page, 'table_number': tn,
                        'lat_combined': best.lat, 'lon_combined': best.lon,
                        'coord_method': best.coord_method, 'precision_km': best.precision_km,
                        'precision_class': best.precision_class,
                        'evidence_scope': best.evidence_scope, 'location_match': how,
                        'n_site_locations': len(pts), 'site_spread_km': round(spread, 2),
                        'meta_row_location': best.meta_row})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Step 3 — assign final water type (hybrid logic)
# ---------------------------------------------------------------------------

def assign_final_type(row) -> tuple[str, str]:
    """
    Three-stage assignment for each table row.

    Stage 1 — LLM unambiguous (trusted):
        Page had exactly one distinct LLM water type → use it directly.

    Stage 2 — LLM ambiguous (header resolves):
        Page had >1 distinct LLM water type → use header if high confidence,
        otherwise fall back to the first LLM type and flag as 'llm_ambiguous'.

    Stage 3 — No LLM data:
        Use header if high confidence, otherwise mark 'uncertain'.

    Returns (water_type_final, assignment_source).
    """
    llm = row.get('water_type_llm')
    llm_has_value = pd.notna(llm) and str(llm) != ''

    # Stage 1: unambiguous LLM — always trust it
    if llm_has_value and not row.get('llm_page_ambiguous', False):
        return llm, 'llm_unambiguous'

    # Stage 2: ambiguous LLM — header resolves if confident
    if llm_has_value and row.get('llm_page_ambiguous', False):
        if row['confidence'] == 'high':
            return row['water_type_header'], 'header_high'
        return llm, 'llm_ambiguous'

    # Stage 3: no LLM data — header only
    if row['confidence'] == 'high':
        return row['water_type_header'], 'header_no_llm'

    return 'uncertain', 'no_data'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Step 1: extract and classify headers
    # ------------------------------------------------------------------
    print('=== Step 1: Extracting and classifying headers ===')
    if OUT_HEADERS.exists():
        print(f'  Loading cached headers from {OUT_HEADERS.name} (delete to re-scan)')
        headers_df = pd.read_csv(OUT_HEADERS)
    else:
        headers_df = collect_headers()
        headers_df.to_csv(OUT_HEADERS, index=False)
    print(f'Loaded {len(headers_df):,} rows from {OUT_HEADERS.name}')
    print()
    print('Header classifier results:')
    print(headers_df['water_type_header'].value_counts().to_string())
    print()
    print('Confidence breakdown:')
    print(headers_df['confidence'].value_counts().to_string())
    print()

    # ------------------------------------------------------------------
    # Step 2: join measurement counts
    # ------------------------------------------------------------------
    print('=== Step 2: Joining measurement counts ===')
    counts = pd.read_csv(COUNTS_PATH)
    merged = headers_df.merge(
        counts, on=['doc_id', 'page_number', 'table_number'], how='outer'
    )
    print(f'Tables in headers only (no count row):  {merged["number_rows"].isna().sum():,}')
    print(f'Tables in counts only (no header row):  {merged["water_type_header"].isna().sum():,}')
    print(f'Matched on both sides:                  '
          f'{(merged["number_rows"].notna() & merged["water_type_header"].notna()).sum():,}')
    print()

    # ------------------------------------------------------------------
    # Step 3: join page-level LLM metadata
    # ------------------------------------------------------------------
    print('=== Step 3: Joining LLM page metadata (Stage 2 fallback) ===')
    page_meta = build_page_meta()
    merged = merged.merge(page_meta, on=['doc_id', 'page_number'], how='left')
    print(f'Tables with LLM water type available: '
          f'{merged["water_type_llm"].notna().sum():,} of {len(merged):,}')
    ambig_pages = merged['llm_page_ambiguous'].sum()
    print(f'Tables on LLM-ambiguous pages:        {int(ambig_pages):,}')
    print()

    print('=== Step 3b: Joining rebuilt table locations ===')
    tl = build_table_locations(merged[['doc_id', 'page_number', 'table_number']].dropna())
    merged = merged.merge(tl, on=['doc_id', 'page_number', 'table_number'], how='left')
    merged['precision_class'] = merged['precision_class'].fillna('unlocated')
    merged['location_match'] = merged['location_match'].fillna('none')
    # several candidate sites >5 km apart: the single point is a guess among them
    merged['location_ambiguous'] = merged['site_spread_km'].fillna(0) > AMBIGUOUS_KM
    print(merged['precision_class'].value_counts().to_string())
    print(merged['location_match'].value_counts().to_string())
    print()

    # ------------------------------------------------------------------
    # Step 4: assign final water type
    # ------------------------------------------------------------------
    print('=== Step 4: Assigning final water type (hybrid) ===')
    results = merged.apply(assign_final_type, axis=1, result_type='expand')
    results.columns = ['water_type_final', 'assignment_source']
    merged = pd.concat([merged, results], axis=1)

    print('Assignment source breakdown:')
    print(merged['assignment_source'].value_counts().to_string())
    print()
    print('Final water type distribution:')
    print(merged['water_type_final'].value_counts().to_string())
    print()

    # table-level dates replace the page-level LLM dates (kept as *_llm)
    merged = apply_table_dates(merged, TABLE_DATES_PATH)
    print('Table date status:')
    print(merged['date_status'].value_counts().to_string())
    print()

    # ------------------------------------------------------------------
    # Step 5: save
    # ------------------------------------------------------------------
    col_order = [
        'doc_id', 'page_number', 'table_number',
        'number_rows',
        'header_text', 'water_type_header', 'confidence',
        'water_type_llm', 'llm_page_ambiguous',
        'water_type_final', 'assignment_source',
        'year_start', 'year_end', 'decade_start', 'date_status', 'date_tier',
        'por_start', 'por_end', 'construction_start', 'construction_end',
        'year_start_llm', 'year_end_llm', 'decade_start_llm', 'date_status_llm',
        'lat_combined', 'lon_combined', 'coord_method', 'precision_km',
        'precision_class', 'evidence_scope', 'location_match', 'n_site_locations',
        'site_spread_km', 'location_ambiguous', 'meta_row_location', 'lat_llm', 'lon_llm', 'watersource_name',
    ]
    final = merged[[c for c in col_order if c in merged.columns]]
    final.to_csv(OUT_TABLE, index=False)

    print('=== Summary ===')
    print(f'Output rows:                    {len(final):,}')
    matched = final[final['number_rows'].notna()]
    print(f'Tables with measurement count:  {len(matched):,}')
    print(f'Total estimated measurements:   {matched["number_rows"].sum():,.0f}')
    print()
    print('Measurements by final water type:')
    by_type = (matched.groupby('water_type_final')['number_rows']
               .agg(['sum', 'count'])
               .rename(columns={'sum': 'measurements', 'count': 'tables'})
               .sort_values('measurements', ascending=False))
    print(by_type.to_string())
    print()
    print(f'Saved -> {OUT_TABLE.name}')


if __name__ == '__main__':
    main()
