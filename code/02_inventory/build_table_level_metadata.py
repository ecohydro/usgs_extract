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
    year_start          earliest year parsed from dates_of_recording
    year_end            latest year parsed from dates_of_recording
    decade_start        floor(year_start / 10) * 10
    lat_combined        actual_latitude if present, else inferred_latitude
    lon_combined        actual_longitude if present, else inferred_longitude
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO        = Path(__file__).resolve().parents[2]
DIGITIZED   = REPO / 'data' / 'digitized'
COUNTS_PATH = REPO / 'data' / 'analysis' / 'csv_row_counts.csv'
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

OPEN_ENDED = ('present', 'current', 'ongoing', 'to date', 'today')

def parse_year_range(date_str):
    if pd.isna(date_str):
        return None, None
    s = str(date_str).lower()
    if any(t in s for t in OPEN_ENDED):
        return None, None
    years = [int(y) for y in re.findall(r'\b(1[89]\d{2}|20[0-7]\d)\b', s)
             if 1800 <= int(y) <= 1980]
    if not years:
        return None, None
    return min(years), max(years)


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

    year_pairs = meta['dates_of_recording'].map(parse_year_range)
    meta['year_start']   = year_pairs.map(lambda x: x[0]).astype('Int64')
    meta['year_end']     = year_pairs.map(lambda x: x[1]).astype('Int64')
    meta['decade_start'] = (meta['year_start'] // 10 * 10).astype('Int64')

    meta['lat_combined'] = meta['actual_latitude'].combine_first(meta['inferred_latitude'])
    meta['lon_combined'] = meta['actual_longitude'].combine_first(meta['inferred_longitude'])

    # Identify ambiguous pages (>1 distinct LLM water type)
    type_counts = (meta.groupby(['id', 'page_number'])['wt_clean']
                   .nunique()
                   .reset_index(name='n_llm_types'))
    meta = meta.merge(type_counts, on=['id', 'page_number'])
    meta['llm_page_ambiguous'] = meta['n_llm_types'] > 1

    keep = ['id', 'page_number', 'wt_clean', 'llm_page_ambiguous',
            'year_start', 'year_end', 'decade_start',
            'lat_combined', 'lon_combined', 'watersource_name']

    # One row per page — keep first occurrence (consistent ordering)
    page_meta = (meta[keep]
                 .drop_duplicates(subset=['id', 'page_number'], keep='first')
                 .rename(columns={'id': 'doc_id', 'wt_clean': 'water_type_llm'}))
    return page_meta


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

    # ------------------------------------------------------------------
    # Step 5: save
    # ------------------------------------------------------------------
    col_order = [
        'doc_id', 'page_number', 'table_number',
        'number_rows',
        'header_text', 'water_type_header', 'confidence',
        'water_type_llm', 'llm_page_ambiguous',
        'water_type_final', 'assignment_source',
        'year_start', 'year_end', 'decade_start',
        'lat_combined', 'lon_combined', 'watersource_name',
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
