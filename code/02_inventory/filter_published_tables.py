"""
filter_published_tables.py
=============================================================================
Produces data/analysis/table_level_metadata_published.csv — the publication
subset of table_level_metadata.csv containing ONLY tables that were processed
by the LLM (Phi-4) metadata extraction.

WHY THIS FILTER EXISTS
----------------------
build_table_level_metadata.py assigns a water type to all 106,191 table CSVs
using a hybrid of LLM classification and a header keyword fallback.  A subset
of those tables were never seen by the LLM — the extraction was run in batches
and some pages were missed — and were typed (or left 'uncertain') by the header
classifier alone.

For publication we restrict the dataset to LLM-processed tables only.  The
header-only tables are dropped, not reclassified, because:

  * They have no LLM-extracted metadata (no watersource_name, dates, or
    coordinates), so they cannot be located, dated, or cross-validated.
  * The entire 'uncertain' bucket (tables the header classifier also could not
    type — classic monthly-format data tables, OCR garbage, and flood-damage
    cost tables) consists exclusively of these non-LLM tables.

  Decision (2026-06-10, A. Hilton): work only with LLM-processed data.  Drop
  every table with no LLM metadata; do not attempt header-only recovery.

WHAT IS DROPPED  (assignment_source)
------------------------------------
    'no_data'        — no LLM metadata AND header could not type it.  This is
                       the entire water_type_final == 'uncertain' bucket.
    'header_no_llm'  — no LLM metadata; typed by header keyword only.

WHAT IS KEPT  (assignment_source)
---------------------------------
    'llm_unambiguous' — page had exactly one LLM water type (trusted directly).
    'llm_ambiguous'   — page had >1 LLM type; header could not resolve, first
                        LLM type used (flagged).
    'header_high'     — page had >1 LLM type; header resolved which table is
                        which.  The LLM DID run on these pages.

The 'Other' category is retained as-is: these are LLM-processed pages where the
LLM detected a data table but could not determine the water type from context.
They are a legitimate, documented category — not reclassified.

OUTPUT
------
    data/analysis/table_level_metadata_published.csv  — one row per LLM-processed
        table CSV.  Same columns as table_level_metadata.csv.  This is the
        correct input for publication measurement counts and Fig 7.

USAGE
-----
    python code/02_inventory/filter_published_tables.py

Run from the repo root.  Reads table_level_metadata.csv (produced by
build_table_level_metadata.py); does not modify it.
=============================================================================
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = Path(__file__).resolve().parents[2]
IN_TABLE = REPO / 'data' / 'analysis' / 'table_level_metadata.csv'
OUT_TABLE = REPO / 'data' / 'analysis' / 'table_level_metadata_published.csv'

# Assignment sources where the LLM saw the page.
LLM_SOURCES = {'llm_unambiguous', 'llm_ambiguous', 'header_high'}
# Assignment sources with no LLM metadata — dropped for publication.
NONLLM_SOURCES = {'no_data', 'header_no_llm'}


def main():
    df = pd.read_csv(IN_TABLE)
    n_all = len(df)
    meas_all = df['number_rows'].sum()

    is_llm = df['assignment_source'].isin(LLM_SOURCES)
    kept = df[is_llm].copy()
    dropped = df[~is_llm]

    print('=== Filter: LLM-processed tables only ===')
    print(f'Input tables:            {n_all:,}  ({meas_all:,.0f} measurements)')
    print(f'Dropped (no LLM data):   {len(dropped):,}  '
          f'({dropped["number_rows"].sum():,.0f} measurements)')
    print(f'Kept (LLM-processed):    {len(kept):,}  '
          f'({kept["number_rows"].sum():,.0f} measurements)')
    print()

    print('Dropped breakdown by assignment_source:')
    print(dropped['assignment_source'].value_counts().to_string())
    print()
    print('Dropped breakdown by water_type_final (header-only guesses, discarded):')
    print(dropped['water_type_final'].value_counts().to_string())
    print()

    # Sanity check: nothing 'uncertain' should survive into the published file.
    n_uncertain = (kept['water_type_final'] == 'uncertain').sum()
    assert n_uncertain == 0, f'{n_uncertain} uncertain tables leaked into kept set'

    kept.to_csv(OUT_TABLE, index=False)

    print('=== Published dataset: measurements by water type ===')
    by_type = (kept.groupby('water_type_final')['number_rows']
               .agg(['sum', 'count'])
               .rename(columns={'sum': 'measurements', 'count': 'tables'})
               .sort_values('measurements', ascending=False))
    by_type['measurements'] = by_type['measurements'].map(lambda x: f'{x:,.0f}')
    print(by_type.to_string())
    print()
    print(f'Saved -> {OUT_TABLE.relative_to(REPO)}')


if __name__ == '__main__':
    main()
