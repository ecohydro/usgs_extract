"""
Published table set (2026-11-18, new metadata): every table in the new metadata that has an answer.

  python filter_published_tables.py
    data/analysis/table_level_metadata.csv -> data/analysis/table_level_metadata_published.csv

Decision (Annette 2026-11-18): the published list is the tables in the new metadata table (Anna's
pipeline). Tables with no usable answer (assignment_source 'no_metadata': unreadable answer, no answer
for the table, empty Reducto page; 149 tables) are EXCLUDED. They cannot be rerun: the files are
corrupted (Anna, 2026-11-18; documented in her methods). List:
data/analysis/manual_review/new_metadata_gaps/rerun_for_anna/rerun_tables.csv.

The earlier rule (2026-06-10: keep only tables the Phi-4 LLM processed) is in git history.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IN_TABLE = REPO / 'data/analysis/table_level_metadata.csv'
OUT_TABLE = REPO / 'data/analysis/table_level_metadata_published.csv'


def main():
    df = pd.read_csv(IN_TABLE, low_memory=False)
    kept = df[df['assignment_source'].ne('no_metadata')]
    kept.to_csv(OUT_TABLE, index=False)
    print(f'tables {len(df):,}; published {len(kept):,}; left out (no metadata) {len(df) - len(kept):,}')
    if 'n_measurements' in kept:
        print(kept.groupby('water_type_final')['n_measurements'].agg(['count', 'sum'])
              .rename(columns={'count': 'tables', 'sum': 'measurements'})
              .sort_values('measurements', ascending=False).to_string())


if __name__ == '__main__':
    main()
