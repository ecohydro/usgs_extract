"""
Per-table digitization-quality metrics and flag, from the Reducto table CSVs.

This describes how well the TABLE STRUCTURE / CELLS were digitized, not whether the
values are hydrologically right (values are not cleaned in this project).

Metrics (data rows = CSV rows after the first/header row):
  n_data_rows, n_cols
  frac_empty        empty cells / all cells in data rows
  frac_numeric      numeric-looking cells / non-empty cells ("1,234", "-0.5", "12e", "45*")
  frac_ocr_suspect  non-empty cells that look like numbers with a letter-for-digit OCR swap
                    ("4,02C", "1O5", "3l2", "2S0"). Letters used: O o l I S Z (never valid
                    in USGS numbers or PLSS tract letters); C only inside a comma number;
                    a lone "o"/"O" cell (OCR zero).
                    Empty cells are NOT penalized much: daily tables legitimately leave months
                    without record blank.
  one_column        table collapsed into a single column (structure lost)

Flag (digitization_quality):
  poor   no data rows, or frac_empty > 0.9, or frac_ocr_suspect > 0.05
  fair   frac_ocr_suspect > 0.01, or one_column
  good   otherwise
table_content: numeric (frac_numeric >= 0.6) | mixed (>= 0.2) | text (< 0.2) -- a text table
(e.g. a station list) is not a digitization problem by itself.
Thresholds are heuristic (not validated against graded tables).

Output: data/analysis/table_quality.csv (one row per table CSV; cached -- delete to re-scan)
"""
import csv
import re
import sys
from pathlib import Path

import pandas as pd

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parents[2]
DIGITIZED = ROOT / 'data/digitized'
OUT = ROOT / 'data/analysis/table_quality.csv'
FILE_RE = re.compile(r'^(\d+)_page_(\d+)_table(\d+)\.csv$')
NUMERIC = re.compile(r'^[-+(]?\$?\d[\d,]*\.?\d*\)?[a-zA-Z*#†‡]?$|^[-+]?\.\d+$')
OCR_SWAP = re.compile(r'^(?:(?=.*\d)(?:\d[\d,.]*[OolISZ][\d,.]*|[OolISZ][\d,.]*\d[\d,.]*|\d{1,3},\d{0,2}C\d*)|[Oo])$')


def metrics(path: Path) -> dict:
    try:
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            rows = list(csv.reader(f))
    except Exception:
        rows = []
    data = rows[1:]
    n_cols = max((len(r) for r in rows), default=0)
    cells = [c.strip() for r in data for c in r]
    n = len(cells)
    filled = [c for c in cells if c and c.lower() not in ('nan', '-', '--', '—', '...', '. . .')]
    nf = len(filled)
    return {
        'n_data_rows': len(data), 'n_cols': n_cols,
        'frac_empty': round(1 - nf / n, 3) if n else 1.0,
        'frac_numeric': round(sum(bool(NUMERIC.match(c.replace(' ', ''))) for c in filled) / nf, 3) if nf else 0.0,
        'frac_ocr_suspect': round(sum(bool(OCR_SWAP.match(c.replace(' ', ''))) for c in filled) / nf, 4) if nf else 0.0,
        'one_column': n_cols <= 1,
    }


def flag(r) -> str:
    if r.n_data_rows == 0 or r.frac_empty > 0.9 or r.frac_ocr_suspect > 0.05:
        return 'poor'
    if r.frac_ocr_suspect > 0.01 or r.one_column:
        return 'fair'
    return 'good'


def main():
    if OUT.exists() and '--force' not in sys.argv:
        print(f'{OUT.name} exists (delete or pass --force to re-scan)')
        return
    files = list(DIGITIZED.rglob('*_table*.csv'))
    print(f'{len(files):,} table CSVs')
    recs = []
    for i, p in enumerate(files):
        if i % 20000 == 0:
            print(f'  {i:,}', flush=True)
        m = FILE_RE.match(p.name)
        if m:
            recs.append({'doc_id': int(m[1]), 'page_number': int(m[2]), 'table_number': int(m[3]),
                         **metrics(p)})
    q = pd.DataFrame(recs).sort_values(['doc_id', 'page_number', 'table_number'])
    q['digitization_quality'] = q.apply(flag, axis=1)
    q['table_content'] = pd.cut(q.frac_numeric, [-1, 0.2, 0.6, 2], right=False,
                                labels=['text', 'mixed', 'numeric']).astype(str)
    q.to_csv(OUT, index=False)
    print(q.digitization_quality.value_counts().to_string())
    print(q.table_content.value_counts().to_string())


if __name__ == '__main__':
    main()
