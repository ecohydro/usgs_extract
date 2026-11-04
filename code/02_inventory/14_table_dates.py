"""
Table-level dates for every Reducto table (tiered protocol in table_dates.py).

Inputs:  data/analysis/page_blocks.parquet  (11_page_blocks_cache.py)
         data/metadata/main_metadata.csv    (publication year per document)
Outputs: data/analysis/table_dates.csv
            one row per table: id, page_number, table_number, year_start,
            year_end, date_tier (A_date_column / B_table_text / C_caption /
            D_page_text / none), date_quote, n_date_values, date_columns,
            por_start, por_end, por_quote, n_body_rows, n_cols
         data/analysis/table_row_dates.parquet
            tier-A tables only: one row per table row that carries a date,
            with the row's site label (first non-date cell, e.g. well number
            or station) -> used to match LLM entries to rows.
Table N on a page = the N-th Table block = {doc}_page_{n}_tableN.csv
(block and CSV counts agree on 99.8% of pages).

Usage: python 14_table_dates.py [doc_id page_number]   (args = debug one page)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from html_table import parse_table, column_headers  # noqa: E402
from table_dates import table_dates, date_columns, YEAR_MAX  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAPTION_BLOCKS = 4  # non-table blocks directly above a table that count as its caption

B = pd.read_parquet(ROOT / 'data/analysis/page_blocks.parquet')
B['id'] = B['id'].astype(str)
md = pd.read_csv(ROOT / 'data/metadata/main_metadata.csv', usecols=['id', 'Year Published'], dtype={'id': str})
md['id'] = md['id'].str.replace(r'\.0$', '', regex=True)
pub = pd.to_numeric(md.drop_duplicates('id').set_index('id')['Year Published'], errors='coerce').to_dict()

debug = None
if len(sys.argv) == 3:
    debug = (sys.argv[1], int(sys.argv[2]))
    B = B[(B.id == debug[0]) & (B.page_number == debug[1])]

B = B.sort_values(['id', 'page_number', 'block_order'])
out, row_out = [], []
for (doc, page), g in B.groupby(['id', 'page_number'], sort=False):
    cap = pub.get(doc)
    cap = int(cap) if pd.notna(cap) else YEAR_MAX
    texts = [c for t, c in zip(g.block_type, g.content) if t != 'Table']
    page_text = '\n'.join(texts)
    since_last = []
    for t, c, tix in zip(g.block_type, g.content, g.table_idx):
        if t != 'Table':
            since_last.append(c)
            continue
        caption = '\n'.join(since_last[-CAPTION_BLOCKS:])
        since_last = []
        r = table_dates(c, caption, page_text, cap)
        ry = r.pop('row_years')
        if ry:
            head, body = parse_table(c)
            dcols = date_columns(column_headers(head), body)
            for ri, ys in ry.items():
                row = body[ri]
                label = next((x for i, x in enumerate(row) if i not in dcols and x and x.strip()), '')
                row_out.append((doc, page, int(tix), ri, label[:120], min(ys), max(ys)))
        title = ' '.join(caption.split())[-300:]
        out.append({'id': doc, 'page_number': page, 'table_number': int(tix), 'pub_year': cap,
                    'caption': title or None, **r})
        if debug:
            print(f'--- table {tix}  caption: {caption[-200:]!r}')
            print({k: v for k, v in r.items()})

df = pd.DataFrame(out)
df['date_tier'] = df['date_tier'].fillna('none')
for c in ['year_start', 'year_end', 'por_start', 'por_end']:
    df[c] = df[c].astype('Int64')
if debug:
    sys.exit()
df.to_csv(ROOT / 'data/analysis/table_dates.csv', index=False)
rd = pd.DataFrame(row_out, columns=['id', 'page_number', 'table_number', 'row_idx', 'site_label',
                                    'year_min', 'year_max'])
rd.to_parquet(ROOT / 'data/analysis/table_row_dates.parquet', index=False)
print(f'tables {len(df):,}  dated {df.year_start.notna().mean():.1%}  row-date rows {len(rd):,}')
print(df.date_tier.value_counts().to_string())
