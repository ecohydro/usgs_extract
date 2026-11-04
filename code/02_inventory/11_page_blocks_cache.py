"""
Cache every Reducto page block (type + text) in one parquet file.

Many downstream steps (table-level dates, tribal-lands text screen, table
quality flags) need the page text and need to know which text is a table and
which is a caption or narrative. Reading 74k JSON files each time is slow, so
this script reads them once.

Input:  data/digitized/{doc}/page_{n}/{doc}_page_{n}.json
Output: data/analysis/page_blocks.parquet
        one row per block: id, page_number, chunk_idx, block_idx, block_order,
        block_type (Header/Title/Text/List Item/Table/...), content
        Table blocks keep their HTML (<table>...); table_idx numbers the Table
        blocks on the page 1..n in reading order (NaN for non-table blocks).
Re-run: delete the parquet.
"""
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'digitized'
OUT = ROOT / 'data' / 'analysis' / 'page_blocks.parquet'

rows = []
n_pages = n_bad = 0
for doc_dir in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith('_')):
    for page_dir in doc_dir.iterdir():
        m = re.fullmatch(r'page_(\d+)', page_dir.name)
        if not m:
            continue
        page = int(m.group(1))
        jp = page_dir / f'{doc_dir.name}_page_{page}.json'
        if not jp.exists():
            continue
        n_pages += 1
        try:
            j = json.loads(jp.read_text(encoding='utf-8'))
            chunks = j['result']['chunks']
        except Exception:
            n_bad += 1
            continue
        order = 0
        tix = 0
        for ci, c in enumerate(chunks):
            blocks = c.get('blocks') or [{'type': 'Text', 'content': c.get('content', '')}]
            for bi, b in enumerate(blocks):
                t = b.get('type', '')
                if t == 'Table':
                    tix += 1
                rows.append((doc_dir.name, page, ci, bi, order, t,
                             str(b.get('content', '') or ''), tix if t == 'Table' else None))
                order += 1

df = pd.DataFrame(rows, columns=['id', 'page_number', 'chunk_idx', 'block_idx', 'block_order',
                                 'block_type', 'content', 'table_idx'])
df['table_idx'] = df['table_idx'].astype('Int64')
df.to_parquet(OUT, index=False)
print(f'pages read {n_pages:,}  unreadable {n_bad:,}  blocks {len(df):,}')
print(df.block_type.value_counts().to_string())
