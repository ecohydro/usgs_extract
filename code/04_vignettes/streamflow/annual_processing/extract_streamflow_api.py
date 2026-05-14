"""
extract_streamflow_api.py — extract annual/monthly streamflow data from pre-digitized USGS pages.

Filters main_metadata.csv for stream-discharge + annual/yearly/year temporal_resolution pages where
the JSON exists, sends each page's content to Claude API for classification and extraction, and
appends results to three output CSVs.

Usage:
    python extract_streamflow_api.py [--limit N] [--workers K] [--model claude-sonnet-4-6] [--dry-run]

Resumes by skipping pages already logged in extraction_log.csv.
"""

import argparse
import csv
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PROMPT_PATH = SCRIPT_DIR / 'annual_streamflow_extraction_instructions.md'
META_PATH = REPO_ROOT / 'data' / 'metadata' / 'main_metadata.csv'
OUTPUT_DIR = REPO_ROOT / 'data' / 'analysis' / 'streamflow' / 'annual'

CANDIDATE_ROOTS = [
    Path('/Volumes/AHILTON_2/usgs_extract_data/digitized'),
    REPO_ROOT / 'data' / 'digitized',
]

ANNUAL_KEYWORDS = ['annual', 'yearly', 'year']

ANNUAL_COLUMNS = [
    'doc_id', 'page_number', 'table_index', 'site_name',
    'json_latitude', 'json_longitude',
    'year', 'year_type', 'peak_date',
    'peak_discharge_cfs', 'peak_gage_height_ft',
    'mean_discharge_cfs', 'total_runoff_acre_ft',
    'discharge_unit', 'quality_flag', 'notes',
    'watersource_name', 'actual_latitude', 'actual_longitude',
    'inferred_latitude', 'inferred_longitude',
    'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
]

MONTHLY_COLUMNS = [
    'doc_id', 'page_number', 'table_index', 'site_name',
    'json_latitude', 'json_longitude',
    'water_year', 'month', 'month_num',
    'max_discharge_cfs', 'min_discharge_cfs', 'mean_discharge_cfs',
    'total_runoff_acre_ft', 'discharge_unit', 'quality_flag', 'notes',
    'watersource_name', 'actual_latitude', 'actual_longitude',
    'inferred_latitude', 'inferred_longitude',
    'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
]

LOG_COLUMNS = [
    'doc_id', 'page_number', 'table_index', 'site_name',
    'actual_content', 'action', 'skip_reason', 'notes',
]

TOOL_SCHEMA = {
    "name": "record_page_extraction",
    "description": (
        "Record the streamflow data extracted from one page. Include exactly one entry per TABLE "
        "chunk on the page (or one entry per station for multi-station tables)."
    ),
    "input_schema": {
        "type": "object",
        "required": ["tables"],
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["table_index", "site_name", "actual_content", "action"],
                    "properties": {
                        "table_index": {
                            "type": "integer",
                            "description": "0-based index of the table chunk among all chunks on the page",
                        },
                        "site_name": {"type": "string"},
                        "batch_metadata_row": {
                            "type": ["integer", "null"],
                            "description": (
                                "Index of the batch metadata row (from the '## Batch metadata' list "
                                "in the input) whose station best matches this table. Null if none "
                                "clearly matches."
                            ),
                        },
                        "actual_content": {
                            "type": "string",
                            "enum": [
                                "annual", "monthly", "annual_and_monthly",
                                "daily", "non_streamflow", "unreadable",
                            ],
                        },
                        "action": {
                            "type": "string",
                            "enum": [
                                "extracted_to_annual", "extracted_to_monthly", "extracted_to_both",
                                "skipped_daily", "skipped_not_streamflow", "skipped_unreadable",
                                "skipped_too_complex",
                            ],
                        },
                        "skip_reason": {"type": "string"},
                        "notes": {"type": "string"},
                        "annual_rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "year": {"type": ["integer", "null"]},
                                    "year_type": {"type": "string"},
                                    "peak_date": {"type": "string"},
                                    "peak_discharge_cfs": {"type": ["number", "null"]},
                                    "peak_gage_height_ft": {"type": ["number", "null"]},
                                    "mean_discharge_cfs": {"type": ["number", "null"]},
                                    "total_runoff_acre_ft": {"type": ["number", "null"]},
                                    "discharge_unit": {"type": "string"},
                                    "quality_flag": {"type": "string"},
                                    "notes": {"type": "string"},
                                    "json_latitude": {"type": ["number", "null"]},
                                    "json_longitude": {"type": ["number", "null"]},
                                },
                            },
                        },
                        "monthly_rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "water_year": {"type": ["integer", "null"]},
                                    "month": {"type": "string"},
                                    "month_num": {"type": ["integer", "null"]},
                                    "max_discharge_cfs": {"type": ["number", "null"]},
                                    "min_discharge_cfs": {"type": ["number", "null"]},
                                    "mean_discharge_cfs": {"type": ["number", "null"]},
                                    "total_runoff_acre_ft": {"type": ["number", "null"]},
                                    "discharge_unit": {"type": "string"},
                                    "quality_flag": {"type": "string"},
                                    "notes": {"type": "string"},
                                    "json_latitude": {"type": ["number", "null"]},
                                    "json_longitude": {"type": ["number", "null"]},
                                },
                            },
                        },
                    },
                },
            }
        },
    },
}


def find_digitized_root():
    for p in CANDIDATE_ROOTS:
        if p.is_dir():
            return p
    sys.exit('ERROR: digitized data root not found — is the external drive plugged in?')


def html_table_to_markdown(html):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        return html
    out = []
    for t in tables:
        for r in t.find_all('tr'):
            cells = [c.get_text(' ', strip=True) for c in r.find_all(['th', 'td'])]
            out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def chunk_to_block(chunk, idx):
    content = chunk.get('content', '') or ''
    if '<table' in content.lower():
        return f"### Chunk {idx} [TABLE]\n{html_table_to_markdown(content)}\n"
    return f"### Chunk {idx} [TEXT]\n{content.strip()}\n"


def load_page_chunks(digitized_root, doc_id, page):
    json_path = digitized_root / doc_id / f'page_{page}' / f'{doc_id}_page_{page}.json'
    if not json_path.is_file():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get('result', {}).get('chunks', [])


def build_user_message(doc_id, page, chunks, batch_rows):
    parts = [f"## Page identifier\n- doc_id: {doc_id}\n- page_number: {page}\n"]
    parts.append("## Batch metadata (LLM-extracted; noisy)")
    for i, r in enumerate(batch_rows):
        parts.append(
            f"- row {i}: watersource_name={r.get('watersource_name','')!r}, "
            f"actual_lat={r.get('actual_latitude','')!r}, "
            f"actual_lon={r.get('actual_longitude','')!r}, "
            f"temporal={r.get('temporal_resolution','')!r}, "
            f"dates={r.get('dates_of_recording','')!r}, "
            f"units={r.get('units_of_measurement','')!r}"
        )
    parts.append("\n## Page chunks\n")
    for idx, ch in enumerate(chunks):
        parts.append(chunk_to_block(ch, idx))
    return '\n'.join(parts)


# Generic geographic words that match between almost any two station names —
# excluded from similarity scoring so distinguishing proper nouns dominate.
_NAME_STOPWORDS = {
    'near', 'calif', 'california', 'ca', 'the', 'of', 'at', 'and',
    'below', 'above', 'nr', 'co', 'county', 'creek', 'river', 'cr',
}


def _name_tokens(name):
    toks = (name or '').lower().replace(',', ' ').replace('.', ' ').split()
    return {t for t in toks if t and t not in _NAME_STOPWORDS}


def pick_batch_metadata(batch_rows, site_name):
    """Match the model's site_name to the best batch metadata row.

    Returns {} (blank metadata) when there is no meaningful token overlap,
    rather than guessing — per the extraction instructions.
    """
    if not batch_rows:
        return {}
    site_tokens = _name_tokens(site_name)

    def score(r):
        return len(_name_tokens(r.get('watersource_name')) & site_tokens)

    best = max(batch_rows, key=score)
    return best if score(best) > 0 else {}


def call_api(client, model, system_text, user_text, max_tokens=32000):
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "record_page_extraction"},
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == 'max_tokens':
        raise RuntimeError(
            f"response truncated at max_tokens={max_tokens} — tool JSON incomplete, page skipped for retry"
        )
    for block in response.content:
        if block.type == 'tool_use' and block.name == 'record_page_extraction':
            return block.input, response.usage
    raise RuntimeError("No tool_use block in API response")


def ensure_header(path, columns):
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def write_rows(lock, paths, log_rows, annual_rows, monthly_rows):
    with lock:
        ensure_header(paths['log'], LOG_COLUMNS)
        ensure_header(paths['annual'], ANNUAL_COLUMNS)
        ensure_header(paths['monthly'], MONTHLY_COLUMNS)
        with open(paths['log'], 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            for r in log_rows:
                w.writerow({k: r.get(k, '') for k in LOG_COLUMNS})
        with open(paths['annual'], 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=ANNUAL_COLUMNS)
            for r in annual_rows:
                w.writerow({k: r.get(k, '') for k in ANNUAL_COLUMNS})
        with open(paths['monthly'], 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=MONTHLY_COLUMNS)
            for r in monthly_rows:
                w.writerow({k: r.get(k, '') for k in MONTHLY_COLUMNS})


def process_page(client, model, system_text, digitized_root, doc_id, page, batch_rows, lock, paths, dry_run=False):
    chunks = load_page_chunks(digitized_root, doc_id, page)
    if chunks is None:
        return {'status': 'no_json', 'doc_id': doc_id, 'page': page}
    user_text = build_user_message(doc_id, page, chunks, batch_rows)
    if dry_run:
        return {'status': 'dry_run', 'doc_id': doc_id, 'page': page, 'chars': len(user_text)}

    try:
        result, usage = call_api(client, model, system_text, user_text)
    except Exception as e:
        return {'status': 'api_error', 'doc_id': doc_id, 'page': page, 'error': str(e)}

    log_rows, annual_out, monthly_out = [], [], []
    meta_keys = (
        'watersource_name', 'actual_latitude', 'actual_longitude',
        'inferred_latitude', 'inferred_longitude',
        'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
    )
    annual_keys = (
        'json_latitude', 'json_longitude', 'year', 'year_type', 'peak_date',
        'peak_discharge_cfs', 'peak_gage_height_ft', 'mean_discharge_cfs',
        'total_runoff_acre_ft', 'discharge_unit', 'quality_flag', 'notes',
    )
    monthly_keys = (
        'json_latitude', 'json_longitude', 'water_year', 'month', 'month_num',
        'max_discharge_cfs', 'min_discharge_cfs', 'mean_discharge_cfs',
        'total_runoff_acre_ft', 'discharge_unit', 'quality_flag', 'notes',
    )

    for t in result.get('tables', []):
        site_name = t.get('site_name', '') or ''
        idx = t.get('batch_metadata_row')
        if (isinstance(idx, int) and 0 <= idx < len(batch_rows)
                and _name_tokens(batch_rows[idx].get('watersource_name')) & _name_tokens(site_name)):
            # Trust the model's pick only if the station names share a real token —
            # guards against matching e.g. "Bear River" to a lone "Georgetown Creek" row.
            meta = batch_rows[idx]
        else:
            meta = pick_batch_metadata(batch_rows, site_name)
        meta_fields = {k: meta.get(k, '') for k in meta_keys}
        common = {
            'doc_id': doc_id, 'page_number': page,
            'table_index': t.get('table_index'),
            'site_name': site_name,
        }
        log_rows.append({
            **common,
            'actual_content': t.get('actual_content', ''),
            'action': t.get('action', ''),
            'skip_reason': t.get('skip_reason', ''),
            'notes': t.get('notes', ''),
        })
        for r in (t.get('annual_rows') or []):
            row = {**common, **{k: r.get(k, '') for k in annual_keys}, **meta_fields}
            annual_out.append(row)
        for r in (t.get('monthly_rows') or []):
            row = {**common, **{k: r.get(k, '') for k in monthly_keys}, **meta_fields}
            monthly_out.append(row)

    if not log_rows:
        log_rows.append({
            'doc_id': doc_id, 'page_number': page, 'table_index': '',
            'site_name': '', 'actual_content': 'non_streamflow',
            'action': 'skipped_not_streamflow',
            'skip_reason': 'no tables on page', 'notes': '',
        })

    write_rows(lock, paths, log_rows, annual_out, monthly_out)
    return {
        'status': 'ok', 'doc_id': doc_id, 'page': page,
        'tables': len(result.get('tables', [])),
        'annual_rows': len(annual_out), 'monthly_rows': len(monthly_out),
        'usage': {
            'input_tokens': getattr(usage, 'input_tokens', 0),
            'output_tokens': getattr(usage, 'output_tokens', 0),
            'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0),
            'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', 0),
        },
    }


def load_processed_pages(log_path):
    if not log_path.is_file():
        return set()
    seen = set()
    with open(log_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            seen.add((r['doc_id'], r['page_number']))
    return seen


def load_candidates(meta_path, digitized_root):
    pages = defaultdict(list)
    with open(meta_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('water_type', '').strip().lower() != 'stream discharge':
                continue
            res = row.get('temporal_resolution', '').strip().lower()
            if not any(k in res for k in ANNUAL_KEYWORDS):
                continue
            doc_id = row['id']
            page = row['page_number']
            json_path = digitized_root / doc_id / f'page_{page}' / f'{doc_id}_page_{page}.json'
            if not json_path.is_file():
                continue
            pages[(doc_id, page)].append({
                'doc_id': doc_id, 'page_number': page,
                'watersource_name': row.get('watersource_name', '').strip(),
                'actual_latitude': row.get('actual_latitude', '').strip(),
                'actual_longitude': row.get('actual_longitude', '').strip(),
                'inferred_latitude': row.get('inferred_latitude', '').strip(),
                'inferred_longitude': row.get('inferred_longitude', '').strip(),
                'temporal_resolution': row.get('temporal_resolution', '').strip(),
                'dates_of_recording': row.get('dates_of_recording', '').strip(),
                'units_of_measurement': row.get('units_of_measurement', '').strip(),
            })
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--model', default='claude-sonnet-4-6')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    load_dotenv(SCRIPT_DIR / '.env')
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if (not api_key or api_key.startswith('sk-ant-...')) and not args.dry_run:
        sys.exit('ERROR: ANTHROPIC_API_KEY not set in .env')

    digitized_root = find_digitized_root()
    print(f"Digitized root: {digitized_root}")

    system_text = PROMPT_PATH.read_text(encoding='utf-8')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        'log': OUTPUT_DIR / 'extraction_log.csv',
        'annual': OUTPUT_DIR / 'annual_streamflow.csv',
        'monthly': OUTPUT_DIR / 'monthly_streamflow.csv',
    }

    processed = load_processed_pages(paths['log'])
    print(f"Already processed: {len(processed)} pages")

    candidates = load_candidates(META_PATH, digitized_root)
    print(f"Candidate pages found: {len(candidates)}")

    todo = [(k, v) for k, v in candidates.items() if k not in processed]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"To process: {len(todo)} pages, workers={args.workers}, model={args.model}, dry_run={args.dry_run}")

    client = None if args.dry_run else Anthropic(api_key=api_key)
    lock = threading.Lock()
    n_done = n_err = 0
    err_summary = defaultdict(int)
    usage_totals = defaultdict(int)

    def task(entry):
        (doc_id, page), batch_rows = entry
        return process_page(client, args.model, system_text, digitized_root,
                            doc_id, page, batch_rows, lock, paths, dry_run=args.dry_run)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(task, e) for e in todo]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                n_err += 1
                err_summary['exception'] += 1
                print(f"  EXCEPTION: {e}", file=sys.stderr)
                continue
            if r['status'] == 'ok':
                n_done += 1
                for k, v in (r.get('usage') or {}).items():
                    usage_totals[k] += v
                if n_done % 25 == 0 or n_done <= 5:
                    print(f"  [{n_done}] {r['doc_id']}/{r['page']}: "
                          f"{r['tables']} tables, {r['annual_rows']} annual, {r['monthly_rows']} monthly")
            elif r['status'] == 'dry_run':
                n_done += 1
                if n_done <= 3:
                    print(f"  DRY [{n_done}] {r['doc_id']}/{r['page']}: prompt={r['chars']} chars")
            else:
                n_err += 1
                err_summary[r['status']] += 1
                print(f"  ERROR ({r['status']}): {r.get('doc_id')}/{r.get('page')}: {r.get('error','')}", file=sys.stderr)

    print(f"\nDone. Processed: {n_done}, Errors: {n_err}")
    if err_summary:
        for k, v in err_summary.items():
            print(f"  {k}: {v}")
    if usage_totals:
        print(f"Usage totals: {dict(usage_totals)}")


if __name__ == '__main__':
    main()
