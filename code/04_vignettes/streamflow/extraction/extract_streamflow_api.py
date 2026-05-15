"""
extract_streamflow_api.py — extract annual and monthly streamflow data from pre-digitized USGS
pages using the Claude Message Batches API.

Filters main_metadata.csv for stream-discharge pages whose temporal_resolution is annual or
monthly and whose JSON exists, sends each page's content to Claude as one batch request, and
appends the classified/extracted results to three output CSVs.

Workflow (subcommands):
    submit     Build requests for all unprocessed pages, create a batch, save batch_state.json.
    status     Poll the active batch and print request counts.
    retrieve   Download results of the (ended) active batch, write CSVs, clear batch_state.json.
    run        submit (or reuse active batch) -> poll until ended -> retrieve. One command.

Options:
    --limit N            Only include the first N unprocessed pages (for test batches).
    --model NAME         Model id (default claude-sonnet-4-6).
    --poll-seconds S     Poll interval for `run`/`status` waiting (default 30).
    --dry-run            (submit/run) Build requests but do not call the API.

Subset filters (apply to `submit` and `run`, for test slices of the dataset):
    --resolution         annual | monthly | both (default both)
    --ca-only            only pages with coordinates inside the California bounding box
    --require-coords     actual | inferred | any — only pages that have such lat-lon metadata

Resumes by skipping pages already logged in extraction_log.csv. A failed or truncated page is
not logged, so a later `submit` picks it up again. The output CSVs are shared across all slices;
the resume log keeps slices from reprocessing each other's pages.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PROMPT_PATH = SCRIPT_DIR / 'streamflow_extraction_instructions.md'
META_PATH = REPO_ROOT / 'data' / 'metadata' / 'main_metadata.csv'
OUTPUT_DIR = REPO_ROOT / 'data' / 'analysis' / 'streamflow'
STATE_PATH = OUTPUT_DIR / 'batch_state.json'

CANDIDATE_ROOTS = [
    Path('/Volumes/AHILTON_2/usgs_extract_data/digitized'),
    REPO_ROOT / 'data' / 'digitized',
]

# temporal_resolution substrings that mark a page's resolution. 'year' also catches 'yearly';
# 'month' also catches 'monthly'.
ANNUAL_RES_KEYWORDS = ['annual', 'year', 'yearly']
MONTHLY_RES_KEYWORDS = ['month', 'monthly']
ALL_RES_KEYWORDS = ANNUAL_RES_KEYWORDS + MONTHLY_RES_KEYWORDS

# California bounding box (generous) — matches scope_filters.py
CA_BBOX = dict(lat_lo=32.3, lat_hi=42.1, lon_lo=-124.6, lon_hi=-114.0)

MAX_TOKENS = 32000

# Batch API pricing for claude-sonnet-4-6, USD per million tokens (50% off standard rates).
# Update if the model changes.
BATCH_PRICING = {
    'input': 1.50,
    'output': 7.50,
    'cache_write': 1.875,
    'cache_read': 0.15,
}


def compute_cost(usage):
    """usage: dict with keys input/output/cache_write/cache_read (token counts) -> USD."""
    return sum(usage.get(k, 0) / 1_000_000 * rate for k, rate in BATCH_PRICING.items())

ANNUAL_COLUMNS = [
    'doc_id', 'page_number', 'table_index', 'site_name',
    'json_latitude', 'json_longitude',
    'year', 'year_type', 'peak_date',
    'peak_discharge', 'peak_gage_height',
    'mean_discharge', 'total_runoff',
    'discharge_unit', 'runoff_unit', 'quality_flag', 'notes',
    'watersource_name', 'actual_latitude', 'actual_longitude',
    'inferred_latitude', 'inferred_longitude',
    'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
]

MONTHLY_COLUMNS = [
    'doc_id', 'page_number', 'table_index', 'site_name',
    'json_latitude', 'json_longitude',
    'water_year', 'month', 'month_num',
    'max_discharge', 'min_discharge', 'mean_discharge',
    'total_runoff', 'discharge_unit', 'runoff_unit', 'quality_flag', 'notes',
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
                            "description": (
                                "1-based table number on the page — copy the N from the "
                                "[TABLE N] label of the chunk this entry describes"
                            ),
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
                                    "peak_discharge": {"type": ["number", "null"]},
                                    "peak_gage_height": {"type": ["number", "null"]},
                                    "mean_discharge": {"type": ["number", "null"]},
                                    "total_runoff": {"type": ["number", "null"]},
                                    "discharge_unit": {"type": "string"},
                                    "runoff_unit": {"type": "string"},
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
                                    "max_discharge": {"type": ["number", "null"]},
                                    "min_discharge": {"type": ["number", "null"]},
                                    "mean_discharge": {"type": ["number", "null"]},
                                    "total_runoff": {"type": ["number", "null"]},
                                    "discharge_unit": {"type": "string"},
                                    "runoff_unit": {"type": "string"},
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


# ----------------------------------------------------------------------------- helpers

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


def chunk_to_block(chunk, idx, table_ord):
    """Format one chunk. table_ord is the 1-based table number, or None for text chunks."""
    content = chunk.get('content', '') or ''
    if table_ord is not None:
        return f"### Chunk {idx} [TABLE {table_ord}]\n{html_table_to_markdown(content)}\n"
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
    table_ord = 0
    for idx, ch in enumerate(chunks):
        content = ch.get('content', '') or ''
        if '<table' in content.lower():
            table_ord += 1
            parts.append(chunk_to_block(ch, idx, table_ord))
        else:
            parts.append(chunk_to_block(ch, idx, None))
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
            if not any(k in res for k in ALL_RES_KEYWORDS):
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


# ----------------------------------------------------------------------------- subset filters

def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _row_coords(row, which):
    """which in {'actual', 'inferred'} -> (lat, lon) or None."""
    lat = _to_float(row.get(f'{which}_latitude'))
    lon = _to_float(row.get(f'{which}_longitude'))
    if lat is None or lon is None:
        return None
    return lat, lon


def page_resolution_kinds(batch_rows):
    """Set of {'annual','monthly'} the page's metadata rows indicate."""
    kinds = set()
    for r in batch_rows:
        res = (r.get('temporal_resolution') or '').lower()
        if any(k in res for k in ANNUAL_RES_KEYWORDS):
            kinds.add('annual')
        if any(k in res for k in MONTHLY_RES_KEYWORDS):
            kinds.add('monthly')
    return kinds


def page_has_coords(batch_rows, which):
    """which in {'actual', 'inferred', 'any'} — True if any metadata row has such coords."""
    for r in batch_rows:
        if which in ('actual', 'any') and _row_coords(r, 'actual'):
            return True
        if which in ('inferred', 'any') and _row_coords(r, 'inferred'):
            return True
    return False


def page_in_ca(batch_rows):
    """True if any metadata row's coords (actual preferred, else inferred) fall in the CA bbox."""
    for r in batch_rows:
        c = _row_coords(r, 'actual') or _row_coords(r, 'inferred')
        if c is None:
            continue
        lat, lon = c
        if (CA_BBOX['lat_lo'] <= lat <= CA_BBOX['lat_hi']
                and CA_BBOX['lon_lo'] <= lon <= CA_BBOX['lon_hi']):
            return True
    return False


def apply_filters(todo, resolution, ca_only, require_coords):
    """Narrow a [(key, batch_rows), ...] list by the submit-time slice options."""
    out = []
    for key, batch_rows in todo:
        if resolution != 'both' and resolution not in page_resolution_kinds(batch_rows):
            continue
        if ca_only and not page_in_ca(batch_rows):
            continue
        if require_coords and not page_has_coords(batch_rows, require_coords):
            continue
        out.append((key, batch_rows))
    return out


def ensure_header(path, columns):
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def output_paths():
    return {
        'log': OUTPUT_DIR / 'extraction_log.csv',
        'annual': OUTPUT_DIR / 'annual_streamflow.csv',
        'monthly': OUTPUT_DIR / 'monthly_streamflow.csv',
    }


def write_rows(paths, log_rows, annual_rows, monthly_rows):
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


# custom_id <-> (doc_id, page) ----------------------------------------------------------

def make_custom_id(doc_id, page):
    return f"{doc_id}__{page}"


def parse_custom_id(cid):
    doc_id, _, page = cid.partition('__')
    return doc_id, page


# ----------------------------------------------------------------------------- requests

def build_requests(system_text, digitized_root, todo, model):
    """Build batch request dicts for the given list of ((doc_id, page), batch_rows)."""
    requests = []
    skipped_no_json = []
    for (doc_id, page), batch_rows in todo:
        chunks = load_page_chunks(digitized_root, doc_id, page)
        if chunks is None:
            skipped_no_json.append((doc_id, page))
            continue
        user_text = build_user_message(doc_id, page, chunks, batch_rows)
        requests.append({
            "custom_id": make_custom_id(doc_id, page),
            "params": {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": [{
                    "type": "text", "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }],
                "tools": [TOOL_SCHEMA],
                "tool_choice": {"type": "tool", "name": "record_page_extraction"},
                "messages": [{"role": "user", "content": user_text}],
            },
        })
    return requests, skipped_no_json


# ----------------------------------------------------------------------------- result parsing

def tool_input_to_rows(doc_id, page, tool_input, batch_rows):
    """Convert one page's tool_use input into (log_rows, annual_rows, monthly_rows)."""
    meta_keys = (
        'watersource_name', 'actual_latitude', 'actual_longitude',
        'inferred_latitude', 'inferred_longitude',
        'temporal_resolution', 'dates_of_recording', 'units_of_measurement',
    )
    annual_keys = (
        'json_latitude', 'json_longitude', 'year', 'year_type', 'peak_date',
        'peak_discharge', 'peak_gage_height', 'mean_discharge',
        'total_runoff', 'discharge_unit', 'runoff_unit', 'quality_flag', 'notes',
    )
    monthly_keys = (
        'json_latitude', 'json_longitude', 'water_year', 'month', 'month_num',
        'max_discharge', 'min_discharge', 'mean_discharge',
        'total_runoff', 'discharge_unit', 'runoff_unit', 'quality_flag', 'notes',
    )

    log_rows, annual_out, monthly_out = [], [], []
    for t in tool_input.get('tables', []):
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
            annual_out.append({**common, **{k: r.get(k, '') for k in annual_keys}, **meta_fields})
        for r in (t.get('monthly_rows') or []):
            monthly_out.append({**common, **{k: r.get(k, '') for k in monthly_keys}, **meta_fields})

    if not log_rows:
        log_rows.append({
            'doc_id': doc_id, 'page_number': page, 'table_index': '',
            'site_name': '', 'actual_content': 'non_streamflow',
            'action': 'skipped_not_streamflow',
            'skip_reason': 'no tables on page', 'notes': '',
        })
    return log_rows, annual_out, monthly_out


def extract_tool_input(message):
    """Return the record_page_extraction tool input from a Message, or raise."""
    if getattr(message, 'stop_reason', None) == 'max_tokens':
        raise RuntimeError(f"response truncated at max_tokens={MAX_TOKENS}")
    for block in message.content:
        if block.type == 'tool_use' and block.name == 'record_page_extraction':
            return block.input
    raise RuntimeError("no record_page_extraction tool_use block in response")


# ----------------------------------------------------------------------------- state

def load_state():
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return None


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def clear_state():
    if STATE_PATH.is_file():
        STATE_PATH.unlink()


# ----------------------------------------------------------------------------- commands

def cmd_submit(client, args, system_text, digitized_root):
    if load_state() is not None:
        sys.exit(f"ERROR: an active batch already exists ({STATE_PATH}). "
                 f"Run `status`/`retrieve`, or delete the state file to start over.")

    processed = load_processed_pages(output_paths()['log'])
    candidates = load_candidates(META_PATH, digitized_root)
    todo = sorted((k, v) for k, v in candidates.items() if k not in processed)
    n_unprocessed = len(todo)

    todo = apply_filters(todo, args.resolution, args.ca_only, args.require_coords)
    n_matched = len(todo)
    active = [f"resolution={args.resolution}"]
    if args.ca_only:
        active.append("ca_only")
    if args.require_coords:
        active.append(f"require_coords={args.require_coords}")
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Candidate pages: {len(candidates)} | already processed: {len(processed)} | "
          f"unprocessed: {n_unprocessed}")
    print(f"Filters [{', '.join(active)}] -> {n_matched} unprocessed pages match this slice")
    if args.limit is not None:
        print(f"  --limit {args.limit}: submitting first {len(todo)} of those {n_matched}")

    requests, skipped = build_requests(system_text, digitized_root, todo, args.model)
    if skipped:
        print(f"  ({len(skipped)} pages skipped — JSON missing on disk)")
    if not requests:
        print("Nothing to submit.")
        return
    if args.dry_run:
        print(f"DRY RUN: would submit {len(requests)} requests. Example custom_id: {requests[0]['custom_id']}")
        return

    batch = client.messages.batches.create(requests=requests)
    state = {
        'batch_id': batch.id,
        'submitted_at': datetime.now(timezone.utc).isoformat(),
        'n_requests': len(requests),
        'model': args.model,
        'filters': {
            'resolution': args.resolution,
            'ca_only': args.ca_only,
            'require_coords': args.require_coords,
            'limit': args.limit,
        },
    }
    save_state(state)
    print(f"Submitted batch {batch.id} with {len(requests)} requests.")
    print(f"State saved to {STATE_PATH}. Use `status` to poll, `retrieve` when ended.")


def cmd_status(client, args):
    state = load_state()
    if state is None:
        sys.exit("No active batch (no batch_state.json).")
    batch = client.messages.batches.retrieve(state['batch_id'])
    print(f"Batch {batch.id}")
    print(f"  processing_status: {batch.processing_status}")
    print(f"  request_counts:    {batch.request_counts}")
    print(f"  submitted:         {state.get('submitted_at')}")
    return batch


def _process_results(client, state, batch):
    """Stream batch results, write CSVs, return summary counts."""
    digitized_root = find_digitized_root()
    candidates = load_candidates(META_PATH, digitized_root)
    paths = output_paths()

    n_ok = n_fail = 0
    n_log = n_annual = n_monthly = 0
    failures = []
    usage = defaultdict(int)

    for entry in client.messages.batches.results(batch.id):
        cid = entry.custom_id
        doc_id, page = parse_custom_id(cid)
        res = entry.result
        if res.type != 'succeeded':
            n_fail += 1
            detail = getattr(res, 'error', res.type)
            failures.append((cid, res.type, str(detail)))
            continue
        # Succeeded API response — billed regardless of whether we can parse it.
        u = res.message.usage
        usage['input'] += getattr(u, 'input_tokens', 0) or 0
        usage['output'] += getattr(u, 'output_tokens', 0) or 0
        usage['cache_write'] += getattr(u, 'cache_creation_input_tokens', 0) or 0
        usage['cache_read'] += getattr(u, 'cache_read_input_tokens', 0) or 0
        try:
            tool_input = extract_tool_input(res.message)
        except Exception as e:
            n_fail += 1
            failures.append((cid, 'parse_error', str(e)))
            continue
        batch_rows = candidates.get((doc_id, page), [])
        log_rows, annual_out, monthly_out = tool_input_to_rows(doc_id, page, tool_input, batch_rows)
        write_rows(paths, log_rows, annual_out, monthly_out)
        n_ok += 1
        n_log += len(log_rows)
        n_annual += len(annual_out)
        n_monthly += len(monthly_out)

    return {
        'n_ok': n_ok, 'n_fail': n_fail,
        'n_log': n_log, 'n_annual': n_annual, 'n_monthly': n_monthly,
        'failures': failures,
        'usage': dict(usage),
    }


def cmd_retrieve(client, args):
    state = load_state()
    if state is None:
        sys.exit("No active batch (no batch_state.json).")
    batch = client.messages.batches.retrieve(state['batch_id'])
    if batch.processing_status != 'ended':
        sys.exit(f"Batch not ended yet (status: {batch.processing_status}). Use `status` to poll.")

    print(f"Retrieving results for batch {batch.id} ...")
    summary = _process_results(client, state, batch)
    print(f"\nPages written: {summary['n_ok']} | failed: {summary['n_fail']}")
    print(f"  log rows: {summary['n_log']} | annual rows: {summary['n_annual']} | "
          f"monthly rows: {summary['n_monthly']}")
    if summary['failures']:
        print(f"  {len(summary['failures'])} failed requests (not logged — will be retried on next submit):")
        for cid, kind, detail in summary['failures'][:20]:
            print(f"    {cid}: {kind} — {detail[:120]}")
        if len(summary['failures']) > 20:
            print(f"    ... and {len(summary['failures']) - 20} more")

    # cost + timing — useful for extrapolating a full-slice run
    usage = summary['usage']
    cost = compute_cost(usage)
    n_billed = summary['n_ok'] + sum(1 for f in summary['failures'] if f[1] == 'parse_error')
    print(f"\nTokens — input: {usage.get('input', 0):,} | output: {usage.get('output', 0):,} | "
          f"cache_write: {usage.get('cache_write', 0):,} | cache_read: {usage.get('cache_read', 0):,}")
    print(f"Cost — ${cost:.4f} total, ${cost / n_billed:.4f}/page "
          f"(batch pricing, {state.get('model', '?')})" if n_billed else f"Cost — ${cost:.4f} total")
    if getattr(batch, 'created_at', None) and getattr(batch, 'ended_at', None):
        proc = batch.ended_at - batch.created_at
        print(f"Batch processing time (created -> ended): {proc}")

    clear_state()
    print(f"Cleared {STATE_PATH}. Re-run `submit` to process any failed/remaining pages.")


def cmd_run(client, args, system_text, digitized_root):
    if load_state() is None:
        cmd_submit(client, args, system_text, digitized_root)
        if args.dry_run or load_state() is None:
            return
    state = load_state()
    print(f"Polling batch {state['batch_id']} every {args.poll_seconds}s ...")
    while True:
        batch = client.messages.batches.retrieve(state['batch_id'])
        counts = batch.request_counts
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {batch.processing_status} | {counts}")
        if batch.processing_status == 'ended':
            break
        time.sleep(args.poll_seconds)
    cmd_retrieve(client, args)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['submit', 'status', 'retrieve', 'run'])
    ap.add_argument('--limit', type=int, default=None, help='only first N unprocessed pages')
    ap.add_argument('--model', default='claude-sonnet-4-6')
    ap.add_argument('--poll-seconds', type=int, default=30)
    ap.add_argument('--dry-run', action='store_true')
    # submit-time subset filters (for test slices); apply to `submit` and `run`
    ap.add_argument('--resolution', choices=['annual', 'monthly', 'both'], default='both',
                    help='only pages whose metadata marks this temporal resolution')
    ap.add_argument('--ca-only', action='store_true',
                    help='only pages with coordinates inside the California bounding box')
    ap.add_argument('--require-coords', choices=['actual', 'inferred', 'any'], default=None,
                    help='only pages with actual / inferred / either lat-lon in the metadata')
    args = ap.parse_args()

    load_dotenv(SCRIPT_DIR / '.env')
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    need_key = not (args.command == 'submit' and args.dry_run)
    if need_key and (not api_key or api_key.startswith('sk-ant-...')):
        sys.exit('ERROR: ANTHROPIC_API_KEY not set in .env')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Anthropic(api_key=api_key) if api_key and not api_key.startswith('sk-ant-...') else None

    if args.command == 'submit':
        system_text = PROMPT_PATH.read_text(encoding='utf-8')
        digitized_root = find_digitized_root()
        cmd_submit(client, args, system_text, digitized_root)
    elif args.command == 'status':
        cmd_status(client, args)
    elif args.command == 'retrieve':
        cmd_retrieve(client, args)
    elif args.command == 'run':
        system_text = PROMPT_PATH.read_text(encoding='utf-8')
        digitized_root = find_digitized_root()
        cmd_run(client, args, system_text, digitized_root)


if __name__ == '__main__':
    main()
