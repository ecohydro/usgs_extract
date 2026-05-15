# Annual Streamflow Extraction

We extract historical streamflow measurements from digitized USGS page JSONs into a unified
dataset. Pages are filtered from `main_metadata.csv` to those the LLM metadata extraction labeled
as stream discharge at annual/yearly resolution. In practice they contain a mix of real annual
data, mislabeled monthly and daily tables, water-budget and sediment tables, and unrelated
content — every table is classified and logged regardless.

Extraction runs through the **Claude Message Batches API**: each candidate page becomes one batch
request; Claude classifies every table on the page and returns structured rows via a tool call.
The Batch API is ~50% cheaper than per-request calls and is not subject to the per-minute output
token rate limit.

## Files here
- `annual_streamflow_extraction_instructions.md` — the system prompt sent with every request
- `extract_streamflow_api.py` — the extraction script (submit / status / retrieve / run)
- `.env` — holds `ANTHROPIC_API_KEY` (gitignored; copy `.env.example` and fill in)

## Outputs (`data/analysis/streamflow/annual/`)
- `annual_streamflow.csv` — one row per year per annual table
- `monthly_streamflow.csv` — one row per month per monthly table
- `extraction_log.csv` — one row per table reviewed, with classification and action
- `batch_state.json` — transient; tracks the active batch between submit and retrieve

## Setup

1. Plug in the external drive (`/Volumes/AHILTON_2/`) — the digitized JSONs live there.
2. Put your Anthropic API key in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Get one at console.anthropic.com (separate from a Claude.ai subscription; billed per token).
3. Dependencies: `anthropic`, `python-dotenv`, `beautifulsoup4`.

## Running

The simplest path — submit, poll, and retrieve in one command:

```
python extract_streamflow_api.py run
```

Or step through manually:

```
python extract_streamflow_api.py submit      # build requests for unprocessed pages, create batch
python extract_streamflow_api.py status      # poll the active batch
python extract_streamflow_api.py retrieve     # download results once ended, write CSVs
```

Useful flags:
- `--limit N` — only the first N unprocessed pages (test batches)
- `--model NAME` — model id (default `claude-sonnet-4-6`)
- `--poll-seconds S` — poll interval for `run` (default 30)
- `--dry-run` — (with `submit`) build requests without calling the API

## Resuming

The pipeline is resumable at two levels:

- **Within a batch:** `submit` saves `batch_state.json`. `status`/`retrieve`/`run` operate on that
  active batch. `retrieve` clears the state file when done.
- **Across batches:** `submit` skips any `(doc_id, page_number)` already present in
  `extraction_log.csv`. Pages whose requests failed or were truncated are *not* logged, so a
  later `submit` picks them up again. Run `submit` repeatedly until no pages remain.

To start completely fresh, delete the three output CSVs and `batch_state.json`.

## How a page is processed

1. The page JSON's chunks are flattened — text chunks kept as-is, HTML table chunks rendered as
   Markdown and labeled `[TABLE 1]`, `[TABLE 2]`, … — and combined with the page's (noisy) batch
   metadata into one user message.
2. Claude classifies every table chunk and returns structured rows through the
   `record_page_extraction` tool (forced tool use, so output is always well-formed).
3. Each table produces one `extraction_log.csv` row; annual/monthly tables also produce data
   rows. Batch metadata columns are joined on by the model's `batch_metadata_row` pick, with a
   token-overlap sanity check that blanks the metadata rather than attaching a wrong station.

## Output schema conventions

- **`table_index`** is the 1-based table number on the page (first table = 1, second = 2, …);
  text chunks are not counted.
- **Faithful transcription, not normalization.** Measurement values (`peak_discharge`,
  `mean_discharge`, `total_runoff`, …) are recorded *exactly as printed* — no unit conversion at
  extraction time. `discharge_unit` and `runoff_unit` capture the unit *as printed in the table*
  (`second-feet`, `sec.-ft.`, `cfs`, `thousands of acre-feet`, …). Normalizing units and
  converting values is the job of downstream analysis code, not the extraction step.
- **`quality_flag`** holds a USGS quality prefix stripped off the value: `e` estimated, `a`
  ice/backwater, `c` revised, `o` zero/trace (value set to 0.0). Other footnote letters are
  described in `notes`.
- **Coordinates** appear in up to six columns: `json_latitude`/`json_longitude` are parsed from
  the page JSON's own text; `actual_*` and `inferred_*` come from the joined batch metadata.
- **Multi-station tables** produce one entry per station; entries share a `table_index` but are
  distinguished by `site_name`.
