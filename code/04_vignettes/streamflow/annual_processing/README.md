# Annual Streamflow Extraction

We are extracting historical streamflow measurements from ~6,900 digitized USGS table CSVs
into a unified dataset. These CSVs come from pages that the LLM metadata extraction labeled
as annual-resolution stream discharge. In practice they contain a mix of actual annual data,
mislabeled monthly and daily tables, water quality tables, and unrelated content.

Each Claude Code session reads a batch of 150 files, identifies what's actually in each one,
extracts any annual or monthly streamflow data it finds, and logs every file it reviewed.

## Files here
- `annual_streamflow_extraction_instructions.md` — complete instructions for each extraction session
- `generate_batches.py` — generates input batch CSVs (run once)
- `concatenate_batches.py` — merges all per-batch outputs into final files (run once when all sessions are done)

## Outputs (`data/analysis/streamflow/annual/`)
- `batch_NNN.csv` — input for each session (pre-built by generate_batches.py)
- `annual_streamflow_batch_NNN.csv` — annual data extracted by session NNN
- `monthly_streamflow_batch_NNN.csv` — monthly data found in annual-labeled files
- `extraction_log_batch_NNN.csv` — every file reviewed in session NNN, with actual content and action

Once all sessions are done, run the concatenation script to merge everything:

```
python code/04_vignettes/streamflow/annual_processing/concatenate_batches.py
```

This produces three final files in `data/analysis/streamflow/annual/`:
- `annual_streamflow.csv`
- `monthly_streamflow.csv`
- `extraction_log.csv`

## Starting an extraction session

Open a new Claude Code session (terminal or VSCode). Paste this prompt, replacing `NNN` with the batch number (001–046):

```
Read code/04_vignettes/streamflow/annual_processing/annual_streamflow_extraction_instructions.md
fully before doing anything else. Then process all (doc_id, page_number) pairs in
data/analysis/streamflow/annual/batch_NNN.csv. Write batch-numbered output files to
data/analysis/streamflow/annual/ — specifically:
  annual_streamflow_batch_NNN.csv
  monthly_streamflow_batch_NNN.csv
  extraction_log_batch_NNN.csv
Append to these files if they already exist (in case you are resuming a partial session).
At the end, print the session summary in the format specified in the instructions.
```

Sessions can run in parallel — each writes to its own batch-numbered files so there are no write conflicts. Check `extraction_log_batch_NNN.csv` to see where a partial session left off before resuming.
