# Metadata extraction: pipeline and evaluation

This folder turns each digitized table into metadata rows, one row per location per table, and measures how
often those rows are right. Each page is read alone: the page text from Reducto, with its tables numbered,
goes to an open language model (Qwen3.6-27B FP8, run with vLLM 0.21 on one L40S GPU, reasoning off) with
the instructions in `prompts.py`. A script then drops any location value whose characters are not on the
page, applies two tidy rules, and picks the best location evidence by a fixed order (`schema.py`).

The evaluation (`evaluation/`) uses three sets of published tables drawn at random within four groups of
data type. Two validation sets of 100 tables were used to develop the instructions. A test set of 200
tables was scored once, after the pipeline was fixed. For every set, the answer key was drafted from the
page scan alone, following `evaluation/KEY_FORMAT.md`.

All outputs are in `data/analysis/evaluation/metadata/` (paths below are relative to it). Every setting is
in `defaults.py`. Decisions and known limits are in `NOTES.md`. `location_validation/` is separate, earlier
work on locations and is not part of this pipeline.

## The pipeline (this folder)

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1 | `01_full_run.py prepare`, `worker`, `reread` | The model reads every page with a published table (batch job `jobs/usgs_metadata_full.sh`); `reread` gives unreadable answers a second reading with longer limits | `full_run/pages.csv`, `full_run/pages_without_table.csv` (raw answers are not in git) |
| 2 | `02_build_metadata.py` | Builds the final table from the answers | `full_run/metadata_per_location.csv`, `full_run/metadata_run_summary.csv` |

Shared code: `defaults.py` (settings), `schema.py` (columns, allowed values, order of location evidence),
`page_input.py` (page text with numbered tables), `prompts.py` (instructions), `reader.py` (model loading
and answer parsing), `pipeline.py` (model answer to final rows: found-on-page check, tidy rules, location),
`tribal.py` (screen for pages on tribal lands, used when drawing the evaluation sets).

## The evaluation (`evaluation/`)

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1 | `01_draw_sets.py test` (then `validation`, `validation2`) | Draws a set; replaces tables on pages withheld for tribal data sovereignty | `{set}_set_tables.csv`, `{set}_set_frame_counts.csv` |
| 2 | `02_collect_drafts.py` | Writes the briefs for the key drafters (`DRAFTER_INSTRUCTIONS.md`) and collects their drafts; `review_server.py` is the page used to check key tables by hand | `key*/answer_key.csv`, `key/checked.csv` |
| 3 | `03_key_runoff_rule.py` | Applies one written key rule uniformly to all keys | `key*/answer_key.csv` |
| 4 | `04_model_reader.py {set}` | The model reads the pages of a set (needs a GPU) | `model_runs/{set}_v5/answers.jsonl` |
| 5 | `05_score.py {set} pipeline` | Scores the pipeline against the key (`scoring.py`) | `scores/{set}_pipeline_v5_long.csv`, `_summary.csv` |
| 6 | `06_ranges.py test_pipeline_v5` | 95% ranges by resampling tables within their group | `scores/test_pipeline_v5_ranges.csv` |
| 7 | `07_on_page_check.py` | Share of location values the found-on-page check drops | `on_page_check.csv`, `on_page_check_by_column.csv` |
| 8 | `08_paper_numbers.py` | Collects every number the paper cites | `paper_numbers.csv` |

The model steps (pipeline 1, evaluation 4) use `~/miniconda3/envs/vllm-rs/bin/python`; everything else
needs pandas (and geopandas to draw a set).

## Where each number in the paper comes from

`paper_numbers.csv` lists all of these with the file each was read from.

| Number | Script | Output file (column or row) |
|---|---|---|
| 200 test tables from 137 reports; 80 / 50 / 35 / 35 per group; at most three per report | `evaluation/01_draw_sets.py` | `test_set_tables.csv` (rows; `doc_id`; `stratum`) |
| Group weights: stream 84%, groundwater 8%, reservoir 4%, other 4% | `evaluation/01_draw_sets.py` | `test_set_tables.csv` (`stratum_tables`, each group's share of the total); also `test_set_frame_counts.csv` |
| 289 key rows | `evaluation/02_collect_drafts.py` | `key/answer_key.csv` (rows) |
| First 20 key tables checked by an author | `evaluation/review_server.py` | `key/checked.csv` |
| Two validation sets of 100 tables | `evaluation/01_draw_sets.py` | `validation_set_tables.csv`, `validation2_set_tables.csv` |
| Table M3: share found per column with 95% range; mean of the eight columns 91% (88 to 93) | `evaluation/05_score.py`, `06_ranges.py` | `scores/test_pipeline_v5_ranges.csv` (`found`, `low`, `high`; last row is the mean) |
| Number of locations 98% | same | same file, row `n_locations` |
| Construction date 81% of 19 rows | same | same file, row `construction_date`; the 19 is `n_page_gives` in `scores/test_pipeline_v5_summary.csv` |
| Start year 87% (81 to 93), end year 96% (92 to 99) | same | same file, rows `date_start_year`, `date_end_year` |
| Supplementary Table 2: scores by group of data type | `evaluation/05_score.py` | `scores/test_pipeline_v5_summary.csv` (`group`, `field`, `found`) |
| 3.5% of location values dropped by the found-on-page check (77 of 2,181) | `evaluation/07_on_page_check.py` | `on_page_check.csv` (row `both`); by kind of evidence in `on_page_check_by_column.csv` |
| Model, vLLM version, GPU, reasoning off | `defaults.py`, `reader.py` | settings, not an output |
| Instructions printed as a supplementary note | `prompts.py` | `V5` |
| Key format printed as a supplementary note | `evaluation/KEY_FORMAT.md` | |
| Full collection: pages, tables, rows, unreadable answers | `01_full_run.py`, `02_build_metadata.py` | `full_run/metadata_run_summary.csv` |

Not in git: copies of the page scans (`pages*/`), the raw model answers for every page (`full_run/in`,
`out`, `reread`), and the tribal screen outputs (`tribal/`, sensitive).
