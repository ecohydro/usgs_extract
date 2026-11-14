# Metadata simplification and evaluation: start here

Handoff from Annette's session of 2026-11-14 to Anna. Paste the prompt below into a new Claude Code
session opened in the repo root (on the `evaluation` branch).

Before starting:
- Work in your own clone of the repo on the `evaluation` branch (not in Annette's Dropbox folder).
- Flowchart of the current pipeline: https://claude.ai/artifact/QkJD9xN6CAbXrrFyrJk1kA
  (private until Annette shares it with you).
- Data: the git repo holds the code and most metadata files, but several large or sensitive files
  are git-ignored and exist only on Annette's machine. If you are missing any of the files below,
  ask Annette; she can transfer them (a ~600 MB bundle without page images) or you can regenerate
  some of them, which is slow.

## Data you need (as of 2026-11-14)

In git (you get these with the `evaluation` branch):
`data/metadata/main_metadata.csv` (Phi-4 rows), `data/analysis/processed_metadata.parquet`,
`data/analysis/location_validation/locations.csv` and the tier files, `data/analysis/page_types.csv`,
`data/analysis/table_level_metadata.csv` / `_published.csv`, `data/analysis/table_headers.csv`,
`data/analysis/table_value_counts.csv`, `data/analysis/table_resolution.csv`,
`data/analysis/manual_review/`, graded samples in `data/analysis/date_parsing/`, the CA GNIS file,
WBD/HUC files, NWIS CA site lists, PLSS caches.

NOT in git (git-ignored) - needed for the evaluation and the Phi-4 pilot:

| File | Size | Needed for | If missing |
|---|---|---|---|
| `data/digitized/{doc}/page_{N}/` (page PNG, Reducto JSON, table CSVs) | tens of GB | grading against page images; the pilot prompt input | Rebuild from your May files with `code/01_digitization/05_final_data_organization/organize_data.py` (expects `Data_Files/ReductCSVs/`, `Data_Files/ReductJson/`, `Data_Files/UpdatedDataJan/` + `UpdatedDataDec/`, `Data_Files/cleaned_metadata_final - Copy.csv`) |
| `data/catalog/restored_tables_catalog.csv` (+ `.gpkg`, `catalog_summary.csv`) | 228 MB | the per-table product being evaluated | Ask Annette (rebuilding needs most rows below) |
| `data/analysis/page_blocks.parquet` | 138 MB | numbered tables per page (pilot prompt input) | `code/02_inventory/11_page_blocks_cache.py` (needs `data/digitized`) |
| `data/analysis/table_dates.csv`, `table_row_dates.parquet` | 41 MB | table dates | `14_table_dates.py` (needs `page_blocks.parquet`) |
| `data/analysis/entry_dates.csv` | 18 MB | which Phi-4 rows are tied to which tables | `15_entry_dates.py` |
| `data/analysis/table_quality.csv` | 6 MB | digitization-quality flag | `17_table_quality.py` |
| `data/analysis/tribal_lands/` (esp. `tribal_screen_pages.csv`) | <1 MB | `paper_filters.drop_tribal()`; every figure/statistic | Ask Annette. Sensitive: lists pages on tribal lands; never publish or share |
| `data/analysis/spatial/tribal_lands/` (Census AIANNH) | 23 MB | re-running the tribal screen only | re-download (see `12_tribal_lands_screen.py`) |
| `data/analysis/spatial/gnis/*_Features_20210825.txt` (non-CA states), `data/analysis/nwis_sites/by_state/` | ~40 MB | re-running out-of-state locations only | re-fetched by `10_out_of_state.py` |

Not needed: `data/withheld_tribal/` (1.2 GB copies of withheld pages, do not transfer),
`archive/`, `data/hydroshare/`, `data/analysis/nwis_sites/ca_groundwater_sites_raw.csv`.

Python: Annette's machine uses Python 3.13 (no conda) with pandas, numpy, geopandas, pyogrio,
pyarrow, matplotlib, seaborn, jupyter/nbconvert.

## Prompt

```
I'm Anna Boser, collaborator on Annette Hilton's USGS historical water data digitization project
(repo: usgs_extract, branch: evaluation). Read CLAUDE.md first: it holds the full history, and the
session logs at the end (Oct 28 - Nov 14, 2026) are the source of truth. Read the "Session Log -
November 14, 2026" notes in full, and code/05_evaluation/detection/NOTES.md for my earlier
evaluation work.

Goal of this session: SIMPLIFY the metadata pipeline where possible and EVALUATE the per-table
metadata product. New evaluation code goes in code/05_evaluation/metadata/, outputs in
data/analysis/evaluation/metadata/. Manuscript target: submission by end of 2026. Annette and I
write the paper; you supply analyses, code, checks and supporting material, not manuscript text
unless asked.

How the metadata is built today (flowchart: https://claude.ai/artifact/QkJD9xN6CAbXrrFyrJk1kA):
1. Reducto digitized each page (JSON + one CSV per table).
2. Phi-4 (UCSB cluster) read each page with a table and returned one row per "water source" it saw:
   data/metadata/main_metadata.csv, 109,440 rows on 64,074 pages. Rows carry NO table number.
   Prompt: code/01_digitization/04_metadata_extraction/01_extract_metadata.py (check it against
   the production prompt in dissertation SI S1 / Yibo's api_run.py before relying on it).
3. Annette and Claude rebuilt locations and dates with deterministic code:
   - locations per metadata row (code/01_digitization/04_metadata_extraction/location_validation/,
     output data/analysis/location_validation/locations.csv): the row's LLM text is only a
     pointer; evidence must be printed on the row's own page (printed coords, well numbers, NWIS
     station number/name, PLSS, GNIS, B118 basin), plus a consensus check;
   - dates per table first (code/02_inventory/14_table_dates.py, table_dates.py), then passed to
     rows (15_entry_dates.py: row / table / page-union / LLM-if-printed);
   - page types incl. report lists (10_page_types.py); tribal-lands screen (12, 16).
4. code/02_inventory/build_table_level_metadata.py joins rows to tables (each table takes the
   location of a row tied to it, else the best-located row on its page), then
   filter_published_tables.py and 18_build_catalog.py -> data/catalog/restored_tables_catalog.csv.
The weak point: step 4's row-to-table join has never been evaluated, and it is complicated
because Phi-4 rows don't map 1:1 to tables (23,557 pages have several tables but one row;
groundwater pages average 8.5 rows; ~27,000 rows could not be tied to a single table).

Two options are on the table; help me decide and then do the work:
A. Evaluate the current per-table product: a stratified sample (~100 tables: 1 row/1 table,
   1 row/many tables, many rows/1 table, many/many), graded against the page images for water
   type, date and location. Gives the paper an "accuracy of the published catalog" number.
B. Re-run Phi-4 on the cluster (Annette can get access) with a per-table prompt: number the
   tables in the input (data/analysis/page_blocks.parquet, table_idx = CSV table number, matches
   99.8% of pages); ask for one JSON object per table with its sites (names, station/well numbers
   copied verbatim), location text and any coordinates copied verbatim (no inferred coordinates),
   date range plus the words it came from, water type, measured quantity, units, record type. Our
   location/date code then verifies these, and the row-to-table join goes away. Start with a
   ~200-page pilot (stratified by page structure, including pages Annette already graded) and
   compare against the current catalog before any full re-run.
Annette's session leaned toward doing A regardless (it is also the baseline for judging B) and
piloting B.

Rules Annette set (keep them):
- Evidence for a row's location/date comes from its own page by default; cross-page evidence
  only through validated, labeled rules.
- Build every figure and statistic from the corrected data (catalog / table-level files /
  processed_metadata.parquet), never from raw LLM fields; fix errors once at the source
  (page types, table_dates.py rules, data/analysis/manual_review/earliest_records_review.csv),
  then rebuild.
- Check a record before calling it an error; say plainly when something is a guess, and how
  many examples a claim rests on.
- When counting restored sites, err toward counting unique sites, but be defensible.
- Ask before committing or pushing.

Open items paused on Nov 14 (don't change them without asking Annette): the site-definition
tolerance in code/02_inventory/21_site_definitions.py (0.5 / 1 / 2 km; review sample
data/analysis/site_definitions/split_station_review.csv awaits Annette's check); syncing 2 newly
withheld tribal pages (13_copy_withheld_tribal.py); re-running figures other than Figure 1.

First steps: I do NOT have Annette's Dropbox folder; I have the Reducto files and other inputs
from May 2026. Check which files in the "Data you need" table of
code/05_evaluation/metadata/START_HERE.md exist on this machine and list what is missing. Do not
start long regeneration runs without asking me: some files are faster to get from Annette, and the
tribal-lands files must come from her. Then confirm the Python environment, summarize your
understanding back to me in a few lines, and propose a concrete plan for A and for the B pilot.
```
