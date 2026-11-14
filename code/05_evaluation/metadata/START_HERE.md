# Metadata simplification and evaluation: start here

Handoff from Annette's session of 2026-11-14 to Anna. Paste the prompt below into a new Claude Code
session opened in the repo root (on the `evaluation` branch).

Before starting:
- Data: most pipeline outputs are git-ignored (catalog, `page_blocks.parquet`, `table_dates.csv`,
  `entry_dates.csv`, tribal files). Work from the shared Dropbox copy of the repo, not a fresh clone.
- Flowchart of the current pipeline: https://claude.ai/artifact/QkJD9xN6CAbXrrFyrJk1kA
  (private until Annette shares it with you).

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

First steps: confirm the repo and data are present and which Python environment to use (on
Annette's machine it was Python 3.13 without conda); summarize your understanding back to me in a
few lines; then propose a concrete plan for A and for the B pilot.
```
