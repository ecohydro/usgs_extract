# Streamflow Extraction Instructions

This is the complete guide for a single extraction session — read it fully before opening any files.

**Project context:** USGS historical water data digitization project. Pages were digitized by Reducto.ai, which produces a JSON per page containing text and table chunks in page order. A separate LLM (Phi-4) extracted metadata per page into a batch CSV; that metadata is noisy. The batch you are processing was filtered to pages likely to contain annual stream discharge data, but many pages will turn out to contain something else — log everything regardless.

---

## Setup

**Batch number** is given in the prompt that invoked you. Use it in all filenames below.

**Digitized data root:** check which path exists and use it as `DIGITIZED_ROOT`:
1. `data/digitized/` (relative to repo root — Annette's machine)
2. `/Volumes/AHILTON_2/usgs_extract_data/digitized/` (Anna's Mac, external drive)

If neither exists, stop and say so — the external drive may need to be plugged in.

**Input:** `data/analysis/streamflow/annual/batch_{batch number}.csv` — one row per metadata entry. A single `(doc_id, page_number)` may appear in multiple rows when more than one metadata entry exists for that page.

**Output files** (all in `data/analysis/streamflow/annual/`):
- `annual_streamflow_batch_{batch number}.csv`
- `monthly_streamflow_batch_{batch number}.csv`
- `extraction_log_batch_{batch number}.csv`

**Append, don't overwrite.** To resume a partial session, check the last entry in `extraction_log_batch_{batch number}.csv` and skip any `(doc_id, page_number)` pairs already logged.

**Write after every page.** After finishing all tables for a `(doc_id, page_number)`, append all results to the output files before moving to the next page. Never accumulate in memory.

---

## Step 1 — Group the Batch by Page

Read the batch CSV. Group rows by `(doc_id, page_number)`. Process each unique page exactly once, carrying all its batch rows into Step 5.

---

## Step 2 — Read the JSON

For each unique `(doc_id, page_number)`, open:

`{DIGITIZED_ROOT}/{doc_id}/page_{page_number}/{doc_id}_page_{page_number}.json`

The JSON contains `result.chunks` — a list of chunks in page order. Work through them sequentially:

- **Text chunks** contain station descriptions, table titles, and sometimes explicit coordinates (e.g., `Lat 34°25'35", long 119°05'15"`). Note the full station description and any coordinates as you encounter them — they describe the table(s) that follow.
- **Table chunks** contain the HTML table. Each is one table to process through Steps 3–5.

---

## Step 3 — Classify the Table and Write One Log Row

For each table chunk, determine what kind of data it contains and write **one row** to `extraction_log_batch_{batch number}.csv` with these columns:

| Column | Description |
|--------|-------------|
| `doc_id` | Document ID |
| `page_number` | Page number within document |
| `table_index` | 0-based index of this table chunk among all chunks on the page |
| `site_name` | Full station description from the JSON text chunk immediately preceding this table, exactly as printed — include geographic qualifiers, tributary references, county, etc. (e.g., `"Santa Paula Creek below Sisar Creek, near Santa Paula, Ventura County, Calif."`) |
| `actual_content` | Classification label (see table below) |
| `action` | What was done with this table (see table below) |
| `skip_reason` | Brief explanation if action is a skip; blank otherwise |
| `notes` | Anything else worth recording about this table |

**Classification:**

| `actual_content` | Meaning | `action` |
|-----------------|---------|----------|
| `annual` | Contains annual streamflow measurements | `extracted_to_annual` |
| `monthly` | Contains monthly streamflow measurements | `extracted_to_monthly` |
| `annual_and_monthly` | Contains both (e.g. monthly table with an annual total row) | `extracted_to_both` |
| `daily` | Contains daily measurements | `skipped_daily` |
| `non_streamflow` | Unrelated content or non-discharge data | `skipped_not_streamflow` |
| `unreadable` | Table structure cannot be determined | `skipped_unreadable` |

**When in doubt, skip.** A skipped row in the log is recoverable; a wrong row in the data is not. If a table looks like streamflow but is too messy to parse reliably, set `actual_content` to the best label you can assign and use `action = skipped_too_complex`.

---

## Step 4 — Write Data Rows (JSON-extracted content only)

For tables classified as `annual`, `monthly`, or `annual_and_monthly`, write data rows to the appropriate output file(s). At this step, include only what is explicitly present in the JSON — **never infer or guess a value**; leave the field blank if the information is not in the table.

### Annual rows → `annual_streamflow_batch_{batch number}.csv`

Write one row per year present in the table.

| Column | Description |
|--------|-------------|
| `doc_id` | Document ID |
| `page_number` | Page number within document |
| `table_index` | 0-based index of this table chunk |
| `site_name` | Full station description from the JSON text chunk (same as log) |
| `json_latitude` | Latitude explicitly stated in the JSON text chunk, converted to decimal degrees (e.g. `Lat 34°25'35"` → `34.4264`) |
| `json_longitude` | Longitude explicitly stated in the JSON text chunk, converted to decimal degrees; west longitudes are negative (e.g. `long 119°05'15"` → `-119.0875`) |
| `year` | Year as integer, exactly as it appears in the table |
| `year_type` | `water_year` or `calendar_year` as labeled in the table; blank if unlabeled |
| `peak_date` | Date of peak discharge — `YYYY-MM-DD` if year is determinable, `MM-DD` otherwise |
| `peak_discharge_cfs` | Peak discharge after stripping quality prefix (cfs-family units; see Units) |
| `peak_gage_height_ft` | Gage height at peak in feet |
| `mean_discharge_cfs` | Mean discharge for the year in cfs |
| `total_runoff_acre_ft` | Total runoff in acre-feet |
| `discharge_unit` | `cfs` for any cfs-family unit (second-feet, sec.-ft., sec-ft, second feet, cfs, cubic feet per second, ft³/s); otherwise the exact unit as written (e.g. `acre-feet`, `gpm`); `unknown` if indeterminate |
| `quality_flag` | Quality prefix stripped from the value: `e` (estimated — also `E`, `#`, `*`), `a` (ice/backwater — also `A`), `c` (revised — also `C`), `o` (zero/trace — set value to 0.0), or footnote letter described in `notes`; blank if none |
| `notes` | Ambiguous values, non-cfs units with raw value, anything unusual |

**Annual totals within monthly tables:** if a monthly table includes a "The year" or "ANNUAL" summary row, extract it here as `year_type = water_year` with `mean_discharge_cfs` populated.

**Two-column layouts:** some pages print two year-ranges side by side — treat each side as separate rows.

**Footnote rows:** rows beginning with a letter (`a`, `b`, `c`…) followed by explanatory text are footnotes, not data — skip them.

### Monthly rows → `monthly_streamflow_batch_{batch number}.csv`

Write one row per month present in the table.

| Column | Description |
|--------|-------------|
| `doc_id` | Document ID |
| `page_number` | Page number within document |
| `table_index` | 0-based index of this table chunk |
| `site_name` | Full station description from the JSON text chunk |
| `json_latitude` | Latitude explicitly stated in the JSON text chunk, converted to decimal degrees |
| `json_longitude` | Longitude explicitly stated in the JSON text chunk, converted to decimal degrees; west longitudes are negative |
| `water_year` | Water year as integer — ending year (e.g. `1910` for Oct 1909–Sep 1910; `1909-10` → `1910`) |
| `month` | Full month name (e.g. `October`; convert any abbreviations) |
| `month_num` | 1–12 in water-year order (October = 1, November = 2, … September = 12) |
| `max_discharge_cfs` | Monthly maximum discharge in cfs |
| `min_discharge_cfs` | Monthly minimum discharge in cfs |
| `mean_discharge_cfs` | Monthly mean discharge in cfs |
| `total_runoff_acre_ft` | Monthly runoff in acre-feet |
| `discharge_unit` | `cfs` for any cfs-family unit; otherwise exact unit as written; `unknown` if indeterminate (same rules as annual) |
| `quality_flag` | Quality prefix stripped from the value; same rules and values as annual |
| `notes` | Anything unusual |

**"The year" / "ANNUAL" rows** in monthly tables: extract to annual output, not monthly.

**Empty cells** (`-`, `--`, blank): leave the measurement column blank; no quality flag.

**`o` values:** treat as `0.0`, set `quality_flag = o`.

---

## Step 5 — Append Matched Metadata Columns

For every row written to the annual or monthly output in Step 4, append the columns from the best-matching batch row for this `(doc_id, page_number)`.

**Matching:** if the page has one batch row, use it. If it has multiple, match by station name similarity between the batch row's `watersource_name` and the `site_name` extracted from the JSON. If no clear match, append blanks for all metadata columns and note it.

Append these columns exactly as they appear in the batch CSV — no modifications:

| Column | Source |
|--------|--------|
| `watersource_name` | Batch row (LLM-extracted station name; useful for joining back to metadata) |
| `actual_latitude` | Batch row |
| `actual_longitude` | Batch row |
| `inferred_latitude` | Batch row |
| `inferred_longitude` | Batch row |
| `temporal_resolution` | Batch row |
| `dates_of_recording` | Batch row |
| `units_of_measurement` | Batch row |

The final rows in both output files will have six latitude/longitude columns: `json_latitude`, `json_longitude`, `actual_latitude`, `actual_longitude`, `inferred_latitude`, `inferred_longitude`.

---

## Common Edge Cases

**"Do." in station name column:** USGS shorthand for "same as above" — use the most recent non-"Do." station name.

**Comma-formatted numbers:** `"5, 610"` is a Reducto artifact for `5610` — strip internal spaces.

**Multi-level headers:** Reducto renders these as 2–3 header rows. Read all together to identify columns.

**No data rows:** table has headers but no numeric data → `non_streamflow`, `skipped_not_streamflow`, `skip_reason = no data rows`.

---

## Session Summary

At the end of each session, report:

```
Session summary (batch NNN):
  Pages processed: N
  Tables reviewed: N
  Extracted to annual: N rows
  Extracted to monthly: N rows
  Skipped daily: N
  Skipped not streamflow: N
  Skipped too complex: N
  Skipped unreadable: N
```

Each batch is self-contained. To resume a partial session, re-run the same prompt — it will append to existing batch files and skip pages already in the log.
