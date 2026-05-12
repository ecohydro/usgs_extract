# Streamflow Extraction Instructions

## Purpose

Extract historical streamflow measurements from USGS digitized table CSVs into three unified output files. This document is the complete guide for a single extraction session — read it fully before opening any CSV files.

**Project context:** USGS historical water data digitization project. The source CSVs are Reducto.ai digitizations of scanned USGS Water Supply Papers and related publications. Metadata was extracted by an LLM (Phi-4) and is noisy — temporal resolution labels in particular are unreliable.

---

## Batching

The ~6,900 files to process are split into 46 batches of 150 files each. Each Claude Code session handles one batch. The batch number for this session is given in the prompt that invoked you.

**Input file:** `data/analysis/streamflow/annual/batch_{batch number}.csv`
Each row is one `(doc_id, page_number)` pair to process, with pre-filled metadata fields.

**Output files** (all in `data/analysis/streamflow/annual/`):

| File | Contents |
|------|----------|
| `annual_streamflow_batch_{batch number}.csv` | One row per station-year extracted this session |
| `monthly_streamflow_batch_{batch number}.csv` | One row per station-month extracted this session |
| `extraction_log_batch_{batch number}.csv` | One row per CSV file reviewed this session — every file, whether extracted or skipped |

**Append, don't overwrite.** If this session is resuming a partial run, append to the existing batch files rather than overwriting them. To find where a previous partial run left off, check the last `doc_id` + `page_number` in `extraction_log_batch_{batch number}.csv` and skip any pairs already logged.

A separate concatenation script merges all batches into final files once all sessions are done — you do not need to worry about that.

---

## Source data background

All files in this workflow come from rows in `data/metadata/main_metadata.csv` where:
- `water_type` == `stream discharge` (case-insensitive)
- `temporal_resolution` contains any of: `annual`, `yearly`, `year`

Because the LLM metadata extraction is noisy, many files are **not** actually annual streamflow — they may be monthly tables, daily tables, water quality tables, or unrelated content. When a file turns out to contain monthly data, extract it to `monthly_streamflow_batch_{batch number}.csv` as normal. The extraction log records what was actually found in every file regardless.

### Finding the digitized data root

Before processing any files, determine which machine you are on by checking which of these paths exists:

1. `data/digitized/` (relative to the repo root — Annette's Dropbox setup)
2. `/Volumes/AHILTON_2/usgs_extract_data/digitized/` (Anna's Mac with external drive)

Use whichever exists as `DIGITIZED_ROOT`. If neither exists, stop and report that the digitized data cannot be found — the external drive may need to be plugged in.

### Processing each pair

Each batch session is given a list of `(doc_id, page_number)` pairs to process. For each pair:

1. Metadata row is at `data/metadata/main_metadata.csv` — match on `id` == doc_id AND `page_number` == page_number. Key fields: `watersource_name`, `inferred_latitude`, `inferred_longitude`, `actual_latitude`, `actual_longitude`, `dates_of_recording`, `temporal_resolution`, `units_of_measurement`.
2. CSV files are at `{DIGITIZED_ROOT}/{doc_id}/page_{page_number}/{doc_id}_page_{page_number}_table*.csv`. A page may have multiple table CSVs — review each one separately and write a log row for each.

**Coordinates:** use `actual_latitude` / `actual_longitude` if non-empty; otherwise use `inferred_latitude` / `inferred_longitude`. If both are empty, leave latitude/longitude blank.

**Write after every file.** After finishing all CSVs for a `(doc_id, page_number)` pair, immediately append the results to the batch output files before moving to the next pair. Do not accumulate rows in memory and write them all at the end — if the session is interrupted, only the current file's work should be lost.

---

## Step 1 — Identify Table Type

Open the CSV. Look at the first 5–10 rows and the column headers. Assign one of these `actual_content` labels:

| Label | How to recognize it |
|-------|---------------------|
| `annual_peaks` | One row per year; columns include water year, a date (of peak), discharge, and often gage height |
| `annual_means` | One row per year (or one row per water year + calendar year); columns include Mean discharge and often total runoff in acre-feet |
| `monthly_means_grid` | Years in the first column, month names (Oct., Nov., … Sept.) as the remaining columns; one row per water year |
| `monthly_stats` | Month names in the first column (October, November…); columns are Max, Min, Mean discharge + runoff; one block per water year |
| `daily` | Day numbers 1–31 in the first column, month names as remaining columns |
| `water_quality` | Columns dominated by chemical parameters: pH, conductance, dissolved solids, ions (Ca, Mg, Na, etc.); discharge may appear as just one column |
| `spot_measurements` | Individual dated discharge measurements (date + gage height + discharge), not summarized into monthly or annual means |
| `multi_station` | Multiple station names appear as column headers or row labels; discharge values for several rivers on one table |
| `non_streamflow` | Content unrelated to streamflow: sediment grain size, biological data, bibliography, station inventory, etc. |
| `unreadable` | Reducto output is so garbled the table structure cannot be determined |

**If a page has multiple CSVs:** each gets its own log row with its own `actual_content` label and action. They may be different types.

---

## Step 2 — Decide Action

| actual_content | Action |
|---------------|--------|
| `annual_peaks` | Extract → `annual_streamflow.csv` |
| `annual_means` | Extract → `annual_streamflow.csv` |
| `monthly_means_grid` | Extract → `monthly_streamflow.csv` |
| `monthly_stats` | Extract → `monthly_streamflow.csv` |
| `daily` | Skip → log as `skipped_daily` |
| `water_quality` | Skip → log as `skipped_not_streamflow` |
| `spot_measurements` | Skip → log as `skipped_not_streamflow` |
| `multi_station` | Skip → log as `skipped_too_complex` (unless small and clean enough to parse by hand — use judgment) |
| `non_streamflow` | Skip → log as `skipped_not_streamflow` |
| `unreadable` | Skip → log as `skipped_unreadable` |

**When in doubt, skip.** A skipped row in the log is recoverable. A wrong row in the data is not. If a table is a genuine type but is too messy to parse reliably (e.g., headers are scrambled, rows are merged, years are ambiguous), use `skipped_too_complex` and describe it in `skip_reason`.

---

## Step 3 — Extract Data

### Annual peaks (`annual_peaks` → `annual_streamflow.csv`)

Classic format: `Water year | Date | Gage height | Discharge`

- One output row per data row in the table (skip header rows and footnote rows)
- `year`: the water year value (integer, e.g., `1938`)
- `year_type`: `water_year`
- `peak_date`: the date of peak discharge — convert to ISO format `YYYY-MM-DD` if the year is clear from context; if only month+day given without year, write `MM-DD` and note in `notes` that year is inferred from water year
- `peak_discharge_cfs`: numeric value after stripping quality prefix; unit must be cfs-family (see Units section below)
- `peak_gage_height_ft`: numeric value if gage height column present, else blank
- `mean_discharge_cfs`: blank
- `total_runoff_acre_ft`: blank
- `discharge_unit`: normalized unit string (see Units section)

**Two-column layouts:** some annual peaks pages print two sets of years side by side (e.g., years 1904–1923 in left columns, 1924–1943 in right columns). Treat each side as separate rows — do not skip the right-side columns.

**Footnotes:** rows at the bottom of the table that begin with `a`, `b`, `c`, etc. followed by explanatory text are footnotes, not data — skip them.

---

### Annual means (`annual_means` → `annual_streamflow.csv`)

Common format: multi-level headers with water year + calendar year blocks; columns include Mean discharge and Total runoff.

- One output row per year per water-year/calendar-year block
- `year`: the year value (integer)
- `year_type`: `water_year` or `calendar_year` as labeled in the table
- `peak_date`: blank
- `peak_discharge_cfs`: blank
- `peak_gage_height_ft`: blank
- `mean_discharge_cfs`: value from Mean column
- `total_runoff_acre_ft`: value from acre-feet column if present, else blank
- `discharge_unit`: normalized unit string

**Multi-level headers from Reducto:** Reducto often renders multi-level headers as 2–3 header rows before the data. Read all header rows together to understand which column maps to which measurement. Look for "Mean", "Discharge", "Second-feet", "cfs", "Runoff", "Acre-feet" across the header rows.

**"The year" / "ANNUAL" rows:** some tables include a summary row at the end of each year block labeled "The year" or "ANNUAL" with the mean for the full year. Extract these as `year_type = water_year` rows with `mean_discharge_cfs` populated.

---

### Monthly means grid (`monthly_means_grid` → `monthly_streamflow.csv`)

Format: year in first column, month names (Oct.–Sept.) as remaining columns.

- Melt the wide table: each cell produces one output row
- `water_year`: value from the first column of that row (integer)
- `month`: full month name, e.g., `October` (convert abbreviation)
- `month_num`: 1–12 (October = 1, November = 2, … September = 12 — water year ordering)
- `mean_discharge_cfs`: cell value after stripping quality prefix
- `max_discharge_cfs`, `min_discharge_cfs`, `total_runoff_acre_ft`: blank
- Skip "The year" / "ANNUAL" column — that is an annual mean, not a monthly value

**`o` values:** in some tables `o` means zero or trace flow. Treat as `0.0` and set `quality_flag = o`.

**Empty cells:** blank or `-` or `--` means no measurement — leave `mean_discharge_cfs` blank, no quality flag needed.

---

### Monthly stats (`monthly_stats` → `monthly_streamflow.csv`)

Format: month names in first column; Max, Min, Mean, Runoff columns; one water-year block per table (sometimes multiple blocks stacked).

- One output row per month per water-year block
- `water_year`: from the year header row immediately above each block (e.g., `1909 -10` → water year `1910`)
- `month`: full month name from first column
- `month_num`: 1–12 (water year ordering)
- `max_discharge_cfs`: Max column value
- `min_discharge_cfs`: Min column value
- `mean_discharge_cfs`: Mean column value
- `total_runoff_acre_ft`: acre-feet column value if present
- Skip "The year" / "The period" summary rows at the bottom of each block

**Water year label parsing:** USGS labels water years as `1909 -10` or `1909-10` meaning October 1909 – September 1910, so the water year integer is `1910` (the ending calendar year).

**Accuracy column:** some older tables have an accuracy rating column (A, B, C, D). Ignore it — do not extract.

---

## Units

**CFS-family** (all identical — normalize `discharge_unit` to `cfs`):
- second-feet, sec.-ft., sec-ft, second feet, cfs, cubic feet per second, ft³/s

**Non-CFS units:** if the discharge column uses acre-feet, gallons per minute, or other units:
- Still extract the numeric value
- Set `discharge_unit` to the actual unit string (e.g., `acre-feet`, `gpm`)
- Leave `mean_discharge_cfs` or `peak_discharge_cfs` blank
- Put the value in `notes` as: `value={X} unit={unit}`

If units are completely unclear from headers and metadata, set `discharge_unit = unknown` and note it.

---

## Quality Flags

USGS data quality prefixes appear attached to numeric values (e.g., `a 230`, `e 45.3`, `*1,390`). Strip the prefix from the value and record it in `quality_flag`:

| Prefix | Meaning | `quality_flag` value |
|--------|---------|----------------------|
| `e` or `E` | estimated | `e` |
| `a` or `A` | affected by ice or backwater | `a` |
| `#` | estimated (alternate) | `e` |
| `*` | estimated (alternate) | `e` |
| `c` or `C` | revised | `c` |
| `o` | zero or trace | `o` (set value to `0`) |
| `-` or `--` | missing | leave value blank, no flag |
| `b`, `d`, `f`, `g`, etc. | footnote reference — look at footnote to understand | describe in `notes` |

If a cell has no prefix, leave `quality_flag` blank.

---

## Output Schemas

All output files go in `data/analysis/streamflow/annual/` with the batch number in the filename.

### `annual_streamflow_batch_{batch number}.csv`

```
doc_id, page_number, table_file, watersource_name, latitude, longitude,
year, year_type, peak_date, peak_discharge_cfs, peak_gage_height_ft,
mean_discharge_cfs, total_runoff_acre_ft, discharge_unit, quality_flag, notes
```

### `monthly_streamflow_batch_{batch number}.csv`

```
doc_id, page_number, table_file, watersource_name, latitude, longitude,
water_year, month, month_num, max_discharge_cfs, min_discharge_cfs,
mean_discharge_cfs, total_runoff_acre_ft, discharge_unit, quality_flag, notes
```

### `extraction_log_batch_{batch number}.csv`

```
doc_id, page_number, table_file, metadata_label, actual_content,
action, skip_reason, notes
```

`action` values: `extracted_to_annual`, `extracted_to_monthly`, `skipped_daily`, `skipped_not_streamflow`, `skipped_too_complex`, `skipped_unreadable`

---

## Common Edge Cases

**Table spans multiple water years in one CSV:** extract all years — one output row per year (annual) or per month (monthly).

**"Do." / "do." in station name column:** USGS shorthand meaning "same as above." Use the most recent non-"Do." station name in that column.

**Comma-formatted numbers:** `"5, 610"` or `"1, 060"` are Reducto artifacts for `5610` and `1060` — strip spaces and commas, parse as integer.

**Merged rows:** occasionally Reducto merges two rows into one cell (e.g., day 27 and 28 together). Note in `notes` and extract what you can; leave the ambiguous value blank.

**Page covers multiple stations:** if the page is genuinely a multi-station table (`multi_station`), skip it unless there are only 2–3 stations and the structure is clean enough to parse each station as a separate set of rows.

**No data rows (all blank):** some tables have headers but no numeric data (e.g., a placeholder page). Log as `actual_content = non_streamflow`, `action = skipped_not_streamflow`, `skip_reason = no data rows`.

---

## Session Summary Format

At the end of each session, report:

```
Session summary (batch NNN):
  Processed: N files
  Extracted to annual: N rows
  Extracted to monthly: N rows
  Skipped daily: N
  Skipped not streamflow: N
  Skipped too complex: N
  Skipped unreadable: N
  Last doc_id processed: XXXXX  page: NNN
```

Each batch is self-contained — there is no handoff between sessions. To resume a partial session, re-run the same batch prompt; the session will append to the existing batch output files and skip pairs already in the log.
