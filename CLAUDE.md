# USGS Historical Water Data Digitization — Project Guide

## What This Project Is

This repository supports the publication of a workflow that digitizes over a century of historical water records from USGS archives. The core paper (Chapter 2 of Hilton's dissertation, `Hilton_Dissertation_USGSDig.docx`) is titled:

**"Over a century of California water data restored by machine learning"**

The project demonstrates three things:
1. AI (ML + LLMs) can rescue and catalogue decades of historical tabular data from scanned PDFs
2. The thousands of resulting files can be inventoried by location, date, and water type
3. The recovered data has real utility for hydrological research

The data (ultimately ~2.5 million measurements, 105,920 tables, 581 documents) will be made available to researchers on [HydroShare](https://www.hydroshare.org/) (hosted by CUAHSI).

---

## What Was Done (Summary from Chapter 2)

### Data Source
- USGS Publications Warehouse search for California water documents published 1879–1980
- Returned 1,042 documents; ~60% available as scanned PDFs
- Documents span water measurements from 1810–1980 (data predate some documents)

### Digitization Pipeline
1. **PDF → PNG preprocessing**: PDFs converted to grayscale single-channel `.png` files using Python PIL and OpenCV
2. **Table Detection**: Microsoft Table Transformer model (threshold = 0.8); recovers 79% of known tables, reduces false positives
3. **Data Digitization**:
   - Tested: PaddleOCR (92% cell accuracy but poor structure preservation)
   - Used: Reducto.ai (100% accuracy on 13 test tables; better structure)
   - Output: `.json` (full Reducto output) and `.csv` (structured table)
4. **Metadata Extraction**: LLM Phi-4 (open-source, run on UCSB HPC cluster) extracts location, dates, water category, and 13 metadata fields per page. Overall accuracy ~87%.

### Metadata Fields (in `cleaned_metadata_final - Copy.csv`)
`id, page_number, inferred_latitude, inferred_longitude, actual_latitude, actual_longitude, location, townships_ranges_sections, watersource_name, actual_county, inferred_county, dates_of_recording, temporal_resolution, units_of_measurement, water_type, keyterms`

### Water Categories
Stream Discharge, Groundwater, Reservoir, Irrigation, Springs, Precipitation, Water Quality, Not Water Related, Other

### Key Stats
- 105,920 tables across 63,175 pages from 581 documents
- ~2,494,166 estimated measurements
- 93% of data has lat/lon coordinates (actual or inferred)
- LLM accuracy: 87% overall; 76–99% by category; lowest for "County" (52–76%)

---

## Repository Structure (Current State)

```
usgs_extract/
├── CLAUDE.md                          ← this file
├── README.md                          ← target repo structure + quickstart for each pipeline step
├── Hilton_Dissertation_USGSDig.docx   ← Chapter 2 draft (primary reference)
│
├── manuscript/
│   └── figures/                       ← all publication figures output here by notebooks
│
├── code/
│   ├── 01_digitization/
│   │   ├── 01_download_and_preprocess/  ← COMPLETE
│   │   │   ├── 01_download.py
│   │   │   ├── 02_verify.py
│   │   │   └── 03_preprocess.py
│   │   ├── 02_table_detection/          ← COMPLETE (workflow); validation script MISSING
│   │   │   └── 01_detect.py
│   │   ├── 03_ocr/                      ← Reducto (final) + PaddleOCR (tested, not used)
│   │   └── 04_metadata_extraction/
│   │       ├── 01_extract_metadata.py   ← LLM (Phi-4) extraction via UCSB HPC API
│   │       └── 02_validate_accuracy.ipynb ← produces manuscript/figures/metadata_accuracy.png
│   ├── 03_inventory/                    ← data inventory notebooks (all manuscript figures)
│   │   ├── water_palette.py             ← shared colorblind-safe color mapping
│   │   ├── 00_data_prep.ipynb           ← clean main_metadata.csv → data/analysis/processed_metadata.parquet
│   │   ├── 01_measurements_by_type.ipynb ← Fig 7: measurements by water type × decade
│   │   ├── 02_spatial_overview.ipynb    ← Fig 8: USA map; Fig 9: CA map; Fig 10: CA by decade
│   │   ├── 03_santa_barbara.ipynb       ← Figs 11–18: SB county coverage + NWIS comparison
│   │   └── 04_santa_ynez.ipynb          ← Fig 19, Tables 7–9: Santa Ynez stream order analysis
│   └── 04_vignettes/                    ← future hydrological analyses (empty for now)
│
├── data/
│   ├── digitized/                       ← final organized output, one folder per doc/page
│   ├── metadata/
│   │   ├── main_metadata.csv            ← comprehensive metadata: LLM fields + USGS pub fields joined
│   │   └── metadata_key.txt             ← column descriptions
│   ├── analysis/
│   │   ├── csv_row_counts.csv           ← estimated measurement counts per page/table
│   │   ├── processed_metadata.parquet   ← output of 00_data_prep.ipynb (parsed dates, clean types)
│   │   └── spatial/
│   │       ├── ca_shapefile/ca_poly.shp
│   │       ├── sb_county_shapefile/sb_county_shp.shp
│   │       ├── usa_shapefile/CUSA_States.shp
│   │       ├── santaynezhuc8/santaynez.shp  ← Santa Ynez HUC8 watershed boundary
│   │       ├── rivers/StreamsandRivers_Clip.shp
│   │       ├── lakes/chap2map.gdb            ← National Wetlands Inventory (GDB format)
│   │       └── streamprox/streamprox_santaynez.txt ← pre-computed stream order proximity
│   │   └── nwis_sites/sb_county_usgs/       ← NWIS site ID reference files for SB county
│   └── digitization_intermediates/
│       ├── 01_download_and_preprocess/
│       │   └── publication_list.csv
│       ├── 02_table_detection/
│       │   └── validation/
│       │       ├── groundwater_table_pages.xlsx
│       │       ├── groundwater_table_pages_expanded.csv
│       │       (MISSING: validation run output — 79% recall figure not preserved)
│       └── 04_metadata_extraction/
│           ├── lm_extracted_metadata.csv
│           ├── USGS_publication_metadata.xlsx
│           └── validation/
│               ├── validation_curated_135.xlsx
│               ├── validation_random_135.xlsx
│               ├── validation_edge_cases_15.xlsx
│               ├── test_set_cat_grades.xlsx   ← per-category accuracy (input to 02_validate_accuracy.ipynb)
│               ├── test_set_indiv_grades.xlsx ← per-entry accuracy (input to 02_validate_accuracy.ipynb)
│               └── jsons/                     ← 260 per-page Reducto JSONs for validation sets
│
├── Data_Files/                        ← ALL digitized data lives here (Annette's side; pre-organization)
│   ├── cleaned_metadata_final - Copy.csv  ← raw main metadata (pre-join source)
│   ├── csv_row_counts.csv
│   ├── 2252_728 example/              ← reference example of target output format
│   ├── ReductCSVs/                    ← 623 document ID subfolders → .csv files
│   ├── ReductJson/                    ← 623 document ID subfolders → .json files
│   ├── UpdatedDataDec/                ← 353 document ID subfolders → .png files
│   └── UpdatedDataJan/                ← 287 document ID subfolders → .png files
│
├── Chapter_2_USGS_Digitization/       ← project documents, legacy code, manuscript drafts
│   ├── Manuscript/
│   ├── Yibo Results/                  ← LLM metadata extraction iterative runs
│   └── Literature-Data Review/
│
├── jupyter_notebooks/                 ← exploratory/working notebooks (not publication notebooks)
│   ├── working_datacrunch.ipynb       ← original data processing (superseded by 00_data_prep.ipynb)
│   ├── archived_notebooks/
│   └── [other exploratory notebooks]
│
└── legacy_ocr/                        ← original nbdev package (superseded by code/)
    └── usgs_extract/model.py
```

---

## Target Output Format (Step 1 Goal)

The final organized data should have one folder per document ID, with a subfolder per page where a table was detected:

```
organized_data/
└── {doc_id}/
    └── page_{N}/
        ├── {doc_id}_page_{N}.png       ← grayscale page image
        ├── {doc_id}_page_{N}.json      ← Reducto full output
        ├── {doc_id}_page_{N}_table1.csv ← digitized table(s)
        ├── {doc_id}_page_{N}_table2.csv ← (if multiple tables on page)
        └── {doc_id}_page_{N}_metadata.csv  ← rows from cleaned_metadata_final for this page
```

The `2252_728 example/` folder in `Data_Files/` shows the file naming convention. The metadata CSV per page is not yet generated — it requires filtering `cleaned_metadata_final - Copy.csv` by `id` and `page_number`.

**Key data gap**: PNGs are split across `UpdatedDataDec/` (353 doc folders) and `UpdatedDataJan/` (287 doc folders), while CSVs and JSONs are each in their own 623-folder directories. These need to be merged into the unified per-page structure.

---

## Planned Work (Four Steps)

### Step 1: Organize Final Data Output
- Write a script to reorganize all files into the unified `{doc_id}/page_{N}/` structure
- Extract metadata rows per page from `cleaned_metadata_final - Copy.csv` and save as individual CSVs
- Join USGS document-level metadata from `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` into each per-page metadata CSV — the `id` column in the main metadata maps directly to the `Publication ID` column in `USGS_ID.xlsx`, which provides the USGS URL, Index ID (series), title, year, author, and ~55 additional document-level fields
- Identify and document any doc IDs present in metadata but missing from data folders (or vice versa)

### Step 2: HydroShare Upload Organization
- Explore groupings of data for upload (e.g., by water_type + decade)
- Understand HydroShare size/file limits
- Create upload-ready copies of grouped data
- May need to compress or chunk large groups

### Step 3: Data Inventory (Notebooks) — COMPLETE
All publication figures are implemented in `code/03_inventory/`. Run notebooks in order (00 → 04); all figures save to `manuscript/figures/`.

- `00_data_prep.ipynb` — parses `data/metadata/main_metadata.csv`, cleans water types, parses dates, joins row counts; outputs `data/analysis/processed_metadata.parquet`
- `01_measurements_by_type.ipynb` — Fig 7: measurements by water type × decade
- `02_spatial_overview.ipynb` — Figs 8–10: USA and California maps
- `03_santa_barbara.ipynb` — Figs 11–18: SB county coverage + NWIS groundwater/stream discharge comparison
- `04_santa_ynez.ipynb` — Fig 19, Tables 7–9: Santa Ynez stream order analysis

Metadata accuracy figure (Fig 5) is in `code/01_digitization/04_metadata_extraction/02_validate_accuracy.ipynb`.

### Step 4: Hydrological Research Vignettes — FUTURE
`code/04_vignettes/` is reserved for future analyses that do hydrological research with the restored data. Two directions have been identified as high-priority; see the **Potential Hydrological Research Directions** section below for details.

---

## Potential Hydrological Research Directions

These are candidate scientific contributions that use the restored data for original hydrological analysis, not just data description. Both focus on stream discharge because it is by far the largest category (84,634 tables, ~2.2M measurements). Key metadata facts relevant to both directions:

- ~76% of stream discharge rows have CFS-family units (second-feet, sec.-ft., cfs, cubic feet per second — all the same unit and trivially normalized)
- The LLM extracted per-column unit mappings as dict strings (e.g., `{'Discharge': 'cfs', 'Gage_Height': 'feet'}`), which identify the relevant column before the CSV is opened
- ~50% of stream discharge rows (~38,000) are daily CFS records across 8,270 unique sites and 219 documents
- 17,557 rows have data starting at or before 1920 (83% in CFS units), spanning 5,098 unique sites

**Prerequisite for both:** The actual table CSVs (`Data_Files/ReductCSVs/`) must be accessible. First step once files are reachable: sample ~50 stream discharge CSVs to understand how Reducto rendered the classic USGS day-row × month-column table structure.

---

### Direction A: Pre-dam Hydrology

**Scientific motivation:** California's major dams fundamentally altered river hydrology, but pre-dam streamflow records are almost entirely absent from digital databases. NWIS stream records in Santa Barbara County don't begin until 1940; most other CA rivers are similar. Even a handful of pre-dam annual peak or mean flow values at a site would be novel and scientifically useful for flood frequency analysis, environmental flow standards, and dam impact studies.

**Key dam dates for California:**
- Gibraltar Dam (Santa Ynez River): 1920
- Jameson Reservoir (Santa Ynez): 1930
- Friant Dam (San Joaquin River): 1942
- Shasta Dam (Sacramento River): 1945
- Cachuma Reservoir (Santa Ynez): 1953
- Folsom Dam (American River): 1956
- Oroville Dam (Feather River): 1968

Scott Dam in Lake County and Cape Horn Dam in Mendocino County, which together are known as the Potter Valley Project. ** next important  dams to look into

**Why this dataset is promising:** 17,557 restored stream discharge rows start at or before 1920; 1,842 annual summary rows start before 1920 across 3,566 unique source sites. Annual tables (one row = one year's mean or peak) are the simplest format to parse and the most directly useful for this analysis.

**Challenges and open questions:**
- CSV files not currently locally accessible — need `Data_Files/` mounted or transferred
- USGS daily tables use a day-row × month-column grid; Reducto's CSV output for this structure is unknown until files are sampled
- Annual summary tables are simpler but still need column identification (headers vary: "Mean", "Annual discharge", "Second-feet", etc.)
- Some "pre-dam" records may be from diversions, canals, or regulated reaches — not natural flow; need to flag these
- Location precision: coordinates may be coarse (township/range inferred), making watershed assignment uncertain for some sites
- Need to check whether any restored site names match NWIS station IDs, enabling cross-validation during the 1940–1980 overlap period
- Verify which specific rivers are best represented in pre-1920 data before committing to a study area

**Suggested first steps:**
1. Filter metadata to stream discharge + CFS + annual temporal resolution + start year ≤ 1945
2. Sample 20–30 of those CSVs to understand column structure
3. Identify a focal river or small set of rivers with the most pre-dam site coverage
4. Build cross-validation using 1940–1980 overlap with NWIS to establish data quality

---

Outline: Clean Unified Streamflow Dataset
Step 1: Scope the problem
Before writing any parsers, understand what we're dealing with:

Filter main_metadata.csv to water_type = Stream Discharge
Tally how many pages/tables exist and their distribution by temporal_resolution (daily, monthly, annual) and units_of_measurement
This tells us which table structures are most common and where to focus effort
From what we already know: ~84,000 stream discharge tables, ~76% in CFS-family units, ~50% daily records.

Step 2: Classify table structures
The tables do not all look the same. From what we've seen already, there are at least three distinct layouts:

Type	Structure	Example
Annual peaks	One row per year, date + discharge	doc 3034/page 199
Monthly means	Water year rows, month columns	doc 15/page 29
Daily (classic USGS)	Day rows (1–31) × month columns	doc 553/page 267
We need to sample ~50–100 stream discharge CSVs across different documents to confirm these are the main types and identify any others. The LLM temporal_resolution field is our best guide, but it's noisy.

Step 3: Write a parser for each table type
Each structure needs its own parser to produce long-form rows. The hardest problem for each:

Annual peaks — simplest; just read row by row, handle two-column printed layouts (already split correctly in the CSV)
Monthly means — melt the month columns; need to know which water year each row belongs to
Daily (USGS format) — hardest: day rows × month columns means the year is usually in the page header, not in the table itself. Reducto may or may not have captured it. Need to recover the year from the filename, metadata, or JSON context
Each parser also needs to handle USGS data quality prefixes: e = estimated, # = estimated, a = affected by backwater, - = missing. These should become a quality_flag column rather than being stripped.

Step 4: Clean units
The good news: ~76% of stream discharge rows are already in the CFS family (second-feet, sec.-ft., cfs, cubic feet per second — all identical, just normalize the label). The remainder:

acre-feet — volumetric, not rate; keep but flag separately
thousands of acre-feet — same
Mixed-unit pages (e.g., discharge in cfs + runoff in acre-feet on same page)
Decision needed: keep all units with a measurement_unit column, or convert everything to CFS where possible and drop the rest. Recommend keeping all with the unit column — downstream researchers can filter.

Step 5: Resolve site identity and coordinates
The same physical station (e.g., "Sacramento River at Kennett") appears across many annual Water Supply Papers as separate rows in the metadata. We need to:

Group by watersource_name + approximate coordinates → assign a canonical site_id
Flag rows where coordinates look wrong (we already saw Kennett coords assigned to southern CA locations in some rows)
Where possible, match to a NWIS station ID — this enables cross-validation and connects to the modern record
Step 6: Handle OCR artifacts
The daily tables in particular have minor digitization errors we saw firsthand:

"4,02C" → 4,020 (letter/number OCR confusion)
"3.860" → 3,860 (decimal instead of thousands comma)
Merged rows (days 27–28 in one cell)
These need a cleaning pass with a small set of regex rules before numeric conversion. Not everything will be recoverable — some cells will need to be marked null.

Step 7: Produce the output table
Final schema:


site_id, doc_id, page_number, watersource_name, latitude, longitude,
date, measurement, measurement_unit, temporal_resolution, quality_flag
One row per measurement (one per day for daily records, one per month for monthly, etc.).

Recommended sequence
Given the scale, don't try to do everything at once. Suggested order:

Annual peaks first — simplest structure, highest scientific value (pre-dam analysis), ~1,800 rows estimated. Proves the pipeline works end to end.
Monthly means second — moderate complexity, large coverage.
Daily last — hardest (year recovery problem), but the biggest dataset.
Want to start with Step 1 — scoping how many tables we're actually dealing with and what the temporal resolution distribution looks like?

### Direction B: Streamflow Permanence / Intermittency Mapping

**Scientific motivation:** Whether a stream reach is perennial, intermittent, or ephemeral matters for ecology, water rights, and land management — but historical permanence is poorly characterized. Even a single flow measurement at a site in a given year establishes that the stream was flowing. With 2,516 restored stream discharge sites in Santa Barbara County alone (vs. 21 NWIS sites), the spatial resolution of this dataset is extraordinary for historical mapping.

**Why this is robust to data quality issues:** The analysis asks a binary question — was there flow? — rather than requiring precise discharge values. Unit normalization is largely irrelevant. Digitization errors (misread digits) rarely convert actual flow to zero. Location imprecision at stream-reach scale is acceptable. This makes the analysis far more tolerant of the messiness inherent in historical data.

**Potential scientific questions:**
- How have perennial reach lengths changed across California between the 1900s and 1980?
- Do wet decades (e.g., 1905, 1938, 1969 flood years) show markedly more flowing reaches than dry decades (1924, 1934, 1977)?
- Has groundwater pumping or land use change caused historically perennial reaches to become intermittent?

**Challenges and open questions:**
- Need to distinguish zero-flow records (actual dry conditions) from missing/null values (measurement not taken) — these look the same in a sparse CSV
- Many sites may have only 1–3 years of data; not enough to establish reliable permanence status without careful uncertainty handling
- Seasonal coverage matters: a site measured only in February tells you less about permanence than one measured in August; need to check seasonal distribution of measurements
- Sites will need to be matched to NHD stream segments for spatial analysis; coordinate precision determines how well this works
- Temporal distribution of the dataset skews toward later decades (1940–1980); pre-1940 coverage may be too sparse for decade-by-decade comparison in most regions
- Need to review existing intermittency datasets (e.g., EPA StreamStats, state stream classifications) to frame the comparison

**Suggested first steps:**
1. Using metadata only (no CSV parsing needed): map all stream discharge sites by decade and season of measurement to assess temporal/spatial coverage
2. Identify regions and time periods with enough site density for a meaningful permanence analysis
3. For the "flow or no flow" question, a simple check of whether any CSV cell is nonzero is sufficient — write a lightweight script that doesn't need to parse the full table structure

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `Data_Files/cleaned_metadata_final - Copy.csv` | Raw main metadata (source, pre-join); links every page to its doc ID, coordinates, water type, dates |
| `data/metadata/main_metadata.csv` | Comprehensive main metadata; all pages with LLM-extracted fields + USGS publication fields joined in |
| `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` | ID crosswalk: maps project number ID (`Publication ID`) → USGS URL, Index ID, title, year, author, and ~55 document-level fields. 1,550 rows covering all documents. |
| `data/digitization_intermediates/01_download_and_preprocess/publication_list.csv` | Canonical 1,042-doc input CSV for the download script; generated from USGS_ID.xlsx |
| `data/digitization_intermediates/02_table_detection/validation/` | Ground truth for table detection validation (see note below) |
| `Data_Files/2252_728 example/` | Reference example of target output folder structure |
| `Hilton_Dissertation_USGSDig.docx` | Full paper draft — authoritative description of all methods and results |
| `Chapter_2_USGS_Digitization/Yibo Results/May 27 2025 R4-6/api_run.py` | Core LLM metadata extraction script (calls UCSB HPC API: llama3 + deepseek-r1) |
| `code/03_inventory/00_data_prep.ipynb` | Loads and cleans main_metadata.csv → processed_metadata.parquet |
| `code/03_inventory/01_measurements_by_type.ipynb` | Fig 7: measurements by type and decade |
| `code/03_inventory/02_spatial_overview.ipynb` | Figs 8–10: USA and California maps |
| `code/03_inventory/03_santa_barbara.ipynb` | Figs 11–18: Santa Barbara County coverage and NWIS comparison |
| `code/03_inventory/04_santa_ynez.ipynb` | Fig 19, Tables 7–9: Santa Ynez stream order analysis |
| `code/01_digitization/04_metadata_extraction/02_validate_accuracy.ipynb` | Fig 5: metadata accuracy by category and test set |
| `data/analysis/processed_metadata.parquet` | Cleaned, parsed metadata output of 00_data_prep — input to all other inventory notebooks |
| `data/analysis/spatial/` | Shapefiles for CA, SB county, USA states, Santa Ynez HUC8, rivers, lakes |
| `legacy_ocr/usgs_extract/tableBbox.py` | Bbox visualization — may be useful when writing the validation script |
| `Chapter_2_USGS_Digitization/Literature-Data Review/LLM-Metadata Testing/Accuracy evaluation/` | Human accuracy evaluation docs and methods |

### Note on table detection ground truth files

`Chapter_2_USGS_Digitization/Yibo Results/hasTable.csv` and `filtered USGS Groundwater Data Tables & Pages.xlsx` are **not** detection output — they are the **validation ground truth**:

- `filtered USGS Groundwater Data Tables & Pages.xlsx` — human-curated list of documents known to have groundwater tables, with page ranges hand-entered (`"88–99"`, `"throughout"`, etc.). 3,053 unique pub IDs, mostly non-California.
- `hasTable.csv` — the xlsx ranges expanded to one row per individual page number (confirmed by cross-checking: every ID in hasTable is in the xlsx, and pages match exactly). Script that did this expansion is missing.

These were copied to `data/digitization_intermediates/02_table_detection/validation/` with cleaner names. The actual detection scores from running the model against this ground truth were not preserved. The 79% recall figure was computed at the time of the run but the results CSV is gone.

---

## Collaborators and Context

- **Annette Hilton** (PI, PhD candidate, UCSB Bren School) — lead researcher, author
- **Anna Boser** (PhD candidate, UCSB Bren School) — collaborator
- **Yibo Liang** — undergrad, primary coder (LLM metadata extraction, data processing)
- **Luma Braconi Lazarini** — undergrad, data digitization and metadata review
- **Henderson Vo** — undergrad, data digitization and metadata review
- **UCSB General Research IT** cluster — runs the Phi-4 LLM metadata extraction workflow
- **CUAHSI / HydroShare** — final data hosting destination

---

## Data Coverage Notes (organized output)

The `data/digitized/` folder was populated by `code/01_digitization/05_final_data_organization/organize_data.py`. Across 622 docs and 74,492 pages, three categories of pages exist but have **no `_metadata.csv` file**:

### Pages with no metadata rows (~10,919 pages, ~14.7%)
The LLM metadata extraction (Phi-4) was only run on pages where a table was detected. Pages that Reducto processed but where no table was found — or where the table was output as plain text rather than structured HTML — were never sent to the LLM and therefore have no entry in `cleaned_metadata_final - Copy.csv`. These fall into three sub-types:

1. **Non-data pages** — title pages, cover pages, introductions, references, figures. No table content present. Expected and not a data loss.

2. **Tables that Reducto could not structure** — the page contains a visible table (e.g., `10166/page_10`: "Table 2.--Age and type of bedrock at each spring") but Reducto returned the content as plain text chunks rather than HTML `<table>` elements. The raw text is preserved in the `.json` file. This represents tables the pipeline did not successfully digitize. **No further digitization will be performed** — this is documented as a known limitation.

3. **Table-of-contents pages** — pages with TOC-style formatting that Reducto parsed as a table but which contain no water measurement data.

### Pages with no table CSVs (~7,127 pages, ~9.6%)
These pages have a `.json` (Reducto ran) but no `_table*.csv` files. This occurs when Reducto processed the page but the downstream CSV conversion was not completed or the page had no digitizable table. The JSON is preserved.

### Pages with no PNG (~260 pages, ~0.3%)
A small number of pages have a JSON and CSVs but no corresponding `.png` file in either `UpdatedDataDec/` or `UpdatedDataJan/`. These are pages from documents that may have been processed outside the main PNG batches.

A full per-page log is at `data/digitized/_organization_log.txt`.

---

## Technical Notes

- Metadata extraction uses Microsoft Phi-4 via UCSB's LLM API (`llm.grit.ucsb.edu`); some iteration scripts also tested llama3 and deepseek-r1
- Table detection threshold: 0.8 (recovers ~79% of known tables)
- Reducto.ai used for final digitization (not open source — requires paid API access)
- PaddleOCR was tested but discarded for structure preservation failures
- Multiple iterations of LLM prompt engineering are preserved in `Chapter_2_USGS_Digitization/Yibo Results/` — the final production prompt is in Supplementary Information S1 of the dissertation
