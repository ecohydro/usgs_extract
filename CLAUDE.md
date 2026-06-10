# USGS Historical Water Data Digitization â€” Project Guide

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
- USGS Publications Warehouse search for California water documents published 1879â€“1980
- Returned 1,042 documents; ~60% available as scanned PDFs
- Documents span water measurements from 1810â€“1980 (data predate some documents)

### Digitization Pipeline
1. **PDF â†’ PNG preprocessing**: PDFs converted to grayscale single-channel `.png` files using Python PIL and OpenCV
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
- LLM accuracy: 87% overall; 76â€“99% by category; lowest for "County" (52â€“76%)

---

## Repository Structure (Current State)

```
usgs_extract/
â”œâ”€â”€ CLAUDE.md                          â† this file
â”œâ”€â”€ README.md                          â† target repo structure + quickstart for each pipeline step
â”œâ”€â”€ Hilton_Dissertation_USGSDig.docx   â† Chapter 2 draft (primary reference)
â”‚
â”œâ”€â”€ manuscript/
â”‚   â””â”€â”€ figures/                       â† all publication figures output here by notebooks
â”‚
â”œâ”€â”€ code/
â”‚   â”œâ”€â”€ 01_digitization/
â”‚   â”‚   â”œâ”€â”€ 01_download_and_preprocess/  â† COMPLETE
â”‚   â”‚   â”‚   â”œâ”€â”€ 01_download.py
â”‚   â”‚   â”‚   â”œâ”€â”€ 02_verify.py
â”‚   â”‚   â”‚   â””â”€â”€ 03_preprocess.py
â”‚   â”‚   â”œâ”€â”€ 02_table_detection/          â† COMPLETE (workflow); validation script MISSING
â”‚   â”‚   â”‚   â””â”€â”€ 01_detect.py
â”‚   â”‚   â”œâ”€â”€ 03_ocr/                      â† Reducto (final) + PaddleOCR (tested, not used)
â”‚   â”‚   â””â”€â”€ 04_metadata_extraction/
â”‚   â”‚       â”œâ”€â”€ 01_extract_metadata.py   â† LLM (Phi-4) extraction via UCSB HPC API
â”‚   â”‚       â””â”€â”€ 02_validate_accuracy.ipynb â† produces manuscript/figures/metadata_accuracy.png
â”‚   â”œâ”€â”€ 02_inventory/                    â† data inventory notebooks (all manuscript figures)
â”‚   â”‚   â”œâ”€â”€ water_palette.py             â† shared colorblind-safe color mapping
â”‚   â”‚   â”œâ”€â”€ 00_data_prep.ipynb           â† clean main_metadata.csv â†’ data/analysis/processed_metadata.parquet
â”‚   â”‚   â”œâ”€â”€ 01_measurements_by_type.ipynb â† Fig 7: measurements by water type Ã— decade
â”‚   â”‚   â”œâ”€â”€ 02_spatial_overview.ipynb    â† Fig 8: USA map; Fig 9: CA map; Fig 10: CA by decade
â”‚   â”‚   â”œâ”€â”€ 03_santa_barbara.ipynb       â† Figs 11â€“18: SB county coverage + NWIS comparison
â”‚   â”‚   â””â”€â”€ 04_santa_ynez.ipynb          â† Fig 19, Tables 7â€“9: Santa Ynez stream order analysis
â”‚   â””â”€â”€ 04_vignettes/                    â† hydrological research vignettes
â”‚       â””â”€â”€ dam_exploring/               â† pre-dam baseline work (notebooks 05, 06)
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ digitized/                       â† COMPLETE; final organized output, one folder per doc/page
â”‚   â”œâ”€â”€ metadata/
â”‚   â”‚   â”œâ”€â”€ main_metadata.csv            â† COMPLETE; comprehensive metadata: LLM fields + USGS pub fields joined
â”‚   â”‚   â””â”€â”€ metadata_key.txt             â† COMPLETE; column descriptions
â”‚   â”œâ”€â”€ analysis/
â”‚   â”‚   â”œâ”€â”€ csv_row_counts.csv           â† estimated measurement counts per page/table
â”‚   â”‚   â”œâ”€â”€ processed_metadata.parquet   â† output of 00_data_prep.ipynb (parsed dates, clean types)
â”‚   â”‚   â””â”€â”€ spatial/
â”‚   â”‚       â”œâ”€â”€ ca_shapefile/ca_poly.shp
â”‚   â”‚       â”œâ”€â”€ sb_county_shapefile/sb_county_shp.shp
â”‚   â”‚       â”œâ”€â”€ usa_shapefile/CUSA_States.shp
â”‚   â”‚       â”œâ”€â”€ santaynezhuc8/santaynez.shp  â† Santa Ynez HUC8 watershed boundary
â”‚   â”‚       â”œâ”€â”€ rivers/StreamsandRivers_Clip.shp
â”‚   â”‚       â”œâ”€â”€ lakes/chap2map.gdb            â† National Wetlands Inventory (GDB format)
â”‚   â”‚       â””â”€â”€ streamprox/streamprox_santaynez.txt â† pre-computed stream order proximity
â”‚   â”‚   â””â”€â”€ nwis_sites/sb_county_usgs/       â† NWIS site ID reference files for SB county
â”‚   â””â”€â”€ digitization_intermediates/
â”‚       â”œâ”€â”€ 01_download_and_preprocess/
â”‚       â”‚   â””â”€â”€ publication_list.csv
â”‚       â”œâ”€â”€ 02_table_detection/
â”‚       â”‚   â””â”€â”€ validation/
â”‚       â”‚       â”œâ”€â”€ groundwater_table_pages.xlsx
â”‚       â”‚       â”œâ”€â”€ groundwater_table_pages_expanded.csv
â”‚       â”‚       (MISSING: validation run output â€” 79% recall figure not preserved)
â”‚       â””â”€â”€ 04_metadata_extraction/
â”‚           â”œâ”€â”€ llm_extracted_metadata.csv
â”‚           â”œâ”€â”€ USGS_publication_metadata.xlsx
â”‚           â””â”€â”€ validation/
â”‚               â”œâ”€â”€ validation_curated_135.xlsx
â”‚               â”œâ”€â”€ validation_random_135.xlsx
â”‚               â”œâ”€â”€ validation_edge_cases_15.xlsx
â”‚               â”œâ”€â”€ test_set_cat_grades.xlsx   â† per-category accuracy (input to 02_validate_accuracy.ipynb)
â”‚               â”œâ”€â”€ test_set_indiv_grades.xlsx â† per-entry accuracy (input to 02_validate_accuracy.ipynb)
â”‚               â””â”€â”€ jsons/                     â† 260 per-page Reducto JSONs for validation sets
â”‚
â”œâ”€â”€ Data_Files/                        â† ARCHIVED; ALL digitized data lives here (Annette's side; pre-organization)
â”‚   â”œâ”€â”€ cleaned_metadata_final - Copy.csv  â† raw main metadata (pre-join source)
â”‚   â”œâ”€â”€ csv_row_counts.csv
â”‚   â”œâ”€â”€ 2252_728 example/              â† reference example of target output format
â”‚   â”œâ”€â”€ ReductCSVs/                    â† 623 document ID subfolders â†’ .csv files
â”‚   â”œâ”€â”€ ReductJson/                    â† 623 document ID subfolders â†’ .json files
â”‚   â”œâ”€â”€ UpdatedDataDec/                â† 353 document ID subfolders â†’ .png files
â”‚   â””â”€â”€ UpdatedDataJan/                â† 287 document ID subfolders â†’ .png files
â”‚
â”œâ”€â”€ Chapter_2_USGS_Digitization/       â† project documents, legacy code, manuscript drafts
â”‚   â”œâ”€â”€ Manuscript/
â”‚   â”œâ”€â”€ Yibo Results/                  â† LLM metadata extraction iterative runs
â”‚   â””â”€â”€ Literature-Data Review/
â”‚
â”œâ”€â”€ jupyter_notebooks/                 â† exploratory/working notebooks (not publication notebooks)
â”‚   â”œâ”€â”€ working_datacrunch.ipynb       â† original data processing (superseded by 00_data_prep.ipynb)
â”‚   â”œâ”€â”€ archived_notebooks/
â”‚   â””â”€â”€ [other exploratory notebooks]
â”‚
â””â”€â”€ legacy_ocr/                        â† original nbdev package (superseded by code/)
    â””â”€â”€ usgs_extract/model.py
```

---

## Target Output Format (Step 1 Goal)

The final organized data should have one folder per document ID, with a subfolder per page where a table was detected:

```
organized_data/
â””â”€â”€ {doc_id}/
    â””â”€â”€ page_{N}/
        â”œâ”€â”€ {doc_id}_page_{N}.png       â† grayscale page image
        â”œâ”€â”€ {doc_id}_page_{N}.json      â† Reducto full output
        â”œâ”€â”€ {doc_id}_page_{N}_table1.csv â† digitized table(s)
        â”œâ”€â”€ {doc_id}_page_{N}_table2.csv â† (if multiple tables on page)
        â””â”€â”€ {doc_id}_page_{N}_metadata.csv  â† rows from cleaned_metadata_final for this page
```

The `2252_728 example/` folder in `Data_Files/` shows the file naming convention. The metadata CSV per page is not yet generated â€” it requires filtering `cleaned_metadata_final - Copy.csv` by `id` and `page_number`.

**Key data gap**: PNGs are split across `UpdatedDataDec/` (353 doc folders) and `UpdatedDataJan/` (287 doc folders), while CSVs and JSONs are each in their own 623-folder directories. These need to be merged into the unified per-page structure.

---

## Planned Work (Four Steps)

### Step 1: Organize Final Data Output - COMPLETE 
- Write a script to reorganize all files into the unified `{doc_id}/page_{N}/` structure
- Extract metadata rows per page from `cleaned_metadata_final - Copy.csv` and save as individual CSVs
- Join USGS document-level metadata from `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` into each per-page metadata CSV â€” the `id` column in the main metadata maps directly to the `Publication ID` column in `USGS_ID.xlsx`, which provides the USGS URL, Index ID (series), title, year, author, and ~55 additional document-level fields
- Identify and document any doc IDs present in metadata but missing from data folders (or vice versa)

### Step 2: HydroShare Upload Organization
- Explore groupings of data for upload (e.g., by water_type + decade)
- Understand HydroShare size/file limits
- Create upload-ready copies of grouped data
- May need to compress or chunk large groups

### Step 3: Data Inventory (Notebooks) â€” COMPLETE
All publication figures are implemented in `code/02_inventory/`. Run notebooks in order (00 â†’ 04); all figures save to `manuscript/figures/`.

- `00_data_prep.ipynb` â€” parses `data/metadata/main_metadata.csv`, cleans water types, parses dates, joins row counts; outputs `data/analysis/processed_metadata.parquet`
- `01_measurements_by_type.ipynb` â€” Fig 7: measurements by water type Ã— decade
- `02_spatial_overview.ipynb` â€” Figs 8â€“10: USA and California maps
- `03_santa_barbara.ipynb` â€” Figs 11â€“18: SB county coverage + NWIS groundwater/stream discharge comparison
- `04_santa_ynez.ipynb` â€” Fig 19, Tables 7â€“9: Santa Ynez stream order analysis

Metadata accuracy figure (Fig 5) is in `code/01_digitization/04_metadata_extraction/02_validate_accuracy.ipynb`.

### Step 3b: NWIS California Stream Site Inventory — COMPLETE
`code/03_nwis_usgs/` contains scripts to download and compile all USGS NWIS stream discharge gauge sites for California. Separate from the digitization inventory; this is modern NWIS data used for cross-reference and comparison.

### Step 4: Hydrological Research Vignettes â€” FUTURE
`code/04_vignettes/` contains hydrological research vignettes. The `dam_exploring/` subfolder holds the pre-dam baseline work started May 2026 (see Vignette Work Log below). Two directions have been identified as high-priority; see the **Potential Hydrological Research Directions** section below for details.

---

## Potential Hydrological Research Directions

These are candidate scientific contributions that use the restored data for original hydrological analysis, not just data description. Both focus on stream discharge because it is by far the largest category (84,634 tables, ~2.2M measurements). Key metadata facts relevant to both directions:

- ~76% of stream discharge rows have CFS-family units (second-feet, sec.-ft., cfs, cubic feet per second â€” all the same unit and trivially normalized)
- The LLM extracted per-column unit mappings as dict strings (e.g., `{'Discharge': 'cfs', 'Gage_Height': 'feet'}`), which identify the relevant column before the CSV is opened
- ~50% of stream discharge rows (~38,000) are daily CFS records across 8,270 unique sites and 219 documents
- 17,557 rows have data starting at or before 1920 (83% in CFS units), spanning 5,098 unique sites

**Prerequisite for both:** The actual table CSVs (`Data_Files/ReductCSVs/`) must be accessible. First step once files are reachable: sample ~50 stream discharge CSVs to understand how Reducto rendered the classic USGS day-row Ã— month-column table structure.

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
- CSV files not currently locally accessible â€” need `Data_Files/` mounted or transferred
- USGS daily tables use a day-row Ã— month-column grid; Reducto's CSV output for this structure is unknown until files are sampled
- Annual summary tables are simpler but still need column identification (headers vary: "Mean", "Annual discharge", "Second-feet", etc.)
- Some "pre-dam" records may be from diversions, canals, or regulated reaches â€” not natural flow; need to flag these
- Location precision: coordinates may be coarse (township/range inferred), making watershed assignment uncertain for some sites
- Need to check whether any restored site names match NWIS station IDs, enabling cross-validation during the 1940â€“1980 overlap period
- Verify which specific rivers are best represented in pre-1920 data before committing to a study area

**Suggested first steps:**
1. Filter metadata to stream discharge + CFS + annual temporal resolution + start year â‰¤ 1945
2. Sample 20â€“30 of those CSVs to understand column structure
3. Identify a focal river or small set of rivers with the most pre-dam site coverage
4. Build cross-validation using 1940â€“1980 overlap with NWIS to establish data quality

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
Daily (classic USGS)	Day rows (1â€“31) Ã— month columns	doc 553/page 267
We need to sample ~50â€“100 stream discharge CSVs across different documents to confirm these are the main types and identify any others. The LLM temporal_resolution field is our best guide, but it's noisy.

Step 3: Write a parser for each table type
Each structure needs its own parser to produce long-form rows. The hardest problem for each:

Annual peaks â€” simplest; just read row by row, handle two-column printed layouts (already split correctly in the CSV)
Monthly means â€” melt the month columns; need to know which water year each row belongs to
Daily (USGS format) â€” hardest: day rows Ã— month columns means the year is usually in the page header, not in the table itself. Reducto may or may not have captured it. Need to recover the year from the filename, metadata, or JSON context
Each parser also needs to handle USGS data quality prefixes: e = estimated, # = estimated, a = affected by backwater, - = missing. These should become a quality_flag column rather than being stripped.

Step 4: Clean units
The good news: ~76% of stream discharge rows are already in the CFS family (second-feet, sec.-ft., cfs, cubic feet per second â€” all identical, just normalize the label). The remainder:

acre-feet â€” volumetric, not rate; keep but flag separately
thousands of acre-feet â€” same
Mixed-unit pages (e.g., discharge in cfs + runoff in acre-feet on same page)
Decision needed: keep all units with a measurement_unit column, or convert everything to CFS where possible and drop the rest. Recommend keeping all with the unit column â€” downstream researchers can filter.

Step 5: Resolve site identity and coordinates
The same physical station (e.g., "Sacramento River at Kennett") appears across many annual Water Supply Papers as separate rows in the metadata. We need to:

Group by watersource_name + approximate coordinates â†’ assign a canonical site_id
Flag rows where coordinates look wrong (we already saw Kennett coords assigned to southern CA locations in some rows)
Where possible, match to a NWIS station ID â€” this enables cross-validation and connects to the modern record
Step 6: Handle OCR artifacts
The daily tables in particular have minor digitization errors we saw firsthand:

"4,02C" â†’ 4,020 (letter/number OCR confusion)
"3.860" â†’ 3,860 (decimal instead of thousands comma)
Merged rows (days 27â€“28 in one cell)
These need a cleaning pass with a small set of regex rules before numeric conversion. Not everything will be recoverable â€” some cells will need to be marked null.

Step 7: Produce the output table
Final schema:


site_id, doc_id, page_number, watersource_name, latitude, longitude,
date, measurement, measurement_unit, temporal_resolution, quality_flag
One row per measurement (one per day for daily records, one per month for monthly, etc.).

Recommended sequence
Given the scale, don't try to do everything at once. Suggested order:

Annual peaks first â€” simplest structure, highest scientific value (pre-dam analysis), ~1,800 rows estimated. Proves the pipeline works end to end.
Monthly means second â€” moderate complexity, large coverage.
Daily last â€” hardest (year recovery problem), but the biggest dataset.
Want to start with Step 1 â€” scoping how many tables we're actually dealing with and what the temporal resolution distribution looks like?

### Direction B: Streamflow Permanence / Intermittency Mapping

**Scientific motivation:** Whether a stream reach is perennial, intermittent, or ephemeral matters for ecology, water rights, and land management â€” but historical permanence is poorly characterized. Even a single flow measurement at a site in a given year establishes that the stream was flowing. With 2,516 restored stream discharge sites in Santa Barbara County alone (vs. 21 NWIS sites), the spatial resolution of this dataset is extraordinary for historical mapping.

**Why this is robust to data quality issues:** The analysis asks a binary question â€” was there flow? â€” rather than requiring precise discharge values. Unit normalization is largely irrelevant. Digitization errors (misread digits) rarely convert actual flow to zero. Location imprecision at stream-reach scale is acceptable. This makes the analysis far more tolerant of the messiness inherent in historical data.

**Potential scientific questions:**
- How have perennial reach lengths changed across California between the 1900s and 1980?
- Do wet decades (e.g., 1905, 1938, 1969 flood years) show markedly more flowing reaches than dry decades (1924, 1934, 1977)?
- Has groundwater pumping or land use change caused historically perennial reaches to become intermittent?

**Challenges and open questions:**
- Need to distinguish zero-flow records (actual dry conditions) from missing/null values (measurement not taken) â€” these look the same in a sparse CSV
- Many sites may have only 1â€“3 years of data; not enough to establish reliable permanence status without careful uncertainty handling
- Seasonal coverage matters: a site measured only in February tells you less about permanence than one measured in August; need to check seasonal distribution of measurements
- Sites will need to be matched to NHD stream segments for spatial analysis; coordinate precision determines how well this works
- Temporal distribution of the dataset skews toward later decades (1940â€“1980); pre-1940 coverage may be too sparse for decade-by-decade comparison in most regions
- Need to review existing intermittency datasets (e.g., EPA StreamStats, state stream classifications) to frame the comparison

**Suggested first steps:**
1. Using metadata only (no CSV parsing needed): map all stream discharge sites by decade and season of measurement to assess temporal/spatial coverage
2. Identify regions and time periods with enough site density for a meaningful permanence analysis
3. For the "flow or no flow" question, a simple check of whether any CSV cell is nonzero is sufficient â€” write a lightweight script that doesn't need to parse the full table structure

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `Data_Files/cleaned_metadata_final - Copy.csv` | Raw main metadata (source, pre-join); links every page to its doc ID, coordinates, water type, dates |
| `data/metadata/main_metadata.csv` | Comprehensive main metadata; all pages with LLM-extracted fields + USGS publication fields joined in |
| `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` | ID crosswalk: maps project number ID (`Publication ID`) â†’ USGS URL, Index ID, title, year, author, and ~55 document-level fields. 1,550 rows covering all documents. |
| `data/digitization_intermediates/01_download_and_preprocess/publication_list.csv` | Canonical 1,042-doc input CSV for the download script; generated from USGS_ID.xlsx |
| `data/digitization_intermediates/02_table_detection/validation/` | Ground truth for table detection validation (see note below) |
| `Data_Files/2252_728 example/` | Reference example of target output folder structure |
| `Hilton_Dissertation_USGSDig.docx` | Full paper draft â€” authoritative description of all methods and results |
| `Chapter_2_USGS_Digitization/Yibo Results/May 27 2025 R4-6/api_run.py` | Core LLM metadata extraction script (calls UCSB HPC API: llama3 + deepseek-r1) |
| `code/02_inventory/00_data_prep.ipynb` | Loads and cleans main_metadata.csv â†’ processed_metadata.parquet |
| `code/02_inventory/01_measurements_by_type.ipynb` | Fig 7: measurements by type and decade |
| `code/02_inventory/02_spatial_overview.ipynb` | Figs 8â€“10: USA and California maps |
| `code/02_inventory/03_santa_barbara.ipynb` | Figs 11â€“18: Santa Barbara County coverage and NWIS comparison |
| `code/02_inventory/04_santa_ynez.ipynb` | Fig 19, Tables 7â€“9: Santa Ynez stream order analysis |
| `code/01_digitization/04_metadata_extraction/02_validate_accuracy.ipynb` | Fig 5: metadata accuracy by category and test set |
| `data/analysis/processed_metadata.parquet` | Cleaned, parsed metadata output of 00_data_prep â€” input to all other inventory notebooks |
| `data/analysis/spatial/` | Shapefiles for CA, SB county, USA states, Santa Ynez HUC8, rivers, lakes |
| `legacy_ocr/usgs_extract/tableBbox.py` | Bbox visualization â€” may be useful when writing the validation script |
| `Chapter_2_USGS_Digitization/Literature-Data Review/LLM-Metadata Testing/Accuracy evaluation/` | Human accuracy evaluation docs and methods |

### Note on table detection ground truth files

`Chapter_2_USGS_Digitization/Yibo Results/hasTable.csv` and `filtered USGS Groundwater Data Tables & Pages.xlsx` are **not** detection output â€” they are the **validation ground truth**:

- `filtered USGS Groundwater Data Tables & Pages.xlsx` â€” human-curated list of documents known to have groundwater tables, with page ranges hand-entered (`"88â€“99"`, `"throughout"`, etc.). 3,053 unique pub IDs, mostly non-California.
- `hasTable.csv` â€” the xlsx ranges expanded to one row per individual page number (confirmed by cross-checking: every ID in hasTable is in the xlsx, and pages match exactly). Script that did this expansion is missing.

These were copied to `data/digitization_intermediates/02_table_detection/validation/` with cleaner names. The actual detection scores from running the model against this ground truth were not preserved. The 79% recall figure was computed at the time of the run but the results CSV is gone.

---

## Collaborators and Context

- **Annette Hilton** (PI, PhD candidate, UCSB Bren School) â€” lead researcher, author
- **Anna Boser** (PhD candidate, UCSB Bren School) â€” collaborator
- **Yibo Liang** â€” undergrad, primary coder (LLM metadata extraction, data processing)
- **Luma Braconi Lazarini** â€” undergrad, data digitization and metadata review
- **Henderson Vo** â€” undergrad, data digitization and metadata review
- **UCSB General Research IT** cluster â€” runs the Phi-4 LLM metadata extraction workflow
- **CUAHSI / HydroShare** â€” final data hosting destination

---

## Data Coverage Notes (organized output)

The `data/digitized/` folder was populated by `code/01_digitization/05_final_data_organization/organize_data.py`. Across 622 docs and 74,492 pages, three categories of pages exist but have **no `_metadata.csv` file**:

### Pages with no metadata rows (~10,919 pages, ~14.7%)
The LLM metadata extraction (Phi-4) was only run on pages where a table was detected. Pages that Reducto processed but where no table was found â€” or where the table was output as plain text rather than structured HTML â€” were never sent to the LLM and therefore have no entry in `cleaned_metadata_final - Copy.csv`. These fall into three sub-types:

1. **Non-data pages** â€” title pages, cover pages, introductions, references, figures. No table content present. Expected and not a data loss.

2. **Tables that Reducto could not structure** â€” the page contains a visible table (e.g., `10166/page_10`: "Table 2.--Age and type of bedrock at each spring") but Reducto returned the content as plain text chunks rather than HTML `<table>` elements. The raw text is preserved in the `.json` file. This represents tables the pipeline did not successfully digitize. **No further digitization will be performed** â€” this is documented as a known limitation.

3. **Table-of-contents pages** â€” pages with TOC-style formatting that Reducto parsed as a table but which contain no water measurement data.

### Pages with no table CSVs (~7,127 pages, ~9.6%)
These pages have a `.json` (Reducto ran) but no `_table*.csv` files. This occurs when Reducto processed the page but the downstream CSV conversion was not completed or the page had no digitizable table. The JSON is preserved.

### Pages with no PNG (~260 pages, ~0.3%)
A small number of pages have a JSON and CSVs but no corresponding `.png` file in either `UpdatedDataDec/` or `UpdatedDataJan/`. These are pages from documents that may have been processed outside the main PNG batches.

A full per-page log is at `data/digitized/_organization_log.txt`.

---

## Technical Notes

- Metadata extraction uses Microsoft Phi-4 via UCSB's LLM API (`llm.grit.ucsb.edu`); some iteration scripts also tested llama3 and deepseek-r1
- Table detection threshold: 0.8 (recovers ~79% of known tables)
- Reducto.ai used for final digitization (not open source â€” requires paid API access)
- PaddleOCR was tested but discarded for structure preservation failures
- Multiple iterations of LLM prompt engineering are preserved in `Chapter_2_USGS_Digitization/Yibo Results/` â€” the final production prompt is in Supplementary Information S1 of the dissertation

---

## Vignette Work Log â€” May 18, 2026

Work toward a **pre-dam streamflow baseline** vignette using metadata only (no CSV parsing). Three things were built:

### 1. Dam inventory scan (metadata only)
Searched `main_metadata.csv` for dam and reservoir name mentions. Key findings:
- **Matilija Dam** (Ventura County, proposed removal ~2030): ~278 rows, records back to 1906 â€” pre-impoundment baseline exists in the dataset.
- **San Clemente Dam** (Carmel River, removed 2015): ~74 rows on the Carmel River.
- **Potter Valley Project** (Scott Dam + Van Arsdale Dam, Eel River): 370 rows, earliest record October 1909 (one year after dam completion). Covers both above-dam and powerhouse tailrace into Russian River. Scott Dam removal is actively proposed â€” these records are policy-relevant.

### 2. Interactive pre-dam gauge map (`code/04_vignettes/dam_exploring/05_predam_gauge_map.ipynb`)
Two-layer Plotly map (3.5 MB HTML, renders in any browser):
- **Gauge sites**: 34,692 unique CA stream discharge sites from restored metadata, color-coded by first-record era (Pre-1900 / 1900â€“1919 / 1920â€“1939 / 1940â€“1959 / 1960+). Pre-1940 layers on by default.
- **NID dams**: 1,534 CA dams from the National Inventory of Dams (downloaded May 2026, saved to `data/analysis/dams.csv` and `dams.geojson`).
- Output: `manuscript/figures/predam_gauge_map.html`

Era breakdown of restored sites: Pre-1900 (1,025), 1900â€“1919 (9,078), 1920â€“1939 (5,941), 1940â€“1959 (4,310), 1960+ (14,338).

### 3. NWIS cross-reference (`code/04_vignettes/dam_exploring/06_nwis_crossref.ipynb`)
Downloaded all CA NWIS stream discharge sites with date ranges (2,418 unique sites, 1891â€“present) and cross-referenced against restored metadata using coordinate proximity (â‰¤0.1Â°) and name token overlap.
- **1,646 strong matches** (close coords + shared name tokens) â€” high-confidence pairs for cross-validation.
- **255 of 256 pre-1920 NWIS sites** have a coordinate match in the restored data.
- Top matched sites have 70â€“84 years of overlap (e.g., Alameda Creek near Niles 1891â€“1975, Sacramento R. above Bend Bridge 1879â€“1975).
- Outputs: `data/analysis/nwis_ca_streams.parquet`, `data/analysis/dam_exploring/nwis_crossref_best_match.csv`

### New files added
| File | Purpose |
|------|---------|
| `data/analysis/dams.csv` | NID California dam inventory (1,534 dams, coordinates + year completed) |
| `data/analysis/dams.geojson` | Same, as GeoJSON |
| `data/analysis/nwis_ca_streams.parquet` | Parsed NWIS CA stream site file (9,821 rows, 2,981 unique sites) |
| `data/analysis/dam_exploring/nwis_streams_ca.txt` | Raw NWIS download (tab-delimited, retrieved 2026-05-18) |
| `data/analysis/dam_exploring/nwis_crossref_best_match.csv` | Best NWISâ€“restored match per NWIS site (2,407 rows) |
| `data/analysis/dam_exploring/nwis_crossref_matches.csv` | All coord-proximity pairs (11,822 rows) |
| `manuscript/figures/predam_gauge_map.html` | Interactive Plotly map â€” gauge sites vs. NID dams |
| `code/04_vignettes/dam_exploring/05_predam_gauge_map.ipynb` | Notebook: builds the gauge/dam map |
| `code/04_vignettes/dam_exploring/06_nwis_crossref.ipynb` | Notebook: NWIS parquet save + cross-reference |

---

## Table-Level Metadata Work Log â€” June 2026

### Problem identified
`main_metadata.csv` is not one row per table. The LLM (Phi-4) was run per page and sometimes returned multiple rows for the same page when it detected distinct water sources. This means `n_measurements` in `processed_metadata.parquet` double-counts measurements for pages with multiple metadata rows â€” the page-level join in `00_data_prep.ipynb` attaches the sum of all table row counts to every metadata row for that page. This inflates Fig 7's total from the correct ~2,666,019 to ~3,175,399.

The old `jupyter_notebooks/measurements_bytype.ipynb` tried to fix this by deduplicating to one row per page â€” but that was also wrong, because it discarded legitimate rows (pages with multiple tables of different water types).

### What was built
`code/02_inventory/build_table_level_metadata.py` â€” produces `data/analysis/table_level_metadata.csv`, a true table-level file with one row per CSV table file. This is the correct input for Fig 7 and any measurement-count analysis.

**Assignment methodology (three stages):**
1. **LLM unambiguous (97,358 tables, 91.7%)** â€” pages where the LLM returned exactly one distinct water type. Trusted directly. The LLM read full page context and is more reliable than column headers for determining the primary purpose of a table.
2. **Header classifier resolves LLM ambiguity (1,756 tables, 1.7%)** â€” pages where the LLM returned >1 distinct water type (e.g. Stream Discharge + Reservoir on same page). Column headers are read from each table CSV and keyword-matched to assign a type per table. If headers are also ambiguous, first LLM type is used (flagged as `llm_ambiguous`, 1,715 tables).
3. **Header classifier only, no LLM (2,097 tables, 2.0%)** â€” pages the LLM never processed. High-confidence header match used; otherwise `uncertain`.

**Why the header classifier is NOT used to override unambiguous LLM classifications:** Many USGS stream discharge tables include water quality parameters (temperature, specific conductance, pH) as secondary columns. The header classifier, seeing "specific conductance", would classify these as Water Quality â€” but the LLM correctly identifies them as Stream Discharge based on document context. Using the header as an override inflated Water Quality from 248 â†’ 7,793 tables; the corrected approach brings it to 786.

### Corrected numbers vs. Dissertation Table 6
| Category | Dissertation | Corrected | Î” |
|---|---|---|---|
| Stream Discharge | 2,206,942 | 2,212,272 | +0.2% |
| Groundwater | 137,875 | 140,281 | +1.7% |
| Reservoir | 91,816 | 94,024 | +2.4% |
| Springs | 17,277 | 16,306 | -5.6% |
| Irrigation | 25,157 | 35,561 | +41% |
| Precipitation | 8,159 | 9,325 | +14% |
| Water Quality | 6,940 | 19,145 | +176% |
| Not Water Related | 6,292 | 25,881 | +311% |
| Other | 128 | 11,682 | large |
| uncertain | â€” | 101,542 | new |
| **Total** | **2,500,586** | **2,666,019** | **+6.6%** |

The +6.6% total increase is explained by the old method's inner join dropping ~42,000 tables that existed in the CSV data but had no matching deduplicated metadata row. The stream discharge, groundwater, reservoir, and springs numbers are all within 6% â€” strong validation that the LLM classifications are stable.

The larger increases in Irrigation, NWR, WQ, and Other reflect categories that had a higher proportion of their tables in the previously-uncounted 42,000.

### Files produced
| File | Purpose |
|---|---|
| `data/analysis/table_headers.csv` | Intermediate: 106,191 rows, one per table CSV â€” raw header text + Stage 2 keyword classification. Cached; delete to re-scan. |
| `data/analysis/table_level_metadata.csv` | Final: one row per table CSV â€” `water_type_final`, `assignment_source`, `number_rows`, dates, coordinates. Input for corrected Fig 7. |

### Residual categories â€” RESOLVED (2026-06-10)

Both residual buckets were resolved by the publication filter below. See the
**Publication Filter Work Log â€” June 10, 2026** section.

- **`uncertain` (3,265 tables, 101,542 measurements):** confirmed to be *entirely*
  non-LLM tables (`assignment_source == 'no_data'` â€” zero are LLM-processed).
  Per the decision to publish only LLM-processed data, these are **dropped**, not
  reclassified. No document concentration to exploit (spread across 380 docs).
- **`Other` (470 tables, 11,682 measurements):** all LLM-processed. **Kept as-is**
  and documented as "data table detected, water type undetermined by the LLM."
  Not reclassified.

---

## Publication Filter Work Log â€” June 10, 2026

### Decision
Publish **only LLM-processed tables.** Tables with no LLM metadata are dropped,
not reclassified. This resolves both the LLM-gap question (drop) and the residual
`uncertain`/`Other` investigation in one step.

### What was built
`code/02_inventory/filter_published_tables.py` â€” reads `table_level_metadata.csv`
(read-only) and writes `data/analysis/table_level_metadata_published.csv`, the
LLM-processed subset. This is the correct input for publication measurement
counts and Fig 7.

**Dropped (5,362 tables, 153,243 measurements):** every table with no LLM
metadata â€” `assignment_source` of `no_data` (3,265, = the entire `uncertain`
bucket) or `header_no_llm` (2,097, typed by header keyword only). These have no
watersource_name, dates, or coordinates and cannot be located or cross-validated.

**Kept (100,829 tables, 2,512,776 measurements):** `llm_unambiguous` (97,358),
`llm_ambiguous` (1,715), `header_high` (1,756 â€” LLM-ambiguous pages where the
header resolved which table is which; the LLM did run on these pages). The
`Other` category (470 tables) is retained and documented, not reclassified.

### Published numbers vs. Dissertation Table 6
| Category | Dissertation | Published | Î” |
|---|---|---|---|
| Stream Discharge | 2,206,942 | 2,193,948 | âˆ’0.6% |
| Groundwater | 137,875 | 136,833 | âˆ’0.8% |
| Reservoir | 91,816 | 91,103 | âˆ’0.8% |
| Irrigation | 25,157 | 30,142 | +19.8% |
| Springs | 17,277 | 16,306 | âˆ’5.6% |
| Precipitation | 8,159 | 8,747 | +7.2% |
| Water Quality | 6,940 | 11,168 | +60.9% |
| Not Water Related | 6,292 | 12,847 | +104% |
| Other | 128 | 11,682 | large |
| **Total** | **2,500,586** | **2,512,776** | **+0.5%** |

Dropping non-LLM tables brings the total to within **0.5%** of the dissertation
(the all-tables figure was +6.6%) and the three largest categories within 1% â€”
strong validation. The remaining larger deltas (WQ, NWR, Irrigation) reflect
the corrected per-table assignment, not the filter.

### Files produced
| File | Purpose |
|---|---|
| `code/02_inventory/filter_published_tables.py` | Filters `table_level_metadata.csv` â†’ LLM-processed subset |
| `data/analysis/table_level_metadata_published.csv` | Publication dataset: 100,829 LLM-processed tables, 2,512,776 measurements |

---

## Mention in Publication / Before Publication

- **Measurement estimate caveat:** The ~2.5M measurement figure (and the `n_measurements` column in `processed_metadata.parquet`) is derived from a raw line count of the Reducto CSV files (`jupyter_notebooks/csv_estimates.ipynb`), meaning it may underestimate true measurements for wide daily tables (where one row spans multiple months) and overestimate for tables where rows represent summary statistics rather than individual observations.
- **Published measurement total (USE THIS):** 2,512,776 measurements across 100,829 LLM-processed tables, from `data/analysis/table_level_metadata_published.csv`. Within 0.5% of the dissertation's 2,500,586 figure. This is the publication number.
- **Dropped from publication:** 5,362 tables (153,243 measurements) had no LLM metadata and were dropped (see Publication Filter Work Log). Document as: "LLM metadata extraction was not completed for ~5% of detected tables; these are excluded from the published water-type inventory."
- **Superseded figure:** The all-tables total of 2,666,019 (in `table_level_metadata.csv`) includes the 5,362 non-LLM tables and should not be cited as the published total.


---

## NWIS CA Stream Pull Work Log — June 10, 2026

### What was built
code/03_nwis_usgs/fetch_nwis_ca_stream_sites.py downloads all USGS NWIS stream discharge gauge sites for California, county by county (58 counties), with full period-of-record information. This is a standalone pull of modern NWIS data, separate from the digitization inventory.

**Query:** NWIS site service, siteType=ST (stream gauging stations) + hasDataTypeCd=dv (sites with daily values) + seriesCatalogOutput=true (adds begin_date / end_date / count_nu per series) + siteStatus=all (active and discontinued). Discharge series identified post-download by filtering to data_type_cd=dv + parm_cd=00060.

**Why county-by-county:** State-level queries with seriesCatalogOutput can silently truncate. County queries are small, resumable, and produce inspectable intermediate files. San Francisco county returns a 404 (no stream gauges); handled as "no sites."

**Results (2026-06-10):**
- 57/58 counties fetched (San Francisco: no stream gauges)
- 127,395 raw rows (all data series across all sites)
- **2,531 unique stream gauge sites** (2,419 with daily discharge / parm_cd=00060; 112 fallback dv-only)
- Earliest record: 1891
- Sites with records beginning ≤1920: **270**
- Sites with records beginning ≤1945: **683**

**Resume:** Re-running skips counties whose CSV already exists in the scratch folder. Delete a county CSV to force re-fetch.

### Files produced
| File | Purpose |
|------|---------|
| `code/03_nwis_usgs/fetch_nwis_ca_stream_sites.py` | Download script |
| `data/analysis/nwis_sites/ca_stream_sites_by_county/` | Scratch: one CSV per county (raw, all series) |
| `data/analysis/nwis_sites/ca_stream_sites_raw.csv` | All series rows pre-dedup (127,395 rows) |
| `data/analysis/nwis_sites/ca_stream_sites.csv` | One row per site (2,531 sites) with begin/end dates |
| `data/analysis/nwis_sites/ca_stream_sites.parquet` | Same, parquet format |