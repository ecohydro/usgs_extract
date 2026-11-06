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

### Step 2: HydroShare Upload Organization — IN PROGRESS (June 2026)

**Grouping decision:** One HydroShare resource per water type category; within each resource, data is split by county (Census TIGER county boundaries). Folder structure:

```
data/hydroshare/{water_type}/{county_name}/
    {doc_id}/
        page_{N}/
            {doc_id}_page_{N}.png
            {doc_id}_page_{N}.json
            {doc_id}_page_{N}_table*.csv
            {doc_id}_page_{N}_metadata.csv
```

**County assignment:** Uses `data/analysis/table_level_metadata_county.csv` — county determined by spatial join of `lat_combined`/`lon_combined` against Census TIGER 2023 CA county shapefile (downloaded to `data/analysis/spatial/ca_counties/tl_2023_us_county.shp`). Consistency check against LLM `actual_county`/`inferred_county` showed 67.6% match, 19.1% mismatch (all adjacent-county border cases, expected).

**Multi-type pages:** Pages containing tables of two or more water types are copied into each matching water_type/county group.

**SKIPPED — to revisit:** ~12% of tables (11,967) have no county assigned because their coordinates fall outside the CA state boundary or coordinates are missing. These are valid data (interstate records, pages without coordinates). They are excluded from the current `data/hydroshare` copy. **TODO:** decide on a separate grouping or upload strategy for these unassigned records before finalizing the HydroShare submission.

**Scripts:**
- `code/02_inventory/county_spatial_join.py` — builds `data/analysis/table_level_metadata_county.csv` (spatial join + consistency check)
- `code/02_inventory/hydroshare_copy.py` — copies page folders from `data/digitized` into `data/hydroshare` structure; resumable (skips existing destinations)
- Log of all copy operations: `data/hydroshare/_copy_log.csv`

**Size estimates:** Stream Discharge ~15–23 GB (53,178 pages); all other types combined ~5–7 GB estimated. Total HydroShare copy ~20–30 GB.

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
---

## Combined 6-Panel Figure Draft — July 14, 2026; Row 3 + caption Oct 1, 2026 (DRAFT COMPLETE)

### Goal
A candidate publication figure: 2 columns (Stream Discharge | Groundwater) × 3 rows,
comparing restored sites vs. sites already available from USGS NWIS, statewide CA.

- **Row 1 (Panels A/B) — DONE:** new sites per decade time series (statewide version
  of dissertation Figs 13/17 Panel A). Restored = unique lat/lon locations clipped to
  CA boundary, counted by decade the record begins. NWIS = sites by `begin_year`
  decade, capped at 1979.
- **Row 2 (Panels C/D) — DONE:** CA maps of restored vs. NWIS site locations (reuses
  logic from notebooks 05/06). Stream: restored (38,741) under NWIS (2,107).
  Groundwater: NWIS (32,618) under restored (5,981) — layering flipped per density.
- **Row 3 (Panels E/F) — REVISED 2026-10-28:** now two classes only: NWIS (any NWIS
  site, incl. basins that also have restored data; stream 1,460 / GW 1,889) vs.
  Restored, no NWIS sites (1,611 / 471 all sites → 1,522 / 372 dated sites only, 2026-10-28). No-site basins not drawn at all (white background inside CA outline), no legend entry.
  Goal: show only where restored data fills basins with zero NWIS data.
  Original version (DONE 2026-10-01): HUC12 basin coverage maps. Each of
  CA's 4,473 HUC12s is classed Restored only / Both / NWIS only / No sites.
  Stream: 1,611 / 1,397 / 63 / 1,402. Groundwater: 471 / 936 / 953 / 2,113.
  "Restored only" counts equal the abstract's previously-ungauged-basin numbers
  (08_scope_extension_stats.py). `BASIN_NWIS_ALL_YEARS = True` counts NWIS sites of
  any start year (matches abstract); False uses the ≤1979 sites shown in Row 2.
  HUC12s are clipped to the CA outline for display only (some straddle the state line).

### Layout (REVISED 2026-10-28)
Transposed to 2 rows × 3 cols to fit the journal's 2-column size (7.25 in wide,
max 7.8 in deep incl. caption; captions set in 7.5 pt Myriad). Figure is built at
7.25 × 4.9 in, saved without bbox tight so the width is exact. Rows = stream (A–C)
/ groundwater (D–F); columns = new sites per decade, site locations, subwatershed
coverage. Arial (Myriad not installed); 6–8 pt text, 9 pt panel letters. The
"Row 1/2/3" names below refer to the old layout (now columns 1/2/3).

### Draft caption (2026-10-01; panel letters updated 2026-10-28)
> **Figure X.** Restored historical (1849–1980) monitoring sites compared with sites
> available from the USGS National Water Information System (NWIS) in California, for
> stream discharge (top row) and groundwater (bottom row). **(A, D)** Number of new monitoring
> sites per decade, by the decade each record begins. A restored location is counted once
> for each decade in which one of its records begins; each NWIS site is counted once, in
> the decade of its first record (records beginning ≤1979). **(B, E)** Locations of
> restored sites and NWIS sites with records beginning ≤1979. **(C, F)** California HUC12
> subwatersheds (n = 4,473; USGS Watershed Boundary Dataset) containing at least one NWIS
> site (blue; includes subwatersheds that also contain restored sites) or restored sites
> but no NWIS site (red). Here, NWIS sites of any
> start year are included, so red marks basins with no NWIS monitoring at any time. The restored records add 1,522 stream discharge basins
> (+104%) and 372 groundwater basins (+20%) with no NWIS sites. Restored sites are limited
> to dated records within California (undated records excluded; see Date Parsing Protocol). NWIS stream sites include daily-value, peak-flow, and
> field-measurement sites.

Caption TODOs: groundwater starts 1856, not 1849 (consider per-type ranges); describe
which site types `ca_groundwater_sites.csv` includes (not yet verified).

### Open items for this figure
- Optional: annotate C/F with headline gain ("+1,522 basins, +104%").
- Row 1 counting asymmetry (restored location counted once per start decade vs. NWIS
  once) — cumulative or "active sites per decade" variant not built.
- Convert script to a notebook once design settles.
- `data/analysis/nwis_sites/ca_groundwater_sites_raw.csv` (193 MB) is untracked: too big
  for GitHub. Add to .gitignore or regenerate on demand.

### Script and output
| File | Purpose |
|------|---------|
| `code/02_inventory/07_ca_combined_6panel.py` | Builds the figure (script for fast iteration; convert to notebook once design settles) |
| `manuscript/figures/ca_nwis_combined_6panel.png` | Draft output |

### Design decisions (flip if needed)
- `USE_EXPANDED_STREAM = True` — NWIS stream sites from `ca_stream_sites_expanded.csv`
  (daily + peak-flow + field measurements, 2,107 sites ≤1979). Set False for daily-value
  gauges only (1,763 sites, `ca_stream_sites.csv`).
- 1980 decade bin dropped from time series — only catches records starting exactly in
  1980 (n=2), plotted as a misleading drop to zero.
- Restored sites clipped to CA boundary (interstate sites excluded — consistent with
  the CA map framing, but note when quoting n's).
- "New sites per decade" semantics mirror the dissertation figures: a restored location
  can appear in multiple decades (one per record start), an NWIS site appears once.
  Statewide this mismatch is more visible — cumulative or "active sites per decade"
  (NWIS begin–end spans) variants would be more apples-to-apples; not yet built.

### Key result from Row 1
Stream discharge: restored sites outnumber NWIS in every decade (e.g., 1910s: 5,429
restored vs. 196 NWIS). Groundwater: crossover story — restored dominates pre-1930,
NWIS explodes after 1940 (13,647 new sites in 1960s). The asymmetry between columns
is the most interesting feature of the figure.

---

## Scope Extension Stats (abstract / manuscript) — July 23, 2026

### Purpose
Quantify how much the restored data extends the **temporal** and **spatial** scope
of CA streamflow and groundwater beyond what USGS NWIS makes digitally available.
Computed for an abstract sentence ("...extend the temporal and spatial scope of
streamflow and groundwater xx% and xx%, including previously ungauged basins").
Will be reused in the manuscript and Row 3 of the 6-panel figure.

### Headline numbers (USE THESE)
| | Streamflow | Groundwater |
|---|---|---|
| **Temporal extension** | **+42 yrs** (1849 vs NWIS 1891) | **+34 yrs** (1856 vs NWIS 1890) |
| — as % of pre-1980 era record | +47% | +38% |
| — as % of record to present (2025) | +31% | +25% |
| CA sites predating NWIS | 280 | 157 |
| **Spatial — previously ungauged basins** | **+1,522 basins (+104%)** | **+372 basins (+20%)** |
| restored HUC12 basins / NWIS basins | 2,909 / 1,460 | 1,177 / 1,889 |
| restored / NWIS % of CA's 4,473 HUC12 | 65% / 33% | 26% / 42% |

*Updated 2026-10-28 to dated restored sites only (Date Parsing Protocol). Previous
(all-sites) values: +1,611 (+110%) stream / +471 (+25%) GW; 272 stream sites predating NWIS.*

### Critical method notes (state in manuscript)
- **CA-clip is mandatory for the temporal figure.** The restored metadata includes
  out-of-state documents (Cape Fear NC, Ohio R. WV, Mississippi, Arkansas R.). The
  *unclipped* earliest streamflow record is 1820 (Cape Fear) — wrong for a CA claim.
  CA-clipped earliest is 1849 (Feather/Yuba/Bear/American Rivers, 5+ sites). Both
  temporal extensions rest on 150–270 CA sites, not single records — robust.
- **"Basin" = HUC12 subwatershed** (4,473 statewide). At HUC8 the ungauged signal
  nearly vanishes (+6% / +5%) because NWIS already touches almost every HUC8. HUC12
  is what makes "previously ungauged basins" true. Both restored + NWIS points are
  assigned to HUC12 by spatial join against the same WBD polygon layer.
- **NWIS streamflow = expanded set** (daily + peak-flow + field, 2,930 sites). Daily-
  value-only would lower NWIS basin coverage and *raise* the restored spatial gain.
- **% record-length is endpoint-sensitive** — report raw years (42, 34) as primary;
  % framings depend on choosing 1980 (era end) vs 2025 (present).

### Files produced
| File | Purpose |
|---|---|
| `code/02_inventory/08_scope_extension_stats.py` | Reproducible script; toggles `USE_EXPANDED_STREAM`, `RECORD_END_ERA` |
| `data/analysis/scope_extension_stats.csv` | One row per water type — all temporal + spatial stats |
| `data/analysis/spatial/ca_wbd/ca_huc8.geojson` | CA HUC8 subbasins (140), from USGS WBD MapServer 2026-07-23 |
| `data/analysis/spatial/ca_wbd/ca_huc12.geojson` | CA HUC12 subwatersheds (4,473), from USGS WBD MapServer 2026-07-23 |

---

## Date Parsing Protocol — October 28, 2026 (APPLIES TO ALL NUMBERS AND FIGURES)

### Why
The raw `dates_of_recording` field is blank for 11.4% of metadata rows, but the old
parser left 19.5% undated: it discarded open-ended records ("10/01/1964-Current Year"),
ignored 2-digit years, and silently dropped post-1980 years. The parser was also
duplicated in 00_data_prep.ipynb and build_table_level_metadata.py.

### Protocol (agreed with Annette)
- **One parser:** `code/02_inventory/date_parsing.py` (`add_date_columns`, `is_dated`,
  `DATED_STATUSES`). Never copy the logic; import it.
- **Valid years:** 1800 ≤ year ≤ document publication year (`Year Published`, ≤1980).
  A record can't postdate its publication; later years are LLM errors.
- **Publication year is never used to impute a start year** — only to cap open-ended
  records and reject impossible years.
- `date_status` per row: `parsed` / `trimmed` (some out-of-range years dropped) /
  `open_ended` (end = pub year) / `two_digit` (only clean MM/YY or MM/DD/YY dates or
  ranges → 19xx) count as **dated**; `implausible` / `unparseable` / `placeholder` /
  `missing` do not. `year_start.notna()` ⇔ dated, so existing `year_start` filters
  already follow the protocol.
- **Counting:** anything time-based (time series, decade maps, date ranges) uses dated
  rows only and reports n excluded. Sites are dated if ≥1 record is dated. Totals that
  don't depend on date (measurements, tables, type shares) keep all rows. **Spatial
  claims use dated sites only**, so all panels/numbers describe the same sites.

### Result
Dated rows: 80.5% → **83.8%** (91,746 of 109,440). Status counts: parsed 87,203;
open_ended 3,377; trimmed 801; two_digit 365; missing 12,527; placeholder 3,096;
unparseable 1,150; implausible 921. 75 previously-dated rows are now excluded (years
after publication). Groundwater is the weak category (~64% parsed; 21% blank).
No year_start value changed for rows dated by both parsers. Non-date columns of
processed_metadata.parquet and both table-level CSVs are unchanged (published total
still 2,512,776 measurements / 100,829 tables).

### Re-run status
- Re-run: 00_data_prep, build_table_level_metadata, filter_published_tables, 01, 02, 03
  notebooks, 07 (6-panel), 08 (scope stats). Figures in manuscript/figures updated.
- **04_santa_ynez.ipynb FAILS (not date-related):** `chap2map.gdb` can't be read in this
  Python env (fiona/GDB driver), so `rivers` is None. Needs an env fix, then re-run.
- Not yet re-run (they read the parquet, so will pick up new dates when run):
  09_sgma_groundwater_basins.py, dam_exploring 05–08, county_spatial_join.py.

### Validation (TODO — needs human grading)
`data/analysis/date_parsing/date_validation_sample.csv`: 100 rows stratified by
date_status (40 parsed, 15 open_ended, 15 two_digit, 10 trimmed, 10 implausible,
5 unparseable, 5 placeholder) with blank grader columns. Page images copied to
`data/analysis/date_parsing/pngs/` (file prefix = sample_no); 6 rows have no page image
anywhere (docs 2057 p1, 23894, 23996) and are skipped, so 94 rows get graded. Report
parser accuracy next to the 87% metadata accuracy. Coverage table:
`data/analysis/date_parsing/date_status_coverage.csv` (status × water type).

---

## Session Log — October 1, 2026

### Done
- **6-panel figure complete** (Row 3 = HUC12 basin coverage, caption drafted) — see 6-panel section above. Commit 7d51e5b.
- **SGMA groundwater basins** (`code/02_inventory/09_sgma_groundwater_basins.py` →
  `data/analysis/sgma_basin_groundwater_coverage.csv`; DWR 2019 prioritization layer in
  `data/analysis/spatial/sgma/`). Only 1 High/Medium basin has restored GW data and zero NWIS
  wells (SCRV–Fillmore) and it is **mislocated** (Sonoma wells, doc 1110 p199). Credible
  "restored is the only pre-1980 record" basins: **Elsinore Valley** (Medium; artesian well
  1904–1915, doc 2886; NWIS starts 2001) and **Coachella–Mission Creek** (Medium; 1936–1973,
  docs 2846/30040, analog-model reports). Stronger story: in 53 of 94 High/Medium basins the
  restored record predates the first NWIS well (e.g., Tulare Lake 1865 vs 1950; Santa Clara
  subbasin 2,169 meas., NWIS has 5 pre-1980 wells).
- **Location accuracy problem documented** — see section below.
- **Dam streamflow inventory** (`code/04_vignettes/dam_exploring/07_dam_streamflow_inventory.py`
  → `data/analysis/dam_exploring/dam_streamflow_inventory.csv` + `_sites.csv`). 64 major CA
  dams (NID, ≥100k af or removal candidate, completed ≤1979, associated structures excluded);
  58 have restored streamflow on the same river within 30 km (name + distance match).
  Most restored-only pre-dam data (years before any NWIS gauge on that river): Prado/Santa Ana
  (21 yrs), New Melones/Stanislaus (13), L.L. Anderson/MF American (12), El Capitan/San Diego
  R. (8), Bradbury/Santa Ynez (8). Removal candidates (Matilija, Scott, Cape Horn, Los Padres,
  Searsville) add little beyond NWIS. Imperial Diversion Dam is a false positive (NWIS file is
  CA-only; Colorado R. gauges are in AZ).
- **Dam vignette timeline figure — DRAFT** (`code/04_vignettes/dam_exploring/08_dam_vignette_timeline.py`
  → `manuscript/figures/dam_vignette_timeline.png`; NHD flowlines/waterbodies via
  `fetch_nhd_focal_rivers.py` → `data/analysis/spatial/nhd/`). Santa Ana + Santa Ynez rows:
  watershed map with numbered stations + station timeline (NWIS bar ≤1980; restored annual
  tables as solid ticks, multi-year tables as thin line; dam years as vertical lines).
  Stations matched by NAME (not coords) and paired to NWIS by location words.
  **Key result:** Santa Ynez station "below Gibraltar Dam nr Santa Barbara" has restored annual
  tables 1904–1916 — before Gibraltar Dam (1920) and before NWIS (1933). Santa Ana restored data
  mostly duplicates NWIS (pre-NWIS only Mentone 1894–95, San Bernardino 1916, unnamed 1887–88).

### Next steps (user to choose)
1. Santa Ynez focus: open the 1903–1920 tables, confirm the early "near Santa Barbara" gauge
   is the same site as NWIS 11123000, and if values parse, plot actual pre/post-dam flows
   (first figure using digitized values, not metadata).
2. Swap Santa Ana for New Melones/Stanislaus or El Capitan/San Diego R.
3. Small fixes: Prado triangle hides markers 3/8; labels drop "near/below".
- Queued figure idea: record length / overlap at the 1,646 restored sites strongly matched to
  NWIS (`data/analysis/dam_exploring/nwis_crossref_best_match.csv`).
- `ca_groundwater_sites_raw.csv` (193 MB) still untracked — too big for GitHub; gitignore it.

## Location Accuracy — Known Problem (Oct 1, 2026; awaiting collaborator decision)

**How coordinates were made:** Phi-4 read ONE page's Reducto text (no title, other pages,
geocoder, or TRS conversion) and returned `Actual_*` and `Inferred_*` lat/lon. Prompt:
"All data is from the State of California. If there is no latitude and longitude data...
use inference". `00_data_prep.ipynb` uses `actual.combine_first(inferred)` → 57,849 actual /
43,594 inferred / 7,997 none. A 52-row OpenStreetMap geocoding test (June 2025) never reached
production; its code is lost.

**Problem:** in a random 400-page sample of "actual" coordinates, **32% had no lat/long printed
on the page** (upper bound; regex-based). Example: doc 1110 p199 (Sonoma "Santa Rosa Valley"
wells, T.7N R.9W) placed at Fillmore, Ventura as "actual". Mislocated points land where NWIS
has no sites → likely **inflate the ungauged-basin numbers** (1,611 stream / 471 GW HUC12) and
affect HydroShare county folders. Document titles are NOT a usable check (reports span many
counties/states — user's point).

**Proposed plan (not approved yet):** (1) parse printed DMS coordinates from page text to
verify/correct; (2) relabel unprinted "actual" as inferred; (3) check inferred points against
the document's own verified-coordinate footprint, page-stated county, duplicate place names;
(4) publish confidence tiers; (5) report spatial claims under all vs verified+consistent.
Also check whether human validation graded "actual" coords against the page (87% figure).
Collaborator write-up: https://claude.ai/code/artifact/080b679c-fa0d-4306-862a-5478e5d6a00c

### Location validation progress — Oct 28, 2026
**Why the dissertation's ~86% lat/long accuracy differs from the 32% problem:** the test
sets graded only the `Actual_*` fields, and an empty Actual field scored as correct. Rows
where Actual was left empty scored 98–100%; rows where the LLM *gave* an actual coordinate
were correct only 39% (curated, n=89) / 60% (random, n=84). **Inferred coordinates were
never graded** (0 grades in either set). Also: the LLM was re-run after grading — actual
coords match production exactly, inferred coords differ between runs (e.g. 1520 p267),
so validation must score the PRODUCTION coordinates.

**Step 1 — Township/Range/Section reference (done):**
`code/01_digitization/04_metadata_extraction/location_validation/01_trs_reference.py`
looks up each T/R(/section) quoted on a test page in the BLM PLSS CadNSDI service and
uses the section (or township) centroid as an independent reference. Meridian is chosen
only from page evidence (named meridian, state abbreviation, county named on page,
or single candidate) — never from the LLM coordinate. Outputs in
`data/digitization_intermediates/04_metadata_extraction/validation/location_reference/`
(`trs_reference.csv`, `trs_reference_summary.csv`, `plss_cache.json`).

| Production coords vs PLSS reference | pages | page-median error | ≤5 km | ≤25 km |
|---|---|---|---|---|
| actual | 31 | 0.8 km | 97% | 100% |
| inferred | 11 | 17 km | 18% | 73% |

Worst inferred: doc 70038399 p529 (Bogard Guard Station, Lassen Co., placed 240 km
away). Unresolved: ~20 rows with >1 meridian candidate and no page evidence, ~10
out-of-state (Oregon) T/Rs not found. Small n — TRS pages only.

**Scale-up opportunity:** 30,901 production rows quote a T/R (7,616 of them have only
inferred coords; 23,285 labeled actual). The PLSS lookup can both VALIDATE these and
REPLACE inferred coords with ~1 km section centroids.
Next planned steps: (2) NWIS station-name reference for stream rows; (3) GNIS place-name
reference + human check; (4) run TRS lookup on all production rows.

**Step 2 — full-dataset PLSS lookup (done, Oct 28):**
`location_validation/02_trs_lookup_all.py` → `data/analysis/location_validation/trs_lookup_all.csv`
(one row per main_metadata row with an LLM T/R; `meta_row` = row position in main_metadata.csv)
+ `trs_lookup_summary.csv`; PLSS downloads cached in the same folder.
- Because the LLM's T/R parsing was unreliable (why the original prompt told it not to
  convert T/R), the LLM T/R is only a pointer: accepted only if the same T and R appear
  in the page's Reducto text (also matches "R,11 E" OCR and state well nos. "4N/1W-4A1");
  section used only if it also appears next to that T/R on the page, else township centroid.
- Meridian rules (page evidence only): named meridian > single candidate > county named on
  page > document meridian (all ≥3 resolved rows in the doc share one meridian). Each
  rule's references agree with "actual" coords 84–91% within 5 km → references are reliable.
- 36,000 rows have an LLM T/R: 32,400 verified on page; **24,752 resolved**; 7,493
  ambiguous meridian; 3,600 not verified; 155 no PLSS match.

| Production coord vs PLSS | rows | median | ≤5 km | ≤25 km | >100 km |
|---|---|---|---|---|---|
| actual, section ref | 15,121 | 0.7 km | 92% | 96% | 1.2% |
| actual, township ref | 2,111 | 4.2 km | 68% | 93% | 3.2% |
| **inferred, section ref** | 4,146 | **33 km** | **7%** | 40% | 18% |
| **inferred, township ref** | 1,771 | **25 km** | **8%** | 51% | 15% |

**Decision pending (Annette):** whether to REPLACE the 5,917 inferred coords that have a
PLSS reference, and flag the 1,381 "actual" coords that disagree with it (>5 km from a
section reference or >10 km from a township reference).
Nothing in main_metadata / processed_metadata has been changed.

## Location Hierarchy — DECISION Oct 28, 2026 (Annette)
LLM-inferred coordinates are presumed inaccurate and are NOT used. Unverified "actual"
coordinates are treated the same. Each row gets a `coord_method` + precision:
1. **printed** lat/long confirmed in page text  2. **PLSS** T/R/section (T/R must appear in
page text)  3. **named site** (NWIS station name, then GNIS town/spring/dam; river name only
with a place qualifier)  4. **coarse** (county; not used for HUC12 claims)  5. **unlocated**.
Raw LLM values stay in main_metadata.csv. Validate each tier's method against tier-1 rows.
Hold figure/stat re-runs until all tiers are built.

**Tier 1 done** — `location_validation/03_printed_coords.py` →
`data/analysis/location_validation/tier1_printed.csv`, `printed_coords_by_page.csv`.
An LLM "actual" coordinate counts only if BOTH lat and lon match a printed value on the page
(±0.005° with seconds, ±0.02° deg-min only). Of 57,845 LLM "actual" rows:
printed_match 29,351 + match without lat/long keyword (OCR-garbled "La", "L") 4,039 =
**33,390 tier 1 (57.7%)**; **not printed 22,520 (38.9%)** (spot-check: no coordinate of
any kind on those pages); page prints other coords 1,620 (2.8%); no page text 303.
Tier 1 = 30.5% of all rows; by type: stream 39%, reservoir 49%, springs 19%,
**groundwater 1.4%** (GW will depend on PLSS / state well numbers).
Leftover: 1,810 non-tier-1 rows sit on pages that print coords (1,257 on pages printing
exactly one coordinate) — possible tier 1b (not approved yet).
**REJECTED (Annette, Oct 28):** no adjacent-page inference (e.g. carrying a station's
printed LOCATION from the previous page). Every location must come from the row's own page.

**Tier 2 note:** the PLSS document-meridian rule (borrows meridian evidence from other pages
of the same report) is now behind `USE_DOCUMENT_MERIDIAN = False` in 02_trs_lookup_all.py,
per the same-page rule; it resolved ~1,225 extra rows when on. Awaiting Annette's call.

**Tier 3a done — NWIS station number on page** (`05_nwis_station_numbers.py` →
`tier3a_nwis_station.csv`): an 8-digit NWIS CA station number printed on the row's page,
AND a distinctive word from the row's water-source name shared with the NWIS station name
(town names in the location text alone don't count — fixed a "Chesbro Reservoir" →
"Pajaro R at Watsonville" false match). Validation on tier-1 rows (n=9,024): median
0.09 km, 93% ≤1 km, 98% ≤5 km. Adds 6,797 locations to unlocated rows.

**Combined table** (`04_build_locations.py` → `data/analysis/location_validation/locations.csv`):
printed 33,390 · nwis_station 6,797 · plss_section 6,143 · plss_township 2,261 ·
**located 48,591 (44%)** · unlocated 60,849. Located share: stream 55%, reservoir 54%,
irrigation 27%, springs 26%, groundwater 14%. Next: tier 3b GNIS named places.

**Tier 3b done — GNIS named places** (`06_gnis_named_places.py` → `tier3b_gnis.csv`,
`tier3b_gnis_validation.csv`). Uses the 2021 archived full GNIS CA file
(`data/analysis/spatial/gnis/CA_Features_20210825.txt`; the current GNIS dropped locales,
dams, post offices). Place candidates from the LLM water-source/location text (pointer),
must also appear in the page text; point-like classes only (no streams/valleys);
duplicates resolved only by a county named on the page. Validation vs printed coords
(n=17,453): "at Place" median 1.0 km (90% ≤5 km), dam 0.7 km (89%), reservoir 1.7 km (94%);
"near Place" 6.0 km (40%), place-only 6.2 km (39%), lake 2.8 km, spring 4.2 km,
below 3.7 km (22% >25 km), above 8.2 km (32% >25 km). Adds 19,383 locations.

**locations.csv now has `precision_class`:** precise = printed, plss_section, nwis_station,
gnis at/dam/reservoir (→ HUC12 claims); approximate = plss_township, gnis near/place/lake/
spring/below/above (→ county/HUC8 scale). Totals: **precise 49,317 · approximate 18,657 ·
unlocated 41,466** (of 109,440). Precise share: stream 57%, reservoir 62%, springs 24%,
irrigation 26%, **groundwater 7%** (GW 80% unlocated).

**Groundwater review + scoping (Oct 28):** `data/analysis/location_validation/groundwater_location_review.csv`
(+ README) = all 20,624 GW rows with LLM fields + every location step. Annette reopened the
adjacent-page question for GW (locations often on the first page of a long table).
40-page scoping sample of unlocated GW pages (`gw_scoping/gw_scoping_classification.csv` + page
PNGs; Claude first pass, Annette to review): **25/40 have the location on the SAME page**, mostly as
well numbers my parser missed — short form "5/2W-26Q1"/"7/35-36E1" (11), full state well no.
incl. OCR variants "165/17E-17801 S", "225/39E-8C8 M" (8), section IDs under a T/R header (2),
report-specific compact numbers "4523Q1" (2), Nevada "11/24-22dc1" (1), "N miles NE of Town"
text (2). Only 2–4/40 truly need a previous page; 2 need document-level ID links (map no.);
5 are not site data (narrative, TOC, area summaries); 3 garbled OCR; 1 surface-water page
misclassified as GW. Parser gap found: the LLM often puts the well number in
watersource_name, which 02_trs_lookup_all.py never read.

**Tier 2b done — same-page well numbers** (`07_well_numbers.py` → `tier2b_well_numbers.csv`,
`_validation.csv`; shared BLM helpers in `plss_utils.py`). Parses full ("10S/6E-23B1", OCR repair
"165/17E-17801 S"), short ("5/2W-26Q1") and section-under-T/R-heading ("4R1") well numbers from
the page text; missing N/S, E/W, meridian settled from the same page only (T/R on page,
meridian letter, single candidate, town in the well heading ≤40 km, county on page). Rows
matched to a well by well no. in LLM text / owner name in heading / only well on page; rows
covering several wells get a centroid only if all within 5 km (approximate). Not applied to
stream/reservoir/precip rows (first run put "Clear Lake at Lakeport" at a well 180 km away).
Location = NWIS GW site coords when the full well no. matches an NWIS name, else section centroid.
Validation: section centroid vs NWIS well ~0.75 km for every direction rule; unbiased direction
check (inferred letters vs NWIS-confirmed option): only candidate 97%, town in heading 91%,
county on page 87%; "T/R on page + county" 0/15 → disabled. plss_wellno vs printed coords:
median 0.8 km (n=21). Adds 2,569 locations.
**Combined now: precise 51,921 · approximate 18,622 · unlocated 38,897.** Groundwater: precise
18% (was 7%), unlocated 68% (was 80%). Remaining unlocated GW (13,905): no well no. on page
8,533; well no. but directions/meridian ambiguous 2,830; row not matched to one well 1,154;
several wells >5 km apart 1,068; section not found 313.

**Document-level + continuation evidence — APPROVED by Annette (Oct 28), labeled `evidence_scope`**
(page | document | continuation) in locations.csv so any figure can include/exclude them.
- *Document level* (07_well_numbers.py pass 2; 02_trs_lookup_all.py USE_DOCUMENT_MERIDIAN=True):
  other pages of the report settle only meridian/N-S/E-W letters; location still from the row's
  page. Unbiased NWIS check: document meridian 100% (n=192), same T/R elsewhere in report 97%
  (n=71). Dropped: report quadrant (82%, n=17); "T/R on page" + any tie-breaker (0/15, 0/14).
  Adds ~709 well rows + TRS doc-meridian rows; 1,191 rows have scope=document.
- *Continuation* (08_continuation.py → tier4_continuation.csv, _validation, _holdout): explicit
  "Continued" marker + same table no./title/header with the previous page; inherit from a located
  row on the chain. Loose rules failed hold-out (precise→precise: owner 62% ≤2 km / 16% >10 km;
  single-site 45% / 16%; section-only well IDs 52% / 42%). STRICT=True kept: full well numbers
  (20/20 exact) and ≥2 shared owner words (73% ≤2 km, 5% >10 km → capped at approximate).
  Adds only 159 rows. Why so few: of 5,398 unlocated non-stream pages, 1,622 lack a continuation
  marker, 960 have the previous page missing (not digitized), 1,213 previous page not similar,
  739 chain has no located page; only 731 reach a located page.
**Combined now: precise 52,711 · approximate 18,985 · unlocated 37,744** (scope: page 70,346;
document 1,191; continuation 159). Groundwater: precise 22%, approximate 15%, unlocated 63%.

## Archived snapshots (Oct 28, 2026)
Before the location rebuild entered the main pipeline, two frozen copies were made in
`archive/` (see `archive/README.md`): `2026-10-01_original_pipeline/` (exact original approach,
from git commit eb05a7b, also tagged `original-pipeline-2026-10-01`) and
`2026-10-28_after_date_protocol/` (date protocol applied, LLM locations). Do not edit them.
~870 MB together and untracked; consider adding `archive/` to .gitignore (the tag already
preserves the original in git). The location rebuild must write a NEW HydroShare folder.
**HydroShare: DEFERRED (Annette, Oct 28).** Do not touch `data/hydroshare/` or the HydroShare
scripts until all date/location edits are finished; the files will be rewritten/corrected at the end.

## Location rebuild IN THE PIPELINE (Oct 28, 2026) — CURRENT NUMBERS
`00_data_prep.ipynb` now joins `locations.csv`: `lat_combined/lon_combined` = rebuilt locations
(`coord_method`, `precision_km`, `precision_class`, `evidence_scope`); old LLM values kept as
`lat_llm/lon_llm/coord_source_llm`. Statistics/figures: HUC12 basins AND temporal stats use
PRECISE, DATED, CA-clipped sites only (`08_scope_extension_stats.py`, `07_ca_combined_6panel.py`).
GNIS step now skips out-of-state pages (Arkansas R. at Poncha Creek, Colo., had been placed in
CA with an LLM-typo date of 1833). Fixed: shell-heredoc edits had turned `\b` into backspace
chars in 06_gnis and 01_trs_reference regexes; 08_continuation targets now independent of its
own earlier output.

| | Original (Oct 1) | + date protocol | **+ location rebuild (current)** |
|---|---|---|---|
| Stream: previously ungauged HUC12 | +1,611 (+110%) | +1,522 (+104%) | **+370 (+25%)** |
| GW: previously ungauged HUC12 | +471 (+25%) | +372 (+20%) | **+65 (+3%)** |
| Stream: earliest CA record / extension | 1849 / +42 yr | 1849 / +42 yr | **1861 / +30 yr** |
| GW: earliest CA record / extension | 1856 / +34 yr | 1856 / +34 yr | **1871 / +19 yr** |
| CA sites predating NWIS (stream / GW) | 272 / 157 | 280 / 157 | **61 / 19** |
| Restored CA sites mapped (stream / GW) | 38,741 / 5,981 | 34,664 / 4,463 | **12,527 / 2,300** (precise only) |

6-panel figure: panel F now shows HUC8 (dark) and HUC12 (grey) boundary lines
(`SHOW_HUC_LINES = {'gw'}`); figure 7.25 × 5.2 in. Not yet re-run with new locations:
notebooks 01–04 (02/03 maps use lat_combined), dam_exploring scripts, 09_sgma. HydroShare deferred.

**DECISION (Annette, Oct 28): analyses and maps use PRECISE + APPROXIMATE locations.**
`PRECISION_CLASSES = {'precise', 'approximate'}` in 07_ca_combined_6panel.py and
08_scope_extension_stats.py (set {'precise'} for the conservative version).
Current numbers (dated, located, CA): stream +543 previously-ungauged HUC12 (+37%), GW +89 (+5%);
mapped restored sites 13,373 stream / 2,772 GW. Raw earliest records are NOT credible yet:
stream "1850" = an INDEX page (doc 2252 p944, LLM-invented date) → first credible stream data
are the Dec 1861–Jan 1862 flood peaks (printed coords; +30 yr vs NWIS 1891); GW "1868" = estimated
recharge average 1868–1964 (doc 2345), GW 1870–75 dates in 1905 well tables look like year-drilled,
not measurement dates → GW start year needs manual review (ties into the date plan: separate
table-data dates from other dates; drop index/contents pages).

**GW recovery round 1 (Oct 28):**
- *County in report title* ("… Fresno and Kings Counties"; only "X County/Counties" phrases count)
  settles well-number directions/meridian (scope document): 98% NWIS-confirmed (n=686), section
  vs NWIS well 0.8 km → kept in 07_well_numbers.py (+1,211 located well rows). Same rule in the
  TRS lookup: 0/6 agreed with printed coords (median 447 km) → `USE_TITLE_COUNTY = False` there.
- *Page types* (`code/02_inventory/10_page_types.py` → `data/analysis/page_types.csv`; joined in
  00_data_prep as `page_type` / `is_data_page`): index_contents 410 pages (2,995 rows), narrative
  (no table at all) 38 pages (71 rows), no_page_text 346 pages (kept), data. Non-data pages are
  excluded from maps/stats (07, 08 filter `is_data_page`). This removed the false "1850" stream
  record (index page doc 2252 p944): stream earliest now 1861 (credible flood peaks).
- Result: GW precise 4,428→5,238, approximate 3,095→3,248, unlocated 13,067→12,104 (41% located);
  CA dated GW sites 2,772→3,025. Stats: stream +542 basins (+37%), GW +94 (+5%); stream 1861 / +30 yr;
  GW "1868" is still the recharge-estimate artifact (needs date-plan fix). Breakdown table:
  `data/analysis/location_validation/location_breakdown_by_water_type.csv`.

**GW recovery round 2 — B118 basin tier (Oct 28):** `location_validation/09_b118_basins.py` →
`tier5_b118_basins.csv`, `tier5_b118_validation.csv`. Groundwater rows (data pages, otherwise
unlocated) whose LLM text names a DWR Bulletin 118 basin/subbasin that also appears on the page
(one-word names need "Valley/Basin/Area") get a point inside the B118 polygon; new
`precision_class = 'basin'` (precision = basin equivalent radius, median ~16 km). NOT used for
HUC12 stats or point maps (PRECISION_CLASSES excludes it); for basin-level summaries only.
Result: +793 GW rows (593 basin, 200 subbasin); validation: precise GW points inside the matched
polygon (+2 km) 82% (basin, n=459), 89% (subbasin, n=18). Small because most area-only pages name
non-B118 areas ("Mendota-Huron area"). NOTE: lat_combined now also holds basin points — any
notebook mapping lat_combined must filter precision_class (07/08 already do).
Totals now: precise 53,513 · approximate 18,738 · basin 793 · unlocated 36,396.

## Session Log — October 28, 2026 (resume here)
Done today: date protocol (date_parsing.py); 6-panel figure redesigned to 2×3 journal layout
(7.25×5.2 in) with C/F = NWIS vs restored-only basins and HUC8/HUC12 lines on F; full location
rebuild (printed → well numbers → PLSS → NWIS station → GNIS → continuation → B118 basin), joined
into processed_metadata; page types; original pipeline archived (archive/, tag
original-pipeline-2026-10-01). Current headline: stream +542 HUC12 (+37%), +30 yr (1861);
GW +94 HUC12 (+5%), GW start year not yet credible.
**Open / next:**
1. Annette is grading `data/analysis/date_parsing/date_validation_sample.csv` (94 rows) and will
   return to the date plan (table-level dates, grounded LLM fallback; fixes GW "1868" artifact).
2. Re-run notebooks 01–04, dam_exploring, 09_sgma with new locations (filter precision_class).
3. Remaining GW options: report-specific decoders (Mokelumne "4523Q1", Mendota-Huron), distance-
   direction text, map-number links. 4. Update abstract/caption numbers.
5. HydroShare: deferred until everything else is final (rebuild into a NEW folder).
6. Nothing committed to git yet today (large set of modified + new files).

## Session Log — October 29, 2026 (resume here)

### Decisions (Annette)
- Stop grading the date sample; use the 69 graded rows. LLM re-run not possible (no UCSB/Phi-4 access).
- Scope: improve locations + dates and findability metadata; NOT cleaning table values.
- Do all suggestions EXCEPT the optional ones (no water-type disagreement flag, no unit-label normalization):
  (1) table-level dates, (2) one searchable table-level catalog (+ GeoPackage), (3) search keys:
  HUC8/HUC12, county, likely NWIS station no. (+ match distance), B118 basin, (4) per-table
  digitization-quality flag, (5) user guide + metadata_key updates, then rebuild figures/stats/HydroShare.
- Out-of-CA data: publish it (labeled), just not featured in the paper.
- Tribal data sovereignty: withhold data on FEDERALLY recognized tribal lands (state-recognized optional,
  toggle off); offer them to tribes directly, do not publish.

### Done
- `code/02_inventory/11_page_blocks_cache.py` → `data/analysis/page_blocks.parquet` (all Reducto blocks,
  852,870; Table blocks numbered table_idx = CSV table number; counts match CSVs on 99.8% of pages).
- **Tribal screen** `12_tribal_lands_screen.py` (Census TIGER 2023 AIANNH in
  `data/analysis/spatial/tribal_lands/aiannh_2023/`) → `data/analysis/tribal_lands/tribal_screen_*.csv`.
  Unit = PAGE. Withheld if a rebuilt location is inside an area, a precise point within its precision
  (min 0.5 km), or the page names a reservation/rancheria. Result: **1,752 pages withheld** (215 docs,
  4,128 metadata rows); 2,129 more pages "possible" (approximate/basin radius touches an area) are a
  REVIEW list, not withheld (`WITHHOLD_POSSIBLE=False`). No hits on state-recognized lands.
  `13_copy_withheld_tribal.py` COPIED (not moved) pages to `data/withheld_tribal/{area}/` (117 folders,
  1.1 GB, git-ignored, manifest `_withheld_manifest.csv`). data/digitized unchanged.
- **Table-level dates** `html_table.py` (HTML parser), `table_dates.py` (tiered protocol: A date column →
  B table text → C caption → D page text; period-of-record and construction years kept separately;
  USGS section headings like "PERIOD OF RECORD.--"/"EXTREMES.--" classify whole sections),
  `14_table_dates.py` → `data/analysis/table_dates.csv` (106,301 tables, 92.6% dated; has caption = table
  title) + `table_row_dates.parquet` (row years + site label for row matching).
  Validation `validation/score_dates_vs_grades.py` → `data/analysis/date_parsing/date_method_validation*.csv`:
  exact 44/64 dated graded rows (old LLM string 26/64); exact or ±1 yr 49 vs 27. By tier: caption 24/27,
  table text 10/17, date column 5/8. Grader "truth" for samples 3/7/11 = year drilled (now construction_*).
- `15_entry_dates.py` → `data/analysis/entry_dates.csv` (per meta_row; match row → table caption/single
  table → page union). Entries dated 82.1% (LLM string 83.8%); same start year as LLM only 51.4% —
  **NEEDS REVIEW before use** (page-union entries may be too broad; check row/caption matching).

### Not done / next
1. Review entry_dates (low agreement), then add an `apply_table_dates()` in date_parsing.py and wire into
   00_data_prep + build_table_level_metadata (use table_dates per table directly; that script also still
   uses raw LLM coords — catalog should use rebuilt locations). Keep LLM dates as *_llm columns.
2. Catalog + search keys + quality flags + GeoPackage (exclude withheld tribal pages; label out-of-CA).
3. Decide: exclude tribal pages from paper figures/stats too (recommended; maps show site points).
4. Re-run figures/stats (07, 08, notebooks 01–04, dam scripts, 09_sgma); user guide; HydroShare into NEW folder.
5. Note: repo is on GitHub (ecohydro/usgs_extract); location/metadata files there include points on tribal lands.
6. Nothing committed today.

## Session Log — November 3, 2026 (resume here)
- Committed + pushed Oct 29 work (6be1204). New data files (page_blocks, table/entry/row dates,
  tribal shapefile + screen outputs) are git-ignored; existing tracked data left as is (Annette).
- **entry_dates review — diagnostics.** Of 86,280 entries dated by both methods, 51.4% same start year.
  Disagreements by explanation (share of all 86,280):
  LLM start = table period-of-record start 18.3% (known LLM error → new is right) ·
  water-year off-by-one 8.8% (new = LLM−1 for 6,527, LLM+1 for 2,566; Oct–Sep convention) ·
  new range wider than LLM 12.2% (median 8 extra years) · LLM range wider 5.9% ·
  partial overlap 1.1% · no overlap 2.1% (median 15 yr apart) · year drilled 0.2%.
  By source: page-union entries agree 43% but 28% of them are period-of-record cases, so
  "page union too broad" is not the main problem.
- **Bugs found (not fixed yet):** (1) start = min / end = max over ALL date mentions, so stray years
  widen ranges: flood notes in captions ("Flood of Dec. 21, 1955"), PERIOD remarks, well IDs/sample
  dates in table text, an "1822" in a Delta table. Mostly tier B_table_text and page union.
  (2) "year ending Sept. 30, YYYY" inconsistent: OCR hyphen "Sep- tember" → start YYYY instead of YYYY−1.
- **Targeted grading sample (Annette to grade):**
  `data/analysis/date_parsing/entry_date_review/entry_date_review_sample.csv` (30 rows + pngs/,
  pngs git-ignored), built by `code/02_inventory/validation/entry_date_review_sample.py`.
  10 new-wider, 5 LLM-wider, 8 no-overlap, 3 partial overlap, 2 period-of-record, 2 off-by-one;
  half from page-union entries. Next: grade → fix range rule (e.g. primary caption water year /
  modal year over min–max) + water-year regex → re-score → wire into pipeline.
- **DECISION (Annette, Nov 3): tribal pages are withheld from paper statistics and figures too**
  (not only from publication). Paper reports only the aggregate amount (below). Figure/stat re-runs
  must filter withheld pages (tribal_screen_pages.csv, withhold=True). The catalog already excludes them.
- **Tribal withheld statistics** — `code/02_inventory/16_tribal_withheld_stats.py` →
  `data/analysis/tribal_lands/tribal_withheld_stats.csv`, `_by_water_type.csv`, `_by_area.csv` (git-ignored).
  Counted on the same basis as the published inventory (LLM-processed tables):
  1,752 pages · 215 documents · 110 federally recognized areas (89 touch CA) · 2,672 tables (2.7%) ·
  **64,386 measurements (2.6% of 2,512,776)**; all digitized tables incl. non-LLM: 2,702 / 65,699.
  California areas 53,956 meas. (1,403 pages); other states 7,278 (199 pages); named in page text
  only (no located point) 3,152 (150 pages). Stream discharge 92% (59,020), reservoir 1,445,
  groundwater 1,309, irrigation 1,213. Top areas: Hoopa Valley, Fort Yuma, Yurok, Karuk, Pala.
  Year range NOT reportable yet (earliest "1800" = parser floor; wait for the date fix).
- **Table-level metadata now uses REBUILT locations** (`build_table_level_metadata.py`, step 3b
  `build_table_locations`): each table takes the location of a metadata entry tied to it (via
  entry_dates.csv `matched_tables`), else the best located entry on its page; best precision class
  wins. New columns: coord_method, precision_km, precision_class, evidence_scope, location_match
  (entry/page/none), n_site_locations, site_spread_km, location_ambiguous (>5 km), meta_row_location;
  old LLM coords kept as lat_llm/lon_llm. Water types / counts unchanged (2,512,776 / 100,829 published).
  Published tables: precise 68,122 · approximate 14,953 · basin 335 · unlocated 17,419;
  location_ambiguous ~5,300. Re-ran filter_published_tables.py. Both CSVs now ~51–52 MB (over GitHub's
  50 MB warning, under the 100 MB limit). Downstream scripts reading lat_combined (county_spatial_join,
  09_sgma, dam scripts, hydroshare_*) not yet re-run.
- **Per-table digitization quality** — `code/02_inventory/17_table_quality.py` → `data/analysis/table_quality.csv`
  (git-ignored): frac_empty, frac_numeric, frac_ocr_suspect (letter-for-digit swaps incl. lone "o"),
  one_column → digitization_quality good/fair/poor + table_content numeric/mixed/text. Heuristic,
  not validated. Empty cells barely penalized (daily tables leave months blank).
- **Catalog** — `code/02_inventory/18_build_catalog.py` → `data/catalog/restored_tables_catalog.csv` +
  `.gpkg` + `catalog_summary.csv` (git-ignored); column key `data/metadata/catalog_key.txt`.
  Published tables minus tribal pages (98,157). Search keys: state/in_california, county (precise+approx),
  HUC8 (precise+approx), HUC12 (precise only), B118 basin, NWIS site (station number on page /
  nearest named stream site ≤5 km / nearest well ≤0.5 km). Validation: nearest_named picks the printed
  station for 17,784 of 18,791 tables (95%). Dates in the catalog are PROVISIONAL (table_dates.csv).
- **APPROVED manuscript sentence (Annette, Nov 3):** "We restored approximately 64,000 measurements
  in 2,672 tables from 110 federally recognized tribal areas, most in California. To respect tribal
  data sovereignty, these data are not published or featured here and are being provided directly
  to the tribes." (Update wording to past tense once the data are actually delivered; numbers from
  tribal_withheld_stats.csv — re-check if the screen or published table set changes.)
- **table_quality.csv done** (106,191 tables): good 93,778 · fair 6,055 · poor 6,358; content numeric
  83,131 · mixed 19,782 · text 3,278. Nearly all poor/fair come from frac_ocr_suspect (poor: 6,197 of
  6,358 are >5% OCR-suspect cells; only 88 no rows, 162 >90% empty, 114 one-column). 95th pct
  frac_ocr_suspect 6.7%, 99th 32%. TODO: spot-check poor tables — the lone "o" = zero rule may
  over-flag daily tables whose values are easily recoverable; consider counting it separately.
- **Tribal sheet** added to 18_build_catalog.py (`write_tribal_sheet`): data/withheld_tribal/
  tribal_tables_catalog.csv + .xlsx ('areas' + 'tables' sheets), one row per table per tribal area,
  with withheld_copy_folder, area_in_california and blank contact columns (tribe_contact,
  date_contacted, date_data_shared, contact_notes). Annette wants this to sort/share with tribes.

### Next session (resume here, Nov 4)
1. Spot-check poor quality flags, then RUN `18_build_catalog.py` (catalog + GeoPackage + tribal sheet);
   check outputs; update catalog_summary numbers here.
2. Annette's grading of entry_date_review_sample.csv → fix range rule + water-year regex → re-score →
   wire table dates into 00_data_prep / build_table_level_metadata → rebuild catalog dates.
3. Remaining list: fix 04_santa_ynez GDB env; user guide; re-run figures/stats with tribal pages
   excluded (07, 08, notebooks 01–04, dam scripts, 09_sgma, county_spatial_join); HydroShare (new folder).
4. Uncommitted: 16/17/18 scripts, entry_date_review_sample.py, build_table_level_metadata.py,
   catalog_key.txt, .gitignore, CLAUDE.md, and the two tracked table-level CSVs (now ~51–52 MB:
   GitHub warns over 50 MB, limit 100 MB).

## Session Log — November 4, 2026 (resume here)
- **Quality flag fixed** (`17_table_quality.py`). Spot-check of 1,500 "poor" tables: 96% were poor only
  because of lone "o"/"O" cells, which are zeros in old USGS type (zero-flow days in daily tables;
  e.g. 658 p126). These now go in a separate `frac_letter_zero` column, not used in the flag.
  All tables: good 103,268 · fair 2,571 · **poor 352** (was 6,358). catalog_key.txt updated.
- **Catalog built** (`18_build_catalog.py` → `data/catalog/`): 98,157 tables · 2,448,390 measurements
  (= 2,512,776 − 64,386 tribal) · 584 docs · 61,982 pages. Located 80,816 (precise 65,880 ·
  approximate 14,602 · basin 334) · unlocated 17,341 · location_ambiguous 4,999. In CA 77,016,
  outside CA 3,333. County 80,308 · HUC8 78,224 · HUC12 62,565 · B118 35,833. Dated (provisional)
  92,139. NWIS: nearest_named 36,062 · station no. on page 18,887 · nearest_well 2,760
  (nearest_named = printed station 17,784 of 18,791). Quality: good 95,447 · fair 2,405 · poor 305.
- **Tribal sheet**: 2,672 tables, 2,891 table-area rows. It lists 113 named areas + an
  "unmatched" bucket. The 110 in tribal_withheld_stats counts areas from located points only;
  3 more (Cahuilla, Jicarilla Apache, Kaibab) are named in page text only (6 tables). The approved
  manuscript sentence says "110"; Annette to decide whether to say 113.
- Entry-date review sample still ungraded → date fix (range rule + water-year regex) still waiting.

### Next
1. Annette grades entry_date_review_sample.csv → fix dates → wire into pipeline → rebuild catalog.
2. Fix 04_santa_ynez GDB env; user guide; re-run figures/stats excluding tribal pages (07, 08,
   notebooks 01–04, dam scripts, 09_sgma, county_spatial_join); HydroShare into a NEW folder.
- **Value counts — a better measurement estimate** (`19_value_counts.py` → `data/analysis/table_value_counts.csv`,
  `_summary.csv`, `_calendar.csv`). number_rows counts printed lines; a daily table row holds up to 12 values.
  Value = numeric token in a data cell (label column, summary rows, date/year/ID/site-description columns
  excluded; merged cells count each number; lone "o" = zero counts; daily grids count only day-labeled rows).
  **Published: 18,775,838 values in 2,512,776 rows (7.5 per row)**; catalog 18,338,405; tribal withheld 437,433.
  Stream 16.9M, reservoir 0.73M, groundwater 0.65M. 62% of values are in daily grids (34,739 tables).
  *Calendar check* (daily grids: one value per day per month with record): found 99.2% of expected values;
  75% of month columns exact; 66% of tables within 2% (28% under, partly real partial-record months; 6% over).
  *Hand check* for non-daily tables: `validation/value_count_review_sample.py` →
  `data/analysis/value_counts/review/` (20 short tables, 20–120 values each, blind; row 4 has no PNG; README has
  counting rules). Annette to grade when time allows.
  **Plan for paper:** keep 2,512,776 as "table rows" (conservative); add "~18.8 million recorded values".
  Abstract / Table 6 / Fig 7 / tribal sentence NOT changed yet (tribal: 437,433 values).
- **Value counts revised after Annette's hand check** (16 scorable tables; 4 marked not-measurement content).
  Fixes: date cells ("Nov. 17", "9, 1961", "1929-36") count 0 and mostly-date columns are dropped; more excluded
  headers (period/record, difference, distance, ratio, regression, bare "Total"). Sample: script/true total 1.47 →
  **1.22**, median per-table ratio 1.74 → **1.04**. Remaining excess is mostly Reducto merging two printed tables
  into one CSV (rows 13, 19). Scorer: `validation/score_value_counts_vs_review.py`.
  **Published values now 18,522,112** (catalog 18,091,550; tribal withheld 430,562). Daily grids (62% of values)
  pass the calendar check (99.2% of expected found); non-daily may still run ~20% high → say "about 18 million".

## Date Fix — November 4, 2026 (after Annette graded entry_date_review_sample.csv)
**Grades (30 rows):** new right/both 20, LLM right 3, neither 7. Failure causes, all fixed in `table_dates.py`:
period-of-record text in captions ("storage began", "to current year"); nearest caption block now wins (not a
union of 4 blocks); "land-surface datum" no longer excludes a "water year ..." title; OCR'd extremes remarks
("Fighest", "below lsd"); word breaks ("com- pleted", "Sep- tember", "Sep tember"); OCR'd month abbreviations
("Har. 1") no longer split clauses; well numbers no longer parsed as dates (mixed separators "4/28-1802",
"6/11-13J1", digit-dash bare years); rotated tables (row labeled "Date"; "Period of record"/"known floods" rows
excluded); unlabeled bare-year columns; isolated stray years (>30 yr from all others, not month-named) dropped;
empty tables get no date. **New labeled fallbacks** (only if no tier A–D date): E_period_of_record (grader's
choice for row 3), F_construction_year (rows 16/21), and in 15_entry_dates G_llm_grounded (LLM date whose start
and end years are printed on the page; only on pages with a non-empty table). Grounded LLM start = table start
57% (ungrounded 17%) where both exist → keep as labeled fallback only.
**Scores:** review sample 17/30 → **25/30** ok (LLM 6/30); earlier 64-row sample exact 44 → **46**,
exact-or-±1 49 → **52** (LLM 26/27). Fallbacks hurt 2 earlier-sample rows (5, 7), helped 4.
Scorer: `validation/score_entry_dates_vs_review.py` (row 3 truth = 1941–53 per grader note).
**Wired in:** `date_parsing.py` has `apply_entry_dates()` / `apply_table_dates()`, `TIER_STATUS`,
`FALLBACK_STATUSES`, `is_dated(df, include_fallbacks=True)`. 00_data_prep → processed_metadata.parquet uses
entry dates (dated 85.5%; strict 81.5%; LLM string 83.8%); build_table_level_metadata uses table dates
(new cols date_tier, por_*, construction_*, *_llm). Measurement/table totals unchanged. Catalog rebuilt:
93,392 of 98,157 tables dated (undated 4,765). `year_start.notna()` ⇔ is_dated still holds.
**Not yet re-run with new dates:** 07, 08, notebooks 01–04, dam scripts, 09_sgma (also need tribal exclusion).
Earliest-record claims (stream 1861, GW start) must be re-checked after that re-run.

## Paper Figures Re-run — November 4, 2026 (priorities: USA map, Fig 7, CA maps, Santa Ynez, 6-panel)
SB notebook (03) skipped for now; dam scripts / 09_sgma not re-run (exploratory).
- **Shared filters** `code/02_inventory/paper_filters.py`: `drop_tribal()` (withheld tribal pages out of all
  paper figures/stats), `located()` (precise + approximate rebuilt points only). Used by 01, 02, 04, 07, 08.
- **Time-based outputs use measurement dates only** (`is_dated(df, include_fallbacks=False)` /
  `~date_status.isin(FALLBACK_STATUSES)`): 07, 08, Fig 7 decades, CA decade map, Santa Ynez decades.
- **More date rules** (`table_dates.py`, found from figure artifacts): century misreads (1834 for 1934),
  pre-1850 years need a month name / water-year range / run of ≥3 years, round-number value columns
  (1800, 1810 ... = elevations/flows), long table titles are not date-column headers, unlabeled year
  columns skipped in day/max/min/mean grids. Review scores unchanged (25/30; 52/64).
- **Earliest record = corroborated**: `08_scope_extension_stats.py` now reports the earliest year confirmed
  by ≥2 documents within ±1 yr (raw min kept as `restored_earliest_yr_raw`); raw minimum was always an
  artifact (1830 = time value in a 1977 WQ table; 1854 = Tennessee page mislocated in CA; 1870 = depth
  "1870-1930 FEET"). Review list: `data/analysis/scope_earliest_records_review.csv` (207 records predating NWIS).
- **Fig 7** (`01`): decades 1850s–1970s (641 rows in 3 pre-1850 decades left out); tiny early decades
  labeled with totals. **USA map** (`02`): legend precise vs approximate (old actual/inferred gone); titles
  no longer state year ranges. **CA decade map**: 1850s–1970s. Eastern US points (Cape Fear, Ohio R.) no
  longer appear: those pages have no rebuilt location (location tiers are CA-focused).
- **Santa Ynez** (`04`): GDB fixed (fiona → pyogrio.list_layers). Stream order recomputed from rebuilt
  locations (sjoin_nearest to StreamOrderEdit), replacing stream_proximity_results.csv (LLM coords).
  Watershed polygon filter only (old text match pulled in points up to 318 km away).
- **Current headline numbers (dated = measurement dates, precise+approximate, CA, tribal excluded):**
  stream +533 previously-ungauged HUC12 (+37%), earliest 1862 vs NWIS 1891 (+29 yr); groundwater +82 (+4%),
  earliest 1876 vs NWIS 1890 (+14 yr). 6-panel: restored sites 12,895 stream / 2,705 GW. Caption and
  abstract text not yet updated.

## Out-of-State Locations — November 4, 2026
`location_validation/10_out_of_state.py` (+ shared `state_cues.py`, `gnis_utils.py`, `nwis_utils.py`; 05/06
refactored to import them, 06 output verified identical). California tiers used CA-only references, so pages
about other states were unlocated or matched to CA namesakes (e.g. Murray, Utah; Lakeview, Oreg. placed in CA).
- **State cue** = state abbreviation/name/postal code right after a comma or hyphen ("..., Nev.", "Ariz.-Calif.",
  "AT MESQUITE, NV"). Pages: 67,790 CA/no cue · 5,563 other state only · 914 mixed.
- **Row class** (page-level rule was wrong: it moved 490 CA rows on index/station-list pages):
  `oos` = row's own LLM place text names a non-CA state cued on the page (5,151 rows) → CA-only references
  skipped; `ca` (98,943); `either` = page cues another state, row text none (3,887) → GNIS name must resolve
  uniquely across CA + page states. Output `row_states.csv`, `page_states.csv`.
- **National references:** NWIS site lists per cued state (`data/analysis/nwis_sites/by_state/`), GNIS 2021
  archive per state (`data/analysis/spatial/gnis/{ST}_Features_20210825.txt`, from prd-tnm S3 Archive/MainDomestic).
- **Validation vs printed coords on out-of-state pages** (`tier_oos_validation.csv`): NWIS station 0.10 km median,
  97% ≤5 km (precise); GNIS "at" 1.05 km, 94% (precise, `gnis_oos_qualifier_at`); dam 1.9 km/71%, reservoir
  3.6 km/55%, spring 8.7 km, near 9.5 km/26% → approximate (`gnis_oos_*`); "below" (90% >25 km) and lakes
  (18 km median) NOT used. Out-of-state GNIS is less reliable than in CA.
- **Result vs before:** +2,014 rows located (587 precise, 1,427 approximate); 59 removed (CA-only matches on
  out-of-state rows); 64 moved >50 km, all from a wrong CA point to the named state. Catalog: outside-CA tables
  3,333 → 4,608; unlocated 17,341 → 15,940.
- **Tribal screen re-run** (new out-of-state points): withheld pages 1,752 → **1,875** (+124, −1), documents
  221, federally recognized areas **118**, tables **2,846**, measurements **70,566 (2.8%)**; other-state share
  rose (11,930 meas., 284 pages). The approved manuscript sentence ("~64,000 measurements in 2,672 tables from
  110 areas") must be updated. 1 page no longer withheld still has a copy in data/withheld_tribal (stale).
- **Paper stats after this:** stream +530 ungauged HUC12 (+36%), GW +81 (+4%); earliest 1862 / 1876 unchanged.
- **Earliest-record review (Annette, Nov 4):** stream 1862 confirmed. Groundwater 1876–78 rejected: doc 2854
  p121 = WSP 237 (1910) "Quality of the surface waters of California", Owens Lake analysis (lake); 2094 p26 =
  lake data; 2293 p21 = precipitation. Stream oddballs confirmed artifacts (1830 = time of day in a 1970s WQ
  table). Verdicts in `data/analysis/manual_review/earliest_records_review.csv`, applied by
  `paper_filters.drop_reviewed()` in 07 and 08 (temporal + spatial). Now: stream 1862 (+29 yr, 124 sites
  predate NWIS), +528 basins; **GW 1883 (+7 yr) — PROVISIONAL**, rests on 1128 p27 (1883-84 borings) + 32842
  p76 (an 1884 citation). Next GW candidates to grade: `manual_review/gw_earliest_candidates/` (8 pages,
  pngs + Claude suggestion). Add verdicts to earliest_records_review.csv and re-run 08 + 07.

## Session Log — November 4, 2026 — END OF DAY (RESUME HERE, Nov 5)
Supersedes the "Next session (resume here, Nov 4)" list above. Details of each item are in the
Nov 4 sections above (quality flag, catalog, value counts, date fix, paper figures, out-of-state locations).

### Done today
- Quality flag fixed (lone "o" = zero); catalog + GeoPackage built (`data/catalog/`).
- Value counts: ~18.5M recorded values in published tables (`19_value_counts.py`); daily grids pass a
  calendar check; non-daily hand check by Annette (median ratio 1.04, total 1.22×).
- Dates: graded samples → table-level dates wired into the whole pipeline (25/30 vs LLM 6/30); extra rules
  for century misreads, round-number columns, pre-1850 years; time-based outputs use measurement dates only.
- Paper figures re-run (Fig 7, USA map, CA maps, Santa Ynez, 6-panel) with tribal exclusion, rebuilt locations,
  table dates. Santa Ynez GDB fixed; stream order recomputed from rebuilt locations.
- Out-of-state locations (`10_out_of_state.py`): +2,014 rows located; tribal screen re-run.
- Earliest-record review started (manual_review/). Commits pushed: b4bc15f, 754aa90, 371ad63.

### Current numbers (paper; CA, dated by measurement dates, precise+approximate, tribal excluded, reviewed pages excluded)
| | Stream | Groundwater |
|---|---|---|
| Previously ungauged HUC12 | +528 (+36%) | +81 (+4%) |
| Earliest record vs NWIS | 1862 vs 1891 (+29 yr), confirmed | 1883 vs 1890 (+7 yr), PROVISIONAL |
| Restored CA sites mapped (6-panel) | ~12.9k | ~2.7k |
Published: 2,512,776 table rows / ~18.5M values / 100,829 tables. Tribal withheld: 1,875 pages, 221 docs,
118 federally recognized areas, 2,846 tables, 70,566 measurements (2.8%).

### Next (in order)
1. **Annette:** grade `data/analysis/manual_review/gw_earliest_candidates/gw_earliest_candidates.csv`
   (8 pages, pngs + Claude suggestions). Claude: copy verdicts into
   `manual_review/earliest_records_review.csv`, re-run `08_scope_extension_stats.py` and `07_ca_combined_6panel.py`.
   GW start likely 1883 (if Hungry Hollow borings kept) or 1889.
2. Update manuscript text: abstract (+29 yr / GW TBD; +528 / +81 basins), 6-panel caption (panel numbers,
   GW start year, NWIS site types), tribal sentence (70,566 meas., 2,846 tables, 118 areas — was 64k/2,672/110;
   also decide 118 vs counting text-only areas), measurements "2.5M table rows, ~18 million values",
   date method + accuracy (25/30), location tiers incl. out-of-state validation.
3. Delete the stale copy in `data/withheld_tribal/` for the 1 page no longer withheld (compare manifest).
4. Not yet re-run with today's changes: 03_santa_barbara (skipped by choice), dam scripts, 09_sgma
   (exploratory), county_spatial_join (needed for HydroShare).
5. Small known issue: state cue needs a period ("Wyo." works, "Wyo" without period does not) — minor.
6. Remaining from the high-level list: user guide; main metadata key update (new date/location columns);
   QGIS spot-check of catalog GeoPackage; HydroShare package into a NEW folder (county grouping from rebuilt
   locations, out-of-state labeled, tribal pages excluded); .gitignore `archive/` and the 193 MB NWIS GW raw
   file; Git LFS decision for the ~57 MB table-level CSVs; optional: table-detection recall script.

## Session Log — November 5, 2026 (resume here)
- **User guide drafted:** `docs/user_guide.md` (for outside data users; [TODO] markers for citation,
  license, contact, HydroShare layout, final counts). Note: per-page `_metadata.csv` files and
  main_metadata.csv still carry raw LLM coords/dates; guide points users to the catalog. Decide whether
  the HydroShare page metadata files should carry the rebuilt values.
- **GW earliest-record grading applied.** `manual_review/earliest_records_review.csv` now has an `action`
  column (drop / retype + new_water_type / undate / construction_year / set_year + corrected_year_* / keep).
  `paper_filters.apply_review(df)` applies it right after load in 07 and 08 (replaces drop_reviewed calls).
  Annette's verdicts: 1128 p27 keep (1883-84 Hungry Hollow borings); 1312 p74 keep, true year 1901;
  48308 p41/p43 undate (depth intervals); 2809 p438 construction_year (year drilled); 2292 p27 → Precipitation;
  32842 p76 and 2888 p197 → Water Quality.
- `corroborated_earliest()` in 08 now also accepts a single reviewer-kept page (otherwise needs ≥2 docs).
- **Numbers:** stream 1862 vs NWIS 1891 (+29 yr, 124 sites predate NWIS), +528 HUC12 (+36%);
  **GW 1883 vs 1890 (+7 yr) — confirmed by review but rests on ONE site (1128 p27)**; +80 HUC12 (+4%).
  6-panel re-run (restored 12,888 stream / 2,694 GW sites).
- **Not yet applied:** the retypes (Precipitation / Water Quality) and the 1901 date fix live only in the paper
  filter, not in processed_metadata / table-level CSVs / catalog. Many unlocated pre-1890 GW tables remain
  (e.g. docs 3133, 2294, 51659 with 1856–1889 table dates) — not in the stats because unlocated.
- **Pre-1890 unlocated GW recovery attempt (Nov 5): no earlier data found; 1883 stands.** The 26 unlocated
  GW pages dated <1890 were mostly "Year completed" columns read as measurement dates (Reducto drops the
  header: docs 3133, 2294, 2749) or a split header "Year com- / pleted" (51659). Others: 2291 p35 = LA rainfall
  (precipitation, mislabeled GW; not yet in review file), 9800 p21 = irrigation well counts, 4768/4821 =
  bibliographies, 3983 = index, 2131 p82 = real 1889 artesian well-group discharge (Riverside).
- **Parser fix (`table_dates.py`):** `\bcom-\W*$` counts as "completed" in headers; rotated "Year | com- pleted"
  rows are construction rows; an unlabeled bare-year column in a well-inventory table (`WELL_INVENTORY`:
  owner / cost of / method of lift / class of well / depth of well / diameter / casing) = construction year.
  74 of 106,301 tables changed (40 → F_construction_year, 11 → undated). Review scores unchanged (25/30;
  46/52 of 64). Re-ran 14, 15, 00_data_prep, build_table_level_metadata, filter_published, 18 catalog, 08, 07.
  Now: stream unchanged; GW 1883 (+7 yr), +79 HUC12 (+4%). Catalog dated 93,158.
- Opportunity not taken: 3133 / 2294 well tables print T/R/S per row but are unlocated (bare-number T.R.S.
  columns, rotated tables). Locating them helps maps/catalog only (their only dates are construction years).

## Temporal Resolution per Table — November 5, 2026
Goal (Annette): show researchers what kind of record exists (daily / monthly / annual / discrete) by water type
and point them to it, without double counting. Replaces the page-level LLM `temporal_resolution` field.
- `code/02_inventory/temporal_resolution.py` `classify(rows, caption)`: daily (day-number column 1-31 counting up
  + month context: classic grid, split grids "Day | Feb | Day.1", "1.0"/"1.." labels, two-level headers) ·
  monthly (month rows, year rows x month columns, "Month" label column) · annual (year rows, no month columns,
  not a well inventory) · discrete (Date column / cells with a day) · no_time_series (logs, inventories,
  rating/capacity tables) · caption keywords · unclassified.
- `19_value_counts.py` now records resolution per table, counts summary-row values separately
  (`n_summary_values`), no longer counts repeated "Day.1" columns, counts only day-labeled rows in all daily
  layouts. Published values 18,522,112 -> **18,270,613**; calendar check unchanged (99.2%).
- `20_temporal_resolution.py` -> `data/analysis/table_resolution.csv`, `table_resolution_summary.csv`,
  `table_resolution_by_parameter.csv`. Adds `primary_parameter` / `parameters` (title first, then headers:
  discharge, gage_height, contents, water_level, temperature, sediment, water_chemistry, precipitation,
  evaporation). `derived_summary` = monthly/annual table on the same or adjacent page as a daily table with the
  same water type AND primary parameter -> its values are secondary. LLM fallback OFF (spot check ~2/6 right).
- Catalog numbers (primary values): stream daily 82.5% (12.8M; discharge 9.17M, gage height 1.17M,
  temperature 0.95M, sediment 0.74M, contents 0.28M), monthly 5.4%, annual 0.8%, discrete 6.3%,
  unclassified 3.6%. Groundwater: discrete 51.8% (water levels 165k, chemistry 78k), no time series 16.8%,
  unclassified 23.1%, daily 5.5% (much of GW "daily" is stream/reservoir data mis-typed as GW by the LLM).
- **Grading sample (Annette):** `data/analysis/temporal_resolution/review/resolution_review_sample.csv`
  (65 tables, blind; key in `resolution_review_key.csv`; pngs git-ignored; README has rules).
  Built by `validation/resolution_review_sample.py`. Next: grade -> score -> fix -> add `data_resolution`,
  `derived_summary`, `primary_parameter`, `parameters` to the catalog (18) + user guide + catalog_key.

## Data cleanup — November 5, 2026 (items 4-6)
- **Reviewer corrections in the published data:** `paper_filters.apply_review(..., drop=False)` now runs in
  `build_table_level_metadata.py` (water_type_final) and `00_data_prep.ipynb` (water_type_clean), adding a
  `review_action` column. Retypes + date fixes reach the table-level CSVs and catalog (2291 p35 / 2292 p27 →
  Precipitation; 32842 p76 / 2888 p197 → Water Quality; 1312 p74 = 1901; 48308 p41/p43 undated; 2809 p438
  construction_year). 'drop' verdicts stay paper-only (tables kept). assignment_source is NOT changed
  (filter_published_tables keeps only LLM sources; changing it dropped the retyped tables once).
  Totals unchanged: 100,829 tables / 2,512,776 rows; paper stats unchanged.
- **'drop' pages fixed in the published data (Annette, Nov 5):** review actions can be combined (';') and a new
  `unlocate` action clears a wrong location. 56808 p127 / 56850 p468 undated; 2293 p21 -> Precipitation; 2094 p26
  -> Water Quality; 2854 p121 (printed p. 122, Owens Lake analyses) -> Water Quality, 1876-1906; 3931 p4 (Wyo.) /
  2987 p69 (Tenn.) unlocated + undated. Only 70038401 p34 (historical flood estimate) remains a pure 'drop'.
  Note: `page_number` = position in the scanned PDF, not the printed page number (why 2854 p121 was hard to find).
- **Stale tribal copies deleted:** 9 page folders (57549 p11 in 6 area folders — no longer withheld; 25119 p38,
  70038392 p42, 251 p20 left in old area folders). data/withheld_tribal now = manifest (2,076 page folders).
- **metadata_key.txt:** header note (LLM coords/dates are raw, superseded) + SECTION 3 documenting
  processed_metadata.parquet rebuilt date / location / page / review columns.

## Session Log — November 5, 2026 — END OF DAY (RESUME HERE, next session)
Done today: user guide draft (docs/user_guide.md); GW earliest-record grading applied (GW start 1883, one
site); year-completed date parser fix; pre-1890 unlocated GW search (nothing earlier); per-table temporal
resolution + parameter (20_temporal_resolution.py); reviewer corrections carried into published data;
stale tribal copies removed; metadata_key.txt Section 3. Commits: ed13544, 9a4b03a.

Current numbers: stream 1862 vs NWIS 1891 (+29 yr, 124 sites predate NWIS), +528 HUC12 (+36%);
GW 1883 vs 1890 (+7 yr, 1 site), +79 HUC12 (+4%). Published 100,829 tables / 2,512,776 rows /
~18.27M values. Tribal withheld: 1,875 pages, 221 docs, 118 areas, 2,846 tables, 70,566 rows.

### Next (in order)
1. Draft manuscript text: abstract, 6-panel caption, tribal sentence (118 vs 121 areas?), "2.5M table rows,
   ~18 million values", methods for date protocol (25/30) and location tiers (incl. out-of-state).
2. Annette (when time allows): grade data/analysis/temporal_resolution/review/resolution_review_sample.csv
   -> score -> fix -> add resolution / derived_summary / primary_parameter / parameters to catalog (18),
   catalog_key.txt and user guide.
3. Re-run 03_santa_barbara (and dam / 09_sgma if used in the paper).
4. Decide per-page _metadata.csv contents (rebuilt vs raw LLM) -> HydroShare package into a NEW folder
   -> finalize user guide (citation, license, contact, counts).
5. Housekeeping: .gitignore archive/ and the 193 MB NWIS GW raw file; Git LFS for ~57 MB table CSVs.
Deferred: row-level PLSS for well inventories (~170 GW tables); table-detection recall script.
