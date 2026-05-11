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
├── Hilton_Dissertation_USGSDig.docx   ← Chapter 2 draft (primary reference)
│
├── Data_Files/                        ← ALL digitized data lives here
│   ├── cleaned_metadata_final - Copy.csv  ← MASTER metadata table (key reference file)
│   ├── csv_row_counts.csv             ← estimated measurement counts per file
│   ├── 2252_728 example/              ← TARGET OUTPUT FORMAT (see below)
│   │   ├── 2252_page_728.json
│   │   ├── 2252_page_728.png
│   │   ├── 2252_page_728_table1.csv
│   │   └── 2252_page_728_table2.csv
│   │   (missing: metadata CSV for this page — still to be added)
│   ├── ReductCSVs/                    ← 623 document ID subfolders → .csv files
│   │   └── {doc_id}/
│   │       └── {doc_id}_page_{N}_table{M}.csv
│   ├── ReductJson/                    ← 623 document ID subfolders → .json files
│   │   └── {doc_id}/
│   │       └── {doc_id}_page_{N}.json
│   ├── UpdatedDataDec/                ← 353 document ID subfolders → .png files
│   ├── UpdatedDataJan/                ← 287 document ID subfolders → .png files
│   ├── updatedSB/
│   │   └── Scanned_SB/               ← Santa Barbara County .csv + .json files
│   │       └── {70300001_pageN.csv, ...}  (different naming convention)
│   └── Scanned_SB/                   ← (duplicate/legacy SB folder at top level)
│
├── Chapter_2_USGS_Digitization/       ← Project documents, code, manuscript
│   ├── Manuscript/
│   │   ├── Methods Report.docx        ← detailed methods
│   │   ├── Supplementary Information.docx
│   │   └── Extra Text.docx
│   ├── Figures/
│   │   └── example_tables/            ← example table images for figures
│   ├── Yibo Results/                  ← LLM metadata extraction code (iterative)
│   │   ├── Sample Table Column Detection/   ← PaddleOCR / column detection scripts
│   │   │   ├── extractTable.py, image.py, table.py, utilities.py
│   │   │   └── ColumnDetection.ipynb, paddle.ipynb
│   │   ├── Working Example for Annette JSON_CSV_HTML/
│   │   │   └── Scripts/json_csv.py, reconstruct_bbox.py
│   │   ├── JSON Context Extraction PoC/    ← proof-of-concept LLM extraction
│   │   ├── May 27 2025 R4-6/          ← iteration of LLM metadata runs
│   │   │   └── api_run.py             ← calls UCSB LLM API (llama3, deepseek-r1)
│   │   ├── June 2025/                 ← latest LLM metadata runs
│   │   │   ├── shortlist/, longlist/, henderson list/, run4v1_062524/, run4v1_062525_2/
│   │   ├── Final Test Sets/
│   │   ├── hasTable.csv               ← table detection results
│   │   └── filtered USGS Groundwater Data Tables & Pages.xlsx
│   ├── Literature-Data Review/
│   │   ├── LLM-Metadata Testing/      ← accuracy evaluation, test sets, reviewer comments
│   │   │   ├── Accuracy evaluation/   ← final accuracy methods and results docs
│   │   │   ├── LLM Runs/              ← archived run zip files
│   │   │   ├── Parameters & Runs/     ← run notes, water category terms
│   │   │   └── {AH, HV, LBL} Comments-Review/  ← human reviewer feedback
│   │   ├── Edited USGS Data Pulls/    ← cleaned document lists
│   │   ├── Raw USGS Data Pulls/       ← raw USGS Publications Warehouse exports
│   │   └── Specific Notes from Luma/ ← workflow documentation from collaborator
│   ├── Logistical Documents/          ← team docs, collaboration compact, passwords
│   ├── Old Meeting Agendas/
│   └── Validation Data Sets (Hilton and Jasechko, 2023)/
│
├── jupyter_notebooks/                 ← analysis, inventory, and vignette notebooks
│   ├── working_datacrunch.ipynb       ← core data processing / merging
│   ├── working_datacrunch_sbaddition.ipynb  ← adds SB county data
│   ├── measurements_bytype.ipynb      ← data inventory by water type
│   ├── summary_figs.ipynb             ← summary figures
│   ├── ca_maps.ipynb                  ← California spatial maps
│   ├── usa_maps.ipynb                 ← national spatial maps
│   ├── accuracy_grades.ipynb          ← LLM accuracy evaluation
│   ├── csv_estimates.ipynb            ← measurement count estimates
│   ├── santabarbara.ipynb             ← Santa Barbara County analysis
│   ├── santabarbara_v2maps.ipynb      ← SB maps v2
│   ├── sb_usgs_gw.ipynb               ← SB groundwater vs NWIS
│   ├── sb_usgs_streamflow.ipynb       ← SB stream discharge vs NWIS
│   ├── santaynez_final.ipynb          ← Santa Ynez River analysis (paper figure)
│   ├── santaynez_analysis*.ipynb      ← iterations of Santa Ynez analysis
│   ├── water_palette.py               ← shared color palette for water categories
│   ├── artifacts_io.py                ← shared artifact I/O utilities
│   ├── archived_notebooks/            ← superseded notebook versions
│   ├── data/                          ← spatial data (GDB files for lakes/rivers, SB USGS data)
│   ├── code/                          ← helper scripts (GeoLocateTemp/)
│   ├── output/, plots/, exports/      ← notebook output directories
│   └── artifacts/                     ← saved analysis artifacts
│
└── legacy_ocr/                        ← original Python package (nbdev) for table detection
    ├── usgs_extract/                  ← package source (model.py loads Table Transformer)
    ├── nbs/                           ← development notebooks
    ├── 01_utilities.ipynb through 04_model.ipynb  ← nbdev source notebooks
    └── scrapper/getpdf.py             ← PDF download script
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

**Santa Barbara exception**: SB County data lives in `updatedSB/Scanned_SB/` with a different naming convention (`70300001_pageN.csv`) and needs special handling.

---

## Planned Work (Four Steps)

### Step 1: Organize Final Data Output
- Write a script to reorganize all files into the unified `{doc_id}/page_{N}/` structure
- Extract metadata rows per page from `cleaned_metadata_final - Copy.csv` and save as individual CSVs
- Join USGS document-level metadata from `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` into each per-page metadata CSV — the `id` column in the master metadata maps directly to the `Publication ID` column in `USGS_ID.xlsx`, which provides the USGS URL, Index ID (series), title, year, author, and ~55 additional document-level fields
- Handle the SB County data exception
- Identify and document any doc IDs present in metadata but missing from data folders (or vice versa)

### Step 2: HydroShare Upload Organization
- Explore groupings of data for upload (e.g., by water_type + decade)
- Understand HydroShare size/file limits
- Create upload-ready copies of grouped data
- May need to compress or chunk large groups

### Step 3: Data Inventory (Notebooks)
- Notebooks already exist (`measurements_bytype.ipynb`, `summary_figs.ipynb`, `ca_maps.ipynb`, `usa_maps.ipynb`)
- Need to: identify which are final/publication-ready vs. exploratory, consolidate duplicates, ensure reproducibility from the cleaned metadata CSV

### Step 4: Hydrological Research Vignettes
- Santa Barbara County analysis exists (`santabarbara.ipynb`, `sb_usgs_gw.ipynb`, `sb_usgs_streamflow.ipynb`)
- Santa Ynez River analysis exists (`santaynez_final.ipynb`)
- Potential larger-scale analyses (e.g., streamflow change across California over decades) require data cleaning/normalization across tables — challenging due to inconsistent formats
- Approach: start with a single water category (e.g., stream discharge) and a limited set of well-structured sites before scaling

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `Data_Files/cleaned_metadata_final - Copy.csv` | Master metadata; links every page to its doc ID, coordinates, water type, dates |
| `Chapter_2_USGS_Digitization/Literature-Data Review/Edited USGS Data Pulls/usgs_to_id/USGS_ID.xlsx` | ID crosswalk: maps project number ID (`Publication ID`) → USGS URL, Index ID, title, year, author, and ~55 document-level fields. 1,550 rows covering all documents. |
| `Data_Files/2252_728 example/` | Reference example of target output folder structure |
| `Hilton_Dissertation_USGSDig.docx` | Full paper draft — authoritative description of all methods and results |
| `Chapter_2_USGS_Digitization/Yibo Results/May 27 2025 R4-6/api_run.py` | Core LLM metadata extraction script (calls UCSB HPC API: llama3 + deepseek-r1) |
| `jupyter_notebooks/working_datacrunch.ipynb` | Core data processing / merging notebook |
| `jupyter_notebooks/santaynez_final.ipynb` | Finalized Santa Ynez vignette |
| `legacy_ocr/usgs_extract/model.py` | Loads Microsoft Table Transformer for table detection |
| `Chapter_2_USGS_Digitization/Literature-Data Review/LLM-Metadata Testing/Accuracy evaluation/` | Human accuracy evaluation docs and methods |

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

The `data/digitized/` folder was populated by `code/01_digitization/06_final_data_organization/organize_data.py`. Across 622 docs and 74,492 pages, three categories of pages exist but have **no `_metadata.csv` file**:

### Pages with no metadata rows (~10,919 pages, ~14.7%)
The LLM metadata extraction (Phi-4) was only run on pages where a table was detected. Pages that Reducto processed but where no table was found — or where the table was output as plain text rather than structured HTML — were never sent to the LLM and therefore have no entry in `cleaned_metadata_final - Copy.csv`. These fall into three sub-types:

1. **Non-data pages** — title pages, cover pages, introductions, references, figures. No table content present. Expected and not a data loss.

2. **Tables that Reducto could not structure** — the page contains a visible table (e.g., `10166/page_10`: "Table 2.--Age and type of bedrock at each spring") but Reducto returned the content as plain text chunks rather than HTML `<table>` elements. The raw text is preserved in the `.json` file. This represents tables the pipeline did not successfully digitize. **No further digitization will be performed** — this is documented as a known limitation.

3. **Table-of-contents pages** — pages with TOC-style formatting that Reducto parsed as a table but which contain no water measurement data.

### Pages with no table CSVs (~7,127 pages, ~9.6%)
These pages have a `.json` (Reducto ran) but no `_table*.csv` files. This occurs when Reducto processed the page but the downstream CSV conversion was not completed or the page had no digitizable table. The JSON is preserved.

### Pages with no PNG (~260 pages, ~0.3%)
A small number of pages have a JSON and CSVs but no corresponding `.png` file in either `UpdatedDataDec/` or `UpdatedDataJan/`. These are pages from documents that may have been processed outside the main PNG batches.

### Santa Barbara County data (excluded)
The SB County data was processed as an early test batch before standardized naming conventions were established. It is **not included** in `data/digitized/`. There are two distinct SB datasets:

**sb1–sb7 + sb_page_full** (`Data_Files/updatedSB/Scanned_SB/`, naming: `sb{N}_page_full_table{M}.csv`): Eight documents, all from the annual "Water levels in observation wells in Santa Barbara County" Open-File Report series (USGS) and one Tecolote Tunnel springs report. Each has a readable Reducto JSON (`Data_Files/updatedSB/Scanned_SB/json/`). The USGS document identities were recovered from the JSON content:

| File | Data year | USGS Index ID | Pub ID |
|------|-----------|--------------|--------|
| sb1 | 1956 | ofr5778 | 23995 |
| sb2 | 1958 | ofr5983 | None |
| sb3 | 1960 | ofr6196 | None |
| sb4 | 1948–49 | ofr49118 (Tecolote Tunnel, 1st progress report) | None |
| sb5 | 1963 | ofr64117 | None |
| sb6 | 1961–63 | ofr6291/63103/64117 (multi-year) | None |
| sb7 | 1962 | ofr63103 | None |
| sb (unnumbered) | 1959 | ofr60102 | None |

**70300001 files** (`Data_Files/Scanned_SB/`, naming: `70300001_page{N}.csv` / `70300001_table{N}.csv`): 434 CSV files from an unidentified document (same annual observation well series, mid-1950s era based on content). Two orphaned Reducto JSONs (`ReductJson/Scanned_SB/report (8).json` and `report (9).json`, 180 and 281 pages respectively) have their content stored at expired S3 URLs (expired 2025-01-11) and cannot be read. Since no readable JSON exists for these CSVs, **this data is permanently excluded**.

See `jupyter_notebooks/santabarbara.ipynb` for analysis using the SB data.

A full per-page log is at `data/digitized/_organization_log.txt`.

---

## Technical Notes

- Metadata extraction uses Microsoft Phi-4 via UCSB's LLM API (`llm.grit.ucsb.edu`); some iteration scripts also tested llama3 and deepseek-r1
- Table detection threshold: 0.8 (recovers ~79% of known tables)
- Reducto.ai used for final digitization (not open source — requires paid API access)
- PaddleOCR was tested but discarded for structure preservation failures
- The SB County data (`updatedSB/Scanned_SB/`) uses a different naming convention (e.g., `70300001_pageN.csv`) — these appear to be separately scanned/processed documents
- Multiple iterations of LLM prompt engineering are preserved in `Chapter_2_USGS_Digitization/Yibo Results/` — the final production prompt is in Supplementary Information S1 of the dissertation
