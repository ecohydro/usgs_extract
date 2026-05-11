# USGS Historical Water Data Digitization — Repo Structure

> **Before making this repo public:** API keys were accidentally committed in `yibo_code/usgs_extract/api_run.py` (UCSB HPC key) and `yibo_code/usgs_extract/testset.py` (UCSB HPC key). The `.env` file containing the Reducto.ai API key is also in git history. All three keys must be scrubbed from commit history (e.g. with `git filter-repo`) and rotated before the repo is made public.

Target structure for organizing this repo. Use this as a reference for where to put things.

```
usgs_extract/
├── manuscript/                        ← dissertation, methods report, supplementary materials
│
├── code/
│   ├── 01_digitization/
│   │   ├── 01_download_and_preprocess/ ← scripts to pull PDFs from USGS and convert to grayscale PNGs
│   │   ├── 02_table_detection/
│   │   │   ├── 01_detect.py           ← Table Transformer detection script
│   │   │   └── validation/            ← MISSING: script to evaluate recall/precision against ground truth
│   │   ├── 03_ocr/
│   │   │   ├── paddle/                ← PaddleOCR (tested, not used in final pipeline)
│   │   │   ├── reducto/               ← Reducto.ai digitization scripts (final pipeline)
│   │   │   └── validation/            ← accuracy evaluation of OCR outputs
│   │   ├── 04_metadata_extraction/
│   │   │   └── 01_extract_metadata.py ← LLM (Phi-4) metadata extraction via UCSB HPC API
│   │   └── 05_final_data_organization/ ← scripts to reorganize outputs into final per-page structure
│   ├── 02_hydroshare/                 ← scripts to package data for HydroShare upload
│   ├── 03_inventory/                  ← data inventory notebooks and summary figures
│   └── 04_vignettes/                  ← hydrological research analyses (Santa Barbara, Santa Ynez, etc.)
│
└── data/
    ├── digitized/                     ← final organized output, one folder per doc/page
    │   └── {doc_id}/
    │       └── page_{N}/
    │           ├── {doc_id}_page_{N}.png
    │           ├── {doc_id}_page_{N}.json
    │           ├── {doc_id}_page_{N}_table1.csv
    │           └── {doc_id}_page_{N}_metadata.csv
    ├── metadata/                      ← main metadata CSV and crosswalk files
    ├── digitization_intermediates/
    │   ├── 01_download_and_preprocess/ ← publication list CSV; PDF download manifest
    │   ├── 02_table_detection/
    │   │   └── validation/
    │   │       ├── groundwater_table_pages.xlsx       ← human-curated ground truth (page ranges)
    │   │       ├── groundwater_table_pages_expanded.csv ← ground truth expanded to one row per page (MISSING: script that did this expansion)
    │   │       └── MISSING: detection scores for ground truth pages (validation run output)
    │   │   (MISSING: production detection results for all CA docs — ran on HPC, not preserved)
    │   ├── 03_ocr/
    │   │   (MISSING: validation ground truth and accuracy scripts — to be recovered)
    │   └── 04_metadata_extraction/
    │       └── validation/
    │           ├── validation_curated_135.xlsx    ← curated set graded by Henderson Vo (orig: HV Comments-Review/Test set 1 (curated)/metadata_HVedits.xlsx)
    │           ├── validation_random_135.xlsx     ← random set graded by Luma Braconi Lazarini (orig: LBL Comments-Review/Random Test Set Grading/random_test_set.xlsx)
    │           ├── validation_edge_cases_15.xlsx  ← edge case answer key (orig: Parameters & Runs/answer_key_shortlist.xlsx)
    │           └── jsons/                         ← 260 per-page Reducto JSONs covering all three validation sets
    ├── hydroshare/                    ← upload-ready packages for HydroShare
    └── analysis/                      ← intermediate data from inventory and vignette notebooks
```

---

## Quickstart: Download & Preprocess (`code/01_digitization/01_download_and_preprocess/`)

**Step 1 — Download PDFs from USGS Publications Warehouse**
```bash
python 01_download.py \
  --input-csv data/digitization_intermediates/01_download_and_preprocess/edited_publication_list.csv \
  --output-dir data/digitization_intermediates/01_download_and_preprocess/pdfs/
```
Input CSV must have `URL` and `Publication ID` columns (standard USGS Publications Warehouse export format). Already-downloaded files are skipped automatically.

**Step 2 — Verify downloads**
```bash
python 02_verify.py \
  --input-csv data/digitization_intermediates/01_download_and_preprocess/edited_publication_list.csv \
  --download-dir data/digitization_intermediates/01_download_and_preprocess/pdfs/ \
  --output-json missing_ids.json
```
Prints a download summary and writes any missing publication IDs to `missing_ids.json`.

**Step 3 — Convert PDFs to grayscale PNGs**
```bash
python 03_preprocess.py \
  --pdf-dir data/digitization_intermediates/01_download_and_preprocess/pdfs/ \
  --output-dir data/digitization_intermediates/01_download_and_preprocess/pngs/
```
Outputs one `{pub_id}_page_{N}.png` per page at 300 DPI, grayscale.

---

## Quickstart: Table Detection (`code/01_digitization/02_table_detection/`)

**Detect tables in PNGs using Microsoft Table Transformer**
```bash
python 01_detect.py \
  --png-dir data/digitization_intermediates/01_download_and_preprocess/pngs/ \
  --output-csv data/digitization_intermediates/02_table_detection/detections.csv \
  --threshold 0.8
```
Output CSV has one row per detected table: `filename`, `doc_id`, `page_num`, `label` (table/table rotated), `score`, `bbox`. Already-processed files are skipped automatically on re-run.

Ground truth validation data is in `data/digitization_intermediates/02_table_detection/validation/`. A script to compute recall/precision against that ground truth is **missing** — the 79% recall figure in the paper was computed manually.

---

## Quickstart: OCR / Digitization (`code/01_digitization/03_ocr/`)

Two digitization approaches are implemented. Reducto was used in the final pipeline; PaddleOCR was tested and discarded due to poor structure preservation.

### Reducto (final pipeline — requires paid API access)

**Step 1 — Send page PNGs to Reducto and save JSON output**
```bash
export REDUCTO_API_KEY=your_key_here
python reducto/01_digitize.py \
  --png-dir data/digitization_intermediates/01_download_and_preprocess/pngs/ \
  --output-dir data/digitization_intermediates/03_ocr/jsons/
```
Uploads each PNG to the Reducto.ai API and saves the full structured JSON response. Already-processed files are skipped. Runs 50 concurrent requests per batch. Errors are appended to `error_log.txt`.

**Step 2 — Convert Reducto JSON to CSV**
```bash
python reducto/02_json_to_csv.py \
  --json-dir data/digitization_intermediates/03_ocr/jsons/ \
  --output-dir data/digitization_intermediates/03_ocr/csvs/
```
Extracts HTML table blocks from each JSON and writes one CSV per table (`{doc_id}_page_{N}_table{M}.csv`).

### PaddleOCR (tested, not used in final pipeline)

PaddleOCR requires table regions to be cropped from pages before processing. Use the Table Transformer detection output from Step 2 as input.

**Step 1 — Crop detected table regions from page PNGs**
```bash
python paddle/01_crop_tables.py \
  --png-dir data/digitization_intermediates/01_download_and_preprocess/pngs/ \
  --detections-csv data/digitization_intermediates/02_table_detection/detections.csv \
  --output-dir data/digitization_intermediates/03_ocr/paddle_crops/
```

**Step 2 — Run PaddleOCR on cropped table images**
```bash
python paddle/02_paddleocr.py \
  --input-dir data/digitization_intermediates/03_ocr/paddle_crops/ \
  --output-dir data/digitization_intermediates/03_ocr/paddle_output/
```

Validation scripts for both approaches are **missing** — accuracy figures in the paper were computed manually.

---

## Quickstart: Metadata Extraction (`code/01_digitization/04_metadata_extraction/`)

**Extract metadata from Reducto per-page JSONs using Phi-4 via UCSB HPC**
```bash
export UCSB_LLM_API_KEY=your_key_here
python 01_extract_metadata.py \
  --json-dir data/digitization_intermediates/03_ocr/jsons/ \
  --output-csv data/digitization_intermediates/04_metadata_extraction/llm_extracted_metadata.csv
```
Requires per-page Reducto JSON files (one file per page, named `{doc_id}_page_{N}.json`). Skips pages with no Table blocks. Already-processed `(ID, PAGE_NUMBER)` pairs are skipped on re-run. Parse failures are written to `{doc_id}_{page_num}_error.txt` in the JSON directory.

Output CSV matches the column schema of `cleaned_metadata_final - Copy.csv`: `ID, PAGE_NUMBER, Inferred_Latitude, Inferred_Longitude, Actual_Latitude, Actual_Longitude, Location, Townships_Ranges_Sections, Watersource_Name, County, Dates_of_Recording, Temporal_Resolution, Units_Of_Measurement, Water_Type, KeyTerms`.

Validation against 3 test sets was done manually in Excel. The graded files are in `data/digitization_intermediates/04_metadata_extraction/validation/`:

| File | Contents |
|------|----------|
| `validation_curated_135.xlsx` | Curated set: 135 tables selected to represent all water categories, graded by Henderson Vo |
| `validation_random_135.xlsx` | Random set: 135 tables drawn at random from the full dataset, graded by Luma Braconi Lazarini |
| `validation_edge_cases_15.xlsx` | Edge cases: 15 tables targeting false positives and ambiguous table types |
| `test_set_cat_grades.xlsx` | Per-category accuracy summary (curated + random sets) — input to the accuracy figure |
| `test_set_indiv_grades.xlsx` | Per-entry accuracy scores (curated + random sets) — input to the accuracy figure |
| `jsons/` | 260 per-page Reducto JSONs covering all three validation sets |

**`02_validate_accuracy.ipynb`** reads `test_set_cat_grades.xlsx` and `test_set_indiv_grades.xlsx` and produces the metadata accuracy figure (saved to `manuscript/figures/metadata_accuracy.png`): histograms of per-entry accuracy for the curated and random test sets, and a lollipop chart of per-category accuracy across both sets.
