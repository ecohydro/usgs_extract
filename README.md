# USGS Historical Water Data Digitization — Repo Structure

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
│   │   │   ├── workflow/              ← LLM (Phi-4) metadata extraction via UCSB HPC API
│   │   │   └── validation/            ← accuracy evaluation of extracted metadata
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
    ├── metadata/                      ← master metadata CSV and crosswalk files
    ├── digitization_intermediates/
    │   ├── 01_download_and_preprocess/ ← publication list CSV; PDF download manifest
    │   ├── 02_table_detection/
    │   │   └── validation/
    │   │       ├── groundwater_table_pages.xlsx       ← human-curated ground truth (page ranges)
    │   │       ├── groundwater_table_pages_expanded.csv ← ground truth expanded to one row per page (MISSING: script that did this expansion)
    │   │       └── MISSING: detection scores for ground truth pages (validation run output)
    │   │   (MISSING: production detection results for all CA docs — ran on HPC, not preserved)
    │   ├── 03_ocr/                    ← raw Reducto JSON and CSV outputs per page
    │   └── 04_metadata_extraction/    ← raw LLM metadata outputs before cleaning
    ├── hydroshare/                    ← upload-ready packages for HydroShare
    └── analysis/                      ← intermediate data from inventory and vignette notebooks
```

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
