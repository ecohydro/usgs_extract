# USGS Historical Water Data Digitization — Repo Structure

Target structure for organizing this repo. Use this as a reference for where to put things.

```
usgs_extract/
├── docs/                              ← manuscript, methods, supplementary materials
│
├── code/
│   ├── 01_digitization/
│   │   ├── 01_download/               ← scripts to pull PDFs from USGS Publications Warehouse
│   │   ├── 02_pre_process/            ← PDF → grayscale PNG conversion
│   │   ├── 03_table_detection/
│   │   │   ├── workflow/              ← Table Transformer detection scripts
│   │   │   └── validation/            ← accuracy evaluation against known tables
│   │   ├── 04_ocr/
│   │   │   ├── paddle/                ← PaddleOCR (tested, not used in final pipeline)
│   │   │   ├── reducto/               ← Reducto.ai digitization scripts (final pipeline)
│   │   │   └── validation/            ← accuracy evaluation of OCR outputs
│   │   ├── 05_metadata_extraction/
│   │   │   ├── workflow/              ← LLM (Phi-4) metadata extraction via UCSB HPC API
│   │   │   └── validation/            ← accuracy evaluation of extracted metadata
│   │   └── 06_final_data_organization/ ← scripts to reorganize outputs into final per-page structure
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
    │   ├── 01_download/               ← Metadata on orginal download
    │   ├── 02_pre_process/            ← Potentially empty
    │   ├── 03_table_detection/        ← table listing which pages have tables, maybe table detection outputs (bounding boxes, confidence scores)
    │   ├── 04_ocr/                    ← raw Reducto JSON and CSV outputs per page
    │   └── 05_metadata_extraction/    ← raw LLM metadata outputs before cleaning
    ├── hydroshare/                    ← upload-ready packages for HydroShare
    └── analysis/                      ← intermediate data from inventory and vignette notebooks
```
