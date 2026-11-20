"""Every tunable of the metadata evaluation. Nothing numeric lives elsewhere."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/analysis/evaluation/metadata"

# Read-only inputs on the cluster (never written to).
CLUSTER = Path("/home/waves/data/usgs_extract")
REDUCTO_CSV = CLUSTER / "ReductCSVs"          # {doc}/{doc}_page_{N}_table{k}.csv
REDUCTO_JSON = CLUSTER / "ReductJson"         # {doc}/{doc}_page_{N}.json
# Second copy, organized by page; used for reports that are missing from ReductJson (six reports, 155 pages).
REDUCTO_JSON_BY_PAGE = Path("/home/waves/data/usgs_extract_data/digitized")   # {doc}/page_{N}/{doc}_page_{N}.json
PAGE_IMAGES = (CLUSTER / "UpdatedDataDec", CLUSTER / "UpdatedDataJan")  # {doc}/{doc}_page_{N}.png
SCAN_FILL_PAGES = Path("/home/waves/annaboser/usgs_extract/scan_fill_pages")   # copies of the page scans read by 03_fill_from_scans.py (waves share: home runs full)

PUBLISHED_TABLES = REPO / "data/analysis/table_level_metadata_published.csv"  # sampling frame

# Test set design (Anna, 2026-11-14/15). Unit = one published table.
STRATA = {                       # water type group -> tables drawn (frozen, Anna 2026-11-14)
    "Stream Discharge": 80,
    "Groundwater": 50,
    "Reservoir": 35,
    "Small types": 35,           # every other water type, pooled
}
SEED = 20261115                  # draw date
ONE_TABLE_PER_PAGE = True        # tables on one page share their heading and their errors
MAX_TABLES_PER_REPORT = 3        # keeps one long report from dominating a stratum
EXCLUDED_PAGES = {}              # (doc, page) -> reason; dropped after the draw and replaced by the next
                                 # table in the stratum's random order (pages a checker flags as tribal data
                                 # that the screen in tribal.py did not catch)

# Pages already used to develop or tune anything are never test pages.
DEVELOPMENT_PAGE_LISTS = [       # (file, document column, page column)
    ("data/analysis/date_parsing/date_validation_sample.csv", "id", "page_number"),
    ("data/analysis/date_parsing/entry_date_review/entry_date_review_sample.csv", "id", "page_number"),
    ("data/analysis/temporal_resolution/review/resolution_review_key.csv", "doc_id", "page_number"),
    ("data/analysis/value_counts/review/value_count_review_key.csv", "doc_id", "page_number"),
    ("data/analysis/manual_review/earliest_records_review.csv", "id", "page_number"),
    ("data/analysis/evaluation/digitization/sample_pages.csv", "doc_id", "page_number"),
]
# pages of the 2025 graded sets
DEVELOPMENT_JSON_DIR = REPO / "data/digitization_intermediates/04_metadata_extraction/validation/jsons"

VALIDATION_STRATA = {            # per validation set: half the test size, same proportions
    "Stream Discharge": 40, "Groundwater": 25, "Reservoir": 18, "Small types": 17}
SETS = {   # name -> strata, seed, table list, page scans, key folder. The test key is never used for tuning.
    "test": dict(strata=STRATA, seed=SEED, tables=OUT / "test_set_tables.csv", pages=OUT / "pages", key_dir=OUT / "key"),
    "validation": dict(strata=VALIDATION_STRATA, seed=SEED + 1, tables=OUT / "validation_set_tables.csv",
                       pages=OUT / "pages_validation", key_dir=OUT / "key_validation"),
    # Second validation set (Anna 2026-11-15): same design; excludes every page of the sets above.
    "validation2": dict(strata=VALIDATION_STRATA, seed=SEED + 2, tables=OUT / "validation2_set_tables.csv",
                        pages=OUT / "pages_validation2", key_dir=OUT / "key_validation2"),
}
REVIEW_PORT = 8788

# The model: run locally with vLLM 0.21 (env vllm-rs), one L40S, reasoning off. Weights on scratch, never in the repo.
MODEL_ROOT = Path("/home/hpc-scratch/annaboser/usgs_models/hf")
MODEL = "Qwen/Qwen3.6-27B-FP8"
MODEL_MAX_CONTEXT = 21000            # tokens
MODEL_MAX_ANSWER = 8000              # tokens; long site lists need room
MODEL_MAX_PAGES_AT_ONCE = 48         # with the default batch settings the 46 GB card ran out of memory
MODEL_RUNS = OUT / "model_runs"      # one folder per (set, prompt version): answers.jsonl
REREAD_MAX_CONTEXT, REREAD_MAX_ANSWER, REREAD_AT_ONCE = 48000, 28000, 16   # second pass for answers that were unreadable
PROMPT_VERSION = "v5"   # prompts.py
SCORES = OUT / "scores"                                          # {set}_{source}_long.csv and _summary.csv

TIDY_RULES = ("runoff_beside_discharge", "generic_water_source")   # pipeline.tidy
REQUIRE_ON_PAGE = True        # a copied location value is kept only if its characters are found in the page text

# 95% ranges of the scores: tables resampled with replacement within their sampling group.
RANGE_DRAWS, RANGE_SEED, RANGE_LEVEL = 5000, 20261116, 0.95
