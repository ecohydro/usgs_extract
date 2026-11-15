"""Every tunable of the metadata evaluation. Nothing numeric lives elsewhere."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/analysis/evaluation/metadata"

# Read-only inputs on the cluster (never written to).
CLUSTER = Path("/home/waves/data/usgs_extract")
REDUCTO_CSV = CLUSTER / "ReductCSVs"          # {doc}/{doc}_page_{N}_table{k}.csv
REDUCTO_JSON = CLUSTER / "ReductJson"         # {doc}/{doc}_page_{N}.json
PAGE_IMAGES = (CLUSTER / "UpdatedDataDec", CLUSTER / "UpdatedDataJan")  # {doc}/{doc}_page_{N}.png

PUBLISHED_TABLES = REPO / "data/analysis/table_level_metadata_published.csv"  # sampling frame

# Test set design (Anna, 2026-11-14/15). Unit = one published table.
STRATA = {                       # water type group -> tables drawn (frozen, Anna 2026-11-14)
    "Stream Discharge": 80,
    "Groundwater": 50,
    "Reservoir": 35,
    "Small types": 35,           # every other water type, pooled
}
SEED = 20261115                  # draw date
ONE_TABLE_PER_PAGE = True        # proposal: tables on one page share their heading and their errors
MAX_TABLES_PER_REPORT = 3        # proposal: keeps one long report from dominating a stratum
EXCLUDED_PAGES = {}              # (doc, page) -> reason; dropped after the draw and replaced by the next
                                 # table in the stratum's random order (tribal withheld pages go here
                                 # once Annette's list arrives)

# Pages already used to develop or tune anything are never test pages.
DEVELOPMENT_PAGE_LISTS = [       # (file, document column, page column)
    ("data/analysis/evaluation/metadata/phi4_grades_long.csv", "doc_id", "page_number"),
    ("data/analysis/date_parsing/date_validation_sample.csv", "id", "page_number"),
    ("data/analysis/date_parsing/entry_date_review/entry_date_review_sample.csv", "id", "page_number"),
    ("data/analysis/temporal_resolution/review/resolution_review_key.csv", "doc_id", "page_number"),
    ("data/analysis/value_counts/review/value_count_review_key.csv", "doc_id", "page_number"),
    ("data/analysis/manual_review/earliest_records_review.csv", "id", "page_number"),
    ("data/analysis/evaluation/digitization/sample_pages.csv", "doc_id", "page_number"),
]
DEVELOPMENT_JSON_DIR = REPO / "data/digitization_intermediates/04_metadata_extraction/validation/jsons"

VALIDATION_STRATA = {            # development set, tuned against freely (Claude 2026-11-15, half the test size)
    "Stream Discharge": 40, "Groundwater": 25, "Reservoir": 18, "Small types": 17}
SETS = {   # name -> strata, seed, table list, page scans, key folder. The test key is never used for tuning.
    "test": dict(strata=STRATA, seed=SEED, tables=OUT / "test_set_tables.csv", pages=OUT / "pages", key_dir=OUT / "key"),
    "validation": dict(strata=VALIDATION_STRATA, seed=SEED + 1, tables=OUT / "validation_set_tables.csv",
                       pages=OUT / "pages_validation", key_dir=OUT / "key_validation"),
    # Anna 2026-11-15: 100 more validation tables, to see how the tuned pipeline does on tables it was
    # not tuned on. Same design; excludes every page of the sets above. Scored as drawn, then usable for tuning.
    "validation2": dict(strata=VALIDATION_STRATA, seed=SEED + 2, tables=OUT / "validation2_set_tables.csv",
                        pages=OUT / "pages_validation2", key_dir=OUT / "key_validation2"),
}
TEST_SET = OUT / "test_set_tables.csv"
KEY = OUT / "key" / "answer_key.csv"          # one row per site per sampled table (see KEY_FORMAT.md)
DRAFTS = OUT / "key" / "drafts"               # Claude's first-pass JSON, one file per sampled table
LONG_TABLE_SITES = 5             # tables with more sites than this: count + first, last, 3 random
REVIEW_PORT = 8788

# Reader 2: open model run locally with vLLM (env vllm-rs). Weights on scratch, never in the repo.
MODEL_ROOT = Path("/home/hpc-scratch/annaboser/usgs_models/hf")
MODEL = "Qwen/Qwen3.6-27B-FP8"       # timed 2026-11-15: 1.4 s/page on one L40S
MODEL_MAX_CONTEXT = 21000            # tokens; largest test page so far ~5,000 in, answers < 1,500
MODEL_MAX_ANSWER = 8000              # tokens; long site lists need room
MODEL_MAX_PAGES_AT_ONCE = 48         # with the default batch settings the 46 GB card ran out of memory
MODEL_RUNS = OUT / "model_runs"      # one folder per (set, prompt version): answers.jsonl
PROMPT_VERSION = "v5"   # v1: 6 of 100 validation answers cut off at 3,000 tokens
TABLE_RESOLUTION = REPO / "data/analysis/table_resolution.csv"   # Annette's per-table record type, variables, value counts
SCORES = OUT / "scores"                                          # {set}_{source}_long.csv and _summary.csv

TIDY_RULES = ("runoff_beside_discharge", "generic_water_source")   # three other rules removed 2026-11-15: they fired on 0 of 519 validation rows
REQUIRE_ON_PAGE = True        # a copied location value is kept only if its characters are found in the page text
