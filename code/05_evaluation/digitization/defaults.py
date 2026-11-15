"""Every tunable of the table digitization evaluation. Nothing numeric lives elsewhere."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/analysis/evaluation/digitization"

# Read-only inputs on the cluster (never written to).
CLUSTER = Path("/home/waves/data/usgs_extract")
REDUCTO_CSV = CLUSTER / "ReductCSVs"          # {doc}/{doc}_page_{N}_table{k}.csv
REDUCTO_JSON = CLUSTER / "ReductJson"         # {doc}/{doc}_page_{N}.json
PADDLE = CLUSTER / "paddleOutput"             # {doc}_page_{N}_table_{k}/ or ..._table_rotated_{k}/
PAGE_IMAGES = (CLUSTER / "UpdatedDataDec", CLUSTER / "UpdatedDataJan")  # {doc}/{doc}_page_{N}.png

TABLE_RESOLUTION = REPO / "data/analysis/table_resolution.csv"  # table kind per Reducto table

# Sample design (Anna, 2026-11-14: frozen). Unit = page.
KINDS = ("daily", "monthly", "annual", "discrete", "no_time_series")  # kinds of table to cover
PAGES_PER_KIND = 4        # frozen 2026-11-14
ONE_PAGE_PER_REPORT = True  # frozen 2026-11-14: no report used twice
SEED = 20261114           # draw date
EXCLUDED_PAGES = {        # dropped after the draw and replaced by the next page of the same kind
    (1312, 239): "lists U.S. Indian Service wells at Pala and Rincon (Anna 2026-11-14: replace)",
}

# Truth set
TRUTH = OUT / "truth"                 # typed grids + tables.csv + uncertain.csv (see TRUTH_FORMAT.md)
SAMPLE = OUT / "sample_pages.csv"

# Scoring (proposals until Anna settles them; variants are sensitivity runs keyed in the store)
SYSTEMS = ("paddle_raw", "reducto_raw", "reducto_csv")  # Anna 2026-11-14: score all three
LABEL_MATCH_MIN_RATIO = 1.0   # 1.0 = labels must be equal after trivia (case, spacing, punctuation,
                              # leaders) is removed. Proposal; 0.9 is the sensitivity run.
REPAIR_NUMBER_FORMAT = True   # frozen, Anna 2026-11-14 (lenient headline): close typesetting gaps
                              # ("21. 2", "4, 400") and read a lone letter o as 0 before comparing.
                              # False is the "_strict" run, reported alongside.
IGNORE_LABEL_FOOTNOTES = True  # frozen, Anna 2026-11-14: a footnote superscript on a printed row label
                               # or header may be dropped by the tool without failing the label
MARK_MAX_CHARS = 3            # a cell is "a number with a mark" if at most this many other characters
LINK_MIN_EQUAL_CELLS = 3      # an output table is scored against a truth table only if at least this
LINK_MIN_EQUAL_SHARE = 0.10   # many, and this share of the smaller table's filled cells, agree exactly
SPLIT_MERGE_MIN_SHARE = 0.10  # a piece counts toward "split"/"merged" if it supplies this share of a
                              # truth table's numbers
VARIANTS = {                  # key suffix -> overrides; "" is the default run
    "": {},
    "_fuzzylabel": {"LABEL_MATCH_MIN_RATIO": 0.9},
    "_strict": {"REPAIR_NUMBER_FORMAT": False},
}
BOOTSTRAP_DRAWS = 5000        # page resamples (within kind) for ranges

# Check of the scorer's own verdicts (Anna 2026-11-14: draw 100, Claude checks)
VERDICT_CHECK_N = 100        # verdicts drawn, spread evenly over (output, outcome) groups
VERDICT_CHECK_VARIANT = "default"   # drawn 2026-11-14 when these settings were the "_repair" variant

# Review pages: scans shown turned, in degrees clockwise (Anna 2026-11-14). The files in pages/ stay as scanned.
REVIEW_ROTATE_CW = {(1110, 231): 90}
REVIEW_PORT = 8787   # Anna 2026-11-14
