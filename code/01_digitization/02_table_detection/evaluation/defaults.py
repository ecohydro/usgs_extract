"""Every tunable and path of the table-detection evaluation. Nothing numeric lives elsewhere."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data/analysis/evaluation/detection"            # all derived outputs
CLUSTER = Path("/home/waves/data/usgs_extract")              # read-only raw data
LIST_XLSX = REPO / "data/digitization_intermediates/02_table_detection/validation/groundwater_table_pages.xlsx"
LIST_SHEET = "Sheet2"                                        # the 142 reports with page ranges
VAL_PDFS = CLUSTER / "validation_test/pdfs"                  # PDFs of the listed reports
VAL_IMAGES = CLUSTER / "validation_test/images"              # page images made in 2024 (126 reports)
NEW_IMAGES = OUT / "rendered_pages"                          # pages rendered now for reports without images
DETECT_SCRIPT = Path(__file__).resolve().parents[1] / "01_detect.py"

RENDER_DPI = 300            # same as 03_preprocess.py
THRESHOLDS_USED = (0.75, 0.80)   # Anna 2026-11-14: report both (batch two used 0.75, batch one 0.8)
CURVE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)] + [0.99]
N_BOOTSTRAP = 2000          # resamples of reports for the uncertainty range
SEED = 20261114
HEADER_LINES = 4            # text lines at the top/bottom of a page searched for the printed page number
ANCHOR_WINDOW = 15          # PDF pages either side searched for pages sharing a page's printed-to-PDF shift
ANCHOR_SUPPORT = 2          # such pages needed before a page's printed number is trusted
N_REDUCTO_SAMPLE = 60       # pages per group (Reducto table / no table) for the false-alarm check
CHECK_DPI = 110             # resolution of the page-number check strips
CHECK_TOP = 0.10            # share of the page height shown from the top
CHECK_BOTTOM = 0.07         # and from the bottom
PAGE_CHECK_SAMPLE = 30      # listed pages shown to Anna to verify the page map
MARGIN = 6                  # extra PDF pages scored either side of a listed page whose position is uncertain
OVERRIDES = Path(__file__).parent / "page_map_overrides.csv"   # corrections read off the scans (hand-entered)
NEIGHBOUR = 2               # PDF pages either side of every listed page also scored (unnumbered fold-out tables)
N_REDUCTO_EXTRA = 240       # extra pages from the 'Reducto returned a table' group (1 error in 60 left the estimate too loose)
ANNA_LABEL_SAMPLE = 20      # table / no-table calls shown to Anna from each of the two checks
BORDERLINE_NOT_TABLE = ["5647_page_158"]   # frozen, Anna 2026-11-14: borderline pages are tables except this one (fossil locality register)
