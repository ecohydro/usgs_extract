# Table detection evaluation: running notes

Append-only, newest at the bottom. Outputs: `data/analysis/evaluation/detection/`. Tunables: `defaults.py`.
Python: `~/.venvs/usgs-eval/bin/python` (venv on top of conda env geofm-vlm, plus pymupdf, openpyxl).

## 2026-11-14 Inventory and what the old records say

- Cluster `/home/waves/data/usgs_extract/validation_test/`: 141 PDFs, 19,598 page images of 126 reports,
  `valiSet.csv` (142 reports, printed page ranges; same as Sheet2 of the repo xlsx), `hasTable.csv` /
  `validationWithPath.csv` (ranges expanded, 3,351 pages, PRINTED numbers used as if PDF positions).
- The file with PDF page numbers (`new_set.csv`, column "New PDF Pages", used by Yibo's ValidationTest notebook)
  is not on the cluster, in the repo, in Anna's home or Drive.
- Two saved score files for the validation images: `validation_test/tablesBbox/results.csv` (Jul 23 2024) and
  `csvs/validation_results.csv` (Aug 8 2024). They differ on 16,916 of 19,598 images.
- Current `01_detect.py` reproduces tablesBbox/results.csv exactly (25/25 random pages, max score within 0.01)
  and the saved production scores (`csvs/cali_1970_results.csv`, `cali_1980_results.csv`) on 13/13 pages
  found. The Aug 8 file is another run of unknown origin (median score difference 0.07).
  Removing the Otsu binarisation does not reproduce it either.
- Neither saved file with the printed-page list reproduces the paper's 0.7796: 0.55 to 0.73 at 0.75/0.8
  depending on the rule. Meaningless anyway (wrong pages).
- Thresholds actually used in production, traced through to what was sent to Reducto: batch two
  (`cali_1980_results`, images in UpdatedDataJan) cut exactly at 0.75 (all 31,131 pages at or above went on, none
  below). Batch one (`cali_1970_results`, UpdatedDataDec): 96% of pages >= 0.8 went on, about 20% of 0.75-0.8,
  7% of < 0.75, so about 0.8 plus unexplained extras. Paper and code default say 0.8.
- Missing report 70174986 (Kansas "Equus beds", 1940): not a USGS-hosted PDF (journal pages 839-866; the
  publications API lists only a thumbnail). Dropped. Report 1389 lists "throughout": dropped (no page list).
- 15 listed reports have a PDF but no 2024 images (1274, 1305, 1343, 1390, 2035, 2278, 23271, 2650, 2773, 2907,
  3133, 32883, 4776, 52018, 52019); rendered now from the PDF with pymupdf at 300 dpi grayscale
  (03_preprocess.py used pdf2image/poppler at 300 dpi; scans are embedded images, so differences should be small;
  not yet measured).

## 2026-11-14 Decisions (Anna, grilling)

- Anna: redo the evaluation properly instead of recovering the old number.
- Anna: report recall at both 0.75 and 0.8 plus the full curve; flag to Annette that the two batches differ.
- Anna: no second hand-labelled California sample for recall; the groundwater list only.
- Anna: page numbers rebuilt from the PDFs, checked against the notebook's surviving answers, she eyeballs ~30.
- Anna: fresh detector run with the current code (first answered "matching old run", then changed to a fresh run
  when she chose to redo the evaluation). Current code is verified to be what was used (above).
- Anna: denominator = every listed page with a score, including reports where nothing is detected
  (Yibo's notebook dropped reports with no detection above threshold, which inflates recall).
- Anna: Claude looks at every listed page that scores low and removes those with no table, plus a random 150
  flagged pages; she gets a sample of the calls and all borderline ones.
- Anna: false alarms from the corpus: share of flagged pages for which Reducto returned no table (her idea),
  corrected by Claude looking at 60 pages with a Reducto table and 60 without.
  First numbers (flagged pages sent to Reducto, share with no table CSV back), by best detector score:
  0.75-0.80 47% of 1,950; 0.80-0.90 42% of 8,166; 0.90-0.95 25% of 5,649; 0.95-0.99 6% of 12,021;
  >= 0.99 2% of 50,928; overall 9.6% of 74,500.
- Anna: a table = data in rows and at least two aligned columns, ruled or not, including continued and rotated
  tables. Not tables: contents pages, indexes, lists of illustrations, single-column well logs, running text.
  Unsure = borderline, shown to her.

## 2026-11-14 Page map (page_offsets.py)

- First attempt (vote for the shift among pages within 6 of the target) got 1789 and 937 wrong (two shifts tied)
  and left 880 pages unplaced. Abandoned.
- Current: trusted pages (printed number corroborated by 2 pages within 15 sharing the shift, then the longest
  chain increasing in page and number); a listed page is placed from the trusted pages before and after it.
  All 7 answers visible in Yibo's notebook reproduced (2706 +5, 3133 -96, 2132 +25, 2096 +1, 1789 +5, 1063 -57,
  937 +5).
- Result before visual corrections: printed_on_page 1,559; between_agreeing_pages 1,128; extrapolated 504;
  between_disagreeing_pages 141; not_found 14 (of 3,346 listed pages in 140 reports).
- The uncertain three groups are being read off the scans (page_check_sheets.py, 118 strips, read by a helper
  agent into page_check/uncertain_read.csv); corrections will go in a hand-entered overrides file.
- Detector run (run_detection.py) scores every mapped page plus 6 pages either side of the uncertain ones
  (5,413 pages), so corrections to the map need no new run.

## 2026-11-14 (later) Page map corrections, false-alarm check

- Running-head numbers ("46 REPUBLICAN VALLEY ...", "AURORA COUNTY. 65") added to the page-number reader:
  printed_on_page 1,477 -> 1,559.
- Helper agent read 118 header strips of the uncertain reports (page_check/uncertain_read.csv): 66 as mapped,
  30 differ, 22 no number visible. Corrections entered by hand in `page_map_overrides.csv` (19 rows) and applied
  by `apply_overrides` -> `page_map_final.csv`: read_off_scan 167, extrapolated 453 (all checked blocks
  confirmed except report 53769, where no number is visible but the three pages checked are tables),
  between_disagreeing_pages 38 (reports 2280 and 2505, the first candidate confirmed), not_found 1.
- Unplaced: report 2278 printed 109 (PDF 109 prints 81; 109 + 28 exceeds the 118 PDF pages).
- Report 2585 and 52938 list PDF positions, not printed numbers (printed 152-160 do not exist in 2585).
- Report 2225 (Florida): the list names the page a fold-out table FACES ("To face page 284"); the table is the
  unnumbered PDF page after it. Seven such pages placed by hand. Because other reports may do the same, every
  listed page that turns out to hold no table has its neighbours (2 either side) looked at before it is
  dropped (list_check.py neighbours); the detector therefore also scores 2 pages either side of every listed page.
- Detector speed: about 0.25 s/page on 2024 images, slower when rendering; 5,413 pages take about 40 minutes
  (estimate given to Anna was 15).
- False alarms (reducto_false_alarms.py, false_alarm_estimate.py). Flagged pages with a saved score sent to
  Reducto: 74,492 (48,823 batch one, 25,669 batch two after removing pages in both score files; 8 have no saved
  score). Blind labels by helper agents, sheets of 4 pages; Claude spot-checked 8, all agreed.
  First draw 60 + 60: no-table-back group 55 no table, 3 real tables, 1 borderline, 1 no image;
  table-back group 58 table, 1 index, 1 borderline (headed well log).
  Claude added 240 more from the table-back group (N_REDUCTO_EXTRA): 1 of 59 errors in the group holding 90%
  of pages left the estimate too loose. Extra draw: 225 table, 8 no table (4 index/contents, 2 stratigraphic
  sections, 2 station/contents lists), 7 borderline (2 headed well logs, 4 ruled tables that are lists of
  references or stations, 1 no image).
  Corrected false-alarm rate among flagged pages: at 0.75, 11.8% (10.1-13.8) with borderline as table, 14.0%
  (11.8-16.5) with borderline as no table; at 0.8, 10.9% (9.1-12.9) and 13.2% (10.8-15.8).
  Uncorrected Reducto share: 9.7% at 0.75, 8.7% at 0.8.
- Open for Anna: the borderline ruling (headed well logs with thickness and depth columns; ruled tables whose
  content is a list of references or stations).

## 2026-11-14 (evening) Fresh detector run, list check, recall

- Fresh run (run_detection.py): 6,153 pages scored (listed pages, 2 either side of each, 6 either side of the
  uncertain ones). Against the matching 2024 run on the 4,593 shared pages: 4,588 within 0.01 of the old best
  score; same side of 0.8 on 4,591, of 0.75 on 4,590. Pages rendered now: 733.
- Scores below about 0.5 do not occur: 01_detect.py keeps a detection only when 'table' beats 'no object', so
  recall is flat (95.8%) below 0.5.
- List check (list_check.py), blind to the score, labels by three helper agents from sheets of 4 pages, Claude
  spot-checked 8 (all agreed): 541 listed pages scoring < 0.8 -> 466 table, 71 no table, 4 borderline;
  150 random pages scoring >= 0.8 -> 141 table, 1 no table, 8 borderline.
  The 72 no-table pages: 45 in report 2907 (text pages inside listed ranges), 9 in 2225 and 8 in 2756
  (list names the page a fold-out faces), a few others.
- Neighbours of the no-table pages (82 unlisted pages, one helper agent): 29 table, 35 no table, 18 borderline.
  18 listed pages moved to an adjacent table page (2225: 9, 2756: 7, 1312: 2; method moved_to_adjacent_table).
  Map before the moves kept as page_map_before_list_check.csv. Remaining no_table 54, removed in 'cleaned'.
  Not looked at: listed pages scoring >= 0.8 outside the sample of 150 (1 of 150 was not a table).
- Recall (score_recall.py; range = 2.5-97.5% of 2,000 resamples of the 139 reports):
  cleaned list, 3,291 pages: 0.75 -> 87.7% (84.2-90.8), 2,886 found; 0.8 -> 85.5% (81.8-88.7), 2,814 found.
  list as given (after the 18 moves), 3,345 pages: 0.75 -> 86.4% (82.7-89.9); 0.8 -> 84.2% (80.4-87.6).
  9 of 139 reports have no listed page found at 0.8.
  Old records for comparison: Yibo's notebook 0.7796 at 0.75 (reports without detections dropped); Methods
  Report 83% at 0.5, 79% at 0.75, 80% at 0.8.
- Excluded reports/pages: 70174986 (no PDF), 1389 ("throughout"), 2278 printed 109 (not in the PDF).
- For Anna: for_anna/check.html (anna_check.py): 30 page-map pages, all 21 borderline pages, 40 calls.
- Not committed; rendered_pages/ (332 MB), sheets and for_anna/img should stay out of git.
- Open: Anna's borderline ruling; Anna's verification; whether Annette knows who compiled the list.

## 2026-11-14 (night) Anna's verification

- Anna looked at for_anna/check.html (30 page-map pages, 21 borderline pages, 40 table / no-table calls, served on
  port 8787) and said: "I checked it is correct". Taken as: page numbers and calls verified.
- Still open: her ruling on whether the borderline kinds count as tables (her answer did not say).
- Anna (ruling, after seeing all 21 borderline pages): "they're all tables except Corpus page 5647_page_158".
  Frozen in defaults.BORDERLINE_NOT_TABLE; false_alarm_estimate.py row 'anna_ruling' is the result of record.
  Recall unchanged (borderline listed pages were already kept as tables in 'cleaned').
