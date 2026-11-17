# Table detection evaluation: decisions and limits

Outputs: `data/analysis/evaluation/detection/`. Tunables: `defaults.py`. Run order and the source of each
number: `README.md`.

## Decisions that hold

- Anna, 2026-11-14: redo the evaluation instead of recovering the old number (the dissertation's 79% at 0.8
  cannot be reproduced: the old run used printed page numbers as PDF positions and dropped reports with no
  detection).
- Anna, 2026-11-14: report recall at both 0.75 and 0.8, plus the full curve. Traced from what was sent to
  Reducto: batch two (scores in `cali_1980_results.csv`) was cut exactly at 0.75; batch one
  (`cali_1970_results.csv`) at about 0.8 (96% of pages at or above 0.8 went on, about 20% of those between 0.75
  and 0.8, 7% of those below 0.75).
- Anna, 2026-11-14: recall is measured on the groundwater list only; no second hand-labelled sample.
- Anna, 2026-11-14: fresh detector run with the current `01_detect.py`. It reproduces the saved 2024 scores
  (4,588 of 4,593 shared pages within 0.01).
- Anna, 2026-11-14: the denominator is every listed page with a score, including reports where nothing is found.
- Anna, 2026-11-14: a table is data in rows and at least two aligned columns, ruled or not, including continued
  and rotated tables. Not tables: contents pages, indexes, lists of illustrations, single-column well logs,
  running text. Unsure = borderline.
- Anna, 2026-11-14, after seeing all 21 borderline pages then on file: "they're all tables except Corpus page
  5647_page_158" (`defaults.BORDERLINE_NOT_TABLE`). Borderline listed pages stay in the list.
- Anna, 2026-11-14: verified `for_anna/check.html` (30 page numbers, 21 borderline pages, 40 calls): "I checked
  it is correct".
- Anna, 2026-11-16: every listed page is checked, not only the low-scoring ones (removing no-table pages only
  among misses can only raise recall). Labels by helper agents, blind to the score; a second blind helper looked
  at 128 pages of the second round (all 32 it had called no table and 96 others) and agreed on 124; the 4
  no-table pages it called borderline (well logs in report 2907) were set to borderline.
- Anna, 2026-11-14: false alarms come from the corpus: share of flagged pages for which Reducto returned no
  table, corrected by a blind look at a sample of each group.
- Claude, 2026-11-14: 240 more pages drawn from the group with a Reducto table (90% of flagged pages; 1 error
  in 60 left the estimate too loose).
- Anna, 2026-11-16: the two drawn pages with no image are replaced by one random page each from the same group,
  so the labelled groups are 60 and 300.
- A listed page with no table moves to the nearest unlisted page within 2 that holds a table (the page after
  wins a tie), because some lists name the page a fold-out table faces (reports 2225, 2756). The rule does not
  use the detector score.

## Limits

- The list covers groundwater tables in 139 reports, mostly outside California; recall on other table kinds
  is not measured. Who compiled the list is not confirmed.
- Excluded from the list: report 70174986 (no USGS PDF), report 1389 (pages given as "throughout"), report 2278
  printed page 109 (not in the PDF).
- 62 of the 73 removed pages are in report 2907 (text pages inside listed ranges).
- Page positions: 453 listed pages are extrapolated from numbered pages on one side only; the blocks checked by
  eye were confirmed, except report 53769 where no number is visible. 15 reports had no 2024 page images and
  were rendered now from the PDF (817 pages with their neighbours; pymupdf at 300 dpi, production used poppler).
- The detector keeps a detection only when "table" beats "no object", so scores below about 0.5 do not occur
  and recall is flat (about 96%) below that.
- The table / no-table labels come from helper agents (Claude models), not people; Anna verified a sample of 40
  calls and the 21 borderline pages of the first round. The 133 borderline pages of the second round (almost
  all headed well logs with thickness and depth columns) follow her ruling without her having seen them.
  Dropping every borderline listed page would give 87.1% and 84.8%.
- False alarms: 8 pages sent to Reducto have no saved detector score and are left out. "Reducto returned no
  table" means no table CSV exists for the page.
- At 0.8 the false-alarm samples are 54 and 296 pages, not 60 and 300.
