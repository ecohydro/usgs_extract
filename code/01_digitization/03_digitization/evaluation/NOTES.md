# Table digitization evaluation: decisions that hold and known limits

The full dated history (attempts, provisional figures, fixes) is in git: commit 7633608 holds the
last complete version of this file.

## Decisions

- **What is scored** (Anna, 2026-11-14). Three outputs for the same pages: raw PaddleOCR, raw
  Reducto (the HTML tables in its JSON), and the published CSV (made from the Reducto JSON by
  `../reducto/02_json_to_csv.py`).
- **Truth set** (Anna, 2026-11-14). 20 pages drawn at random, 4 per kind of table (daily, monthly,
  annual, discrete, no time series), no report used twice, from the 10,852 pages of 57 reports that
  have both Reducto and PaddleOCR output. Unit = page, because the tools split a page into different
  numbers of tables. Seed 20261114.
- **Kind of a page** (Claude, 2026-11-14) = kind of its largest published Reducto table
  (`data/analysis/table_resolution.csv`). Pages mixing kinds are kept: requiring one kind per page
  would drop 61% of the frame.
- **Replaced page** (Anna, 2026-11-14). 1312 p239 lists U.S. Indian Service wells at Pala and
  Rincon; it is dropped and replaced by the next page of its kind in the random order (623 p441).
  The other 19 pages are unchanged. Its typed files are kept out of git (`excluded/`).
- **Typing** (Anna, 2026-11-14). The printed tables were typed from the page images by Claude, with
  no access to any tool output, following TRUTH_FORMAT.md, then read against the scans by Anna
  (review pages: `03_review_pages.py`, `review_server.py`). Agreed wording: "The transcription was
  typed from the page images by Claude and then read against the scans by one author (A.B.), who
  found no errors beyond the typing of superscripts and subscripts, which was corrected."
- **Headline metric** (Anna, 2026-11-14): numbers right and findable. A printed number counts only
  if the number matches and its row label and its column header match. Every printed number gets
  exactly one outcome (listed at the top of scoring.py). Reported beside it: output numbers that
  match nothing printed, tables found / split / merged / missed, marks kept, and a GriTS-style
  score (own implementation, supplement only).
- **Lenient numbers are the default** (Anna, 2026-11-14): gaps of typesetting are closed ("21. 2",
  "4, 400") and a lone letter o is read as 0. The strict run (`_strict`) does neither.
- **Labels** (Claude; Anna accepted 2026-11-14): compared as bare letters and digits, so case,
  spacing, punctuation, leaders, line-break hyphens and superscript or subscript form do not count.
  A footnote mark on a printed label may be dropped by the tool (Anna, 2026-11-14). A table with no
  printed header cannot fail on column headers.
- **648 p556 is one table** (Anna, 2026-11-14): four ruled blocks under a header printed once.
- **Published CSV read in its favour** (Claude, 2026-11-14): pandas "Unnamed: ..." header cells count
  as blank and ".1" duplicate-name suffixes are removed.
- **Corpus figure** (Anna, 2026-11-16, relayed by her methods-draft session). Every printed number
  counts equally; a kind of table stands for its share of published pages times the mean number of
  printed numbers on its test pages (`corpus_share` in 05_summarize.py). The project's value counts
  are not used. 95% range from 5,000 resamples of the test pages within each kind.
- **Check of the scorer** (Anna, 2026-11-14): 100 verdicts drawn evenly over (output, outcome)
  groups and checked by Claude against the typed truth and the tool output, not the scan. 98
  correct; 0 wrong about right and findable; 2 in the wrong bin (641 p109), which led to the rule
  that an output cell is credited to one printed table only. The graded sample was drawn before
  that fix and is kept as drawn; it names the default run "_repair", its name at the time.

## Known limits

- 20 pages, 4,696 printed numbers, 32 printed tables. Ranges are wide for the rarer kinds.
- 17 of the 20 pages are stream discharge (the frame is about 90% stream discharge).
- Pages whose largest table has no kind (8% of published pages) are not represented.
- The frame is the pages with PaddleOCR output: 57 of 586 reports.
- The kind of a page comes from the corpus classification, which is not perfect: 623 p441, filed
  under no time series, is a daily grid.
- 95 cells the typists flagged as unsure stand as typed (`truth/*_uncertain.csv`).
- The tribal screen file was not on this machine, so the frame was not screened; the 20 pages were
  checked by eye.
- Most of the published CSV's loss against raw Reducto is column headers lost when multi-level
  headers are flattened by the conversion script. That script is not changed here.
- The original test set of the dissertation (13 tables, 2,734 cells) and its scoring code are lost
  and were not reproduced.
- GriTS figures come from our own implementation, not the official code.
