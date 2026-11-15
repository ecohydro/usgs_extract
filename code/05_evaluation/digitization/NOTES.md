# Table digitization evaluation: notes

## 2026-11-14

**Brief (Anna).** Defensible, reproducible evaluation of the table digitization step (Reducto used,
PaddleOCR tested and rejected). Paper claims: PaddleOCR 92% of cell values, Reducto 100% on 13
hand-digitized tables (2,734 cells). Answers, table list and scoring code are lost.

**Old sample pages located.** `Yibo Results/Sample Table Column Detection/images/` matched pixel for
pixel (normalised 48x64 thumbnail correlation = 1.0) to corpus page images:
1166 p31, 1258 p104, 1312 p162, 1553 p101, 23423 p28/29/30, 24284 p20/25, 366 p36.
23995 page_1 has no exact match (best 0.62); `sampleTables/23995_page26.png` suggests page 26, which
exists only in `cali_images/`, with no Reducto or PaddleOCR output. Reducto output exists for the
other 10 (17 tables, 2,866 cells, 2,297 non-empty: same order as the paper's 2,734, not confirmed as
the test set). PaddleOCR on the cluster for 1166, 1312, 1553, 366 only; the repo sample folder
`output/` has PaddleOCR for all 11.

**Hilton and Jasechko.** None of the five reports is in publication_list.csv (author + year search)
or in ReductJson. "Underground waters of eastern United States" (wsp114, id 1389; probably Fuller
1905, matched by title only) has 305 page images in validation_test/images, no Reducto, no PaddleOCR.

**Anna: truth set** = pages that already have Reducto and PaddleOCR output, several kinds of table;
Claude digitizes, Anna checks; criteria to be worked out together (structure + headers, and numbers).
Hilton and Jasechko not used.

**Anna: sample design (frozen).** 20 random pages, 4 per kind of table (daily, monthly, annual,
discrete, no_time_series), no report used twice. Unit = page (the two tools split a page into
different numbers of tables).

**Claude: page kind** = kind of the page's largest published Reducto table (most printed rows),
from data/analysis/table_resolution.csv. First draw required all tables on the page to share one
kind; abandoned because it removed 61% of pages (10,852 -> 4,266), i.e. every daily page printed
with its monthly summary. Mixed pages are kept and flagged (`n_kinds`, `kinds_on_page`).
Frame: 10,852 pages / 57 reports with both outputs; 10,521 of a kind in the design
(daily 7,215; monthly 1,677; annual 334; discrete 1,212; no_time_series 83).
Command: `python 01_draw_sample.py` (seed in defaults.py) -> sample_pages.csv, pages/.
Draw: 17 of 20 pages are stream discharge (frame is 90% stream discharge).

**Not done: tribal screen.** data/analysis/tribal_lands/tribal_screen_pages.csv is not on this
machine, so withheld tribal pages could not be excluded from the frame. Check the 20 pages by eye.

**Finding that shapes the scoring (1553 p81).** Raw Reducto HTML (ReductJson) is not error free:
"16." for 16.6 / 16.3 / 16.8 / 16.2 (last digit dropped), 13.2 for 13.4 (Sept, day 8), "21. 2".
The published CSV (code/01_digitization/03_ocr/reducto/02_json_to_csv.py, pandas) adds changes of
its own: 26 -> 26.0, two-level headers flattened with "Unnamed: 2_level_1" and misaligned.
So "Reducto" has two scoreable artifacts: raw HTML and published CSV.
PaddleOCR artifacts: res_0.txt (JSON with cell boxes + HTML) and an .xlsx per table.

**Open (waiting on Anna).** Which Reducto artifact is scored; transcription rules; metric.

**Anna: Reducto artifacts.** Score both raw Reducto (JSON HTML) and the published CSV, separately.

**Anna: metrics (settled in conversation).** Headline = "numbers right and findable": a printed
number counts only if the number matches AND its row label AND its column header match. Every
printed number gets exactly one outcome (right_findable / row_label_wrong / col_label_wrong /
both_labels_wrong / incorrect / missed_cell / missed_table); invented numbers reported separately
(output side); tables found/missed/split/merged reported as counts and folded in as causes;
GriTS as a supplement line only; share of printed marks kept as a small supporting number.
Dropped by Anna: "printed form preserved" (26 vs 26.0) and the old position-free method replica.
Labels are forgiving of trivia; repeated/absent labels fall back on order (the alignment is
order-preserving, so this is automatic).

**Anna: "go"** on typing. Typing rules of TRUTH_FORMAT.md were my recommendation; she did not
answer that question separately. Treat as proposals until she has seen the review pages.

**Typing.** Five subagents (one per kind, 4 pages each), image only, forbidden to open any tool
output; zoomed crops via crop.py; each wrote per-page grids, a tables file and an uncertain file.
First launch of the daily batch went out with the page list missing (my slip); it typed nothing and
was resumed with the list. 93 cells flagged uncertain. Typists' conventions are appended to
TRUTH_FORMAT.md. Caveat for the paper: the truth is machine-typed (Claude) then human-checked; some
uncertain digits were chosen so that column totals agree (all listed in the uncertain files).
- 648 p556: typist typed four ruled blocks as four tables (header printed once, above block 1).
  I merged them into one table (header applies to all). This matters: Reducto split the page into
  two tables and the second (62 rows) has no header, so 59 numbers are col_label_wrong. Under the
  four-table typing they would be right. Anna to decide.
- 1312 p239 names "U. S. Indian Service" as owner of wells at Pala and Rincon. Tribal screen file
  is not on this machine, so I cannot tell whether the page is on the withheld list. Anna to decide.
- 251 p309 is in Arizona, 1276 p80 in Colorado (out-of-state pages are in the corpus; kept).

**Scorer decisions (Claude).**
- Alignment: order-preserving row and column pairing maximising cell agreement (GriTS two-step
  heuristic, own code, on span-expanded grids). Every truth table is compared with every output
  table on the page; a pair is linked if >= 3 cells and >= 10% of the smaller table agree exactly;
  each printed number takes its best outcome over linked output tables.
- Column header of an output column = the output cells above the first body row (or in the rows
  aligned to truth header rows), joined; compared with the truth header joined over levels.
  Declared th / thead tags are ignored, so a header row emitted as data still counts if it sits
  above the right column. A truth table with no printed header cannot fail on column labels.
- Label trivia: case, spacing, punctuation, leaders, and line-break hyphens ("Diam- eter") are
  ignored (labels compared as bare letters and digits). Before this, 1084 p275 lost 351 numbers to
  hyphenated headers that any reader can use.
- Published CSV read as text; pandas "Unnamed: n_level_m" cells treated as blank and ".1"
  duplicate-name suffixes removed, both in the tool's favour.
- MARK_MAX_CHARS 2 -> 3 so "a(2.55)" counts as a number with a mark.
- Variant `_repair` (REPAIR_NUMBER_FORMAT): closes typesetting gaps ("21. 2", "4, 400") and reads a
  lone letter o as 0. Reason: the pages really print a gap after the point, the typists dropped it,
  and Reducto often keeps it; Reducto reads zeros in some typewritten tables as "o" (593 p236: 61
  cells; the repo's value counts already treat lone o as zero). Which is the headline: Anna.
- Abandoned: none yet. Not done: official GriTS code (own implementation only); weighting by kind
  and bootstrap ranges (04_summarize.py not written).

**Provisional results (before Anna's check; `python 02_score.py`), 4,351 printed numbers, 20 pages.**
Pooled share right and findable: raw Reducto 80.5% strict / 85.0% with number repair; published
CSV 53.9% / 57.8%; raw PaddleOCR 5.2% / 5.4%.
Mean over pages by kind, with repair (raw Reducto | CSV | PaddleOCR):
daily .99 | .96 | .00; monthly .94 | .11 | .01; annual .96 | .68 | .34; discrete .77 | .77 | .01;
no_time_series .47 | .07 | .00.
Raw Reducto outcomes (repair): right 85.0%, column label wrong 1.4%, row label wrong 0.3%,
incorrect 2.5%, missed cell 0.9%, whole table missed 9.9% (the two rotated tables, 1110 p231 and
1312 p239: Reducto returns a scrambled, transposed grid). CSV: column label wrong 28.7%
(multi-level headers flattened by pandas). PaddleOCR: several printed rows crammed into one cell.
Not yet verified by eye: PaddleOCR's 9 missed tables, CSV split/merged counts.

**Review material.** `python 03_review_pages.py` -> review/index.html (scan beside typed tables;
yellow = typist unsure, orange = raw Reducto differs), flagged_cells.csv (423 cells),
corrections.csv (template, never overwritten).

**Anna: check of the scorer's verdicts.** Draw 100, Claude checks. `python 04_verdict_check.py`
draws 5 verdicts from each of the 20 (output, outcome) groups (variant `_repair`, seed in
defaults); `... show A B` prints, for each, the truth row and the exact output row / header /
label cell the scorer looked at. Checked by reading those (truth grid vs tool output), NOT against
the scan: this tests the scorer, not the typing. Result in scores/verdict_check_graded.csv:
- 98 of 100 verdicts correct under the stated rules. 0 of 100 wrong about the headline (all 15
  right_findable confirmed; all 85 others confirmed not right and findable).
- 2 in the wrong bin (641 p109, published CSV): the small headerless rating list was "found"
  inside the daily grid through coincidental zeros; should be missed_table.
  FIX: every (printed table, output table) pair is compared, then each output table's cells are
  shared out, the printed table with most agreement first; a claimed output cell cannot be
  credited again.
- Latent bug (no wrong verdict in the sample): "2½" parsed as 2 with a mark. FIX: vulgar
  fractions parsed (2½ = 2.5). Three cells per variant flipped right -> incorrect on 1084 p275.
- Effect of both fixes on the store: 110 of 39,159 verdict rows changed (3 variants); headline
  shares unchanged to 0.1 point (raw Reducto 85.0%, CSV 57.8%, PaddleOCR 5.4%, repair variant).
  The graded sample was drawn before the fixes; the draw is kept as is.
- Leniencies seen (all PaddleOCR, none in the headline): stray characters next to a number count
  as a mark ("B 10", "70!", "$18, 000").
- Harshness seen: one misread letter in a unit line ("Scc.-ft") fails the whole column under the
  strict label rule (passes under `_fuzzylabel`).
- 13 sampled verdicts hinge on a truth cell the typist flagged unsure (M. Lahse / M. Lause, 338 / 339).
- 04_verdict_check.py needed out_row / out_col in cells.csv; added to the scorer for this.

**Anna: replace 1312 p239** (U.S. Indian Service wells at Pala and Rincon). `EXCLUDED_PAGES` in
defaults.py; 01_draw_sample.py makes the full draw first, then replaces an excluded page by the
next page in its kind's random order, so the other 19 pages are unchanged (verified). Replacement:
623 p441 (Upper Klamath Lake near Klamath Falls, Oreg.: daily elevation grid + monthly elevation
and contents; filed under no_time_series because table_resolution.csv calls its largest table
that; typist found no tribal name on the page). The 1312 typed files are in
data/.../digitization/excluded/, its image removed from pages/. Sample numbers 17-20 shifted.
Note: the new "no_time_series" page is really a daily grid, so that stratum now holds 2 genuine
inventory/analysis tables, 1 duration table and 1 misfiled daily grid. The kind label comes from
the corpus classification (59/65 accurate), so some misfiling is expected and is left as drawn.

**05_summarize.py** (kind weights = share of published pages by kind of largest table: daily .659,
monthly .116, annual .046, discrete .149, no_time_series .030; range = 2.5-97.5 percentiles of
5,000 resamples of pages within kind). Provisional, 20 pages, 4,697 printed numbers, before
Anna's check. Share right and findable, corpus-weighted (95% range):
  _repair : raw Reducto 93.9% (87.3-98.2) | published CSV 80.4% (73.0-86.5) | PaddleOCR 1.9% (0.6-3.2)
  default : 86.3% (76.7-94.6) | 73.6% (64.3-82.2) | 1.8%
  _fuzzylabel: 86.5% | 78.7% | 2.0%
Number right, labels aside (repair): 97.0% both Reducto outputs; PaddleOCR 14.0%.
Pooled over all numbers (repair): 87.3% | 62.1% | 5.0%.
By kind (repair; raw | CSV | Paddle): daily .99 .96 .00; monthly .94 .11 .01; annual .96 .68 .34;
discrete .77 .77 .01; no_time_series .72 .32 .00.
Tables (32 printed): raw Reducto 29 found, 1 split, 2 missed; CSV same; PaddleOCR 22 found, 10 missed.
Output numbers matching nothing printed: Reducto 1.4%, PaddleOCR 3.8%. Marks kept (50 marked
numbers): Reducto 42%, PaddleOCR 22%. GriTS-style content F: .82 / .80 / .30.
- Anna 2026-11-14: lenient numbers are the headline (REPAIR_NUMBER_FORMAT frozen True; default run). Strict is the `_strict` variant. Old `_repair` rows = new `default` rows; old `default` = new `_strict`. Numbers in earlier entries keep their old labels.

**Anna: review feedback (2026-11-14).**
- corrections.csv "hard to work with" -> removed (it was empty). review_server.py now serves the
  review pages on REVIEW_PORT (8787, localhost only); every cell is editable in place, a change is
  written to the truth CSV and logged in review/edit_log.csv (old, new, time); a tick per page goes
  to review/checked_pages.csv. Rebuilding the pages (03) shows changed cells in green.
  The plain `python -m http.server` first used is gone. Server dies with the session: restart with
  `python review_server.py`.
- 1110 p231 scan shown turned 90 degrees clockwise (REVIEW_ROTATE_CW; review copy only).
- 1110 p231: "superscripts not caught". The typist had typed footnote superscripts as plain digits
  ("5L3 1", "4 329", "(2)") and chemical subscripts as plain digits. Retyped as superscripts /
  subscripts (29 cells there; 4 sub-headings on 1061 p39, incl. "CaCO²" which is printed with a
  superscript 2; 10 footnote letters "a" -> "ᵃ" on 1137, 1147, 1276, 648). Rule added to
  TRUTH_FORMAT.md. Scorer: labels compared after NFKC (SiO₂ = SiO2); a superscript digit is a mark,
  not part of the number ("⁴329" = 329 with mark ⁴; before, "4 329" was not a number at all);
  marks compared after NFKC. Headline unchanged (raw Reducto 93.9%, CSV 80.4%, PaddleOCR 1.9%).
  Not yet re-checked for superscripts by eye: the other 18 pages (typists' reports mention none
  beyond those converted).

**Anna: "done with corrections, no further mistakes" (2026-11-14, about 23:03).**
What the record shows: review/edit_log.csv and checked_pages.csv do not exist, i.e. no cell was
changed through the page and no page was ticked (the only two POSTs in the server log are my own
tests). The only truth changes that came out of her check are the super/subscript retyping she
asked for. Server log: all 20 pages were opened on the editing server (19 of them within one
minute at 23:01); the first, read-only server ran 21:57-22:51 and its log is not kept, so how long
she spent per page is unknown. The 95 cells the typists flagged unsure stand as typed.
How the check is described in the paper is hers to state; asked.
Still open: Q6 (648 p556 as one table; currently one table) and Q7 (footnote marks in labels;
currently a lost mark fails the label).

**Anna, 2026-11-14 late: answers.**
- Check wording (frozen): "The transcription was typed from the page images by Claude and then read
  against the scans by one author (A.B.), who found no errors beyond the typing of superscripts
  and subscripts, which was corrected."
- Q6: 648 p556 stays one table (frozen).
- Q7: IGNORE_LABEL_FOOTNOTES = True (frozen): a footnote superscript on a printed row label or
  header may be dropped by the tool without failing the label. Changed 0 verdicts on this sample
  (the only labels with footnotes that carry numbers are on 1110 p231, which Reducto and PaddleOCR
  miss whole). "Count under marks kept" for labels is NOT implemented: nothing to count here;
  `out_row_label` is now in cells.csv if it is ever needed.
- PaddleOCR's 10 missed tables checked: every one is a page where PaddleOCR produced a single
  table (the table-detector crop) that is another printed table, a crammed grid, or noise
  (383 p43: one text cell repeated; 623 p441: daily grid only; 1110: rotated noise). Genuine.

**Weighted figures (Anna: "what we'd expect if we'd done a random draw").** 05_summarize.py now
weights kinds two ways (summary/kind_weights.csv, overall.csv column `weighted_by`):
pages (daily .659, monthly .116, annual .046, discrete .149, no_time_series .030) and recorded
values (.833, .064, .024, .066, .013). Unclassified pages (8% of published pages) have no stratum.
FINAL for this truth set, default (lenient) run, share right and findable (95% range):
  by pages : raw Reducto 93.9% (87.3-98.2) | published CSV 80.4% (73.0-86.6) | PaddleOCR 1.9% (0.6-3.3)
  by values: raw Reducto 96.5% (93.5-98.8) | published CSV 88.2% (82.7-92.5) | PaddleOCR 1.1% (0.4-1.8)
  strict, by pages: 86.3% | 73.6% | 1.8%.   Number right labels aside, by pages: 97.0% | 97.0% | 14.0%.
Page-weighted outcome shares (raw Reducto | CSV | PaddleOCR): row label wrong 0.1 | 0.1 | 1.6;
column label wrong 3.1 | 16.6 | 3.1; both 0.0 | 0.0 | 7.4; incorrect 1.9 | 1.9 | 22.3;
missed cell 0.3 | 0.3 | 48.0; whole table missed 0.8 | 0.8 | 15.7.
Not done: official GriTS code (own implementation only); human README for the folder; fix of
the CSV converter (out of scope: code/01_digitization).
