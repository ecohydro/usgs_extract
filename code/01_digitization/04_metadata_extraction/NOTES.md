# Metadata extraction: decisions that hold and known limits

How to run everything and where each number comes from: README.md. The full history of what was tried
(prompt versions v1 to v4, dropped rules, a counting script, the comparison with the earlier published
files) is in git at checkpoint e2f0513 (2026-11-17), where this code lived in code/05_evaluation/metadata.

## Decisions

Product
- 2026-11-15, Anna and Annette: one row per location per table; each page is processed alone and the
  process never looks at other pages. Keys: doc_id, page_number, table_number, location_number.
- 2026-11-15, Anna: graded columns are location, location_type, water_source, data_type, date_start,
  date_end, temporal_resolution, variables, construction_date. Dates as fine as printed (1916 / 1916-10 /
  1916-10-01). `data_type` is Annette's nine categories, unchanged. `variables` holds measured quantities
  and site descriptions in one list. A table with no water data is one row, Not Water Related.
- 2026-11-15, Anna: no `units` column. Fact told to her: one variable does come in different units in
  these reports, so a user must open the table to know which.
- 2026-11-15, Anna: the number of measurements is Annette's postprocessing, not part of this pipeline. The
  answer keys still hold n_measurements and n_variables (schema.COUNTS_IN_KEY), ungraded.
- 2026-11-15, Anna and Annette: all location evidence printed on the page is collected (coordinates, well
  number, station number, station name, township-range-section, place, basin or area, county, state) and
  a fixed order picks `location` and `location_type` (schema.EVIDENCE; Annette's ranking extended
  downward). Turning a location into a point or basin is Annette's work, not this pipeline's.

Evaluation
- 2026-11-14, Anna: the unit is one published table; 200 test tables (80 stream discharge, 50 groundwater,
  35 reservoir, 35 other types pooled), random within group, reweighted to the published tables for the
  overall figure.
- 2026-11-15, Claude (not put to Anna as a question): one table per page and at most three tables per
  report; pages used in any earlier development or grading (508 pages) are excluded from the frame.
- 2026-11-15, Anna: any drawn table on a page withheld for tribal data sovereignty is removed and
  redrawn. tribal.py mirrors Annette's rule; 3 test tables were replaced (and 1 and 5 in the validation
  sets). The screen outputs are sensitive and git-ignored.
- 2026-11-15, Anna: the evaluation asks only whether the pipeline extracted what the PAGE gives. The answer
  key is page-only: blank means the page does not give it.
- 2026-11-15, Anna: answer keys are drafted by Claude from the page scan alone (blind to every pipeline
  output), following KEY_FORMAT.md. Anna checked the first 20 test tables and found no errors
  (key/checked.csv); the rest is not checked by a person. Five cells of those 20 were later edited by
  Claude to match the written rules (key/edit_log.csv).
- 2026-11-15, Anna: two validation sets of 100 tables for development; the test set is scored once, after
  her go, with the pipeline fixed beforehand. Done 2026-11-15 (evaluation/05_score.py test pipeline v5 --final).
- 2026-11-15, Anna (go read from "ok good let's go then"): the written key rule "runoff beside discharge in
  a daily or monthly table is not a separate variable" was applied to all keys by script
  (evaluation/03_key_runoff_rule.py): 1 test row, 8 validation rows, 0 in the second validation set.
- Score reported = "found": of the rows where the page gives a value, the share the output has right,
  weighted by group. Scorer conventions: coordinates compared by their digits; gage height and elevation
  count as one variable; a date is right if it agrees with the key as far as the coarser of the two goes.
- 2026-11-17: 95% ranges by resampling tables within their group, 5,000 draws, seed 20261116.

Pipeline (fixed 2026-11-15, Anna: "lock in the pipeline")
- Model Qwen3.6-27B FP8 on our own GPUs (vLLM 0.21, one L40S, reasoning off, text only), chosen from open
  models that fit one card; no bake-off against other models was run.
- No input from the 2025 Phi-4 output and no fallback to Annette's per-table files (Anna asked whether
  they were needed; on validation the scores were identical without them).
- Two tidy rules kept (pipeline.tidy); the found-on-page check is kept as a guarantee that location words
  are copied, not invented.
- 2026-11-17, Anna: the comparison with the earlier published per-table files and the recomputation of the
  2025 Phi-4 grades are out of the paper and out of this code; the code moved here from
  code/05_evaluation/metadata, pipeline at the top and evaluation in evaluation/.
- 2026-11-16, Anna: the final table is committed as is, including rows for pages on tribal lands (the
  repository is public; she was told); it is to be cleaned before publication.

## Known limits

- Groundwater is the weak group (test: location 61%, variables 59%). Causes seen: short well numbers without their township prefix,
  Reducto misreads copied faithfully, the wrong column taken in wide well inventories, and an exact-list
  rule for variables (any overlap: 93%).
- The key is machine-drafted; only 20 of 200 test tables were checked by a person.
- The model reads Reducto's text, not the scan, so a digitization misread ("165/43E" for "16S/43E") is
  copied into the metadata.
- Of the misses on the test set, about 87% are a wrong value, 7% an omission, 5% a value where the page
  has none. Location words must be on the page; 71 of 73 wrong dates use a year printed on the page.
- Found-on-page check: on the two validation sets it drops 77 of 2,181 location values (3.5%), nearly all
  states (43) and counties (30) the model supplied without the page printing them (on_page_check.csv).
  An earlier figure of 0.4% came from an older prompt on the first validation set only.
- Full collection (2026-11-16): 63,524 pages read, 100,679 tables, 177,015 rows. 72 tables on 65 pages have
  an unreadable answer even after a second reading with longer limits (very long site lists); 59 tables
  got no answer. 7 tables carry the data type "Evaporation", which is not one of the nine (left as written).
- Six reports (52056, 23997, 23879, 23996, 23894, 52119; 155 published tables) have no Reducto files on the
  cluster copy and were not run (full_run/pages_without_table.csv). They need the files from Annette.
- The tribal screen catches only what its rule describes (a point inside or near a federally recognized
  area, or a reservation or rancheria named on the page).
- The weights still count tables on tribal lands in the frame (about 3%).

## Open

- Annette: the order of location types and the list of variables.
