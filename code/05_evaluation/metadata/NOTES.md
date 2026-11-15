# Metadata extraction evaluation: running notes

## 2026-11-14 Fact finding (before any decisions)

Environment: `~/miniconda3/envs/smh_irr_labels_2/bin/python` (pandas, pyarrow, geopandas). No conda
env has openpyxl; installed to the session scratchpad and used via PYTHONPATH. Needs a durable home
(ask Anna before installing into an env).

### Can the 87% be recomputed? Yes
`01_recompute_phi4_grades.py` reads the grader rows in validation_curated_135.xlsx and
validation_random_135.xlsx (outputs: phi4_grades_*.csv).
- Pooled lenient accuracy (1 / 0.5 / 0): curated 87.3%, random 87.2%. Per-field values match
  test_set_cat_grades.xlsx to within 0.1 point. Strict (1 only): 85.5% / 84.8%.
- Document-cluster bootstrap 95% interval: curated 85.0 to 89.7, random 83.7 to 90.5.
- Actual size: curated 445 entries on 122 pages from 49 documents (graders LB 286, HV 159);
  random 292 entries on 123 pages from 89 documents, of which only 229 entries (118 pages) were graded
  (grader LB). Not 135 pages each. One grader per entry, no double grading.
- Edge-case workbook (15 pages) holds two answer keys only, no grades of LLM output.
- Unit of the 87% is the grade (entry x field), ten fields weighted equally.
- Blank-is-correct effect: when the LLM gave an "actual" latitude it was right 39% (curated, n=89) /
  60% (random, n=84); printed county given: 66% / 29%; township-range given: 56% / 57%.
  Fields nearly always filled (water type, water source name, units, dates) are not affected.
- Inferred latitude/longitude, inferred county, key terms were never graded.

### The graded run is not the published run
- Graded values for the random set are identical to /home/waves/data/usgs_extract/randomSetTest/metadata.csv
  (2025-07-01). Production output metadataResults/metadata.csv is dated 2025-08-23.
- Share of graded entries whose exact value appears on the same page in data/metadata/main_metadata.csv:
  water type 86% (curated) / 97% (random); actual latitude 75 / 69; dates 44 / 46; units 41 / 38;
  water source name 32 / 66; location text 15 / 25. Same number of entries per page on 76% of pages.
  Text differences may be cosmetic; not yet checked semantically.
- 119 of 122 curated and 118 of 123 random test pages are in the published table set.

### Where each published field comes from
- Still Phi-4: water_type (page level; headers split multi-type pages: 97,358 tables single-type,
  1,756 header-resolved, 1,715 first-type fallback), watersource_name, location_text, units, keyterms,
  temporal_resolution (kept raw, superseded by data_resolution).
- Rebuilt by rules (Annette, Oct to Nov 2026): dates, coordinates and precision class, search keys
  (state, county, HUC8/12, B118 basin, NWIS site), record type, derived flag, primary parameter,
  value counts, digitization quality, page type. Phi-4 text is used only as a pointer that must be
  confirmed in the page text.
- Not ML: document fields (USGS Publications Warehouse), table caption (Reducto block).

### Existing validation of the rebuilt fields (all in CLAUDE.md, files under data/analysis/)
- Dates: 30-row targeted review (25/30) and 64 dated rows of a 100-row status-stratified sample
  (46 exact, 52 within a year); graded by Annette; both samples were used to tune the rules.
- Record type: 65 tables, blind, Annette; 59/65 after rules were fixed on the same sample.
- Locations: no human grading; each method scored against printed coordinates or NWIS on rows that
  have both (large n, but only rows that have a printed coordinate).
- Value counts: calendar check on daily grids (99.2%); 16 hand-counted tables.
- Water type header split, parameter (53/65 on the record-type sample), quality flag, page type,
  search keys: no independent validation beyond spot checks.
- Missing on this checkout (git-ignored, on Annette's side): data/catalog/, table_dates.csv,
  entry_dates.csv, table_quality.csv, data/digitized/.

Open: everything about how to evaluate each field (grilling with Anna starts here).

## 2026-11-14 Decisions (grilling)
- Anna: grade BOTH the final (August 2025, production) Phi-4 output and the final published values
  after Annette's rebuild. The paper reports the two side by side.
- Anna: fresh random sample drawn from the published catalog; both versions graded against one answer
  key. Old curated / random / edge-case sets are history only (the recomputed 87% stays as the
  description of the original check).
- PARKED (Anna waiting on Annette): the sampled unit (table vs page vs Phi-4 entry). Claude recommended
  table. Downstream and also waiting: stratification, sample size, how a Phi-4 entry is tied to a table.

## 2026-11-14 Annette's handoff (pulled origin/evaluation, 311f15d; START_HERE.md)
- Annette frames two options: A = evaluate the current per-table catalog on a stratified sample
  (~100 tables by page structure: 1 entry/1 table, 1/many, many/1, many/many; water type, date,
  location); B = re-run Phi-4 with a per-table prompt (200-page pilot first), which would remove the
  entry-to-table join. Her session leaned: do A regardless, pilot B. So Phi-4 access may exist again.
- She names the entry-to-table join (build_table_level_metadata.py) as the never-evaluated weak point:
  23,557 pages have several tables but one entry; ~27,000 entries not tied to a single table.
- Her framing implies the unit is the TABLE (matches Claude's recommendation; Anna still to confirm).
- Her handoff scope is "simplify and evaluate"; Anna's brief to this session is evaluate only.
- Missing on this machine: data/digitized, data/catalog, page_blocks.parquet, table_dates.csv,
  entry_dates.csv, table_quality.csv, tribal_lands/. Raw Reducto inputs are on /home/waves
  (ReductCSVs, ReductJson, UpdatedDataDec/Jan). Annette offers a ~600 MB bundle (no page images);
  tribal files must come from her.
- Today's upstream changes that touch the evaluation: printed coordinates now stored as printed
  (locations.csv changed), new page type report_list, site definitions (open).
- Anna: evaluate first. This session evaluates the final Phi-4 output and the published catalog; the
  per-table Phi-4 re-run pilot is decided later, with this answer key as its baseline.
- Anna: the evaluation is per table (one sampled unit = one published table). Unparks stratification,
  sample size, and how Phi-4 entries are scored against a table.
- Anna: Phi-4 is scored through the pipeline's own tie (the entry build_table_level_metadata.py ties
  to the table, else the page's first entry). Claude recommended the grader picking the entry; not
  chosen. Consequence to state in methods: the Phi-4 per-table score includes errors of the tying step.
- Anna: strata = water type groups (stream discharge, groundwater, reservoir, small types pooled),
  random within group, reweighted to the catalog for the overall figure. Page structure and kind of
  table recorded per sampled table and reported as breakdowns. (Her first reply on this question
  mixed in the digitization session's design and was withdrawn.)
- Anna asked whether Annette already has an evaluation set. Answer given: several small graded sets
  (dates 64 + 30, record type 65, value counts 16, earliest records), none a random sample of published
  tables with a full answer key, most used to tune the rules; no human grading of locations, water
  type per table, units, water source name. Anna: continue with the fresh sample. Existing sets are
  reported as development checks. Not verified: how the 65 record-type tables were drawn.
- Anna: 200 tables: 80 stream discharge, 50 groundwater, 35 reservoir, 35 small types pooled.
  Expected precision at 85% accuracy: headline about +/- 7 points, groundwater about +/- 10.
- Anna: answer key = Claude drafts the true values from the page image without seeing the Phi-4 or
  catalog values; Anna (or someone she names) confirms or corrects every table; a 40-table subset is
  graded independently by a second person for agreement. Second person not yet named.
- Anna: answer key covers seven fields: water type, start/end year, location, kind of record,
  measured quantity (primary parameter), units, water source name. County and watershed codes scored
  automatically from the true point. Not graded: quality flag, value counts, key terms, document
  fields, page type, NWIS station match.
- Anna: location truth = best human point by any means a careful person would use (printed
  coordinates, NWIS station lookup, township/section, well number, maps), with how it was found and
  confidence recorded, or "cannot be located". Scoring by distance: precise within 5 km, approximate
  within 25 km; tables left unlocated that a person could locate count as misses.
  Caveat told to Anna: the grader uses the same reference sources as Annette's rules (NWIS, PLSS,
  GNIS), so truth and method share reference errors; the grader records the source used so agreement
  can be split by whether truth came from the same source as the catalog method.

## 2026-11-15 Location grading, revised in discussion (supersedes the distance-threshold rule above)
- Anna: evaluation asks only whether the pipeline correctly extracted the location the DOCUMENT gives
  for the table. How fine-grained the document's location is belongs to results (usefulness of the
  data), not to the evaluation.
- Truth = the document's own location evidence for the table, turned into a point by a fixed written
  ladder (printed coordinates > station number/name in NWIS > well number / township-section > named
  place > cannot be located), written before any catalog value is seen. No free "best human point".
- Verdicts: correct / less specific than the document allows / wrong (with km) / missed (document
  locates it, catalog blank) / correctly blank. "Same place" decided by a fixed rule per evidence
  type (same station, same section), not by judgement.
- Anna ("ok great"): truth is taken from the whole report; each verdict is also marked evidence on the
  same page vs elsewhere in the report, so both numbers can be given.
- Anna: location method / precision label is not graded separately; checked automatically against the
  evidence type the grader records. County, state, watershed codes, groundwater basin checked
  automatically from the true point. Field plan confirmed ("great good plan"). Fields with no Phi-4
  counterpart (kind of record partly, measured quantity, location method) are reported as "not
  produced" for Phi-4.

## 2026-11-15 Direction change (Anna)
- Multi-site tables: the catalog keeps one point, one water type, one year range per table
  (7,749 published tables have 2+ candidate sites, groundwater 25%; 49,174 tables take their location
  from "best located entry on the page", not an entry tied to the table). Anna: this is not the product
  a scientist needs. Wanted: one row per site and quantity within a table (location, dates, units),
  pointing to document, page, table.
- Anna is open to re-running Phi-4 with a new prompt to get that shape. Original run took about 8 days
  (error logs 2025-08-15 to 08-23, ~11 s/page, sequential requests).
- Anna: before fixing the answer key, scope what can be redone on the engineering side, because the
  product shape decides what is scored. Supersedes "evaluate first, pilot later". Decisions on
  sample (200 tables by water type), grader, fields, location verdicts stand but the key's shape waits.

### Scoping facts for an engineering redo (2026-11-15)
- Production prompt input = all Reducto chunk text of the page, tables as HTML, no table numbers;
  sent one page per request to https://llm.grit.ucsb.edu (model phi4:latest), key from env
  UCSB_LLM_API_KEY. Service answers (401 without key); no key on this machine. Available models unknown.
- Page size is small: 260 local test pages, median ~4,650 chars (~1,300 tokens), max ~16,800 chars
  (~4,800 tokens); 1 table on 63%, 2 on 28%, 3+ on 7%; table rows per page median 30, max 80.
  Numbering the tables in the input is easy (Reducto block type "Table").
- Fields that do NOT need the LLM (already read from the table itself by rules): dates, kind of record,
  measured quantity, value counts, quality flag.
- Fields that depend on the LLM: which sites a table covers and their location words (pointer for the
  location ladder), water type, units, water source name. The entry-to-table tie exists only because
  the old prompt had no table numbers.
- Idea to test: two table layouts. "One site, many times" (daily/monthly grids): LLM returns the site
  once. "Rows are sites" (well inventories, spring lists): LLM should not retype 80 rows; it names the
  columns holding site ID / location / date and code reads the rows.
- Proposed flow (Claude, not yet accepted): scripts keep years / kind of record / quantity / values;
  Phi-4 re-run per numbered table, copy-only (water type, units, layout, site and location evidence);
  script found-on-page check; script reads sites from rows for "each row is a site" tables; existing
  location ladder; join on table number -> one row per site per table. Flowchart with confidence per
  step: https://claude.ai/artifact/E8Hm6kVDvx12tJDV3PrvDJ (source in session scratchpad only).
  Least trusted: Phi-4 layout/column naming (untested), reading sites from rows (low to medium).
- Quick check (scratch script, seed 0, table CSVs read from /home/waves ReductCSVs): do table ROWS
  carry distinct location codes (well number, section id, coordinates, township-range, station no.)?
  Share of tables with 2+ distinct codes in rows: groundwater tables the pipeline counts as multi-site
  11% (n=249); other types multi-site 14% (n=250); groundwater single/no site 7% (n=247); stream
  single/no site 3% (n=250). Regexes are simplified versions of 07_well_numbers.py; names not counted.
- Reading 12 of the multi-site groundwater tables without row codes: 8 are small water-level tables
  (Date | Water level) for ONE well each, several wells per page, the well number and owner printed in
  the heading above each table (some headings swallowed into the table header by Reducto); 3 are
  driller's logs; 1 is a dated depth list. So "n_site_locations >= 2" mostly means several sites on
  the PAGE, one per table. The distinguishing evidence is the heading above the table, not the rows.
  Implication: a script that ties each table to the text block just above it (Reducto reading order;
  table_dates.py already takes captions this way) may do most of the per-table site assignment
  without Phi-4. Not yet measured: how often that heading holds a parsable code (needs Reducto JSONs).
- Check 2 (scratch, seed 1, Reducto JSONs from /home/waves; simplified patterns for station number,
  station name "X River at/near Y", well number, printed coordinates):
  | water type | n | site heading before the table or in its header | any on page | none on page |
  | stream discharge | 300 | 91% | 94% | 8% |
  | groundwater | 297 | 49% | 62% | 39% |
  | reservoir | 200 | 90% | 90% | 15% |
  | small types | 198 | 43% | 49% | 59% |
  Only the 3 blocks directly above the table: stream 34%, groundwater 42%, reservoir 12% (station
  heading sits at the top of the page, remarks in between), so the rule must be "last site heading
  before the table", not "block above". One table on the page: about a third of tables.
  Caveat: presence of a heading is not a verified correct tie; other stations mentioned in remarks.
- Redesign proposed to Anna ("evidence first, model last"): per table, scripts gather evidence by
  closeness (table, heading before it, rest of page, rest of report labeled); existing ladder resolves;
  existing Phi-4 output kept as a free second opinion (water type, site name cross-check); a current
  model only on the residual (no script site, disagreement, unclear multi-site pages), est. ~15% of
  tables; no full re-run. Awaiting her decision.

### Compute and model scoping (2026-11-15)
- Phi-4 is open (MIT); the 2025 run used UCSB's hosted service. This session's node (hpc-12) has
  2 x NVIDIA L40S 46 GB, idle; env `vllm-rs` has vLLM 0.21.0 (installed 2026-05-27), torch 2.11 cuda.
- Storage: home 7 GB free, /tmp 16 GB free -> model weights go to /home/hpc-scratch/annaboser/usgs_models
  (5.7 TB free on the scratch file system).
- Current open models that fit one L40S, verified on the Hugging Face API (name, size, license, date):
  Qwen/Qwen3.6-27B-FP8 30.9 GB apache-2.0 2026-04; Qwen/Qwen3.8-27B-FP8 30.9 GB apache-2.0 2026-08;
  google/gemma-4-31B-it 62.5 GB full precision (needs a 4-bit version) apache-2.0 2026-03;
  numind/NuExtract3 9.3 GB (4B, built for document-to-JSON extraction) apache-2.0 2026-04;
  microsoft/phi-4 29.3 GB (2024-12) for reference. All but Phi-4 also read images.
  No trustworthy public benchmark for this task at this size; blog claims not relied on.
- Timing test in progress: Qwen3.6-27B-FP8, per-table copy-only prompt, 260 old test pages
  (development material, not the future test set). Script in session scratchpad (timing_test.py).
- Proposal version 3 (Claude, awaiting Anna): "two independent readers per table". Reader 1 = scripts
  (evidence nearest first). Reader 2 = a current open model on our own GPUs reading EVERY page,
  per numbered table, copy-only, with a found-on-page check. Compared per table: agree / one reader
  only / disagree (label published). Then the existing ladder. The 2025 Phi-4 output is no longer an
  input, only the evaluation baseline. Known risk: both readers share the Reducto text.
  Model picked by a bake-off on development pages. Drawn at the artifact link (version 3).
- First timing run failed on my script (no __main__ guard for vLLM's spawned engine); fixed, rerun.

## 2026-11-15 Test set drawn; first 20 answer keys drafted
- Anna: likes the two-reader plan; wants to start on the evaluation set now. Test set and development
  material are separate: the 200 test tables are never used to build or tune anything.
- `defaults.py`, `02_draw_test_set.py` -> test_set_tables.csv (200 tables, 137 reports; seed 20261115;
  80 stream / 50 groundwater / 35 reservoir / 35 small types), test_set_frame_counts.csv, pages/ (96 MB).
  Frame: table_level_metadata_published.csv (100,829) minus tables on 508 development pages (old Phi-4
  test pages, Annette's graded samples, earliest-record review, digitization sample) = 100,064.
  Proposals in defaults (not yet put to Anna): one table per page, at most 3 tables per report.
  The list is shuffled, so any prefix is a mix of strata. Weight = stratum tables / stratum draws.
  NOT yet excluded: pages withheld for tribal data sovereignty (list only on Annette's machine);
  they go into EXCLUDED_PAGES and are replaced by the next table of the stratum's random order.
- `KEY_FORMAT.md`: one key row per site per sampled table; evidence copied as printed; point computed
  later by the ladder, not typed.
- Drafts: 4 subagents x 5 tables, blind (scan + brief only; brief shows the first lines of the Reducto
  CSV only to recognise the table). `03_collect_drafts.py` -> key/answer_key.csv (20 samples, 28 site
  rows); never overwrites a sample already in the key.
- `review_server.py` + `review_page.html` (port 8788): scan left, editable site cards right; edits
  saved to key/answer_key.csv, logged in key/edit_log.csv; checked marks in key/checked.csv.
- Format questions raised by the drafters (to settle with Anna, one at a time):
  1. water type when a stream station table reports temperature / sediment / chemistry (Water Quality
     or Stream Discharge?), and inflow to a reservoir, lake samples;
  2. years for a block that is part of a larger printed table (Reducto split), and for capacity /
     rating tables;
  3. kind of record for rating tables (Annette: annual) and capacity tables (no time series);
  4. evidence_kind holds one value but pages print several (coordinates + station number + section);
     and weaker evidence on the page vs stronger evidence elsewhere in the report;
  5. do footer / summary rows and inventory columns (depth, casing) count as quantities;
  6. state / county not printed on the page: search the report or "cannot tell";
  7. n_sites for a table that continues over many pages: this page only (used so far);
  8. non-data tables: water_type "none" or "Not Water Related".
- Anna: remove and redraw any drawn table that is tribal data. `tribal.py` mirrors Annette's rule
  (12_tribal_lands_screen.py: inside a federally recognized area, precise point within its precision
  or 0.5 km, or page text names an Indian Reservation / TIGER rancheria; "possible" pages not withheld).
  Census TIGER 2023 AIANNH downloaded to data/analysis/evaluation/metadata/tribal/ (folder git-ignored;
  page lists are sensitive, never commit or share). Spatial rule gives 1,894 pages over the whole
  corpus (Annette reports 2,038 withheld incl. text-named: consistent). Text rule is applied only to
  drawn candidates (needs the page's Reducto JSON).
  `02_draw_test_set.py` now replaces a withheld pick in place by the next table of its stratum, so
  other sample numbers do not change. Result: 3 replaced (sample numbers 57, 111, 199); the 20 drafted
  tables are untouched. Stratum weights still count tribal tables in the frame (about 3%); to refine
  with Annette's full list.
  Limit: the rule only catches what it describes; the digitization session met a page of U.S. Indian
  Service wells that names no reservation. Anna flags such pages while checking -> EXCLUDED_PAGES.
- Timing test result (Qwen3.6-27B-FP8, one L40S, vLLM 0.21, 256 old test pages, per-table copy-only
  prompt, thinking off): 1.36 s/page (348 s total after a 218 s model load), 3,246 input and 176 output
  tokens per page, 256/256 answers valid JSON with the right number of tables.
  Full corpus estimate: 64,074 pages x 1.36 s = about 24 GPU-hours (12 h on both cards) vs 8 days in 2025.
  Settings were conservative: max 48 pages at once, slower sampler (no CUDA compiler on the node for
  the fast one: VLLM_USE_FLASHINFER_SAMPLER=0), text-only (limit_mm_per_prompt image 0), 12k context.
  Two failed starts first: missing __main__ guard (mine); out of memory at the default batch settings.
- Anna: answer key keeps the full truth from the report (e.g. sample 1, title and wells split across
  facing pages 21/22), each part marked with its page. Anna's idea: a table that says "continued"
  may look at the pages before it.
  Check 3 (scratch, same samples as check 2): of tables with NO site heading before them, share that
  say "continued" / have the previous page scanned / both: stream 57% / 89% / 50% (n=28);
  groundwater 49% / 76% / 43% (n=148); reservoir 45% / 65% / 35% (n=20); small types 28% / 60% / 22%
  (n=111). So a continued rule could reach up to about half of the no-site tables.
  Annette's earlier continuation tier (08_continuation.py, page level): loose inheritance failed its
  hold-out (45-62% within 2 km); strict version kept (full well numbers 20/20). What may be inherited
  must depend on the layout: site for one-site tables; title and headers only for rows-are-sites.
- Anna (2026-11-15): do NOT change much from Annette's rules; she did a lot of validated work; move fast,
  no mistakes. Change only where something is really missing. Consequence: no "continued" rule, no new
  units script, no layout-specific rules now; the redesign (versions 2-3 of the flowchart) is parked.
  Claude's view given to her: the one thing really missing is a check (and if needed a fix) of the
  step that ties Phi-4 entries to tables; the evaluation itself will show whether it matters.
- CORRECTION (Anna, same day): the "don't change much from Annette's rules" steer applies ONLY to looking
  at previous pages. So: keep Annette's cross-page rules exactly as they are (no new "continued" rule).
  The two-reader plan and the one-row-per-location table are NOT parked; the entry above over-read her.

## 2026-11-15 Target product agreed by Anna and Annette (supersedes earlier shapes)
- Each page is processed alone; the process never looks at other pages.
- Final table: one row per location per table. Keys: document, page, table, location id.
  Graded: location (best available from the page), location type, water source, water type, start
  date, end date (granularity as available), temporal resolution, measurement type(s), units, number
  of measurements, date of construction (wells only).
- On the way: readers (scripts + open model) extract the same fields EXCEPT location / location type,
  plus all location evidence on the page (gauge/well id, water source, city, county, state, landmark,
  township-range-section, lat/lon ...) and reach a consensus. Annette's hierarchy then picks the best
  evidence -> location + location type. Our work ends there; matching to points/basins is Annette's.
- Claude's proposal (artifact version 5, same link): column names doc_id, page_number, table_number,
  location_number, location, location_type, water_source, water_type, date_start, date_end,
  temporal_resolution, measurement_types, units, n_measurements, construction_date; evidence columns
  coordinates, well_number, station_number, station_name, township_range_section, place_name,
  place_relation, basin_or_area, county, state (+ evidence_place, agreement).
  Category sources checked in the data: locations.csv coord_method (order from 04_build_locations:
  GNIS < PLSS < NWIS name < NWIS number < well no. < printed), 9 water types, table_resolution.csv
  classes (daily 44,734; monthly 19,972; discrete 18,404; annual 11,321; unclassified 8,522;
  no_time_series 3,238) and parameters (primary "unknown" for 11,491 tables), 6,988 distinct unit
  spellings in the Phi-4 output.
- Open decisions put to Anna/Annette: date format (ISO reduced precision recommended); meaning of water
  type now that measurement types exist; location type list and order; extra measurement types; well
  inventory attributes not measurements; answer key becomes page-only (drafts 1, 9, 15, 19 used a
  neighbouring page); non-data tables = one row, Not Water Related.
- Anna (2026-11-15) on the version 5 schema: date format YES (1916 / 1916-10 / 1916-10-01); answer key
  page-only YES; tables with no water data = one row, Not Water Related, YES. Water type: keep
  Annette's original categories; maybe call the column "data type"; unsure about site type vs
  measurement type; wants casing/depth kept in the measurement column (fold in or rename). Asked
  Claude to work out a better design.
- Sleuthing (table_resolution.csv, published tables, Annette's water type x her primary quantity):
  Reservoir 78% contents; Groundwater 46% water level, 10% chemistry, 36% unknown; Stream Discharge
  69% discharge, 8% gage height, 7% chemistry, 5% sediment, 5% temperature; Springs 41% discharge;
  Irrigation 48% discharge; Precipitation 61% precipitation; Water Quality (476 tables) 68% chemistry.
  So the nine names already behave as "kind of site" for all but Water Quality.
  Header words of the 11,491 tables with unknown quantity: stream (5,016): "run-off" in acre-feet
  (~1,800), sediment size; groundwater (3,097): driller's logs (sand / clay / gravel / thickness,
  ~800) and well inventories (depth, use, method of lift); springs: use descriptions; reservoir:
  elevation; irrigation: irrigated acres, use.
- Claude's suggestion: (1) `data_type` = Annette's nine, unchanged; no separate site type column
  (it would repeat data_type on ~97% of rows); (2) rename measurement_types -> `variables`, one list
  holding measured quantities (her nine + runoff, elevation, well_yield) and site descriptions
  (well_construction, lithology, water_use, irrigated_area), + other; n_measurements counts measured
  values only. Awaiting Anna.
- Anna (2026-11-15): YES `data_type` (Annette's nine, unchanged; site-first rule for clashes, nothing
  changes in her figures); YES `variables`; wants both `n_measurements` and `n_variables`
  (= n_measurements + the other numbers recorded). Claude's naming concern, raised once: "n_variables"
  can be read as "number of kinds of variable"; `n_values` (the catalog's existing name) would avoid
  that. Her name is used until she says otherwise.
- Implemented: `schema.py` (columns, vocabularies, hierarchy, best_location) is the single source;
  KEY_FORMAT.md rewritten (page-only; blank = not on page; evidence columns filled, location and
  location_type derived); `03_collect_drafts.py` (briefs / collect); review server and page moved to
  the new columns (location and location_type shown read-only and recomputed on every save).
  Old-format drafts and key moved to key/old_format_2026-11-15/ (not deleted).
  First 20 tables being redrafted by 4 subagents, page-only. Artifact version 6 published.
- First 20 tables redrafted in the new shape, page-only (4 subagents), collected: 28 location rows.
  Location type by table: coordinates 12, station number 2, station name 2, well number 2, county 1,
  none 1. Data type: Stream Discharge 10, Reservoir 5, Groundwater 3, Springs 1, Not Water Related 1.
  Drafter unsure on 12 of 20 tables, mostly because of the counts. Review server restarted (8788).
- Rules the drafters could not settle (to put to Anna one at a time):
  COUNTS: (a) one estimated value bracketing several days: 1 or one per day (sample 16: 264 vs 365);
  (b) whole months printed as no flow (4: 212 vs 365); "No flow" / "Dry" entries (11, 15);
  (c) a second variable printed in the foot rows of a daily table, e.g. end-of-month elevation
  (8: 377 vs 365; 20: 365 vs 389); (d) monthly max / min / mean in a monthly summary table (13);
  (e) rating and capacity table points (6, 12); (f) one value printed in two units (11);
  (g) a table Reducto split into blocks or printed in stacked blocks: which block is "the table"
  (3: 143 vs 94; 18: 55 vs 21; 9: years of the block vs of the printed table).
  DATES: extremes table: dates of the extremes or the water years covered (2).
  LOCATION: two places in one station name ("at X, near Y"); land grants ("in Yokaya Grant") have no
  field; a report's own station number vs a USGS number (11); a table-title name used as station
  name when there is no formal station (9); several place relations on one page (12).
  OTHER: units for chemistry tables (many units); continuation pages with no column headings (15);
  inflow to a reservoir and lake samples: Stream Discharge or Reservoir (9, 14, 18).
- Anna (2026-11-15): drop `units` ("not interesting if we know the variables"). Removed from schema.py,
  KEY_FORMAT.md, the review page, the key file and the artifact. The drafts still hold the units the
  drafters read (ignored by the collector). Fact told to her once: one variable does come in different
  units in these reports (contents in acre-feet or thousands of acre-feet; water level as depth below
  land surface or as altitude), so a user must open the table to know which.

## 2026-11-15 Anna: build both sets and the pipeline without her checking; come back with validation numbers
- Anna looked over the first 20 redrafted tables and found no errors. She asked for: full validation and
  test keys without her checking each table, pipeline (scripts + open model) engineered and tuned on
  validation, then numbers; test set scored only after she says go. Agreed: status at each milestone,
  ask before any GPU run over an hour. Claude told her the key is then machine-made and suggested a
  human check of 40 random test tables for the paper (not blocking).
- Sets: `defaults.SETS`. Test = 200 tables (unchanged by the refactor, verified byte-identical).
  Validation = 100 tables (40 stream / 25 groundwater / 18 reservoir / 17 small types; seed SEED+1;
  excludes every test page and development page; 1 replaced after the tribal screen). Scripts take the
  set name: 02_draw_test_set.py, 03_collect_drafts.py (briefs / collect), review_server.py.
- Counting and special-case rules written into KEY_FORMAT.md by Claude (two blocks, 2026-11-15), from the
  cases drafters raised; principle "count what is printed". Five cells of the first 20 test tables were
  edited through the review server to match the rules (20: n_measurements 389 and variables; 11: 11;
  9: Reservoir; 15: construction date blank) -> they appear in key/edit_log.csv though made by Claude.
  Validation sample 69 n_measurements 60 -> 36 (monthly summary rule).
- Drafting: DRAFTER_INSTRUCTIONS.md; subagents of 5 tables each, page-only, blind. Validation key 100
  tables / 144 rows (24 tables marked unsure). Test key 200 tables. Briefs from 121 on also say how many
  columns the digitized table has (earlier drafters sometimes thought columns were missing).
  Claude does not open the test key or its scores while tuning. (Drafters' short reports on test
  tables did pass through this session: rule gaps only.)
- Recurring rule gaps still open after the clarifications: computed-only / modelled tables (recharge
  estimates, model nodes, projections); depth to water AND altitude of water table printed for the same
  reading (counted once by most); driller's logs (0 measurements); "(a)" footnote cells; daily grid
  plus peak block in one digitized table; report-own station numbers printed like USGS numbers
  ("308.", "1882."); 15-digit site numbers; rain gauges and weather tables.
- Code: page_input.py, prompts.py (v1), 05_model_reader.py (vLLM, stdlib only: env vllm-rs has no
  pandas), scoring.py, predictions.py (current_rows = Annette's per-table files in the new columns;
  model_rows), 04_score.py.
- Baseline "current" on the first 85 validation tables (before any new pipeline): data type found 95%,
  temporal resolution 91%, end year 96%, start year 73%, location type 73%, water source 88%,
  variables exact 59% (any overlap 89%), n_measurements 59%, n_variables 61%.

## 2026-11-15 Pipeline on the validation set (100 tables, 144 location rows)
Scores = share of tables where the page gives the value and the output has it right ("found"),
weighted to the published tables. Files: data/analysis/evaluation/metadata/scores/validation_*.
| column | current files | model v1 | model v2 | pipeline v2 + tidy |
| n_locations | 0.95 | 0.94 | 0.97 | 0.97 |
| location | (not stored) | 0.90 | 0.92 | 0.94 |
| location_type | 0.73 | 0.93 | 0.98 | 0.97 |
| water_source | 0.86 | 0.90 | 0.93 | 0.93 |
| data_type | 0.93 | 0.78 | 0.95 | 0.96 |
| date_start (as fine as the key) | year only | 0.74 | 0.83 | 0.83 |
| date_end | year only | 0.70 | 0.85 | 0.85 |
| start year | 0.70 | 0.81 | 0.84 | 0.84 |
| end year | 0.93 | 0.95 | 0.98 | 0.98 |
| temporal_resolution | 0.87 | 0.85 | 0.90 | 0.90 |
| variables (exact set) | 0.56 | 0.83 | 0.86 | 0.93 |
| n_measurements (within 5%) | 0.56 | none | none | 0.56 (current counts) |
| n_variables | 0.59 | none | none | 0.59 |
- v1 -> v2 prompt changes (from the v1 errors): data type follows the site, not the measurement (v1 called
  chemistry at stream gauges Water Quality and canals Stream Discharge); water-year dating; dates of
  the block; fields left out when blank (6 of 100 v1 answers were cut off at 3,000 tokens; v2: 1 at 5,000).
  Speed: v1 4.1 s/page, v2 4.6 s/page (richer answers than the 1.4 s/page timing prompt)
  -> full corpus about 80 GPU-hours on one card at this prompt, 40 on two.
- Readers compared (model v2 vs current): where both give a value and agree, they are right 92-99%
  (data type 0.99, n=85; resolution 0.95, n=84; start year 0.92, n=64; end year 0.99, n=72). Where they
  differ, the model is right 75-100% and the current files 0-25% (13, 4, 13, 5 rows). So RECONCILE is
  now model-first for every column except the counts, and agreement is worth publishing as a label.
- tidy() rules in pipeline.py (each from validation errors): two-digit years -> 19xx; capacity tables
  (no_time_series with only contents / elevation / gage height) get no dates; "runoff" dropped beside
  "discharge" in daily tables; generic water source words blanked; coordinates need digits.
- Scorer: coordinates compared by digits; gage_height and elevation count as one variable (stage);
  dates agree when one is a prefix of the other at month or day level.
- Counts: Annette's n_values match the key on 56% of tables (daily 78%, monthly 44%, discrete 40%,
  annual 43%, capacity tables 0%). Mostly definition: her count includes monthly runoff columns
  (60 vs the key's 36) and drops the first column of capacity tables. A counter written to the key's
  rules from the Reducto CSV (counts.py, deleted) did worse (28%; daily grids undercounted) -> abandoned.
  Open decision for Anna.
- Remaining errors are concentrated in a few hard tables: multi-location pages without headings or
  rotated (validation 4, 11, 36, 52, 55, 72), Reducto misreads copied faithfully ("165/43E" for
  "16S/43E"), tables with water-year rows (start year), inventories with dated measurements.
- v3 prompt (running): water-year rows rule; water source for wells never generic; answer limit 8,000.
- Pipeline v3 on validation (prompt v3, model-first reconcile, tidy rules; 5.3 s/page, 100/100 answers
  parsed): n_locations 0.97, location 0.94, location_type 0.98, water_source 0.93, data_type 0.99,
  date_start 0.87, date_end 0.83, start year 0.93, end year 0.98, temporal_resolution 0.90,
  variables 0.92, n_measurements 0.56, n_variables 0.59.
  By group (location / data type / start year / variables): stream 0.96 / 1.00 / 0.92 / 0.95;
  reservoir 0.94 / 1.00 / 0.89 / 0.94; groundwater 0.70 / 0.93 / 0.94 / 0.68; small types 0.87 / 1.00 /
  0.99 / 0.77. Groundwater water_source 0.62 (what counts as the source of a well is ill-defined).
  These are tuned-on numbers (three prompt rounds against this key); the test set is untouched.
  Full corpus at 5.3 s/page: about 94 GPU-hours on one card, 47 on two -> batch job, needs Anna's OK.
- Anna (2026-11-15): "do we need data type from the Phi-4 run? it complicates the story." Tested on
  validation (v3): as built (model, else Annette's files), data type from the model only, and every
  column from the model only give identical scores on every column (data type 99, resolution 90, dates
  87 / 83, years 93 / 98, variables 92, water source 93): the model left no data type blank (0 of 292
  rows) and answered every table. RECONCILE is now "model" for every column; only the two counts come
  from Annette's value-count script (no Phi-4). Her per-table files stay as an outside agreement check,
  not as an input. Artifact version 9 redrawn accordingly.
- Anna: "what do the scripts really add?" Ablation on validation (v3 answers; pipeline.py with each
  step switched off; defaults.TIDY_RULES, REQUIRE_ON_PAGE):
  | step | effect on validation scores |
  | fallback to Annette's per-table files | none (model blank on 0 of 292 rows) |
  | found-on-page check | none; removed 5 of 1,232 location values (state "Calif." x3, county "Orange", place "Troy") |
  | tidy: two-digit years, capacity tables undated, generic water source, coordinates need digits | none each (prompt v3 already avoids these; they fixed v1/v2 errors) |
  | tidy: drop "runoff" beside "discharge" in daily tables | variables 83.4 -> 92.3 (6 rows) |
  | Annette's value counts | the only source of n_measurements / n_variables (56% / 59% vs key) |
  | table numbering in the input, location hierarchy | structural: cannot be switched off |
  Also: 208 of 292 rows carry 3 or more kinds of location evidence, so the hierarchy does choose.
  Proposal to Anna: keep numbering, hierarchy, the on-page check (as a guarantee, with its 0.4% removal
  rate reported), the runoff rule (or fold into the prompt), and the counts; drop the four no-op rules
  after the second validation set confirms they do nothing there either. Not yet tested: asking the
  model for the counts.

## 2026-11-15 Second validation set (Anna: "add another 100 tables and see how it does")
- Set `validation2` (defaults.SETS): 100 tables, seed SEED+2, no page shared with test or validation;
  5 replaced after the tribal screen; key 100 tables / 152 rows. Scored with the pipeline exactly as it
  stood (prompt v3, every column from the model, counts from Annette's script). Model 4.2 s/page.
  | column | validation (tuned on) | validation2 (unseen) | validation2, Annette's current files |
  | n_locations | 97 | 98 | 93 |
  | location | 94 | 85 | not stored |
  | location_type | 98 | 90 | 56 |
  | water_source | 93 | 95 | 89 |
  | data_type | 99 | 96 | 90 |
  | date_start / date_end | 87 / 83 | 86 / 86 | years only |
  | start year / end year | 93 / 98 | 91 / 94 | 80 / 86 |
  | temporal_resolution | 90 | 94 | 84 |
  | variables (exact) | 92 | 80 | 68 |
  | n_measurements / n_variables | 56 / 59 | 65 / 56 | 66 / 56 |
  By group on validation2 (location / data type / variables): stream 87 / 98 / 82; reservoir 100 / 94 /
  93; groundwater 69 / 99 / 64; small types 54 / 59 / 57.
- Why location is lower: of the 5 stream tables wrong, 3 are Reducto misreads copied faithfully
  ("long. 100°09'" for 120°09'; "39032'" for 39°32'; "DAESF." for "Calif.") and 2 are a printed
  station number the model did not return (it gave the station name). With 40 stream tables carrying
  84% of the weight, 3 or 4 tables move the headline by 8 points: sampling noise is as large as any
  tuning effect. Variables: 4 of the 7 stream misses are "discharge;runoff" on MONTHLY tables (the
  runoff rule only covered daily tables).
- Ablations repeated on validation2: on-page check and the four dead tidy rules change nothing; the
  runoff rule again carries variables (80.1 with, 69.2 without).
- validation2 has now been looked at, so it is tuning material from here on.
- Anna: fine to keep the extra tidy rules if they help at all. Checked how often each fires on the v3
  answers of both validation sets (519 rows): two-digit years 0, capacity tables undated 0, coordinates
  need digits 0, generic water source 12 (all in validation2; turns a wrong "Groundwater"-style source
  into a blank; no verdict becomes right, none becomes wrong). Kept: runoff beside discharge (now daily
  and monthly) and generic water source. Dropped: the three that never fire.
- Counting redesign (Anna: think about what a script should really do): counts.py reads Reducto's
  original HTML table (not the CSV), and the model (prompt v4) says which columns are measured, which
  rows are summaries / separately measured foot rows, and whether rows are locations. Without hints,
  on both validation sets (155 single-location tables): n_measurements 0.56, n_variables 0.65; daily
  0.85 / 0.87 (Annette's counts: 0.80 / 0.67) after two fixes: HTML source, and a lone letter o = zero.
  Monthly 0.39, annual 0.38, capacity tables 0.08 without hints. v4 runs in progress.

## 2026-11-15 (late): prompt v4 scored on both validation sets; tuning stopped
- v4 answers: 200 of 200 pages well formed, none truncated; 5.6 and 4.2 s per page.
- v3 was rescored with today's code so the comparison is like for like (the earlier v3 table was
  scored with the older rule set; e.g. variables 92 / 80 then, 81 / 89 now). Found %, weighted:
  | column | v3 set1 | v3 set2 | v3 both | v4 set1 | v4 set2 | v4 both | Annette's files, both |
  |---|---|---|---|---|---|---|---|
  | n_locations | 97 | 98 | 98 | 99 | 98 | 99 | 94 |
  | location | 94 | 85 | 90 | 93 | 89 | 91 | not stored |
  | location_type | 98 | 90 | 94 | 95 | 92 | 94 | 65 |
  | water_source | 93 | 95 | 94 | 91 | 93 | 92 | 88 |
  | data_type | 99 | 96 | 98 | 97 | 95 | 96 | 92 |
  | date_start / date_end | 87 / 83 | 86 / 86 | 87 / 84 | 87 / 83 | 84 / 84 | 86 / 83 | years only |
  | start year / end year | 93 / 98 | 91 / 94 | 92 / 96 | 90 / 98 | 88 / 95 | 89 / 96 | 75 / 89 |
  | temporal_resolution | 90 | 94 | 92 | 90 | 95 | 93 | 86 |
  | variables | 81 | 89 | 85 | 84 | 91 | 87 | 62 |
  | n_measurements | 53 | 61 | 57 | 54 | 73 | 63 | 61 |
  | n_variables | 62 | 70 | 66 | 62 | 70 | 66 | 58 |
- v4 against v3: counts +6 (n_measurements), location +1, variables +2; start year -3 (three stream
  tables where v4 gave the water year instead of the block's own dates: validation 22, validation2 15
  and 99), data type -2, water source -2. Apart from counts every difference is one to three stream
  tables, i.e. inside the noise. v4 kept; no further tuning.
- KEY INCONSISTENCY found: 8 tables of the FIRST validation key list "discharge;runoff" for monthly
  summary tables (samples 14, 15, 21, 37, 69, 77, 80 and one more). KEY_FORMAT.md ("runoff in
  acre-feet ... summary values"; "a variable that appears only in summary rows is not listed") was
  written after that key was drafted; validation2 follows the rule. With those 8 scored as right,
  variables = 97 / 91 / 94 (set1 / set2 / both). The key has NOT been changed; Anna to decide. The
  test key was drafted before the clarification too and probably has the same pattern (not opened).
- Tidy rules on v4 answers (553 rows): runoff beside discharge fires 2 times (the v4 model leaves
  runoff out by itself), generic water source 16 times. Both kept; neither matters much any more.
- Counts by kind of table, v4, both sets (n_measurements / n_variables found): daily 93 / 84 (49
  tables; Annette's counts 83 / 70); monthly 25 / 88 (33); annual 33 / 26 (16); discrete 53 / 41
  (40); no time series 6 / 31. Rows within 10% of the key: 69% / 68% (Annette's: 65% / 72%). Summed
  over tables, the pipeline total is 0.95 / 1.01 of the key total (Annette's 0.94 / 0.95).
  Monthly n_measurements misses are one pattern: key 36 (max, min, mean x 12), pipeline 48 (it also
  counts the second-foot-days or runoff column as measured); a script-side exclusion of those
  columns would fix about 8 stream tables without rerunning the model. Not done (tuning stopped).
- Other things seen, not fixed: two-digit years returned as "47-05-19" (validation 57); owner names
  with reversed word order on rotated tables (validation 36, validation2 92: Reducto's text order);
  v4 leaves the owner out on some well-inventory rows where v3 gave it (validation2 6, 7, 54, 63).

## 2026-11-15 (late): counts taken out of the pipeline (Anna)
- Anna: the number of measurements is Annette's postprocessing, not part of the metadata pipeline.
- Done: n_measurements and n_variables removed from schema.GRADED (the answer keys keep the columns,
  schema.COUNTS_IN_KEY, ungraded); counts.py deleted; counts branch removed from pipeline.py, scoring.py
  and defaults.RECONCILE. The final table has 12 columns.
- Prompt v5 = v4 minus the counting hints (measured_columns, summary_rows, measured_rows,
  rows_are_locations). A changed prompt changes the other answers a little, so both validation sets
  are rerun with v5; PROMPT_VERSION = "v5".
- v5 scored (found %, set1 / set2 / both): n_locations 99 / 98 / 99; location 93 / 89 / 91;
  location_type 95 / 94 / 95; water_source 90 / 95 / 93; data_type 94 / 96 / 95; date_start 87 / 86 / 87;
  date_end 81 / 84 / 82; start year 91 / 89 / 90; end year 98 / 92 / 95; temporal_resolution 90 / 94 / 92;
  variables 81 / 89 / 85 (94 / 89 / 92 if the 8 runoff tables of the first key follow the written rule).
  Annette's files, both sets: location_type 65, water_source 88, data_type 91, years 75 / 89, resolution
  86, variables 64. By group (location / data type / variables): stream 94 / 96 / 87; reservoir 97 /
  100 / 91; groundwater 68 / 92 / 66; small types 71 / 79 / 64.
- Tidy rules on v5 (500 rows): runoff beside discharge 21, generic water source 14 (v5 lists runoff
  again, unlike v4, so the rule matters again).
- Parser bug fixed (not tuning): 2 of 200 v5 answers failed to parse because a copied coordinate's
  seconds mark (38°49'26", long ...) is an unescaped quote. 05_model_reader.parse now retries with
  inner quotes escaped; stored answers re-parsed (v5: 200 of 200). Matters for the full run, where
  coordinates with seconds are common.

## 2026-11-15 (night): pipeline locked, full run started (Anna: "lock in the pipeline and start ... as fast as you can")
- Locked: prompt v5, Qwen3.6-27B-FP8, found-on-page check, two tidy rules, hierarchy. pipeline.py is now one
  function per table (table_rows) used by the evaluation sets and the full run; the reconcile options and
  the fallback to Annette's files are gone (defaults.RECONCILE removed). Validation output byte-identical
  before and after the refactor (271 and 229 rows).
- reader.py holds the model code (load, read, parse); 05_model_reader.py and 06_full_run.py use it.
- Runoff rule applied to the keys by 08_key_runoff_rule.py (Anna's go read from "ok good let's go then";
  backups answer_key_before_runoff_rule.csv): test 1 row, validation 8 rows, validation2 0. Variables on
  validation now 94 / 89 (set1 / set2).
- Full run: 63,679 pages hold a published table (table_level_metadata_published.csv); 63,524 have a table
  in their Reducto page file, 155 do not (full_run/pages_without_table.csv). 318 shards of 200 pages,
  page texts stored in full_run/in (72 MB) so workers never read the data share.
  06_full_run.py worker = shared work list (claim by mkdir, stale after 90 min, out file renamed into
  place). Two workers started in the session on hpc-12 (they die with the session; their shards are
  taken over). Batch job data/analysis/evaluation/metadata/jobs/usgs_metadata_full.sh (array 0-15, one
  L40S each, excludes hpc-10/13/15) staged in composer project 57 (a spare default job of 2026-11-10);
  Anna submits. Estimate: 63,524 pages x about 5 s = about 88 GPU-hours; 18 GPUs -> about 5 h.
- 07_build_metadata.py builds full_run/metadata_per_location.csv. It includes pages on tribal lands:
  Annette's screen comes after; the file is not to be shared beyond her.
- Test set: to be scored once from the full-run answers of its 200 pages (same prompt), with --final.
