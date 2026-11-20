# Instructions for reading metadata from page scans (03_fill_from_scans.py)

You record the metadata of tables the language-model run could not give. You are given a list of numbers N.
For each N you read ONE page scan (a PNG image; use the Read tool on it) and write down what that page says
about one table, one record per location the table reports data for. The rules are those of the test-set
answer key.

1. Read the format and ALL rules, including the special-case rules at the end, and follow them exactly:
   `/home/annaboser/usgs_extract/code/01_digitization/04_metadata_extraction/evaluation/KEY_FORMAT.md`
2. For each N read its brief: `/home/annaboser/usgs_extract/data/analysis/evaluation/metadata/scan_fill/briefs/N.txt`.
   It gives the page scan path, which table on the page it is (tables numbered top to bottom), the first and last
   lines of the digitizer's version of that table (only to recognise which part of the page is the table; it may
   be empty or garbled) and the output path.

Strict rules
- THIS PAGE ONLY. Open only the page scan named in each brief. Never open another page of the report. If the page
  does not give something, leave the field blank (empty string).
- Do not open any other file under `/home/annaboser/usgs_extract/data` or `/home/waves` (no metadata, no
  analysis files, no other drafts).
- Copy names, numbers and location words exactly as printed. No conversion, no lookup, no inference from general
  knowledge.
- Fill EVERY kind of location evidence the page prints for the location, each in its own field. Do not fill
  `location` or `location_type`.
- Dates as fine as the page gives: "1916", "1916-10" or "1916-10-01"; water year "1917" = "1916-10" to "1917-09".
- Leave `n_measurements` and `n_variables` blank (not needed here).
- EVERY LOCATION. Unlike the answer key, give a full row for every location the table reports, in printed
  order, even when there are more than 5 (KEY_FORMAT's "first, last and three in between" rule does NOT apply).
- If the scan is rotated or small you may crop, enlarge or rotate a copy with Python (PIL is in
  `/home/annaboser/miniconda3/envs/smh_irr_labels_2/bin/python`), saving only into
  `/tmp/claude-15187/-home-annaboser-usgs-extract/68d58847-6d4b-437f-84fa-115b66c21490/scratchpad/`.

Output
For each N write a JSON file at the path given in its brief: a JSON list with one object per location, with
exactly these fields: location_number, n_locations_in_table, coordinates, well_number, station_number,
station_name, township_range_section, place_name, place_relation, basin_or_area, county, state, evidence_place,
water_source, data_type, date_start, date_end, temporal_resolution, variables, n_measurements, n_variables,
construction_date, confidence, notes. Blank = empty string. Check that each file
is valid JSON.

Final report: at most one short line per N, only for things that were hard or where the rules did not cover the
case. If nothing was hard, say "no issues".
