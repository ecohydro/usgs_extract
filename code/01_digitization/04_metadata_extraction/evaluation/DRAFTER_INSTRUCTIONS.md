# Instructions for drafting answer keys

You are drafting an answer key for an evaluation of metadata extracted from scanned historical USGS
water reports. You are given a set name (`test`, `validation` or `validation2`) and a list of sample numbers. For
each sample you read ONE page scan (a PNG image; use the Read tool on it) and write down what that
page says about one table, one record per location the table reports data for.

1. Read the format and ALL rules, including the special-case rules at the end, and follow them
   exactly: `/home/annaboser/usgs_extract/code/01_digitization/04_metadata_extraction/evaluation/KEY_FORMAT.md`
2. For each sample number N read its brief:
   - set `validation`: `/home/annaboser/usgs_extract/data/analysis/evaluation/metadata/key_validation/briefs/N.txt`
   - set `validation2`: `/home/annaboser/usgs_extract/data/analysis/evaluation/metadata/key_validation2/briefs/N.txt`
   - set `test`: `/home/annaboser/usgs_extract/data/analysis/evaluation/metadata/key/briefs/N.txt`
   The brief gives the page scan path, which table on the page is the sampled one, the first and
   last lines of the digitizer's version of that table (only to recognise which part of the page is
   the sampled table), and the output path.

Strict rules
- THIS PAGE ONLY. Open only the one page scan named in each brief. Never open another page of the
  report. If the page does not give something, leave the field blank (empty string) and say so in notes.
- BLIND. Do not open any other file under `/home/annaboser/usgs_extract/data` (no metadata, no
  analysis files, no other key or draft files), and nothing under `/home/waves`.
- Copy names, numbers and location words exactly as printed. No conversion, no lookup, no inference
  from general knowledge.
- Fill EVERY kind of location evidence the page prints for the location, each in its own field. Do
  not fill `location` or `location_type`.
- Dates as fine as the page gives: "1916", "1916-10" or "1916-10-01"; water year "1917" = "1916-10"
  to "1917-09".
- Counts: follow the counting rules in KEY_FORMAT.md (count what is printed). Count carefully, month
  by month or column by column; say in notes how you counted; set confidence to unsure if the count
  may be off by more than a few.
- If the scan is rotated or small you may crop, enlarge or rotate a copy with Python (PIL is in
  `/home/annaboser/miniconda3/envs/smh_irr_labels_2/bin/python`), saving only into
  `/tmp/claude-15187/-home-annaboser-usgs-extract/733d78d3-539f-4610-a93c-bf9cab9ea263/scratchpad/`.

Output
For each sample write a JSON file at the path given in its brief: a JSON list with one object per
location, with exactly these fields: location_number, n_locations_in_table, coordinates, well_number,
station_number, station_name, township_range_section, place_name, place_relation, basin_or_area,
county, state, evidence_place, water_source, data_type, date_start, date_end, temporal_resolution,
variables, n_measurements, n_variables, construction_date, confidence, notes. Blank = empty string.
Check that each file is valid JSON.

Final report: at most one short line per sample, only for things that were hard or where the rules
did not cover the case. If nothing was hard, say "no issues".
