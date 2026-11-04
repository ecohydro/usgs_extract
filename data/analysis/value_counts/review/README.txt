Value-count hand check (blind). One row per table; pngs/ holds the page image (file prefix = review_no).
If the page has several tables, table_number says which one (table_csv shows its digitized text).

Count in grader_true_values: every printed measurement value in that table.
  Count: discharge, gage height, water level / depth to water, storage, contents, chemistry values,
         precipitation, irrigated amounts, estimated values (e, *, footnoted), zeros.
  Do NOT count: dates, years, day numbers, station/well numbers, owner names, well depth or other
         site description, totals, means, maxima/minima, acre-foot summary lines.
  Blank or dash = no value. Note anything ambiguous in grader_notes.
Row 4 (doc 70038392 p34) has no page image anywhere: skip it.
The script's counts are in value_count_review_key.csv; join after grading.
