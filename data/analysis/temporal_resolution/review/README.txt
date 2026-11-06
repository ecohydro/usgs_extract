Temporal resolution grading (blind)

For each row, open the png and find the table (table_number = its order on the page, top to
bottom; the caption is the text just above it). Fill in:

true_resolution  what ONE table's own observations are:
  daily           one value per day (day 1-31 rows x months, or a day-by-day list)
  monthly         one value per month (month rows, or year rows x month columns)
  annual          one value per year (annual peaks, yearly totals or means)
  discrete        individual measurements on given dates (field discharge measurements,
                  water levels on dates, water-quality samples, peaks above a base)
  no_time_series  no time dimension (well log, well inventory, rating/capacity table,
                  undated analyses)
  unsure / not a table   if the table can't be read or isn't on the page
derived_summary  y if the table is a summary computed from another table on this or the
                 adjacent page (e.g. "Monthly discharge" printed with the daily table), else n
true_parameter   main quantity in the table: discharge, gage_height, contents, water_level,
                 temperature, sediment, water_chemistry, precipitation, evaporation, other
note             anything useful (e.g. "daily but water temperature, not discharge")

Summary rows inside a table (Mean / Total / Max / Min) don't change the class: a daily table
with monthly means at the bottom is daily.
