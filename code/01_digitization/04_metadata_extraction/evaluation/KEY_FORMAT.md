# Answer key format (metadata test set)

The key has the same shape as the final metadata table (`schema.py`): **one row per location per
sampled table**. It records what the sampled page says. Agreed by Anna and Annette, 2026-11-15.

## Rules for the reader

1. **This page only.** Work from the scan `pages/{doc}_page_{N}.png`. Never open another page of the
   report, and never use general knowledge to fill a gap. If the page does not give something, the
   field stays blank.
2. The key is written from the scan, never from the pipeline's output (Phi-4 entries, catalog,
   table-level files are not opened). The first lines of the digitized table are shown only so you
   can tell which printed table is "table k" (tables are numbered top to bottom).
3. Copy names, numbers and location words exactly as printed. No conversion, no lookup.
4. A field is: a value; blank (the page does not give it); or `cannot tell` (it is on the page but
   cannot be read or decided). Say why in `notes`.
5. One row per location the table reports data for. A table about one site: one row. A table that
   lists many sites (well inventory, list of springs, several stations as rows or columns): one row
   per site; if there are more than 5, write `n_locations_in_table` and give full rows for the first,
   the last and three in between, evenly spaced, with `location_number` = the site's position in the table.
6. A table that holds no water data (index, contents, list of reports, text set as a table): one row,
   `data_type` = Not Water Related, everything else blank.

## Columns the reader fills (JSON list of objects in `key/drafts/{sample_no}.json`)

Location evidence: fill EVERY kind the page prints for this location, each as printed.

| field | content |
|---|---|
| `location_number` | 1, 2, ... (position of the site within the table) |
| `n_locations_in_table` | number of distinct locations the table reports |
| `coordinates` | printed latitude and longitude, e.g. `lat 34°38'00", long 120°26'10"` |
| `well_number` | state well number or the report's own well number, e.g. `4S/5W-4A1` |
| `station_number` | USGS station (gauge) number, e.g. `11-1335` |
| `station_name` | full station name: water body, relation, place, e.g. `Santa Ynez River near Lompoc, Calif.` |
| `township_range_section` | land survey description as printed, with meridian if printed |
| `place_name` | city, town, dam, bridge, ranch or other landmark the site is tied to |
| `place_relation` | how the site relates to that place: `at`, `near`, `above`, `below`, `3 miles north of` |
| `basin_or_area` | groundwater basin, valley, river basin or study area named for this table |
| `county` | only if printed on the page |
| `state` | only if printed on the page |
| `evidence_place` | where the best evidence is printed: in table / heading before table / elsewhere on page |
| `water_source` | name of the river, creek, canal, lake, reservoir, spring or well (owner or well name), as printed |
| `data_type` | Stream Discharge, Groundwater, Reservoir, Irrigation, Springs, Precipitation, Water Quality, Not Water Related, Other. It names the kind of water the site is on: a temperature or sediment table at a stream gauge is Stream Discharge. Water Quality only when the page does not say what kind of water it is. |
| `date_start`, `date_end` | first and last measurement in THIS table for THIS location, as fine as the page gives: `1916`, `1916-10` or `1916-10-01`. A water year "1917" is `1916-10` to `1917-09`. Never the station's period of record, never a construction year. |
| `temporal_resolution` | daily / monthly / annual / discrete / no_time_series. Annual includes annual max/min, peaks above a base and stream rating tables; discrete = measurements on given dates; no_time_series = logs, inventories, capacity tables, undated analyses. |
| `variables` | everything the table records for the location, `;`-separated, from: discharge, runoff, gage_height, contents, elevation, water_level, well_yield, temperature, sediment, water_chemistry, precipitation, evaporation (measured); well_construction, lithology, water_use, irrigated_area (described); other |
| `n_measurements` | number of measured values printed in this table for this location (values of the measured variables; not summary rows such as totals and means, not dates, not identifiers) |
| `n_variables` | n_measurements plus the other numbers recorded for the location (depths, diameters, altitudes, areas, summary values) |
| `construction_date` | wells only: year (or date) drilled or completed, same format as the dates |
| `confidence` | sure / unsure |
| `notes` | what was hard; how n_measurements was counted |

`location` and `location_type` are not typed: they are taken from the evidence columns by the
hierarchy in `schema.py` (coordinates > well number > station number > station name >
township-range-section > named place > basin or area > county > state).

## Rules for the cases the first 20 tables raised (set by Claude 2026-11-15; Anna to confirm)

What "the table" is
- The sampled table is exactly what the digitizer numbered as table k: the part of the page that the
  lines in your brief cover, from its first to its last line. If one printed table was split into
  blocks, only that block counts (its own dates, its own counts). If several printed blocks were kept
  as one digitized table, all of them count.

Counting (`n_measurements`, `n_variables`): count what is printed
- One printed value is one measurement, also when a brace or note says it stands for several days.
- A cell that says "No flow", "Dry" or 0 is a measurement. "No meas.", a dash or an empty cell is not.
  A whole month replaced by a note ("no flow during month") has no printed daily values: count nothing.
- A value printed in two units (cfs and gpm) is one measurement.
- A different variable printed in the foot rows of a table (end-of-month elevation under a daily
  contents grid) is a measurement of that variable: count it and list the variable.
- Totals, means, maxima, minima and runoff conversions computed from the table's own rows are summary
  values: only in `n_variables`. In a monthly summary table the monthly max / min / mean cells ARE the
  table's values (measurements); its calendar-year and water-year rows are summaries.
- Rating and capacity tables: every printed point counts (both columns of a gage height / discharge
  pair), `temporal_resolution` annual for rating tables, no_time_series for capacity tables. Dates of
  a rating table = the period it applies to; a capacity table has none.
- Time-of-day columns count like dates: not at all. Numbers inside remarks text are not counted.

Dates
- A table of extremes (annual maximum / minimum with their dates): the printed dates of the first and
  last extreme. A water-year table: the water year, also when some months are blank.

Location evidence
- `station_number`: USGS station numbers only. A report's own numbering ("Station 15", "Map no. 29")
  goes in `notes`.
- `station_name`: any printed name of the form water body + relation + place, also when it is only a
  table title ("inflow into Crane Valley reservoir near North Fork, Cal.").
- Two places in one name ("at X, near Y"): `place_name` X; Y, `place_relation` at; near. The page
  gives several distances to places: use the one in the station name, others in `notes`.
- A land grant ("in Yokaya Grant") goes in `township_range_section`, as printed.

Data type
- It follows the water body the site is on. Inflow into a reservoir measured at the reservoir, and
  samples from a lake: Reservoir. A stream station next to a reservoir ("Boulder Creek at Cuyamaca
  Reservoir"): Stream Discharge.

Continuation pages without column headings
- Record what the page prints. If the meaning of a column cannot be told from this page alone, leave
  the fields that depend on it blank and say so in `notes`; do not guess from the numbers' look.

Clarifications after the validation set (2026-11-15)
- Monthly summary table (one row per month): the monthly maximum, minimum and mean cells are its
  measurements. Its "second-foot-days" and "runoff in acre-feet" columns are totals and conversions:
  summary values, only in `n_variables`. Yearly rows: summary values.
- A variable that appears only in summary rows (runoff in acre-feet under a daily discharge grid) is
  not listed in `variables`.
- "Change in contents" is computed from the contents column: summary value.
- Chemical analyses: every printed analysis result counts, also computed ones (dissolved solids as
  sum, hardness, percent sodium, sodium adsorption ratio). The same quantity restated in other units
  (tons per acre-foot, tons per day next to mg/l) counts once.
- Numbers in the heading text above a table (well depth, diameter, altitude of land surface) are not
  in the table: not counted. They may still fill `construction_date` or location fields.
- A printed range ("500-700") is one value. Values given only in a footnote are not counted.
- A table whose first row is the last day of the previous period (Sept. 30 before an October to
  September table): use the printed rows for dates and counts.
- A complete daily water-year table may be dated to the day (1952-10-01 to 1953-09-30) or to the
  month (1952-10 to 1953-09); both are right. When values are printed only for part of the year,
  still give the water year in the title, unless the sampled block has its own dates.
- A canal or ditch that diverts water (also "spreading diversion"): `data_type` Irrigation.
- Well and spring numbers printed in short form under a group heading ("-26F1" under "5/3W-"): write
  the full number and say so in `notes`. Spring numbers go in `well_number`.
- A site with several measurement dates about a month apart is `discrete` unless the table is laid
  out as one value per month.
- A block that holds only column headings: Not Water Related, everything else blank.
- State from a running head or table title on the page counts as printed. A county that appears only
  inside an agency name does not.
