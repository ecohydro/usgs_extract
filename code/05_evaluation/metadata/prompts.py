"""Prompts for reader 2, keyed by version so runs with different prompts coexist in model_runs/."""
import schema as S

V1 = """You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
For EACH table, list every location (site) the table reports data for, and describe it using ONLY words printed on this page.
Copy names, numbers and location words exactly as printed. Never guess, convert, or use outside knowledge. Use "" when the page does not say.

A table about one station, well, spring or reservoir has one location. A table whose rows (or columns) are different wells, springs or stations has one location per row (or column): list them all, in order.
A table that holds no water data (index, contents, list of reports, text set as a table) has one location with data_type "Not Water Related" and everything else "".

Return ONLY a JSON list, one object per table:
[{{"table": n,
  "locations": [{{
    "coordinates": printed latitude and longitude,
    "well_number": state well number or the report's own well number,
    "station_number": USGS station (gauge) number,
    "station_name": full station name as printed (water body, relation, place),
    "township_range_section": land survey description as printed (also land grants),
    "place_name": city, town, dam, bridge or other landmark the site is tied to,
    "place_relation": how the site relates to that place ("at", "near", "above", "below", "3 miles north of"),
    "basin_or_area": groundwater basin, valley, river basin or study area named for this table,
    "county": county, only if printed,
    "state": state, only if printed,
    "evidence_place": where the best location evidence is printed: "in table", "heading before table" or "elsewhere on page",
    "water_source": name of the river, creek, canal, lake, reservoir, spring or well (owner or well name),
    "data_type": one of {data_types}. It names the kind of water the site is on: a temperature or sediment table at a stream gauge is "Stream Discharge"; "Water Quality" only when the page does not say what kind of water it is,
    "date_start": first measurement in this table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01" (a water year 1917 starts "1916-10"),
    "date_end": last measurement, same format (a water year 1917 ends "1917-09"). Not the station's period of record, not a construction year,
    "temporal_resolution": one of {resolutions} (annual includes annual maximum/minimum and rating tables; discrete = measurements on given dates; no_time_series = logs, inventories, capacity tables, undated analyses),
    "variables": list from {variables},
    "construction_date": wells only: year drilled or completed
  }}]
}}]

PAGE:
{{page}}
""".format(data_types=S.DATA_TYPES, resolutions=S.TEMPORAL_RESOLUTIONS, variables=S.VARIABLES)

V2 = """You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
For EACH table, list every location (site) the table reports data for, and describe it using ONLY words printed on this page.
Copy names, numbers and location words exactly as printed. Never guess, convert, or use outside knowledge. Leave out any field the page does not give.

How many locations
- A table about one station, well, spring or reservoir has one location, even if it has many rows of dates.
- A table whose rows (or columns) are different wells, springs or stations has one location per row (or column): list them all, in the printed order.
- A table that holds no water data (index, contents, list of reports, column headings only, text set as a table) has one location with only "data_type": "Not Water Related".

Which words belong to a table
- Use the heading printed before the table (station or well heading), the table itself, and the rest of the page, in that order. On a page with several stations or wells, each table belongs to the last station or well heading printed before it.
- If the table is a continuation and its site is not named on this page, give only what the page prints (for example the area in the running head).

Fields (per location):
- "coordinates": printed latitude and longitude, as printed.
- "well_number": state well number, spring number or the report's own well number, as printed (write the full number if the page prints a short form under a group heading).
- "station_number": USGS station (gauge) number only. Not a report's own numbering such as "Station 15" or "Site 42".
- "station_name": the full station name as printed: water body, relation and place ("Santa Ynez River near Lompoc, Calif.").
- "township_range_section": land survey description as printed (also a land grant, "in Yokaya Grant").
- "place_name": city, town, dam, bridge or other landmark the site is tied to. "place_relation": at / near / above / below / "3 miles north of".
- "basin_or_area": groundwater basin, valley, river basin or study area named for this table (also from a running head).
- "county", "state": only if printed on this page (not when the word only appears inside an agency or owner name).
- "evidence_place": where the best location evidence is printed: "in table", "heading before table" or "elsewhere on page".
- "water_source": name of the river, creek, canal, lake, reservoir or spring; for a well, the owner or well name.
- "data_type": one of {data_types}. It names the kind of water the SITE is on, not what was measured:
    a stream or river station is "Stream Discharge" also when the table gives temperature, sediment or chemical analyses;
    a well is "Groundwater" also for chemical analyses; a lake or reservoir (contents, elevation, capacity table, inflow measured at the reservoir, lake samples) is "Reservoir";
    a canal, ditch or diversion is "Irrigation"; a spring is "Springs"; a rain gauge is "Precipitation";
    "Water Quality" only when the page does not say what kind of water the samples come from.
- "date_start", "date_end": first and last measurement in THIS table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01".
    A water year (or "year ending Sept. 30") 1917 runs "1916-10" to "1917-09": never write just "1917" for a water year.
    If the table is one block of a longer table (for example only April to June), give the dates of the rows in this block.
    Dates printed without a year take the year from the table title or page heading. Not the station's period of record ("records available"), not the year a well was drilled. A capacity table has no dates.
- "temporal_resolution": one of {resolutions}.
    daily = a value for each day; monthly = one row or value per month; annual = one per year, also annual maximum/minimum tables, peaks above a base, and stream rating tables;
    discrete = measurements on given dates (water levels, samples, field measurements), also when they are about a month apart; no_time_series = well logs, inventories, capacity tables, undated analyses.
- "variables": list of what the table records, from {variables}.
    Use "elevation" for the water-surface elevation or stage of a lake or reservoir and "gage_height" for a stream. Do not list a quantity that appears only in total or summary rows (runoff in acre-feet under a daily discharge table).
    Depth, casing, diameter of a well = "well_construction"; a driller's log = "lithology".
- "construction_date": wells only: year drilled or completed, if printed.

Return ONLY a JSON list, one object per table, and inside it one object per location with only the fields the page gives:
[{{"table": 1, "locations": [{{"station_name": "...", "data_type": "...", ...}}]}}, {{"table": 2, "locations": [...]}}]

PAGE:
{{page}}
""".format(data_types=S.DATA_TYPES, resolutions=S.TEMPORAL_RESOLUTIONS, variables=S.VARIABLES)

V3 = """You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
For EACH table, list every location (site) the table reports data for, and describe it using ONLY words printed on this page.
Copy names, numbers and location words exactly as printed. Never guess, convert, or use outside knowledge. Leave out any field the page does not give.

How many locations
- A table about one station, well, spring or reservoir has one location, even if it has many rows of dates.
- A table whose rows (or columns) are different wells, springs or stations has one location per row (or column): list them all, in the printed order.
- A table that holds no water data (index, contents, list of reports, column headings only, text set as a table) has one location with only "data_type": "Not Water Related".

Which words belong to a table
- Use the heading printed before the table (station or well heading), the table itself, and the rest of the page, in that order. On a page with several stations or wells, each table belongs to the last station or well heading printed before it.
- If the table is a continuation and its site is not named on this page, give only what the page prints (for example the area in the running head).

Fields (per location):
- "coordinates": printed latitude and longitude, as printed.
- "well_number": state well number, spring number or the report's own well number, as printed (write the full number if the page prints a short form under a group heading).
- "station_number": USGS station (gauge) number only. Not a report's own numbering such as "Station 15" or "Site 42".
- "station_name": the full station name as printed: water body, relation and place ("Santa Ynez River near Lompoc, Calif.").
- "township_range_section": land survey description as printed (also a land grant, "in Yokaya Grant").
- "place_name": city, town, dam, bridge or other landmark the site is tied to. "place_relation": at / near / above / below / "3 miles north of".
- "basin_or_area": groundwater basin, valley, river basin or study area named for this table (also from a running head).
- "county", "state": only if printed on this page (not when the word only appears inside an agency or owner name).
- "evidence_place": where the best location evidence is printed: "in table", "heading before table" or "elsewhere on page".
- "water_source": name of the river, creek, canal, lake, reservoir or spring; for a well, the owner or well name as printed. Never a generic word such as "Groundwater" or "Well"; leave it out if no name is printed.
- "data_type": one of {data_types}. It names the kind of water the SITE is on, not what was measured:
    a stream or river station is "Stream Discharge" also when the table gives temperature, sediment or chemical analyses;
    a well is "Groundwater" also for chemical analyses; a lake or reservoir (contents, elevation, capacity table, inflow measured at the reservoir, lake samples) is "Reservoir";
    a canal, ditch or diversion is "Irrigation"; a spring is "Springs"; a rain gauge is "Precipitation";
    "Water Quality" only when the page does not say what kind of water the samples come from.
- "date_start", "date_end": first and last measurement in THIS table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01".
    A water year (or "year ending Sept. 30") 1917 runs "1916-10" to "1917-09": never write just "1917" for a water year.
    In a table with one row per water year, the first row labelled 1930 starts "1929-10" and the last row labelled 1950 ends "1950-09".
    If the table is one block of a longer table (for example only April to June), give the dates of the rows in this block.
    Dates printed without a year take the year from the table title or page heading. Not the station's period of record ("records available"), not the year a well was drilled. A capacity table has no dates.
- "temporal_resolution": one of {resolutions}.
    daily = a value for each day; monthly = one row or value per month; annual = one per year, also annual maximum/minimum tables, peaks above a base, and stream rating tables;
    discrete = measurements on given dates (water levels, samples, field measurements), also when they are about a month apart; no_time_series = well logs, inventories, capacity tables, undated analyses.
- "variables": list of what the table records, from {variables}.
    Use "elevation" for the water-surface elevation or stage of a lake or reservoir and "gage_height" for a stream. Do not list a quantity that appears only in total or summary rows (runoff in acre-feet under a daily discharge table).
    Depth, casing, diameter of a well = "well_construction"; a driller's log = "lithology".
- "construction_date": wells only: year drilled or completed, if printed.

Return ONLY a JSON list, one object per table, and inside it one object per location with only the fields the page gives:
[{{"table": 1, "locations": [{{"station_name": "...", "data_type": "...", ...}}]}}, {{"table": 2, "locations": [...]}}]

PAGE:
{{page}}
""".format(data_types=S.DATA_TYPES, resolutions=S.TEMPORAL_RESOLUTIONS, variables=S.VARIABLES)

V4 = """You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
For EACH table, list every location (site) the table reports data for, and describe it using ONLY words printed on this page.
Copy names, numbers and location words exactly as printed. Never guess, convert, or use outside knowledge. Leave out any field the page does not give.

How many locations
- A table about one station, well, spring or reservoir has one location, even if it has many rows of dates.
- A table whose rows (or columns) are different wells, springs or stations has one location per row (or column): list them all, in the printed order.
- A table that holds no water data (index, contents, list of reports, column headings only, text set as a table) has one location with only "data_type": "Not Water Related".

Which words belong to a table
- Use the heading printed before the table (station or well heading), the table itself, and the rest of the page, in that order. On a page with several stations or wells, each table belongs to the last station or well heading printed before it.
- If the table is a continuation and its site is not named on this page, give only what the page prints (for example the area in the running head).

Fields (per location):
- "coordinates": printed latitude and longitude, as printed.
- "well_number": state well number, spring number or the report's own well number, as printed (write the full number if the page prints a short form under a group heading).
- "station_number": USGS station (gauge) number, also in its short form printed before the station name ("11-1335.", "1680.", "308."). Always give it when it is printed, together with the station name. Not a report's own numbering such as "Station 15" or "Site 42".
- "station_name": the full station name as printed: water body, relation and place ("Santa Ynez River near Lompoc, Calif.").
- "township_range_section": land survey description as printed (also a land grant, "in Yokaya Grant").
- "place_name": city, town, dam, bridge or other landmark the site is tied to. "place_relation": at / near / above / below / "3 miles north of".
- "basin_or_area": groundwater basin, valley, river basin or study area named for this table (also from a running head).
- "county", "state": only if printed on this page (not when the word only appears inside an agency or owner name).
- "evidence_place": where the best location evidence is printed: "in table", "heading before table" or "elsewhere on page".
- "water_source": name of the river, creek, canal, lake, reservoir or spring; for a well, the owner or well name as printed. Never a generic word such as "Groundwater" or "Well"; leave it out if no name is printed.
- "data_type": one of {data_types}. It names the kind of water the SITE is on, not what was measured:
    a stream or river station is "Stream Discharge" also when the table gives temperature, sediment or chemical analyses;
    a well is "Groundwater" also for chemical analyses; a lake or reservoir (contents, elevation, capacity table, inflow measured at the reservoir, lake samples) is "Reservoir";
    a canal, ditch or diversion is "Irrigation"; a spring is "Springs"; a rain gauge is "Precipitation";
    "Water Quality" only when the page does not say what kind of water the samples come from.
- "date_start", "date_end": first and last measurement in THIS table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01".
    A water year (or "year ending Sept. 30") 1917 runs "1916-10" to "1917-09": never write just "1917" for a water year.
    In a table with one row per water year, the first row labelled 1930 starts "1929-10" and the last row labelled 1950 ends "1950-09".
    If the table is one block of a longer table (for example only April to June), give the dates of the rows in this block.
    Dates printed without a year take the year from the table title or page heading. Not the station's period of record ("records available"), not the year a well was drilled. A capacity table has no dates.
- "temporal_resolution": one of {resolutions}.
    daily = a value for each day; monthly = one row or value per month; annual = one per year, also annual maximum/minimum tables, peaks above a base, and stream rating tables;
    discrete = measurements on given dates (water levels, samples, field measurements), also when they are about a month apart; no_time_series = well logs, inventories, capacity tables, undated analyses.
- "variables": list of what the table records, from {variables}.
    Use "elevation" for the water-surface elevation or stage of a lake or reservoir and "gage_height" for a stream. Do not list a quantity that appears only in total or summary rows (runoff in acre-feet under a daily discharge table).
    Depth, casing, diameter of a well = "well_construction"; a driller's log = "lithology".
- "construction_date": wells only: year drilled or completed, if printed.

For each table also say how its numbers should be counted (a script does the counting):
- "measured_columns": the column headings, copied from the table, of the columns that hold measured values. For a grid of days by months write ["all month columns"]. Leave out columns of dates, times, day numbers, identifiers, owners, remarks, and columns that are totals or conversions of other columns (run-off in acre-feet, second-foot-days, change in contents, tons per acre-foot next to mg/l).
- "summary_rows": the first-cell labels, copied from the table, of rows that are totals, means, maxima, minima or yearly lines computed from the other rows (for example ["Total", "Mean", "Max", "Min", "Ac-ft", "CAL YR 1964", "WTR YR 1965"]). A foot row that gives a different measured quantity (end-of-month elevation under a daily contents grid) is NOT a summary row: list its label in "measured_rows" instead.
- "rows_are_locations": true if each row of the table is a different location, else false.

Return ONLY a JSON list, one object per table, and inside it one object per location with only the fields the page gives:
[{{"table": 1, "measured_columns": [...], "summary_rows": [...], "measured_rows": [...], "rows_are_locations": false, "locations": [{{"station_name": "...", "data_type": "...", ...}}]}}, {{"table": 2, ...}}]

PAGE:
{{page}}
""".format(data_types=S.DATA_TYPES, resolutions=S.TEMPORAL_RESOLUTIONS, variables=S.VARIABLES)

# v5 (2026-11-15): v4 without the counting hints; counts left the pipeline (Anna).
V5 = """You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
For EACH table, list every location (site) the table reports data for, and describe it using ONLY words printed on this page.
Copy names, numbers and location words exactly as printed. Never guess, convert, or use outside knowledge. Leave out any field the page does not give.

How many locations
- A table about one station, well, spring or reservoir has one location, even if it has many rows of dates.
- A table whose rows (or columns) are different wells, springs or stations has one location per row (or column): list them all, in the printed order.
- A table that holds no water data (index, contents, list of reports, column headings only, text set as a table) has one location with only "data_type": "Not Water Related".

Which words belong to a table
- Use the heading printed before the table (station or well heading), the table itself, and the rest of the page, in that order. On a page with several stations or wells, each table belongs to the last station or well heading printed before it.
- If the table is a continuation and its site is not named on this page, give only what the page prints (for example the area in the running head).

Fields (per location):
- "coordinates": printed latitude and longitude, as printed.
- "well_number": state well number, spring number or the report's own well number, as printed (write the full number if the page prints a short form under a group heading).
- "station_number": USGS station (gauge) number, also in its short form printed before the station name ("11-1335.", "1680.", "308."). Always give it when it is printed, together with the station name. Not a report's own numbering such as "Station 15" or "Site 42".
- "station_name": the full station name as printed: water body, relation and place ("Santa Ynez River near Lompoc, Calif.").
- "township_range_section": land survey description as printed (also a land grant, "in Yokaya Grant").
- "place_name": city, town, dam, bridge or other landmark the site is tied to. "place_relation": at / near / above / below / "3 miles north of".
- "basin_or_area": groundwater basin, valley, river basin or study area named for this table (also from a running head).
- "county", "state": only if printed on this page (not when the word only appears inside an agency or owner name).
- "evidence_place": where the best location evidence is printed: "in table", "heading before table" or "elsewhere on page".
- "water_source": name of the river, creek, canal, lake, reservoir or spring; for a well, the owner or well name as printed. Never a generic word such as "Groundwater" or "Well"; leave it out if no name is printed.
- "data_type": one of {data_types}. It names the kind of water the SITE is on, not what was measured:
    a stream or river station is "Stream Discharge" also when the table gives temperature, sediment or chemical analyses;
    a well is "Groundwater" also for chemical analyses; a lake or reservoir (contents, elevation, capacity table, inflow measured at the reservoir, lake samples) is "Reservoir";
    a canal, ditch or diversion is "Irrigation"; a spring is "Springs"; a rain gauge is "Precipitation";
    "Water Quality" only when the page does not say what kind of water the samples come from.
- "date_start", "date_end": first and last measurement in THIS table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01".
    A water year (or "year ending Sept. 30") 1917 runs "1916-10" to "1917-09": never write just "1917" for a water year.
    In a table with one row per water year, the first row labelled 1930 starts "1929-10" and the last row labelled 1950 ends "1950-09".
    If the table is one block of a longer table (for example only April to June), give the dates of the rows in this block.
    Dates printed without a year take the year from the table title or page heading. Not the station's period of record ("records available"), not the year a well was drilled. A capacity table has no dates.
- "temporal_resolution": one of {resolutions}.
    daily = a value for each day; monthly = one row or value per month; annual = one per year, also annual maximum/minimum tables, peaks above a base, and stream rating tables;
    discrete = measurements on given dates (water levels, samples, field measurements), also when they are about a month apart; no_time_series = well logs, inventories, capacity tables, undated analyses.
- "variables": list of what the table records, from {variables}.
    Use "elevation" for the water-surface elevation or stage of a lake or reservoir and "gage_height" for a stream. Do not list a quantity that appears only in total or summary rows (runoff in acre-feet under a daily discharge table).
    Depth, casing, diameter of a well = "well_construction"; a driller's log = "lithology".
- "construction_date": wells only: year drilled or completed, if printed.

Return ONLY a JSON list, one object per table, and inside it one object per location with only the fields the page gives:
[{{"table": 1, "locations": [{{"station_name": "...", "data_type": "...", ...}}]}}, {{"table": 2, ...}}]

PAGE:
{{page}}
""".format(data_types=S.DATA_TYPES, resolutions=S.TEMPORAL_RESOLUTIONS, variables=S.VARIABLES)


PROMPTS = {"v1": V1, "v2": V2, "v3": V3, "v4": V4, "v5": V5}
