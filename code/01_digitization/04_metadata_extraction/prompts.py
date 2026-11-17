"""The instructions given to the model for one page (printed in the paper as a supplementary note).

Version v5 is the only one that ran on the test set and on the full collection. `{page}` is replaced by the
page text, in which every table is marked [TABLE n] (page_input.page_text).
"""
import schema as S

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


PROMPTS = {"v5": V5}
