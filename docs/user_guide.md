# Restored Historical California Water Data (1850s–1980) — User Guide

> **DRAFT (2026-11-13).** Items marked **[TODO]** depend on decisions or steps not yet
> finished (HydroShare packaging, citation, license, final counts). Counts are from
> `data/catalog/catalog_summary.csv` as of 2026-11-13 and will be refreshed before release.

## 1. What this dataset is

This dataset contains tables of historical water measurements digitized from scanned
U.S. Geological Survey (USGS) reports about California, published between 1879 and 1980.
Most of these records were available only as printed or scanned pages and were not in
any USGS database such as the National Water Information System (NWIS).

The published release contains:

| | |
|---|---|
| Digitized tables | 97,717 |
| Table rows (conservative measurement count) | about 2.4 million |
| Recorded numeric values (see §6.4) | about 17.8 million (17.2 million excluding derived summary tables) |
| Source documents | 583 USGS reports |
| Pages with tables | 61,698 |
| Tables with a location | 83,262 (85%; precise, approximate or basin) |
| Tables with dates | 92,899 (95%) |
| Record type (§4.4) | daily 42,355 · monthly 18,606 · discrete 16,815 · annual 10,492 · no time series 2,881 · unclassified 6,568 |

Water types (by table): Stream Discharge (about 84%), Groundwater, Reservoir, Irrigation,
Springs, Precipitation, Water Quality, Not Water Related, and Other.

Most data describe California, but some reports include sites in neighboring and other
states. These are included and labeled (§4.3).

**Citation:** [TODO: dataset DOI and paper citation — Hilton et al., "Over a century of
California water data restored by machine learning."]
**License:** [TODO]
**Contact:** [TODO]

## 2. How the data were made (short version)

Full methods are in the paper. In brief:

1. **Scanning to images.** Report PDFs from the USGS Publications Warehouse were converted
   to grayscale page images.
2. **Table detection.** A Microsoft Table Transformer model found pages containing tables
   (about 79% of known tables recovered in a test set).
3. **Digitization.** Reducto.ai converted each table to a CSV file and saved the full page
   reading (all text blocks and tables) as JSON.
4. **Page metadata.** A large language model (Microsoft Phi-4) read each page's text and
   recorded the site name, location description, water type, dates, units and key terms.
   About 87% of these fields were correct in hand-graded test sets.
5. **Rebuilt locations and dates.** Coordinates and dates determined by Microsoft Phi-4 were passed through a second series of checks; both were rebuilt from evidence printed on each page to improve accuracy (§5).
6. **Screening.** Tables with no language-model metadata (about 5% of detected tables) and
   pages on federally recognized tribal lands (§7) are not included.

The table values themselves were **not** cleaned or corrected. Each CSV is the
digitizer's reading of the printed table.

## 3. Files and folders

**[TODO: final HydroShare layout.]** The planned layout is one HydroShare resource per
water type, split into county folders. Inside, each page has its own folder:

```
{doc_id}/
  page_{N}/
    {doc_id}_page_{N}.png            scanned page image (grayscale)
    {doc_id}_page_{N}.json           full Reducto output for the page (all text and tables)
    {doc_id}_page_{N}_table1.csv     digitized table 1
    {doc_id}_page_{N}_table2.csv     digitized table 2 (when a page has several tables)
    {doc_id}_page_{N}_metadata.csv   page metadata (language-model fields + report details)
```

- `doc_id` is the USGS Publications Warehouse publication ID. The report itself can be
  viewed at the `usgs_url` given in the catalog.
- `page_number` is the page's position in the scanned PDF. It is not the number printed
  on the page.
- Table numbers follow the order of the tables on the page.

**Catalog files** (use these to search):

| File | Contents |
|---|---|
| `restored_tables_catalog.csv` | One row per table: report, water type, dates, location, search keys, quality flags, and paths to the table, page image and JSON |
| `restored_tables_catalog.gpkg` | The same, for located tables only, as a GIS layer (layer `tables`, WGS84 / EPSG:4326) |
| `catalog_key.txt` | Description of every catalog column |
| `main_metadata.csv`, `metadata_key.txt` | The original page-level language-model output with report details (§5.3) |

> **Use the catalog for locations and dates.** The per-page `_metadata.csv` files and
> `main_metadata.csv` hold the original language-model fields, including the
> coordinates and date strings that were replaced (§5). [TODO: decide whether the
> HydroShare page metadata files will carry the rebuilt values.]

## 4. Finding data

Open `restored_tables_catalog.csv` in any spreadsheet or data tool, or load the
GeoPackage in QGIS or ArcGIS, and filter on these columns:

### 4.1 By place

| Column | Use | Filled for |
|---|---|---|
| `county` | California or other-state county (Census TIGER 2023) | precise and approximate locations |
| `huc8`, `huc8_name` | USGS subbasin (California only) | precise and approximate locations |
| `huc12` | USGS subwatershed (California only) | precise locations only |
| `b118_basin_id`, `b118_basin_name` | DWR Bulletin 118 groundwater basin/subbasin | located points inside a basin, and groundwater pages that name a basin |
| `state`, `in_california` | State of the site | precise and approximate locations |
| `latitude`, `longitude` | Point location | all located tables |

Search keys are left blank when the location is too coarse for that scale (for example,
an approximate point is not assigned a HUC12).

### 4.2 By station: link to NWIS

`nwis_site_no` gives a likely matching USGS NWIS site, so restored records can be joined
to the modern record. `nwis_match` says how the match was made:

- `station_number_on_page` — the NWIS station number is printed on the page. Most reliable.
- `nearest_named` — the nearest NWIS stream site within 5 km that shares a name word with
  the restored site. Where a station number was also printed, this picked the same station
  for about 95% of tables.
- `nearest_well` — the nearest NWIS groundwater site within 0.5 km. Treat as a candidate
  only; nearby wells are often different wells.

`nwis_match_km` is the distance between the restored point and the NWIS site.

### 4.3 By water type, date and report

- `water_type` — the type assigned to the table (see `water_type_source` in
  `catalog_key.txt` for how).
- `year_start`, `year_end` — years of the data in the table (§5.2).
- `doc_title`, `series_title`, `year_published`, `authors`, `usgs_url`, `doi` — the source
  report.
- `table_caption` — the table's printed title, when the digitizer found it. Often the
  quickest way to see what a table holds.
- Out-of-state data: filter `in_california == False`. Tables with no location have both
  `state` and `in_california` blank.

### 4.4 By record type and parameter

Each table is classified from its own layout and title (not from the page-level language-model
field `temporal_resolution`, which is kept but superseded):

| `data_resolution` | What the table holds |
|---|---|
| `daily` | One value per day: the classic USGS grid of day rows (1–31) by month columns |
| `monthly` | One value per month: month rows, year rows by month columns, end-of-month readings, or samples taken about once a month |
| `annual` | One value per year: annual peaks, yearly totals or means, annual maximum/minimum, peaks above a base, stream rating tables |
| `discrete` | Measurements on given dates at irregular times: field discharge measurements, well water levels, water-quality samples |
| `no_time_series` | No time dimension: well logs, well inventories, reservoir capacity tables, undated analyses |
| `unclassified` | Could not be decided from the table or its title |

- `primary_parameter` — the main quantity (`discharge`, `gage_height`, `contents`,
  `water_level`, `temperature`, `sediment`, `water_chemistry`, `precipitation`,
  `evaporation`, `unknown`); `parameters` lists every parameter named in the title and headers.
- `derived_summary` — `True` for a monthly or annual table printed beside the daily table it
  summarizes (same or adjacent page, same water type and parameter). Its values repeat the
  daily record. To count values without double counting, sum `n_values_primary`, which is 0
  for these tables. `n_values` counts every numeric value in the table.

In a hand check of 65 tables, the classification matched the settled definitions for 59 of 65 tables, and the derived-summary flag for 62 of 65.

### 4.5 Example

Daily stream discharge tables in Santa Barbara County before 1920, with precise locations:

```python
import pandas as pd
cat = pd.read_csv("restored_tables_catalog.csv")
sel = cat[(cat.water_type == "Stream Discharge")
          & (cat.county == "Santa Barbara")
          & (cat.year_start < 1920)
          & (cat.precision_class == "precise")
          & (cat.data_resolution == "daily")
          & (cat.primary_parameter == "discharge")]
for path in sel.table_csv:
    table = pd.read_csv(path)   # paths are relative to the data root
```

County names are given without the word "County" (e.g. `Santa Barbara`).

## 5. Accuracy

### 5.1 Locations

Every location comes from evidence printed **on the table's own page**, with a few
labeled exceptions (`evidence_scope`, below). Coordinates guessed by the language model
are not used: when checked against coordinates printed on the page, most guessed
coordinates were tens of kilometres off. Each table records how its location was found
(`coord_method`) and how exact it is (`precision_km`, `precision_class`).

| `coord_method` | Evidence on the page | Checked against | Typical error | Class |
|---|---|---|---|---|
| `printed` | Latitude and longitude printed on the page | (is the reference) | — | precise |
| `nwis_station` / `nwis_well` | NWIS station number printed on the page | printed coordinates | 0.1 km median; 98% within 5 km | precise |
| `plss_section` | Township, range and section (Public Land Survey) | printed coordinates | 0.7 km median; 92% within 5 km | precise |
| `plss_wellno` | California state well number (e.g. 10S/6E-23B1) | NWIS well locations, printed coordinates | about 0.8 km | precise |
| `gnis_*` "at" a place, dam, reservoir | Named place (USGS GNIS) | printed coordinates | 0.7–1.7 km median; about 90% within 5 km | precise |
| `plss_township` | Township and range only | printed coordinates | 4.2 km median | approximate |
| `wellno_group` | Several wells on one row, all within 5 km | — | up to 5 km | approximate |
| `gnis_*` "near", "below", "above", lake, spring, place name only | Named place | printed coordinates | 3–8 km median; some > 25 km | approximate |
| `b118_basin` / `b118_subbasin` | Groundwater basin named on the page | located wells | point inside the basin; about 16 km radius | basin |

How to use the classes:

- **precise** — suitable for subwatershed (HUC12) work and point maps.
- **approximate** — county or HUC8 scale.
- **basin** — basin-level summaries only. The point is a placeholder inside the basin
  polygon, not the site.
- **unlocated** — no usable location evidence on the page (15,940 tables). Many are
  groundwater tables that identify wells only by local numbers.

Other location columns:

- `evidence_scope`: `page` (all evidence on the table's page; nearly all tables),
  `document` (the meridian or direction letters of a township/well number were settled
  from elsewhere in the same report; checked at 97–100% against NWIS), or `continuation`
  (location carried from the first page of a table continued across pages; strict rules
  only, capped at approximate). Drop `document` and `continuation` rows for a strictly
  same-page set.
- `location_ambiguous = True` (5,107 tables): the table covers several sites more than
  5 km apart, or its row could not be tied to one site. The point is one of the
  candidates.
- Out-of-state sites were located with national NWIS and GNIS references. Named-place
  matches are less reliable outside California, so only NWIS station numbers and "at"
  place names are classed precise there.

### 5.2 Dates

`year_start` and `year_end` describe the data in the table, read from the table and page
itself. `date_tier` (also `date_status`) says where the years came from, best first:

| Tier | Source |
|---|---|
| A `table_date_column` | A Date or Year column or row in the table |
| B `table_text` | Explicit dates elsewhere in the table |
| C `table_caption` | The table title (e.g. "Discharge ... for the year ending Sept. 30, 1915") |
| D `page_text` | Other text on the same page |
| E `period_of_record` | Fallback: the station's stated "records available" years |
| F `construction_year` | Fallback: years the wells were drilled |
| — `undated` | No date found |

**Tiers E and F are not measurement dates.** For any time-based analysis, use tiers
A–D only.

Notes:

- USGS stream records use the **water year** (October 1 – September 30, named for the
  year it ends). A table "for the year ending September 30, 1915" has
  `year_start = 1914`, `year_end = 1915`.
- `period_of_record_start/end` and `construction_start/end` are kept as separate columns
  and never used as the table's own dates.
- Years must fall between 1800 and the report's publication year.
- `year_start_llm`, `year_end_llm` hold the original language-model dates for comparison.

Accuracy, checked against hand grading of the page images: in a sample of 64 pages, the
table-level years matched the grader exactly for 46 and within one year for 52 (the
language-model dates matched 26 and 27). In a second sample of 30 pages where the two
methods disagreed, the table-level years were right for 25 and the language-model dates
for 6.

### 5.3 Water type and page metadata

- `water_type` comes from the language model's reading of the whole page. Where a page had
  tables of several types, column headers decided which table was which
  (`water_type_source`).
- `Other` means a data table was found but the language model could not decide its water
  type.
- `watersource_name`, `location_text`, `temporal_resolution`, `units` and `keyterms` are
  language-model fields (about 87% accurate overall in hand-graded tests). For the type of
  record, use the table-level `data_resolution` (§4.4) instead of `temporal_resolution`. Units are as
  printed: "second-feet", "sec.-ft." and "cfs" are the same unit (cubic feet per second)
  and are not standardized.
- `page_type`: `data`, `index_contents` (indexes and tables of contents), `narrative` or
  `no_page_text`. Exclude `index_contents` and `narrative` pages from analyses.

### 5.4 Digitization quality

`digitization_quality` is a heuristic flag describing how cleanly the cells were read.
It does not measure whether values are correct:

- **poor** (305 tables): no data rows, more than 90% empty cells, or more than 5% of cells
  with a letter read in place of a digit ("4,02C", "1O5").
- **fair** (2,403): more than 1% such cells, or the table collapsed to one column.
- **good** (95,275): everything else.

Old USGS typefaces print zero as a lone "o", for example on zero-flow days in daily
tables. These cells are counted separately (`frac_letter_zero`) and not treated as errors.
Read them as 0.

## 6. Known limitations

1. **Values are not cleaned.** Expect occasional letter/digit confusions ("4,02C" for
   4,020), decimal points for thousands commas ("3.860" for 3,860), merged cells (two days
   in one cell) and, rarely, two printed tables merged into one CSV. Always check
   surprising values against the page image.
2. **Data-quality marks are kept as printed.** USGS prefixes such as "e" (estimated) or
   "a" (affected by backwater) stay in the cell text.
3. **Table layouts vary.** Daily stream tables usually have days as rows and months as
   columns, with the year in the page title rather than the table. Annual and monthly
   tables differ by report. No single parser reads them all.
4. **Row and value counts are estimates.** "Table rows" counts CSV lines. A daily table
   row holds up to 12 values, so the count of numeric values (about 18 million) better
   reflects the data volume. Daily tables passed a calendar check (99% of expected days
   found). Counts for other tables may run up to about 20% high. Monthly and annual summary
   tables printed beside a daily table repeat its values; they are flagged
   `derived_summary` and left out of `n_values_primary`.
5. **Coverage gaps.** Table detection recovers about 79% of known tables. Some visible
   tables were read as plain text, not as a table; their text is still in the page JSON
   but there is no CSV. About 5% of detected tables had no language-model metadata and are
   not included.
6. **Groundwater locations are the weakest.** Many well tables give only local well
   numbers or owner names, so a large share of groundwater tables are unlocated or located
   only to a basin.
7. **One location per table.** A table listing many wells or stations gets one point
   (flagged by `location_ambiguous`). The page and table give the row-level detail.
8. **Out-of-state data** are labeled but were not the focus of the location work.

## 7. Data not included: tribal lands

To respect tribal data sovereignty, pages whose sites fall on federally recognized tribal
lands (Census TIGER 2023 American Indian/Alaska Native areas), or that name a reservation
or rancheria, are withheld from this release: about 3,100 tables (78,826 table rows,
3.1% of the published total) on 2,036 pages from 225 reports, covering 112 federally
recognized areas, most in California. [TODO: confirm final counts before release.]
These data are being offered directly to the tribes.

## 8. Column reference

See `catalog_key.txt` (catalog) and `metadata_key.txt` (page metadata) for every column.

[TODO before release: refresh counts, add citation/license/contact, finalize §3 to match
the HydroShare layout, and update §7 counts.]
