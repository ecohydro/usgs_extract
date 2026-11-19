# Restored Historical California Water Data (1850s–1980) — User Guide

> **DRAFT (updated 2026-11-19 for the new metadata).** Items marked **[TODO]** depend on
> decisions or steps not yet finished (HydroShare packaging, citation, license, final
> counts). Counts are from `data/catalog/restored_tables_catalog.csv` (2026-11-18 build)
> and will be refreshed before release.

## 1. What this dataset is

This dataset contains tables of historical water measurements digitized from scanned
U.S. Geological Survey (USGS) reports about California, published up to 1980.
Most of these records were available only as printed or scanned pages and were not in
any USGS database such as the National Water Information System (NWIS).

The published release contains:

| | |
|---|---|
| Digitized tables | 97,321 |
| Estimated measurements (§6.4) | about 17.5 million |
| Source documents | 583 USGS reports |
| Pages with tables | 61,421 |
| Tables with a location | 82,864 (85%; precise, approximate or basin) |
| Tables with dates | 89,509 (92%) |
| Record type (§4.4) | daily 43,756 · discrete 22,274 · monthly 15,510 · annual 9,460 · no time series 3,331 · not given 2,990 |

Water types (by table): Stream Discharge (about 77%), Groundwater, Irrigation, Reservoir,
Water Quality, Springs, Precipitation, Not Water Related, and Other.

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
2. **Table detection.** A Microsoft Table Transformer model flagged pages containing tables
   (85% of known table pages found in a hand-checked test set).
3. **Digitization.** Reducto.ai converted each table to a CSV file and saved the full page
   reading (all text blocks and tables) as JSON.
4. **Metadata.** An open-weight language model (Qwen3.6-27B) read each page and recorded,
   for every table and every site (location) in it: the location evidence as printed,
   the water source, the water type, the first and last measurement dates, the record
   type and the variables recorded. On a 200-table test set the eight graded fields were
   right 91% of the time on average (83–96% by field).
5. **Checked and rebuilt fields.** Locations were rebuilt from evidence printed on each
   page (coordinates, station and well numbers, township/range, named places), dates were
   checked against the report's publication year, and water types were completed and
   reviewed (§5).
6. **Screening.** Tables whose metadata answer was corrupted (131 tables) and pages on
   federally recognized tribal lands (§7) are not included.

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
    {doc_id}_page_{N}_metadata.csv   page metadata [TODO: content to be decided]
```

- `doc_id` is the USGS Publications Warehouse publication ID. The report itself can be
  viewed at the `usgs_url` given in the catalog.
- `page_number` is the page's position in the scanned PDF. It is not the number printed
  on the page.
- Table numbers follow the order of the tables on the page.

**Catalog and metadata files** (use these to search):

| File | Contents |
|---|---|
| `restored_tables_catalog.csv` | One row per table: report, water type, dates, location, search keys, quality flags, and paths to the table, page image and JSON |
| `restored_tables_catalog.gpkg` | The same, for located tables only, as a GIS layer (layer `tables`, WGS84 / EPSG:4326) |
| `catalog_key.txt` | Description of every catalog column |
| [TODO] per-location metadata file, `metadata_key.txt` | One row per site per table: the language-model fields as printed plus the rebuilt location, water type and dates |

> **Use the catalog for searching.** It holds the checked and rebuilt locations and
> dates. [TODO: decide the content of the public per-location metadata file and of the
> per-page `_metadata.csv` files.]

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
- `station_name_on_page` — the NWIS station name (stream + at/near place, e.g. "Santa Ynez
  River near Lompoc") is printed on the page without a number.
- `nearest_named` — the nearest NWIS stream site within 5 km that shares a name word with
  the restored site.
- `nearest_well` — the nearest NWIS groundwater site within 0.5 km. Treat as a candidate
  only; nearby wells are often different wells.

`nwis_match_km` is the distance between the restored point and the NWIS site.

### 4.3 By water type, date and report

- `water_type` — the kind of water the site is on (see §5.3).
- `year_start`, `year_end` — years of the measurements in the table (§5.2).
- `doc_title`, `series_title`, `year_published`, `authors`, `usgs_url`, `doi` — the source
  report.
- `table_caption` — the table's printed title, when the digitizer found it. Often the
  quickest way to see what a table holds.
- Out-of-state data: filter `in_california == False`. Tables with no location have both
  `state` and `in_california` blank.

### 4.4 By record type and variable

| `temporal_resolution` | What the table holds |
|---|---|
| `daily` | A value for each day: the classic USGS grid of day rows (1–31) by month columns |
| `monthly` | One row or value per month |
| `annual` | One per year: annual peaks, yearly totals or means, annual maximum/minimum, peaks above a base, stream rating tables |
| `discrete` | Measurements on given dates: field discharge measurements, well water levels, water-quality samples (also when about a month apart) |
| `no_time_series` | No time dimension: well logs, well inventories, reservoir capacity tables, undated analyses |
| blank | Not given |

`variables` lists what the table records (`;`-separated): `discharge`, `runoff`,
`gage_height` (streams), `contents`, `elevation` (lake or reservoir water surface),
`water_level` (wells), `well_yield`, `temperature`, `sediment`, `water_chemistry`,
`precipitation`, `evaporation`; descriptive: `well_construction`, `lithology`,
`water_use`, `irrigated_area`; and `other`.

Both fields come from the language model. On the 200-table test set, `temporal_resolution`
was right for 95% of tables and `variables` exactly right for 92%.

### 4.5 Example

Daily stream discharge tables in Santa Barbara County before 1920, with precise locations:

```python
import pandas as pd
cat = pd.read_csv("restored_tables_catalog.csv")
sel = cat[(cat.water_type == "Stream Discharge")
          & (cat.county == "Santa Barbara")
          & (cat.year_start < 1920)
          & (cat.precision_class == "precise")
          & (cat.temporal_resolution == "daily")
          & cat.variables.fillna("").str.contains("discharge")]
for path in sel.table_csv:
    table = pd.read_csv(path)   # paths are relative to the data root
```

County names are given without the word "County" (e.g. `Santa Barbara`).

## 5. Accuracy

### 5.1 Locations

Every location comes from evidence printed **on the table's own page**, with one
labeled exception (`evidence_scope`, below). Coordinates guessed by a language model are
never used. Each table records how its location was found (`coord_method`) and how exact
it is (`precision_km`, `precision_class`). Each method was checked against coordinates
printed on the same pages:

| `coord_method` | Evidence on the page | Median error | Within 5 km | Class |
|---|---|---|---|---|
| `printed` | Latitude and longitude printed on the page | (the reference) | — | precise |
| `nwis_station` / `nwis_well` | NWIS station number | 0.09 km | 98% | precise |
| `nwis_name` | NWIS station name, no number | 0.1 km | 98% | precise |
| `nwis_name_cluster` | Station name matching several gauges within 5 km (centroid) | 2.4 km | 96% | precise |
| `plss_section` | Township, range and section (Public Land Survey) | 0.7 km | 98% | precise |
| `plss_wellno` | California state well number (e.g. 10S/6E-23B1) | about 0.75 km (vs NWIS wells) | — | precise |
| `gnis_*` "at" a place, dam, reservoir | Named place (USGS GNIS) | 0.8–1.6 km | 91–96% | precise |
| `plss_township` | Township and range only | 4.1 km | 74% | approximate |
| `wellno_group` | Several wells on one row, all within 5 km | up to 5 km | — | approximate |
| `gnis_*` "near", "below", "above", lake, spring | Named place | 2.9–8.2 km | 37–75% | approximate |
| `b118_basin` / `b118_subbasin` | Groundwater basin named on the page | point inside the basin | 93% / 67% of located wells inside | basin |

How to use the classes:

- **precise** — suitable for subwatershed (HUC12) work and point maps.
- **approximate** — county or HUC8 scale.
- **basin** — basin-level summaries only. The point is a placeholder inside the basin
  polygon, not the site.
- **conflict** (593 tables) — independent location methods disagree by more than 25 km,
  or a named-place point lies more than 30 km from every NWIS gauge on the stream the page
  names. The coordinates are kept but should not be relied on.
- **unlocated** — no usable location evidence on the page (13,864 tables). Many are
  groundwater tables that identify wells only by local numbers or owner names.

Other location columns:

- `evidence_scope`: `page` (all evidence on the table's page; nearly all tables) or
  `document` (947 tables: the meridian or direction letters of a township/well number
  were settled from elsewhere in the same report). Drop `document` rows for a strictly
  same-page set.
- `location_ambiguous = True` (2,233 tables): the table's sites are more than 5 km apart.
  The point is the best-located of them.
- A site is placed outside California when the page names another state for it (in the
  place name, e.g. a station name ending in ", Nev.", or in the metadata's `state` field).
  Out-of-state sites were located with national NWIS and GNIS references. Named-place
  matches are less reliable outside California, so only NWIS station numbers and "at"
  place names are classed precise there.

### 5.2 Dates

`year_start` and `year_end` are the first and last measurement years in the table, as
read by the language model from the table, its title and the page heading. They are not
the station's period of record and not the year a well was drilled.

- USGS stream records use the **water year** (October 1 – September 30, named for the
  year it ends). Water year 1915 runs from October 1914, so a table "for the year ending
  September 30, 1915" has `year_start = 1914`, `year_end = 1915`.
- Years must fall between 1800 and the report's publication year; impossible years are
  dropped (`date_status`).
- `construction_year_min` / `_max` give the years wells were drilled, kept separate from
  measurement dates.

Accuracy on the 200-table test set: start year right for 87% of key rows, end year for
96%.

### 5.3 Water type

- `water_type` is the kind of water the **site** is on, not what was measured: a stream
  station's temperature or chemistry table is Stream Discharge, and a well's chemical
  analyses are Groundwater. Canals, ditches and diversions are Irrigation. "Water Quality"
  is used only when the page does not say what kind of water was sampled.
- A table with several sites takes the type of most of its sites
  (`water_types_in_table` lists all of them).
- On the test set, the water type was right for 96% of key rows.
- `Other` means a data table was found but its water type could not be decided.
- `page_type`: `data`, `station_index` (station catalogues in the "Index of surface-water
  records" volumes: site information, no measurements), `index_contents` (indexes and
  tables of contents) or `report_list` (lists of earlier reports). Exclude
  `index_contents` and `report_list` pages from analyses, and `station_index` from
  analyses of measurements.

### 5.4 Digitization quality

`digitization_quality` is a heuristic flag describing how cleanly the cells were read.
It does not measure whether values are correct:

- **poor** (305 tables): no data rows, more than 90% empty cells, or more than 5% of cells
  with a letter read in place of a digit ("4,02C", "1O5").
- **fair** (2,390): more than 1% such cells, or the table collapsed to one column.
- **good** (94,612): everything else.

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
4. **Measurement counts are estimates.** `n_measurements` counts printed values of
   measured variables in the digitized table. Dates, identifiers, site descriptions,
   summary rows and columns computed from the table's own values are not counted. Daily
   tables passed a calendar check (99% of expected days found).
5. **Coverage gaps.** Table detection finds about 85% of known table pages. Some visible
   tables were read as plain text, not as a table; their text is still in the page JSON
   but there is no CSV.
6. **Groundwater locations are the weakest.** Many well tables give only local well
   numbers or owner names, so about half of groundwater tables are unlocated or located
   only to a basin.
7. **One point per table in the catalog.** A table listing many wells or stations gets one
   point (flagged by `location_ambiguous`). The per-location metadata and the page give the
   site-level detail.
8. **Out-of-state data** are labeled but were not the focus of the location work.

## 7. Data not included: tribal lands

To respect tribal data sovereignty, pages whose sites fall on federally recognized tribal
lands (Census TIGER 2023 American Indian/Alaska Native areas), or that name a reservation
or rancheria, are withheld from this release: 3,373 tables (about 555,000 estimated
measurements, 3.1% of the published total) on 2,231 pages from 232 reports, covering 172
federally recognized areas (91 in California). [TODO: confirm final counts before
release.] These data are being offered directly to the tribes.

## 8. Column reference

See `catalog_key.txt` (catalog) and `metadata_key.txt` (metadata) for every column.

[TODO before release: refresh counts, add citation/license/contact, finalize §3 to match
the HydroShare layout and the public metadata file, and update §7 counts.]
