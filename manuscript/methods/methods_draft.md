# Methods draft: digitization pipeline

Nov 16, 2026 · @Anna Boser

## Overview

We turned scanned USGS reports into machine-readable tables, each with metadata giving its location, dates and content, in three automated steps: table detection, digitization and metadata extraction (Fig. M1; one example table followed through all three steps in Fig. M2). Each step was evaluated against hand-checked pages.

Reports were found by searching the USGS Publications Warehouse for California reports published 1879–1980 that matched water-related keywords (wells, groundwater, stream, discharge, springs, irrigation and 21 other terms) (Supplementary Note 1). The search returned 1,042 reports. Of these, 629 (60%) were available as scanned PDFs, 104,729 pages in all. Every page was rendered at 300 dpi and saved as a grayscale PNG (pdf2image).

&#91;image: Flowchart of the pipeline from the USGS Publications Warehouse search to metadata for each table\]

**Figure M1.** The digitization pipeline. Black boxes are inputs and intermediate products, blue boxes are the published products, and blue labels are the steps. Under each automated step is its measured accuracy: the share of known table pages the detector found at the stricter of its two thresholds (0.8), the share of printed numbers Reducto read correctly and placed under the right row and column, and the share of metadata fields right, averaged over eight fields. Surrounding text is the page text Reducto returns along with the tables.

&#91;image: One page with two detected tables (a), the digitized first table (b), and its metadata (c)\]

**Figure M2.** One table followed through the pipeline, from printed page C26 of Floods of 1966 in the United States. (a, left) The scanned page with the detector's boxes and confidence scores. (b, top right) Reducto's output for the first table, unedited. (c, bottom right) The metadata returned for that table, one row per location; six fields are shown. Accuracies are those of Figure M1.

## Table detection

Because not every page in the reports contained usable tabular data and digitizing every page would be prohibitively expensive, we first used Table Transformer to locate where in the reports there were tables to be digitized (Smock et al. 2022). The detector found 85% of known tables at the stricter of the two thresholds used (0.8), and about 11% of the pages it flagged held no table (Table M1). This process led to 74,492 of the 104,729 pages (71%) being flagged for digitization.&#32;

**Approach.** Each page image was binarized (Otsu), resized so its longer side was 800 pixels, and passed to Table Transformer (Smock et al. 2022), which returns boxes labeled table or rotated table with a confidence score (Figure M2a). A page was sent on to digitization if any box scored at or above the threshold. The threshold was 0.75 for the 270 reports published 1971–1980 (43% of reports, 34% of pages) and 0.8 for the 359 reports published 1898–1970 (57% of reports, 66% of pages). In total, 74,492 pages were flagged.

**Evaluation.** While a false positive leads us to digitize a page without a table, leading to inefficiencies in the pipeline, a false negative leads to losses in the amount of data digitized. Therefore, our main evaluation metric is recall, or the number of pages with tables that were identified.&#32;

Recall was measured against an existing hand-compiled list of report pages that hold groundwater tables (139 USGS reports, mostly outside California, CITE Annette's paper). To clean the list of pages with no table, we manually checked every listed page. Listed pages with no table were removed (73 pages), and 27 listed pages that pointed to the page beside a table were moved to it, leaving 3,272 pages. Recall is the share of these pages with a box at or above the threshold. Its 95% interval comes from resampling reports (2,000 resamples).

In order to capture the inefficiency of the pipeline, we also calculate the false-alarm rate: the share of flagged pages with no table. We first took the share of flagged pages for which Reducto returned no table (9.7% at 0.75). We then corrected it by looking at 60 of those pages and 300 pages for which Reducto did return a table. Reducto was wrong in both directions: 3 of the 60 pages held a table it had missed (57 had none), and 9 of the 300 held no table. Weighting each sample's no-table rate by its group's share of flagged pages gives the corrected rate.

**Table M1.** Table detection at the two thresholds used. Recall is the share of the 3,272 listed pages with a table that have a detector box scoring at or above the threshold. False alarms are the share of flagged pages with no table. Values in brackets are 95% intervals, from 2,000 resamples of reports for recall and from resamples of the hand-labelled pages within each group for false alarms.

| Threshold | Recall | Tables found | False alarms among flagged pages |
| --- | --- | --- | --- |
| 0.75 | 87.5% (84.0–90.7) | 2,863 of 3,272 | 11.9% (10.3–13.9) |
| 0.8 | 85.3% (81.5–88.5) | 2,791 of 3,272 | 11.0% (9.3–13.0) |

## Digitization

Once the tables to be digitized were located, we used Reducto to digitize the information within each table. Reducto read 96% of printed numbers correctly and placed them under the right row and column (Table M2). The open-source model PaddleOCR, which we tested first, only achieved 1%.

**Approach.** Each flagged page was sent to Reducto (reducto.ai, commercial API, 2024), which returns the page as text blocks and tables (HTML) with bounding boxes. We kept the full output as JSON, though we also converted each table to a CSV file to improve accessibility. PaddleOCR (open source) was tested on the same pages and rejected.

**Evaluation.** We transcribed tables from 20 pages at random from 57 reports: 4 pages for each kind of table (daily, monthly, annual, discrete measurements, no time series), at most one page per report. The tables were transcribed from the page images by Claude Fable 5.2 (Anthropic) without access to either tool's output, then read against the scans by one author, who found no errors beyond the typing of superscripts and subscripts, which was corrected. The 20 pages hold 32 tables and 4,696 printed numbers.

A model is evaluated by scoring the percent of printed numbers that are both correct and sorted under a correct row label and column header, so that a user could find it. Rows and columns of the output were matched to the printed table in order, maximizing agreement (Supplementary Note 2). Labels were compared ignoring case, spacing, punctuation, leader dots and line-break hyphens. Number formatting was compared leniently: printed gaps ("21. 2", "4, 400") were closed and a lone letter o was read as zero (under strict formatting Reducto scores 91.1%). Because the test pages were drawn in equal numbers for each kind of table, the counts for each kind were scaled by the number of pages of that kind in the corpus (daily 66%, discrete 15%, monthly 12%, annual 5%, no time series 3%), so that the results estimate the share of all numbers in the corpus. Performance by table type is reported in Supplementary Table 1. The 95% ranges come from 5,000 resamples of pages within kinds.

**Table M2.** Digitization accuracy by tool and output format: Reducto (raw HTML and converted to CSV) and PaddleOCR. Percentages are shares of printed numbers, with every number counting equally and each kind of table scaled by its number of pages in the corpus; each column adds to 100% up to rounding. Values in brackets are 95% ranges from 5,000 resamples of pages within kinds.

| Outcome | Reducto, raw output (HTML) | Reducto, converted to CSV | PaddleOCR |
| --- | --- | --- | --- |
| Right number under the right row and column | 96.1% (93.5–98.3) | 86.9% (80.7–92.9) | 1.2% (0.4–2.3) |
| Right number, wrong row label | 0.0% | 0.0% | 1.7% |
| Right number, wrong column header | 1.0% | 10.2% | 1.9% |
| Right number, both labels wrong | 0.0% | 0.0% | 8.5% |
| Wrong number | 1.7% | 1.7% | 22.5% |
| Cell missing from a table that was found | 0.2% | 0.2% | 48.7% |
| Whole table missed | 1.0% | 1.0% | 15.6% |
| Tables found (of 32) | 29 (1 split, 2 missed) | 29 (1 split, 2 missed) | 22 (10 missed) |

The CSV conversion loses accuracy because multi-level column headers are flattened (10.2% of numbers end up under the wrong column header, against 1.0% in the raw output); the numbers themselves are unchanged. The two tables Reducto missed are rotated tables, returned as a transposed, scrambled grid.&#32;

## Metadata extraction

In order for the digitized tables to be discoverable and therefore useful to researchers, it is important to know what they contain, or metadata. We therefore deigned a pipeline to extract location, date, and content information for each table. On a held-out test set of 200 tables, the pipeline got 91% of metadata fields right on average (Table M3).

**Approach.** In addition to the digitized tables, Reducto returned all text of each page fully digitized. Each page was processed on its own by an open-weight language model, Qwen3.6-27B (FP8 weights, Apache-2.0 license) with reasoning turned off. The input was Reducto's text for the page with its tables numbered in reading order. For each numbered table the model returned every location the table covers and, for each location, the evidence printed on the page, copied verbatim: coordinates, well number, station number or name, township-range-section, place name with its relation (at, near, below), basin or area, county and state. It also returned the water source, data type (one of nine categories: stream discharge, groundwater, reservoir, irrigation, springs, precipitation, water quality, not water related, other), start and end dates (as precise as printed: 1916, 1916-10 or 1916-10-01), temporal resolution (daily, monthly, annual, discrete, no time series), the variables recorded (from a fixed list) and, for wells, the date of construction. The full prompt is found under Supplementary Note 3, in addition to three data cleaning steps that follow the prompt. Each location was then given the most specific evidence available, in the order coordinates, well number, station number, station name, township-range-section, place name, basin or area, county, state. The output has one row per location per table. The pipeline was run on all 106,319 digitized tables and extracted metadata for all but 152 (0.1%), where the model's answer was unreadable (72 tables) or left the table out (62), or Reducto had kept no page text (18). For these 152 tables, Claude Fable 5.2 (Anthropic) read the page scan and recorded the same fields following the answer-key rules (Supplementary Note 4).

**Evaluation.** The test set is 200 tables from 137 reports, drawn at random within four groups of data type (80 stream discharge, 50 groundwater, 35 reservoir, 35 other types), one table per page and at most three per report. Pages used during development and pages on tribal lands were excluded. The prompt and rules were developed on two separate validation sets of 100 tables each; the test set was scored once, after the pipeline was frozen.

The answer key was written from the page images by Claude Fable 5.2 (Anthropic), blind to the pipeline's output, following written rules (Supplementary Note 4). One author checked the first 20 tables and found no errors; the rest of the key was not checked by hand. A field is counted right if the page gives a value and the output matches it. Dates match if they agree to the precision of the key, variables if the sets are equal, and coordinates if their digits agree. Results are weighted to the published tables by the share of each data-type group.&#32;

**Table M3.** Metadata fields and their accuracy on the test set (200 tables, 289 rows, one per location per table). Accuracy is the share of rows for which the page gives the field and the output matches it. Tables are weighted by how common their data type is among the published tables (stream discharge 84%, groundwater 8%, reservoir 4%, other types 4%). Values in brackets are 95% ranges from 5,000 resamples of tables within data-type groups. Examples are the first row of the table in Figure M2.

| Field | What it records | Example | Accuracy (95% range) |
| --- | --- | --- | --- |
| Location | The most specific location evidence printed on the page for the site | Kern River near Kernville | 92% (88–96) |
| Location type | The kind of evidence the location is: coordinates, well number, station number, station name, township-range-section, named place, basin or area, county or state | station name | 94% (90–97) |
| Water source | Name of the river, creek, canal, lake, reservoir, spring or well | Kern River | 91% (86–95) |
| Data type | The kind of water the site is on, one of nine categories | Stream discharge | 96% (93–99) |
| Start date | First measurement in the table for the location, as precise as printed | 1950-11 | 83% (76–90) |
| End date | Last measurement in the table for the location, as precise as printed | 1966-12 | 84% (77–91) |
| Temporal resolution | Daily, monthly, annual, discrete or no time series | discrete | 95% (91–98) |
| Variables | What the table records, from a fixed list of 17 | discharge | 92% (88–96) |
| Mean of the eight fields |  |  | 91% (88–93) |

The number of locations per table was right for 98% of tables, and the construction date for 81% of the 19 rows that print one. The 91% headline is the mean of the eight fields from location to variables. Judged by year alone, the start date was right for 87% of rows (95% range 81–93) and the end date for 96% (92–99). Results for each group of data types are in Supplementary Table 2.

## Supplementary information

### Supplementary notes

#### Supplementary Note 1: Search terms

Reports were found by searching the USGS Publications Warehouse with the state filter set to California and the following 27 water-related terms: Wells, Borings, Underground, Waters, Water, Groundwater, Ground-water, Artesian, Flowing, Precipitation, Rain, Rainfall, Weather, Snow, Stream, Streams, Discharge, River, Rivers, Flow, Creek, Spring, Springs, Water-supply, Irrigation, Water-right, Water allocation. Two searches were run, one for reports published up to 1970 and one for 1970 to 1980, and the results were merged with duplicates removed. Reports on Santa Barbara County were also searched by the keyword "Santa Barbara", because some early reports carry no state tag.

#### Supplementary Note 2: How each printed number was scored

The scoring script compares two grids of cells for each table: the hand transcription (the printed table) and a tool's output. The transcription also records which rows are header rows and which columns hold row labels. Merged cells are repeated in every row or column they span. The script never uses raw position: row 5 of the printed table is not assumed to be row 5 of the output, because a tool may add, drop, split or merge rows and columns.

**Step 1: agreement between two cells.** For a printed cell *a* and an output cell *b*, the agreement *s*(*a*, *b*) is 1 if the two hold the same text, the same number, or the same label after trivia is removed (case, spacing, punctuation, leader dots). Otherwise it is 0.8 times the share of characters the two have in common, and 0 if either cell is empty.

**Step 2: lining up rows and columns.** Printed rows are paired with output rows, and printed columns with output columns, so that total agreement is as large as possible. A pairing *M* must be one-to-one and keep the printed order (if row 3 is paired with output row 4, row 5 cannot be paired with output row 2); rows and columns may be left unpaired. For a matrix of scores *S*, the best pairing is

```latex
M^* = \arg\max_{M} \sum_{(i,k) \in M} S_{ik}
```

found exactly by dynamic programming, as in sequence alignment. Rows are paired first, with *S* the agreement between printed row *i* and output row *k* (itself the best ordered pairing of their cells). Columns are then paired given the rows, and rows again given the columns, using

```latex
V_{jl} = \sum_{(i,k) \in M_{\mathrm{rows}}} s(T_{ij}, O_{kl}), \qquad W_{ik} = \sum_{(j,l) \in M_{\mathrm{cols}}} s(T_{ij}, O_{kl})
```

where *T* is the printed grid and *O* the output grid. The two steps are run twice. This is the alignment used by the GriTS table metric (Smock et al. 2022). Its result is a map from each printed row *i* to an output row *r*(*i*) and from each printed column *j* to an output column *c*(*j*), or to nothing.

**Step 3: the outcome of each printed number.** A printed number is a cell outside the header rows and label columns that holds exactly one number, with at most three other characters (a footnote mark, for example). For the printed number in row *i* and column *j*, the script looks at the output cell in row *r*(*i*) and column *c*(*j*):

- *Cell missing:* row *i* or column *j* has no partner in the output, or the output cell is empty.
- *Wrong number:* the output cell holds something else. Numbers are compared by value, so "1,300" equals "1300".
- *Right number:* the values are equal. The two labels are then checked.

The row label is right if the output cell in row *r*(*i*) and in the column paired with the printed label column holds the printed row label. The column header is right if the printed header of column *j* (all header levels joined, top to bottom) equals the output's header for column *c*(*j*), read either from the output rows paired with the printed header rows or from every output row above the first data row. Labels are compared after removing case, spacing, punctuation, leader dots and footnote marks. The four combinations give the outcomes right and findable, wrong row label, wrong column header, and both labels wrong. A table printed without headers cannot have a wrong column header.

**Step 4: pages with several tables.** Every printed table is compared with every output table on the page. An output table is linked to a printed table only if at least 3 cells, and at least 10% of the filled cells of the smaller table, agree exactly. Each output cell can be credited to one printed table only; the printed table that agrees most with an output table claims its cells first. A printed number takes its best outcome over the linked output tables. If no output table is linked, every number in the printed table is counted as *whole table missed*. A printed table is counted as split when two or more output tables each supply at least 10% of its numbers.

**Step 5: from pages to the reported shares.** For outcome *o*, the reported share is

```latex
\hat{p}_o = \frac{\sum_k w_k \, \bar{x}_{k,o}}{\sum_k w_k \, \bar{n}_k}
```

where *k* runs over the five kinds of table, *w* is the share of corpus pages of that kind, and the two bars are means over the four test pages of that kind: the number of printed numbers with outcome *o* per page, and the number of printed numbers per page. Every printed number therefore counts equally within a kind, and kinds count in proportion to how many numbers they are expected to hold in the corpus. The 95% ranges are the 2.5th to 97.5th percentiles of this share over 5,000 resamples, each drawing four pages with replacement within every kind.

**Check of the scoring script.** We drew 100 outcomes, 5 from each combination of output and outcome that occurred, and Claude (Anthropic) compared each with the transcription and the output cells the script had used. 98 were correct, and none was wrong about whether a number was right and findable. The two errors had one cause (a small table credited with numbers from an unrelated table on the same page), which was fixed by the rule in Step 4 that an output cell is credited once.

#### Supplementary Note 3: Prompt and data cleaning for metadata extraction

The prompt below was sent to the language model once for each page. The page text, with every table marked \[TABLE n\], replaced {page}.

```text
You read ONE scanned page of a historical USGS water report. The page text follows; every table is marked [TABLE n].
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
- "data_type": one of ['Stream Discharge', 'Groundwater', 'Reservoir', 'Irrigation', 'Springs', 'Precipitation', 'Water Quality', 'Not Water Related', 'Other']. It names the kind of water the SITE is on, not what was measured:
    a stream or river station is "Stream Discharge" also when the table gives temperature, sediment or chemical analyses;
    a well is "Groundwater" also for chemical analyses; a lake or reservoir (contents, elevation, capacity table, inflow measured at the reservoir, lake samples) is "Reservoir";
    a canal, ditch or diversion is "Irrigation"; a spring is "Springs"; a rain gauge is "Precipitation";
    "Water Quality" only when the page does not say what kind of water the samples come from.
- "date_start", "date_end": first and last measurement in THIS table for this location, as fine as printed: "1916", "1916-10" or "1916-10-01".
    A water year (or "year ending Sept. 30") 1917 runs "1916-10" to "1917-09": never write just "1917" for a water year.
    In a table with one row per water year, the first row labelled 1930 starts "1929-10" and the last row labelled 1950 ends "1950-09".
    If the table is one block of a longer table (for example only April to June), give the dates of the rows in this block.
    Dates printed without a year take the year from the table title or page heading. Not the station's period of record ("records available"), not the year a well was drilled. A capacity table has no dates.
- "temporal_resolution": one of ['daily', 'monthly', 'annual', 'discrete', 'no_time_series'].
    daily = a value for each day; monthly = one row or value per month; annual = one per year, also annual maximum/minimum tables, peaks above a base, and stream rating tables;
    discrete = measurements on given dates (water levels, samples, field measurements), also when they are about a month apart; no_time_series = well logs, inventories, capacity tables, undated analyses.
- "variables": list of what the table records, from ['discharge', 'runoff', 'gage_height', 'contents', 'elevation', 'water_level', 'well_yield', 'temperature', 'sediment', 'water_chemistry', 'precipitation', 'evaporation', 'well_construction', 'lithology', 'water_use', 'irrigated_area', 'other'].
    Use "elevation" for the water-surface elevation or stage of a lake or reservoir and "gage_height" for a stream. Do not list a quantity that appears only in total or summary rows (runoff in acre-feet under a daily discharge table).
    Depth, casing, diameter of a well = "well_construction"; a driller's log = "lithology".
- "construction_date": wells only: year drilled or completed, if printed.

Return ONLY a JSON list, one object per table, and inside it one object per location with only the fields the page gives:
[{"table": 1, "locations": [{"station_name": "...", "data_type": "...", ...}]}, {"table": 2, ...}]

PAGE:
{page}
```

**Data cleaning after the prompt.** Three cleaning steps were applied to the model's answers.

1. Location evidence not found in the page text was dropped (3.5% of location values on the two validation sets, 77 of 2,181; nearly all were states and counties the model supplied without the page printing them). This guarantees that every piece of location evidence in the output is printed on the page.
2. "Runoff" was removed from the variables of daily and monthly tables that also list discharge. In these tables runoff appears only in summary rows, as a total of the discharge values.
3. Generic water-source words were blanked, because they are not names: "groundwater", "well", "wells", "spring", "springs", "stream", "river", "reservoir" and the ditto mark "do.".

#### Supplementary Note 4: Rules for writing the metadata answer key

The answer key has one row per location per sampled table and records what the sampled page says. The rules below were given to the reader in writing. The written rules also covered counts of printed values, which are not reported here and are left out.

**General rules**

1. This page only. The reader works from the scan of the sampled page, never opens another page of the report, and never uses general knowledge to fill a gap. If the page does not give something, the field stays blank.
2. The key is written from the scan, never from the pipeline's output. The first and last lines of the digitized table are shown only so that the reader can tell which printed table was sampled (tables are numbered top to bottom).
3. Names, numbers and location words are copied exactly as printed, with no conversion and no lookup.
4. A field is a value, blank (the page does not give it), or "cannot tell" (it is on the page but cannot be read or decided), with the reason in a notes field.
5. One row per location the table reports data for. A table about one site has one row. A table that lists many sites (a well inventory, a list of springs, several stations as rows or columns) has one row per site; if there are more than five, the reader records the number of locations and gives full rows for the first, the last and three evenly spaced in between.
6. A table that holds no water data (index, contents, list of reports, text set as a table, a block holding only column headings) has one row with data type "not water related" and everything else blank.
7. The sampled table is exactly what the digitizer numbered as that table. If one printed table was split into blocks, only the sampled block counts, with its own dates. If several printed blocks were kept as one digitized table, all of them count.

**Location evidence.** Every kind of evidence the page prints for the location is filled, each in its own field: coordinates, well number, station number, station name, township-range-section, place name and its relation, basin or area, county, state. Location and location type are not typed by the reader; they are taken from these fields in the order coordinates, well number, station number, station name, township-range-section, named place, basin or area, county, state.

- Station number: USGS station numbers only. A report's own numbering ("Station 15", "Map no. 29") goes in the notes.
- Station name: any printed name of the form water body, relation, place, also when it is only a table title ("inflow into Crane Valley reservoir near North Fork, Cal.").
- Two places in one name ("at X, near Y"): both are recorded, in order. If the page gives several distances to places, the one in the station name is used.
- A land grant ("in Yokaya Grant") goes in township-range-section, as printed.
- Well and spring numbers printed in short form under a group heading ("-26F1" under "5/3W-") are written in full. Spring numbers go in the well number field.
- County and state are recorded only if printed on the page. A state in a running head or table title counts as printed; a county that appears only inside an agency name does not.

**Water source.** The name of the river, creek, canal, lake, reservoir, spring or well (owner or well name), as printed.

**Data type.** One of stream discharge, groundwater, reservoir, irrigation, springs, precipitation, water quality, not water related, other. It names the kind of water the site is on, not what was measured: a temperature or sediment table at a stream gauge is stream discharge. Inflow into a reservoir measured at the reservoir, and samples from a lake, are reservoir. A stream station next to a reservoir ("Boulder Creek at Cuyamaca Reservoir") is stream discharge. A canal or ditch that diverts water is irrigation. Water quality is used only when the page does not say what kind of water it is.

**Dates.** The first and last measurement in this table for this location, as fine as the page gives: 1916, 1916-10 or 1916-10-01. Never the station's period of record and never a construction year.

- A water year 1917 is 1916-10 to 1917-09, also when some months are blank. When values are printed only for part of the year, the water year in the title is still given, unless the sampled block has its own dates.
- A complete daily water-year table may be dated to the day or to the month; both are right.
- A table of extremes (annual maximum or minimum with their dates) takes the printed dates of the first and last extreme.
- A rating table takes the period it applies to. A capacity table has no dates.
- A table whose first row is the last day of the previous period (Sept. 30 before an October to September table) is dated from its printed rows.
- Construction date (wells only): the year or date drilled or completed, in the same format.

**Temporal resolution.** Daily, monthly, annual, discrete or no time series. Annual includes annual maximum and minimum tables, peaks above a base and stream rating tables. Discrete means measurements on given dates; a site with several measurement dates about a month apart is discrete unless the table is laid out as one value per month. No time series covers logs, inventories, capacity tables and undated analyses.

**Variables.** Everything the table records for the location, from a fixed list: discharge, runoff, gage height, contents, elevation, water level, well yield, temperature, sediment, water chemistry, precipitation, evaporation, well construction, lithology, water use, irrigated area, other. A variable that appears only in summary rows (runoff in acre-feet under a daily discharge grid) is not listed. A different variable printed in the foot rows of a table (end-of-month elevation under a daily contents grid) is listed.

**Continuation pages without column headings.** The reader records what the page prints. If the meaning of a column cannot be told from this page alone, the fields that depend on it are left blank; nothing is guessed from the look of the numbers.

### Supplementary tables

**Supplementary Table 1.** Digitization results by kind of table. Every printed number in the four test pages of a kind counts equally. Outcome columns are for Reducto's raw output and add to 100%; a wrong row label accounts for at most 0.6% in any kind and is included in "other". The last two columns give the share right and findable for the other two outputs.

| Kind of table | Pages | Printed numbers | Right and findable | Wrong column header | Wrong number | Cell missing | Table missed | Other | Reducto CSV, right and findable | PaddleOCR, right and findable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Daily | 4 | 1,396 | 98.4% | 0.2% | 1.4% | 0.0% | 0.0% | 0.0% | 95.8% | 0.1% |
| Monthly | 4 | 560 | 94.8% | 0.0% | 4.3% | 0.9% | 0.0% | 0.0% | 4.1% | 1.6% |
| Annual | 4 | 832 | 97.1% | 0.0% | 1.8% | 0.6% | 0.5% | 0.0% | 80.4% | 26.1% |
| Discrete measurements | 4 | 384 | 80.5% | 15.4% | 3.4% | 0.5% | 0.0% | 0.3% | 80.5% | 1.8% |
| No time series | 4 | 1,524 | 70.9% | 0.0% | 2.6% | 1.7% | 24.3% | 0.6% | 37.9% | 0.0% |

In the discrete kind, the wrong column headers all come from one page; in the no-time-series kind, the missed table is one rotated table of 370 numbers.

**Supplementary Table 2.** Share of metadata fields right on the test set (200 tables, 289 rows, one per location per table), for all rows and for each group of data types. The start year and end year rows compare years only.

| Field | All (n = 289 rows) | Stream discharge | Groundwater | Reservoir | Other types |
| --- | --- | --- | --- | --- | --- |
| Location | 92% | 95% | 61% | 92% | 93% |
| Location type | 94% | 96% | 75% | 97% | 80% |
| Water source | 91% | 95% | 63% | 61% | 71% |
| Data type | 96% | 98% | 83% | 97% | 92% |
| Start date | 83% | 85% | 67% | 68% | 86% |
| End date | 84% | 86% | 64% | 93% | 83% |
| Start year | 87% | 89% | 70% | 86% | 91% |
| End year | 96% | 97% | 76% | 96% | 91% |
| Temporal resolution | 95% | 96% | 83% | 97% | 97% |
| Variables | 92% | 96% | 59% | 88% | 82% |
