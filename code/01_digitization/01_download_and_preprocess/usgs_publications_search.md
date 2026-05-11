# USGS Publications Warehouse Search

This document describes how to reproduce the publication list used in this project. The result of this process is already saved as `data/digitization_intermediates/01_download_and_preprocess/publication_list.csv` (1,042 documents). Only follow these steps if you need to re-run the search from scratch.

---

## Search terms

The following water-related terms were used in all searches:

> Wells, Borings, Underground, Waters, Water, Groundwater, Ground-water, Artesian, Flowing, Precipitation, Rain, Rainfall, Weather, Snow, Stream, Streams, Discharge, River, Rivers, Flow, Creek, Spring, Springs, Water-supply, Irrigation, Water-right, Water allocation

---

## How to reproduce the search

Go to the [USGS Publications Warehouse](https://pubs.usgs.gov/search) and run **two searches**, then combine and filter the results:

**Search A — California water documents published before 1970**
- Keywords: water-related terms above
- Filter: State = California
- Filter: Year Published ≤ 1970

**Search B — California water documents published 1970–1980**
- Keywords: water-related terms above
- Filter: State = California
- Filter: Year Published 1970–1980

Export both results as CSV (standard Publications Warehouse export format, which includes `URL`, `Publication ID`, `Title`, and other metadata columns).

---

## Filtering to the final list

After downloading the raw CSVs:

1. Merge Search A and Search B, drop duplicates on `Publication ID`
2. Keep only documents where a PDF is available (check `Number of Links > 0` and confirm the pub page has a downloadable PDF — the download script filters for links with class `usa-link Document`)
3. Manually review any ambiguous entries (non-water documents that matched the keyword list)

The final filtered list should match `publication_list.csv`: **1,042 documents** spanning publications from 1879–1980 with data coverage from ~1810–1980.

---

## Notes

- Santa Barbara County documents were searched separately (using "Santa Barbara" as a keyword rather than state filter) because some early documents were not tagged with a state. These are included in the final 1,042.
- The original searches were run in batches between May–August 2024 and are archived in `Chapter_2_USGS_Digitization/Literature-Data Review/Raw USGS Data Pulls/`.
- For full instructions on navigating the Publications Warehouse interface, see `docs/usgs_publications_warehouse_job_aid.docx`.
