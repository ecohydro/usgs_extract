Station-index check (2026-11-14)

Question: are the tables from the 13 "Index of surface-water records" volumes (USGS, 1948-1970)
station catalogues (station name, location, drainage area, period of record) with NO measurements?
If yes, their pages become a non-data page type ("station_index"), like the report lists, and drop
out of Figures 1-3 (about 5,600 estimated measurements; 305 records at 260 restored stream sites).

Files
  station_index_tables.csv  one row per table (188 tables on 170 pages), with catalog fields
                            (water type, dates, location class, caption) and three columns to fill:
                              is_station_index        y / n
                              contains_measurements   y / n  (any discharge, stage, etc. values?)
                              notes
  pages/                    page images, {doc}_p{page}.png (git-ignored copies of data/digitized)
  tables/                   digitized table CSVs, {doc}_p{page}_table{N}.csv
