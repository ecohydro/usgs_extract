# Truth set: typing rules and file format

The truth is typed from the page image only. Whoever types never opens the Reducto or PaddleOCR
output for that page, so the truth cannot lean toward either tool.

## What is typed
- Every printed table on the page, top to bottom (numbered 1, 2, ... in reading order), whether or
  not a tool found it. A table continued from the previous page counts. Running text, station
  descriptions (LOCATION.--, GAGE.--), captions and footnote text are not tables.
- The table title or caption goes in `tables.csv`, not in the grid.

## Files (data/analysis/evaluation/digitization/truth/)
- `{doc}_page_{N}_table{k}.csv`: the grid, plain CSV, no index, every cell text, every row the same
  number of columns. Header rows first, then body rows in printed order.
- `{doc}_page_{N}_tables.csv`: one row per table on that page: `doc_id, page_number, table_no, title, n_header_rows,
  label_cols, n_body_rows, n_cols, notes`.
- `{doc}_page_{N}_uncertain.csv`: one row per cell the typist could not read with confidence:
  `doc_id, page_number, table_no, row, col, typed, alternative, reason` (row and col are 0-based
  positions in the CSV grid, header rows included).

## Typing rules (Anna, 2026-11-14)
1. Type exactly what is printed: "1,300" keeps its comma, ".5" stays ".5", "26" is not "26.0",
   "Sept." keeps its period. No corrections of apparent printing errors.
2. Dotted leaders and dashes that mean "no value" are an empty cell. Leaders that trail a row label
   ("October........") are dropped: the label is "October".
3. A mark stays with its number in the same cell as printed: "a 12.3", "*450", "12.3 b".
4. Ditto marks and "do." are typed as printed ("do."), not expanded.
5. Headers keep every printed level, one CSV row per level.
   - A label spanning several columns is repeated in each column it covers.
   - A label spanning several header rows is typed once, in the top row it occupies; the cells
     beneath it stay empty.
   - A header broken across lines inside one cell is joined with a space; a word hyphenated at the
     line break is rejoined ("Dis- charge" -> "Discharge").
6. Blank spacer lines between groups of rows (every 5 days in a daily grid) are not typed.
   Summary rows inside the ruled table (Mean, Total, The year) are typed as body rows.
7. Layout is kept as printed. A table printed as side-by-side blocks ("Year | Discharge | Year |
   Discharge") is typed that wide, not stacked.
8. `label_cols`: 0-based column positions that hold row labels, joined by ";" (e.g. "0", or "0;3"
   for two side-by-side blocks). A value column's row label is the nearest label column to its left.
9. A rotated table is typed in its reading orientation.

## Conventions the typists applied (2026-11-14), to be confirmed by Anna
- Old typesetting leaves a gap after a decimal point or thousands comma ("11. 3", "1, 300"); typed
  without the gap ("11.3", "1,300").
- A unit line under the header rule (Feet., Second-feet., Inches.) is the last header row.
- Sub-headings inside a table (a water year, a quadrangle, a parameter) are body rows with empty
  value cells. Basin sub-headings between ruled blocks of one listing (648 p556) are not typed.
- Footnote superscripts are typed as plain characters attached to the cell ("a 8,230", "†464").
- Summary lines printed under a grid that do not follow its columns are not typed (641 p109).
- Superscripts and subscripts are typed as such (Anna, 2026-11-14, after checking 1110 p231):
  footnote marks as superscript characters attached to the cell ("⁴329", "5L3¹", "(²)", "ᵃ8,230"),
  chemical subscripts as subscripts ("HCO₃", "SiO₂"). Symbols (* † ‡) stay as printed. For
  scoring, a superscript footnote digit is a mark, never part of the number; in labels,
  super/subscript digits compare equal to plain digits.
