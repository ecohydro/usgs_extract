# Which well does this table belong to? (grading sample, 2026-11-15)

Built by `code/01_digitization/04_metadata_extraction/location_validation/validation/well_owner_review_sample.py`.

**Why:** in water-level reports each well's description ("7/34-25D1. City of Santa Barbara, well 9-D ...")
is printed above its table, and tables often run onto the next page. Two methods disagree on which
well a table belongs to for ~2,100 records statewide. Your answers decide which method to use, or
whether to drop the well link.

**How to grade `well_owner_review_sample.csv`** (42 rows):
1. Open `pngs/NN_page.png`. Find the table: `table_on_page` (e.g. "2 of 4" = the 2nd table on the
   page, top to bottom) and `table_first_rows` (the first rows as digitized).
2. Find the well description that belongs to that table. If the table continues from the previous
   page, open `pngs/NN_prev_page.png`.
3. In `answer` write **A**, **B**, **other** (put the well number in `true_well_if_other`), or
   **unclear**. Some rows show the same well as A and B; answer A if it is right.
4. `notes`: anything useful (e.g. "well listed in a separate section", "table is not a single well").

Don't open `well_owner_review_key.csv` until you're done: it says which method picked A and B.
Well numbers are written in short form (7/34-25D1 = T.7 N., R.34 W., section 25, tract D, well 1).
