"""
Turn Annette's graded earliest-records sheet (2026-11-19) into the review file the pipeline applies.

  python earliest_review_to_actions.py
    in:  data/analysis/manual_review/earliest_records_new/earliest_pages_review.csv  (Annette; not changed)
    out: data/analysis/manual_review/earliest_records_review.csv  (read by paper_filters.apply_review)

Annette graded 31 of 121 pages (every water type), writing NOTES rather than verdict words. Each note
is turned into an action here, with the reason; pages she did not grade get no action.

Page-level rows (table_number blank) apply to every record of that page AND water type.
Record-level rows (table_number + location_number) apply to that one record only; used where only some
records on the page were wrong (the sheet's year_min / year_max span all records of the page).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MR = ROOT / 'data/analysis/manual_review'
SHEET = MR / 'earliest_records_new/earliest_pages_review.csv'
OUT = MR / 'earliest_records_review.csv'
REVIEWER = 'Annette Hilton 2026-11-19 (actions transcribed by Claude)'

# page-level decisions from Annette's notes, keyed by review_no
PAGE = {
    1: ('construction_year', "note: dates are 'year completed' (drill date), not measurement dates"),
    2: ('construction_year', "note: dates are 'year completed' (drill date), not measurement dates"),
    6: ('keep', "note: dates are real; 'stream discharge' written in new_water_type: the 1889 record "
                "(Santa Paula Creek, loc 3) is already Stream Discharge; the table is Irrigation by "
                "majority (8 of 13 sites are ditches/canals), no change"),
    12: ('keep', 'note: dates are real for first table'),
    49: ('keep', 'note: table 1 (fig 1) 1861-1904 is correct; page range 1861-1912 written in the '
                 'shifted columns covers all three tables, each already dated on its own'),
}
# any other graded page whose note says the dates are correct / real -> keep

# record-level corrections (read from the page images, 2026-11-19)
RECORDS = [
    # 2246 p511 (WSP, Santa Clara R. misc. measurements): 'Aug. 11, 1898' read as 1808 (Piru Creek at French house)
    (2246, 511, 'Stream Discharge', 1, 13, 'set_year', 1898, 1898,
     "note: '1808' probably OCR; page prints Aug. 11, 1898"),
]
# 52651 p29 (springs): the model took years from the References column; Annette: use 'Date of observation'
for loc, y in [(1, 1966), (2, 1966), (3, 1965), (4, 1954), (5, 1954)]:
    RECORDS.append((52651, 29, 'Springs', 1, loc, 'set_year', y, y,
                    "note: use 'date of observation' column (model used reference years)"))
# 2447 p66 (flood maxima, out of state): the model used 'Period of known floods'; Annette: use the 'Date'
# column. The 26 records follow the printed rows in order (their periods match the printed periods);
# each gets the year of its printed maximum. Record 7 = Colorado R nr Topock, printed 'g 1862'.
FLOOD = {1: 1914, 2: 1914, 3: 1956, 4: 1960, 5: 1941, 6: 1957, 7: 1862, 8: 1941, 9: 1951, 10: 1939,
         11: 1891, 12: 1941, 13: 1949, 14: 1941, 15: 1941, 16: 1941, 17: 1941, 18: 1941, 19: 1960,
         20: 1955, 21: 1960, 22: 1956, 23: 1949, 24: 1906, 25: 1952, 26: 1952}
for loc, y in FLOOD.items():
    RECORDS.append((2447, 66, 'Stream Discharge', 1, loc, 'set_year', y, y,
                    "note: use 'date' column (year of the printed maximum, not the period of known floods)"))


def main():
    s = pd.read_csv(SHEET, dtype=str)
    graded = s[s['note'].notna() | s['verdict'].notna()].copy()
    rows = []
    for _, g in graded.iterrows():
        no = int(g.review_no)
        if no in PAGE:
            act, why = PAGE[no]
        elif any(int(g.id) == r[0] and int(g.page_number) == r[1] for r in RECORDS):
            continue                                   # handled record by record
        elif str(g.note).strip().lower().startswith(('dates are correct', 'dates are real', '1875 date is real',
                                                      '1879 date is real')):
            act, why = 'keep', 'note: ' + str(g.note).strip()
        else:
            raise ValueError(f'review_no {no}: note not mapped: {g.note!r}')
        rows.append(dict(id=int(g.id), page_number=int(g.page_number), water_type=g.water_type,
                         table_number=pd.NA, location_number=pd.NA, action=act, new_water_type=pd.NA,
                         corrected_year_start=pd.NA, corrected_year_end=pd.NA,
                         reviewer=REVIEWER, note=why, review_no=no))
    for d, pg, wt, t, loc, act, ys, ye, why in RECORDS:
        no = s.loc[(s.id.astype(int) == d) & (s.page_number.astype(int) == pg), 'review_no']
        rows.append(dict(id=d, page_number=pg, water_type=wt, table_number=t, location_number=loc, action=act,
                         new_water_type=pd.NA, corrected_year_start=ys, corrected_year_end=ye,
                         reviewer=REVIEWER, note=why, review_no=int(no.iat[0]) if len(no) else pd.NA))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'{len(graded)} graded pages -> {len(out)} review rows -> {OUT}')
    print(out.groupby([out.table_number.notna().map({True: 'record', False: 'page'}), 'action']).size().to_string())


if __name__ == '__main__':
    main()
