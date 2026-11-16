"""
Which well does a water-level table belong to? (2026-11-15; used by 07_well_numbers.py)

Water-level reports print each well's DESCRIPTION ("7/34-25D1. City of Santa Barbara, well 9-D ...")
above its table, and tables run onto the next page. 07 matches a metadata row to a well printed on the
same page, so a table continued from the previous page could get the next well's number (doc 2060 p203).

Table owner = the last well description before the table in reading order (Reducto block order,
across pages). description block = non-table block whose text STARTS with a well number.
"single-well" table = a 'water level' title since the description, or a header with 'date' + 'level';
tables whose header has 'owner' are inventories and skipped.
Reducto also merges several printed well tables into one table, with the later wells' headings inside
it ("7N/35W-30G1 --Continued", item 21), and some "single-well" tables are really one-row-per-well
lists. So each table also carries the well numbers printed INSIDE it.

Rule (graded sample, data/analysis/manual_review/well_owner_check/, Annette 2026-11-15):
  agree             row's well = table owner                                  -> keep
  in_table_llm      row's well printed inside the table and named in the LLM text
                    (4/4 text check; sample items 7, 21, 39)                   -> keep
  in_table_other    row's well printed inside the table but matched only as 'only well on page' or by
                    owner name (one-row-per-well lists with short ids "6Q1 7D1": 3/4 wrong) -> unlink
  owner_same_page   owner described on the SAME page, row has another well (8/8 rule right) -> reassign
  owner_earlier     owner described only on an earlier page (neither method right in the graded items),
                    or several owners                                          -> unlink
"""
import re

import pandas as pd

TRACT = r'[A-HJ-NP-R8]'
RE_WELL = re.compile(r'(?<![\d/.])(\d{1,2})\s?([NS5])?\s?/\s?(\d{1,2})\s?([EWF])?\s?[-–—]\s?'
                     r'(\d{1,2})\s?(' + TRACT + r')\s?(\d{1,3})(?:\s?([MSH])(?![a-z]))?')
RE_WL_TITLE = re.compile(r'water[\s-]*levels?', re.I)
RE_TAG = re.compile(r'<[^>]+>')


def _key(m):
    t, _, r, _, sec, tract, seq, _ = m.groups()
    if not (1 <= int(t) <= 60 and 1 <= int(r) <= 60 and 1 <= int(sec) <= 36):
        return None
    return (int(t), int(r), int(sec), 'B' if tract == '8' else tract, int(seq))   # = 07 well_key


def lead_well(text):
    m = RE_WELL.search(text[:40])
    return _key(m) if m and m.start() <= 12 else None


def wells_in(text):
    return {k for k in (_key(m) for m in RE_WELL.finditer(RE_TAG.sub(' ', text))) if k}


def table_owners(page_blocks):
    """(doc, page, table_idx) -> dict(owner, owner_page, inside) for single-well tables."""
    b = page_blocks.sort_values(['id', 'page_number', 'block_order'])
    out = {}
    for doc, d in b.groupby('id', sort=False):
        owner, wl_title = None, False
        for r in d.itertuples(index=False):
            c = str(r.content)
            if r.block_type != 'Table':
                w = lead_well(c)
                if w:
                    owner, wl_title = (w, r.page_number), False
                elif RE_WL_TITLE.search(c[:200]):
                    wl_title = True
                continue
            head = c[:600].lower()
            if owner and 'owner' not in head and (wl_title or ('date' in head and 'level' in head)):
                out[(int(doc), int(r.page_number), int(r.table_idx))] = {
                    'owner': owner[0], 'owner_page': int(owner[1]), 'inside': wells_in(c)}
    return out


def check(owners, doc, page, tables, pick_key, how):
    """Verdict for a row matched to well `pick_key` by rule `how`.
    tables = the row's matched table numbers (or all tables on the page). Returns (verdict, new_key)."""
    info = [owners[(doc, page, t)] for t in tables if (doc, page, t) in owners]
    if not info:
        return None, None
    own = {i['owner'] for i in info}
    inside = set().union(*(i['inside'] for i in info))
    if pick_key in own:
        return 'agree', None
    if pick_key in inside:
        return ('in_table_llm', None) if how == 'well number in LLM text' else ('in_table_other', None)
    if len(own) == 1 and all(i['owner_page'] == page for i in info):
        return 'owner_same_page', next(iter(own))
    return 'owner_earlier', None


def matched_tables(root):
    """meta_row -> list of table numbers the row is tied to (15_entry_dates.py)."""
    e = pd.read_csv(root / 'data/analysis/entry_dates.csv', usecols=['meta_row', 'matched_tables'],
                    low_memory=False).dropna()
    return {int(m): [int(float(x)) for x in str(t).split(',') if x.strip() not in ('', 'nan')]
            for m, t in zip(e.meta_row, e.matched_tables)}
