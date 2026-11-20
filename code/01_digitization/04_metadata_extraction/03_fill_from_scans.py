"""Metadata read from the page scan for the tables the model run could not give (Anna, 2026-11-20).

  python 03_fill_from_scans.py briefs    -> scan_fill/briefs/{n}.txt, scan_fill/tables.csv; page scans copied to SCAN_FILL_PAGES
  (Claude Fable 5.2 helpers then follow evaluation/SCAN_FILL_INSTRUCTIONS.md and write scan_fill/drafts/{n}.json)
  python 03_fill_from_scans.py collect   -> scan_fill/rows.csv, which 02_build_metadata.py puts in the final table (status "ok",
                                            not marked; scan_fill/tables.csv is the record of which tables these are)

Tables: those in full_run/metadata_per_location.csv whose status is not "ok" (answer not readable, no answer for this
table) and the tables on the pages Reducto kept no text for (full_run/pages_without_table.csv; identified from their CSV).
The reader follows the answer-key rules (evaluation/KEY_FORMAT.md), as for the test set. Rows pass the same tidy step
and location hierarchy as the model's rows (pipeline.table_rows), without the found-on-page check (that check compares
with Reducto's text; these rows come from the scan).
"""
import csv
import html as htmllib
import json
import re
import shutil
import sys
from pathlib import Path

import defaults as D
import page_input
import pipeline
import schema as S

RUN = D.OUT / "full_run"
FILL = D.OUT / "scan_fill"
PAGES = D.SCAN_FILL_PAGES



def scan(doc, page):
    for root in D.PAGE_IMAGES:
        f = root / str(doc) / f"{doc}_page_{page}.png"
        if f.exists():
            return f
    f = D.REDUCTO_JSON_BY_PAGE / str(doc) / f"page_{page}" / f"{doc}_page_{page}.png"
    return f if f.exists() else None


def html_rows(h):
    rows = []
    for tr in re.findall(r"<tr\b.*?</tr>", h, flags=re.S | re.I):
        cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)
        rows.append([htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c))).strip() for c in cells])
    return rows


def table_lines(doc, page, k):
    tabs = [t for typ, t in page_input.page_blocks(doc, page) if typ == "Table"]
    if len(tabs) >= k:
        return html_rows(tabs[k - 1]), len(tabs)
    f = D.REDUCTO_JSON_BY_PAGE / str(doc) / f"page_{page}" / f"{doc}_page_{page}_table{k}.csv"
    if not f.exists():
        f = D.REDUCTO_CSV / str(doc) / f"{doc}_page_{page}_table{k}.csv"
    n = len(list(f.parent.glob(f"{doc}_page_{page}_table*.csv")))
    return (list(csv.reader(open(f, encoding="utf-8", errors="replace"))) if f.exists() else []), n


def targets():
    out = set()
    for r in csv.DictReader(open(RUN / "metadata_per_location.csv", encoding="utf-8")):
        if r["status"] != "ok":
            out.add((int(r["doc_id"]), int(r["page_number"]), int(r["table_number"])))
    for r in csv.DictReader(open(RUN / "pages_without_table.csv", encoding="utf-8")):
        doc, page = int(r["doc_id"].strip()), int(r["page_number"].strip())
        for f in (D.REDUCTO_JSON_BY_PAGE / str(doc) / f"page_{page}").glob(f"{doc}_page_{page}_table*.csv"):
            out.add((doc, page, int(re.search(r"_table(\d+)\.csv$", f.name).group(1))))
    return sorted(out)


def briefs():
    (FILL / "briefs").mkdir(parents=True, exist_ok=True)
    (FILL / "drafts").mkdir(exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    show = lambda rr: "\n".join(" | ".join(c[:28] for c in row[:8]) for row in rr)
    with open(FILL / "tables.csv", "w", newline="") as t:
        w = csv.writer(t)
        w.writerow(["n", "doc_id", "page_number", "table_number"])
        for n, (doc, page, k) in enumerate(targets(), start=1):
            w.writerow([n, doc, page, k])
            src = scan(doc, page)
            dst = PAGES / f"{doc}_page_{page}.png"
            if not dst.exists():
                shutil.copy(src, dst)
            rows, n_tabs = table_lines(doc, page, k)
            head = show(rows[:4]) + f"\n... ({len(rows)} lines in all; last lines:)\n" + show(rows[-2:]) if rows else "(no digitized lines)"
            (FILL / "briefs" / f"{n}.txt").write_text(
                f"n: {n}\n"
                f"report (doc_id): {doc}   page: {page}   table on the page: {k} of {n_tabs} (tables numbered top to bottom)\n"
                f"page scan (the ONLY page you may open): {dst}\n"
                f"write your answer to: {FILL}/drafts/{n}.json\n\n"
                f"To recognise table {k} on the scan, these are the first and last lines of what the digitizer made of it "
                f"(may contain reading errors; do not copy values from here; only the first 8 columns shown):\n{head}\n",
                encoding="utf-8")
    print("briefs:", n, "tables ->", FILL / "briefs")


def collect():
    tabs = list(csv.DictReader(open(FILL / "tables.csv")))
    D.REQUIRE_ON_PAGE = False
    out, missing = [], []
    for t in tabs:
        f = FILL / "drafts" / f"{t['n']}.json"
        try:
            locs = json.load(open(f, encoding="utf-8"))
            assert isinstance(locs, list) and locs
        except Exception as e:
            missing.append((t["n"], str(e)[:60]))
            continue
        for r in pipeline.table_rows(int(t["doc_id"]), int(t["page_number"]), int(t["table_number"]), locs, ""):
            r["n_locations_in_table"] = len(locs)
            out.append(r)
    cols = S.KEYS + ["n_locations_in_table"] + S.GRADED + S.EVIDENCE + S.EVIDENCE_EXTRA + ["evidence_place"]
    with open(FILL / "rows.csv", "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{len(tabs) - len(missing)} of {len(tabs)} tables, {len(out)} rows -> {FILL / 'rows.csv'}; missing: {missing}")


if __name__ == "__main__":
    {"briefs": briefs, "collect": collect}[sys.argv[1]]()
