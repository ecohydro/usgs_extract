"""Final metadata table from the full run: one row per location per table, for every page that was read.

  python 02_build_metadata.py        -> full_run/metadata_per_location.csv (+ full_run/metadata_run_summary.csv)

Each table of each page goes through pipeline.table_rows, the same function the evaluation sets are scored with.
A page whose answer could not be read gives one blank row per table with status "answer not readable".
on_published_list: whether the table is on data/analysis/table_level_metadata_published.csv (the June 2026 list);
pages read from 2026-11-18 on (01_full_run.py prepare_unlisted) hold tables that are not.
NOTE: the file covers every page, including pages on tribal lands; Annette's tribal screen is applied after this.
"""
import csv
import gzip
import json

import defaults as D
import pipeline
import schema as S

RUN = D.OUT / "full_run"
SCAN_STATUS = "ok"   # rows from 03_fill_from_scans.py are not marked (Anna 2026-11-20); the record is scan_fill/tables.csv
COLUMNS = S.KEYS + ["n_locations_in_table"] + S.GRADED + S.EVIDENCE + S.EVIDENCE_EXTRA + ["evidence_place", "status", "on_published_list"]


def main():
    n = dict(pages=0, tables=0, rows=0, unreadable_pages=0, cut_off_pages=0, tables_without_answer=0)
    shards = sorted((RUN / "in").glob("shard_*.jsonl.gz"))
    missing = [s.name for s in shards if not (RUN / "out" / (s.name[:-9] + ".jsonl")).exists()]
    again = {}
    if (RUN / "reread" / "answers.jsonl").exists():      # second pass with a longer answer limit
        again = {(a["doc_id"], a["page_number"]): a for a in map(json.loads, open(RUN / "reread" / "answers.jsonl", encoding="utf-8"))
                 if a["parsed"] is not None}
    n["pages_fixed_by_reread"] = 0
    listed = {(int(r["doc_id"]), int(r["page_number"]), int(r["table_number"]))
              for r in csv.DictReader(open(D.PUBLISHED_TABLES))}
    n["tables_not_on_published_list"] = 0
    scan = {}                                             # 03_fill_from_scans.py: tables the model could not give
    if (D.OUT / "scan_fill" / "rows.csv").exists():
        for r in csv.DictReader(open(D.OUT / "scan_fill" / "rows.csv", encoding="utf-8")):
            scan.setdefault((int(r["doc_id"]), int(r["page_number"]), int(r["table_number"])), []).append(r)
    n["tables_read_from_scan"] = 0
    with open(RUN / "metadata_per_location.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in shards:
            out = RUN / "out" / (s.name[:-9] + ".jsonl")
            if not out.exists():
                continue
            text = {(d, p): t for d, p, k, t in map(json.loads, gzip.open(s, "rt", encoding="utf-8"))}
            for a in map(json.loads, open(out, encoding="utf-8")):
                doc, page = a["doc_id"], a["page_number"]
                if a["parsed"] is None and (doc, page) in again:
                    a = again[(doc, page)]
                    n["pages_fixed_by_reread"] += 1
                norm = pipeline.page_norm(text[(doc, page)])
                n["pages"] += 1
                n["unreadable_pages"] += a["parsed"] is None
                n["cut_off_pages"] += a["finish"] != "stop"
                for k in range(1, a["n_tables"] + 1):
                    locs = pipeline.table_locations(a["parsed"], k)
                    status = "ok" if locs else ("answer not readable" if a["parsed"] is None else "no answer for this table")
                    n["tables"] += 1
                    n["tables_without_answer"] += not locs
                    on_list = (doc, page, k) in listed
                    n["tables_not_on_published_list"] += not on_list
                    if not locs and (doc, page, k) in scan:
                        n["tables_read_from_scan"] += 1
                        for r in scan.pop((doc, page, k)):
                            w.writerow({**r, "status": SCAN_STATUS, "on_published_list": on_list})
                            n["rows"] += 1
                        continue
                    for r in pipeline.table_rows(doc, page, k, locs, norm):
                        w.writerow({**r, "status": status, "on_published_list": on_list})
                        n["rows"] += 1
        for (doc, page, k), rows in sorted(scan.items()):  # tables on pages Reducto kept no text for
            n["tables"] += 1
            n["tables_read_from_scan"] += 1
            on_list = (doc, page, k) in listed
            n["tables_not_on_published_list"] += not on_list
            for r in rows:
                w.writerow({**r, "status": SCAN_STATUS, "on_published_list": on_list})
                n["rows"] += 1
    n["shards_missing"] = len(missing)
    with open(RUN / "metadata_run_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([["what", "n"]] + [[k, v] for k, v in n.items()])
    print(n)


if __name__ == "__main__":
    main()
