"""Write briefs for the drafters and add finished drafts to the answer key.

  python code/05_evaluation/metadata/03_collect_drafts.py briefs test 1 20   # briefs for sample numbers 1..20
  python code/05_evaluation/metadata/03_collect_drafts.py collect test       # drafts/*.json -> answer_key.csv
(second word: test or validation)

A sample already in key/answer_key.csv is never touched, so edits made in the review page are safe.
`location` and `location_type` are filled from the evidence columns by schema.best_location.
"""
import csv
import json
import sys

import pandas as pd

import defaults as D
import schema as S

FILLED = [c for c in S.KEY_COLUMNS if c not in S.KEYS[:3] + ["location", "location_type"]]


def briefs(set_name, first, last):
    cfg = D.SETS[set_name]
    out = cfg["key_dir"] / "briefs"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(cfg["tables"])
    for r in s[(s.sample_no >= first) & (s.sample_no <= last)].itertuples():
        files = sorted((D.REDUCTO_CSV / str(r.doc_id)).glob(f"{r.doc_id}_page_{r.page_number}_table*.csv"))
        f = D.REDUCTO_CSV / str(r.doc_id) / f"{r.doc_id}_page_{r.page_number}_table{r.table_number}.csv"
        head = ""
        if f.exists():
            rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))
            show = lambda rr: "\n".join(" | ".join(c[:28] for c in row[:8]) for row in rr)
            head = show(rows[:4]) + f"\n... ({len(rows)} lines in all; last lines:)\n" + show(rows[-2:])
        (out / f"{r.sample_no}.txt").write_text(
            f"sample_no: {r.sample_no}\n"
            f"report (doc_id): {r.doc_id}   page: {r.page_number}   table on the page: {r.table_number} of {len(files)}\n"
            f"page scan (the ONLY page you may open): {cfg['pages']}/{r.doc_id}_page_{r.page_number}.png\n"
            f"write your answer to: {cfg['key_dir']}/drafts/{r.sample_no}.json\n\n"
            f"To recognise table {r.table_number} on the scan, these are the first lines of what the digitizer made "
            f"of it (may contain reading errors; do not copy values from here). The sampled table is exactly the part "
            f"of the page these lines cover, from its first to its last line (only the first 8 columns of each line "
            f"are shown here; the digitized table has {max(len(x) for x in rows) if f.exists() and rows else 0} columns):\n{head}\n", encoding="utf-8")
    print("briefs written to", out)


def collect(set_name):
    cfg = D.SETS[set_name]
    KEY, DRAFTS = cfg["key_dir"] / "answer_key.csv", cfg["key_dir"] / "drafts"
    KEY.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if KEY.exists():
        have = {int(r["sample_no"]) for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    s = pd.read_csv(cfg["tables"]).set_index("sample_no")
    new, bad = [], []
    for f in sorted(DRAFTS.glob("*.json"), key=lambda p: int(p.stem)):
        n = int(f.stem)
        if n in have:
            continue
        try:
            sites = json.load(open(f, encoding="utf-8"))
            assert isinstance(sites, list) and sites
        except Exception as e:
            bad.append((n, str(e)))
            continue
        for i, site in enumerate(sites, start=1):
            row = {"sample_no": n, "doc_id": s.doc_id[n], "page_number": s.page_number[n],
                   "table_number": s.table_number[n]}
            row.update({k: S.clean(site.get(k)) for k in FILLED})
            row["location_number"] = row["location_number"] or i
            row["location"], row["location_type"] = S.best_location(row)
            new.append(row)
    write_header = not KEY.exists()
    with open(KEY, "a", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, ["sample_no"] + S.KEY_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerows(new)
    print(f"added {len({r['sample_no'] for r in new})} samples ({len(new)} location rows); "
          f"{len(have)} already in the key; unreadable drafts: {bad}")


if __name__ == "__main__":
    if sys.argv[1] == "briefs":
        briefs(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        collect(sys.argv[2])
