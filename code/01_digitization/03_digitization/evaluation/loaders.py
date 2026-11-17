"""Load the truth grids and the three tool outputs for one page, all as plain grids
(list of rows of strings, header rows included, spans expanded, rectangular).

truth_tables(doc, page)  -> list of dicts {table_no, grid, n_header_rows, label_cols, title}
output_tables(system, doc, page) -> list of dicts {name, grid}
"""
import csv
import json
import re
import sys

import defaults as D

sys.path.insert(0, str(D.REPO / "code/02_inventory"))
from html_table import parse_table  # noqa: E402  (the repo's Reducto HTML parser)

_UNNAMED = re.compile(r"^Unnamed: \d+(_level_\d+)?$")


def rectangular(rows):
    w = max((len(r) for r in rows), default=0)
    return [list(r) + [""] * (w - len(r)) for r in rows]


def _html_grid(html):
    head, body = parse_table(html)
    return rectangular(head + body)


def truth_tables(doc, page):
    out = []
    meta = D.TRUTH / f"{doc}_page_{page}_tables.csv"
    for m in csv.DictReader(open(meta, encoding="utf-8")):
        f = D.TRUTH / f"{doc}_page_{page}_table{m['table_no']}.csv"
        grid = rectangular(list(csv.reader(open(f, encoding="utf-8"))))
        labels = [int(x) for x in str(m["label_cols"]).split(";") if x.strip() != ""]
        out.append({"table_no": int(m["table_no"]), "grid": grid, "title": m["title"],
                    "n_header_rows": int(m["n_header_rows"]), "label_cols": labels})
    return out


def reducto_raw(doc, page):
    f = D.REDUCTO_JSON / str(doc) / f"{doc}_page_{page}.json"
    if not f.exists():
        return []
    j = json.load(open(f, encoding="utf-8"))
    out = []
    for ch in j.get("result", {}).get("chunks", []):
        for b in ch.get("blocks", []):
            if b.get("type") == "Table" and b.get("content"):
                g = _html_grid(b["content"])
                if g:
                    out.append({"name": f"block{len(out) + 1}", "grid": g})
    return out


def reducto_csv(doc, page):
    """The published CSVs, read as a user's spreadsheet shows them. Two pandas artifacts are
    treated as blank / undone, in the tool's favour: 'Unnamed: 3_level_1' header cells and the
    '.1' suffix pandas adds to a repeated column name."""
    files = sorted((D.REDUCTO_CSV / str(doc)).glob(f"{doc}_page_{page}_table*.csv"),
                   key=lambda p: int(re.search(r"table(\d+)", p.name)[1]))
    out = []
    for f in files:
        rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))
        rows = [["" if _UNNAMED.match(c.strip()) else c for c in r] for r in rows]
        if rows:
            seen = set()
            for i, c in enumerate(rows[0]):
                m = re.match(r"^(.*)\.(\d+)$", c)
                if m and m[1] in seen:
                    rows[0][i] = m[1]
                seen.add(rows[0][i])
        g = rectangular(rows)
        if g:
            out.append({"name": f.name, "grid": g})
    return out


def paddle_raw(doc, page):
    out = []
    dirs = sorted(D.PADDLE.glob(f"{doc}_page_{page}_table_*"))
    for d in dirs:
        f = d / "res_0.txt"
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            res = j.get("res")
            if isinstance(res, dict) and res.get("html"):
                g = _html_grid(res["html"])
                if g:
                    out.append({"name": d.name, "grid": g})
    return out


def output_tables(system, doc, page):
    return {"paddle_raw": paddle_raw, "reducto_raw": reducto_raw, "reducto_csv": reducto_csv}[system](doc, page)
