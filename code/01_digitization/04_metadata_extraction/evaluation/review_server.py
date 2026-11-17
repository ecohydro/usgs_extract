"""Review page for the metadata answer key: page scan on the left, key rows on the right.

Edits are saved straight into key/answer_key.csv (the single source) and logged, with old and new
value, in key/edit_log.csv. "Checked" marks go to key/checked.csv.
Run:  ~/miniconda3/envs/smh_irr_labels_2/bin/python code/01_digitization/04_metadata_extraction/review_server.py [port] [test|validation]
then open http://localhost:<port>/ (VS Code forwards the port).
"""
import csv
import datetime
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # shared modules are in the pipeline folder
import defaults as D
import schema as S

HERE = Path(__file__).resolve().parent
SET = D.SETS[sys.argv[2] if len(sys.argv) > 2 else "test"]
KEY = SET["key_dir"] / "answer_key.csv"
LOG = SET["key_dir"] / "edit_log.csv"
CHECKED = SET["key_dir"] / "checked.csv"
FIELDS = [c for c in S.KEY_COLUMNS if c not in S.KEYS]   # editable; location / location_type are derived


def read_csv(path):
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def write_key(rows):
    cols = ["sample_no"] + S.KEY_COLUMNS
    with open(KEY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def append(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def checked_set():
    state = {}
    for r in read_csv(CHECKED):
        state[int(r["sample_no"])] = r["checked"] == "True"
    return {n for n, v in state.items() if v}


def table_preview(doc, page, k, n=6):
    f = D.REDUCTO_CSV / str(doc) / f"{doc}_page_{page}_table{k}.csv"
    if not f.exists():
        return []
    rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))
    return [[c[:30] for c in r[:10]] for r in rows[:n]]


def state():
    samples = read_csv(SET["tables"])
    key = read_csv(KEY)
    done = checked_set()
    drafted = {int(r["sample_no"]) for r in key}
    out = []
    for s in samples:
        n = int(s["sample_no"])
        out.append({"sample_no": n, "stratum": s["stratum"], "doc_id": s["doc_id"], "page_number": s["page_number"],
                    "table_number": s["table_number"], "drafted": n in drafted, "checked": n in done})
    return out


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self.send((HERE / "review_page.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/list":
            return self.send(state())
        if p.startswith("/api/sample/"):
            n = int(p.rsplit("/", 1)[1])
            s = next(r for r in read_csv(SET["tables"]) if int(r["sample_no"]) == n)
            sites = [r for r in read_csv(KEY) if int(r["sample_no"]) == n]
            return self.send({"sample": s, "sites": sites, "schema": {"data_types": S.DATA_TYPES, "resolutions": S.TEMPORAL_RESOLUTIONS, "variables": S.VARIABLES, "places": S.EVIDENCE_PLACES}, "checked": n in checked_set(),
                              "preview": table_preview(s["doc_id"], s["page_number"], s["table_number"])})
        if p.startswith("/pages/") and p.endswith(".png") and "/" not in p[7:]:
            f = SET["pages"] / p[7:]
            if f.exists():
                return self.send(f.read_bytes(), "image/png")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            now = datetime.datetime.now().isoformat(timespec="seconds")
            n = int(d["sample_no"])
            rows = read_csv(KEY)
            mine = [r for r in rows if int(r["sample_no"]) == n]
            path = "/" + self.path.split("?")[0].rsplit("/", 1)[-1]
            if path == "/save":
                site, field, value = int(d["location_number"]), d["field"], str(d["value"]).strip()
                assert field in FIELDS and field not in ("location", "location_type")
                r = next(r for r in mine if int(r["location_number"]) == site)
                old = r[field]
                if old != value:
                    r[field] = value
                    r["location"], r["location_type"] = S.best_location(r)
                    write_key(rows)
                    append(LOG, ["time", "sample_no", "location_number", "field", "old", "new"], [now, n, site, field, old, value])
                return self.send({"ok": True, "location": r["location"], "location_type": r["location_type"]})
            elif path == "/add_site":
                s = next(r for r in read_csv(SET["tables"]) if int(r["sample_no"]) == n)
                base = dict(mine[-1]) if mine else {f: "" for f in FIELDS}
                new = {**base, "sample_no": n, "doc_id": s["doc_id"], "page_number": s["page_number"],
                       "table_number": s["table_number"], "location_number": max([int(r["location_number"]) for r in mine] + [0]) + 1}
                for f in S.EVIDENCE + S.EVIDENCE_EXTRA + ["location", "location_type", "water_source", "n_measurements",
                                                          "n_variables", "construction_date", "notes"]:
                    new[f] = ""
                last = max([i for i, r in enumerate(rows) if int(r["sample_no"]) == n] + [len(rows) - 1])
                rows.insert(last + 1, new)
                write_key(rows)
                append(LOG, ["time", "sample_no", "location_number", "field", "old", "new"], [now, n, new["location_number"], "(site added)", "", ""])
            elif path == "/delete_site":
                site = int(d["location_number"])
                rows = [r for r in rows if not (int(r["sample_no"]) == n and int(r["location_number"]) == site)]
                write_key(rows)
                append(LOG, ["time", "sample_no", "location_number", "field", "old", "new"], [now, n, site, "(site deleted)", "", ""])
            elif path == "/checked":
                append(CHECKED, ["time", "sample_no", "checked"], [now, n, bool(d["checked"])])
            else:
                raise ValueError("unknown path")
            self.send({"ok": True})
        except Exception as e:   # the page shows the field in red if a save fails
            self.send({"ok": False, "error": str(e)}, code=500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else D.REVIEW_PORT
    print(f"review page: http://localhost:{port}/")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
