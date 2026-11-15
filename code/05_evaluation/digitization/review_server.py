"""Serve the review pages and save Anna's edits straight into the truth files.

GET   anything under data/analysis/evaluation/digitization/ (review/, pages/)
POST  /save   {"doc":..,"page":..,"table":..,"row":..,"col":..,"value":".."}
        -> rewrites that cell of truth/{doc}_page_{page}_table{table}.csv and appends the change
           (old value, new value, time) to review/edit_log.csv
POST  /checked {"doc":..,"page":..,"checked":true|false} -> review/checked_pages.csv
The truth CSVs stay the single source; edit_log.csv is the record of what the human check changed.
Run:  ~/miniconda3/envs/geofm-vlm/bin/python code/05_evaluation/digitization/review_server.py [port]
"""
import csv
import datetime
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import defaults as D

LOG = D.OUT / "review" / "edit_log.csv"
CHECKED = D.OUT / "review" / "checked_pages.csv"


def append(path, header, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # always show the current page and scan
        super().end_headers()

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            now = datetime.datetime.now().isoformat(timespec="seconds")
            doc, page = int(d["doc"]), int(d["page"])
            if self.path == "/save":
                k, i, j, value = int(d["table"]), int(d["row"]), int(d["col"]), str(d["value"]).strip()
                f = D.TRUTH / f"{doc}_page_{page}_table{k}.csv"
                g = list(csv.reader(open(f, encoding="utf-8")))
                old = g[i][j]
                if old != value:
                    g[i][j] = value
                    with open(f, "w", newline="", encoding="utf-8") as out:
                        csv.writer(out).writerows(g)
                    append(LOG, ["time", "doc_id", "page_number", "table_no", "row", "col", "old", "new"],
                           [now, doc, page, k, i, j, old, value])
            elif self.path == "/checked":
                append(CHECKED, ["time", "doc_id", "page_number", "checked"], [now, doc, page, bool(d["checked"])])
            else:
                raise ValueError("unknown path")
            body = b'{"ok": true}'
            self.send_response(200)
        except Exception as e:  # the page shows the cell in red if a save fails
            body = json.dumps({"ok": False, "error": str(e)}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else D.REVIEW_PORT
    print(f"review pages on http://127.0.0.1:{port}/review/index.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(D.OUT))).serve_forever()
