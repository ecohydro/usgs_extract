#!/usr/bin/env python3
"""
extract_ofr60102_missing.py

Extracts 16 pages from ofr60102 (sb) that have structured HTML Table blocks
in the Reducto JSON but were never converted to CSV or organized into folders.

For each missing page:
  - Creates data/digitized/ofr60102/page_{N}/
  - Writes per-page JSON  (ofr60102_page_{N}.json)
  - Writes table CSVs    (ofr60102_page_{N}_table{M}.csv)
  - Writes metadata CSV  (ofr60102_page_{N}_metadata.csv) if available

Source JSON: Data_Files/sb_data_updates/Scanned_SB/Scanned_SB/sb_page_full.json
Metadata:    Data_Files/sb_data_updates/final_sbmetadata.csv
"""

import csv
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "data" / "digitized"
SB_JSON     = REPO_ROOT / "Data_Files" / "sb_data_updates" / "Scanned_SB" / "Scanned_SB" / "sb_page_full.json"
SB_META_CSV = REPO_ROOT / "Data_Files" / "sb_data_updates" / "final_sbmetadata.csv"
DOC_ID      = "ofr60102"


class _TableParser(HTMLParser):
    """Extract rows of cell-text from an HTML table string."""
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append(" ".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr":
            if any(c for c in self._row):   # skip empty rows
                self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            cleaned = data.strip()
            if cleaned:
                self._cell.append(cleaned)


def html_table_to_rows(html_content: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(html_content)
    return parser.rows


def make_page_json(full_data: dict, page_num: int) -> dict:
    chunks = full_data.get("result", {}).get("chunks", [])
    page_chunks = []
    for chunk in chunks:
        blocks = chunk.get("blocks", [])
        if not blocks:
            continue
        if min(b["bbox"]["page"] for b in blocks) == page_num:
            page_chunks.append(chunk)
    return {
        "usage":  {"num_pages": 1},
        "result": {"type": "full", "chunks": page_chunks},
    }


def main():
    with open(SB_JSON, encoding="utf-8") as f:
        full_data = json.load(f)
    chunks = full_data["result"]["chunks"]

    # Existing folders
    doc_dir = OUTPUT_ROOT / DOC_ID
    existing_pages = {int(p.name.replace("page_", "")) for p in doc_dir.iterdir() if p.is_dir()}

    # Pages in JSON that have Table blocks and no folder yet
    pages_tables: dict[int, list[str]] = defaultdict(list)
    for chunk in chunks:
        for block in chunk.get("blocks", []):
            if block["type"] == "Table":
                pages_tables[block["bbox"]["page"]].append(block["content"])

    missing_pages = sorted(pages_tables.keys() - existing_pages)
    print(f"Pages to extract: {missing_pages}")

    # Load SB metadata for ofr60102 (stored under old id 'sb')
    meta_by_page: dict[str, list[dict]] = defaultdict(list)
    meta_cols = None
    with open(SB_META_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        meta_cols = reader.fieldnames
        for row in reader:
            if row["id"].strip() == "sb":
                meta_by_page[row["page_number"].strip()].append(row)

    stats = {"pages": 0, "csvs": 0, "meta_written": 0}

    for page_num in missing_pages:
        page_dir = doc_dir / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        # Per-page JSON (no job_id/duration/pdf_url — matches stripped format)
        page_json = make_page_json(full_data, page_num)
        with open(page_dir / f"{DOC_ID}_page_{page_num}.json", "w", encoding="utf-8") as f:
            json.dump(page_json, f)

        # Table CSVs
        for local_num, html_content in enumerate(pages_tables[page_num], start=1):
            rows = html_table_to_rows(html_content)
            if not rows:
                continue
            # Pad all rows to the same width
            width = max(len(r) for r in rows)
            csv_path = page_dir / f"{DOC_ID}_page_{page_num}_table{local_num}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row + [""] * (width - len(row)))
            stats["csvs"] += 1

        # Metadata CSV
        meta_rows = meta_by_page.get(str(page_num), [])
        if meta_rows and meta_cols:
            meta_out = page_dir / f"{DOC_ID}_page_{page_num}_metadata.csv"
            with open(meta_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=meta_cols, extrasaction="ignore")
                writer.writeheader()
                for r in meta_rows:
                    updated = dict(r)
                    updated["id"] = DOC_ID
                    writer.writerow(updated)
            stats["meta_written"] += 1

        stats["pages"] += 1

    print(f"Done.")
    print(f"  Pages created:   {stats['pages']}")
    print(f"  CSVs written:    {stats['csvs']}")
    print(f"  Metadata CSVs:   {stats['meta_written']}")


if __name__ == "__main__":
    main()
