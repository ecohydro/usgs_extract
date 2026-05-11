#!/usr/bin/env python3
"""
organize_sb_pages.py

Reorganizes SB County documents in data/digitized/ from the flat page_full
structure into the standard per-page folder structure:

    data/digitized/{index_id}/page_{N}/
        {index_id}_page_{N}.json          ← chunks belonging to page N
        {index_id}_page_{N}_table{M}.csv  ← tables on page N, re-numbered from 1

Source: the Reducto full-document JSON for each SB document contains per-block
page numbers in bbox.page. The N-th Table block (1-indexed, across all chunks)
corresponds to {index_id}_page_full_table{N}.csv. This script reads those page
numbers and reorganises accordingly.

Documents processed (sb1 = pub ID 23995 is already in the main pipeline):
    ofr5983  (sb2, 1958)    ofr6196  (sb3, 1960)    ofr49118 (sb4, Tecolote)
    ofr64117 (sb5, 1963)    ofr6291  (sb6, 1961)    ofr63103 (sb7, 1962)
    ofr60102 (unnumbered sb, 1959)
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "data" / "digitized"

SB_DOCS = [
    "ofr5983", "ofr6196", "ofr49118",
    "ofr64117", "ofr6291", "ofr63103", "ofr60102",
]


def get_table_block_pages(chunks: list) -> dict:
    """
    Walk chunks in order, counting Table blocks sequentially (1-indexed).
    Returns dict: table_number -> page_number.
    """
    table_num = 0
    mapping = {}
    for chunk in chunks:
        for block in chunk.get("blocks", []):
            if block.get("type") == "Table":
                table_num += 1
                mapping[table_num] = block["bbox"]["page"]
    return mapping


def make_page_json(full_data: dict, page_num: int) -> dict:
    """
    Build a per-page JSON from the full-document JSON.
    Includes chunks where the minimum block page equals page_num.
    """
    chunks = full_data.get("result", {}).get("chunks", [])
    page_chunks = []
    for chunk in chunks:
        blocks = chunk.get("blocks", [])
        if not blocks:
            continue
        min_page = min(b["bbox"]["page"] for b in blocks)
        if min_page == page_num:
            page_chunks.append(chunk)

    return {
        "job_id":   full_data.get("job_id"),
        "duration": full_data.get("duration"),
        "pdf_url":  full_data.get("pdf_url"),
        "usage":    {"num_pages": 1},
        "result":   {"type": "full", "chunks": page_chunks},
    }


def organize_doc(index_id: str) -> dict:
    doc_dir = OUTPUT_ROOT / index_id
    json_path = doc_dir / f"{index_id}_page_full.json"

    if not json_path.exists():
        return {"error": f"{json_path} not found"}

    with open(json_path, encoding="utf-8") as f:
        full_data = json.load(f)

    chunks = full_data["result"]["chunks"]
    table_page_map = get_table_block_pages(chunks)   # table_num -> page_num

    # Existing CSVs: extract table number from filename
    csv_files = {
        int(p.stem.split("_table")[1]): p
        for p in doc_dir.glob(f"{index_id}_page_full_table*.csv")
    }

    # Group CSVs by page, preserving document-order for per-page re-numbering
    pages_tables: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    skipped = []
    for tbl_num, csv_path in sorted(csv_files.items()):
        page = table_page_map.get(tbl_num)
        if page is None:
            skipped.append(tbl_num)
            continue
        pages_tables[page].append((tbl_num, csv_path))

    stats = {"pages_created": 0, "csvs_moved": 0, "csvs_skipped": len(skipped)}

    for page_num, table_list in sorted(pages_tables.items()):
        page_dir = doc_dir / f"page_{page_num}"
        page_dir.mkdir(exist_ok=True)

        # Per-page JSON
        page_json = make_page_json(full_data, page_num)
        with open(page_dir / f"{index_id}_page_{page_num}.json", "w", encoding="utf-8") as f:
            json.dump(page_json, f)

        # CSVs: re-number from 1 within this page
        for local_num, (_, csv_src) in enumerate(table_list, start=1):
            dst = page_dir / f"{index_id}_page_{page_num}_table{local_num}.csv"
            shutil.copy2(csv_src, dst)
            stats["csvs_moved"] += 1

        stats["pages_created"] += 1

    # Remove old page_full files now that everything is reorganised
    json_path.unlink()
    for csv_path in csv_files.values():
        csv_path.unlink()

    return stats


def main():
    for index_id in SB_DOCS:
        print(f"{index_id}...", end=" ", flush=True)
        result = organize_doc(index_id)
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(
                f"{result['pages_created']} pages, "
                f"{result['csvs_moved']} CSVs moved"
                + (f", {result['csvs_skipped']} CSVs skipped (no table match)" if result['csvs_skipped'] else "")
            )


if __name__ == "__main__":
    main()
