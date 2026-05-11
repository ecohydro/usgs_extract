#!/usr/bin/env python3
"""
add_sb_metadata.py

Adds LLM-extracted metadata for the 7 SB County documents to:

  1. Per-page metadata CSVs in data/digitized/{index_id}/page_{N}/
         {index_id}_page_{N}_metadata.csv
     Written only for pages that exist in both the metadata file and the
     organized folder structure.

  2. The main metadata CSV:
         Data_Files/cleaned_metadata_final - Copy.csv
     The SB rows are appended with id updated to the USGS Index ID so they
     are consistent with the folder naming convention.

Source metadata: Data_Files/sb_data_updates/final_sbmetadata.csv
     Uses old IDs (sb, sb2-sb7); sb1 is already in the main pipeline as
     pub ID 23995 and is skipped here.

ID mapping (old -> USGS Index ID used as folder name):
    sb  -> ofr60102    sb2 -> ofr5983    sb3 -> ofr6196
    sb4 -> ofr49118    sb5 -> ofr64117   sb6 -> ofr6291
    sb7 -> ofr63103
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[3]
OUTPUT_ROOT  = REPO_ROOT / "data" / "digitized"
SB_META_CSV  = REPO_ROOT / "Data_Files" / "sb_data_updates" / "final_sbmetadata.csv"
MAIN_META    = REPO_ROOT / "Data_Files" / "cleaned_metadata_final - Copy.csv"

ID_MAP = {
    "sb":  "ofr60102",
    "sb2": "ofr5983",
    "sb3": "ofr6196",
    "sb4": "ofr49118",
    "sb5": "ofr64117",
    "sb6": "ofr6291",
    "sb7": "ofr63103",
}


def main() -> None:
    # Load SB metadata, grouped by (old_id, page_number)
    sb_rows: dict[tuple, list] = defaultdict(list)
    with open(SB_META_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old_id = row["id"].strip()
            if old_id not in ID_MAP:
                continue  # skip sb1 and anything unexpected
            sb_rows[(old_id, row["page_number"].strip())].append(row)

    cols = list(next(iter(sb_rows.values()))[0].keys())  # column order from source

    stats = {"pages_written": 0, "pages_no_folder": 0, "rows_appended": 0}

    # ── 1. Per-page metadata CSVs ─────────────────────────────────────────────
    for (old_id, page_num), rows in sorted(sb_rows.items()):
        new_id  = ID_MAP[old_id]
        page_dir = OUTPUT_ROOT / new_id / f"page_{page_num}"

        if not page_dir.exists():
            stats["pages_no_folder"] += 1
            continue

        out_path = page_dir / f"{new_id}_page_{page_num}_metadata.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                updated = dict(r)
                updated["id"] = new_id   # use index ID, not old sb* label
                writer.writerow(updated)

        stats["pages_written"] += 1

    print(f"Per-page metadata CSVs written: {stats['pages_written']}")
    print(f"Skipped (no folder):            {stats['pages_no_folder']}")

    # ── 2. Append to main metadata CSV ─────────────────────────────────────
    # Read existing columns from main metadata to preserve header order
    with open(MAIN_META, encoding="utf-8") as f:
        main_cols = csv.DictReader(f).fieldnames

    # Build flat list of all SB rows with updated IDs
    all_sb_rows = []
    for (old_id, page_num), rows in sorted(sb_rows.items()):
        new_id = ID_MAP[old_id]
        for r in rows:
            updated = dict(r)
            updated["id"] = new_id
            all_sb_rows.append(updated)

    with open(MAIN_META, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=main_cols, extrasaction="ignore")
        for r in all_sb_rows:
            writer.writerow(r)
        stats["rows_appended"] = len(all_sb_rows)

    print(f"Rows appended to main metadata: {stats['rows_appended']}")
    print("Done.")


if __name__ == "__main__":
    main()
