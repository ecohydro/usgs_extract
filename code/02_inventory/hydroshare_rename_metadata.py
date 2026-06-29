"""
hydroshare_rename_metadata.py

1. Renames each data/hydroshare/{type}/{county}/metadata.csv
   -> metadata_{county_slug}.csv

2. Writes one data/hydroshare/{type}/metadata_{type_slug}.csv
   covering all counties for that water type.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DST_BASE = ROOT / "data/hydroshare"


def main():
    # ------------------------------------------------------------------
    # Step 1 – rename per-county metadata.csv -> metadata_{county}.csv
    # ------------------------------------------------------------------
    renamed = 0
    already = 0
    for meta_file in sorted(DST_BASE.glob("*/*/metadata.csv")):
        county_slug = meta_file.parent.name          # e.g. alameda_county
        new_name = meta_file.parent / f"metadata_{county_slug}.csv"
        if new_name.exists():
            already += 1
        else:
            meta_file.rename(new_name)
            renamed += 1

    print(f"Renamed:        {renamed}")
    print(f"Already named:  {already}")

    # ------------------------------------------------------------------
    # Step 2 – write per-water-type metadata_{type_slug}.csv
    # ------------------------------------------------------------------
    type_dirs = [d for d in DST_BASE.iterdir() if d.is_dir() and not d.name.startswith("_")]
    written = skipped = 0

    for type_dir in sorted(type_dirs):
        type_slug = type_dir.name
        out_path = type_dir / f"metadata_{type_slug}.csv"

        if out_path.exists():
            skipped += 1
            continue

        # Concatenate all per-county metadata files for this type
        county_files = sorted(type_dir.glob("*/metadata_*.csv"))
        if not county_files:
            print(f"  WARNING: no county metadata files found in {type_dir.name}/")
            continue

        combined = pd.concat(
            [pd.read_csv(f, low_memory=False) for f in county_files],
            ignore_index=True,
        )
        combined.to_csv(out_path, index=False)
        written += 1
        print(f"  {type_slug}: {len(combined):,} rows -> {out_path.name}")

    print(f"\nType-level CSVs written:  {written}")
    print(f"Type-level CSVs skipped:  {skipped}  (already existed)")


if __name__ == "__main__":
    main()
