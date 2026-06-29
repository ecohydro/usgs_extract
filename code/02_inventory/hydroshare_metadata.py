"""
hydroshare_metadata.py

For each water_type/county group in data/hydroshare, writes a metadata.csv
containing all main_metadata rows for the pages in that group.

Placed at:
    data/hydroshare/{water_type}/{county}/metadata.csv

Source columns: everything from main_metadata.csv, plus county_spatial and
water_type_final from table_level_metadata_county.csv.

Resumable: skips groups whose metadata.csv already exists.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DST_BASE = ROOT / "data/hydroshare"
TABLE_META = ROOT / "data/analysis/table_level_metadata_county.csv"
MAIN_META = ROOT / "data/metadata/main_metadata.csv"

# ---------------------------------------------------------------------------
# Name slug helpers (must match hydroshare_copy.py)
# ---------------------------------------------------------------------------
def type_slug(w): return w.strip().lower().replace(" ", "_")
def county_slug(c): return c.strip().lower().replace(" ", "_") + "_county"


def main():
    print("Loading metadata …")
    tl = pd.read_csv(TABLE_META)
    mm = pd.read_csv(MAIN_META, low_memory=False)
    mm = mm.rename(columns={"id": "doc_id"})

    # Join table-level county/type info onto main_metadata
    # One row per (doc_id, page_number, table_number) in tl;
    # main_metadata can have multiple rows per page — keep all of them.
    tl_slim = tl[["doc_id", "page_number", "water_type_final", "county_spatial"]].copy()
    tl_slim["doc_id"] = tl_slim["doc_id"].astype(int)

    mm = mm.dropna(subset=["doc_id", "page_number"])
    mm["doc_id"] = mm["doc_id"].astype(int)
    mm["page_number"] = mm["page_number"].astype(int)

    # Drop pages with no county (outside CA / no coords — same as copy script)
    tl_assigned = tl_slim[tl_slim["county_spatial"].notna()]

    # Unique (doc_id, page_number, water_type_final, county_spatial) pairs
    page_groups = (
        tl_assigned[["doc_id", "page_number", "water_type_final", "county_spatial"]]
        .drop_duplicates()
    )
    page_groups["type_slug"] = page_groups["water_type_final"].map(type_slug)
    page_groups["county_slug"] = page_groups["county_spatial"].map(county_slug)

    # Join page_groups -> main_metadata to get all metadata rows per page
    enriched = page_groups.merge(mm, on=["doc_id", "page_number"], how="left")
    print(f"Enriched rows (page_groups x main_metadata): {len(enriched):,}")

    groups = page_groups.groupby(["type_slug", "county_slug"])
    print(f"Groups: {len(groups)}")
    print("Writing metadata CSVs …")

    written = skipped = 0
    for (ts, cs), group_pages in groups:
        out_path = DST_BASE / ts / cs / "metadata.csv"

        if out_path.exists():
            skipped += 1
            continue

        # All enriched rows for this group
        mask = (enriched["type_slug"] == ts) & (enriched["county_slug"] == cs)
        group_meta = enriched[mask].drop(columns=["type_slug", "county_slug"])
        group_meta.to_csv(out_path, index=False)
        written += 1

        if (written + skipped) % 50 == 0:
            print(f"  {written + skipped}/{len(groups)}  written={written}  skipped={skipped}")

    print(f"\nDone.")
    print(f"  Written:  {written}")
    print(f"  Skipped:  {skipped}  (already existed)")


if __name__ == "__main__":
    main()
