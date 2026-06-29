"""
hydroshare_copy.py

Copies data from data/digitized into upload-ready groups under data/hydroshare,
organized as:

    data/hydroshare/{water_type}/{county_name}/
        {doc_id}/
            page_{N}/
                {doc_id}_page_{N}.png
                {doc_id}_page_{N}.json
                {doc_id}_page_{N}_table1.csv
                ...

Rules:
- Grouping uses county_spatial (Census TIGER spatial join) from
  table_level_metadata_county.csv — clean, standardized county names.
- Pages with no county assigned (outside CA boundary or no coordinates)
  are SKIPPED. These will be handled separately in a future step.
- Pages that contain tables of more than one water type are copied into
  EACH matching water_type/county group (not deduplicated to one type).
- Original data in data/digitized is never modified.

Resumable: if a destination page directory already exists, it is skipped.

Outputs
-------
data/hydroshare/{water_type}/{county_name}/{doc_id}/page_{N}/  (copied files)
data/hydroshare/_copy_log.csv  -- one row per (water_type, county, doc_id, page)
                                   with status: copied / skipped / src_missing
"""

import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_BASE = ROOT / "data/digitized"
DST_BASE = ROOT / "data/hydroshare"
TABLE_META = ROOT / "data/analysis/table_level_metadata_county.csv"
LOG_PATH = DST_BASE / "_copy_log.csv"

# ---------------------------------------------------------------------------
# Name slug helpers
# ---------------------------------------------------------------------------
def type_slug(water_type: str) -> str:
    return water_type.strip().lower().replace(" ", "_")


def county_slug(county: str) -> str:
    # Census TIGER names are already clean title-case e.g. "Los Angeles"
    return county.strip().lower().replace(" ", "_") + "_county"


# ---------------------------------------------------------------------------
# Build copy plan: unique (water_type, county, doc_id, page_number) tuples
# ---------------------------------------------------------------------------
def build_plan(tl: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with no county assigned (outside CA / no coords)
    assigned = tl[tl["county_spatial"].notna()].copy()
    print(f"Tables with county assigned: {len(assigned):,} / {len(tl):,}")
    print(f"Skipped (no county):         {tl['county_spatial'].isna().sum():,}")

    # One copy task per unique (water_type, county, doc_id, page_number)
    # A page with 2 water types on it appears twice — once per type — which is correct.
    plan = (
        assigned[["water_type_final", "county_spatial", "doc_id", "page_number"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    plan["type_slug"] = plan["water_type_final"].map(type_slug)
    plan["county_slug"] = plan["county_spatial"].map(county_slug)
    plan["doc_id"] = plan["doc_id"].astype(int)
    plan["page_number"] = plan["page_number"].astype(int)
    print(f"Copy tasks (unique type+county+page): {len(plan):,}")
    return plan


# ---------------------------------------------------------------------------
# Execute copies
# ---------------------------------------------------------------------------
def run_copy(plan: pd.DataFrame) -> pd.DataFrame:
    log_rows = []
    n_copied = n_skipped = n_missing = 0
    total = len(plan)

    for i, row in enumerate(plan.itertuples(), 1):
        src = SRC_BASE / str(row.doc_id) / f"page_{row.page_number}"
        dst = (DST_BASE / row.type_slug / row.county_slug
               / str(row.doc_id) / f"page_{row.page_number}")

        if dst.exists():
            status = "skipped"
            n_skipped += 1
        elif not src.exists():
            status = "src_missing"
            n_missing += 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            status = "copied"
            n_copied += 1

        log_rows.append(
            {
                "water_type": row.water_type_final,
                "county": row.county_spatial,
                "doc_id": row.doc_id,
                "page_number": row.page_number,
                "status": status,
            }
        )

        if i % 1000 == 0 or i == total:
            print(
                f"  {i:>6}/{total}  copied={n_copied}  skipped={n_skipped}"
                f"  src_missing={n_missing}"
            )

    log_df = pd.DataFrame(log_rows)
    return log_df, n_copied, n_skipped, n_missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    DST_BASE.mkdir(parents=True, exist_ok=True)

    tl = pd.read_csv(TABLE_META)
    plan = build_plan(tl)

    # Summary of groups before starting
    groups = (
        plan.groupby(["type_slug", "county_slug"])
        .size()
        .reset_index(name="n_pages")
        .sort_values(["type_slug", "n_pages"], ascending=[True, False])
    )
    print(f"\nGroups to create: {len(groups)} (water_type x county)")
    print("\nLargest 10 groups:")
    print(groups.nlargest(10, "n_pages")[["type_slug", "county_slug", "n_pages"]].to_string(index=False))

    print(f"\nStarting copy to {DST_BASE} …\n")
    log_df, n_copied, n_skipped, n_missing = run_copy(plan)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)

    print(f"\nDone.")
    print(f"  Copied:      {n_copied:,}")
    print(f"  Skipped:     {n_skipped:,}  (destination already existed)")
    print(f"  Src missing: {n_missing:,}")
    print(f"  Log saved:   {LOG_PATH}")


if __name__ == "__main__":
    main()
