"""
county_spatial_join.py

Assigns a clean CA county name to every published table via spatial join of
lat_combined / lon_combined against the Census TIGER county shapefile.

Also checks consistency between the spatially-assigned county and the LLM-
extracted actual_county / inferred_county fields, flagging mismatches as a
data quality indicator.

Outputs
-------
data/analysis/spatial/ca_counties/           -- TIGER county shapefile (downloaded once)
data/analysis/table_level_metadata_county.csv -- table_level_metadata_published + county_spatial
data/analysis/county_consistency_report.csv  -- per-table mismatch flags
data/analysis/county_consistency_summary.txt -- summary statistics
"""

import io
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SPATIAL_DIR = ROOT / "data/analysis/spatial/ca_counties"
COUNTY_SHP = SPATIAL_DIR / "tl_2023_us_county.shp"
TABLE_META = ROOT / "data/analysis/table_level_metadata_published.csv"
MAIN_META = ROOT / "data/metadata/main_metadata.csv"
OUT_TABLE = ROOT / "data/analysis/table_level_metadata_county.csv"
OUT_CONSISTENCY = ROOT / "data/analysis/county_consistency_report.csv"
OUT_SUMMARY = ROOT / "data/analysis/county_consistency_summary.txt"

TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)

# ---------------------------------------------------------------------------
# Step 1 – Download CA county shapefile if not already present
# ---------------------------------------------------------------------------
def download_county_shapefile():
    if COUNTY_SHP.exists():
        print(f"County shapefile already present: {COUNTY_SHP}")
        return
    SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CA county shapefile from Census TIGER …")
    resp = requests.get(TIGER_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(SPATIAL_DIR)
    print(f"Saved to {SPATIAL_DIR}")


# ---------------------------------------------------------------------------
# Step 2 – Load data
# ---------------------------------------------------------------------------
def load_data():
    raw = gpd.read_file(COUNTY_SHP)
    # Filter to California (state FIPS 06) and keep only county name + geometry
    counties = raw[raw["STATEFP"] == "06"][["NAME", "geometry"]].rename(
        columns={"NAME": "county_spatial"}
    ).copy()
    # TIGER is in EPSG:4269 (NAD83); reproject to WGS84 for consistency with
    # the lat/lon coordinates in our metadata (which are WGS84 decimal degrees)
    if counties.crs and counties.crs.to_epsg() != 4326:
        counties = counties.to_crs(epsg=4326)
    print(f"County shapefile: {len(counties)} counties, CRS={counties.crs}")

    tl = pd.read_csv(TABLE_META)
    print(f"Published tables: {len(tl)} rows")

    mm = pd.read_csv(MAIN_META, low_memory=False)
    mm = mm.rename(columns={"id": "doc_id"})
    # Deduplicate to one row per page; prefer rows with actual_county filled in
    mm = (
        mm.sort_values("actual_county", na_position="last")
        .drop_duplicates(subset=["doc_id", "page_number"], keep="first")
    )
    # main_metadata has raw lat/lon; lat_combined is in table_level_metadata_published
    mm = mm[["doc_id", "page_number", "actual_county", "inferred_county"]]
    print(f"main_metadata (deduplicated): {len(mm)} rows")

    # tl already has lat_combined / lon_combined (computed by 00_data_prep)
    merged = tl.merge(mm, on=["doc_id", "page_number"], how="left")
    print(f"After merge with main_metadata: {len(merged)} rows")
    return counties, merged


# ---------------------------------------------------------------------------
# Step 3 – Spatial join
# ---------------------------------------------------------------------------
def spatial_join(counties, merged):
    has_coords = merged["lat_combined"].notna() & merged["lon_combined"].notna()
    print(f"Rows with coordinates: {has_coords.sum()} / {len(merged)}")

    pts = gpd.GeoDataFrame(
        merged[has_coords].copy(),
        geometry=gpd.points_from_xy(
            merged.loc[has_coords, "lon_combined"],
            merged.loc[has_coords, "lat_combined"],
        ),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts, counties, how="left", predicate="within")
    # Drop the index_right column added by sjoin
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Merge county_spatial back into full merged table
    merged = merged.merge(
        joined[["doc_id", "page_number", "table_number", "county_spatial"]],
        on=["doc_id", "page_number", "table_number"],
        how="left",
    )

    in_ca = merged["county_spatial"].notna().sum()
    no_coords = (~has_coords).sum()
    outside_ca = has_coords.sum() - in_ca
    print(f"Spatially assigned to CA county: {in_ca}")
    print(f"No coordinates (unassigned):     {no_coords}")
    print(f"Coords outside CA boundary:      {outside_ca}")
    return merged


# ---------------------------------------------------------------------------
# Step 4 – Consistency check
# ---------------------------------------------------------------------------

# The 58 CA county names (without "County" suffix) for normalization
CA_COUNTIES = {
    "alameda", "alpine", "amador", "butte", "calaveras", "colusa",
    "contra costa", "del norte", "el dorado", "fresno", "glenn",
    "humboldt", "imperial", "inyo", "kern", "kings", "lake", "lassen",
    "los angeles", "madera", "marin", "mariposa", "mendocino", "merced",
    "modoc", "mono", "monterey", "napa", "nevada", "orange", "placer",
    "plumas", "riverside", "sacramento", "san benito", "san bernardino",
    "san diego", "san francisco", "san joaquin", "san luis obispo",
    "san mateo", "santa barbara", "santa clara", "santa cruz", "shasta",
    "sierra", "siskiyou", "solano", "sonoma", "stanislaus", "sutter",
    "tehama", "trinity", "tulare", "tuolumne", "ventura", "yolo", "yuba",
}


def normalize_county(text: str) -> set[str]:
    """
    Extract CA county names from a raw LLM county string.
    Returns a set of lowercase county names (without 'County' suffix).
    Handles multi-county strings, extra qualifiers, and non-CA counties.
    """
    if not isinstance(text, str):
        return set()
    text_lower = text.lower()
    found = set()
    for name in CA_COUNTIES:
        # Match 'name county' or 'name' at a word boundary, case-insensitive
        if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
            found.add(name)
    return found


def check_consistency(merged):
    """
    For each table, compare county_spatial to the CA counties mentioned in
    actual_county and inferred_county.

    Consistency categories:
      match        -- county_spatial appears in one or both LLM fields
      mismatch     -- county_spatial does NOT appear in either LLM field
                      (and LLM field has at least one recognizable CA county)
      llm_no_ca    -- LLM fields mention no recognizable CA county (may be
                      out-of-state, blank, or free-text only)
      no_spatial   -- no county_spatial assigned (no coords or outside CA)
      no_llm       -- both LLM county fields are null/empty
    """
    df = merged[["doc_id", "page_number", "table_number", "water_type_final",
                 "county_spatial", "actual_county", "inferred_county"]].copy()

    # Vectorized: extract CA county sets from LLM fields
    def llm_ca_str(series):
        return series.apply(
            lambda x: ", ".join(sorted(normalize_county(x))) if isinstance(x, str) else ""
        )

    df["actual_ca"] = df["actual_county"].apply(normalize_county)
    df["inferred_ca"] = df["inferred_county"].apply(normalize_county)
    df["llm_ca"] = df.apply(lambda r: r["actual_ca"] | r["inferred_ca"], axis=1)
    df["llm_ca_counties"] = df["llm_ca"].apply(
        lambda s: ", ".join(sorted(s)) if s else ""
    )

    spatial_lower = df["county_spatial"].str.lower()

    no_spatial = df["county_spatial"].isna()
    no_llm = df["actual_county"].isna() & df["inferred_county"].isna()
    llm_has_ca = df["llm_ca"].apply(bool)

    # Check if spatial county appears in llm_ca set (vectorized via apply)
    in_llm = df.apply(
        lambda r: (r["county_spatial"].lower() in r["llm_ca"])
        if isinstance(r["county_spatial"], str) and r["llm_ca"]
        else False,
        axis=1,
    )

    conditions = [
        no_spatial,
        (~no_spatial) & no_llm,
        (~no_spatial) & (~no_llm) & (~llm_has_ca),
        (~no_spatial) & (~no_llm) & llm_has_ca & in_llm,
        (~no_spatial) & (~no_llm) & llm_has_ca & (~in_llm),
    ]
    choices = ["no_spatial", "no_llm", "llm_no_ca", "match", "mismatch"]
    import numpy as np
    df["consistency"] = np.select(conditions, choices, default="unknown")

    return df[["doc_id", "page_number", "table_number", "water_type_final",
               "county_spatial", "actual_county", "inferred_county",
               "llm_ca_counties", "consistency"]]


# ---------------------------------------------------------------------------
# Step 5 – Produce cross-tab scoping matrix
# ---------------------------------------------------------------------------
def build_scoping_matrix(consistency_df, merged):
    # Add county_spatial back to merged for the cross-tab
    ct = consistency_df[["doc_id", "page_number", "table_number", "county_spatial"]]
    m = merged.merge(ct, on=["doc_id", "page_number", "table_number"], how="left",
                     suffixes=("_orig", ""))
    # Use county_spatial_y (from consistency_df) preferentially
    if "county_spatial_y" in m.columns:
        m["county_spatial"] = m["county_spatial_y"]

    cross = (
        m.groupby(["water_type_final", "county_spatial"])
        .agg(n_tables=("table_number", "count"),
             n_measurements=("number_rows", "sum"))
        .reset_index()
        .sort_values(["water_type_final", "n_tables"], ascending=[True, False])
    )
    return cross


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    download_county_shapefile()
    counties, merged = load_data()
    merged = spatial_join(counties, merged)

    print("\nRunning consistency check …")
    consistency_df = check_consistency(merged)

    # Consistency summary
    counts = consistency_df["consistency"].value_counts()
    pct = (counts / len(consistency_df) * 100).round(1)
    summary_lines = [
        "County consistency check — spatial join vs. LLM-extracted county fields",
        "=" * 70,
        f"Total published tables: {len(consistency_df):,}",
        "",
        "Status breakdown:",
    ]
    for status in ["match", "mismatch", "llm_no_ca", "no_llm", "no_spatial"]:
        n = counts.get(status, 0)
        p = pct.get(status, 0)
        summary_lines.append(f"  {status:<14} {n:>7,}  ({p:.1f}%)")

    # Mismatch detail: top mismatched county pairs
    mm_rows = consistency_df[consistency_df["consistency"] == "mismatch"]
    if len(mm_rows):
        summary_lines += [
            "",
            f"Mismatch detail (spatial county vs. LLM county):",
            f"  Total mismatches: {len(mm_rows):,}",
            "",
            "  Top 20 spatial->LLM mismatch pairs:",
        ]
        pairs = (
            mm_rows.groupby(["county_spatial", "llm_ca_counties"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(20)
        )
        for _, r in pairs.iterrows():
            summary_lines.append(
                f"    spatial={r['county_spatial']!r:25s}  llm={r['llm_ca_counties']!r}  n={r['n']}"
            )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text.encode("ascii", errors="replace").decode("ascii"))

    # Scoping cross-tab
    cross = build_scoping_matrix(consistency_df, merged)
    print(f"\nScoping cross-tab: {len(cross)} (water_type, county) groups")

    # Write outputs
    merged_out = merged[[
        "doc_id", "page_number", "table_number", "water_type_final",
        "number_rows", "year_start", "year_end", "decade_start",
        "lat_combined", "lon_combined", "watersource_name", "county_spatial",
    ]]
    merged_out.to_csv(OUT_TABLE, index=False)
    print(f"Saved: {OUT_TABLE}")

    consistency_df.to_csv(OUT_CONSISTENCY, index=False)
    print(f"Saved: {OUT_CONSISTENCY}")

    cross.to_csv(ROOT / "data/analysis/water_type_county_matrix.csv", index=False)
    print(f"Saved: data/analysis/water_type_county_matrix.csv")

    OUT_SUMMARY.write_text(summary_text, encoding="utf-8")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
