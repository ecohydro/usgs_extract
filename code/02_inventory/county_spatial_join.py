"""
county_spatial_join.py

Assigns a clean CA county name to every published table via spatial join of
lat_combined / lon_combined against the Census TIGER county shapefile.

Also checks consistency between the spatially-assigned county and the county
field of the metadata (new per-location metadata, `county`, 2026-11-19; was the
Phi-4 actual_county / inferred_county fields), flagging mismatches as a data
quality indicator.

Rebuilt on the new metadata 2026-11-19:
  - county only from precise or approximate points (same rule as the catalog,
    18_build_catalog.py); basin / conflict / unlocated tables get no county
  - tribal_withheld column (paper_filters.tribal_mask): pages withheld from
    publication; hydroshare_copy must skip them
  - measurements = n_measurements (19_value_counts.py), not printed row counts

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

import sys

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_filters import tribal_mask  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SPATIAL_DIR = ROOT / "data/analysis/spatial/ca_counties"
COUNTY_SHP = SPATIAL_DIR / "tl_2023_us_county.shp"
TABLE_META = ROOT / "data/analysis/table_level_metadata_published.csv"
MAIN_META = ROOT / "data/analysis/metadata_base.csv"  # 00a_metadata_base.py
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

    # metadata county field: all of the table's locations, joined
    mm = pd.read_csv(MAIN_META, usecols=["doc_id", "page_number", "table_number", "county"],
                     low_memory=False)
    mm = (mm.dropna(subset=["county"])
            .groupby(["doc_id", "page_number", "table_number"])["county"]
            .agg(lambda x: "; ".join(dict.fromkeys(x.astype(str))))
            .rename("meta_county").reset_index())
    print(f"tables with a metadata county: {len(mm)}")
    merged = tl.merge(mm, on=["doc_id", "page_number", "table_number"], how="left")

    # county only from precise / approximate points
    ok = merged["precision_class"].isin(["precise", "approximate"])
    merged.loc[~ok, ["lat_combined", "lon_combined"]] = float("nan")
    merged["tribal_withheld"] = tribal_mask(merged, id_col="doc_id").values
    print(f"tribal-withheld tables: {int(merged.tribal_withheld.sum())}")
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


# county names that are also states or common words: need "County"/"Co." after them
NEEDS_COUNTY_WORD = {"nevada", "lake", "orange", "sierra", "kings"}


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
        pat = r"\b" + re.escape(name) + (r"\s+(county|co\b)" if name in NEEDS_COUNTY_WORD else r"\b")
        if re.search(pat, text_lower):
            found.add(name)
    return found


def check_consistency(merged):
    """
    For each table, compare county_spatial to the CA counties named in the
    metadata county field (meta_county, all locations of the table).

    Consistency categories:
      match        -- county_spatial appears in the county field
      mismatch     -- county_spatial does NOT appear in the county field
                      (and the field names at least one recognizable CA county)
      llm_no_ca    -- the field names no recognizable CA county (e.g. "Calif.",
                      out-of-state, free text)
      no_spatial   -- no county_spatial assigned (no precise/approximate point,
                      or outside CA)
      no_llm       -- county field empty
    """
    df = merged[["doc_id", "page_number", "table_number", "water_type_final",
                 "county_spatial", "meta_county"]].copy()

    # Vectorized: extract CA county sets from LLM fields
    def llm_ca_str(series):
        return series.apply(
            lambda x: ", ".join(sorted(normalize_county(x))) if isinstance(x, str) else ""
        )

    df["llm_ca"] = df["meta_county"].apply(normalize_county)
    df["llm_ca_counties"] = df["llm_ca"].apply(
        lambda s: ", ".join(sorted(s)) if s else ""
    )

    spatial_lower = df["county_spatial"].str.lower()

    no_spatial = df["county_spatial"].isna()
    no_llm = df["meta_county"].isna()
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
               "county_spatial", "meta_county",
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
             n_measurements=("n_measurements", "sum"))
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
        "County consistency check — spatial join (precise/approximate points) vs. metadata county field",
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
        "n_measurements", "year_start", "year_end", "decade_start",
        "lat_combined", "lon_combined", "precision_class", "watersource_name",
        "county_spatial", "tribal_withheld",
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
