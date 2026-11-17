"""The metadata table agreed by Anna and Annette (2026-11-15): one row per location per table.

Column lists, allowed values and the location hierarchy live here and nowhere else; the answer key,
the review page and the scorer all import them.
"""
KEYS = ["doc_id", "page_number", "table_number", "location_number"]

# Location evidence, best first (Annette's ranking in 04_build_locations.py, extended downward).
# Each is copied from the page as printed. The first one filled becomes `location`.
EVIDENCE = ["coordinates", "well_number", "station_number", "station_name", "township_range_section",
            "place_name", "basin_or_area", "county", "state"]
LOCATION_TYPES = ["coordinates", "well_number", "station_number", "station_name", "township_range_section",
                  "named_place", "basin_or_area", "county", "state", "none"]
EVIDENCE_EXTRA = ["place_relation"]   # at / near / above / below / "3 miles north of": goes with place_name

GRADED = ["location", "location_type", "water_source", "data_type", "date_start", "date_end",
          "temporal_resolution", "variables", "construction_date"]
# Anna 2026-11-15: the two counts (n_measurements, n_variables) are Annette's postprocessing, not part of this
# pipeline. The answer keys still hold them (COUNTS_IN_KEY) but nothing produces or grades them here.
COUNTS_IN_KEY = ["n_measurements", "n_variables"]

DATA_TYPES = ["Stream Discharge", "Groundwater", "Reservoir", "Irrigation", "Springs", "Precipitation",
              "Water Quality", "Not Water Related", "Other"]            # Annette's nine, unchanged
TEMPORAL_RESOLUTIONS = ["daily", "monthly", "annual", "discrete", "no_time_series"]
VARIABLES_MEASURED = ["discharge", "runoff", "gage_height", "contents", "elevation", "water_level", "well_yield",
                      "temperature", "sediment", "water_chemistry", "precipitation", "evaporation"]
VARIABLES_DESCRIBED = ["well_construction", "lithology", "water_use", "irrigated_area"]
VARIABLES = VARIABLES_MEASURED + VARIABLES_DESCRIBED + ["other"]

BOOKKEEPING = ["n_locations_in_table", "evidence_place", "confidence", "notes"]
EVIDENCE_PLACES = ["in table", "heading before table", "elsewhere on page"]

KEY_COLUMNS = KEYS + GRADED + COUNTS_IN_KEY + EVIDENCE + EVIDENCE_EXTRA + BOOKKEEPING
BLANK = {"", "none", "None", "null", "nan"}


def clean(v):
    v = "" if v is None else str(v).strip()
    return "" if v in BLANK else v


def best_location(row):
    """(location, location_type) from a row's evidence columns, by the hierarchy."""
    for col in EVIDENCE:
        v = clean(row.get(col))
        if v and v != "cannot tell":
            if col == "place_name":
                rel = clean(row.get("place_relation"))
                return (f"{rel} {v}".strip(), "named_place")
            return (v, col)
    return ("", "none")
