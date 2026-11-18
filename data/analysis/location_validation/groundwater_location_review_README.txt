groundwater_location_review.csv -- column guide (built 2026-10-28)

One row per LLM metadata row in data/metadata/main_metadata.csv whose water_type
contains "groundwater" (20,624 rows; includes ~195 rows whose water_type is a
list such as ['Stream Discharge', 'Groundwater']). Nothing in main_metadata.csv
was changed; this is a review copy.

IDENTIFIERS
  meta_row            row position (0-based) in main_metadata.csv
  id, page_number     document and page
  page_png            path to the page image (blank if no image)
  document_title      USGS publication title

ORIGINAL LLM FIELDS (pointers only; not used as locations)
  watersource_name, location, llm_trs, llm_actual_county, llm_inferred_county,
  dates_of_recording
  llm_actual_lat/lon     coordinates the LLM labeled "actual"
  llm_inferred_lat/lon   coordinates the LLM inferred (dropped from the pipeline)

NEW LOCATION (result of the hierarchy)
  coord_method     which method located the row (highest tier wins):
                     printed        LLM "actual" lat/long confirmed printed on the page
                     plss_section   Township/Range/Section on the page -> section centroid
                     plss_township  Township/Range on the page -> township centroid
                     nwis_station   NWIS station number printed on the page
                     gnis_<type>    named place on the page -> GNIS point
                                    (qualifier_at, qualifier_near, place, feature_lake, ...)
                     unlocated      none of the above
  precision_class  precise (HUC12-usable) | approximate (county/HUC8) | unlocated
  precision_km     typical error of the method (median vs printed coordinates)
  new_lat/new_lon  the new coordinate

STEP-BY-STEP DIAGNOSTICS
  tier1_status       printed_match / printed_match_no_keyword = confirmed on page;
                     not_printed = LLM "actual" coords are NOT on the page;
                     printed_other_values = page prints coords, but different ones;
                     no_llm_actual = LLM gave no "actual" coords
  trs_parsed         T/R(/sec) read from the LLM TRS or location field
  trs_on_page        verified = same T and R found in the page text;
                     tr_not_on_page = LLM's T/R is not on the page (rejected)
  trs_status         ok | ambiguous_meridian (several meridians fit, no page evidence) |
                     not_verified | no_match | no_trs_in_llm_fields
  trs_meridian_rule  how the meridian was chosen (only candidate, county named on page, ...)
  trs_meridian, trs_precision, trs_lat, trs_lon   the PLSS reference (if ok)
  nwis_status        ok | no_name_match | tie | no_nwis_number_on_page
  nwis_site_no, nwis_station_nm
  gnis_status        ok | no_candidate (no place name found in LLM text) |
                     not_in_gnis | name_not_on_page | ambiguous (duplicate names) | not_run
  gnis_place, gnis_match_type, gnis_lat, gnis_lon

UPDATED 2026-10-28 (after the same-page well-number step, 07_well_numbers.py)
  coord_method adds:  nwis_well     well number matched to an NWIS groundwater site (exact)
                      plss_wellno   section centroid from the well number
                      wellno_group  row covers several wells within 5 km (centroid; approximate)
  wellno                    well number as resolved, e.g. 6S/32W-9B2
  wellno_status             ok | ok_group | ambiguous (directions/meridian not settled from the page) |
                            group (several wells >5 km apart) | no_row_match | no_well_number_on_page
  wellno_match_rule         well number in LLM text | owner name in well heading | only well on page
  wellno_direction_rule     how missing N/S, E/W letters / meridian were settled
                            (as printed, T/R on page, meridian letter, only candidate,
                             town in well heading, county on page)
  wellno_meridian, wellno_n_on_page, wellno_nwis_site_no
