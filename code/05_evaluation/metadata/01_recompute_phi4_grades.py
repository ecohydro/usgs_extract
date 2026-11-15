"""Recompute the original Phi-4 metadata accuracy from the hand-graded test sets.

Reads the two graded workbooks (curated, random) in
data/digitization_intermediates/04_metadata_extraction/validation/, where each
LLM output row is followed by a grader row (initials_ID in column A, 1 / 0.5 / 0
per graded field), and writes to data/analysis/evaluation/metadata/:
  phi4_grades_long.csv        one row per graded entry x field
  phi4_grades_by_field.csv    accuracy per field and set, with the spreadsheet's value
  phi4_grades_overall.csv     overall accuracy per set under each averaging rule
  phi4_grades_blank_vs_given.csv  accuracy split by whether the LLM left the field blank

Run from the repo root (needs pandas, numpy, openpyxl):
  python code/05_evaluation/metadata/01_recompute_phi4_grades.py
"""
import re
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd

VAL = Path("data/digitization_intermediates/04_metadata_extraction/validation")
OUT = Path("data/analysis/evaluation/metadata")
SETS = {"curated": ("validation_curated_135.xlsx", "Sheet1"),
        "random": ("validation_random_135.xlsx", "metadata")}
FIELDS = ["Actual_Latitude", "Actual_Longitude", "Location", "Townships_Ranges_Sections",
          "Watersource_Name", "Actual_County", "Dates_of_Recording", "Temporal_Resolution",
          "Units_Of_Measurement", "Water_Type"]
CAT_NAMES = {"Latitude": "Actual_Latitude", "Longitude": "Actual_Longitude", "Location": "Location",
             "Township Range Section": "Townships_Ranges_Sections", "Water source name": "Watersource_Name",
             "County": "Actual_County", "Date": "Dates_of_Recording",
             "Temporal resolution": "Temporal_Resolution", "Units": "Units_Of_Measurement",
             "Water category": "Water_Type"}
N_BOOT = 2000   # document-cluster bootstrap draws
SEED = 0


def blank(v):
    return v is None or str(v).strip() in ("", "[]", "{}", "None", "nan")


def read_set(name):
    fname, sheet = SETS[name]
    rows = list(openpyxl.load_workbook(VAL / fname)[sheet].iter_rows(values_only=True))
    col = {h: i for i, h in enumerate(rows[0]) if h}
    out, prev = [], None
    for xl_row, r in enumerate(rows[1:], start=2):
        a = r[0]
        if isinstance(a, (int, float)):
            prev = r
        elif isinstance(a, str) and re.match(r"^[A-Z]{2}_\d+", a.strip()) and prev is not None:
            for f in FIELDS:
                g = r[col[f]]
                out.append(dict(test_set=name, entry=a.strip(), grader=a.strip()[:2], xl_row=xl_row,
                                doc_id=int(prev[col["ID"]]), page_number=int(prev[col["PAGE_NUMBER"]]),
                                field=f, llm_value=prev[col[f]], llm_blank=blank(prev[col[f]]),
                                grade=g if isinstance(g, (int, float)) else np.nan))
            prev = None
    return pd.DataFrame(out)


def boot_ci(df, value="grade"):
    """95% interval resampling documents (entries on one report are not independent)."""
    rng = np.random.default_rng(SEED)
    g = df.dropna(subset=[value]).groupby("doc_id")[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.concat([read_set(s) for s in SETS], ignore_index=True)
    long.to_csv(OUT / "phi4_grades_long.csv", index=False)

    cat = pd.read_excel(VAL / "test_set_cat_grades.xlsx").dropna(subset=["category"])
    cat["field"] = cat["category"].str.strip().map(CAT_NAMES)
    sheet = cat.melt(id_vars="field", value_vars=["random_set", "curated_set"],
                     var_name="test_set", value_name="spreadsheet_pct")
    sheet["test_set"] = sheet["test_set"].str.replace("_set", "")

    rows = []
    for (s, f), d in long.groupby(["test_set", "field"]):
        lo, hi = boot_ci(d)
        rows.append(dict(test_set=s, field=f, n_graded=d.grade.notna().sum(), n_ungraded=d.grade.isna().sum(),
                         n_full=(d.grade == 1).sum(), n_half=(d.grade == .5).sum(), n_zero=(d.grade == 0).sum(),
                         lenient_pct=100 * d.grade.mean(), strict_pct=100 * (d.grade.dropna() == 1).mean(),
                         ci_lo=100 * lo, ci_hi=100 * hi))
    by_field = pd.DataFrame(rows).merge(sheet, on=["test_set", "field"], how="left")
    by_field["diff_vs_spreadsheet"] = by_field.lenient_pct - by_field.spreadsheet_pct
    by_field.to_csv(OUT / "phi4_grades_by_field.csv", index=False)

    rows = []
    for s, d in long.groupby("test_set"):
        per_entry = d.groupby(["entry", "xl_row"]).grade.mean()          # graded fields only
        per_entry10 = d.groupby(["entry", "xl_row"]).grade.sum() / 10    # blanks count as 0 (random sheet formula)
        per_page = d.groupby(["doc_id", "page_number"]).grade.mean()
        lo, hi = boot_ci(d)
        rows.append(dict(test_set=s, entries=d.xl_row.nunique(), pages=len(per_page), documents=d.doc_id.nunique(),
                         graders=";".join(f"{k}:{v}" for k, v in d.drop_duplicates("xl_row").grader.value_counts().items()),
                         pooled_grades_pct=100 * d.grade.mean(), ci_lo=100 * lo, ci_hi=100 * hi,
                         mean_of_fields_pct=100 * d.groupby("field").grade.mean().mean(),
                         mean_of_entries_pct=100 * per_entry.mean(),
                         mean_of_entries_sum_over_10_pct=100 * per_entry10.mean(),
                         mean_of_pages_pct=100 * per_page.mean(),
                         strict_pooled_pct=100 * (d.grade.dropna() == 1).mean()))
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "phi4_grades_overall.csv", index=False)

    bv = (long.dropna(subset=["grade"]).groupby(["test_set", "field", "llm_blank"]).grade
          .agg(n="count", lenient_pct=lambda x: 100 * x.mean()).reset_index())
    bv.to_csv(OUT / "phi4_grades_blank_vs_given.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(by_field.round(2).to_string(index=False)); print(overall.round(2).T.to_string())
    print(bv.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
