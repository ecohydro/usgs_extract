"""
Score per-table temporal resolution, derived-summary flag and primary parameter
(20_temporal_resolution.py -> data/analysis/table_resolution.csv) against Annette's blind
grades (data/analysis/temporal_resolution/review/resolution_review_sample.csv, 65 tables,
graded 2026-11-12).

Grading conventions applied here:
  - blank true_resolution = grader found no time dimension ("no dates on page") ->
    no_time_series; 'unclassified' from the script is counted as a soft miss for those.
  - true_parameter is free text; it is mapped to the script's categories with keywords and
    the script's primary_parameter is correct if it is ANY category the grader named
    (the grader often wrote two, e.g. "stream discharge, and gage height").
  - DEFINITION rows: the blind grade differed from the README definition; Annette settled
    these on 2026-11-13 (rating tables = annual; revised peaks for selected years = discrete;
    annual max/min per water year = annual; peaks above base = annual; dated samples taken
    about monthly = monthly). Those settled labels are the definition truth below.
  - DERIVED_OVERRIDE: tables the grader marked "related to other tables on the page" (n) count
    as derived (Annette 2026-11-13: "if I said related, include them as derived").

Output: data/analysis/temporal_resolution/review/resolution_review_scored.csv
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / 'data/analysis/temporal_resolution/review'
KEY = ['doc_id', 'page_number', 'table_number']

# sample_no -> class per the README definitions (grader's label differs; see note)
DEFINITION = {
    12: ('annual', 'rating table (settled 2026-11-13)'),
    54: ('annual', 'rating table (settled 2026-11-13)'),
    65: ('annual', 'rating table with two periods of use (settled 2026-11-13)'),
    21: ('discrete', 'revised peaks for selected years (settled 2026-11-13)'),
    31: ('annual', 'annual maximum and minimum per water year (settled 2026-11-13)'),
    55: ('annual', 'peak discharges above a base + annual maximum (settled 2026-11-13)'),
    61: ('monthly', 'water-quality samples about once a month (settled 2026-11-13)'),
    64: ('monthly', 'spring discharge measured about once a month (settled 2026-11-13)'),
    2: ('no_time_series', 'stratigraphy; grader note: no dates'),
    5: ('discrete', 'one water-quality sample dated 75-11-12'),
}
# grader said "related" (not continuation) -> derived
# 59 = extremes (max/min) table of the same kind as 31 -> scored like 31 for consistency
DERIVED_OVERRIDE = {25: True, 31: True, 58: True, 63: True, 59: True}

PMAP = [('water_chemistry', r'water quality|chloride|chemi'),
        ('sediment', r'sediment'),
        ('temperature', r'temperature'),
        ('gage_height', r'gage height|elevation'),
        ('contents', r'acre-?feet|contents|storage'),
        ('water_level', r'well level|water level'),
        ('precipitation', r'precipitation|rainfall'),
        ('discharge', r'discharge|runoff|pumpage')]


def grader_params(text):
    t = str(text).lower()
    return {n for n, rx in PMAP if re.search(rx, t)} or {'other'}


g = pd.read_csv(R / 'resolution_review_sample.csv')
res = pd.read_csv(ROOT / 'data/analysis/table_resolution.csv',
                  usecols=KEY + ['resolution', 'resolution_evidence', 'derived_summary', 'primary_parameter'])
m = g.merge(res, on=KEY, how='left')
m['truth'] = m.true_resolution.fillna('no_time_series').str.strip()
m['truth_def'] = [DEFINITION.get(s, (t, ''))[0] for s, t in zip(m.sample_no, m.truth)]
m['def_note'] = [DEFINITION.get(s, ('', ''))[1] for s in m.sample_no]
m['ok_grader'] = m.resolution.eq(m.truth)
m['ok_definition'] = m.resolution.eq(m.truth_def)
m['true_derived'] = m['derived_summary (y/n)'].str.strip().str.lower().eq('y')
m['true_derived'] = [DERIVED_OVERRIDE.get(s, d) for s, d in zip(m.sample_no, m.true_derived)]
m['ok_derived'] = m.true_derived.eq(m.derived_summary.astype(bool))
m['true_params'] = m.true_parameter.map(lambda s: ';'.join(sorted(grader_params(s))))
# grader "other" (stratigraphy, bank erosion, aquifer units ...) = script 'unknown'
m['ok_parameter'] = [p in set(tp.split(';')) or (p == 'unknown' and 'other' in tp)
                     for p, tp in zip(m.primary_parameter, m.true_params)]
m.to_csv(R / 'resolution_review_scored.csv', index=False)

n = len(m)
print(f'Tables graded: {n}')
print(f'Resolution = grader label:        {m.ok_grader.sum()}/{n}')
print(f'Resolution = README definition:   {m.ok_definition.sum()}/{n}')
print(f'Derived-summary flag agrees:      {m.ok_derived.sum()}/{n}')
print(f'Primary parameter in grader set:  {m.ok_parameter.sum()}/{n}')
print('\nResolution confusion (rows = definition truth, cols = script):')
print(pd.crosstab(m.truth_def, m.resolution).to_string())
bad = m[~m.ok_definition]
print('\nMisses vs definition:')
print(bad[['sample_no', 'doc_id', 'page_number', 'table_number', 'truth_def', 'resolution',
           'resolution_evidence']].to_string(index=False))
print('\nDerived-summary disagreements:')
print(m[~m.ok_derived][['sample_no', 'resolution', 'true_derived', 'derived_summary', 'note']]
      .to_string(index=False))
print('\nParameter misses:')
print(m[~m.ok_parameter][['sample_no', 'water_type_final', 'primary_parameter', 'true_parameter']]
      .to_string(index=False))
