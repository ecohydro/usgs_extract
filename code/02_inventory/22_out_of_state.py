"""
Out-of-state split of the published tables (Annette, 2026-11-18): the paper uses California data only;
out-of-state data (tribal included) are kept separate and reported only as numbers in the methods.

  python 22_out_of_state.py
    -> data/analysis/table_level_metadata_published_ca.csv        California tables (paper set, incl. tribal rows;
                                                                   the paper also drops withheld tribal pages)
    -> data/analysis/out_of_state/out_of_state_tables.csv          tables outside California (incl. tribal)
    -> data/analysis/out_of_state/out_of_state_summary.csv         numbers for the methods
    -> data/analysis/out_of_state/out_of_state_by_water_type.csv, out_of_state_by_state.csv

California rule (same as Figure 1, paper_filters.ca_scope):
  precise / approximate point -> the state the point lies in (no state = outside the US, e.g. Baja California)
  basin point (DWR Bulletin 118)  -> California
  unlocated / conflicting coords  -> California unless the page names only other states
                                     (location_validation/page_states.csv, page_state_class == 'out_of_state')
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'data/analysis'
OUT = A / 'out_of_state'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_filters import ca_scope  # noqa: E402


def main():
    OUT.mkdir(exist_ok=True)
    t = pd.read_csv(A / 'table_level_metadata_published.csv', low_memory=False)
    t = ca_scope(t)
    trib = pd.read_csv(A / 'tribal_lands/tribal_screen_pages.csv', usecols=['id', 'page_number', 'withhold'])
    trib = trib[trib.withhold].rename(columns={'id': 'doc_id'})[['doc_id', 'page_number']].assign(tribal=True)
    t = t.merge(trib, on=['doc_id', 'page_number'], how='left')
    t['tribal'] = t['tribal'].fillna(False).astype(bool)

    ca, oos = t[t.in_ca_scope], t[~t.in_ca_scope]
    ca.drop(columns=['tribal']).to_csv(A / 'table_level_metadata_published_ca.csv', index=False)
    oos.to_csv(OUT / 'out_of_state_tables.csv', index=False)

    pages_all = t.groupby(['doc_id', 'page_number']).in_ca_scope
    page_oos_only = (~pages_all.any()).sum()

    def block(d, label):
        return {f'{label}: tables': len(d), f'{label}: measurements': int(d.n_measurements.sum()),
                f'{label}: pages': d[['doc_id', 'page_number']].drop_duplicates().shape[0],
                f'{label}: documents': d.doc_id.nunique()}

    s = {}
    s.update(block(t, 'published (all)'))
    s.update(block(ca, 'California'))
    s.update(block(oos, 'out of state'))
    s.update(block(oos[~oos.tribal], 'out of state, not tribal'))
    s.update(block(oos[oos.tribal], 'out of state, tribal (withheld)'))
    s.update(block(ca[ca.tribal], 'California, tribal (withheld)'))
    s['out of state: pages with only out-of-state tables'] = int(page_oos_only)
    s['out of state: share of published measurements'] = round(oos.n_measurements.sum() / t.n_measurements.sum(), 4)
    s['out of state: tables located by a point'] = int(oos.oos_basis.eq('point').sum())
    s['out of state: tables unlocated, page names only other states'] = int(oos.oos_basis.eq('page').sum())
    s['out of state: year range (table dates)'] = f'{oos.year_start.min():.0f}-{oos.year_end.max():.0f}'
    sm = pd.Series(s, name='value')
    sm.to_csv(OUT / 'out_of_state_summary.csv', header=True)

    bw = (oos.groupby('water_type_final').agg(tables=('n_measurements', 'size'), measurements=('n_measurements', 'sum'),
                                             tribal_tables=('tribal', 'sum'))
             .sort_values('measurements', ascending=False))
    bw.to_csv(OUT / 'out_of_state_by_water_type.csv')
    bs = (oos.groupby(oos.state_name.fillna('(no point; page names other states)'))
             .agg(tables=('n_measurements', 'size'), measurements=('n_measurements', 'sum'),
                  documents=('doc_id', 'nunique'))
             .sort_values('measurements', ascending=False))
    bs.to_csv(OUT / 'out_of_state_by_state.csv')
    print(sm.to_string())
    print(bw.to_string())
    print(bs.head(20).to_string())


if __name__ == '__main__':
    main()
