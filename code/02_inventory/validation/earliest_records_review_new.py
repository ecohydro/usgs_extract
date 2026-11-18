"""
Earliest-record review list on the new metadata (2026-11-18).

  python earliest_records_review_new.py
    -> data/analysis/manual_review/earliest_records_new/earliest_pages_review.csv  (+ pngs/, git-ignored)

One row per page x water type, from two sources:
  scope    records predating NWIS (08_scope_extension_stats.py -> scope_earliest_records_review.csv;
           CA, dated, precise/approximate, tribal pages excluded)
  fig1     catalog tables in California (core water types, data pages, measurement dates) starting before
           EARLY_YEAR -- the early bars of Figure 1
Earlier verdicts on the same page (any water type) (Phi-4 era, manual_review/earliest_records_review_phi4.csv)
are copied in as a hint only; they are not applied. Grader columns are blank: verdict (keep / drop / retype /
undate / construction_year / set_year), new_water_type, corrected_year_start, corrected_year_end, note.
When graded, save as manual_review/earliest_records_review.csv (paper_filters.apply_review reads that name).
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / 'data/analysis'
OUT = A / 'manual_review/earliest_records_new'
EARLY_YEAR = 1890
CORE = ['Stream Discharge', 'Groundwater', 'Reservoir', 'Irrigation', 'Springs', 'Precipitation', 'Water Quality']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(A / 'scope_earliest_records_review.csv')
    s = (s.groupby(['id', 'page_number', 'water_type'])
          .agg(year_min=('year_start', 'min'), year_max=('year_end', 'max'), n_records=('meta_row', 'size'),
               names=('watersource_name', lambda x: '; '.join(sorted({str(v) for v in x if pd.notna(v)})[:5])))
          .reset_index().assign(source='scope'))
    c = pd.read_csv(ROOT / 'data/catalog/restored_tables_catalog.csv', low_memory=False)
    c = c[c.water_type.isin(CORE) & c.in_california.eq(True) & c.page_type.eq('data') & (c.year_start < EARLY_YEAR)]
    f = (c.rename(columns={'doc_id': 'id'}).groupby(['id', 'page_number', 'water_type'])
          .agg(year_min=('year_start', 'min'), year_max=('year_end', 'max'), n_records=('table_number', 'size'),
               names=('watersource_name', lambda x: '; '.join(sorted({str(v) for v in x if pd.notna(v)})[:5])),
               measurements=('n_measurements', 'sum'), title=('doc_title', 'first'))
          .reset_index().assign(source='fig1'))
    r = pd.concat([s, f])
    r = (r.groupby(['id', 'page_number', 'water_type'])
          .agg(source=('source', lambda x: '+'.join(sorted(set(x)))), year_min=('year_min', 'min'),
               year_max=('year_max', 'max'), n_records=('n_records', 'max'), names=('names', 'first'),
               measurements=('measurements', 'max'), title=('title', 'first'))
          .reset_index().sort_values(['water_type', 'year_min', 'id', 'page_number']))
    old = pd.read_csv(A / 'manual_review/earliest_records_review_phi4.csv')
    # matched by page only: water types changed with the new metadata
    old = (old.assign(t=old.water_type + ': ' + old.action.astype(str) + ' (' + old.note.fillna('').astype(str) + ')')
              .groupby(['id', 'page_number'])['t'].agg(' | '.join).rename('phi4_era_verdict').reset_index())
    r = r.merge(old, on=['id', 'page_number'], how='left')
    r.insert(0, 'review_no', range(1, len(r) + 1))
    png = []
    for _, x in r.iterrows():
        p = ROOT / f'data/digitized/{x.id}/page_{x.page_number}/{x.id}_page_{x.page_number}.png'
        dst = OUT / 'pngs' / f'{x.review_no:03d}_{x.id}_p{x.page_number}.png'
        if p.exists():
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
            png.append(dst.name)
        else:
            png.append('')
    r['png'] = png
    for col in ['verdict', 'new_water_type', 'corrected_year_start', 'corrected_year_end', 'note']:
        r[col] = ''
    dst = OUT / 'earliest_pages_review.csv'
    if dst.exists():   # never overwrite the file the reviewer is grading; write a dated rerun copy instead
        dst = OUT / 'earliest_pages_review_rerun.csv'
        print(f'review file exists (kept as is); new list -> {dst.name}')
    r.to_csv(dst, index=False)
    print(f'pages to review: {len(r)} (no png: {(r.png == "").sum()})')
    print(r.groupby(['water_type', 'source']).size().to_string())
    print('with a Phi-4-era verdict:', int(r.phi4_era_verdict.notna().sum()))


if __name__ == '__main__':
    main()
