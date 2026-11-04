"""Score the value counts (19_value_counts.py, recomputed live) against Annette's hand counts
of 20 non-daily tables (value_count_review_sample.csv, graded 2026-11-04).
Rows the grader marked as not measurements (blank or 0 count: station list, rock units,
records of other agencies) are reported separately; they are a water-type / content issue.
Output: data/analysis/value_counts/review/value_count_review_scored.csv"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / 'data/analysis/value_counts/review'
spec = importlib.util.spec_from_file_location('vc', ROOT / 'code/02_inventory/19_value_counts.py')
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)

s = pd.read_csv(D / 'value_count_review_sample.csv')
k = pd.read_csv(D / 'value_count_review_key.csv')[['review_no', 'n_values']].rename(columns={'n_values': 'n_values_first'})
s = s.merge(k, on='review_no')
s['n_values_now'] = [vc.metrics(ROOT / p)['n_values'] for p in s.table_csv]
s['truth'] = pd.to_numeric(s.grader_true_values, errors='coerce')
s['scored'] = s.truth > 0
s.to_csv(D / 'value_count_review_scored.csv', index=False)
pd.set_option('display.width', 200)
print(s[['review_no', 'water_type_final', 'truth', 'n_values_first', 'n_values_now']].to_string(index=False))
g = s[s.scored]
for c in ['n_values_first', 'n_values_now']:
    r = g[c] / g.truth
    print(f'{c}: total {g[c].sum()} vs true {g.truth.sum():.0f} (ratio {g[c].sum() / g.truth.sum():.2f}); '
          f'exact {int((g[c] == g.truth).sum())}/{len(g)}; within 10% {int(((r - 1).abs() <= 0.1).sum())}/{len(g)}; '
          f'median ratio {r.median():.2f}')
print('not measurements (grader):', s.loc[~s.scored, 'review_no'].tolist(), 'script counts', s.loc[~s.scored, 'n_values_now'].tolist())
