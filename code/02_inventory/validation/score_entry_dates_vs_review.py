"""Score entry_dates.csv (15_entry_dates.py) against Annette's grades of the targeted
entry-date review sample (entry_date_review_sample.csv, graded 2026-11-04).

Truth = grader_true_start / grader_true_end. Row 3 (doc 51801 p37): grader said the table
years cannot be told apart and the 'Records available: 1941-1953' range is the best last
option, so truth = 1941-1953 (period-of-record fallback). Row 23 is an index page: truth = no date.
Output: data/analysis/date_parsing/entry_date_review/entry_date_review_scored.csv"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / 'data/analysis/date_parsing/entry_date_review'
pd.set_option('display.width', 250)
s = pd.read_csv(D / 'entry_date_review_sample.csv')
s.loc[s.review_no == 3, ['grader_true_start', 'grader_true_end']] = [1941, 1953]
e = pd.read_csv(ROOT / 'data/analysis/entry_dates.csv',
                usecols=['meta_row', 'year_start', 'year_end', 'date_source', 'date_tier'])
m = s[['review_no', 'category', 'meta_row', 'grader_true_start', 'grader_true_end',
       'year_start_llm', 'year_end_llm', 'new_start', 'new_end']].merge(e, on='meta_row', how='left')


def score(a, b, ts, te):
    if pd.isna(ts):
        return 'no_date_ok' if pd.isna(a) else 'gave_date'
    if pd.isna(a):
        return 'missed'
    if a == ts and b == te:
        return 'exact'
    if abs(a - ts) <= 1 and abs(b - te) <= 1:
        return 'within_1yr'
    if a == ts:
        return 'start_ok'
    return 'wrong'


m['score_llm'] = [score(*x) for x in zip(m.year_start_llm, m.year_end_llm, m.grader_true_start, m.grader_true_end)]
m['score_before'] = [score(*x) for x in zip(m.new_start, m.new_end, m.grader_true_start, m.grader_true_end)]
m['score_now'] = [score(*x) for x in zip(m.year_start, m.year_end, m.grader_true_start, m.grader_true_end)]
m.to_csv(D / 'entry_date_review_scored.csv', index=False)
print(m[['review_no', 'category', 'grader_true_start', 'grader_true_end', 'new_start', 'new_end',
         'year_start', 'year_end', 'date_tier', 'score_before', 'score_now']].to_string(index=False))
ok = ['exact', 'within_1yr', 'no_date_ok']
print({c: f"{m[c].isin(ok).sum()}/{len(m)} ok, {m[c].eq('exact').sum()} exact" for c in ['score_llm', 'score_before', 'score_now']})
