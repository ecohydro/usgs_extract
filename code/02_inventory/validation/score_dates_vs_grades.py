"""Score table-level dates (table_dates.csv) and the old LLM-string dates against
Annette's graded rows (date_validation_sample.csv; graded rows only).
Truth = grader_true_start / grader_true_end for the page; new method = union of the
page's tables. Output: data/analysis/date_parsing/date_method_validation.csv"""
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 90)
g = pd.read_csv(ROOT / 'data/analysis/date_parsing/date_validation_sample.csv', encoding='cp1252', dtype={'id': str})
g['id'] = g.id.str.replace(r'\.0$', '', regex=True)
G = 'grader_correct (Y/N)'
g[G] = g[G].astype(str).str.strip().str.upper()
g = g[g[G].isin(['Y', 'N']) | g.grader_true_start.notna()].copy()
td = pd.read_csv(ROOT / 'data/analysis/table_dates.csv', dtype={'id': str})
pgs = td.groupby(['id', 'page_number']).agg(
    new_start=('year_start', 'min'), new_end=('year_end', 'max'),
    tiers=('date_tier', lambda s: ','.join(sorted(set(s)))),
    quote=('date_quote', lambda s: ' || '.join(s.dropna())[:150])).reset_index()
m = g.merge(pgs, on=['id', 'page_number'], how='left')


def score(s, e, ts, te):
    if pd.isna(ts):
        return 'no_date_ok' if pd.isna(s) else 'no_date_but_gave'
    if pd.isna(s):
        return 'missed'
    if s == ts and e == te:
        return 'exact'
    if s == ts:
        return 'start_ok'
    if abs(s - ts) <= 1 and abs(e - te) <= 1:
        return 'within_1yr'
    return 'wrong'


m['score_new'] = [score(*x) for x in zip(m.new_start, m.new_end, m.grader_true_start, m.grader_true_end)]
m['score_old'] = [score(*x) for x in zip(m.year_start, m.year_end, m.grader_true_start, m.grader_true_end)]
m.to_csv(ROOT / 'data/analysis/date_parsing/date_method_validation.csv', index=False)
print(pd.crosstab(m.score_old, m.score_new, margins=True))
print(m[['sample_no', 'date_status', 'year_start', 'year_end', 'grader_true_start', 'grader_true_end',
         'new_start', 'new_end', 'tiers', 'score_new', 'quote']].to_string())

# per-tier summary (pages whose tables all used one tier)
dated = m[m.grader_true_start.notna()].copy()
dated['tier'] = dated.tiers.where(~dated.tiers.fillna('').str.contains(','), 'mixed')
dated['exact_or_1yr'] = dated.score_new.isin(['exact', 'within_1yr'])
tier = dated.groupby('tier').agg(n=('score_new', 'size'), exact=('score_new', lambda s: (s == 'exact').sum()),
                                 exact_or_within_1yr=('exact_or_1yr', 'sum')).reset_index()
tot = pd.DataFrame([{'tier': 'ALL (new)', 'n': len(dated), 'exact': (dated.score_new == 'exact').sum(),
                     'exact_or_within_1yr': dated.exact_or_1yr.sum()},
                    {'tier': 'ALL (old LLM string)', 'n': len(dated), 'exact': (dated.score_old == 'exact').sum(),
                     'exact_or_within_1yr': dated.score_old.isin(['exact', 'within_1yr']).sum()}])
tier = pd.concat([tier, tot])
tier.to_csv(ROOT / 'data/analysis/date_parsing/date_method_validation_by_tier.csv', index=False)
print(tier.to_string(index=False))
