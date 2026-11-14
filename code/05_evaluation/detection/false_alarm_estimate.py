"""Corrected false-alarm rate of the detector in the real corpus.

Reducto's answer (table back / no table back) is known for every flagged page; a blind look at a sample
of each group says how often that answer is wrong. For threshold t:
  false alarms(t) = share(no table back) x P(really no table | no table back)
                  + share(table back)    x P(really no table | table back)
with shares from all flagged pages scoring >= t and the two P's from the sampled pages scoring >= t.
'anna_ruling' is the result of record (borderline pages are tables except those in
BORDERLINE_NOT_TABLE); the two other rows count every borderline page one way or the other; pages with no image are left out. Range: 2.5-97.5% of
N_BOOTSTRAP resamples of the labelled pages within each group.
Produces OUT/false_alarm_estimate.csv. Run: python false_alarm_estimate.py
"""
import numpy as np, pandas as pd
import defaults as D

F = D.OUT / "false_alarm_check"


def labelled():
    a = pd.read_csv(F / "key.csv").merge(pd.read_csv(F / "sample_labeled.csv")[["n", "label", "note"]], on="n")
    b = pd.read_csv(F / "key_extra.csv").merge(pd.read_csv(F / "sample_extra_labeled.csv")[["n", "label", "note"]], on="n")
    d = pd.concat([a, b])
    return d[~d.note.str.contains("no image", case=False, na=False)]


def main():
    lab = labelled()
    pop = pd.read_csv(D.OUT / "corpus_flagged_pages.csv").dropna(subset=["best_score"])
    rng = np.random.default_rng(D.SEED)
    rows = []
    for t in D.THRESHOLDS_USED:
        p, l = pop[pop.best_score >= t], lab[lab.best_score >= t]
        w_no = (~p.reducto_table).mean()
        ruled = l.assign(label=l.label.where(~l.page.isin(D.BORDERLINE_NOT_TABLE), "no_table"))
        for name, src, bad in (("anna_ruling", ruled, ["no_table"]),
                               ("borderline_counts_as_table", l, ["no_table"]),
                               ("borderline_counts_as_no_table", l, ["no_table", "borderline"])):
            g_no = src[~src.reducto_table].label.isin(bad).to_numpy()
            g_yes = src[src.reducto_table].label.isin(bad).to_numpy()
            est = w_no * g_no.mean() + (1 - w_no) * g_yes.mean()
            boot = [w_no * rng.choice(g_no, len(g_no)).mean() + (1 - w_no) * rng.choice(g_yes, len(g_yes)).mean()
                    for _ in range(D.N_BOOTSTRAP)]
            rows.append(dict(threshold=t, borderline_rule=name, flagged_pages=len(p), reducto_no_table_share=w_no,
                             looked_no_table_group=len(g_no), really_no_table_there=int(g_no.sum()),
                             looked_table_group=len(g_yes), really_no_table_there_2=int(g_yes.sum()),
                             false_alarm_rate=est, low=np.percentile(boot, 2.5), high=np.percentile(boot, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(D.OUT / "false_alarm_estimate.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
