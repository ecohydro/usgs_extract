"""Draw scorer verdicts for a hand check, and print what is needed to judge each one.

A verdict = the outcome the scorer gave one printed number for one output. The draw is spread
evenly over (output, outcome) groups so rare outcomes are checked too; the error rate is then
weighted back by group size (see 05 / notes).
Produces  scores/verdict_check_sample.csv  (the draw; a `checker_verdict` / `checker_note` pair is
          filled in by hand in scores/verdict_check_graded.csv)
Run:      python 04_verdict_check.py            draw + save
          python 04_verdict_check.py show A B   print context for sample rows A..B (0-based, end excl.)
"""
import sys

import numpy as np
import pandas as pd

import defaults as D
import loaders
import scoring

OUT = D.OUT / "scores"


def draw():
    c = pd.read_csv(OUT / "cells.csv", keep_default_na=False)
    c = c[c.variant == D.VERDICT_CHECK_VARIANT].reset_index(drop=True)
    rng = np.random.default_rng(D.SEED)
    groups = [g for _, g in c.groupby(["system", "outcome"], sort=True)]
    take = {i: 0 for i in range(len(groups))}
    left = D.VERDICT_CHECK_N
    while left > 0:                      # one per group per round, until a group runs out
        for i, g in enumerate(groups):
            if left > 0 and take[i] < len(g):
                take[i] += 1
                left -= 1
    picks = [g.iloc[rng.permutation(len(g))[:take[i]]] for i, g in enumerate(groups)]
    s = pd.concat(picks).reset_index(drop=True)
    s["group_size"] = s.groupby(["system", "outcome"]).outcome.transform(
        lambda x: len(c[(c.system == x.name[0]) & (c.outcome == x.name[1])]))
    s.to_csv(OUT / "verdict_check_sample.csv", index=False)
    print(s.groupby(["system", "outcome"]).agg(drawn=("row", "size"), group_size=("group_size", "first")).to_string())


def short(x, n=14):
    x = x.strip()
    return x if len(x) <= n else x[:n - 1] + "~"


def show(a, b):
    s = pd.read_csv(OUT / "verdict_check_sample.csv", keep_default_na=False)
    for n, r in s.iloc[a:b].iterrows():
        tt = next(t for t in loaders.truth_tables(r.doc_id, r.page_number) if t["table_no"] == r.table_no)
        T = tt["grid"]
        print(f"\n=== #{n} {r.system} | {r.doc_id} p{r.page_number} t{r.table_no} | VERDICT {r.outcome}"
              f" ({r.why}) | scorer saw output cell: [{r.output_text}]")
        print(f"  truth: row label [{r.row_label}]  header [{r.col_header}]  value [{r.truth_text}]  (grid row {r.row}, col {r.col})")
        print("  truth row :", " | ".join(short(x) for x in T[r.row]))
        if r.output_table == "" or r.out_row == "":
            for o in loaders.output_tables(r.system, r.doc_id, r.page_number):
                g = o["grid"]
                lab = scoring.norm_text(r.row_label)
                hits = [k for k, row in enumerate(g) if lab and any(lab in scoring.norm_text(x) for x in row)][:3]
                print(f"  output {o['name']} ({len(g)}x{len(g[0])}):")
                for k in list(range(min(3, len(g)))) + hits:
                    print(f"     r{k}:", " | ".join(short(x) for x in g[k]))
            continue
        o = next(o for o in loaders.output_tables(r.system, r.doc_id, r.page_number) if o["name"] == r.output_table)
        g, k, l = o["grid"], int(float(r.out_row)), int(float(r.out_col))
        fb = int(float(r.out_first_body_row)) if r.out_first_body_row != "" else 3
        print(f"  output {o['name']} ({len(g)}x{len(g[0])}); scorer looked at row {k}, col {l}")
        print(f"     header above that column: {[g[x][l] for x in range(min(fb, 4))]}")
        if r.out_label_col != "":
            print(f"     row label cell: [{g[k][int(float(r.out_label_col))]}]")
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < len(g):
                print(f"     r{kk}:", " | ".join(("»" if x == l and kk == k else "") + short(c) for x, c in enumerate(g[kk])))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show(int(sys.argv[2]), int(sys.argv[3]))
    else:
        draw()
