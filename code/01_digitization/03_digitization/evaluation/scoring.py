"""Scoring engine: one truth page against one tool's tables for that page.

score_page(truth_tables, output_tables, params) returns
    cells   one dict per printed number (truth numeric cell) with its outcome
    tables  one dict per truth table (found / missed / split / merged, GriTS-style scores)
    extras  one dict per output table (numbers that match nothing printed)

Outcome of a printed number (exactly one per cell; Anna 2026-11-14):
    right_findable     number right, row label right, column header right   <- the headline
    row_label_wrong    number right, column header right, row label wrong
    col_label_wrong    number right, row label right, column header wrong
    both_labels_wrong  number right, both wrong
    incorrect          a different number (or unreadable content) sits in that cell
    missed_cell        nothing in that cell, or its row / column is absent from the output
    missed_table       no output table corresponds to the printed table

How a tool's table is compared with a printed table: rows are lined up with rows and columns with
columns, each in printed order, so as to maximise the agreement of the cells (the two-step
alignment used by GriTS, Smock et al. 2022). No row or column is identified by its raw position.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

RANK = ["right_findable", "row_label_wrong", "col_label_wrong", "both_labels_wrong",
        "incorrect", "missed_cell", "missed_table"]

_LEADERS = re.compile(r"[.…_\-–—]{2,}|(?:\s[.\-_]){2,}")
_NUM = re.compile(r"\d[\d,]*\.?\d*|\.\d+")
_FRACTIONS = {"\u00bd": .5, "\u00bc": .25, "\u00be": .75, "\u2153": 1 / 3, "\u2154": 2 / 3,
              "\u215b": .125, "\u215c": .375, "\u215d": .625, "\u215e": .875}
_SPACED = re.compile(r"(?<=\d)\s*([.,])\s+(?=\d)")


def strip_leaders(s):
    return _LEADERS.sub(" ", s or "").strip()


def norm_text(s):
    """Label text with trivia removed: case, spacing, punctuation, dotted leaders."""
    s = unicodedata.normalize("NFKC", strip_leaders(s)).lower()   # SiO\u2082 = SiO2, 5L3\u00b9 = 5L31
    return re.sub(r"[^0-9a-z]+", "", s)  # also joins words hyphenated at a line break ("Diam- eter")


def parse_number(s, mark_max=2, repair_spaced=False):
    """(value, mark) if the cell is one number with at most `mark_max` other characters, else None."""
    s = strip_leaders(s)
    if repair_spaced:  # typesetting gaps ("21. 2", "4, 400") closed; a lone letter o read as zero
        s = _SPACED.sub(r"\1", s)
        if s.strip() in ("o", "O"):
            s = "0"
    frac = 0.0
    for ch, val in _FRACTIONS.items():   # "2½" is 2.5, not the number 2 with a mark
        if ch in s:
            frac += val
            s = s.replace(ch, " ")
    toks = list(_NUM.finditer(s))
    if frac and not toks:
        return frac, s.replace(" ", "")[:8]
    if len(toks) != 1:
        return None
    t = toks[0]
    rest = (s[:t.start()] + s[t.end():]).replace(" ", "")
    sign = 1.0
    before = s[:t.start()].rstrip()
    if before.endswith(("-", "−", "+")) and len(s[:t.start()]) - len(before) == 0:
        if before[-1] != "+":
            sign = -1.0
        rest = rest.replace(before[-1], "", 1)
    if len(rest) > mark_max:
        return None
    try:
        v = float(t.group().replace(",", ""))
    except ValueError:
        return None
    return sign * (v + frac), rest


@lru_cache(maxsize=500000)
def _sim(a, b):
    """Agreement of two cells for lining up rows and columns: 1 if the same, else partial."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    pa, pb = parse_number(a), parse_number(b)
    if pa and pb and pa[0] == pb[0]:
        return 1.0
    na, nb = norm_text(a), norm_text(b)
    if na and na == nb:
        return 1.0
    return 0.8 * SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _align(S):
    """Order-preserving one-to-one pairing of rows of S with columns of S that maximises the sum."""
    n, m = S.shape
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        row, prev, Si = dp[i], dp[i - 1], S[i - 1]
        for j in range(1, m + 1):
            best = prev[j - 1] + Si[j - 1]
            if prev[j] > best:
                best = prev[j]
            if row[j - 1] > best:
                best = row[j - 1]
            row[j] = best
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if dp[i, j] == dp[i - 1, j - 1] + S[i - 1, j - 1] and S[i - 1, j - 1] > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1], dp[n, m]


def align_grids(T, O):
    """Row pairs and column pairs between truth grid T and output grid O, plus the cell
    agreement matrix. Rows first (row-to-row agreement = best ordered pairing of their cells),
    then columns given the rows, then rows once more given the columns."""
    R, C, r, c = len(T), len(T[0]), len(O), len(O[0])
    sim = np.zeros((R, C, r, c), dtype=np.float32)
    strip_t = [[x.strip() for x in row] for row in T]
    strip_o = [[x.strip() for x in row] for row in O]
    for i in range(R):
        for j in range(C):
            a = strip_t[i][j]
            if not a:
                continue
            for k in range(r):
                for l in range(c):
                    b = strip_o[k][l]
                    if b:
                        sim[i, j, k, l] = _sim(a, b)
    W = np.zeros((R, r))
    for i in range(R):
        for k in range(r):
            W[i, k] = _align(sim[i, :, k, :])[1]
    rows, _ = _align(W)
    for _ in range(2):
        V = np.zeros((C, c))
        for i, k in rows:
            V += sim[i, :, k, :]
        cols, _ = _align(V)
        W = np.zeros((R, r))
        for j, l in cols:
            W += sim[:, j, :, l]
        rows, _ = _align(W)
    return dict(rows), dict(cols), sim


def _joined(parts):
    out = []
    for p in parts:
        p = p.strip()
        if p and (not out or out[-1] != p):
            out.append(p)
    return " ".join(out)


_FOOTNOTE = re.compile("[\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u1d43\u1d47\u1d9c\u1d48]")


def _labels_match(a, b, ignore_footnotes=True):
    """a = printed label, b = the tool's. A footnote superscript on the printed label may be
    kept (in any form) or dropped by the tool; either way the label is the same label."""
    if ignore_footnotes and _FOOTNOTE.search(a) and norm_text(_FOOTNOTE.sub("", a)) == norm_text(b):
        return True
    return norm_text(a) == norm_text(b)


def _label_col(j, label_cols):
    left = [c for c in label_cols if c < j]
    return max(left) if left else None


def compare(tt, og, P):
    """One truth table against one output grid. Returns per-truth-numeric-cell outcomes and
    summary numbers (n exactly agreeing cells, GriTS-style sums)."""
    T, H, L = tt["grid"], tt["n_header_rows"], tt["label_cols"]
    num = lambda s: parse_number(s, P["MARK_MAX_CHARS"], P["REPAIR_NUMBER_FORMAT"])
    rows, cols, sim = align_grids(T, og)
    body_out = [k for i, k in rows.items() if i >= H]
    first_body = min(body_out) if body_out else len(og)
    res = {}
    for i in range(H, len(T)):
        for j in range(len(T[0])):
            if j in L:
                continue
            t = num(T[i][j])
            if t is None:
                continue
            if i not in rows or j not in cols:
                res[(i, j)] = {"outcome": "missed_cell", "output_text": "", "why": "row or column absent"}
                continue
            k, l = rows[i], cols[j]
            otext = og[k][l].strip()
            o = num(otext)
            if o is None or o[0] != t[0]:
                if not strip_leaders(otext):
                    res[(i, j)] = {"outcome": "missed_cell", "output_text": otext, "why": "cell empty",
                                   "out_row": k, "out_col": l}
                else:
                    near = [og[kk][ll] for kk, ll in ((k, l - 1), (k, l + 1), (k - 1, l), (k + 1, l))
                            if 0 <= kk < len(og) and 0 <= ll < len(og[0])]
                    shifted = any((n := num(x)) is not None and n[0] == t[0] for x in near)
                    res[(i, j)] = {"outcome": "incorrect", "output_text": otext,
                                   "why": "shifted to a neighbouring cell" if shifted else "misread",
                                   "out_row": k, "out_col": l}
                continue
            lc = _label_col(j, L)
            if lc is None:
                row_ok = True
            else:
                row_ok = lc in cols and _labels_match(T[i][lc], og[k][cols[lc]], P["IGNORE_LABEL_FOOTNOTES"])
            want = _joined(T[h][j] for h in range(H))
            aligned = _joined(og[rows[h]][l] for h in range(H) if h in rows)
            above = _joined(og[kk][l] for kk in range(first_body))
            col_ok = H == 0 or any(_labels_match(want, cand, P["IGNORE_LABEL_FOOTNOTES"]) for cand in (aligned, above))  # no printed header: nothing to get wrong
            outcome = {(True, True): "right_findable", (False, True): "row_label_wrong",
                       (True, False): "col_label_wrong", (False, False): "both_labels_wrong"}[(row_ok, col_ok)]
            res[(i, j)] = {"outcome": outcome, "output_text": otext, "why": "", "out_row": k, "out_col": l,
                           "out_label_col": cols.get(lc) if lc is not None else None, "out_first_body_row": first_body,
                           "out_row_label": og[k][cols[lc]] if lc is not None and lc in cols else "",
                           "mark_kept": (unicodedata.normalize("NFKC", o[1]) == unicodedata.normalize("NFKC", t[1])) if t[1] else None}
    equal = sum(1 for i, k in rows.items() for j, l in cols.items() if sim[i, j, k, l] == 1.0)
    content = sum(float(sim[i, j, k, l]) / (1.0 if sim[i, j, k, l] == 1.0 else 0.8)
                  for i, k in rows.items() for j, l in cols.items())
    return res, {"equal": equal, "content": content, "matched_rows": len(rows), "matched_cols": len(cols),
                 "out_rows": len(og), "out_cols": len(og[0])}


def _filled(g):
    return sum(1 for r in g for x in r if x.strip())


def score_page(truth, outputs, P):
    num = lambda s: parse_number(s, P["MARK_MAX_CHARS"], P["REPAIR_NUMBER_FORMAT"])
    cells, tables = [], []
    used = {o["name"]: [] for o in outputs}          # output table -> truth tables it serves
    credited = {o["name"]: set() for o in outputs}   # output cells equal to a printed number
    # every printed table against every output table; then each output table is shared out:
    # the printed table it agrees with most claims its cells first, and an output cell already
    # claimed cannot also be credited to another printed table (a small table would otherwise
    # "find" its zeros inside an unrelated grid).
    pair = {}
    for ti, tt in enumerate(truth):
        for o in outputs:
            pair[(ti, o["name"])] = compare(tt, o["grid"], P)
    accepted = {}
    for o in outputs:
        claimed = set()
        order = sorted(range(len(truth)), key=lambda ti: -pair[(ti, o["name"])][1]["equal"])
        for rank, ti in enumerate(order):
            res, s = pair[(ti, o["name"])]
            need = max(P["LINK_MIN_EQUAL_CELLS"],
                       P["LINK_MIN_EQUAL_SHARE"] * min(_filled(truth[ti]["grid"]), _filled(o["grid"])))
            for v in res.values():
                if v["outcome"] in RANK[:4] and (v["out_row"], v["out_col"]) in claimed:
                    v.update(outcome="missed_cell", why="output cell belongs to another printed table")
            right = [v for v in res.values() if v["outcome"] in RANK[:4]]
            if (s["equal"] if rank == 0 else len(right)) < need:
                continue
            claimed.update((v["out_row"], v["out_col"]) for v in right)
            accepted[(ti, o["name"])] = (res, s)
    for ti, tt in enumerate(truth):
        T, H, L = tt["grid"], tt["n_header_rows"], tt["label_cols"]
        best, pieces, links = {}, {}, []
        for o in outputs:
            if (ti, o["name"]) not in accepted:
                continue
            res, s = accepted[(ti, o["name"])]
            links.append((o["name"], s))
            got = 0
            for key, v in res.items():
                if v["outcome"] in RANK[:4]:
                    got += 1
                if key not in best or RANK.index(v["outcome"]) < RANK.index(best[key]["outcome"]):
                    best[key] = {**v, "output_table": o["name"]}
            pieces[o["name"]] = got
        n_num = 0
        for i in range(H, len(T)):
            for j in range(len(T[0])):
                if j in L or num(T[i][j]) is None:
                    continue
                n_num += 1
                v = best.get((i, j), {"outcome": "missed_table" if not links else "missed_cell",
                                      "output_text": "", "why": "no output table", "output_table": ""})
                lc = _label_col(j, L)
                cells.append({"table_no": tt["table_no"], "row": i, "col": j, "truth_text": T[i][j],
                              "row_label": T[i][lc] if lc is not None else "",
                              "col_header": _joined(T[h][j] for h in range(H)),
                              "has_mark": bool(num(T[i][j])[1]), **v})
        big = [n for n, g in pieces.items() if n_num and g >= P["SPLIT_MERGE_MIN_SHARE"] * n_num]
        for n in big:
            used[n].append(tt["table_no"])
        top = max(links, key=lambda x: x[1]["equal"])[1] if links else None
        R, C = len(T), len(T[0])
        tables.append({
            "table_no": tt["table_no"], "n_numbers": n_num, "n_output_tables_linked": len(links),
            "n_pieces": len(big), "truth_rows": R, "truth_cols": C,
            # GriTS-style, own implementation on span-expanded grids, best linked output table
            "layout_recall": (top["matched_rows"] * top["matched_cols"]) / (R * C) if top else 0.0,
            "layout_precision": (top["matched_rows"] * top["matched_cols"]) / (top["out_rows"] * top["out_cols"]) if top else 0.0,
            "content_recall": top["content"] / (R * C) if top else 0.0,
            "content_precision": top["content"] / (top["out_rows"] * top["out_cols"]) if top else 0.0,
        })
    for t in tables:
        merged = any(len(v) > 1 and t["table_no"] in v for v in used.values())
        t["status"] = ("missed" if t["n_output_tables_linked"] == 0 else
                       "split" if t["n_pieces"] > 1 else "merged" if merged else "found")
    # numbers in the output that match nothing printed
    extras = []
    truth_values = {}
    for tt in truth:
        for row in tt["grid"]:
            for x in row:
                n = num(x)
                if n is not None:
                    truth_values[n[0]] = truth_values.get(n[0], 0) + 1
    for o in outputs:
        vals = [n[0] for row in o["grid"] for x in row if (n := num(x)) is not None]
        left, unmatched = dict(truth_values), 0
        for v in vals:
            if left.get(v, 0) > 0:
                left[v] -= 1
            else:
                unmatched += 1
        extras.append({"output_table": o["name"], "n_output_numbers": len(vals),
                       "n_not_printed": unmatched, "linked": bool(used[o["name"]])})
    return cells, tables, extras
