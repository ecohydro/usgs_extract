"""Turns the list's printed page numbers into PDF page positions.

Produces OUT/page_map.csv (one row per listed page: report, printed page, pdf page, how it was found)
and OUT/page_map_reports.csv (one row per report). Run: python page_offsets.py
Method: the text layer of each PDF page is searched for a bare number in its first/last lines (the
printed page number). Pages whose number is corroborated by neighbours are 'trusted'. A listed printed
page is placed by the trusted pages just before and after it; when those two imply different shifts
(an unnumbered plate between them) both candidates are kept for a visual check.
"""
import re, collections
import pandas as pd, pymupdf
import defaults as D

METHODS = ["printed_on_page", "between_agreeing_pages", "between_disagreeing_pages", "extrapolated", "not_found"]


def parse_pages(text):
    """'88-90; 96, 105-107' -> [88, 89, 90, 96, 105, 106, 107]; None if not numeric (e.g. 'throughout')."""
    pages = []
    for part in re.split(r"[;,]", str(text)):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pages.extend(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            pages.append(int(part))
        else:
            return None
    return sorted(set(pages))


def printed_numbers(page):
    """Candidate printed page numbers of a PDF page: a bare 1-4 digit number among its first/last
    HEADER_LINES text lines, or a number opening/closing a running head ('46 REPUBLICAN VALLEY', 'AURORA COUNTY. 65')."""
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    found = set()
    for line in lines[:D.HEADER_LINES] + lines[-D.HEADER_LINES:]:
        m = re.fullmatch(r"[-. ]*(\d{1,4})[-. ]*", line)
        if m:
            found.add(int(m.group(1)))
    for line in lines[:D.HEADER_LINES]:
        m = re.fullmatch(r"(\d{1,4})\s+[A-Z][A-Z ,.'\-]{8,}", line) or re.fullmatch(r"[A-Z][A-Z ,.'\-]{8,}\s(\d{1,4})", line)
        if m:
            found.add(int(m.group(1)))
    return found


def anchors(doc):
    """PDF pages whose printed number is trusted: [(pdf_page, printed_number)], increasing in both.

    A (page, number) pair is kept when ANCHOR_SUPPORT other pages within ANCHOR_WINDOW share its shift
    (drops table values that happen to sit on a line alone); the longest chain that increases in both
    page and number is then kept (drops what is left of the noise).
    """
    n = len(doc)
    pairs = [(q, x) for q in range(1, n + 1) for x in printed_numbers(doc[q - 1])]
    by_shift = collections.defaultdict(list)
    for q, x in pairs:
        by_shift[q - x].append(q)
    good = [(q, x) for q, x in pairs
            if sum(1 for v in by_shift[q - x] if v != q and abs(v - q) <= D.ANCHOR_WINDOW) >= D.ANCHOR_SUPPORT]
    good.sort()
    if not good:
        return [], n
    best = [1] * len(good); prev = [-1] * len(good)
    for i in range(len(good)):
        for j in range(i):
            if good[j][0] < good[i][0] and good[j][1] < good[i][1] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(len(good)), key=best.__getitem__)
    chain = []
    while i != -1:
        chain.append(good[i]); i = prev[i]
    return chain[::-1], n


def map_report(pdf_path, printed_pages):
    """Each listed printed page -> PDF page, from the trusted pages just before and after it."""
    doc = pymupdf.open(pdf_path)
    anc, n = anchors(doc)
    rows = []
    for p in printed_pages:
        before = [(q, x) for q, x in anc if x <= p]
        after = [(q, x) for q, x in anc if x >= p]
        lo = before[-1] if before else None
        hi = after[0] if after else None
        row = dict(printed_page=p, pdf_page=None, shift=None, method="not_found",
                   anchor_before=lo[1] if lo else None, anchor_after=hi[1] if hi else None, pdf_page_alt=None)
        if lo and lo[1] == p:
            row.update(pdf_page=lo[0], shift=lo[0] - p, method="printed_on_page")
        elif lo and hi and lo[0] - lo[1] == hi[0] - hi[1]:
            row.update(pdf_page=p + lo[0] - lo[1], shift=lo[0] - lo[1], method="between_agreeing_pages")
        elif lo and hi:      # an unnumbered page (plate, map) was inserted somewhere between the two
            row.update(pdf_page=p + lo[0] - lo[1], shift=lo[0] - lo[1], pdf_page_alt=p + hi[0] - hi[1],
                       method="between_disagreeing_pages")
        elif lo or hi:
            q, x = lo or hi
            if 1 <= p + q - x <= n:
                row.update(pdf_page=p + q - x, shift=q - x, method="extrapolated")
        rows.append(row)
    return rows, n


def apply_overrides(pm):
    """page_map with the hand-entered corrections (page numbers read off the scans) applied; pages the
    visual check confirmed keep their method, corrected ones become 'read_off_scan'."""
    final = pm.copy()
    final["pdf_page_alt"] = None
    ov = pd.read_csv(D.OVERRIDES, dtype={"report": str})
    for o in ov.itertuples():
        m = (final.report == o.report) & final.printed_page.between(o.printed_from, o.printed_to)
        final.loc[m, "pdf_page"] = final.loc[m, "printed_page"] + o.shift
        final.loc[m, "shift"] = o.shift
        final.loc[m, "method"] = "read_off_scan"
    return final


def main():
    lst = pd.read_excel(D.LIST_XLSX, sheet_name=D.LIST_SHEET)
    out, reports = [], []
    for _, r in lst.iterrows():
        rid = str(r["Publication ID"])
        pages = parse_pages(r["Pages"])
        pdf = D.VAL_PDFS / f"{rid}.pdf"
        status = "ok"
        if pages is None:
            status = "pages_not_numeric"
        elif not pdf.exists():
            status = "no_pdf"
        if status != "ok":
            reports.append(dict(report=rid, title=r["Title"], pages_text=r["Pages"], status=status))
            continue
        rows, n = map_report(pdf, pages)
        for x in rows:
            x.update(report=rid)
        out.extend(rows)
        df = pd.DataFrame(rows)
        shifts = df["shift"].dropna()
        reports.append(dict(report=rid, title=r["Title"], pages_text=r["Pages"], status=status, pdf_pages=n,
                            n_listed=len(df), **{"n_" + m: int((df.method == m).sum()) for m in METHODS},
                            shifts=";".join(str(int(s)) for s in sorted(shifts.unique()))))
    D.OUT.mkdir(parents=True, exist_ok=True)
    pm = pd.DataFrame(out)[["report", "printed_page", "pdf_page", "pdf_page_alt", "shift", "method", "anchor_before", "anchor_after"]]
    pm.to_csv(D.OUT / "page_map.csv", index=False)
    final = apply_overrides(pm)
    final.to_csv(D.OUT / "page_map_final.csv", index=False)
    print(final.method.value_counts().to_string())
    rp = pd.DataFrame(reports)
    rp.to_csv(D.OUT / "page_map_reports.csv", index=False)
    print(rp.status.value_counts().to_string())
    print(pm.method.value_counts().to_string())


if __name__ == "__main__":
    main()
