"""
Minimal parser for Reducto's HTML <table> blocks (no lxml dependency, fast).

parse_table(html) -> (header_rows, body_rows)
    each row is a list of cell strings, colspan expanded (the text is repeated
    in every spanned column) and rowspan carried down into following rows.
    Leading rows made only of <th> cells are header rows.
column_headers(header_rows) -> one joined header string per column.
"""
import html as _html
import re

_TR = re.compile(r'<tr[^>]*>(.*?)</tr>', re.S | re.I)
_CELL = re.compile(r'<(th|td)([^>]*)>(.*?)</\1>', re.S | re.I)
_SPAN = re.compile(r'(col|row)span\s*=\s*"?(\d+)', re.I)
_TAG = re.compile(r'<[^>]+>')
_WS = re.compile(r'\s+')


def _clean(s):
    return _WS.sub(' ', _html.unescape(_TAG.sub(' ', s))).strip()


def parse_table(html):
    rows, kinds = [], []
    carry = {}  # col index -> [text, rows remaining]
    for tr in _TR.findall(html or ''):
        row, kind = [], []
        col = 0
        cells = _CELL.findall(tr)
        ci = 0
        while ci < len(cells) or col in carry:
            if col in carry:
                text, left = carry[col]
                row.append(text)
                kind.append('td')
                if left <= 1:
                    del carry[col]
                else:
                    carry[col][1] = left - 1
                col += 1
                continue
            tag, attrs, content = cells[ci]
            ci += 1
            spans = {k.lower(): int(v) for k, v in _SPAN.findall(attrs)}
            text = _clean(content)
            for _ in range(max(1, spans.get('col', 1))):
                row.append(text)
                kind.append(tag.lower())
                if spans.get('row', 1) > 1:
                    carry[col] = [text, spans['row'] - 1]
                col += 1
        rows.append(row)
        kinds.append(kind)
    n_head = 0
    for k in kinds:
        if k and all(t == 'th' for t in k):
            n_head += 1
        else:
            break
    return rows[:n_head], rows[n_head:]


def column_headers(header_rows):
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    out = []
    for c in range(width):
        parts = []
        for r in header_rows:
            if c < len(r) and r[c] and (not parts or parts[-1] != r[c]):
                parts.append(r[c])
        out.append(' '.join(parts))
    return out
