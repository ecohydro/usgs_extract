"""One page as the readers see it: Reducto blocks in reading order, tables numbered as their CSV files.

page_blocks(doc, page) -> list of (type, text)            (tables keep their HTML)
page_text(doc, page)   -> (text with "[TABLE k]" before each table, number of tables)
"""
import json
import re

import defaults as D


def page_blocks(doc, page):
    f = D.REDUCTO_JSON / str(doc) / f"{doc}_page_{page}.json"
    if not f.exists():
        f = D.REDUCTO_JSON_BY_PAGE / str(doc) / f"page_{page}" / f"{doc}_page_{page}.json"
    if not f.exists():
        return []
    chunks = json.load(open(f, encoding="utf-8")).get("result", {}).get("chunks", [])
    return [(b.get("type"), b.get("content") or "") for c in chunks for b in c.get("blocks", [])]


def compact_html(html):
    """Drop attributes and whitespace the model does not need (saves about a third of the tokens)."""
    html = re.sub(r"<(/?)(table|thead|tbody|tr|th|td)\b[^>]*>", r"<\1\2>", html)
    return re.sub(r"\s+", " ", html).strip()


def page_text(doc, page):
    out, k = [], 0
    for typ, txt in page_blocks(doc, page):
        if typ == "Table":
            k += 1
            out.append(f"[TABLE {k}]\n{compact_html(txt)}")
        elif txt.strip():
            out.append(txt.strip())
    return "\n".join(out), k


def plain(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
