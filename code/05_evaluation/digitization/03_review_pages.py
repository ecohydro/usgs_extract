"""Build the pages Anna uses to check the typed truth against the scans.

Produces (data/analysis/evaluation/digitization/review/):
    index.html and one {doc}_page_{N}.html per page: the scan beside the typed tables. Every cell
      can be clicked and retyped; review_server.py saves the change into the truth CSV and logs it
      in edit_log.csv. Colours:
        yellow  the typist was unsure (hover shows the alternative reading),
        orange  raw Reducto holds a different number there (hover shows Reducto's value),
        green   already changed by the human check (hover shows the old value).
      Flags only direct the eye: the scan decides, never the tool.
    flagged_cells.csv  every yellow / orange cell, for reference
Needs scores/cells.csv (run 02_score.py first). Serve with review_server.py.
Run:  ~/miniconda3/envs/geofm-vlm/bin/python code/05_evaluation/digitization/03_review_pages.py
"""
import csv
import html

import pandas as pd

import defaults as D
import loaders

CSS = """body{font:13px sans-serif;margin:0;display:flex;height:100vh}
#scan{flex:1;overflow:auto;border-right:2px solid #888}#scan img{width:160%}
#typed{flex:1;overflow:auto;padding:8px}table{border-collapse:collapse;margin:6px 0 18px}
td{border:1px solid #bbb;padding:1px 5px;white-space:nowrap}tr.h td{background:#e8e8e8;font-weight:bold}
td.l{background:#f4f4f4}td.u{background:#ffe95c}td.d{background:#ffb37a}td.u.d{background:#ff8f8f}
td.e{background:#9be29b !important}td.err{outline:3px solid red}td:focus{outline:2px solid #06c;background:#fff !important}
td{min-width:1.2em;height:1.2em}#bar{position:sticky;top:0;background:#fff;padding:4px 0;border-bottom:1px solid #ccc;z-index:2}
.k span{padding:1px 6px;margin-right:8px}"""


JS = """
document.querySelectorAll('td[contenteditable]').forEach(td=>{
  td.dataset.orig=td.textContent;
  td.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();td.blur();}});
  td.addEventListener('blur',()=>{
    const v=td.textContent.trim(); if(v===td.dataset.orig) return;
    fetch('/save',{method:'POST',body:JSON.stringify({doc:DOC,page:PAGE,table:+td.dataset.t,row:+td.dataset.r,col:+td.dataset.c,value:v})})
     .then(r=>r.json()).then(j=>{ if(j.ok){td.title='changed by you; was: '+(td.dataset.first||td.dataset.orig||'(empty)');
        td.dataset.first=td.dataset.first||td.dataset.orig; td.dataset.orig=v; td.classList.add('e'); td.classList.remove('err');}
        else{td.classList.add('err'); td.title='NOT SAVED: '+j.error;} })
     .catch(()=>{td.classList.add('err'); td.title='NOT SAVED: server not reachable';});
  });
});
document.getElementById('done').addEventListener('change',e=>{
  fetch('/checked',{method:'POST',body:JSON.stringify({doc:DOC,page:PAGE,checked:e.target.checked})});
});
"""


def main():
    rev = D.OUT / "review"
    rev.mkdir(exist_ok=True)
    sample = pd.read_csv(D.SAMPLE)
    cells = pd.read_csv(D.OUT / "scores" / "cells.csv", keep_default_na=False)
    cells = cells[(cells.system == "reducto_raw") & (cells.variant == "default")]
    wrong = cells[cells.outcome.isin(["incorrect", "missed_cell"])]
    flagged, index = [], []
    lf, cf = rev / "edit_log.csv", rev / "checked_pages.csv"
    log = pd.read_csv(lf, keep_default_na=False) if lf.exists() else pd.DataFrame(
        columns=["time", "doc_id", "page_number", "table_no", "row", "col", "old", "new"])
    checked = set()
    if cf.exists():
        last = pd.read_csv(cf).groupby(["doc_id", "page_number"]).checked.last()
        checked = {k for k, v in last.items() if v}
    for s in sample.itertuples():
        doc, page = s.doc_id, s.page_number
        if not (D.TRUTH / f"{doc}_page_{page}_tables.csv").exists():
            continue
        unsure = {}
        f = D.TRUTH / f"{doc}_page_{page}_uncertain.csv"
        if f.exists():
            for u in csv.DictReader(open(f, encoding="utf-8")):
                unsure[(int(u["table_no"]), int(u["row"]), int(u["col"]))] = u
        edits = {(int(r.table_no), int(r.row), int(r.col)): r.old for r in log.itertuples()
                 if r.doc_id == doc and r.page_number == page}
        # first old value per cell = what the typist had
        first = {}
        for r in log.itertuples():
            if r.doc_id == doc and r.page_number == page:
                first.setdefault((int(r.table_no), int(r.row), int(r.col)), r.old)
        edits = first
        w = wrong[(wrong.doc_id == doc) & (wrong.page_number == page)]
        differ = {(r.table_no, r.row, r.col): r.output_text for r in w.itertuples()}
        parts = [f"<h3>Sample {s.sample_no}: report {doc}, page {page} ({s.kind})</h3>",
                 "<p class=k><span class='u' style='background:#ffe95c'>typist unsure</span>"
                 "<span style='background:#ffb37a'>Reducto reads a different number</span>"
                 "<span style='background:#ff8f8f'>both</span><span style='background:#9be29b'>changed by you</span>"
                 " Click any cell and type to correct it; it saves when you leave the cell. Hover a coloured cell for details.</p>",
                 f"<div id=bar><label><input type=checkbox id=done {'checked' if (doc, page) in checked else ''}> "
                 "I have checked this page</label> &nbsp; <a href='index.html'>all pages</a></div>"]
        for tt in loaders.truth_tables(doc, page):
            k, H, L = tt["table_no"], tt["n_header_rows"], tt["label_cols"]
            parts.append(f"<b>Table {k}.</b> {html.escape(tt['title'])}<table>")
            for i, row in enumerate(tt["grid"]):
                parts.append("<tr class=h>" if i < H else "<tr>")
                for j, x in enumerate(row):
                    cls, tip = (["l"] if j in L else []), []
                    key = (k, i, j)
                    if key in unsure:
                        cls.append("u")
                        tip.append(f"typist unsure; could be: {unsure[key]['alternative']} ({unsure[key]['reason']})")
                    if key in differ:
                        cls.append("d")
                        tip.append(f"Reducto: {differ[key] or '(empty)'}")
                    if key in edits:
                        cls.append("e")
                        tip.append(f"changed by you; was: {edits[key] or '(empty)'}")
                    if key in unsure or key in differ:
                        lc = max([c for c in L if c < j], default=None)
                        flagged.append({
                            "sample_no": s.sample_no, "doc_id": doc, "page_number": page, "table_no": k,
                            "row_label": row[lc] if lc is not None else "", "row": i, "col": j,
                            "column_header": " ".join(dict.fromkeys(t[j] for t in tt["grid"][:H] if t[j])),
                            "typed": x, "typist_alternative": unsure.get(key, {}).get("alternative", ""),
                            "reducto": differ.get(key, ""), "correct_value": ""})
                    parts.append(f"<td contenteditable data-t={k} data-r={i} data-c={j} class='{' '.join(cls)}' title='{html.escape(' | '.join(tip), quote=True)}'>"
                                 f"{html.escape(x)}</td>")
                parts.append("</tr>")
            parts.append("</table>")
        src = f"../pages/{doc}_page_{page}.png"
        if (doc, page) in D.REVIEW_ROTATE_CW:   # a turned copy for reading; pages/ is untouched
            from PIL import Image
            turned = Image.open(D.OUT / "pages" / f"{doc}_page_{page}.png").rotate(
                -D.REVIEW_ROTATE_CW[(doc, page)], expand=True)
            turned.save(rev / f"{doc}_page_{page}_turned.png")
            src = f"{doc}_page_{page}_turned.png"
        name = f"{doc}_page_{page}.html"
        (rev / name).write_text(
            f"<!doctype html><meta charset=utf-8><title>{doc} p{page}</title><style>{CSS}</style>"
            f"<div id=scan><img src='{src}'></div><div id=typed>{''.join(parts)}</div>"
            f"<script>const DOC={doc},PAGE={page};{JS}</script>",
            encoding="utf-8")
        n_edit = len(edits)
        index.append(f"<li><a href='{name}'>Sample {s.sample_no}: report {doc}, page {page}</a> ({s.kind})"
                     f"{' &#10003; checked' if (doc, page) in checked else ''}{f', {n_edit} cells changed' if n_edit else ''}</li>")
    (rev / "index.html").write_text("<!doctype html><meta charset=utf-8><h3>Truth set check</h3><ol>"
                                    + "".join(index) + "</ol>", encoding="utf-8")
    pd.DataFrame(flagged).to_csv(rev / "flagged_cells.csv", index=False)
    print(len(index), "pages;", len(flagged), "flagged cells")


if __name__ == "__main__":
    main()
