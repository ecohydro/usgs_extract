"""Reader 2: the open model reads each page of a set and answers per numbered table.

  ~/miniconda3/envs/vllm-rs/bin/python code/05_evaluation/metadata/05_model_reader.py validation [prompt_version]

Writes model_runs/{set}_{prompt}/answers.jsonl: one line per page (doc_id, page_number, n_tables,
raw answer, parsed answer or null). Pages already in the file are skipped, so a run can be resumed.
Needs a GPU. The model code is in reader.py.
"""
import csv
import json
import sys
import time

import defaults as D
import page_input
import reader


def main(set_name, version):
    out_dir = D.MODEL_RUNS / f"{set_name}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "answers.jsonl"
    done = set()
    if out.exists():
        done = {(r["doc_id"], r["page_number"]) for r in map(json.loads, open(out))}
    pages = sorted({(int(r["doc_id"]), int(r["page_number"])) for r in csv.DictReader(open(D.SETS[set_name]["tables"]))})
    todo = []
    for doc, page in pages:
        if (doc, page) in done:
            continue
        text, k = page_input.page_text(doc, page)
        if k:
            todo.append((doc, page, k, text))
    print(f"{len(todo)} pages to read, {len(done)} already done")
    if not todo:
        return
    llm = reader.load()
    t0 = time.time()
    rows = reader.read(llm, todo, version)
    dt = time.time() - t0
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"RESULT pages={len(todo)} seconds={dt:.0f} per_page={dt / len(todo):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else D.PROMPT_VERSION)
