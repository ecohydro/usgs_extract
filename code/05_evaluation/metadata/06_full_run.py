"""The full run: every page with a published table goes through the open model (prompt defaults.PROMPT_VERSION).

  python 06_full_run.py prepare     any Python; reads the Reducto page files once and writes the shard inputs
  python 06_full_run.py worker      vllm-rs Python on a GPU; any number of workers, on any nodes, at any time
  python 06_full_run.py status

Layout under data/analysis/evaluation/metadata/full_run/:
  pages.csv                 doc_id, page_number, n_tables, shard
  in/shard_0000.jsonl.gz    the page texts of one shard (so workers never touch the slow data share)
  out/shard_0000.jsonl      the model's answers (written whole, then renamed: a file that exists is complete)
  claims/shard_0000/        a worker is on this shard; a claim older than CLAIM_STALE_MIN is taken over
Every worker runs the same loop: claim a free shard, read it, write it, repeat; exit when every shard has its out file.
"""
import csv
import gzip
import json
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import defaults as D
import page_input

RUN = D.OUT / "full_run"
SHARD_PAGES = 200
CLAIM_STALE_MIN = 90


def prepare():
    for d in ("in", "out", "claims"):
        (RUN / d).mkdir(parents=True, exist_ok=True)
    pages = sorted({(int(r["doc_id"]), int(r["page_number"]))
                    for r in csv.DictReader(open(D.REPO / "data/analysis/table_level_metadata_published.csv"))})
    print(len(pages), "pages with a published table", flush=True)

    def one(p):
        text, k = page_input.page_text(*p)
        return p[0], p[1], k, text
    with ThreadPoolExecutor(32) as ex:
        got = list(ex.map(one, pages))
    keep = [g for g in got if g[2]]
    print(len(keep), "have a table in the Reducto page file;", len(got) - len(keep), "do not", flush=True)
    with open(RUN / "pages.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "page_number", "n_tables", "shard"])
        for i in range(0, len(keep), SHARD_PAGES):
            s = i // SHARD_PAGES
            with gzip.open(RUN / "in" / f"shard_{s:04d}.jsonl.gz", "wt", encoding="utf-8") as g:
                for doc, page, k, text in keep[i:i + SHARD_PAGES]:
                    g.write(json.dumps([doc, page, k, text]) + "\n")
                    w.writerow([doc, page, k, s])
    with open(RUN / "pages_without_table.csv", "w", newline="") as f:
        csv.writer(f).writerows([["doc_id", "page_number"]] + [[g[0], g[1]] for g in got if not g[2]])
    print("PREPARED", (len(keep) - 1) // SHARD_PAGES + 1, "shards", flush=True)


def shards():
    return sorted(p.name[:-9] for p in (RUN / "in").glob("shard_*.jsonl.gz"))


def claim(name):
    c = RUN / "claims" / name
    try:
        c.mkdir()
    except FileExistsError:
        try:
            if time.time() - c.stat().st_mtime < CLAIM_STALE_MIN * 60:
                return False
            shutil.rmtree(c, ignore_errors=True)
            c.mkdir()
        except (FileExistsError, FileNotFoundError):
            return False
    (c / f"{socket.gethostname()}_{os.getpid()}").touch()
    return True


def worker():
    import reader
    llm = reader.load()
    me = f"{socket.gethostname()}_{os.getpid()}"
    while True:
        left = [s for s in shards() if not (RUN / "out" / f"{s}.jsonl").exists()]
        if not left:
            print("ALL SHARDS DONE", flush=True)
            return
        mine = next((s for s in left if claim(s)), None)
        if mine is None:                       # others hold the rest: wait in case one of them dies
            time.sleep(300)
            continue
        pages = [tuple(json.loads(l)) for l in gzip.open(RUN / "in" / f"{mine}.jsonl.gz", "rt", encoding="utf-8")]
        t0 = time.time()
        rows = reader.read(llm, pages, D.PROMPT_VERSION)
        tmp = RUN / "out" / f".{mine}.{me}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, RUN / "out" / f"{mine}.jsonl")
        shutil.rmtree(RUN / "claims" / mine, ignore_errors=True)
        print(f"SHARDDONE {mine} {me} pages={len(pages)} seconds={time.time() - t0:.0f} "
              f"unparsed={sum(r['parsed'] is None for r in rows)} left={len(left) - 1}", flush=True)


def reread():
    """Second pass (Anna 2026-11-16): pages whose answer could not be read (mostly cut off at the answer limit) are
    read again with a longer limit. Writes reread/answers.jsonl; 07_build_metadata.py uses a reread answer if readable."""
    import reader
    bad = set()
    for f in sorted((RUN / "out").glob("shard_*.jsonl")):
        bad |= {(a["doc_id"], a["page_number"]) for a in map(json.loads, open(f, encoding="utf-8")) if a["parsed"] is None}
    pages = []
    for f in sorted((RUN / "in").glob("shard_*.jsonl.gz")):
        pages += [tuple(p) for p in map(json.loads, gzip.open(f, "rt", encoding="utf-8")) if (p[0], p[1]) in bad]
    print(len(pages), "pages to reread", flush=True)
    llm = reader.load(D.REREAD_MAX_CONTEXT, D.REREAD_AT_ONCE)
    t0 = time.time()
    rows = reader.read(llm, pages, D.PROMPT_VERSION, D.REREAD_MAX_CONTEXT, D.REREAD_MAX_ANSWER)
    (RUN / "reread").mkdir(exist_ok=True)
    with open(RUN / "reread" / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"REREAD pages={len(rows)} seconds={time.time() - t0:.0f} readable={sum(r['parsed'] is not None for r in rows)} "
          f"cut_off={sum(r['finish'] != 'stop' for r in rows)}", flush=True)


def status():
    s = shards()
    done = [x for x in s if (RUN / "out" / f"{x}.jsonl").exists()]
    held = sorted(p.name for p in (RUN / "claims").iterdir())
    print(f"{len(done)} of {len(s)} shards done; {len(held)} being read")


if __name__ == "__main__":
    {"prepare": prepare, "worker": worker, "status": status, "reread": reread}[sys.argv[1]]()
