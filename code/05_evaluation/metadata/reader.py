"""The open model reader: load the model once, read pages, parse the answers. Standard library only
(the vllm-rs environment has no pandas). Used by 05_model_reader.py (evaluation sets) and 06_full_run.py (all pages).
"""
import json
import os
import re
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HOME", "/home/hpc-scratch/annaboser/usgs_models/hf")

import defaults as D
import prompts


def escape_inner_quotes(s):
    """A seconds mark inside a copied coordinate (38°49'26", long ...) is an unescaped quote. Keep a quote only where
    JSON needs one: after { [ , : or before : } ] or before a comma that is followed by the next string or object."""
    out = []
    for i, ch in enumerate(s):
        if ch == '"' and (i == 0 or s[i - 1] != "\\"):
            before = s[:i].rstrip()[-1:]
            after = s[i + 1:].lstrip()
            opens = before in "{[,:"
            closes = after[:1] in ":}]" or (after[:1] == "," and after[1:].lstrip()[:1] in '"{[')
            if not (opens or closes):
                out.append('\\"')
                continue
        out.append(ch)
    return "".join(out)


def parse(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    for text in (m.group(0), escape_inner_quotes(m.group(0))):
        try:
            v = json.loads(text)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return None


def model_path():
    snap = D.MODEL_ROOT / "hub" / ("models--" + D.MODEL.replace("/", "--")) / "snapshots"
    return str(sorted(snap.iterdir())[-1])


def load():
    from vllm import LLM
    return LLM(model=model_path(), max_model_len=D.MODEL_MAX_CONTEXT, gpu_memory_utilization=0.90,
               max_num_seqs=D.MODEL_MAX_PAGES_AT_ONCE, max_num_batched_tokens=8192,
               limit_mm_per_prompt={"image": 0, "video": 0})


def read(llm, pages, version):
    """pages: list of (doc, page, n_tables, text) -> list of answer records, same order."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=D.MODEL_MAX_ANSWER)
    tok = llm.get_tokenizer()
    budget = D.MODEL_MAX_CONTEXT - D.MODEL_MAX_ANSWER - 1800   # room for the instructions
    msgs = []
    for doc, page, k, text in pages:
        ids = tok.encode(text)
        if len(ids) > budget:                                   # very long page: keep the start
            text = tok.decode(ids[:budget])
        msgs.append([{"role": "user", "content": prompts.PROMPTS[version].replace("{page}", text)}])
    try:
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(msgs, sp)
    return [dict(doc_id=doc, page_number=page, n_tables=k, finish=o.outputs[0].finish_reason,
                 raw=o.outputs[0].text, parsed=parse(o.outputs[0].text))
            for (doc, page, k, _), o in zip(pages, outs)]
