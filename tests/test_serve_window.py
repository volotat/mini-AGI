#!/usr/bin/env python3
"""
A chat reply that runs past the attention window keeps its characters in order.

Keys are cached with their rotary position already applied, so a cached key
says forever where it was read. serve.py used to make room in a full window
by trimming the oldest key and placing each new character at the cache's
length - block-1, every time. The keys it went on to cache all said block-1,
and attention could no longer tell the newest characters' order apart. The
window fills before a reply starts whenever the conversation is long in BYTES:
the prompt budget counts characters, and a Cyrillic letter is two bytes.

  every key in the cache sits at the position attention assumes for it -
  0, 1, 2, ... in order - at every forward of a reply, through several
  window crossings, in English and in Cyrillic

  the reply is, character for character, what model.generate writes for the
  same prompt, decoding and random numbers: when the window fills, both
  re-read the newest half of the text from position 0
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-serve-window-")
try:
    from minagi.create import create                           # noqa: E402
    from minagi.pool import PooledMLP                          # noqa: E402
    from minagi.recur import RecurCoder                        # noqa: E402
    from minagi.tokenizer import ByteTokenizer                 # noqa: E402
    import minagi.decode as D                                  # noqa: E402
    import serve                                               # noqa: E402
    import train as T                                          # noqa: E402

    BLOCK = 64
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=BLOCK, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=3)
    torch.manual_seed(0)
    for site in (m for m in model.modules() if isinstance(m, PooledMLP)):
        site.router.weight.data.normal_(0.0, 1.0)      # a router with opinions
    model.eval()
    tok = ByteTokenizer()
    serve.STATE.update(model=model, tok=tok)

    # THE POSITION EACH CACHED KEY WAS READ AT, followed through every forward:
    # a trim from the front drops the oldest, a forward appends its own.
    track = {"caches": None, "stamps": [], "bad": None}
    forward = RecurCoder.forward

    def watched(self, idx, targets=None, caches=None, pos_offset=0, collect=False):
        if caches is not None:
            k = caches[0].get("k")
            held = 0 if k is None else k.shape[-2]
            if caches is not track["caches"]:
                track["caches"], track["stamps"] = caches, []
            st = track["stamps"][len(track["stamps"]) - held:] if held else []
            st += range(pos_offset, pos_offset + idx.shape[1])
            track["stamps"] = st
            if st != list(range(len(st))) and track["bad"] is None:
                track["bad"] = st[-4:]
        return forward(self, idx, targets=targets, caches=caches,
                       pos_offset=pos_offset, collect=collect)

    RecurCoder.forward = watched

    # what stream picks, id by id: config.yaml's decoding, its random numbers
    # fixed, so that model.generate given the same sampler writes the same
    picked = []
    SEED = 1234

    def recording(sampler):
        def pick(logits, history):
            nxt = sampler(logits, history)
            picked.append(nxt)
            return nxt
        return pick

    def by_serve(prompt, n):
        track.update(caches=None, stamps=[], bad=None)
        picked.clear()
        for _ in serve.stream(prompt, n,
                              sampler=recording(D.sampler_from(seed=SEED))):
            pass
        return list(picked), track["bad"]

    def by_generate(prompt, n):
        ids = torch.tensor([tok.encode(prompt).ids[-BLOCK:]])
        with torch.no_grad():
            out = model.generate(ids, n, sampler=D.sampler_from(seed=SEED))
        return out[0, ids.shape[1]:].tolist()

    for label, prompt in (
            ("English", "<user>\nWhat is the capital of France?\n</user>\n<bot>\n"),
            ("Cyrillic, past the window in bytes before the reply",
             "<user>\n" + "Какая столица у Франции? Расскажи подробно. " * 2
             + "\n</user>\n<bot>\n")):
        n_bytes = len(tok.encode(prompt).ids)
        a, bad = by_serve(prompt, 3 * BLOCK)
        crossings = (min(n_bytes, BLOCK) + len(a)) // (BLOCK // 2) - 1
        check(f"{label}: every cached key at the position attention assumes",
              bad is None,
              f"prompt {n_bytes} bytes, reply {len(a)}, window {BLOCK}"
              + (f"; newest keys read at {bad}" if bad else ""))
        b = by_generate(prompt, len(a))
        same = a == b[:len(a)]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        check(f"{label}: the reply is what model.generate writes",
              same and len(a) > BLOCK,
              f"{len(a)} characters, ~{crossings} window crossings"
              + ("" if same else f"; first difference at character {first}"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all serve-window tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
