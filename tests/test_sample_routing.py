#!/usr/bin/env python3
"""
A sample is its own text.

Each reading of each prompt starts at position 0, so it begins a new text
with an empty vote: the prompt's forward admits the experts its characters
ask for most, and then every character of the reply is a forward of its own
that adds its requests and routes among what the prompt and the reply so far
have voted for. Nothing from another text counts.

What this asserts, on a paged model:

  every reading of every prompt starts a new text

  a prompt's experts do not depend on the prompt read before it

  a reply never uses more experts than the card holds

  sampling leaves nothing behind that changes the next training step: a
  training window admits the same experts before a round of samples and
  after it
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


tmp = tempfile.mkdtemp(prefix="minagi-sample-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.tokenizer import ByteTokenizer                # noqa: E402
    import train                                              # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=256, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = train.build_paged(wdir, "cpu", read_only=True)
    tok = ByteTokenizer()
    dev = torch.device("cpu")
    RAW = [("raw", {})]

    def admitted():
        return tuple(sorted(pool._admitted))

    WINDOW = torch.tensor([list(b"def merge(a, b):\n    return sorted(a + b)\n"
                                b"class Node:\n    pass\n" * 3)])

    def train_window():
        """Admit the way a training step does: a fresh window at position 0."""
        model.train()
        with torch.no_grad():
            model(WINDOW[:, :-1], WINDOW[:, 1:])
        return admitted()

    before = train_window()

    keep = train.SAMPLE_PROMPTS
    texts_seen, admitted_after = [], {}
    try:
        for first in ("1. e4 e5 2. Nf3 Nc6 ", "The Roman Empire was "):
            train.SAMPLE_PROMPTS = [("x", first), ("chess", "1. d4 Nf6 2. c4 ")]
            s0 = pool.segments
            out = train.sample_now(model, tok, dev, n_new=60, variants=RAW)
            texts_seen.append(pool.segments - s0)
            admitted_after[first] = admitted()
    finally:
        train.SAMPLE_PROMPTS = keep

    check("the sampler produced text", out and len(out[0][2][0][1]) > 0)
    check("every reading of every prompt starts a new text",
          all(n == 2 for n in texts_seen), f"texts begun per round: {texts_seen}")
    a, b = admitted_after.values()
    check("a prompt's experts do not depend on the prompt before it",
          a == b, f"{a} vs {b}")
    check("a reply never uses more experts than the card holds",
          len(a) <= pool.resident, f"{len(a)} of {pool.resident}")
    check("sampling leaves nothing behind for the next training step",
          train_window() == before, f"{before}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all sample-routing tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
