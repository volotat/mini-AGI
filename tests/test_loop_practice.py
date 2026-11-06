#!/usr/bin/env python3
"""
Loop practice (minagi/practice.py) and the loop detector the meter uses.

What this asserts:

  a practice span begins where a word does, ends with a space or a newline,
  keeps to its length bounds, holds a letter or digit and no marker, and is
  never one the real text goes on to repeat - repetition the text intends is
  not practised against

  the practice window is the span's real preceding text and then the span
  written out `reps` times

  the term is zero when each copy is already at most lam times as likely as
  the one before, positive when a copy is more likely than that, and pushes
  only on the repetitions - not on the real text before them

  on a paged model, FileReader.practice returns a finite term whose gradient
  reaches the model and leaves the reader's cursor where it was

  loop_tail finds a block repeated three times at a text's end, and not a run
  of one character, a block of punctuation, or ordinary text

Every repeated string here is built in code rather than written out, so this
file shows the self-knowledge lane no loop to learn from.
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.practice import (loop_tail, loop_window, pick_span,  # noqa: E402
                             practice_loss)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


PROSE = ("When the river rose in spring the farmers moved their herds to "
         "higher ground, and the village waited for the water to fall. "
         "Nobody knew how long it would take this time.\n"
         "The old bridge had held for forty years, though each flood left "
         "it a little more crooked than the last one had.\n"
         "Children counted the posts that still stood above the current. ")
ids = np.frombuffer((PROSE * 3).encode(), dtype=np.uint8).astype(np.int64)

print("spans")
rng = np.random.default_rng(0)
spans = [pick_span(ids, rng) for _ in range(200)]
spans = [s for s in spans if s is not None]
check("spans are found in ordinary text", len(spans) > 150, f"{len(spans)}")
good = True
for a, b in spans:
    span = ids[a:b]
    good &= ids[a - 1] in (32, 10)
    good &= span[-1] in (32, 10)
    good &= 8 <= b - a <= 200
    good &= any(chr(c).isalnum() for c in span)
    good &= not np.array_equal(ids[b:b + (b - a)], span)
check("each starts a word, ends with a space or newline, keeps its bounds "
      "and is not repeated by the text after it", good)

# a text that is one line said again and again: the line, or several of it,
# is what the text repeats and is never practised against; a part of it may
# be, since the text never says a part of it twice running
line = np.frombuffer(b"a line that the text itself says again\n",
                     dtype=np.uint8).astype(np.int64)
rng = np.random.default_rng(1)
got = [pick_span(np.tile(line, 12), rng) for _ in range(300)]
got = [g for g in got if g is not None]
check("never the line the text repeats, nor several of it",
      len(got) > 0 and all((b - a) % len(line) for a, b in got),
      f"{len(got)} spans")

marked = ids.copy()
marked[::7] = 300                       # a structural marker everywhere
got = [pick_span(marked, np.random.default_rng(k)) for k in range(50)]
check("no span holds a marker",
      all(g is None or (marked[g[0]:g[1]] < 256).all() for g in got))

print("window")
w = loop_window(ids, np.random.default_rng(3), reps=5, prefix=100)
window, p, L = w
span = window[p:p + L]
check("the window is real text, then the span five times",
      len(window) == p + 5 * L and all(
          np.array_equal(window[p + n * L:p + (n + 1) * L], span)
          for n in range(5)) and p <= 100)

print("term")
V, reps, Lx, px = 6, 4, 3, 5
window = np.array([0, 1, 2, 3, 4] + [1, 2, 3] * reps)


def logits_for(probs_by_rep, head=0.5):
    """Logits under which the k-th character of repetition n has probability
    probs_by_rep[n], and every character of the real head has `head`."""
    T = len(window) - 1
    z = torch.full((1, T, V), -30.0)
    for i in range(T):
        tgt = int(window[i + 1])
        n = (i + 1 - px) // Lx
        q = probs_by_rep[n] if i + 1 >= px else head
        z[0, i, tgt] = math.log(q / (1 - q)) if q < 1 else 30.0
        z[0, i, (tgt + 1) % V] = 0.0       # the rest of the mass
    return z


decaying = logits_for([0.8, 0.3, 0.12, 0.05])
loss, logp = practice_loss(decaying, window, px, Lx, reps, lam=0.5)
check("zero when every copy is at most half as likely as the one before",
      float(loss) == 0.0, f"{float(loss):.4f}")
z = logits_for([0.3, 0.6, 0.8, 0.9]).requires_grad_(True)
loss, logp = practice_loss(z, window, px, Lx, reps, lam=0.5)
loss.backward()
check("positive when copies grow more likely", float(loss) > 0,
      f"{float(loss):.4f}, log P by copy {logp}")
g = z.grad[0].abs().sum(-1)
check("it pushes on the repetitions, never on the real text before them",
      float(g[:px + Lx - 1].sum()) == 0.0 and float(g[px + Lx - 1:].sum()) > 0)

print("on a paged model")
tmp = tempfile.mkdtemp(prefix="minagi-practice-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.stream import FileReader                      # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=8, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=512, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", read_only=True)
    model.train()
    data = np.frombuffer((PROSE * 4).encode(), dtype=np.uint8).astype(np.uint16)
    r = FileReader(model, data, "prose", 128, 512, torch.device("cpu"))
    for _ in range(4):                       # a full window read
        loss = r.step(learn=True)
        loss.backward()
        model.zero_grad(set_to_none=True)
    pos = r.pos
    pl = None
    prng = np.random.default_rng(5)
    for _ in range(10):
        pl = r.practice(0.5, 4, prng)
        if pl is not None:
            break
    check("a practice term comes out of the window just read", pl is not None)
    if pl is not None:
        check("it is finite", bool(torch.isfinite(pl)), f"{float(pl):.4f}")
        pl.backward()
        reached = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0))
                      for p in model.parameters())
        check("its gradient reaches the model", reached > 0, f"{reached} tensors")
        check("the reader's cursor did not move", r.pos == pos)

    # chat learning practises the same way, after its own steps
    from minagi.live import LiveLearner                       # noqa: E402
    from minagi.tokenizer import ByteTokenizer                # noqa: E402
    wdir2 = os.path.join(tmp, "w2")
    create(wdir2, seed=1, verbose=False,
           experts=8, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=512, max_steps=3, top_k=2, depth=1)
    m2, _, _, man2 = T.build_paged(wdir2, "cpu", ram_capacity=8)
    live = LiveLearner(m2, lr=1e-3, save_every=0, chunk=128, context=512,
                       manifest=man2, practice_every=1)
    recs = live.feed(PROSE * 2, ByteTokenizer())
    took = [r["practice"] for r in recs if "practice" in r]
    check("the chat learner takes practice steps after its own",
          len(took) > 0 and all(math.isfinite(v) for v in took),
          f"{len(took)} of {len(recs)} steps")
    check("...without counting them as steps of the stream",
          live.steps == len(recs))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("loop detector")
block = "the cart rolled on "
found = loop_tail("Once the gate was open " + block * 3)
check("a phrase said three times at the end is a loop",
      found is not None and found[0] == len(block)
      and found[1] >= 3 * len(block), f"{found}")
check("ordinary text is not", loop_tail(PROSE) is None)
check("a run of one character is not", loop_tail("text" + " " * 60) is None)
check("a block of punctuation is not", loop_tail("text" + "-=" * 30) is None)
check("two copies are not three", loop_tail("start " + block * 2) is None)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
