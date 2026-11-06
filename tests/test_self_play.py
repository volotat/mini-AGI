#!/usr/bin/env python3
"""
Self-play (minagi/practice.py): the model writes on from a place in the text
it has just read, and one step raises the real continuation and lowers its
own.

What this asserts:

  the term pushes the real continuation up and the model's own down, never
  the shared prefix, and where the two continuations agree character for
  character their pushes cancel

  on a paged model, a self-play term comes out of a reader's window: the
  model writes greedily, reads both continuations as one training forward
  of two rows, the gradient reaches it and one optimiser step can be taken;
  the reader's cursor does not move and the model is left training

  drawn rather than greedy, the same random generator gives the same
  continuation
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.practice import self_play_pair_loss                # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


print("the term")
V, P, C = 8, 4, 3
real = [1, 2, 3, 4, 5, 6, 7]                 # prefix 1 2 3 4, real 5 6 7
own = [1, 2, 3, 4, 5, 0, 0]                  # its own: agrees on 5, then 0 0
rows = torch.tensor([real, own])
y = rows[:, 1:]
z = torch.zeros(2, rows.shape[1] - 1, V)
z[1, :, 0] = 2.0                             # its own text is the likelier
z.requires_grad_(True)
loss, lp_real, lp_own = self_play_pair_loss(z, y, P, C)
loss.backward()
g = z.grad
check("positive while its own continuation is the likelier",
      float(loss) > 0 and float(lp_own) > float(lp_real),
      f"{float(loss):.3f}")
check("the real continuation goes up",
      all(float(g[0, P - 1 + k, int(y[0, P - 1 + k])]) < 0 for k in range(C)))
check("its own goes down where it differs",
      all(float(g[1, P - 1 + k, int(y[1, P - 1 + k])]) > 0 for k in (1, 2)))
check("the shared prefix is not pushed",
      float(g[:, :P - 1].abs().sum()) == 0.0)
z2 = torch.zeros(2, rows.shape[1] - 1, V, requires_grad=True)
same = torch.tensor([real, real])
l2, _, _ = self_play_pair_loss(z2, same[:, 1:], P, C)
l2.backward()
check("identical continuations cancel: the push on one row is the exact "
      "negative of the push on the other, so through the same weights "
      "nothing moves", float((z2.grad[0] + z2.grad[1]).abs().sum()) == 0.0
      and float(z2.grad.abs().sum()) > 0)

print("on a paged model")
PROSE = ("When the river rose in spring the farmers moved their herds to "
         "higher ground, and the village waited for the water to fall. "
         "Nobody knew how long it would take this time.\n"
         "The old bridge had held for forty years, though each flood left "
         "it a little more crooked than the last one had.\n")
tmp = tempfile.mkdtemp(prefix="minagi-selfplay-")
try:
    from minagi.create import create                          # noqa: E402
    from minagi.stream import FileReader                      # noqa: E402
    from minagi.practice import self_play_term                # noqa: E402
    import train as T                                         # noqa: E402

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=8, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=1024, max_steps=3, top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu", ram_capacity=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pool.attach_optimiser(opt)
    model.train()
    data = np.frombuffer((PROSE * 6).encode(), dtype=np.uint8).astype(np.uint16)
    r = FileReader(model, data, "prose", 256, 1024, torch.device("cpu"))
    for _ in range(3):
        loss = r.step(learn=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    pos = r.pos
    sp = r.self_play(32, np.random.default_rng(1))
    check("a self-play term comes out of the window just read",
          sp is not None and bool(torch.isfinite(sp)),
          f"{float(sp):.4f}" if sp is not None else "")
    if sp is not None:
        sp.backward()
        reached = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0))
                      for p in model.parameters())
        check("its gradient reaches the model", reached > 0,
              f"{reached} tensors")
        opt.step()
        opt.zero_grad(set_to_none=True)
        check("an optimiser step on it goes through", True)
        check("it wrote what it was asked to",
              len(r.played["own"]) == 32
              and 0 <= r.played["agree"] <= 32,
              f"agreeing at {r.played['agree']} of 32")
    check("the reader's cursor did not move", r.pos == pos)
    check("the model is left training", model.training)

    ids = data[:1000].astype(np.int64)
    a = self_play_term(model, ids, 1024, torch.device("cpu"), 24,
                       np.random.default_rng(7), sample=True)
    b = self_play_term(model, ids, 1024, torch.device("cpu"), 24,
                       np.random.default_rng(7), sample=True)
    check("drawn with the same generator, the same continuation",
          a is not None and b is not None
          and np.array_equal(a[1]["own"], b[1]["own"]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
