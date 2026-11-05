#!/usr/bin/env python3
"""
Muon for the weight matrices and AdamW for the rest (minagi/muon.py,
training.optimizer), as Liu et al. (2025) apply it.

What these tests hold it to:
  orthogonalising takes a matrix to one with every singular value near one,
  pointing the way the matrix points - and a stack of them each on its own
  the AdamW it runs for everything else is torch's AdamW, step for step
  a Muon step has AdamW's typical size - an RMS of 0.2 times the rate,
  whatever the matrix's shape - and AdamW's decoupled weight decay
  attention's query, key and value are stepped as the three matrices they are
  an expert on the card that no character chose is not stepped and keeps its
  momentum, while its weight decay applies; the experts beside it are stepped
  Muon keeps its momentum where AdamW keeps its first moment, and keeps no
  second (tests/test_muon_one_moment.py)
  on a real paged model every weight matrix is Muon's - the trunk's, the
  experts', the router's - and the embedding, the head, the norms, the
  halting unit, the gate and the depth embedding are AdamW's, the pool's
  AdamW group last; a router that growth replaces stays Muon's
  `adamw` builds the optimiser the run has always had
  train.py reads with it end to end - it steps, saves, loads again, and the
  experts' files carry its momentum and no second moment - and chat's
  learner builds the same optimiser from the config
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minagi.muon import (MuonAdamW, adopt, build, orthogonalise,  # noqa: E402
                         takes_muon)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


torch.manual_seed(0)

# --- orthogonalising ---------------------------------------------------------------
G = torch.randn(64, 256)
O = orthogonalise(G).float()
sv = torch.linalg.svdvals(O)
U, _, Vh = torch.linalg.svd(G, full_matrices=False)
cos = float((O * (U @ Vh)).sum() / (O.norm() * (U @ Vh).norm()))
T_ = orthogonalise(torch.randn(256, 64)).float()
stack = torch.randn(5, 64, 256)
each = torch.stack([orthogonalise(m) for m in stack]).float()
check("orthogonalising gives singular values near one, pointing the way the "
      "matrix points - wide or tall",
      float(sv.min()) > 0.5 and float(sv.max()) < 1.3 and cos > 0.95
      and float(torch.linalg.svdvals(T_).min()) > 0.5,
      f"singular values {float(sv.min()):.2f}-{float(sv.max()):.2f}, "
      f"cosine with UV' {cos:.3f}")
check("...and a stack of matrices each on its own",
      torch.allclose(orthogonalise(stack).float(), each, atol=2e-2))

# --- the AdamW it runs is torch's ------------------------------------------------
def pair():
    a = [torch.nn.Parameter(torch.randn(16, 8)), torch.nn.Parameter(torch.randn(8))]
    b = [torch.nn.Parameter(x.detach().clone()) for x in a]
    return a, b


a, b = pair()
ta = torch.optim.AdamW(a, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
mb = MuonAdamW([{"params": b, "muon": False}], lr=1e-2, betas=(0.9, 0.95),
               weight_decay=0.1)
for _ in range(6):
    gs = [torch.randn_like(x) for x in a]
    for x, y, g in zip(a, b, gs):
        x.grad, y.grad = g.clone(), g.clone()
    ta.step()
    mb.step()
check("the AdamW it runs for the rest is torch's AdamW, step for step",
      all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b)))

# --- the size of a Muon step ---------------------------------------------------------
rms = {}
for shape in ((512, 2048), (64, 512), (2048, 512)):
    p = torch.nn.Parameter(torch.zeros(shape))
    p.grad = torch.randn(shape)
    MuonAdamW([{"params": [p], "muon": True}], lr=1e-3, weight_decay=0.0).step()
    rms[shape] = float(p.detach().pow(2).mean().sqrt()) / 1e-3
check("a Muon step has AdamW's typical size: an RMS of 0.2 x the rate, "
      "whatever the shape",
      all(0.14 < v < 0.24 for v in rms.values()),
      " ".join(f"{s[0]}x{s[1]}: {v:.3f}" for s, v in rms.items()))
p = torch.nn.Parameter(torch.ones(32, 64))
p.grad = torch.zeros(32, 64)
MuonAdamW([{"params": [p], "muon": True}], lr=1e-2, weight_decay=0.1).step()
check("...with AdamW's decoupled weight decay",
      torch.allclose(p.detach(), torch.full((32, 64), 1 - 1e-3)))

# --- query, key and value as three matrices ---------------------------------------
p = torch.nn.Parameter(torch.zeros(96, 32))
p.grad = torch.randn(96, 32) * torch.tensor([1.0, 30.0, 0.03]).repeat_interleave(32).view(96, 1)
MuonAdamW([{"params": [p], "muon": True, "split": 3}], lr=1.0,
          weight_decay=0.0).step()
blocks = [torch.linalg.svdvals(p.detach()[i * 32:(i + 1) * 32]) / (0.2 * 32 ** 0.5)
          for i in range(3)]
check("attention's query, key and value are stepped as three matrices, each "
      "its own full step however different their gradients' sizes",
      all(float(s.min()) > 0.5 and float(s.max()) < 1.3 for s in blocks),
      " ".join(f"{float(s.min()):.2f}-{float(s.max()):.2f}" for s in blocks))

# --- an expert on the card with no gradient ----------------------------------------
p = torch.nn.Parameter(torch.randn(4, 16, 8))
before = p.detach().clone()
p.grad = torch.randn(4, 16, 8)
p.grad[2] = 0.0
o = MuonAdamW([{"params": [p], "muon": True}], lr=1e-2, weight_decay=0.1)
st = o.state[p]
st["exp_avg"] = torch.randn(4, 16, 8)
st["exp_avg_sq"] = torch.full((4, 16, 8), 7.0)
m_before = st["exp_avg"].clone()
o.step()
moved = (p.detach() - before * (1 - 1e-3)).flatten(1).abs().amax(1)
check("an expert on the card that no character chose is not stepped and "
      "keeps its momentum; its weight decay applies; the others are stepped",
      float(moved[2]) < 1e-7 and torch.equal(st["exp_avg"][2], m_before[2])
      and all(float(moved[i]) > 1e-4 for i in (0, 1, 3))
      and not torch.equal(st["exp_avg"][0], m_before[0]),
      f"{[round(float(x), 6) for x in moved]}")
check("Muon keeps its momentum in AdamW's first moment and lets the second go",
      "exp_avg_sq" not in st)

# --- on a real paged model ------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix="minagi-muon-")
try:
    import train as T
    from minagi.create import create
    from minagi.pool import split_trunk_pool
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    model, cfg, pool, man = T.build_paged(wdir, "cpu")
    trunk, pool_ps = split_trunk_pool(model)
    opt = build(model, trunk, pool_ps, 1e-3, 1e-4, 0.1, kind="muon")
    names = {id(p): n for n, p in model.named_parameters()}
    on = {names[id(p)] for g in opt.param_groups if g["muon"]
          for p in g["params"]}
    off = {names[id(p)] for g in opt.param_groups if not g["muon"]
           for p in g["params"]}
    mats = {n for n, p in model.named_parameters()
            if (p.dim() == 2 and min(p.shape) > 1 or p.dim() == 3)
            and not any(s in n for s in ("tok_emb", "head", "halt"))}
    split = {names[id(p)] for g in opt.param_groups if g.get("split") == 3
             for p in g["params"]}
    check("on a paged model every weight matrix is Muon's - the trunk's, the "
          "experts', the router's",
          on == mats and {"pool.w1", "pool.w3", "pool.w2"} <= on
          and any("router" in n for n in on)
          and split == {n for n in on if n.endswith("qkv.weight")}
          and len(split) >= 1, f"{sorted(on)}")
    check("...and the embedding, the head, the norms, the halting unit, the "
          "gate and the depth embedding AdamW's, the pool's AdamW group last",
          off == {n for n, _ in model.named_parameters()} - on
          and any("tok_emb" in n for n in off) and "pool.gate" in off
          and any("depth_emb" in n for n in off)
          and any("halt" in n for n in off)
          and opt.param_groups[-1]["name"] == "pool"
          and not opt.param_groups[-1]["muon"]
          and all(g["base_lr"] == (1e-4 if g["name"] == "trunk" else 1e-3)
                  for g in opt.param_groups))
    # growth writes expert files at once: on a copy
    gdir = os.path.join(tmp, "g")
    shutil.copytree(wdir, gdir)
    gm, _, gp, _ = T.build_paged(gdir, "cpu")
    gt, gps = split_trunk_pool(gm)
    gopt = build(gm, gt, gps, 1e-3, 1e-4, 0.1, kind="muon")
    gp.attach_optimiser(gopt)
    gp.add_experts(1, step=1, birth_gate=0.001)
    T._resync_opt(gopt, gm, types.SimpleNamespace(lr=1e-3, wd=0.1))
    gp.attach_optimiser(gopt)
    names = {id(p): n for n, p in gm.named_parameters()}
    rows = [g["muon"] for g in gopt.param_groups for p in g["params"]
            if "router" in names.get(id(p), "")]
    gate = [g["muon"] for g in gopt.param_groups for p in g["params"]
            if names.get(id(p)) == "pool.gate"]
    check("a router that growth replaces stays Muon's, a gate AdamW's",
          rows == [True] * len(rows) and len(rows) >= 1 and gate == [False])
    check("`adamw` builds the optimiser the run has always had",
          type(build(model, trunk, pool_ps, 1e-3, 1e-4, 0.1)) is
          torch.optim.AdamW)

    # --- end to end ------------------------------------------------------------------
    corpus = os.path.join(tmp, "train")
    for subj, text in (("prose", "the cat sat on the mat and looked out. " * 60),
                       ("code", "def f(x):\n    return x * 2 + 1\n" * 60)):
        os.makedirs(os.path.join(corpus, subj))
        with open(os.path.join(corpus, subj, "a.txt"), "w") as f:
            f.write(text)
    log = os.path.join(tmp, "loss.jsonl")
    r = subprocess.run(
        [sys.executable, "train.py", "--device", "cpu", "read", corpus,
         "--save", "--weights-dir", wdir, "--held-out", "",
         "--sample-every", "0", "--save-every", "0", "--no-plots",
         "--history", os.path.join(tmp, "h.jsonl"),
         "--sample-log", os.path.join(tmp, "s.txt"), "--loss-log", log,
         "--max-chars", "3000", "--chunk", "64", "--optimizer", "muon"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    rows = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    m2, _, pool2, man2 = T.build_paged(wdir, "cpu", read_only=True)
    moms, seconds = [], 0
    for f in sorted(os.listdir(os.path.join(wdir, "experts"))):
        z = np.load(os.path.join(wdir, "experts", f))
        if "w1_m" in z.files:
            moms.append(int(np.abs(z["w1_m"]).max()))
        seconds += sum(k.endswith("_v") for k in z.files)
    check("train.py reads with it end to end: steps, saves, and loads again",
          r.returncode == 0 and "Muon for the weight matrices" in r.stdout
          and len(rows) >= 20 and all(x["model"] is not None for x in rows)
          and int(man2["step"]) >= len(rows) - 1,
          f"exit {r.returncode}, {len(rows)} steps, muon line "
          f"{'Muon for the weight matrices' in r.stdout}, model "
          f"{sum(x['model'] is None for x in rows)} none, step {man2['step']}"
          + ("" if r.returncode == 0 else "\n" + r.stdout[-1500:]
             + r.stderr[-1500:]))
    check("...and the experts' files carry its momentum, and no second moment",
          sum(m > 0 for m in moms) >= 2 and seconds == 0,
          f"{moms}, {seconds} second moments")

    from minagi.live import LiveLearner
    conf = {"training": {"lr": 1e-3, "trunk_lr_mult": 0.1, "chunk": 16,
                         "optimizer": "muon"}}
    m3, _, _, man3 = T.build_paged(wdir, "cpu", read_only=True)
    ln = LiveLearner.for_run(m3, manifest=dict(man3, checks={
        "every": 10, "held_out": os.path.join(tmp, "nowhere")}),
        save_every=0, config=conf)
    check("chat's learner builds the same optimiser from the config",
          isinstance(ln.opt, MuonAdamW))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all muon tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
