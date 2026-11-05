#!/usr/bin/env python3
"""
Muon keeps one moment, and nothing carries a second for what it steps.

  a matrix Muon steps holds its momentum and no second moment after a step -
  none is made, and one restored from an AdamW checkpoint is let go - while
  the parameters AdamW steps keep both; the optimiser says which keeps one

  on a paged model Muon steps, the experts' files, their entries in RAM, the
  card and optim.npz hold the momentum and no second moment, and what the
  pool costs on disk - what growth.max_disk_gb is held against - is six
  bytes a parameter; under AdamW the files carry both and it is eight

  a file AdamW wrote, read where Muon steps the experts: the entries already
  in RAM let the second moment go when the optimiser arrives, the file is
  read without it, and it is written back without it

  moving back to AdamW: a tensor with only a momentum starts its second
  moment from the momentum squared - from optim.npz for what is not an
  expert, on the card for the experts - so its first step stays near the
  rate, where a zero second moment would step it many times that
"""
import os
import shutil
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minagi import muon as M                                   # noqa: E402
from minagi import store                                       # noqa: E402
from minagi.precision import pack_bf16, unpack_bf16            # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


keeps_second = getattr(M, "keeps_second", None)

# ---- the optimiser ------------------------------------------------------------
torch.manual_seed(0)
a = torch.nn.Parameter(torch.randn(16, 8))                     # Muon's
b = torch.nn.Parameter(torch.randn(8))                         # AdamW's
o = M.MuonAdamW([{"params": [a], "muon": True},
                 {"params": [b], "muon": False}], lr=1e-2)
o.state[a]["exp_avg"] = torch.randn(16, 8)
o.state[a]["exp_avg_sq"] = torch.rand(16, 8)    # as an AdamW checkpoint left it
for _ in range(2):
    a.grad, b.grad = torch.randn(16, 8), torch.randn(8)
    o.step()
check("a matrix Muon steps holds its momentum and no second moment, even "
      "one restored from AdamW",
      "exp_avg" in o.state[a] and "exp_avg_sq" not in o.state[a],
      str(sorted(o.state[a])))
check("...while the parameters AdamW steps keep both",
      {"exp_avg", "exp_avg_sq"} <= set(o.state[b]))
check("the optimiser says which keeps one",
      keeps_second is not None and not keeps_second(o, a)
      and keeps_second(o, b)
      and keeps_second(torch.optim.AdamW([b], lr=1e-3), b))

# ---- moving back to AdamW: the first step -------------------------------------
LR, T = 1e-3, 50_000.0
torch.manual_seed(1)
m0 = torch.randn(1000) * 0.05                   # the momentum Muon left
g0 = torch.randn(1000) * 1e-4                   # a small gradient beside it
p = torch.nn.Parameter(torch.zeros(1000))
o = M.MuonAdamW([{"params": [p], "muon": False}], lr=LR, weight_decay=0.0)
o.state[p].update(exp_avg=m0.clone(), step=torch.tensor(T))
p.grad = g0.clone()
try:
    o.step()
    worst = float(p.detach().abs().max()) / LR
except KeyError as e:
    worst = f"no second moment: {e}"
check("moving back to AdamW, a tensor with only a momentum starts its second "
      "moment from it, and its first step stays near the rate",
      isinstance(worst, float) and worst < 1.5,
      f"largest step {worst:.2f}x the rate" if isinstance(worst, float) else worst)
q = torch.nn.Parameter(torch.zeros(1000))
z = torch.optim.AdamW([q], lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
z.state[q] = {"step": torch.tensor(T), "exp_avg": m0.clone(),
              "exp_avg_sq": torch.zeros(1000)}
q.grad = g0.clone()
z.step()
blind = float(q.detach().abs().max()) / LR
check("...where a zero second moment would step it many times that",
      blind > 10, f"{blind:.0f}x the rate")

# ---- moving back to AdamW: what optim.npz holds -------------------------------
tmp = tempfile.mkdtemp(prefix="minagi-one-moment-")
try:
    lin = torch.nn.Sequential(torch.nn.Linear(8, 8))
    mom = torch.randn(8, 8) * 0.05
    od = os.path.join(tmp, "optim")
    os.makedirs(os.path.join(od, "experts"))
    np.savez(os.path.join(od, "optim.npz"),
             **{"0.weight|m": pack_bf16(mom), "0.weight|t": np.asarray(T)})
    adam = torch.optim.AdamW(lin.parameters(), lr=LR)
    try:
        store._load_optim(adam, lin, od)
    except KeyError as e:
        print(f"  (restore raised KeyError {e})")
    st = adam.state[lin[0].weight]
    want = unpack_bf16(pack_bf16(mom)) ** 2
    check("optim.npz written while Muon stepped a tensor gives AdamW a second "
          "moment started from the momentum",
          "exp_avg_sq" in st and torch.allclose(st["exp_avg_sq"], want),
          "" if "exp_avg_sq" in st else "none")
    np.savez(os.path.join(od, "optim.npz"),
             **{"0.weight|m": pack_bf16(mom),
                "0.weight|v": pack_bf16(torch.rand(8, 8))})
    mu = M.MuonAdamW([{"params": [lin[0].weight], "muon": True},
                      {"params": [lin[0].bias], "muon": False}], lr=LR)
    store._load_optim(mu, lin, od)
    check("...and Muon, given an AdamW second moment, does not take it",
          "exp_avg" in mu.state[lin[0].weight]
          and "exp_avg_sq" not in mu.state[lin[0].weight])

    # ---- on a paged model ------------------------------------------------------
    import train as TR
    from minagi.create import create
    from minagi.pool import PooledMLP, split_trunk_pool

    TEXTS = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1",
             "def merge(a, b):\n    return sorted(a + b)\n\nclass Node:\n",
             "The Roman Empire was the post-Republican period of ancient Rome",
             "add 4917 + 388 = 5305\nadd 72459 + 1 = 72460\nsub 33 - 3 = 30",
             "Once upon a time, there was a little boy named Tom. One day he"]
    D, F = 32, 32

    def ids(t, block):
        return torch.tensor([[c for c in t.encode()][:block]])

    def build(name, fresh=True):
        d = os.path.join(tmp, name)
        if fresh:
            create(d, seed=0, verbose=False, experts=12, resident=4, d_ff=F,
                   d_model=D, trunk_d_ff=32, n_head=2, block=128,
                   max_steps=3, top_k=2, depth=1)
        model, cfg, pool, _ = TR.build_paged(d, "cpu", ram_capacity=2)
        pool.balance = 0.0
        torch.manual_seed(0)
        for site in (s for s in model.modules() if isinstance(s, PooledMLP)):
            site.router.weight.data.normal_(0.0, 1.0)
        return d, model, cfg, pool

    def optimiser(model, kind):
        trunk, ps = split_trunk_pool(model)
        return M.build(model, trunk, ps, 1e-3, 1e-4, 0.1, kind=kind)

    def train(model, cfg, opt, steps=10):
        for i in range(steps):
            x = ids(TEXTS[i % len(TEXTS)], cfg.block)
            model.train()
            _, loss = model(x[:, :-1], x[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    def files(d):
        out = {}
        for f in sorted(os.listdir(os.path.join(d, "experts"))):
            with np.load(os.path.join(d, "experts", f)) as zz:
                out[f] = set(zz.files)
        return out

    def disk_ok(pool, per):
        return pool.disk_bytes(0) == pool.n_experts() * 3 * D * F * per

    md, model, cfg, pool = build("muon")
    opt = optimiser(model, "muon")
    pool.attach_optimiser(opt)
    train(model, cfg, opt)
    store.save(model, md, step=1, opt=opt, cfg=asdict(cfg))
    fm = files(md)
    with np.load(os.path.join(md, "optim.npz")) as zz:
        ov = [k for k in zz.files if k.endswith("|v")]
    names = {id(p): n for n, p in model.named_parameters()}
    muon_names = {names[id(p)] for g in opt.param_groups if g["muon"]
                  for p in g["params"]}
    check("under Muon the experts' files hold the momentum and no second "
          "moment",
          sum("w1_m" in k for k in fm.values()) >= 2
          and not any(n.endswith("_v") for k in fm.values() for n in k),
          f"{sum('w1_m' in k for k in fm.values())} with a momentum, "
          f"{sum('w1_v' in k for k in fm.values())} with a second moment")
    check("...nor do their entries in RAM, the card or optim.npz",
          not any(k.endswith("_v") for e in pool.tiers.ram.values() for k in e)
          and all("exp_avg_sq" not in opt.state[p]
                  for p in (pool.w1, pool.w3, pool.w2))
          and not [k for k in ov if k[:-2] in muon_names],
          f"optim.npz second moments for Muon's tensors: "
          f"{[k for k in ov if k[:-2] in muon_names][:4]}")
    check("...and the pool costs six bytes a parameter on disk",
          disk_ok(pool, 6), f"{pool.disk_bytes(0):,} bytes")

    ad, amodel, acfg, apool = build("adamw")
    aopt = optimiser(amodel, "adamw")
    apool.attach_optimiser(aopt)
    train(amodel, acfg, aopt)
    store.save(amodel, ad, step=1, opt=aopt, cfg=asdict(acfg))
    fa = files(ad)
    check("under AdamW the files carry both moments, at eight bytes a "
          "parameter",
          sum("w1_v" in k for k in fa.values()) >= 2
          and all(("w1_m" in k) == ("w1_v" in k) for k in fa.values())
          and disk_ok(apool, 8))

    # ---- a file AdamW wrote, read where Muon steps -----------------------------
    od2 = os.path.join(tmp, "adamw_then_muon")
    shutil.copytree(ad, od2)
    _, m3, c3, p3 = build("adamw_then_muon", fresh=False)
    m3.eval()
    with torch.no_grad():                    # the start's check, before any optimiser
        m3(ids(TEXTS[2], c3.block)[:, :-1])
    had = sum(k.endswith("_v") for e in p3.tiers.ram.values() for k in e)
    written = set()
    to_disk = p3.tiers._to_disk

    def spy(i, ent):
        written.add(i)
        return to_disk(i, ent)
    p3.tiers._to_disk = spy
    o3 = optimiser(m3, "muon")
    p3.attach_optimiser(o3)
    gone = not any(k.endswith("_v") for e in p3.tiers.ram.values() for k in e)
    old = next(f for f in sorted(fa) if "w1_v" in fa[f])
    read = p3.tiers._from_disk(int(old[1:6]))
    check("a file AdamW wrote, where Muon steps: entries in RAM let the second "
          "moment go when the optimiser arrives, and the file is read without it",
          had > 0 and gone and "w1_m" in read
          and not any(k.endswith("_v") for k in read),
          f"{had} second moments in RAM before")
    train(m3, c3, o3)
    store.save(m3, od2, step=2, opt=o3, cfg=asdict(c3))
    f3 = files(od2)
    back = [i for i in written if "w1_v" in fa.get("e%05d.npz" % i, ())]
    check("...and written back without it",
          back and all("w1_v" not in f3["e%05d.npz" % i] for i in written),
          f"{len(written)} written back, {len(back)} of them had one")

    # ---- moving back to AdamW on a paged model ---------------------------------
    bd = os.path.join(tmp, "muon_then_adamw")
    shutil.copytree(md, bd)
    _, m4, c4, p4 = build("muon_then_adamw", fresh=False)
    m4.eval()
    with torch.no_grad():
        m4(ids(TEXTS[1], c4.block)[:, :-1])
    o4 = optimiser(m4, "adamw")
    p4.attach_optimiser(o4)
    store._load_optim(o4, m4, bd)
    p4._own_moments()
    sts = [o4.state[p] for p in (p4.w1, p4.w3, p4.w2)]
    occ = [s for s, e in enumerate(p4.slots) if e >= 0]
    fill = [("exp_avg_sq" in st) and bool(torch.allclose(
        st["exp_avg_sq"][s], st["exp_avg"][s] ** 2)) for st in sts for s in occ]
    live = sum(float(st["exp_avg"][s].abs().max()) > 0 for st in sts for s in occ)
    check("moving back to AdamW, each expert on the card starts its second "
          "moment from its momentum",
          occ and all(fill) and live > 0,
          f"{sum(fill)} of {len(fill)} slot tensors, {live} with a momentum")
    try:
        train(m4, c4, o4, steps=2)
        stepped = True
    except KeyError as e:
        stepped = f"KeyError {e}"
    check("...and AdamW steps from there", stepped is True, str(stepped))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  " + ("all one-moment tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
