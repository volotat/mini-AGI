"""
The optimiser every learner steps the model with: AdamW, as the run has
always used, or Muon for the weight matrices (training.optimizer).

MUON (Jordan et al., 2024) steps a weight matrix by its gradient's momentum,
orthogonalised: five Newton-Schulz iterations take the momentum to the
nearest matrix whose singular values are all about one, so a step moves the
weights as far along every direction the gradient points as along its
strongest few. The step is scaled by 0.2 * sqrt(max(rows, cols)) - the
match Liu et al. (2025, "Muon is Scalable for LLM Training") use to give it
the size of AdamW's typical step - so one learning rate, and the controller
that scales it (minagi/plasticity.py), mean the same thing under either.

WHAT IT STEPS, as Liu et al. do: every weight matrix - attention's, the
dense MLPs', the adapter's, each expert's three, and the router's. The pool
keeps the experts' as stacks, one matrix per slot of the card, and each is
orthogonalised on its own; attention's query, key and value are one tensor
here and three matrices there, so they are orthogonalised as three.
Everything that is not a matrix stays on AdamW, which this optimiser runs
for it, as there: the embedding and the head, the norms, the halting unit,
the gates and the depth embeddings. training.muon_params narrows Muon to the
trunk's matrices or to the pool's.

SWITCHING A RUN TO IT. Liu et al. found that a model pretrained with AdamW
and then fine-tuned with Muon did no better than one fine-tuned with AdamW
(their section 3.5): the gain they measured came from training with Muon
from the start. A run that moves to it mid-way is that case.

ITS MOMENT IS ADAMW'S FIRST. Muon keeps one moment, a momentum, and keeps it
where AdamW keeps its first moment - exp_avg - so the experts' files carry
it with them as they carry AdamW's (minagi/paged.py), checkpoints save it as
they save AdamW's (minagi/store.py), and a run that moves from AdamW starts
from the momentum AdamW had built rather than from zero. It keeps no second
moment: a matrix Muon steps carries none on the card, in RAM or in its
file, where AdamW's would be a quarter of an expert's bytes, and one restored
from an AdamW checkpoint is let go at its first step. A run that moves back
to AdamW starts each such matrix's second moment from its momentum squared
(second_moment), which holds its first steps to about the rate until AdamW's
own average - beta2 0.95, about twenty steps - takes over.

AN EXPERT ON THE CARD THAT NO CHARACTER CHOSE has no gradient. AdamW steps
it anyway, by a momentum that fades. Muon's step has the same size whatever
the momentum's, so it would take a full step along a stale direction - so a
slot without a gradient is not stepped and keeps its momentum. Its weight
decay applies, as AdamW's does.
"""

import math

import torch

# the quintic Newton-Schulz coefficients of Jordan et al.
NS = (3.4445, -4.7750, 2.0315)
SLOTS_AT_ONCE = 8          # experts orthogonalised together: memory, not maths


def orthogonalise(G, steps=5, eps=1e-7):
    """The matrix (or each of a stack of them, over the leading dimension)
    nearest to G with every singular value about one - in bfloat16, as Muon
    computes it."""
    a, b, c = NS
    X = G.to(torch.bfloat16)
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


def second_moment(m):
    """The second moment AdamW starts from for a tensor that has none - one
    Muon stepped, which kept only a momentum: the momentum squared. Zeros for
    a tensor with no history; otherwise a denominator that holds each
    coordinate's step to about the rate, where zeros would let a small
    gradient beside an old momentum take a step of any size."""
    return m * m


def keeps_second(opt, p):
    """Whether `opt` keeps AdamW's second moment for parameter p: AdamW
    does, Muon does not."""
    if not isinstance(opt, MuonAdamW):
        return True
    return not any(g["muon"] for g in opt.param_groups
                   if any(q is p for q in g["params"]))


def takes_muon(name, p, which="all", pool_ids=()):
    """Whether Muon steps this parameter: a weight matrix - two dimensions,
    or a stack of the experts' - that is not the embedding, the head or the
    halting unit."""
    if which == "trunk" and id(p) in pool_ids:
        return False
    if which == "pool" and id(p) not in pool_ids:
        return False
    if p.dim() == 3:
        return True
    if p.dim() != 2 or min(p.shape) < 2:
        return False
    return not any(s in name for s in ("tok_emb", "head", "halt",
                                       "depth_emb"))


class MuonAdamW(torch.optim.Optimizer):
    """Muon for the groups marked `muon`, AdamW for the rest."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, muon=False,
                                      momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps))

    def _state(self, p, second=True):
        st = self.state[p]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(p)
        if not second:
            st.pop("exp_avg_sq", None)
        elif "exp_avg_sq" not in st:
            st["exp_avg_sq"] = second_moment(st["exp_avg"])
        # counted on the host: a counter on the card would make every
        # parameter's step wait for the one before it
        st["step"] = torch.tensor(float(st.get("step", 0.0)) + 1.0)
        return st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            (self._muon if g["muon"] else self._adamw)(g)
        return loss

    def _adamw(self, g):
        """torch's AdamW, step for step."""
        b1, b2 = g["betas"]
        lr, wd, eps = g["lr"], g["weight_decay"], g["eps"]
        for p in g["params"]:
            if p.grad is None:
                continue
            st = self._state(p)
            t = float(st["step"])
            grad, m, v = p.grad, st["exp_avg"], st["exp_avg_sq"]
            p.mul_(1.0 - lr * wd)
            m.lerp_(grad, 1.0 - b1)
            v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            denom = (v.sqrt() / math.sqrt(1.0 - b2 ** t)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / (1.0 - b1 ** t))

    def _muon(self, g):
        lr, wd = g["lr"], g["weight_decay"]
        mom, nest, ns = g["momentum"], g["nesterov"], g["ns_steps"]
        k = int(g.get("split", 1))
        for p in g["params"]:
            if p.grad is None:
                continue
            st = self._state(p, second=False)
            P, grad, buf = p, p.grad, st["exp_avg"]
            p.mul_(1.0 - lr * wd)
            if k > 1:
                # one tensor holding k matrices along its rows - attention's
                # query, key and value - orthogonalised as the k they are
                P, grad, buf = (t.view(k, t.shape[0] // k, t.shape[1])
                                for t in (p, grad, buf))
            scale = 0.2 * math.sqrt(max(P.shape[-2], P.shape[-1]))
            if P.dim() == 2:
                buf.lerp_(grad, 1.0 - mom)
                u = grad.lerp(buf, mom) if nest else buf
                P.add_(orthogonalise(u, ns).to(p.dtype), alpha=-lr * scale)
                continue
            for i in range(0, P.shape[0], SLOTS_AT_ONCE):
                sl = slice(i, i + SLOTS_AT_ONCE)
                gr, bf = grad[sl], buf[sl]
                # a slot no character chose this step: no step, and its
                # momentum kept as it was
                live = (gr.flatten(1).abs().amax(1) > 0).to(gr.dtype)
                live = live.view(-1, 1, 1)
                bf.lerp_(gr, live * (1.0 - mom))
                u = gr.lerp(bf, mom) if nest else bf
                P[sl].add_(orthogonalise(u, ns).to(p.dtype) * live,
                           alpha=-lr * scale)


def build(model, trunk, pool, lr, trunk_lr, wd, kind="adamw", fused=False,
          which="all"):
    """The optimiser for a model split into `trunk` and `pool` parameters
    (minagi/pool.py, split_trunk_pool), each group at its own rate with its
    schedule's anchor beside it: AdamW, or Muon for the weight matrices."""
    if kind != "muon":
        return torch.optim.AdamW(
            [{"params": trunk, "name": "trunk", "weight_decay": wd,
              "lr": trunk_lr, "base_lr": trunk_lr},
             {"params": pool, "name": "pool", "weight_decay": wd,
              "lr": lr, "base_lr": lr}],
            lr=lr, betas=(0.9, 0.95), fused=fused)
    names = {id(p): n for n, p in model.named_parameters()}
    pool_ids = {id(p) for p in pool}

    def split(ps, name, rate):
        on = [p for p in ps if takes_muon(names.get(id(p), ""), p, which,
                                          pool_ids)]
        qkv = [p for p in on if names.get(id(p), "").endswith("qkv.weight")]
        on = [p for p in on if not names.get(id(p), "").endswith("qkv.weight")]
        off = [p for p in ps if not takes_muon(names.get(id(p), ""), p,
                                               which, pool_ids)]
        return [dict({"params": q, "name": name, "muon": m,
                      "weight_decay": wd, "lr": rate, "base_lr": rate}, **x)
                for q, m, x in ((on, True, {}), (qkv, True, {"split": 3}),
                                (off, False, {})) if q]
    # the pool's AdamW group last: the progress line reads the last group's
    # rate as the pool's
    opt = MuonAdamW(split(trunk, "trunk", trunk_lr) + split(pool, "pool", lr),
                    lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    opt.which = which
    return opt


def adopt(opt, model, ps):
    """New parameters of the pool's - a gate or router rows that growth or
    pruning replaced - sorted by whether Muon steps them: [(muon, params)].
    A plain AdamW steps them all."""
    if not isinstance(opt, MuonAdamW):
        return [(False, ps)]
    names = {id(p): n for n, p in model.named_parameters()}
    ids = {id(p) for p in ps}
    which = getattr(opt, "which", "all")
    on = [p for p in ps if takes_muon(names.get(id(p), ""), p, which, ids)]
    off = [p for p in ps if not takes_muon(names.get(id(p), ""), p, which,
                                           ids)]
    return [(m, q) for m, q in ((True, on), (False, off)) if q]


def describe(opt):
    """One line on what steps what."""
    if not isinstance(opt, MuonAdamW):
        return "AdamW for every parameter"
    mu = sum(p.numel() for g in opt.param_groups if g["muon"]
             for p in g["params"])
    ad = sum(p.numel() for g in opt.param_groups if not g["muon"]
             for p in g["params"])
    return (f"Muon for the weight matrices ({mu / 1e6:.1f}M on the card), "
            f"AdamW for the rest ({ad / 1e6:.2f}M)")
