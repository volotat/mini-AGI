#!/usr/bin/env python3
"""
The Sampler (minagi/decode.py): the established ways of choosing the next
character, one stream at a time.

What this asserts:

  temperature 0 takes the highest logit and draws nothing; the same seed
  draws the same characters and another seed does not

  each cut keeps what its paper keeps: top_k the k likeliest, top_p the
  fewest likeliest holding p, min_p those at least min_p as likely as the
  likeliest, typical_p those whose surprise is nearest the entropy, eta those
  above min(eta, sqrt(eta) e^-H); every cut but typical_p keeps the
  likeliest, and draws come only from what is kept, in proportion

  mirostat's threshold falls after a draw more surprising than its target and
  rises after one less surprising

  DRY charges nothing for a repeat shorter than `dry_allowed`, charges
  multiplier * base ** (length - allowed) to the one character that would
  continue a longer one and nothing to the others, and on a toy model that
  always continues a cycle, greedy writing with DRY leaves the cycle where
  greedy alone never does

  config.yaml's `decoding` is read as the Sampler's settings, a name that is
  none of them is refused rather than ignored, and the rule describes itself
  in a line for the sample log

Every repeated sequence here is built in code rather than written out, so
this file shows the self-knowledge lane no loop to learn from.
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minagi.decode import Sampler, dry_penalty, sampler_from  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


V = 12
probs = torch.tensor([0.30, 0.22, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02,
                      0.015, 0.01, 0.003, 0.002])
logits = probs.log()
hist = [1, 2, 3]


def kept(**kw):
    """The characters a setting ever draws, over many draws."""
    s = Sampler(temperature=1.0, seed=0, **kw)
    return {s(logits, hist) for _ in range(3000)}


print("greedy and seeds")
check("temperature 0 takes the highest logit",
      Sampler()(logits, hist) == 0)
a = [Sampler(temperature=1.0, seed=5)(logits, hist) for _ in range(1)]
s1, s2, s3 = (Sampler(temperature=1.0, seed=5), Sampler(temperature=1.0, seed=5),
              Sampler(temperature=1.0, seed=6))
d1 = [s1(logits, hist) for _ in range(200)]
d2 = [s2(logits, hist) for _ in range(200)]
d3 = [s3(logits, hist) for _ in range(200)]
check("the same seed draws the same characters", d1 == d2)
check("another seed draws others", d1 != d3)

print("cuts")
check("top_k 3 keeps the three likeliest", kept(top_k=3) == {0, 1, 2})
check("top_k 1 is the likeliest alone", kept(top_k=1) == {0})
# 0.30 + 0.22 + 0.15 = 0.67 < 0.7, so the fourth is needed to reach 0.7
check("top_p 0.7 keeps the fewest holding 0.7", kept(top_p=0.7) == {0, 1, 2, 3})
check("top_p 0.3 keeps the likeliest alone", kept(top_p=0.3) == {0})
check("min_p 0.3 keeps those at least 0.3 as likely as the top",
      kept(min_p=0.3) == {0, 1, 2, 3})
check("min_p 1 keeps the likeliest alone", kept(min_p=1.0) == {0})
lp = probs.log()
ent = float(-(probs * lp).sum())
order = torch.argsort((-lp - ent).abs()).tolist()
want, mass = set(), 0.0
for i in order:
    want.add(i)
    mass += float(probs[i])
    if mass >= 0.5:
        break
check("typical_p 0.5 keeps the characters nearest the entropy",
      kept(typical_p=0.5) == want, f"{sorted(want)}")
cut = min(0.05, math.sqrt(0.05) * math.exp(-ent))
check("eta keeps those above its threshold",
      kept(eta=0.05) == {i for i in range(V) if probs[i] > cut},
      f"threshold {cut:.4f}")
s = Sampler(temperature=1.0, top_k=4, seed=1)
n = 20000
counts = np.bincount([s(logits, hist) for _ in range(n)], minlength=V)
want = (probs[:4] / probs[:4].sum()).numpy()
check("draws follow what is kept, in proportion",
      np.abs(counts[:4] / n - want).max() < 0.015 and counts[4:].sum() == 0,
      f"{np.round(counts[:4] / n, 3)} against {np.round(want, 3)}")

print("mirostat")
s = Sampler(temperature=1.0, mirostat=3.0, mirostat_lr=0.1, seed=2)
mu0 = s.mu
s(logits, hist)
check("its threshold rises after a draw less surprising than the target",
      s.mu > mu0, f"{mu0:.3f} -> {s.mu:.3f}")
s = Sampler(temperature=1.0, mirostat=0.05, mirostat_lr=0.1, seed=2)
s.mu = 50.0                                 # nothing is cut
mu0 = s.mu
s(torch.zeros(V), hist)                     # every draw is log2(12) bits
check("and falls after one more surprising",
      s.mu < mu0, f"{mu0:.3f} -> {s.mu:.3f}")

print("DRY")
rng = np.random.default_rng(0)
block = list(rng.integers(0, V, 20))
for b in range(len(block) - 1):          # no character follows itself twice
    if block[b + 1] == block[b]:
        block[b + 1] = (block[b] + 1) % V
head = list(rng.integers(0, V, 40))
for length in (5, 12):
    h = head + block + list(rng.integers(0, V, 30)) + block[:length]
    pen = dry_penalty(h, V, 0.8, 1.15, 8)
    nxt = block[length]
    if length < 8:
        check(f"a repeat of {length} costs nothing", float(pen.max()) == 0.0)
    else:
        want = 0.8 * 1.15 ** (length - 8)
        others = [float(pen[c]) for c in range(V) if c != nxt]
        check(f"a repeat of {length} costs its next character "
              f"0.8 x 1.15^{length - 8}",
              abs(float(pen[nxt]) - want) < 1e-4 and max(others) < want,
              f"{float(pen[nxt]):.4f} against {want:.4f}")

cycle = list(rng.permutation(V))[:7]


def toy(history):
    """A model that always continues the cycle it is in, and starts it again
    from anywhere else."""
    z = torch.full((V,), -2.0)
    at = cycle.index(history[-1]) if history[-1] in cycle else -1
    z[cycle[(at + 1) % len(cycle)]] = 2.0
    return z


def write(sampler, n=120):
    h = [cycle[0]]
    for _ in range(n):
        h.append(sampler(toy(h), h))
    return h


g = write(Sampler())
check("greedy alone never leaves the cycle", all(c in cycle for c in g))
d = write(Sampler(dry=0.8, dry_base=1.15, dry_allowed=8))
off = [i for i, c in enumerate(d) if c not in cycle]
check("greedy with DRY leaves it", bool(off), f"first at {off[:1]}")

print("config.yaml")
s = sampler_from({"decoding": {"dry": 0.8, "dry_allowed": 16}}, seed=3)
check("decoding's keys are the Sampler's settings",
      s.dry == 0.8 and s.dry_allowed == 16 and s.temperature == 0.0)
try:
    sampler_from({"decoding": {"rep_penalty": 1.0}})
    refused = False
except ValueError as e:
    refused = "rep_penalty" in str(e)
check("a name that is no setting is refused, by name", refused)
check("no decoding at all is greedy", sampler_from({})(logits, hist) == 0)
check("the rule says itself in a line",
      s.describe() == "greedy, DRY 0.8 x 1.15^(repeat - 16)"
      and Sampler(temperature=0.7, min_p=0.1).describe()
      == "temperature 0.7, min_p 0.1", f"{s.describe()!r}")

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
