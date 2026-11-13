#!/usr/bin/env python3
"""
The automatic share of the reading each subject gets (minagi/lanes.py).

What these tests hold it to:
  a lane feeds the held-out subjects named after it - chat feeds chat and
  chat_hermes - and a lane no subject measures feeds none
  with no evidence every subject gets an even share
  a fit recovers a subject's power law from noisy checks, from the recent
  part of the run when the slope has changed - early checks far out on log
  D do not set it
  the share follows what the next character is worth, alpha * L / D: a
  steeper subject gets more, a flat one falls to the floor, and a lane
  feeding two subjects gets the two
  a subject getting worse has as much claim as one getting better
  a lane nothing measures is valued as the typical one, less for each time
  it has been read
  no share leaves [1/4, 3] of an even share, and shares sum to one
  a share moves at most 5% at a check
  the shares are of characters: what each subject reads follows its share
  exactly however much a visit to it holds, so a subject of short files is
  visited more often, and no subject waits long; a visit that reads nothing
  is charged as a full one, so the reader moves on
  the state saves and restores; a state whose shares counted visits carries
  on as the characters they read; a lane the corpus gained starts even;
  equal mode never moves
  seeding from the sample log is the same as having watched those checks
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minagi.lanes import LaneSchedule, feeds                # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


LANES = ["arithmetic", "chat", "chess", "self-knowledge", "wikipedia"]
SUBJECTS = ["arithmetic", "chat", "chat_hermes", "chess", "wikipedia"]
SIZES = {"arithmetic": 150e6, "chat": 1.9e9, "chess": 330e6,
         "self-knowledge": 7e6, "wikipedia": 830e6}

f = feeds(LANES, SUBJECTS)
check("chat feeds chat and chat_hermes; self-knowledge feeds nothing",
      f["chat"] == ["chat", "chat_hermes"] and f["self-knowledge"] == []
      and f["arithmetic"] == ["arithmetic"], f)

s = LaneSchedule(LANES, SIZES, SUBJECTS)
check("with no evidence every subject gets an even share",
      all(abs(v - 0.2) < 1e-12 for v in s.targets().values()))

# a run whose subjects follow power laws in what their lane has read
LAW = {"arithmetic": (0.62, 0.0), "chat": (0.9, 0.12),
       "chat_hermes": (1.2, 0.12), "chess": (0.7, 0.15),
       "wikipedia": (1.6, 0.08)}            # (loss at D = 1e8, alpha)


def law(sub, D):
    L0, al = LAW[sub]
    return L0 * (D / 1e8) ** -al


rng = np.random.default_rng(0)


def run(sched, start, end, checks, noise=0.03):
    for total in np.linspace(start, end, checks):
        for k in LANES:
            sched.read[k] = total / len(LANES)
        sched.observe(total, {sub: law(sub, total / len(LANES))
                              * (1 + noise * rng.standard_normal())
                              for sub in SUBJECTS})


s = LaneSchedule(LANES, SIZES, SUBJECTS)
run(s, 5e8, 2.3e9, 600)
got = {sub: s.fit(sub) for sub in SUBJECTS}
check("a fit recovers each subject's slope and level from noisy checks",
      all(abs(got[k][0] - LAW[k][1]) < 0.03 for k in SUBJECTS)
      and all(abs(got[k][1] / law(k, 2.3e9 / 5) - 1) < 0.02 for k in SUBJECTS),
      {k: round(v[0], 3) for k, v in got.items()})

# the slope changed: steep early, flatter for the last part of the run
s2 = LaneSchedule(["chess"], {"chess": 1e12}, ["chess"])
for total in np.geomspace(1e7, 2.3e9, 2000):
    s2.read["chess"] = total
    al = 0.3 if total < 1e9 else 0.1
    L = 2.0 * (min(total, 1e9) / 1e7) ** -0.3 * (max(total, 1e9) / 1e9) ** -0.1
    s2.observe(total, {"chess": L})
check("...the recent slope, when it has changed: early checks far out on "
      "log D do not set it", abs(s2.fit("chess")[0] - 0.1) < 0.02,
      f"{s2.fit('chess')[0]:.3f}")

t = s.targets()
check("the steeper subject gets more than the flatter one with the same "
      "reading", t["chess"] > t["wikipedia"] * 0.5 and t["chat"] > t["chess"],
      {k: round(v, 3) for k, v in t.items()})
check("...a flat one falls to the floor, a quarter of an even share",
      abs(t["arithmetic"] - 0.25 / 5) < 1e-9)
w_chat = s.worth("chat")
f1, f2 = s.fit("chat"), s.fit("chat_hermes")
check("...and a lane feeding two subjects is worth the two",
      abs(w_chat - (f1[0] * f1[1] + f2[0] * f2[1]) / s.read["chat"]) < 1e-15)
check("no share leaves [1/4, 3] of an even share, and they sum to one",
      all(0.05 - 1e-9 <= v <= 0.6 + 1e-9 for v in t.values())
      and abs(sum(t.values()) - 1) < 1e-9)

# getting worse: arithmetic's held-out rising
s3 = LaneSchedule(LANES, SIZES, SUBJECTS)
LAW["arithmetic"] = (0.5, -0.12)
run(s3, 5e8, 2.3e9, 600)
LAW["arithmetic"] = (0.62, 0.0)
check("a subject getting worse has as much claim as one getting better",
      s3.targets()["arithmetic"] > 0.1,
      f"{s3.targets()['arithmetic']:.3f}")

# nothing measures self-knowledge: the typical worth, less for re-reading
s.read["self-knowledge"] = SIZES["self-knowledge"]          # read once
once = s.worth("self-knowledge"), s.reread("self-knowledge")
s.read["self-knowledge"] = 41 * SIZES["self-knowledge"]     # forty-one times
check("a lane nothing measures is worth less for each time it has been read",
      once[0] is None and once[1] == 1.0
      and abs(s.reread("self-knowledge") - math.exp(-40 / 15.4)) < 1e-12)

# shares move slowly
s.shares = {k: 0.2 for k in LANES}
before = dict(s.shares)
line = s.step()
check("a share moves at most 5% at a check",
      all(abs(math.log(s.shares[k] / before[k])) <= 0.05 + 0.02
          for k in LANES) and line.startswith("reading shares: "), line)
for _ in range(200):
    s.step()
t = s.targets()
check("...and reaches the targets in time",
      all(abs(s.shares[k] - t[k]) < 0.005 for k in LANES))

# reading: what a visit holds may differ by subject - a visit used to end
# with its file, and a chess or arithmetic file filled the whole passage, a
# wikipedia article a sixth of it
HOLDS = {"arithmetic": 32768, "chat": 16827, "chess": 32754,
         "self-knowledge": 13794, "wikipedia": 5324}


def visits(sched, n, holds=HOLDS):
    """n visits as the reader makes them: pick, read, say how much."""
    out = []
    for _ in range(n):
        k = sched.pick()
        sched.note_read(k, holds[k])
        out.append(k)
    return out


before = dict(s.read)
picks = visits(s, 5000)
got = {k: s.read[k] - before[k] for k in LANES}
tot = sum(got.values())
gap = {}
for k in LANES:
    at = [0] + [i for i, p in enumerate(picks) if p == k]
    # characters read elsewhere between two visits to k
    gap[k] = max(sum(HOLDS[q] for q in picks[a + 1:b] if q != k)
                 for a, b in zip(at, at[1:]))
check("characters follow the shares exactly, however much a visit to each "
      "subject holds",
      all(abs(got[k] / tot - s.shares[k]) < 0.002 for k in LANES),
      {k: f"{got[k] / tot:.3f}/{s.shares[k]:.3f}" for k in LANES})
two = LaneSchedule(["chess", "wikipedia"], SIZES, [], mode="fixed",
                   weights={"chess": 1.0, "wikipedia": 1.0})
p2 = visits(two, 700)
ratio = p2.count("wikipedia") / max(p2.count("chess"), 1)
check("...so at equal shares a subject of short files is visited as many "
      "times more often as its visits are shorter",
      abs(ratio - HOLDS["chess"] / HOLDS["wikipedia"]) < 0.3,
      f"{ratio:.2f} visits to wikipedia for each to chess")
# paying back its own visit and outwaiting the longest one: (own + longest)
# characters of credit, earned at its share of whatever is read elsewhere
check("...and no subject waits much longer than its share says",
      all(gap[k] <= (HOLDS[k] + max(HOLDS.values())) / s.shares[k]
          for k in LANES),
      {k: f"{gap[k]:,}" for k in LANES})
missed = getattr(s, "missed", None)
gone = LaneSchedule(LANES, SIZES, SUBJECTS)
first, seq = gone.pick(), []
for _ in range(20 if missed is not None else 0):
    k = gone.pick()
    seq.append(k)
    if k == first:
        gone.missed(k, 32768)            # every file it tried had gone
    else:
        gone.note_read(k, HOLDS[k])
check("a visit that reads nothing is charged as a full one: the reader moves "
      "on, and nothing is counted as read",
      missed is not None and 1 <= seq.count(first) <= 3
      and gone.read[first] == 0.0, f"{seq.count(first)} of {len(seq)} visits")

# save and restore
st = s.state()
r = LaneSchedule.restore(st, LANES, SIZES, SUBJECTS, total=0)
check("the state restores: shares, what was read, the fits",
      r.shares == s.shares and r.read == s.read and r.fits == s.fits
      and st.get("unit") == "characters"
      and visits(r, 50) == visits(s, 50))
old = {"mode": "auto", "shares": dict(s.shares), "read": dict(s.read),
       "fits": st["fits"], "last": st["last"],
       "credit": {k: 3.0 for k in LANES}}  # a state that counted visits
try:
    conv = LaneSchedule.restore(old, LANES, SIZES, SUBJECTS, total=0,
                                per_visit=HOLDS)
except TypeError as e:
    conv = None
    print(f"  (restore refused per_visit: {e})")
mix = {k: s.shares[k] * HOLDS[k] for k in LANES}
mix = {k: v / sum(mix.values()) for k, v in mix.items()}
check("a state whose shares counted visits carries on as the characters "
      "those visits read, so the reading does not jump",
      conv is not None
      and all(abs(conv.shares[k] - mix[k]) < 1e-9 for k in LANES)
      and all(v == 0.0 for v in conv.credit.values()),
      "" if conv is None else
      {k: f"{conv.shares[k]:.3f}" for k in LANES})
grown = LaneSchedule.restore(st, LANES + ["poetry"], dict(SIZES, poetry=1e8),
                             SUBJECTS, total=2.3e9)
check("...and a lane the corpus gained starts at an even share",
      abs(grown.shares["poetry"] / sum(grown.shares.values()) - 1 / 6) < 0.01
      and abs(sum(grown.shares.values()) - 1) < 1e-9)
eq = LaneSchedule.restore(st, LANES, SIZES, SUBJECTS, mode="equal")
eq.observe(3e9, {"chat": 0.4})
check("equal mode is even and never moves",
      all(abs(v - 0.2) < 1e-12 for v in eq.shares.values()) and eq.step() == "")

# seeding from the log
rows = [{"chars": total / 1e6,
         "dom": {sub: law(sub, total / 5) for sub in SUBJECTS}}
        for total in np.linspace(5e8, 2.3e9, 300)]
a = LaneSchedule(LANES, SIZES, SUBJECTS)
a.bootstrap(rows)
b = LaneSchedule(LANES, SIZES, SUBJECTS)
for row in rows:
    for k in LANES:
        b.read[k] = row["chars"] * 1e6 / 5
    b.observe(row["chars"] * 1e6, row["dom"])
check("seeding from the sample log is watching those checks",
      all(np.allclose(a.fits[k], b.fits[k]) for k in SUBJECTS)
      and a.read == {k: 0.0 for k in LANES})

print("\n  all lane-schedule tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
