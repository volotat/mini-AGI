#!/usr/bin/env python3
"""
The switches the expert-pool A/B tests (tools/ab_pool.py): how the chosen
experts are weighted, the routing bias, a newborn's gate - and the
config file a run reads them from.

What these tests hold it to:
  MINAGI_CONFIG names the config file every part of a run reads
  weighted before the gates (as the run is), choosing an expert gated near
  zero takes weight from the expert chosen beside it; weighted after them
  (pool.gate_after) every character's weights add up to one and that
  expert's share is near zero - on the same choices, since gates do not
  choose; weighted after them and scaled, experts gated alike are mixed
  exactly as before the gates, and a weak one among them takes nothing
  the routing bias is not used while it is off; on, it moves which experts
  are chosen - for the card's vote and each character's pick - but not the
  weights they are mixed with
  after a step it moves each offset toward its rank's share of the curve
  1 / (rank + offset): an expert taking more than that comes down, one taking
  less goes up, centred so they do not drift together, within the clamp;
  with the setting off it only clears the step's counts
  under the floor rule (pool.route_floor) only an expert inside its trial
  is lifted: below the floor it rises a step, at or above it it lets go a
  step at a time but never goes below 0, within the clamp; when its trial
  ends, or while no trial is set, it has no offset - and an old expert's
  leftover offset is cleared; learning outside training reads the same
  trial, in the run's steps
  a forward's requests carry the router's own probability, or under the
  floor rule the lifted one, so an expert the router barely wants can win
  the card's vote once it is lifted - and the forward reads which rule is
  in force from the pool
  the offsets and the shares they steer by are saved with the weights, and a
  newborn starts with none
  a newborn's gate is a share of its parents' gates under
  growth.birth_gate_parents, the birth_gate it is given otherwise
  build_paged reads all of them from the config, and train.py can write
  every step's loss and source for comparing runs chunk by chunk
"""
import inspect
import json
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-switches-")
try:
    from minagi import config as C                           # noqa: E402
    from minagi.create import create                         # noqa: E402
    from minagi.pool import PooledMLP, capture_routes        # noqa: E402
    import train as T                                        # noqa: E402

    # --- the config file -----------------------------------------------------
    alt = os.path.join(tmp, "alt.yaml")
    with open(alt, "w") as f:
        f.write("pool:\n  gate_after: scaled\n  route_bias: 2.0e-4\n"
                "  route_bias_offset: 10\n  route_floor: 0.3\n"
                "growth:\n  birth_gate_parents: 0.1\n")
    os.environ["MINAGI_CONFIG"] = alt
    got = C.load()
    os.environ.pop("MINAGI_CONFIG")
    check("MINAGI_CONFIG names the config file a run reads",
          C.get(got, "pool.gate_after") == "scaled"
          and C.get(C.load(), "pool.gate_after") != "scaled")

    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False,
           experts=12, resident=4, d_ff=32, d_model=32, trunk_d_ff=32,
           n_head=2, block=128, max_steps=3, top_k=2, depth=1)
    os.environ["MINAGI_CONFIG"] = alt
    model, cfg, pool, man = T.build_paged(wdir, "cpu", read_only=True)
    os.environ.pop("MINAGI_CONFIG")
    check("build_paged reads the switches from the config",
          pool.gate_after == "scaled" and pool.route_bias == 2e-4
          and pool.route_bias_offset == 10 and pool.birth_gate_parents == 0.1
          and pool.route_floor == 0.3)
    pool.gate_after, pool.route_bias, pool.birth_gate_parents = False, 0.0, 0.0
    pool.route_floor = 0.0
    model.eval()
    TEXT = ("add 2856 + 741 = 3597\nadd 12 + 30 = 42\nmul 7 * 6 = 42\n"
            "the cat sat on the mat and the dog sat on the log\n")
    X = torch.tensor([[b for b in TEXT.encode()][:cfg.block]])

    def routes():
        """(experts, weights) per character, at every call site, in order."""
        with torch.no_grad(), capture_routes() as got:
            model(X)
        return [(e.reshape(-1, e.shape[-1]), w.reshape(-1, w.shape[-1]))
                for e, w in got]

    # --- weighted before or after the gates ------------------------------------
    pool.gate.data.fill_(1.0)
    first = routes()
    hits = torch.bincount(torch.cat([e.reshape(-1) for e, _ in first]),
                          minlength=pool.n_experts())
    weak = int(hits.argmax())
    pool.gate.data[weak] = 1e-6
    before = routes()
    pool.gate_after = True
    after = routes()
    pool.gate_after = False
    with_weak = [(eb, wb, wa) for (eb, wb), (ea, wa) in zip(before, after)]
    rows_b, rows_a, ws = [], [], []
    for eb, wb, wa in with_weak:
        m = (eb == weak).any(-1)
        rows_b.append(wb[m].sum(-1))
        rows_a.append(wa.sum(-1))
        ws.append(wa[eb == weak])
    rb, ra, wk = torch.cat(rows_b), torch.cat(rows_a), torch.cat(ws)
    same = all(torch.equal(eb, ea) for (eb, _), (ea, _) in zip(before, after))
    check("weighted before the gates, choosing an expert gated near zero "
          "takes weight from the one beside it",
          len(rb) > 0 and float(rb.max()) < 0.999, f"{len(rb)} characters")
    check("...weighted after them, every character's weights add up to one "
          "and that expert's share is near zero, on the same choices",
          same and torch.allclose(ra, torch.ones_like(ra), atol=1e-5)
          and float(wk.abs().max()) < 1e-4)
    # scaled: experts gated alike mix as before the gates; a weak one among
    # them takes nothing, and the rest keep the whole of their gates' scale
    pool.gate.data.fill_(1.5)
    alike_b = routes()
    pool.gate_after = "scaled"
    alike_s = routes()
    pool.gate.data[weak] = 1e-6
    weak_s = routes()
    pool.gate_after = False
    same_mix = all(torch.allclose(wb, ws, atol=1e-5)
                   for (_, wb), (_, ws) in zip(alike_b, alike_s))
    tot = torch.cat([w.sum(-1) for _, w in weak_s])
    wk_s = torch.cat([w[e == weak] for e, w in weak_s])
    check("...weighted after them and scaled, experts gated alike mix exactly "
          "as before the gates, and a weak one among them takes nothing from "
          "the rest",
          same_mix and torch.allclose(tot, torch.full_like(tot, 1.5), atol=1e-4)
          and float(wk_s.abs().max()) < 1e-4)
    pool.gate.data.fill_(1.0)

    # --- the routing bias -------------------------------------------------------
    check("the routing bias is not used while it is off",
          pool.selection_bias() is None)
    base = routes()
    never = int((torch.bincount(torch.cat([e.reshape(-1) for e, _ in base]),
                                minlength=pool.n_experts())).argmin())
    pool.route_bias = 1e-3
    pool.bias.zero_()
    pool.bias[never] = 100.0
    forced = routes()
    took = all(bool((e == never).any(-1).all()) for e, _ in forced)
    wf = torch.cat([w[e == never] for e, w in forced])
    check("on, it moves which experts are chosen - the card's vote and every "
          "character's pick take the one it lifts",
          took and never in pool._admitted, f"expert {never}")
    check("...but not the weights they are mixed with: the lifted expert "
          "keeps the router's own share, far from all of it",
          float(wf.max()) < 0.95, f"max weight {float(wf.max()):.3f}")
    pool.bias.zero_()

    # --- the controller ---------------------------------------------------------
    n = pool.n_experts()
    pool.chosen.zero_()
    pool.chosen[:n] = 1.0 / n
    pool._step_hits.zero_()
    pool._step_hits[0] = 1000.0                # one expert takes every choice
    pool.chosen_steps = 1.0                    # this step is the whole share
    pool._steer_bias()
    # centred: one comes down by (n - 1) steps of the others' rise
    up = 1e-3 * 2.0 / n
    check("after a step an expert taking more than its rank's share comes "
          "down, one taking less goes up - centred, so the offsets add up to "
          "what they did",
          abs(float(pool.bias[0]) + (n - 1) * up) < 1e-7
          and all(abs(float(pool.bias[i]) - up) < 1e-7 for i in range(1, n))
          and abs(float(pool.bias[:n].sum())) < 1e-6
          and float(pool._step_hits.sum()) == 0.0)
    pool.bias[:n] = 3.9999
    pool._step_hits[1] = 5.0
    pool._steer_bias()
    check("...within the clamp", float(pool.bias[:n].max()) <= 4.0)
    pool.route_bias = 0.0
    pool.bias.zero_()
    pool._step_hits[2] = 7.0
    pool._steer_bias()
    check("...and with the setting off it only clears the step's counts",
          float(pool.bias.abs().sum()) == 0.0
          and float(pool._step_hits.sum()) == 0.0)

    # --- the floor ----------------------------------------------------------------
    def steer(starved=None):
        """One step in which every expert takes an even share of the choices,
        but `starved` takes none."""
        pool._step_hits[:n] = 10.0
        if starved is not None:
            pool._step_hits[starved] = 0.0
        pool._steer_bias()

    pool.route_bias, pool.route_floor = 1e-3, 0.1
    pool.chosen_steps = 1.0                    # each step is the whole share
    pool.trial, pool.now = 100, 1000
    pool.born[:n] = 0.0
    pool.born[:3] = 950.0                      # three experts on trial
    pool.bias.zero_()
    pool.bias[5] = 0.7                         # an old expert's leftover offset
    steer(starved=0)
    check("under the floor rule an expert on trial below the floor rises a "
          "step, one on trial at an even share stays at 0, and an old expert "
          "has no offset - its leftover is cleared",
          abs(float(pool.bias[0]) - 1e-3) < 1e-9
          and float(pool.bias[1:n].abs().sum()) == 0.0)
    steer(starved=0)
    steer()
    check("...it lets go a step at a time once its own row earns the share",
          abs(float(pool.bias[0]) - 1e-3) < 1e-9)
    steer()
    steer()
    check("...but never below 0", float(pool.bias[0]) == 0.0)
    pool.bias[0] = 3.9995
    steer(starved=0)
    check("...within the clamp", abs(float(pool.bias[0]) - 4.0) < 1e-6)
    pool.now = 1100                            # 150 steps old: its trial is over
    steer(starved=0)
    check("...when its trial ends it has no offset, whatever it took",
          float(pool.bias[:n].abs().sum()) == 0.0)
    pool.trial, pool.now = 0, 960
    steer(starved=0)
    check("...and while no trial is set nothing is lifted",
          float(pool.bias[:n].abs().sum()) == 0.0)

    # --- what the card's vote counts ----------------------------------------------
    from minagi.pool import requests                     # noqa: E402
    sc = torch.zeros(64, 6)
    sc[:, 5] = -20.0                           # an expert the router barely wants
    sc[:, 0] = 2.0
    lift = torch.zeros(6)
    lift[5] = 30.0
    own = requests(sc, 2, lift)
    lifted_mass = requests(sc, 2, lift, lifted=True)
    plain = requests(sc, 2)
    check("a forward's requests carry the router's own probability: lifted into "
          "every character's top 2, the barely-wanted expert still brings "
          "almost nothing to the vote",
          float(plain[5]) == 0.0 and float(own[5]) < 1e-6
          and int(own.argmax()) == 0)
    check("...and the lifted probability under the floor rule, so once lifted "
          "it can win the card's vote",
          int(lifted_mass.argmax()) == 5 and float(lifted_mass[5]) > 60.0)
    pool.route_bias, pool.route_floor = 1e-3, 0.1
    rules = (pool.vote_lifted(), )
    pool.route_floor = 0.0
    rules += (pool.vote_lifted(), )
    pool.route_bias = 0.0
    pool.route_floor = 0.1
    rules += (pool.vote_lifted(), )
    pool.route_floor = 0.0
    route_src = inspect.getsource(PooledMLP._route)
    check("...the forward reads which rule is in force from the pool: lifted "
          "only under the floor rule with the bias on",
          rules == (True, False, False) and "vote_lifted()" in route_src
          and "requests(z" in route_src)
    pool.bias.zero_()
    pool.chosen_steps = 50.0
    from minagi.live import LiveLearner                  # noqa: E402
    pool.trial = 0
    LiveLearner._for_run(model, 3e-4, man,
                         {"prune": {"survival_chars": 12_800},
                          "training": {"chunk": 128}}, None, 128, None, 8)
    check("learning outside training - chat, the benchmarks - reads a "
          "newborn's trial too, in the run's steps, so the floor rule does not "
          "take every lift away there",
          pool.trial == 100, f"trial {pool.trial}")
    pool.trial = 0

    # --- saved with the weights; growth; a newborn's gate ------------------------
    pool.bias[:n] = torch.linspace(-1, 1, n)
    pool.chosen[:n] = torch.linspace(0, 0.1, n)
    tel = pool.telemetry()
    m2, _, pool2, _ = T.build_paged(wdir, "cpu", read_only=True)
    pool2.load_telemetry(json.loads(json.dumps(tel)))
    check("the offsets and the shares they steer by are saved with the "
          "weights",
          torch.allclose(pool2.bias[:n], pool.bias[:n], atol=1e-5)
          and torch.allclose(pool2.chosen[:n], pool.chosen[:n], atol=1e-7))

    gw = os.path.join(tmp, "g")
    shutil.copytree(wdir, gw)
    m3, _, pool3, _ = T.build_paged(gw, "cpu")
    pool3.gate.data.fill_(2.0)
    pool3.bias[:n] = 0.5
    pool3.birth_gate_parents = 0.1
    pool3.add_experts(2, step=1, birth_gate=0.001)
    check("a newborn starts with no offset and no share",
          pool3.bias.numel() == n + 2 and float(pool3.bias[n:].abs().sum()) == 0
          and pool3.chosen.numel() == n + 2
          and pool3._step_hits.numel() == n + 2)
    check("...and under growth.birth_gate_parents its gate is that share of its "
          "parents' gates",
          torch.allclose(pool3.gate.data[n:], torch.full((2,), 0.2)),
          f"{pool3.gate.data[n:].tolist()}")
    pool3.birth_gate_parents = 0.0
    pool3.add_experts(1, step=2, birth_gate=0.001)
    check("...the birth_gate it is given otherwise",
          abs(float(pool3.gate.data[-1]) - 0.001) < 1e-9)

    src = inspect.getsource(T.cmd_read)
    check("train.py can write every step's loss and source, for comparing "
          "runs chunk by chunk",
          "loss_log.write(" in src and '"--loss-log"' in inspect.getsource(T))
finally:
    os.environ.pop("MINAGI_CONFIG", None)
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  all pool-switch tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
