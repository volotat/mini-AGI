#!/usr/bin/env python3
"""
Everything samples.txt records, as one picture.

The sample log is the only continuous record of a run - it is appended at every
checkpoint and survives restarts, so it reaches back further than anything
held in memory. This parses it and plots against CHARACTERS READ rather than
step, because the context window moves during a run and a step is not a fixed
amount of reading.
"""
import argparse
import re

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi import heldout                                  # noqa: E402

# The header gained the corpus total partway through the project, so both
# shapes appear in one file and the older half is the early history.
HEAD = heldout.LOG_HEAD


DYING_AT, SURVIVAL_CHARS, CHUNK = 0.75, 0, 1536
try:                                                        # noqa: E402
    import yaml
    _c = yaml.safe_load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml")))
    DYING_AT = float(_c["prune"].get("dying_at", 0.75))
    SURVIVAL_CHARS = int(str(_c["prune"].get("survival_chars", 0)).replace("_", ""))
    CHUNK = int(_c["training"].get("chunk", 1536))
except Exception:
    pass


def _dead_pct(r):
    """Share of the pool that has gone `dying_at` of the way to being deleted.

    DEAD MEANS UNADDRESSED. This read the gate until the gate turned out to be
    backwards - asked which experts would take no traffic over the next 11M
    characters, lowest-gate picked them at 8% against a 63% base rate, because
    the smallest gates belong to the busiest experts. It is now the same
    staleness prune acts on.

    Older rows carry no `last_seen`, so `since` - the segment an expert was
    last ADMITTED - stands in. It differs only for an expert that stayed
    resident without being re-admitted, which makes it a slight OVER-estimate
    of how dead the pool is.
    """
    n = len(r.get("gate") or [])
    if not n or not SURVIVAL_CHARS:
        return 0.0
    seen = r.get("last_seen") or r.get("since")
    if not seen:
        return 0.0
    seg, step = float(r.get("segments") or 0), float(r.get("step") or 0)
    if seg <= 0 or step <= 0:
        return 0.0
    window = (SURVIVAL_CHARS / CHUNK) * (seg / step)
    if window <= 0:
        return 0.0
    idle = seg - np.asarray(seen[:n], dtype=float)
    born = np.asarray((r.get("born") or [0] * n)[:n], dtype=float)
    young = (step - born) < (SURVIVAL_CHARS / CHUNK)
    dying = (idle / window >= DYING_AT) & ~young
    return 100.0 * float(dying.mean())


def parse(path):
    """Every entry in a sample log, oldest first, with held-out on one level
    - minagi/heldout.py parse_log, which the trainer reads the log with too."""
    return heldout.parse_log(path)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/samples.txt")
    ap.add_argument("--out", default="runs/training_progress.png")
    ap.add_argument("--last", type=float, default=None,
                    help="show only the last N million characters. A "
                         "trailing span rather than a fixed start, because a "
                         "fixed one goes stale: `since 40` was set when the "
                         "run had read 60M and by 176M it was showing almost "
                         "the whole history again")
    ap.add_argument("--logx", action="store_true",
                    help="log the character axis too. A power law is a "
                         "straight line on log-log, so the fitted trend "
                         "becomes a ruler - worth it across decades, useless "
                         "inside one")
    ap.add_argument("--since", type=float, default=None,
                    help="start at this many million characters read, rather "
                         "than at the beginning of the log")
    ap.add_argument("--linear", action="store_true",
                    help="linear loss axis; the default is log, which is what "
                         "makes a slow steady decline visible at all")
    ap.add_argument("--min-rows", type=int, default=100,
                    help="ignore pool-log segments shorter than this; they "
                         "are tool runs, not training")
    ap.add_argument("--pool-log", default="runs/expert_history.jsonl",
                    help="denser than the sample log for anything about the "
                         "pool - it is written every growth decision, not "
                         "every checkpoint")
    a = ap.parse_args()
    rows = parse(a.log)
    if not rows:
        raise SystemExit("nothing parsed")
    if a.last is not None:
        # A trailing window: where it starts is decided by where the run has
        # got to, so it stays the same width for the life of the run.
        a.since = max(0.0, max(r["chars"] for r in rows) - a.last)
    if a.since is not None:
        rows = [r for r in rows if r["chars"] >= a.since]
        if not rows:
            raise SystemExit(f"nothing at or after {a.since}M characters")
    if a.logx:
        # log(0) is not a point on any axis
        rows = [r for r in rows if r["chars"] > 0]
    print(f"  {len(rows)} checkpoints, step {rows[0]['step']:,} -> "
          f"{rows[-1]['step']:,}")

    x = np.array([r["chars"] for r in rows])
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    span = (f"  ·  the last {a.last:.0f}M characters" if a.last is not None
            else f"  ·  from {a.since:.0f}M characters"
            if a.since is not None else "")
    fig.suptitle(
        f"mini-AGI  ·  {rows[-1]['step']:,} steps  ·  {rows[-1]['chars']:.1f}M "
        f"of {rows[-1]['corpus']:,.0f}M characters read "
        f"({100*rows[-1]['chars']/rows[-1]['corpus']:.2f}%)" + span,
        fontsize=13, y=0.985)

    # 1. the score
    p = ax[0][0]
    v = np.array([r.get("val", np.nan) for r in rows], float)
    se = np.array([r.get("se", 0) or 0 for r in rows], float)
    t = np.array([r.get("train", np.nan) for r in rows], float)
    p.plot(x, t, lw=1, color="#c0c0c0", label="train")
    p.fill_between(x, v - se, v + se, color="#2b7bba", alpha=0.18)
    p.plot(x, v, lw=1.3, color="#2b7bba", label="held-out", zorder=4)
    ok = ~np.isnan(v)
    if ok.any():
        i = int(np.nanargmin(v))
        # under the held-out line, so the newest stretch of the curve shows
        p.plot(x[i], v[i], "o", ms=6, color="#d62728", zorder=3)
        p.annotate(f"best {v[i]:.4f}", (x[i], v[i]), textcoords="offset points",
                   xytext=(6, 10), fontsize=9, color="#d62728")
    if not a.linear:
        p.set_yscale("log")
        p.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        p.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    if a.logx:
        # On log-log a power law is a straight line, so the fitted trend
        # becomes a ruler and any departure from it is visible rather than
        # inferred. Worth it across the whole history, which spans decades;
        # pointless inside a trailing window, where the axis covers a factor
        # of one and a bit.
        p.set_xscale("log")
        p.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        p.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # a fitted power law over everything in view
    fit = ok & (x > 0)
    if fit.sum() > 6:
        b_, a_ = np.polyfit(np.log(x[fit]), np.log(v[fit]), 1)
        p.plot(x[fit], np.exp(a_) * x[fit] ** b_, lw=1, ls="--",
               color="#d62728", alpha=.8,
               label=f"trend  loss ~ chars^{b_:.2f}")
    p.set_title("loss, nats per character")
    p.set_xlabel("million characters read"); p.legend(fontsize=8)
    p.grid(alpha=.25, which="both")

    # 2. per domain
    p = ax[0][1]
    doms = sorted({k for r in rows if r.get("dom") for k in r["dom"]})
    cmap = plt.get_cmap("tab10")
    for j, d in enumerate(doms):
        y = np.array([r["dom"].get(d, np.nan) if r.get("dom") else np.nan
                      for r in rows], float)
        if np.isnan(y).all():
            continue
        p.plot(x, y, lw=1.3, color=cmap(j % 10), label=d)
    if not a.linear:
        p.set_yscale("log")
        p.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        p.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    p.set_title("held-out by domain")
    p.set_xlabel("million characters read")
    p.legend(fontsize=7, ncol=2); p.grid(alpha=.25, which="both")

    # 3. context
    p = ax[1][0]
    c = np.array([r.get("context") or np.nan for r in rows], float)
    p.plot(x, c, lw=1.8, color="#7b3294")
    p.set_title(f"context window  (now {rows[-1].get('context') or 0:,} characters)")
    p.set_xlabel("million characters read"); p.grid(alpha=.25)

    # 4. experts - from the pool log, which is written far more often
    p = ax[1][1]
    ph = []
    if os.path.exists(a.pool_log):
        with open(a.pool_log) as f:
            for line in f:
                if line.strip():
                    try:
                        ph.append(json.loads(line))
                    except ValueError:
                        pass
    if ph and a.since is not None:
        ph = [r for r in ph if r["chars"] / 1e6 >= a.since]
    if ph:
        px = np.array([r["chars"] / 1e6 for r in ph])
        pe = np.array([r["experts"] for r in ph], float)
        dead = np.array([_dead_pct(r) for r in ph])
        # The log spans several runs and characters-read restarts with each
        # one, so plotting it as a single line draws a diagonal backwards
        # across the whole figure. Break it wherever the count goes down.
        # A run ends where characters-read goes backwards, and ALSO where the
        # expert count jumps by more than the grower could have moved it - a
        # tool run against a copy appends here too, and its 18-expert pool
        # followed by the real 157 is one continuous line unless it is cut.
        breaks = [i for i in range(1, len(px))
                  if px[i] < px[i - 1] or abs(pe[i] - pe[i - 1]) > 0.3 * pe[i - 1]]
        cut = [0] + breaks + [len(px)]
        segs = [slice(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]
        # Short segments are not training. A tool run against a copy of the
        # weights appends here too, and a handful of rows from a twelve-expert
        # smoke test drawn on the same axes as a 158-expert pool is a diagonal
        # across the whole figure. Real reading leaves hundreds of rows.
        segs = [g for g in segs if (g.stop - g.start) >= a.min_rows]
        q = p.twinx()
        for g in segs:
            p.plot(px[g], pe[g], lw=1.4, color="#2ca25f")
            q.plot(px[g], dead[g], lw=1, color="#bbbbbb")
        q.set_ylabel(f"% of experts unaddressed past {int(100 * DYING_AT)}% "
                     f"of the window", color="#888888", fontsize=8)
        q.set_ylim(0, max(10, float(np.percentile(dead, 99)) * 1.4))
        p.set_title(f"pool: {int(pe[-1])} experts, "
                    f"{dead[-1]:.0f}% dying")
    else:
        e = np.array([r.get("experts") or np.nan for r in rows], float)
        p.plot(x, e, lw=1.8, color="#2ca25f")
        p.set_title(f"experts in the pool (now {rows[-1].get('experts') or 0})")
    p.set_xlabel("million characters read"); p.grid(alpha=.25)

    # 5. reading speed
    p = ax[2][0]
    rt = np.array([r.get("rate") or np.nan for r in rows], float)
    p.plot(x, rt, lw=1, color="#e6842a")
    p.set_title("reading speed, characters per second")
    p.set_xlabel("million characters read"); p.grid(alpha=.25)

    # 6. lr and grad norm
    p = ax[2][1]
    lr = np.array([r.get("lr") or np.nan for r in rows], float)
    p.plot(x, lr, lw=1.6, color="#377eb8", label="learning rate")
    p.set_yscale("log"); p.set_ylabel("lr", color="#377eb8")
    ev = np.array([r.get("evidence") if r.get("evidence") is not None else np.nan
                   for r in rows], float)
    if np.isfinite(ev).any():
        # what the controller can prove, so the rate is readable even when it
        # is not moving: above T_MID (2.2) it eases up, below it down
        pe = p.twinx()
        pe.plot(x, ev, lw=1.0, color="#984ea3", alpha=.75)
        pe.axhline(2.2, color="#984ea3", ls=":", lw=.8)
        pe.set_ylabel("evidence t", color="#984ea3", fontsize=9)
        pe.tick_params(axis="y", labelcolor="#984ea3", labelsize=8)
        pe.spines["right"].set_position(("outward", 38))
    q = p.twinx()
    gn = np.array([r.get("gnorm") or np.nan for r in rows], float)
    q.plot(x, gn, lw=.9, color="#999999", alpha=.8, label="grad norm")
    q.axhline(1.0, color="#d62728", ls=":", lw=1)
    q.set_ylabel("grad norm", color="#666666")
    p.set_title("learning rate and gradient norm (clip at 1)")
    p.set_xlabel("million characters read"); p.grid(alpha=.25)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=125)
    print(f"  wrote {a.out}")

    last = rows[-1]
    print(f"\n  held-out now {last.get('val')}  best {np.nanmin(v):.4f}")
    if last.get("dom"):
        print("  " + "  ".join(f"{k} {v_:.3f}" for k, v_ in sorted(last["dom"].items())))


if __name__ == "__main__":
    main()
