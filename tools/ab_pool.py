#!/usr/bin/env python3
"""
Settings of the expert pool, compared on the model as it is: every arm a copy
of weights/ that reads the same characters of data/train under its own
settings, and then the same held-out blocks.

WHAT AN ARM IS. A copy of weights/ in a scratch folder, and config.yaml with
the arm's settings changed, in a file of its own that MINAGI_CONFIG names
(minagi/config.py). The arm reads `--chars` characters through train.py as
the run reads them - the same subjects in the same order, the same files at
the same places - with the held-out checks off, so the learning rate and the
subjects' shares stay where the run left them and every arm reads the same
text at the same rate. train.py --loss-log writes every step's loss and
where it was read. Then the arm reads the held-out blocks, the same ones for
every arm, and its copy is deleted.

WHAT IS COMPARED, arm against the first arm:
  training loss   chunk by chunk, on the same chunks: the model's own loss,
                  as held-out reads it, each scored before the step that
                  learns from it - text the arm has not seen. The mean
                  difference over the arm, over its last quarter, and its
                  trend per ten million characters
  held-out        block by block, on the same blocks
  experts in use  on held-out: the experts chosen at least once, the share
                  of choices the 31 most chosen take, how many experts carry
                  90% and 99% of the choices, and exp(entropy) of the shares
  reading speed   characters a second while the arm reads

A second arm with the first one's settings shows how far two runs of the same
settings land apart: the noise every other difference is read against.

A PROBE changes a setting and reads held-out with nothing trained - on
weights/ itself, read only - which says what the setting does to the model
as it stands before learning has had a chance to adjust to it.

It needs the card to itself: it waits while anything else is on the GPU, and
never trains on weights/.

    python3 tools/ab_pool.py --out runs/ab_pool/NAME --probe pool.gate_after=true
    python3 tools/ab_pool.py --out runs/ab_pool/NAME --chars 14e6 \\
        --arm control --arm bias:pool.route_bias=3e-4 --arm control2
    python3 tools/ab_pool.py --out runs/ab_pool/NAME --report
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ORIGINALS = 31          # experts the pool started with: the cliff in its use


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- settings ------------------------------------------------------------------

def value(s):
    """A setting's value as YAML would read it, with numbers like 3e-4 - which
    YAML reads as text - read as numbers."""
    import yaml
    v = yaml.safe_load(s)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def parse_arm(spec):
    """"name:key=val,key=val" -> (name, {key: val})."""
    name, _, rest = spec.partition(":")
    sets = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        sets[k.strip()] = value(v.strip())
    return name.strip(), sets


def write_config(sets, path):
    """config.yaml with `sets` ({dotted.key: value}) changed, at `path`."""
    import yaml
    with open(os.path.join(ROOT, "config.yaml")) as f:
        c = yaml.safe_load(f) or {}
    for k, v in sets.items():
        d = c
        *head, last = k.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = v
    with open(path, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    return path


# --- the card ------------------------------------------------------------------

def gpu_users():
    """Processes on the GPU, as nvidia-smi lists them."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def wait_for_card(poll=60):
    """Return once nothing else is on the GPU and no training run is going."""
    from tools.measure_heldout import training_run
    told = False
    while True:
        busy = gpu_users()
        run = training_run(os.path.join(ROOT, "weights"))
        if not busy and run is None:
            return
        if not told:
            say(f"  the card is in use ({'; '.join(busy) or f'train.py pid {run}'})"
                f" - waiting")
            told = True
        time.sleep(poll)


def corpus_print(folder):
    """What would change the text an arm reads: every file's name and size."""
    got = []
    for dp, _, fs in sorted(os.walk(folder)):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            try:
                got.append((os.path.relpath(p, folder), os.path.getsize(p)))
            except OSError:
                pass
    return hash(tuple(got)), len(got)


# --- one arm -------------------------------------------------------------------

def run_arm(name, sets, a, state):
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    cfg = write_config(sets, os.path.join(out, "config.yaml"))
    wdir = os.path.join(a.scratch, f"ab-{name}")
    if os.path.exists(wdir):
        shutil.rmtree(wdir)
    wait_for_card()
    fp = corpus_print(os.path.join(ROOT, a.train))
    if state.setdefault("corpus", list(fp)) != list(fp):
        say(f"  WARNING: {a.train} changed since the first arm - the arms "
            f"will not read the same text past the change")
    say(f"{name}: copying weights/ to {wdir}")
    t = time.time()
    subprocess.run(["cp", "-a", os.path.join(ROOT, "weights"), wdir],
                   check=True)
    say(f"  copied in {time.time() - t:.0f} s")
    with open(os.path.join(wdir, "manifest.json")) as f:
        man = json.load(f)
    start = {"step": man.get("step"), "read_chars": man.get("read_chars")}
    first = state.setdefault("start", start)
    if start != first:
        say(f"  weights/ has moved since the first arm ({first} -> {start}): "
            f"{name} would not start where the others did - not run")
        shutil.rmtree(wdir, ignore_errors=True)
        return False
    with open(os.path.join(out, "before.json"), "w") as f:
        json.dump(dict(start, telemetry=man.get("telemetry")), f)
    env = dict(os.environ, MINAGI_CONFIG=cfg)
    llog = os.path.join(out, "loss.jsonl")
    if os.path.exists(llog):
        os.remove(llog)
    cmd = [sys.executable, "train.py", "read", a.train, "--save",
           "--weights-dir", wdir, "--held-out", "", "--sample-every", "0",
           "--save-every", "0", "--no-plots",
           "--history", os.path.join(out, "history.jsonl"),
           "--sample-log", os.path.join(out, "samples.txt"),
           "--loss-log", llog, "--max-chars", str(int(a.chars)),
           "--minutes", str(a.minutes)]
    say(f"{name}: reading {a.chars / 1e6:g}M characters "
        f"{json.dumps(sets) if sets else '(as the run is)'}")
    t = time.time()
    with open(os.path.join(out, "train.log"), "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    took = time.time() - t
    say(f"  read in {took / 60:.0f} min (exit {rc})")
    if rc != 0:
        say(f"  {name} failed - see {os.path.join(out, 'train.log')}")
        shutil.rmtree(wdir, ignore_errors=True)
        return False
    evaluate_in(wdir, cfg, os.path.join(out, "eval.json"), a.blocks)
    with open(os.path.join(wdir, "manifest.json")) as f:
        man = json.load(f)
    with open(os.path.join(out, "after.json"), "w") as f:
        json.dump({"telemetry": man.get("telemetry"),
                   "n_experts": man.get("n_experts"),
                   "read_chars": man.get("read_chars"),
                   "step": man.get("step")}, f)
    shutil.rmtree(wdir, ignore_errors=True)
    say(f"  {name} done; copy deleted")
    return True


def evaluate_in(wdir, cfg, out, blocks):
    """Held-out on `wdir` under the settings in `cfg`, in a process of its
    own, so the card is free again when it returns."""
    env = dict(os.environ, MINAGI_CONFIG=cfg)
    t = time.time()
    subprocess.run([sys.executable, os.path.abspath(__file__), "--eval", wdir,
                    "--eval-out", out, "--blocks", str(blocks)],
                   cwd=ROOT, env=env, check=True)
    say(f"  held-out read in {time.time() - t:.0f} s -> "
        f"{os.path.relpath(out, ROOT)}")


def evaluate(wdir, out, blocks):
    """The held-out blocks, each block's chunk losses, and which experts were
    chosen how often while they were read."""
    import numpy as np
    import torch
    import train as T
    from minagi.config import load, get
    from minagi.precision import set_compute_dtype
    from minagi.stream import FolderEvaluator
    c = load()
    held = os.path.join(ROOT, str(get(c, "data.held_out", "data/val")))
    chunk = int(get(c, "training.chunk", 2048))
    set_compute_dtype(str(get(c, "training.precision", "bf16")))
    device = torch.device("cuda")
    model, cfg, pool, man = T.build_paged(
        wdir, device, ram_capacity=int(get(c, "pool.ram_cache", 96)),
        read_only=True)
    model.eval()
    ev = FolderEvaluator(model, held, chunk, cfg.block, device)
    n = pool.n_experts()
    use0 = pool.use[:n].detach().double().cpu().clone()
    t = time.time()
    got = {}
    with torch.no_grad():
        for name in ev.groups:
            got[name] = [ev.score_block(d, name) for d in ev._texts(name,
                                                                   blocks)]
    used = (pool.use[:n].detach().double().cpu() - use0).numpy()
    rec = {"weights": wdir, "config": os.environ.get("MINAGI_CONFIG"),
           "gate_after": getattr(pool, "gate_after", False),
           "route_bias": getattr(pool, "route_bias", 0.0),
           "route_floor": getattr(pool, "route_floor", 0.0),
           "step": man.get("step"), "chars": man.get("read_chars"),
           "blocks": got, "seconds": round(time.time() - t, 1),
           "uid": [int(v) for v in pool.uid[:n]],
           "gate": [round(float(v), 6) for v in pool.gate.data[:n]],
           "born": [float(v) for v in pool.born[:n]],
           "used": [float(v) for v in used]}
    if getattr(pool, "route_bias", 0.0):
        rec["bias"] = [round(float(v), 5) for v in pool.bias[:n]]
    with open(out, "w") as f:
        json.dump(rec, f)
    every = [x for bl in got.values() for b in bl for x in b]
    print(f"  held-out {float(np.mean(every)):.4f} over "
          f"{sum(len(b) for b in got.values())} blocks, "
          f"{int((used > 0).sum())} of {n} experts chosen", flush=True)


# --- what the arms say ---------------------------------------------------------

def usage(used):
    """The shape of a pool's use from its choice counts."""
    import numpy as np
    u = np.sort(np.asarray(used, float))[::-1]
    tot = u.sum()
    if tot <= 0:
        return {}
    s = u / tot
    cum = np.cumsum(s)
    nz = s[s > 0]
    return {"chosen": int((u > 0).sum()),
            "top31": round(float(cum[min(ORIGINALS, len(cum)) - 1]), 4),
            "n90": int(np.searchsorted(cum, 0.90) + 1),
            "n99": int(np.searchsorted(cum, 0.99) + 1),
            "effective": round(float(np.exp(-(nz * np.log(nz)).sum())), 1)}


def trained_use(out):
    """Each expert's choices while the arm read, by expert: its use after
    the arm less its use before."""
    try:
        with open(os.path.join(out, "before.json")) as f:
            b = json.load(f)["telemetry"]
        with open(os.path.join(out, "after.json")) as f:
            a = json.load(f)["telemetry"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    was = dict(zip(b["uid"], b["use"]))
    return [max(0.0, u - was.get(i, 0.0)) for i, u in zip(a["uid"], a["use"])]


def gates(e, born_before=None):
    """The gates at the end of an arm: how many are above 0.1, and the
    median of those of the experts grown after the first 31."""
    import numpy as np
    g = np.abs(np.asarray(e["gate"], float))
    order = np.argsort(np.asarray(e["born"], float), kind="stable")
    grown = g[order[ORIGINALS:]]
    return {"above_0.1": int((g > 0.1).sum()),
            "grown_median": round(float(np.median(grown)), 4)
            if len(grown) else None}


def read_losses(path, key="model"):
    """(where, loss, time) of every step in a loss log."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            rows.append(((d["lane"], d["file"], d["pos"], d["j"]),
                         d.get(key) if d.get(key) is not None else d["loss"],
                         d.get("t")))
    return rows


def paired(a_rows, b_rows, bin_steps=256, chunk=2048):
    """b minus a, step by step on the steps both read alike: the mean, the
    mean over the last quarter and the trend per ten million characters, with
    standard errors from bins of `bin_steps` (a step's neighbours read the
    same passage, so single steps are not independent)."""
    import numpy as np
    n = 0
    for (wa, _, _), (wb, _, _) in zip(a_rows, b_rows):
        if wa != wb:
            break
        n += 1
    if n < 2 * bin_steps:
        return {"aligned": n}
    d = np.array([b_rows[i][1] - a_rows[i][1] for i in range(n)])
    nb = n // bin_steps
    bins = d[:nb * bin_steps].reshape(nb, bin_steps).mean(1)
    x = (np.arange(nb) + 0.5) * bin_steps * chunk / 1e7    # tens of millions
    q = max(1, nb // 4)
    out = {"aligned": n, "of": min(len(a_rows), len(b_rows)),
           "mean": float(bins.mean()),
           "se": float(bins.std(ddof=1) / math.sqrt(nb)) if nb > 1 else None,
           "last_quarter": float(bins[-q:].mean()),
           "last_quarter_se": (float(bins[-q:].std(ddof=1) / math.sqrt(q))
                               if q > 1 else None),
           "first_bin": float(bins[0]),
           "bins": [round(float(v), 5) for v in bins]}
    if nb >= 3:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, bins, rcond=None)
        resid = bins - A @ coef
        s2 = float(resid @ resid) / max(nb - 2, 1)
        sxx = float(((x - x.mean()) ** 2).sum())
        out["trend"] = float(coef[0])                       # per 10M chars
        out["trend_se"] = math.sqrt(s2 / sxx) if sxx > 0 else None
    return out


def speed(rows, chunk=2048):
    ts = [t for _, _, t in rows if t is not None]
    if len(ts) < 100:
        return None
    # past the first hundred steps: the first ones fill the card and RAM
    return round((len(ts) - 101) * chunk / max(ts[-1] - ts[100], 1e-9))


def held_out_paired(a, b):
    """b minus a on held-out, block by block over the blocks both read."""
    import numpy as np
    diffs, per = [], {}
    for name, blocks in a["blocks"].items():
        other = b["blocks"].get(name, [])
        ds = [float(np.mean(y)) - float(np.mean(x))
              for x, y in zip(blocks, other) if x and y and len(x) == len(y)]
        if ds:
            per[name] = round(float(np.mean(ds)), 5)
            diffs.extend(ds)
    d = np.asarray(diffs)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1)
                                                 / math.sqrt(len(d))),
            "blocks": len(d), "per_subject": per}


def held_out_mean(e):
    import numpy as np
    return float(np.mean([x for bl in e["blocks"].values() for b in bl
                          for x in b]))


# the reference palette's categorical slots, in order, and its inks
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


NEAR = 0.02    # nats: an arm further off than this gets a row without it


def plot(summ, path, bin_steps=256, chunk=2048):
    """Every arm's training loss against the first arm's, bin by bin, and
    its held-out difference beside it - control arms in grey. When an arm is
    so far off that the rest lie flat on the zero line, a second row draws
    the rest at their own scale."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return None
    arms = [(n, r) for n, r in summ["arms"].items() if r.get("train")]
    if not arms:
        return None

    def size(r):
        h = r.get("held_out") or {}
        return max([abs(v) for v in r["train"].get("bins", [])]
                   + [abs(h.get("mean", 0)) + h.get("se", 0)])
    near = [(n, r) for n, r in arms if size(r) <= NEAR]
    far = [n for n, r in arms if size(r) > NEAR]
    rows = [arms] + ([near] if near and far else [])
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.8 * len(rows)),
                             gridspec_kw={"width_ratios": [2.2, 1]},
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    colour, k = {}, 0
    for n, r in arms:
        if r["sets"]:
            colour[n] = SERIES[k % len(SERIES)]
            k += 1
        else:
            colour[n] = MUTED
    ends = []
    for row, (ax, bx) in zip(rows, axes):
        for n, r in row:
            b = r["train"].get("bins", [])
            x = [(i + 0.5) * bin_steps * chunk / 1e6 for i in range(len(b))]
            ax.plot(x, b, color=colour[n], lw=2,
                    ls="--" if not r["sets"] else "-", label=n)
            if b:
                ends.append((ax, n, x[-1], b[-1]))
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("characters read in the arm (millions)", color=INK2)
        ax.set_ylabel("training loss minus the first arm's (nats)",
                      color=INK2)
        names = [n for n, r in row if r.get("held_out")]
        ys = range(len(names))
        for y, n in zip(ys, names):
            h = summ["arms"][n]["held_out"]
            bx.errorbar(h["mean"], y, xerr=h["se"], fmt="o", ms=7,
                        color=colour[n], ecolor=colour[n], elinewidth=2,
                        capsize=0)
        bx.axvline(0, color=MUTED, lw=1)
        bx.set_yticks(list(ys), names)
        bx.invert_yaxis()
        bx.set_xlabel("held-out minus the first arm's (nats)", color=INK2)
        bx.xaxis.set_major_locator(MaxNLocator(4))
        for a_ in (ax, bx):
            a_.set_facecolor(SURFACE)
            a_.grid(True, color="#e6e5e1", lw=0.8)
            a_.tick_params(colors=INK2)
            for sp in a_.spines.values():
                sp.set_color("#d5d3cd")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
    titles = [("on the same chunks, each scored before it is learned",
               "held-out, the same blocks")]
    if len(rows) > 1:
        titles.append((f"the same without {', '.join(far)}, at the scale "
                       f"of the rest", f"held-out without {', '.join(far)}"))
    for (ax, bx), (t1, t2) in zip(axes, titles):
        ax.set_title(t1, fontsize=10, color=INK, loc="left")
        bx.set_title(t2, fontsize=10, color=INK, loc="left")
    fig.suptitle("Expert-pool A/B: negative is better", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    # each line named at its end, the names pushed apart where lines meet
    for ax in {e[0] for e in ends}:
        px = [(n, x, y, ax.transData.transform((x, y))[1])
              for a_, n, x, y in ends if a_ is ax]
        px.sort(key=lambda t: t[3])
        gap, last, placed = 13.0, None, []
        for n, x, y, py in px:
            at = py if last is None else max(py, last + gap)
            placed.append((n, x, y, at - py))
            last = at
        for n, x, y, dy in placed:
            ax.annotate(n, (x, y), xytext=(6, dy * 72.0 / fig.dpi),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK2, annotation_clip=False)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def report(a):
    """Every arm against the first, and the probes against weights/ as it
    is, into summary.json and summary.txt in --out."""
    import numpy as np
    plan = load_plan(a.out)
    base_ev = os.path.join(a.out, "before", "eval.json")
    before = json.load(open(base_ev)) if os.path.exists(base_ev) else None
    lines, summ = [], {"plan": plan, "arms": {}, "probes": {}}
    if before:
        lines.append(f"weights/ as it is: held-out {held_out_mean(before):.4f}"
                     f" over {sum(len(b) for b in before['blocks'].values())} "
                     f"blocks; experts {usage(before['used'])}")
        for p in plan.get("probes", []):
            f = os.path.join(a.out, f"probe-{p['name']}", "eval.json")
            if not os.path.exists(f):
                continue
            e = json.load(open(f))
            h = held_out_paired(before, e)
            summ["probes"][p["name"]] = {"sets": p["sets"], "held_out": h,
                                         "use": usage(e["used"])}
            lines.append(f"probe {p['name']} {json.dumps(p['sets'])}: "
                         f"held-out {h['mean']:+.4f} +/-{h['se']:.4f}, "
                         f"experts {usage(e['used'])}")
    arms = [x for x in plan.get("arms", [])
            if os.path.exists(os.path.join(a.out, x["name"], "eval.json"))]
    if arms:
        first = arms[0]["name"]
        A = read_losses(os.path.join(a.out, first, "loss.jsonl"))
        Ae = json.load(open(os.path.join(a.out, first, "eval.json")))
        lines.append("")
        lines.append(f"against {first}, on the same chunks and the same "
                     f"held-out blocks (nats; negative is better):")
        hdr = (f"{'arm':<14}{'train all':>18}{'last quarter':>18}"
               f"{'trend /10M':>18}{'held-out':>18}{'chosen':>8}"
               f"{'top31':>7}{'n90':>5}{'n99':>5}{'eff':>7}{'char/s':>8}")
        lines.append(hdr)
        for x in arms:
            nm = x["name"]
            B = read_losses(os.path.join(a.out, nm, "loss.jsonl"))
            Be = json.load(open(os.path.join(a.out, nm, "eval.json")))
            u = usage(Be["used"])
            tu = trained_use(os.path.join(a.out, nm))
            rec = {"sets": x["sets"], "use": u, "speed": speed(B),
                   "held_out_value": held_out_mean(Be), "steps": len(B),
                   "trained_use": usage(tu) if tu else None,
                   "gates": gates(Be)}
            if nm != first:
                rec["train"] = paired(A, B)
                rec["held_out"] = held_out_paired(Ae, Be)
            summ["arms"][nm] = rec

            def pm(v, se):
                return (f"{v:+.4f}+/-{se:.4f}" if v is not None and se
                        else (f"{v:+.4f}" if v is not None else "-"))
            tr, ho = rec.get("train", {}), rec.get("held_out", {})
            lines.append(
                f"{nm:<14}{pm(tr.get('mean'), tr.get('se')):>18}"
                f"{pm(tr.get('last_quarter'), tr.get('last_quarter_se')):>18}"
                f"{pm(tr.get('trend'), tr.get('trend_se')):>18}"
                f"{pm(ho.get('mean'), ho.get('se')):>18}"
                f"{u.get('chosen', 0):>8}{u.get('top31', 0):>7.2f}"
                f"{u.get('n90', 0):>5}{u.get('n99', 0):>5}"
                f"{u.get('effective', 0):>7.1f}{rec['speed'] or 0:>8}")
            t = rec["trained_use"] or {}
            lines.append(f"{'':<14}while reading: {t.get('chosen', '-')} "
                         f"experts chosen, top31 {t.get('top31', 0):.2f}, "
                         f"n90 {t.get('n90', '-')}, n99 {t.get('n99', '-')}, "
                         f"eff {t.get('effective', 0):.1f}; gates above 0.1: "
                         f"{rec['gates']['above_0.1']}, grown median "
                         f"{rec['gates']['grown_median']}; held-out "
                         f"{rec['held_out_value']:.4f}")
            if tr and tr.get("aligned") is not None and tr.get("of") \
                    and tr["aligned"] < tr["of"]:
                lines.append(f"{'':<14}aligned for {tr['aligned']:,} of "
                             f"{tr['of']:,} steps only")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    if plot(summ, os.path.join(a.out, "summary.png")):
        lines.append(f"chart: {os.path.join(a.out, 'summary.png')}")
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --- the plan ------------------------------------------------------------------

def load_plan(out):
    p = os.path.join(out, "plan.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_plan(out, plan):
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="folder the arms and the summary go to")
    ap.add_argument("--arm", action="append", default=[],
                    help="name:key=val,key=val - an arm, in the order given; "
                         "the first is what the others are compared with")
    ap.add_argument("--probe", action="append", default=[],
                    help="[name:]key=val,... - held-out on weights/ with a "
                         "setting changed and nothing trained")
    ap.add_argument("--chars", type=float, default=14e6,
                    help="characters each arm reads")
    ap.add_argument("--minutes", type=float, default=200,
                    help="a cap on each arm's reading, in case it is slow")
    ap.add_argument("--blocks", type=int, default=120,
                    help="held-out blocks of each subject, spread over all "
                         "of it, the same for every arm")
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--scratch", default=os.environ.get(
        "AB_SCRATCH", os.path.join(ROOT, "runs", "ab_pool", "scratch")),
        help="where each arm's copy of weights/ is made and deleted")
    ap.add_argument("--report", action="store_true",
                    help="only write the summary of what --out holds")
    ap.add_argument("--eval", help=argparse.SUPPRESS)
    ap.add_argument("--eval-out", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.eval:
        evaluate(a.eval, a.eval_out, a.blocks)
        return
    if not a.out:
        sys.exit("  --out is needed")
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a)
        return
    os.makedirs(a.scratch, exist_ok=True)
    plan = load_plan(a.out)
    plan.setdefault("arms", [])
    plan.setdefault("probes", [])
    plan.update(chars=a.chars, blocks=a.blocks, train=a.train)
    state = {}

    # weights/ as it is, once - every probe is read against it
    if a.probe or a.arm:
        base = os.path.join(a.out, "before")
        if not os.path.exists(os.path.join(base, "eval.json")):
            os.makedirs(base, exist_ok=True)
            wait_for_card()
            say("weights/ as it is: held-out")
            evaluate_in(os.path.join(ROOT, "weights"),
                        write_config({}, os.path.join(base, "config.yaml")),
                        os.path.join(base, "eval.json"), a.blocks)
    for i, spec in enumerate(a.probe):
        name, sets = parse_arm(spec if ":" in spec else f"p{i}:{spec}")
        d = os.path.join(a.out, f"probe-{name}")
        os.makedirs(d, exist_ok=True)
        plan["probes"] = [p for p in plan["probes"] if p["name"] != name] + [
            {"name": name, "sets": sets}]
        save_plan(a.out, plan)
        wait_for_card()
        say(f"probe {name}: {json.dumps(sets)}")
        evaluate_in(os.path.join(ROOT, "weights"),
                    write_config(sets, os.path.join(d, "config.yaml")),
                    os.path.join(d, "eval.json"), a.blocks)
        report(a)
    for spec in a.arm:
        name, sets = parse_arm(spec)
        plan["arms"] = [x for x in plan["arms"] if x["name"] != name] + [
            {"name": name, "sets": sets}]
        save_plan(a.out, plan)
        if run_arm(name, sets, a, state):
            report(a)
    say("all done")


if __name__ == "__main__":
    main()
