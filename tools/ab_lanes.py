#!/usr/bin/env python3
"""
Orders of reading, compared on the model as it is: subjects read side by side,
a passage from one and then one from another as training reads them, against
one subject at a time for an hour or more before the next.

AN ARM is a copy of weights/ that reads a list of PHASES, each a train.py
session of its own: some subjects at even shares, read the way training reads
them - a passage of one subject, then the subject the schedule picks next - for
so many characters. "chat+code+wikipedia:6.9e6" is an hour of the three side
by side at the run's speed, "code:6.9e6" an hour of code alone, and "*3"
after a phase repeats it. After every phase the arm reads the held-out blocks,
the same blocks for every arm and every phase, so arms are compared where they
have read the same amount, and the phases show what each one did.

The checks are off, as in tools/ab_pool.py: the learning rate stays where the
run left it and the order of reading is all that differs. Each phase draws
its own files and offsets, so two arms read the same subjects in the same
amounts but not the same documents; an arm repeated with another
data.shuffle_seed (--set) shows how much that alone moves held-out.

A SUBJECT is a folder of data/train. train.py names a subject after the first
folder under a path it reads, so each subject is linked file by file - hard
links, no copies - under a folder of its own in the scratch folder, and a
phase reads those folders. Sessions run in a working folder of their own,
which keeps their corpus index (runs/corpus_index.json, where train.py runs)
apart from the run's.

An arm stopped part way carries on from its last finished phase. It needs the
card to itself: it waits while anything else is on the GPU, and never trains
on weights/.

    python3 tools/ab_lanes.py --out runs/ab_lanes/NAME \\
        --arm "mixed=chat+code+wikipedia:6.9e6*3,chat+code+wikipedia:1.73e6" \\
        --arm "blocked=chat:6.9e6,code:6.9e6,wikipedia:6.9e6,chat+code+wikipedia:1.73e6" \\
        --arm "mixed2=chat+code+wikipedia:6.9e6*3,chat+code+wikipedia:1.73e6" \\
        --set "mixed2:data.shuffle_seed=1"
    python3 tools/ab_lanes.py --out runs/ab_lanes/NAME --report
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ab_pool import (say, write_config, wait_for_card, evaluate_in,  # noqa: E402
                     read_losses, speed, parse_arm as parse_sets)


def parse_arm(spec):
    """"name=a+b:chars,c:chars*n" -> (name, [([a, b], chars), ([c], chars),
    ...]): subjects joined by +, and *n repeating a phase n times."""
    name, _, rest = spec.partition("=")
    phases = []
    for p in filter(None, (x.strip() for x in rest.split(","))):
        p, _, times = p.partition("*")
        lanes, _, chars = p.partition(":")
        for _ in range(int(times or 1)):
            phases.append((sorted(lanes.split("+")), float(chars)))
    if not name.strip() or not phases:
        sys.exit(f"  --arm {spec!r}: expected name=subjects:characters,...")
    return name.strip(), phases


def linked(lane, a):
    """The folder a phase reads `lane` from: data/train/<lane> linked file by
    file under a folder of its own, made once and kept for every arm."""
    root = os.path.join(a.scratch, "corpus", lane)
    if not os.path.isdir(os.path.join(root, lane)):
        src = os.path.join(ROOT, a.train, lane)
        if not os.path.isdir(src):
            sys.exit(f"  {src}: no such subject")
        os.makedirs(root, exist_ok=True)
        tmp = os.path.join(root, f".{lane}-linking")    # train.py skips dot-folders
        shutil.rmtree(tmp, ignore_errors=True)
        say(f"linking {os.path.relpath(src, ROOT)} under "
            f"{os.path.relpath(root, ROOT)}")
        subprocess.run(["cp", "-al", src, tmp], check=True)
        os.rename(tmp, os.path.join(root, lane))
    return root


def manifest(wdir):
    with open(os.path.join(wdir, "manifest.json")) as f:
        return json.load(f)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def run_arm(name, phases, sets, a, plan):
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    cfg = write_config(dict({"data.lane_schedule": "equal"}, **sets),
                       os.path.join(out, "config.yaml"))
    wdir = os.path.join(a.scratch, f"ab-{name}")
    prog_path = os.path.join(out, "progress.json")
    prog = load_json(prog_path, {})
    done = 0
    if prog.get("done") and os.path.isdir(wdir):
        if (prog.get("plan") == [[l, c] for l, c in phases]
                and prog.get("sets", {}) == sets
                and manifest(wdir).get("step") == prog["done"][-1]["step"]):
            done = len(prog["done"])
    if done == len(phases):
        say(f"{name}: all {done} phases read already")
        return True
    wait_for_card()
    if done:
        say(f"{name}: carrying on after phase {done} of {len(phases)}")
    else:
        prog = {"plan": [[l, c] for l, c in phases], "sets": sets, "done": []}
        shutil.rmtree(wdir, ignore_errors=True)
        say(f"{name}: copying weights/ to {wdir}")
        t = time.time()
        subprocess.run(["cp", "-a", os.path.join(ROOT, "weights"), wdir],
                       check=True)
        say(f"  copied in {time.time() - t:.0f} s")
        m = manifest(wdir)
        start = {"step": m.get("step"), "read_chars": m.get("read_chars")}
        if plan.setdefault("start", start) != start:
            say(f"  weights/ has moved since the first arm ({plan['start']} -> "
                f"{start}): {name} would not start where the others did - "
                f"not run")
            shutil.rmtree(wdir, ignore_errors=True)
            return False
        save_json(os.path.join(a.out, "plan.json"), plan)
        save_json(prog_path, prog)
    work = os.path.join(a.scratch, "cwd")
    os.makedirs(os.path.join(work, "runs"), exist_ok=True)
    for k in range(done, len(phases)):
        lanes, chars = phases[k]
        roots = [linked(l, a) for l in lanes]
        wait_for_card()
        before = manifest(wdir)
        llog = os.path.join(out, f"loss-{k}.jsonl")
        if os.path.exists(llog):
            os.remove(llog)
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "read", *roots,
               "--save", "--weights-dir", wdir, "--held-out", "",
               "--sample-every", "0", "--save-every", "0", "--no-plots",
               "--history", os.path.join(out, "history.jsonl"),
               "--sample-log", os.path.join(out, "samples.txt"),
               "--loss-log", llog, "--max-chars", str(int(chars)),
               "--minutes", str(a.minutes)]
        say(f"{name}: phase {k + 1} of {len(phases)}, {'+'.join(lanes)} for "
            f"{chars / 1e6:g}M characters")
        t = time.time()
        with open(os.path.join(out, f"train-{k}.log"), "w") as f:
            rc = subprocess.run(cmd, cwd=work,
                                env=dict(os.environ, MINAGI_CONFIG=cfg),
                                stdout=f, stderr=subprocess.STDOUT).returncode
        took = time.time() - t
        after = manifest(wdir)
        read = (after.get("read_chars") or 0) - (before.get("read_chars") or 0)
        say(f"  read {read / 1e6:.2f}M in {took / 60:.0f} min (exit {rc})")
        if rc != 0 or after.get("step") == before.get("step"):
            say(f"  {name} failed - see {os.path.join(out, f'train-{k}.log')}")
            return False
        evaluate_in(wdir, cfg, os.path.join(out, f"eval-{k}.json"), a.blocks)
        prog["done"].append({"lanes": lanes, "chars": chars, "read": read,
                             "step": after.get("step"),
                             "read_chars": after.get("read_chars"),
                             "minutes": round(took / 60, 1)})
        save_json(prog_path, prog)
        report(a)
    if not a.keep:
        shutil.rmtree(wdir, ignore_errors=True)
        say(f"  {name} done; copy deleted")
    return True


# --- what the arms say ---------------------------------------------------------

def subject_means(e):
    """{subject: [mean loss of each block]} of an eval.json."""
    return {s: [sum(b) / len(b) for b in bl if b]
            for s, bl in e["blocks"].items()}


def compare(a_means, b_means, subjects):
    """b minus a over `subjects`, block by block on the same blocks: the
    mean over the subjects of each one's mean difference, and its standard
    error from the blocks."""
    ds, var = [], 0.0
    for s in subjects:
        x, y = a_means.get(s, []), b_means.get(s, [])
        d = [q - p for p, q in zip(x, y)]
        if len(d) < 2:
            continue
        m = sum(d) / len(d)
        ds.append(m)
        var += sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d)
    if not ds:
        return None, None
    return sum(ds) / len(ds), math.sqrt(var) / len(ds)


def mean_over(means, subjects):
    v = [sum(means[s]) / len(means[s]) for s in subjects if means.get(s)]
    return sum(v) / len(v) if v else float("nan")


def lane_losses(path):
    """{lane: (mean training loss on text not yet learned from, steps)} of
    one phase's loss log, and its reading speed."""
    rows = read_losses(path)
    by = {}
    for (lane, *_), loss, _ in rows:
        by.setdefault(lane, []).append(loss)
    return ({l: (sum(v) / len(v), len(v)) for l, v in by.items()},
            speed(rows))


def report(a):
    from minagi.lanes import feeds
    plan = load_json(os.path.join(a.out, "plan.json"), {})
    arms = plan.get("arms", [])
    before = load_json(os.path.join(a.out, "before", "eval.json"))
    if not arms or before is None:
        return
    b0 = subject_means(before)
    subjects = sorted(b0)
    lanes = sorted({x for arm in arms for ls, _ in arm["phases"] for x in ls})
    fed = sorted({s for v in feeds(lanes, subjects).values() for s in v})
    rest = [s for s in subjects if s not in fed]
    lines = [f"held-out, mean of the subjects' means over "
             f"{len(before['blocks'][subjects[0]])} blocks each; read: the "
             f"subjects {', '.join(lanes)} feed ({', '.join(fed)}); "
             f"rest: {', '.join(rest)}",
             f"before: read {mean_over(b0, fed):.4f}   rest "
             f"{mean_over(b0, rest):.4f}   all {mean_over(b0, subjects):.4f}",
             ""]
    summ = {"before": {s: round(mean_over(b0, [s]), 5) for s in subjects},
            "fed": fed, "arms": {}}
    first = None
    for arm in arms:
        name = arm["name"]
        out = os.path.join(a.out, name)
        prog = load_json(os.path.join(out, "progress.json"), {})
        rows, total = [], 0.0
        lines.append(f"{name}: " + ", ".join(
            f"{'+'.join(l)} {c / 1e6:g}M" for l, c in arm["phases"])
            + "".join(f"   {k}={v}" for k, v in arm.get("sets", {}).items()))
        for k, d in enumerate(prog.get("done", [])):
            e = load_json(os.path.join(out, f"eval-{k}.json"))
            if e is None:
                break
            m = subject_means(e)
            total += d["read"]
            ll, sp = lane_losses(os.path.join(out, f"loss-{k}.jsonl"))
            row = {"phase": k + 1, "lanes": d["lanes"],
                   "chars": round(total), "speed": sp,
                   "read": round(mean_over(m, fed), 5),
                   "rest": round(mean_over(m, rest), 5),
                   "subjects": {s: round(mean_over(m, [s]), 5)
                                for s in subjects},
                   "train": {l: round(v, 5) for l, (v, _) in ll.items()}}
            dv, se = compare(b0, m, fed)
            row["read_vs_before"] = [round(dv, 5), round(se, 5)]
            if first is not None and k < len(first):
                dv, se = compare(first[k], m, fed)
                row["read_vs_first"] = [round(dv, 5), round(se, 5)]
                dv, se = compare(first[k], m, rest)
                row["rest_vs_first"] = [round(dv, 5), round(se, 5)]
            rows.append((row, m))
            vs = (f"   vs {arms[0]['name']} {row['read_vs_first'][0]:+.4f} "
                  f"+/-{row['read_vs_first'][1]:.4f} (rest "
                  f"{row['rest_vs_first'][0]:+.4f})"
                  if "read_vs_first" in row else "")
            lines.append(
                f"  {k + 1}. {'+'.join(d['lanes']):<22} {total / 1e6:6.2f}M  "
                f"read {row['read']:.4f} ({row['read_vs_before'][0]:+.4f})  "
                f"rest {row['rest']:.4f}{vs}   "
                f"{sp or 0:,} char/s")
            lines.append("       " + "   ".join(
                f"{s} {row['subjects'][s]:.4f}" for s in subjects))
            lines.append("       training loss, text not yet learned from: "
                         + "   ".join(f"{l} {v:.4f}"
                                      for l, v in row["train"].items()))
        if first is None:
            first = [m for _, m in rows]
        summ["arms"][name] = [r for r, _ in rows]
        lines.append("")
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    save_json(os.path.join(a.out, "summary.json"), summ)
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True,
                    help="folder the arms and the summary go to")
    ap.add_argument("--arm", action="append", default=[],
                    help="name=subjects:characters,... - an arm, in the order "
                         "given; the first is what the others are compared "
                         "with")
    ap.add_argument("--set", action="append", default=[],
                    help="name:key=val,key=val - config.yaml settings of the "
                         "arm of that name, beside its phases")
    ap.add_argument("--minutes", type=float, default=150,
                    help="a cap on each phase's reading, in case it is slow")
    ap.add_argument("--blocks", type=int, default=120,
                    help="held-out blocks of each subject, spread over all "
                         "of it, the same for every arm and phase")
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--scratch", default=os.environ.get(
        "AB_SCRATCH", os.path.join(ROOT, "runs", "ab_lanes", "scratch")),
        help="where the arms' copies of weights/ and the linked subjects go")
    ap.add_argument("--keep", action="store_true",
                    help="keep each arm's copy of weights/ when it is done")
    ap.add_argument("--report", action="store_true",
                    help="only write the summary of what --out holds")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.scratch = os.path.abspath(a.scratch)
    os.makedirs(a.out, exist_ok=True)
    if a.report:
        report(a)
        return
    os.makedirs(a.scratch, exist_ok=True)
    plan = load_json(os.path.join(a.out, "plan.json"), {})
    plan.setdefault("arms", [])
    plan.update(blocks=a.blocks, train=a.train)
    arms = [parse_arm(s) for s in a.arm]
    sets = dict(parse_sets(s) for s in a.set)
    unknown = sorted(set(sets) - {n for n, _ in arms})
    if unknown:
        sys.exit(f"  --set names {', '.join(unknown)}, which no --arm is")
    for name, phases in arms:
        plan["arms"] = [x for x in plan["arms"] if x["name"] != name] + [
            {"name": name, "phases": phases, "sets": sets.get(name, {})}]
    save_json(os.path.join(a.out, "plan.json"), plan)
    base = os.path.join(a.out, "before")
    if not os.path.exists(os.path.join(base, "eval.json")):
        os.makedirs(base, exist_ok=True)
        wait_for_card()
        say("weights/ as it is: held-out")
        evaluate_in(os.path.join(ROOT, "weights"),
                    write_config({}, os.path.join(base, "config.yaml")),
                    os.path.join(base, "eval.json"), a.blocks)
    for name, phases in arms:
        if run_arm(name, phases, sets.get(name, {}), a, plan):
            report(a)
    say("all done")


if __name__ == "__main__":
    main()
