#!/usr/bin/env python3
"""
What train.py reads of each subject follows the shares in characters.

  two subjects at equal fixed shares, one of long files and one of short
  ones: a visit to the first reads a passage of one file, a visit to the
  second a passage of its files joined. train.py reads about as many
  characters of each, and saves the lane state as characters, which a
  resume carries on from
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


tmp = tempfile.mkdtemp(prefix="minagi-lane-chars-")
try:
    from minagi.create import create
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    corpus = os.path.join(tmp, "train")
    os.makedirs(os.path.join(corpus, "long"))
    os.makedirs(os.path.join(corpus, "short"))
    with open(os.path.join(corpus, "long", "a.txt"), "w") as f:
        f.write("the cat sat on the mat and looked out of the window. " * 1000)
    for i in range(40):
        with open(os.path.join(corpus, "short", f"{i}.txt"), "w") as f:
            f.write(f"def f{i}(x):\n    return x * {i} + 1\n" * 6)
    passage = 2048
    short_len = os.path.getsize(os.path.join(corpus, "short", "0.txt"))

    def run(max_chars):
        r = subprocess.run(
            [sys.executable, "train.py", "--device", "cpu", "read", corpus,
             "--save", "--weights-dir", wdir, "--held-out", "",
             "--sample-every", "0", "--save-every", "0", "--no-plots",
             "--history", os.path.join(tmp, "h.jsonl"),
             "--sample-log", os.path.join(tmp, "s.txt"),
             "--loss-log", os.path.join(tmp, "loss.jsonl"),
             "--max-chars", str(max_chars), "--chunk", "64",
             "--passage", str(passage), "--lane-weights", "long=1,short=1"],
            cwd=ROOT, capture_output=True, text=True, timeout=1200)
        man = json.load(open(os.path.join(wdir, "manifest.json")))
        return r, man

    r, man = run(30000)
    lanes = man.get("lanes") or {}
    read = lanes.get("read", {})
    tot = sum(read.values()) or 1.0
    check("train.py reads", r.returncode == 0 and tot > 20000,
          f"exit {r.returncode}, {tot:,.0f} characters"
          + ("" if r.returncode == 0 else "\n" + r.stdout[-1200:]
             + r.stderr[-1200:]))
    share = read.get("long", 0.0) / tot
    check("at equal shares it reads about as many characters of a subject of "
          "long files as of one of files a tenth as long",
          0.4 < share < 0.6,
          f"long {read.get('long', 0):,.0f}, short {read.get('short', 0):,.0f}"
          f" (short files of {short_len} characters)")
    check("...and saves its lane state as characters",
          lanes.get("unit") == "characters", str(lanes.get("unit")))
    visits = []
    with open(os.path.join(tmp, "loss.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["j"] == 0:
                visits.append([d["lane"], 0, len(d.get("files", []))])
            if visits:
                visits[-1][1] += 1
    short_visits = [v for v in visits[:-1] if v[0] == "short"]
    check("a visit to the subject of short files reads a whole passage of "
          "them, joined",
          short_visits and all(n == passage // 64 and k > 1
                               for _, n, k in short_visits),
          f"{len(short_visits)} visits, steps {sorted({n for _, n, _ in short_visits})}, "
          f"files {sorted({k for _, _, k in short_visits})}")

    before = dict(read)
    r2, man2 = run(60000)
    read2 = (man2.get("lanes") or {}).get("read", {})
    more = {k: read2.get(k, 0.0) - before.get(k, 0.0) for k in read2}
    tot2 = sum(more.values()) or 1.0
    check("a resume carries on in characters",
          r2.returncode == 0 and 0.4 < more.get("long", 0.0) / tot2 < 0.6,
          f"long {more.get('long', 0):,.0f}, short {more.get('short', 0):,.0f}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  " + ("all lane-character tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
