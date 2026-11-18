#!/usr/bin/env python3
"""
A read without --save leaves the weights directory exactly as it was.

  the pool writes an expert it trained back to disk whenever RAM makes room
  for another, and a dry read did that straight into the weights it promised
  not to touch. Read without --save, with RAM too small to hold what the read
  trains, every file of the weights is byte for byte what it was and no
  scratch directory is left beside them; one left by a dry read that never
  finished is cleared away, one still in use is not; and the same read with
  --save does change the weights
"""
import hashlib
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


def digest(d):
    out = {}
    for root, _, names in os.walk(d):
        for n in names:
            p = os.path.join(root, n)
            with open(p, "rb") as f:
                out[os.path.relpath(p, d)] = hashlib.sha256(f.read()).hexdigest()
    return out


tmp = tempfile.mkdtemp(prefix="minagi-dry-read-")
try:
    from minagi.create import create
    wdir = os.path.join(tmp, "w")
    create(wdir, seed=0, verbose=False, experts=12, resident=4, d_ff=32,
           d_model=32, trunk_d_ff=32, n_head=2, block=128, max_steps=3,
           top_k=2, depth=1)
    corpus = os.path.join(tmp, "train")
    os.makedirs(os.path.join(corpus, "prose"))
    os.makedirs(os.path.join(corpus, "code"))
    with open(os.path.join(corpus, "prose", "a.txt"), "w") as f:
        f.write("the cat sat on the mat and looked out of the window. " * 400)
    with open(os.path.join(corpus, "code", "b.txt"), "w") as f:
        f.write("".join(f"def f{i}(x):\n    return x * {i} + 1\n"
                        for i in range(600)))

    def run(*extra):
        return subprocess.run(
            [sys.executable, "train.py", "--device", "cpu", "read", corpus,
             "--weights-dir", wdir, "--held-out", "",
             "--sample-every", "0", "--save-every", "0", "--no-plots",
             "--history", os.path.join(tmp, "h.jsonl"),
             "--sample-log", os.path.join(tmp, "s.txt"),
             "--max-chars", "12000", "--chunk", "64", "--passage", "1024",
             "--lane-weights", "prose=1,code=1",
             "--resident", "4", "--ram-capacity", "1", *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=1200)

    # A scratch directory whose process is gone, and one whose process is
    # alive (this test's own), named the way a dry read names them.
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = os.path.join(tmp, f".w.dry-read-{dead.pid}-x")
    live = os.path.join(tmp, f".w.dry-read-{os.getpid()}-x")
    for d in (stale, live):
        os.makedirs(d)
        with open(os.path.join(d, "core.npz"), "w") as f:
            f.write("x")

    before = digest(wdir)
    r = run()
    after = digest(wdir)
    tail = "" if r.returncode == 0 else "\n" + r.stdout[-1500:] + r.stderr[-1500:]
    check("a dry read runs", r.returncode == 0, f"exit {r.returncode}{tail}")
    changed = sorted(k for k in set(before) | set(after)
                     if before.get(k) != after.get(k))
    check("...and leaves every file of the weights as it was", not changed,
          f"{len(changed)} of {len(before)} files changed: {changed[:4]}")
    left = sorted(n for n in os.listdir(tmp)
                  if n.startswith(".w.dry-read-") and n != os.path.basename(live))
    check("...leaves no scratch directory of its own beside them", not left,
          str(left))
    check("...clears away one left by a dry read that never finished",
          not os.path.exists(stale))
    check("...and leaves one whose read is still running", os.path.exists(live))

    r2 = run("--save")
    saved = digest(wdir)
    moved = sum(1 for k in before
                if k.startswith("experts") and saved.get(k) != before[k])
    check("the same read with --save changes the experts it trained",
          r2.returncode == 0 and moved > 0,
          f"exit {r2.returncode}, {moved} expert files changed")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  " + ("all dry-read tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
