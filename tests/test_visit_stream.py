#!/usr/bin/env python3
"""
A visit reads one subject as one stream (train.py's _Lane.open).

  a subject of short files: the visit is a whole passage of them, each
  joined to the one before by one <|endoftext|> - none added after a file
  that already ends in one - and the reader's `files` say where each file
  starts, so the text can be put back together exactly
  the files are drawn one after another from the lane's generator, so a
  session's reading can be replayed from its seed
  a subject of long files: the visit is one passage of one file, from an
  offset drawn after the file - what the reader always drew
  a subject whose files have all gone: no visit
"""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                   # noqa: E402
from minagi.ingest import as_stream                 # noqa: E402
from minagi.tokenizer import SPECIAL_ID             # noqa: E402

EOT = SPECIAL_ID["<|endoftext|>"]
SPACE = (9, 10, 13, 32)
ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


tmp = tempfile.mkdtemp(prefix="minagi-visit-stream-")
try:
    span = 2048
    # every third file closes itself, as a packed file of stories does
    short = [write(os.path.join(tmp, "short", f"{i:02d}.txt"),
                   f"story {i}: the cat sat on mat number {i}.\n" * 3
                   + ("<|endoftext|>\n" if i % 3 == 0 else ""))
             for i in range(30)]
    lane = T._Lane("short", short, np.random.default_rng([0, 0, 0]))
    r = lane.open(None, 64, 256, "cpu", span)
    data = r.data
    check("a visit to a subject of short files reads a whole passage",
          len(data) == span + 1, f"{len(data)} ids for a passage of {span}")
    check("...of several of them", len(r.files) > 5, f"{len(r.files)} files")

    placed = joins = True
    end = 0
    for k, (path, at) in enumerate(r.files):
        ids = as_stream(path)
        if k:
            prev = as_stream(r.files[k - 1][0])
            want = end if T._ends_in_eot(prev) else end + 1
            joins = joins and at == want and (
                at == end or int(data[end]) == EOT)
        got = data[at:at + len(ids)]
        placed = placed and np.array_equal(got, ids[:len(got)])
        end = at + len(ids)
    check("each file sits where `files` says, whole but the last", placed)
    check("one <|endoftext|> between two files, none after a file that "
          "already ends in one", joins)
    doubled = False
    last = None
    for v in data.tolist():
        if v in SPACE:
            continue
        doubled = doubled or (v == EOT and last == EOT)
        last = v
    check("no <|endoftext|> follows another", not doubled)

    rng = np.random.default_rng([0, 0, 0])
    drawn = [short[int(rng.integers(0, len(short)))]
             for _ in range(len(r.files))]
    check("the files are the lane generator's draws, one after another",
          drawn == [p for p, _ in r.files])

    long = [write(os.path.join(tmp, "long", f"{i}.txt"),
                  f"line {i} of a long file about chess openings.\n" * 500)
            for i in range(3)]
    lane2 = T._Lane("long", long, np.random.default_rng([0, 1, 5]))
    r2 = lane2.open(None, 64, 256, "cpu", span)
    rng = np.random.default_rng([0, 1, 5])
    path = long[int(rng.integers(0, len(long)))]
    ids = as_stream(path)
    pos = int(rng.integers(0, len(ids) - span - 1))
    check("a visit to a long file reads that file from an offset drawn after "
          "it, as before",
          r2.name == path and r2.pos == pos and len(r2.data) == len(ids)
          and r2.files == [[path, pos]],
          f"{os.path.basename(r2.name)} at {r2.pos}, expected "
          f"{os.path.basename(path)} at {pos}")

    lane3 = T._Lane("gone", [os.path.join(tmp, "gone", "x.txt")],
                    np.random.default_rng(0))
    check("a subject whose files have all gone gives no visit",
          lane3.open(None, 64, 256, "cpu", span) is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n  " + ("all visit-stream tests pass" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
