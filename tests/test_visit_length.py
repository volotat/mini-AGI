#!/usr/bin/env python3
"""
How long a subject is read for before the next one: train.py's
_visit_chunks, which cmd_read reads its visits by - --passage characters of
one subject, a chunk a step, and never less than one full window.

What these tests hold it to:
  a visit is the passage in chunks, rounded up, so the number in the config
  is the number in effect - the rule this replaced counted chunks and
  rounded them up to whole windows, so the configured number was not the
  one read
  never less than one window, or the window never fills and the model reads
  with less context than it has
  at this run's settings a visit is 32,768 characters: 16 chunks of 2,048,
  8 windows of 4,096
  cmd_read reads its visits through it
"""
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T                                           # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


v = T._visit_chunks
check("a visit is the passage in chunks: the number configured is the one "
      "read", all(v(4096, 2048, p) * 2048 == p for p in (8192, 10240, 65536)),
      f"{[v(4096, 2048, p) for p in (8192, 10240, 65536)]}")
check("...rounded up when the passage is not whole chunks",
      v(4096, 2048, 5000) == 3)
check("never less than one full window",
      v(32768, 2048, 4096) == 16 and v(4097, 2048, 2048) == 3
      and v(4096, 512, 1) == 8)
check("at this run's settings: 16 chunks of 2,048, 8 windows of 4,096",
      v(4096, 2048, 32768) == 16)

src = inspect.getsource(T.cmd_read)
check("cmd_read reads its visits through it",
      "_visit_chunks(ctx, args.chunk, args.passage)" in src)

print("\n  all visit-length tests pass" if ok else "\n  FAILURES")
sys.exit(0 if ok else 1)
