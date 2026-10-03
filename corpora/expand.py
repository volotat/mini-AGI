#!/usr/bin/env python3
"""
Write the corpora out as text files, in folders, the way a user would have them.

The generators produced uint16 `.bin` streams, which meant training read
something no person could open and no user could reproduce - and it meant the
project trained through one path while documenting another. Text on disk is the
only format the model actually needs, since its alphabet is the 256 byte
values, so the corpora become ordinary files and training becomes

    python3 train.py read data/train

which is exactly what someone pointing it at their own notes would type.

Each corpus is split across many files rather than one large one, because
`read` opens a fresh window per file and a single multi-hundred-megabyte file
would be one long window with no boundaries in it.

    python3 -m corpora expand --out data
"""

import argparse
import os
import shutil
import sys

import numpy as np

# The repo root, found rather than assumed, so that data_* directories resolve
# against the checkout no matter where this is invoked from. minagi/ marks it.
def _find_root(start):
    d = os.path.dirname(os.path.abspath(start))
    while True:
        if os.path.isdir(os.path.join(d, "minagi")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return os.path.dirname(os.path.abspath(start))
        d = up


ROOT = _find_root(__file__)
sys.path.insert(0, ROOT)

SOURCES = [
    ("code", "data_char", ".py"),
    ("arithmetic", "data_math_char", ".txt"),
    ("chat", "data_chat_char", ".txt"),
    ("self-knowledge", "data_self_knowledge_char", ".txt"),
    ("chess", "data_chess_char", ".txt"),
]


def decode(arr, tok):
    """
    The character stream back to the text it was made from.

    Not a cast to uint8. The structural markers - <user>, <bot>, <g>, <think>
    and the rest - are ids 256 and above, and casting wraps them onto control
    bytes, silently deleting every turn boundary in the conversation corpus and
    every game boundary in the chess one. The tokenizer knows they are markers
    and writes them back as the text they stand for, so re-reading the file
    produces the same ids it started from.
    """
    return tok.decode([int(v) for v in arr]).encode("utf-8", "surrogateescape")


def write_shards(raw, out_dir, stem, ext, shard_chars, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    n = len(raw) if limit is None else min(len(raw), limit)
    written = chars = 0
    i = 0
    while i < n:
        j = min(i + shard_chars, n)
        # end on a line boundary where one is nearby, so a file does not stop
        # mid-token and teach the model that documents end that way
        k = raw.rfind(b"\n", i + shard_chars // 2, j)
        if k > i:
            j = k + 1
        path = os.path.join(out_dir, f"{stem}-{written:05d}{ext}")
        with open(path, "wb") as f:
            f.write(raw[i:j])
        written += 1
        chars += j - i
        i = j
    return written, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--shard-chars", type=int, default=200_000,
                    help="characters per file; a read opens a window per file")
    ap.add_argument("--limit-mb", type=float, default=0,
                    help="cap each corpus, for a quick smaller build")
    ap.add_argument("--clean", action="store_true",
                    help="remove the output directory first")
    ap.add_argument("--only", action="append", default=[],
                    help="rebuild just these corpora; repeatable. The lane is "
                         "emptied first, because shard counts change and a "
                         "shorter rebuild would leave the tail of the old one "
                         "behind, mixing two formats in one subject.")
    a = ap.parse_args()

    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()

    if a.clean and os.path.exists(a.out):
        shutil.rmtree(a.out)
    limit = int(a.limit_mb * 1e6) if a.limit_mb else None

    total = {}
    for name, src, ext in SOURCES:
        if a.only and name not in a.only:
            continue
        for split, binf in (("train", "train.bin"), ("val", "val.bin")):
            p = os.path.join(ROOT, src, binf)
            if not os.path.exists(p):
                print(f"  {src}/{binf} missing - skipped", file=sys.stderr)
                continue
            arr = np.memmap(p, dtype=np.uint16, mode="r")
            cap = limit if split == "train" else (
                None if limit is None else max(limit // 50, 200_000))
            raw = decode(np.asarray(arr[:cap] if cap else arr), tok)
            out_dir = os.path.join(a.out, split, name)
            if a.only and os.path.isdir(out_dir):
                for f in os.listdir(out_dir):
                    if f.endswith(ext):
                        os.remove(os.path.join(out_dir, f))
            n, chars = write_shards(raw, out_dir, name, ext, a.shard_chars)
            total[(split, name)] = (n, chars)
            print(f"  {split}/{name:<11} {n:>5} files  {chars/1e6:>7.1f}M chars",
                  flush=True)

    tr = sum(c for (s, _), (_, c) in total.items() if s == "train")
    va = sum(c for (s, _), (_, c) in total.items() if s == "val")
    nf = sum(n for (n, _) in total.values())
    print(f"\n  {nf:,} files, {tr/1e6:.1f}M characters to train on, "
          f"{va/1e6:.1f}M held out")
    print(f"  train:  python3 train.py read {a.out}/train --save")
    print(f"  score:  --held-out {a.out}/val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
