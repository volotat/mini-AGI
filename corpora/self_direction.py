#!/usr/bin/env python3
"""The self-direction curriculum lane.

Deterministic telemetry -> directive pairs teaching the model to read its own
<self> block and steer: improving -> lr up, settling -> lr down, regressing ->
grow 0. Same contract as corpora/tool_use.py.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from minagi.tokenizer import ByteTokenizer

S0, S1 = "<self>", "</self>"
P0, P1 = "<policy>", "</policy>"


def _block(step, held, t, e, idle):
    return (f"{S0}\nstep {step} chars 500000 loss 0.62 held {held:.4f}\n"
            f"lr scale x0.51 verdict t {t:+.2f} e {e:+.3f}\n"
            f"pool experts 160 idle {idle} keep 0.82\n{S1}\n")


CASES = [
    # (kind, held, t, e, idle, directive)
    ("improving", 0.70, 3.0, 0.08, 2, "lr x1.05"),
    ("settling", 0.75, 0.5, 0.01, 3, "lr x0.99"),
    ("regressing", 0.90, -4.0, -0.12, 8, "grow 0"),
]


def render_turn(rng):
    kind, held, t, e, idle, directive = rng.choice(CASES)
    return _block(1000, held, t, e, idle) + f"{P0}\n{directive}\n{P1}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_self_direction_char")
    ap.add_argument("--conversations", type=int, default=10000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = ByteTokenizer()
    rng = random.Random(args.seed)

    for split, count in (("train", args.conversations), ("val", args.val)):
        path = os.path.join(args.out, f"{split}.bin")
        total = 0
        with open(path, "wb") as f:
            batch = []
            for _ in range(count):
                batch.append(render_turn(rng))
                if len(batch) >= 2048:
                    for enc in tok.encode_batch(batch):
                        a = np.array(enc.ids, dtype=np.uint16)
                        f.write(a.tobytes())
                        total += len(a)
                    batch.clear()
            for enc in tok.encode_batch(batch):
                a = np.array(enc.ids, dtype=np.uint16)
                f.write(a.tobytes())
                total += len(a)
        print(f"{split}: {total:,} tokens -> {path}")

    json.dump({"vocab_size": tok.get_vocab_size(), "tokenizer": "byte",
               "format": f"{S0}...{S1}{P0}...{P1}"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
