#!/usr/bin/env python3
"""The tool-use corpus lane.

Few-shot round-trips teaching the model to emit <tool> calls and read <result>.
Deterministic (seeded), so a rebuild reproduces the same files. Same contract as
corpora/chat.py: main() writes data_tool_use_char/{train,val}.bin + meta.json.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from minagi.tokenizer import ByteTokenizer

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"
T0, T1 = "<tool>", "</tool>"
R0, R1 = "<result>", "</result>"

SCREEN_TEXTS = ["2 + 2 = 4.", "This is a fine statement.",
                "The tokenizer has 9 markers.", "Reading is training."]
SCREEN_BANDS = [{"ok": True, "band": "Green", "reasons": []},
                {"ok": True, "band": "Yellow", "reasons": ["empty"]}]

JUDGES = [
    {"name": "torch", "args": {"name": "torch", "is_verified": True,
                               "is_open_source": True, "has_lock_in": False,
                               "is_proportional": True, "is_maintainable": True,
                               "security_scan_result": "clean",
                               "evidence_level": "runtime"}},
    {"name": "mystery-sdk", "args": {"name": "mystery-sdk", "is_verified": False,
                                     "is_open_source": False, "has_lock_in": True,
                                     "is_proportional": False,
                                     "is_maintainable": False,
                                     "security_scan_result": "unknown",
                                     "evidence_level": "assumption"}},
]


def render_turn(rng):
    kind = rng.random()
    if kind < 0.5:
        text = rng.choice(SCREEN_TEXTS)
        call = {"name": "maat_screen", "args": {"text": text}}
        result = {"ok": True, "band": "Green", "reasons": []}
        answer = "It passed the screen."
    else:
        ex = rng.choice(JUDGES)
        call = {"name": "maat_judge", "args": ex["args"]}
        result = ({"ok": True, "verdict": "PASS", "gates": []}
                  if ex["name"] == "torch" else
                  {"ok": True, "verdict": "FAIL", "gates": []})
        answer = ("I would adopt it." if ex["name"] == "torch"
                  else "I would not adopt it.")
    body = f"{T0}\n{json.dumps(call)}\n{T1}\n"
    res = f"{R0}\n{json.dumps(result)}\n{R1}\n"
    q = "Screen this for me." if call["name"] == "maat_screen" else \
        "Should I adopt this dependency?"
    return f"{U0}\n{q}\n{U1}\n{B0}\n{body}{res}{answer}\n{B1}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_tool_use_char")
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
               "format": f"{U0}...{U1}{B0}...{B1}"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
