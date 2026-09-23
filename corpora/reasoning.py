"""
Bring in reasoning traces, in the markers this model already uses.

OpenThoughts-114k is DeepSeek-R1 deliberating on maths, code and science: a
long chain of thought and then an answer. The shape is already in the corpus -
the arithmetic data writes its working between think markers - so these
reinforce a format the model is learning rather than introducing a new one.
The source wraps its sections in <|begin_of_thought|> and
<|begin_of_solution|>; those become <think> and the answer body, which are
single symbols in this alphabet rather than spellings.

Length is not filtered. A trace longer than the window is read across several
windows: every chunk still trains, and only the ability to attend back to the
problem while writing the answer is lost, which for a deliberation that is
mostly locally structured costs little.

    python3 -m corpora reasoning --limit 2000     # try it small first
    python3 -m corpora reasoning                  # all of it

Writes one file per trace, sharded into subdirectories - a hundred thousand
files in one directory is slow to walk and unpleasant to look at.
"""

import argparse
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

THOUGHT = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.S)
SOLUTION = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.S)


def convert(row):
    """One record as this model reads text, or None if it does not fit the shape."""
    turns = row.get("conversations") or []
    out, saw_answer = [], False
    for t in turns:
        who = (t.get("from") or "").lower()
        val = (t.get("value") or "").strip()
        if not val:
            continue
        if who in ("user", "human"):
            out.append(f"<user>\n{val}\n</user>\n")
        else:
            think = THOUGHT.search(val)
            answer = SOLUTION.search(val)
            body = ""
            if think:
                body += f"<think>\n{think.group(1).strip()}\n</think>\n"
            if answer:
                body += answer.group(1).strip()
                saw_answer = True
            elif not think:
                body += val
                saw_answer = True
            else:
                # a deliberation with the solution outside the markers
                rest = SOLUTION.sub("", THOUGHT.sub("", val)).strip()
                body += rest
                saw_answer = bool(rest)
            out.append(f"<bot>\n{body}\n</bot>\n")
    text = "".join(out)
    return text if (saw_answer and text.startswith("<user>")) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="open-thoughts/OpenThoughts-114k")
    ap.add_argument("--out", default="data/train/reasoning")
    ap.add_argument("--limit", type=int, default=0, help="0 for everything")
    ap.add_argument("--shard", type=int, default=500,
                    help="files per subdirectory")
    ap.add_argument("--held-out", default="data/val/reasoning",
                    help="where to put the traces kept back from training")
    ap.add_argument("--hold", type=int, default=400,
                    help="traces held back, so the domain can be measured at "
                         "all; every other domain has a held-out split")
    ap.add_argument("--stream", action="store_true",
                    help="read over the network instead of downloading; "
                         "fine for a small --limit, hours for all of it")
    a = ap.parse_args()

    from datasets import load_dataset
    if a.stream:
        print(f"  streaming {a.dataset}", flush=True)
        d = load_dataset(a.dataset, split="train", streaming=True)
    else:
        # Streaming a hundred thousand long records takes hours, because each
        # one is a separate request. The parquet is a gigabyte and a half and
        # reads locally in minutes.
        print(f"  downloading {a.dataset} (about 1.5 GB)", flush=True)
        d = load_dataset(a.dataset, split="train")
        print(f"  {len(d):,} records", flush=True)
    os.makedirs(a.out, exist_ok=True)
    n = chars = skipped = 0
    for i, row in enumerate(d):
        if a.limit and n >= a.limit:
            break
        text = convert(row)
        if text is None:
            skipped += 1
            continue
        # the first few are kept back and never trained on, so the domain
        # has a held-out score like every other one
        if n < a.hold:
            os.makedirs(a.held_out, exist_ok=True)
            with open(os.path.join(a.held_out, f"trace-{n:06d}.txt"), "w") as f:
                f.write(text)
        else:
            sub = os.path.join(a.out, f"{(n - a.hold) // a.shard:04d}")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, f"trace-{n:06d}.txt"), "w") as f:
                f.write(text)
        n += 1
        chars += len(text)
        if n % 2000 == 0:
            print(f"    {n:,} traces, {chars/1e6:,.0f}M characters", flush=True)
    print(f"  wrote {n - a.hold:,} traces to {a.out} and {min(a.hold, n):,} "
          f"to {a.held_out}, {chars/1e6:,.1f}M characters"
          f"  ({skipped:,} skipped)", flush=True)
    print(f"  median trace is about {chars//max(n,1):,} characters", flush=True)
    # A STREAMED dataset leaves a worker thread alive, and CPython aborts while
    # finalising it: "PyGILState_Release: thread state ... must be current when
    # releasing". That abort lands AFTER every file is written, so the lane
    # does its whole job and then reports failure with exit 134 - which reads
    # as a broken lane and hides a complete one. fetch.py exits the same way
    # for the same reason. Nothing below has anything left to clean up.
    if a.stream:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
