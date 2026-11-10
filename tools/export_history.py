#!/usr/bin/env python3
"""
The run's history as one tidy record per evaluation.

runs/samples.txt is written to be READ - by a person, at four in the morning,
wanting to know whether the thing is still working. That makes it the wrong
shape for drawing from later: the numbers are embedded in prose, the format
has changed twice, and the generated samples sit in between the records.

This turns it into JSONL and CSV, one row per held-out evaluation, with every
number the log carries and nothing invented. Nothing here reads the model, so
it runs while training does.

    python3 tools/export_history.py --log runs/samples.txt --out runs/archive/run-1

Fields, where the log has them:

    step chars pct minutes experts context ctx_max char_s gaining
    grad_norm clipped train lr evidence_t evidence_n effect rate logit
    val val_se bits_per_char perplexity gap
    dom_<domain> ...            per-domain held-out, one column each
    chess_legal_raw chess_legal_adapted chess_legal_written
                                moves before an impossible one
    repeats_pct                 greedy's repeated 8-grams
    written_repeats_pct         the written reading's, under its rule
    swaps_per_chunk ram_hit_rate dropped_pct
    session                     which restart this record belongs to
"""
import argparse
import csv
import json
import os
import re
import sys

PATS = {
    "step":        r"^step ([\d,]+)\s",
    "chars":       r"^step [\d,]+\s+([\d.]+)M of",
    "pct":         r"of [\d,]+M characters \(([\d.]+)%\)",
    "minutes":     r"\)\s+([\d,]+) min",
    "experts":     r"min\s+([\d,]+) experts",
    "context":     r"^context ([\d,]+) characters of ([\d,]+)",
    "char_s":      r"reading ([\d,]+) char/s",
    "gaining":     r"still gaining ([+-][\d.]+) deep into it",
    "grad_norm":   r"^grad norm ([\d.]+) against a clip of",
    "train":       r"^train loss ([\d.]+)",
    "lr":          r"\slr ([\d.eE+-]+)",
    "evidence_t":  r"evidence t ([+-][\d.]+) over ([\d.]+)",
    "effect":      r"\(effect ([+-][\d.]+)\)",
    "rate":        r"rate x([\d.]+)",
    "logit":       r"^attention logit ([\d.]+) at most",
    "val":         r"^held-out loss ([\d.]+) \+/-([\d.]+) nats",
    "bits":        r"nats\s+([\d.]+) bits/char",
    "perplexity":  r"perplexity ([\d.]+)",
    "gap":         r"gap ([+-][\d.]+)",
    "repeats":     r"^repeats (\d+)% of 8-grams",
    "written":     r"^written .*: repeats (\d+)% of 8-grams",
    "swaps":       r"([\d.]+) experts loaded\s*onto the card per chunk",
    "ram_hit":     r"RAM hit rate ([\d.]+)",
    "dropped":     r"capacity dropped ([\d.]+)% of routing",
}


def num(s):
    return float(s.replace(",", "")) if s is not None else None


def parse(path):
    """One record per held-out evaluation, in the order the log has them."""
    rows, cur, session = [], {}, 0
    last_min = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()
            for key, pat in PATS.items():
                m = re.search(pat, line)
                if not m:
                    continue
                g = m.groups()
                if key == "step":
                    # a fresh progress block: whatever was half-built is stale
                    cur = {"step": int(num(g[0]))}
                elif key == "context":
                    cur["context"], cur["ctx_max"] = int(num(g[0])), int(num(g[1]))
                elif key == "evidence_t":
                    cur["evidence_t"], cur["evidence_n"] = num(g[0]), num(g[1])
                elif key == "val":
                    cur["val"], cur["val_se"] = num(g[0]), num(g[1])
                elif key in ("experts", "minutes"):
                    cur[key] = int(num(g[0]))
                else:
                    field = {"bits": "bits_per_char", "repeats": "repeats_pct",
                             "written": "written_repeats_pct",
                             "swaps": "swaps_per_chunk",
                             "ram_hit": "ram_hit_rate",
                             "dropped": "dropped_pct"}.get(key, key)
                    # The repeat lines come after held-out, which closed the
                    # record, so they belong to the one just appended - put
                    # on `cur`, they were dropped from every record.
                    if key in ("repeats", "written") and rows and "step" not in cur:
                        rows[-1][field] = num(g[0])
                    else:
                        cur[field] = num(g[0])
            # The per-domain line FOLLOWS held-out, so it has to be attached to
            # the record already appended rather than to `cur` - which is the
            # mistake the first version of this made, reporting zero domains.
            if rows and line and rows[-1].get("val") is not None:
                d = re.findall(r"([a-z][a-z_]*)\s+([\d.]+)", line)
                # The domain line is NOTHING BUT name/value pairs. Matching
                # loosely instead picked words out of the generated samples -
                # "add 6", "travels 60" - and invented fifty domains.
                rebuilt = " ".join(f"{k} {v}" for k, v in d)
                whole = re.sub(r"\s+", " ", rebuilt) == re.sub(r"\s+", " ", line)
                if (len(d) >= 5 and whole
                        and all(0.0 < float(v) < 20.0 for _, v in d)):
                    for k, v in d:
                        rows[-1][f"dom_{k}"] = float(v)
            # Chess legality is written in the sample block, which also
            # comes after the held-out line, so it attaches to the record
            # already appended. Only the chess prompt emits "legal moves",
            # so the label alone identifies it.
            if rows:
                m = re.search(r"^\[(raw|adapted|written)\].*?(\d+) legal moves", line)
                if m:
                    rows[-1][f"chess_legal_{m.group(1)}"] = int(m.group(2))
            # a record is complete once it has both a step and a held-out value
            if cur.get("val") is not None and cur.get("step") is not None:
                mins = cur.get("minutes")
                if mins is not None and last_min is not None and mins < last_min:
                    session += 1
                if mins is not None:
                    last_min = mins
                cur["session"] = session
                rows.append(cur)
                cur = {}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/samples.txt")
    ap.add_argument("--out", default="runs/archive/run-1")
    a = ap.parse_args()
    if not os.path.exists(a.log):
        print(f"  no log at {a.log}")
        return 1
    rows = parse(a.log)
    if not rows:
        print(f"  {a.log} carries no complete evaluations")
        return 1
    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, "history.jsonl")
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    cols = []
    for r in rows:                              # union, first-seen order
        for k in r:
            if k not in cols:
                cols.append(k)
    cv = os.path.join(a.out, "history.csv")
    with open(cv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    doms = sorted(c[4:] for c in cols if c.startswith("dom_"))
    print(f"  {len(rows):,} evaluations -> {jl}")
    print(f"  {len(rows):,} evaluations -> {cv}")
    print(f"  {len(cols)} columns, {len(doms)} domains: {', '.join(doms)}")
    print(f"  sessions (restarts): {rows[-1]['session'] + 1}")
    print(f"  characters {rows[0].get('chars', 0):.0f}M -> {rows[-1].get('chars', 0):.0f}M")
    have = [c for c in cols if sum(1 for r in rows if r.get(c) is not None) > len(rows) * 0.5]
    thin = [c for c in cols if c not in have]
    if thin:
        print(f"  present in under half the rows (format changed mid-run): "
              f"{', '.join(thin)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
