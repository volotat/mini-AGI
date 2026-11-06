#!/usr/bin/env python3
"""
How much the model repeats itself when it writes greedily, against how much
real text repeats in the same place. Read only: it never writes to the
weights it measures.

Four measures, each the same set of texts every time it runs:

  samples   the run's sample prompts (train.py SAMPLE_PROMPTS), plain greedy,
            written exactly as sample_now writes the raw reading: the share of
            repeated 8-grams, and whether the text ends in a loop.
  openings  real openings from the held-out folder: the model's greedy
            continuation against the real one of the same length - how often
            each ends in a loop, and each one's share of repeated 8-grams. The
            real continuation is the yardstick: code repeats itself
            legitimately, a story does not.
  feeding   a real phrase, line or sentence after its own preceding text,
            repeated six times: the mean probability of its characters at each
            repetition (TP) and the share of them that are the argmax (WR).
            Rising means a repetition feeds itself (Xu et al. 2022).
  copying   real windows read with the true text: at open choices (entropy
            above 1 nat), how often the argmax continues an earlier passage
            and how often the real text does; and where the real text does
            continue an earlier passage of 8+ characters, how often the argmax
            does too - repetition the writer intended, which must survive.

A loop is a block repeated three times or more at a text's end, covering 24
characters or more, holding a letter or digit and not one character repeated
(minagi/practice.py loop_tail).

    python3 tools/loop_meter.py --weights weights --out runs/loop_meter/NAME
    python3 tools/loop_meter.py --weights DIR --out OUT --quick   # no writing
    python3 tools/loop_meter.py --compare OUT1 OUT2 ...           # side by side

The texts it writes go to OUT/texts.txt, under runs/, which the self-knowledge
lane does not read.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def copy_candidates(ids, lengths=(32, 16, 8, 4)):
    """For every position t: the character that followed the most recent
    earlier occurrence of the longest of `lengths` characters ending at t, and
    that length (0 and -1 where there is none). Looks only at ids[:t+1]."""
    ids = np.asarray(ids)
    n = len(ids)
    cand = np.full(n, -1)
    klen = np.zeros(n, int)
    raw = ids.astype(np.uint16).tobytes()
    last = {m: {} for m in lengths}
    for t in range(n):
        for m in lengths:
            if t + 1 >= m:
                j = last[m].get(raw[2 * (t - m + 1):2 * (t + 1)])
                if j is not None:
                    cand[t], klen[t] = ids[j + 1], m
                    break
        for m in lengths:
            if t + 1 >= m:
                last[m][raw[2 * (t - m + 1):2 * (t + 1)]] = t
    return cand, klen


def subject_files(val):
    out = {}
    for name in sorted(os.listdir(val)):
        fs = []
        for dp, _, names in os.walk(os.path.join(val, name)):
            fs += [os.path.join(dp, f) for f in names]
        if fs:
            out[name] = sorted(fs)
    return out


def read_text(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def compare(paths):
    """Several meter.json files side by side, the first as the reference."""
    recs = []
    for p in paths:
        p = os.path.join(p, "meter.json") if os.path.isdir(p) else p
        with open(p) as f:
            recs.append((os.path.basename(os.path.dirname(p)), json.load(f)))
    w = max(12, *(len(n) for n, _ in recs))
    print(f"{'':<34}" + "".join(f"{n:>{w + 2}}" for n, _ in recs))

    def row(label, get, fmt):
        vals = []
        for _, r in recs:
            try:
                vals.append(fmt(get(r)))
            except (KeyError, TypeError, IndexError, ZeroDivisionError):
                vals.append("-")
        print(f"{label:<34}" + "".join(f"{v:>{w + 2}}" for v in vals))

    pct = "{:.1%}".format
    row("step", lambda r: r["step"], "{:,}".format)
    row("samples: repeated 8-grams", lambda r: r["samples_rep8"], pct)
    row("samples: ending in a loop", lambda r: r["samples_loops"], str)
    row("openings: loops written / real",
        lambda r: (r["openings_all"]["loops_written"],
                   r["openings_all"]["loops_real"],
                   r["openings_all"]["n"]),
        lambda v: f"{v[0]}/{v[1]} of {v[2]}")
    row("openings: rep. 8-grams written",
        lambda r: r["openings_all"]["rep8_written"], pct)
    row("openings: rep. 8-grams real",
        lambda r: r["openings_all"]["rep8_real"], pct)
    for k in range(6):
        row(f"feeding: TP at repetition {k + 1}",
            lambda r, k=k: r["feeding_all"]["TP"][k], "{:.3f}".format)
    row("copying: argmax at open choices",
        lambda r: r["copying_all"]["argmax_copies_at_open"], pct)
    row("copying: real at open choices",
        lambda r: r["copying_all"]["real_copies_at_open"], pct)
    row("intended repetition kept",
        lambda r: r["copying_all"]["intended_kept"], pct)
    names = sorted(set().union(*(r.get("samples", {}) for _, r in recs)))
    for n in names:
        row(f"  sample {n}: rep8 / loop",
            lambda r, n=n: r["samples"][n],
            lambda v: f"{v['rep8']:.0%}{' L' if v['loop'] else ''}")
    return 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        return compare(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None,
                    help="cuda when there is one, else cpu")
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--openings", type=int, default=6,
                    help="real openings per subject")
    ap.add_argument("--prefix", type=int, default=512,
                    help="characters of real text before each continuation")
    ap.add_argument("--chars", type=int, default=256,
                    help="characters written after each opening")
    ap.add_argument("--spans", type=int, default=12,
                    help="repeated spans per subject (feeding)")
    ap.add_argument("--windows", type=int, default=3,
                    help="windows of 2,048 characters per subject (copying)")
    ap.add_argument("--quick", action="store_true",
                    help="only the measures that read: no writing")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subjects to measure; all by default")
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    os.chdir(ROOT)
    import train as T
    from minagi.practice import loop_tail, pick_span
    from minagi.tokenizer import ByteTokenizer

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    model, cfg, pool, man = T.build_paged(
        os.path.abspath(a.weights), device, ram_capacity=96, read_only=True)
    model.eval()
    tok = ByteTokenizer()
    os.makedirs(a.out, exist_ok=True)
    texts = open(os.path.join(a.out, "texts.txt"), "w", encoding="utf-8")
    rec = {"weights": os.path.abspath(a.weights), "step": man.get("step"),
           "experts": len(man["telemetry"]["uid"]), "device": dev}
    say(f"weights at step {man.get('step'):,}, {rec['experts']} experts, "
        f"on {dev}")
    files = subject_files(os.path.join(ROOT, a.held_out))
    keep = set(a.subjects.split(",")) if a.subjects else None
    if keep:
        files = {k: v for k, v in files.items() if k in keep}

    @torch.no_grad()
    def write(ids, n):
        """Greedy, as sample_now writes its raw reading."""
        ids = list(ids)[-cfg.block:]
        cur = torch.tensor([ids], device=device)
        caches = model.empty_caches()
        off, got = 0, []
        for _ in range(n):
            if off + cur.shape[1] > cfg.block:
                break
            logits, _ = model(cur, caches=caches, pos_offset=off)
            off += cur.shape[1]
            t = int(logits[0, -1].float().argmax())
            got.append(t)
            cur = torch.tensor([[t]], device=device)
        return got

    def judged(s):
        lp = loop_tail(s)
        return {"rep8": round(T.repeat_rate(s), 4),
                "loop": list(lp) if lp else None}

    t0 = time.time()
    if not a.quick:
        # ---- the run's sample prompts, as the raw reading writes them
        rec["samples"] = {}
        for name, prompt in T.SAMPLE_PROMPTS:
            if keep and name not in keep:
                continue
            s = tok.decode(write(tok.encode(prompt).ids, 140))
            rec["samples"][name] = judged(s)
            texts.write(f"=== sample {name}\n{prompt!r}\n--- written\n{s}\n\n")
        sm = rec["samples"].values()
        rec["samples_rep8"] = round(float(np.mean([v["rep8"] for v in sm])), 4)
        rec["samples_loops"] = sum(v["loop"] is not None for v in sm)
        say(f"samples: repeated 8-grams {rec['samples_rep8']:.1%}, "
            f"{rec['samples_loops']} of {len(rec['samples'])} end in a loop "
            f"({time.time() - t0:.0f} s)")
        # ---- real openings, written on, against their real continuations
        rec["openings"] = {}
        for k, (name, fs) in enumerate(files.items()):
            rng = np.random.default_rng([a.seed, k, 1])
            rows = []
            for i in range(a.openings):
                text = ""
                for _ in range(20):
                    text = read_text(fs[int(rng.integers(0, len(fs)))])
                    if len(text) > a.prefix + a.chars + 400:
                        break
                if len(text) <= a.prefix + a.chars + 400:
                    continue
                pos = int(rng.integers(0, len(text) - a.prefix - a.chars - 400))
                nl = text.find("\n", pos, pos + 300)
                pos = nl + 1 if nl >= 0 else pos
                prefix = text[pos:pos + a.prefix]
                real = text[pos + a.prefix:pos + a.prefix + a.chars]
                gen = tok.decode(write(tok.encode(prefix).ids, a.chars))
                rows.append({"gen": judged(gen), "real": judged(real)})
                texts.write(f"=== {name} opening {i}\n...{prefix[-160:]}\n"
                            f"--- real\n{real}\n--- written\n{gen}\n\n")
            if rows:
                rec["openings"][name] = {
                    "n": len(rows),
                    "loops_written": sum(r["gen"]["loop"] is not None
                                         for r in rows),
                    "loops_real": sum(r["real"]["loop"] is not None
                                      for r in rows),
                    "rep8_written": round(float(np.mean(
                        [r["gen"]["rep8"] for r in rows])), 4),
                    "rep8_real": round(float(np.mean(
                        [r["real"]["rep8"] for r in rows])), 4)}
        op = rec["openings"].values()
        rec["openings_all"] = {
            "n": sum(v["n"] for v in op),
            "loops_written": sum(v["loops_written"] for v in op),
            "loops_real": sum(v["loops_real"] for v in op),
            "rep8_written": round(float(np.mean([v["rep8_written"]
                                                 for v in op])), 4),
            "rep8_real": round(float(np.mean([v["rep8_real"] for v in op])), 4)}
        o = rec["openings_all"]
        say(f"openings: {o['loops_written']} of {o['n']} written continuations "
            f"end in a loop (real: {o['loops_real']}); repeated 8-grams "
            f"{o['rep8_written']:.1%} written, {o['rep8_real']:.1%} real "
            f"({time.time() - t0:.0f} s)")
    texts.close()

    @torch.no_grad()
    def probs(ids):
        x = torch.tensor([list(ids[:-1])], device=device)
        logits, _ = model(x)
        return F.log_softmax(logits[0].float(), -1)

    # ---- feeding: does a repetition raise its own odds?
    REP, CTX = 6, 200
    rec["feeding"] = {}
    for k, (name, fs) in enumerate(files.items()):
        rng = np.random.default_rng([a.seed, k, 2])
        TP, WR = [], []
        for _ in range(a.spans * 6):
            if len(TP) >= a.spans:
                break
            text = read_text(fs[int(rng.integers(0, len(fs)))])
            if len(text) < 1200:
                continue
            p0 = int(rng.integers(0, len(text) - 1100))
            ids = np.asarray(tok.encode(text[p0:p0 + 1100]).ids)
            got = pick_span(ids, rng, min_len=12, max_len=120,
                            min_prefix=CTX)
            if got is None:
                continue
            s, e = got
            span = list(ids[s:e])
            seq = list(ids[s - CTX:s]) + span * REP
            lp = probs(seq)
            y = torch.tensor(seq[1:], device=device)
            tp, wr = [], []
            for r in range(REP):
                lo = CTX + r * len(span) - 1
                rows = lp[lo:lo + len(span)]
                tgt = y[lo:lo + len(span)]
                tp.append(float(rows.gather(1, tgt[:, None]).exp().mean()))
                wr.append(float((rows.argmax(-1) == tgt).float().mean()))
            TP.append(tp)
            WR.append(wr)
        if TP:
            rec["feeding"][name] = {
                "n": len(TP),
                "TP": [round(float(v), 4) for v in np.mean(TP, 0)],
                "WR": [round(float(v), 4) for v in np.mean(WR, 0)]}
    fd = rec["feeding"].values()
    rec["feeding_all"] = {
        "TP": [round(float(np.mean([v["TP"][r] for v in fd])), 4)
               for r in range(REP)],
        "WR": [round(float(np.mean([v["WR"][r] for v in fd])), 4)
               for r in range(REP)]}
    f = rec["feeding_all"]
    say(f"feeding: TP by repetition {' '.join(f'{v:.3f}' for v in f['TP'])}; "
        f"WR {f['WR'][0]:.3f} -> {f['WR'][-1]:.3f} ({time.time() - t0:.0f} s)")

    # ---- copying: at open choices, and where the real text copies
    rec["copying"] = {}
    W = 2048
    for k, (name, fs) in enumerate(files.items()):
        rng = np.random.default_rng([a.seed, k, 3])
        acc = dict(n=0, open=0, open_cand=0, am_copy_open=0, gold_copy_open=0,
                   intended=0, intended_kept=0)
        done = 0
        for _ in range(a.windows * 6):
            if done >= a.windows:
                break
            text = read_text(fs[int(rng.integers(0, len(fs)))])
            if len(text) < W + 200:
                continue
            p0 = int(rng.integers(0, len(text) - W - 100))
            ids = np.asarray(tok.encode(text[p0:p0 + W + 1]).ids)[:W + 1]
            lp = probs(ids)
            p = lp.exp()
            H = -(p * lp).sum(-1).cpu().numpy()
            am = lp.argmax(-1).cpu().numpy()
            gold = ids[1:]
            cand, klen = copy_candidates(ids[:-1])
            for t in range(256, len(gold)):
                acc["n"] += 1
                if H[t] > 1.0:
                    acc["open"] += 1
                    if cand[t] >= 0:
                        acc["open_cand"] += 1
                        acc["am_copy_open"] += int(am[t] == cand[t])
                        acc["gold_copy_open"] += int(gold[t] == cand[t])
                if klen[t] >= 8 and gold[t] == cand[t]:
                    acc["intended"] += 1
                    acc["intended_kept"] += int(am[t] == gold[t])
            done += 1
        if acc["n"]:
            o = max(acc["open"], 1)
            rec["copying"][name] = {
                "positions": acc["n"],
                "open_share": round(acc["open"] / acc["n"], 4),
                "argmax_copies_at_open": round(acc["am_copy_open"] / o, 4),
                "real_copies_at_open": round(acc["gold_copy_open"] / o, 4),
                "intended": acc["intended"],
                "intended_kept": round(acc["intended_kept"]
                                       / max(acc["intended"], 1), 4)}
    cp = rec["copying"].values()
    rec["copying_all"] = {
        k: round(float(np.mean([v[k] for v in cp])), 4)
        for k in ("argmax_copies_at_open", "real_copies_at_open",
                  "intended_kept")}
    c = rec["copying_all"]
    say(f"copying: at open choices the argmax copies "
        f"{c['argmax_copies_at_open']:.1%}, the real text "
        f"{c['real_copies_at_open']:.1%}; intended repetition kept "
        f"{c['intended_kept']:.1%} ({time.time() - t0:.0f} s)")
    rec["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "meter.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    say(f"-> {os.path.join(a.out, 'meter.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
