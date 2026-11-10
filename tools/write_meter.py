#!/usr/bin/env python3
"""
How well each way of choosing characters writes. The same prompts are written
on by every setting of minagi/decode.py's Sampler, and each text is measured
against the real text that followed its prompt. Read only: it never writes to
the weights it reads.

THE PROMPTS, the same every run for the same --seed:

  openings  real held-out text, `--openings` per subject: `--prefix`
            characters, then `--chars` written, kept beside the real
            continuation of the same length. An arithmetic opening ends at a
            problem's "= ", a chess one a few moves into a game.
  samples   the run's sample prompts (train.py SAMPLE_PROMPTS); a reply in
            chat markers ends at </bot>.

Every setting draws with the same seed on the same prompt, so two settings
differ by their rule and not by their luck.

THE MEASURES of each text:

  rep8      the share of its 8-grams seen earlier in it (train.repeat_rate)
  loop      whether it ends in a loop (minagi.practice.loop_tail)
  nonwords  the share of its words of three letters or more found nowhere in
            the subject's held-out text or a sample of its training text -
            spelling
  sums      arithmetic: the problems it answers, and how many correctly
  moves     chess: legal moves before the first impossible one, or the end
            of the game
  judge     with --judge, an outside model's loss on the text after its
            prompt, in bits a character, beside its loss on the real
            continuation of the same prompt. Below the real text's is too
            predictable - repetition, stock phrases; above it is less
            coherent than real text.
  new_judge the same over the text's new material only: every character
            inside an 8-gram already seen, in the prompt or the text, is
            left out, in the written text and the real one alike. A
            repeated span is easy to predict, so the whole-text judge
            rewards repeating; this one cannot be bought that way.

    python3 tools/write_meter.py --weights weights --out runs/sampling/NAME \\
        --settings SETTINGS.json [--only a,b]
    python3 tools/write_meter.py --out runs/sampling/NAME --judge gpt2-large
    python3 tools/write_meter.py --out runs/sampling/NAME --report \\
        [--by-subject] [--against SETTING]

SETTINGS.json maps a name to the Sampler's settings, e.g.
{"greedy": {}, "min_p 0.1": {"temperature": 1.0, "min_p": 0.1}}. One file of
texts per setting, OUT/<name>.jsonl, so settings can be written by several
processes at once. Texts go under runs/, which the self-knowledge lane does
not read.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

WORD = re.compile(r"[A-Za-z]{3,}")
MOVE_NO = re.compile(r"\d+\.")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# ---------------------------------------------------------------- prompts
def text_opening(text, rng, prefix, chars):
    if len(text) <= prefix + chars + 400:
        return None
    pos = int(rng.integers(0, len(text) - prefix - chars - 400))
    nl = text.find("\n", pos, pos + 300)
    pos = nl + 1 if nl >= 0 else pos
    return text[pos:pos + prefix], text[pos + prefix:pos + prefix + chars]


def sum_opening(text, rng, prefix, chars):
    """Whole lines of problems, then one problem up to its "= "."""
    lines = text.split("\n")
    if len(lines) < 40:
        return None
    i = int(rng.integers(20, len(lines) - 10))
    eq = lines[i].find(" = ")
    if eq < 0:
        return None
    head = "\n".join(lines[:i])[-prefix:]
    head = head[head.find("\n") + 1:]
    real = lines[i][eq + 3:] + "\n" + "\n".join(lines[i + 1:])
    return head + "\n" + lines[i][:eq + 3], real[:chars]


def chess_opening(text, rng, prefix, chars):
    """Whole games, then one game cut after a few of its moves."""
    games = [g for g in text.split("\n") if g.startswith("<g>")]
    if len(games) < 3:
        return None
    i = int(rng.integers(1, len(games)))
    toks = games[i].split(" ")
    nums = [j for j, t in enumerate(toks) if MOVE_NO.fullmatch(t)]
    if len(nums) < 6:
        return None
    cut = nums[int(rng.integers(3, min(15, len(nums) - 1)))]
    game = " ".join(toks[:cut + 1]) + " "
    head = "\n".join(games[:i])[-max(prefix - len(game), 0):]
    head = head[head.find("\n") + 1:] if "\n" in head else ""
    real = " ".join(toks[cut + 1:])
    return (head + "\n" if head else "") + game, real[:chars]


def openings(files, n, prefix, chars, seed):
    from loop_meter import read_text
    out = []
    for k, (name, fs) in enumerate(files.items()):
        rng = np.random.default_rng([seed, k, 1])
        make = {"chess": chess_opening,
                "arithmetic": sum_opening}.get(name, text_opening)
        got = 0
        for _ in range(n * 40):
            if got >= n:
                break
            item = make(read_text(fs[int(rng.integers(0, len(fs)))]), rng,
                        prefix, chars)
            if item:
                out.append({"subject": name, "kind": "opening",
                            "prompt": item[0], "real": item[1]})
                got += 1
    return out


# --------------------------------------------------------------- measures
def sums(text):
    """(problems answered, answered correctly) over the whole lines of an
    arithmetic text; the first line is the answer to the prompt's problem."""
    done = right = 0
    for line in text.split("\n"):
        got = check_sum(line)
        if got is not None:
            done += 1
            right += int(got)
    return done, right


def check_sum(line):
    """Whether one arithmetic line's answer is right; None when it is not a
    whole line of a known problem."""
    m = re.match(r"^\s*(add|sub|mul|mod|sum|gcd|round|cmp)\s+(.*?)\s+=\s+(.+)$",
                 line)
    if not m:
        return None
    op, lhs, rhs = m.groups()
    ans = rhs.split("</think>")[-1].strip()
    if "<think>" in ans or not ans:
        return None
    n = [int(x) for x in re.findall(r"\d+", lhs)]
    try:
        if op == "cmp":
            o = re.search(r"==|<|>", lhs)
            if not o or len(n) < 2:
                return None
            want = {"<": n[0] < n[1], ">": n[0] > n[1],
                    "==": n[0] == n[1]}[o.group(0)]
            return ans == ("yes" if want else "no")
        if len(n) < 2:
            return None
        want = {"add": lambda: n[0] + n[1], "sub": lambda: n[0] - n[1],
                "mul": lambda: n[0] * n[1], "mod": lambda: n[0] % n[1],
                "sum": lambda: sum(n), "gcd": lambda: math.gcd(n[0], n[1]),
                "round": lambda: (n[0] + n[1] // 2) // n[1] * n[1]}[op]()
        return ans == str(want)
    except (ZeroDivisionError, KeyError):
        return None


def chess_moves(prompt, text):
    """Legal moves the text makes in the prompt's last game before an
    impossible one or the game's end, and whether it went wrong."""
    import train as T
    game = prompt[prompt.rfind("<g>"):]
    got = T.chess_legality(game, text.split("</g>")[0])
    if got is None:
        return None
    return got[0], got[1] is not None


_lex = {}


def lexicon(subject, train_mb=24):
    """Words of the subject's held-out text and of a sample of its lane's
    training text. Spelling is judged against both, so a correctly spelled
    word the held-out text happens not to hold is not counted as a slip."""
    if subject in _lex:
        return _lex[subject]
    words = set()
    lane = "chat" if subject.startswith("chat") else subject
    for root, cap in ((os.path.join(ROOT, "data/val", subject), None),
                      (os.path.join(ROOT, "data/train", lane), train_mb << 20)):
        size = 0
        paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root)
                       for f in fs)
        if cap:
            rng = np.random.default_rng(7)
            paths = [paths[i] for i in rng.permutation(len(paths))]
        for p in paths:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
            words.update(w.lower() for w in WORD.findall(t))
            size += len(t)
            if cap and size >= cap:
                break
    _lex[subject] = words
    return words


def measure(rec):
    import train as T
    from minagi.practice import loop_tail
    s = rec["written"]
    out = {"rep8": round(T.repeat_rate(s), 4),
           "loop": bool(loop_tail(s)), "chars": len(s)}
    ws = [w.lower() for w in WORD.findall(s)]
    if ws and rec["subject"] not in ("arithmetic", "chess"):
        lex = lexicon(rec["subject"])
        out["nonwords"] = round(sum(w not in lex for w in ws) / len(ws), 4)
    if rec["subject"] == "arithmetic":
        first = check_sum(rec["prompt"].split("\n")[-1] + s.split("\n")[0])
        out["first_right"] = first
        out["sums"] = sums(s)
    if rec["subject"] == "chess":
        out["moves"] = chess_moves(rec["prompt"], s)
    if rec.get("real") is not None:
        out["real_rep8"] = round(T.repeat_rate(rec["real"]), 4)
        out["real_loop"] = bool(loop_tail(rec["real"]))
    return out


# ---------------------------------------------------------------- writing
def write_all(a):
    import torch
    import train as T
    from loop_meter import subject_files
    from minagi.decode import Sampler
    from minagi.precision import amp
    from minagi.tokenizer import ByteTokenizer, SPECIAL_ID

    torch.set_num_threads(2)
    with open(a.settings) as f:
        settings = json.load(f)
    if a.only:
        keep = [s.strip() for s in a.only.split(",")]
        settings = {k: settings[k] for k in keep}
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available()
                                    else "cpu"))
    model, cfg, pool, man = T.build_paged(
        os.path.abspath(a.weights), dev, ram_capacity=a.ram, read_only=True)
    model.eval()
    tok = ByteTokenizer()
    stop = SPECIAL_ID["</bot>"]
    files = subject_files(os.path.join(ROOT, a.held_out))
    prompts = openings(files, a.openings, a.prefix, a.chars, a.seed)
    prompts += [{"subject": n, "kind": "sample", "prompt": p, "real": None}
                for n, p in T.SAMPLE_PROMPTS]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prompts.json"), "w") as f:
        json.dump({"weights": os.path.abspath(a.weights),
                   "step": man.get("step"), "seed": a.seed,
                   "prompts": prompts}, f)
    say(f"weights at step {man.get('step'):,}; {len(prompts)} prompts; "
        f"settings {', '.join(settings)}")

    @torch.no_grad()
    def write(ids, n, sampler, until=None):
        ids = list(ids)[-cfg.block:]
        hist = list(ids)
        cur = torch.tensor([ids], device=dev)
        caches = model.empty_caches()
        off, got = 0, []
        for _ in range(n):
            if off + cur.shape[1] > cfg.block:
                break
            with amp(dev):
                logits, _ = model(cur, caches=caches, pos_offset=off)
            off += cur.shape[1]
            t = sampler(logits[0, -1], hist)
            got.append(t)
            hist.append(t)
            if t == until:
                break
            cur = torch.tensor([[t]], device=dev)
        return got

    for name, kw in settings.items():
        path = os.path.join(a.out, safe(name) + ".jsonl")
        t0, n_chars = time.time(), 0
        with open(path, "w", encoding="utf-8") as fh:
            for i, p in enumerate(prompts):
                sampler = Sampler(seed=a.seed * 1000 + i, **kw)
                ids = tok.encode(p["prompt"]).ids
                got = write(ids, a.chars, sampler,
                            until=stop if p["kind"] == "sample" else None)
                rec = dict(p, setting=name, index=i,
                           written=tok.decode(got))
                rec["measures"] = measure(rec)
                n_chars += len(got)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        say(f"{name}: {n_chars:,} characters in {dt:.0f} s "
            f"({n_chars / max(dt, 1e-9):.1f}/s) -> {path}")


# ------------------------------------------------------------------ judge
def repeated(prompt, text, n=8):
    """For each character of `text`, whether it lies in an n-gram already
    seen earlier - in the prompt or the text before it."""
    s, P = prompt + text, len(prompt)
    seen, mark = set(), [False] * len(text)
    for i in range(len(s) - n + 1):
        g = s[i:i + n]
        if i + n > P and g in seen:
            for j in range(max(i, P), i + n):
                mark[j - P] = True
        seen.add(g)
    return mark


def judge(a):
    """An outside model's bits a character on every written text and on the
    real continuation of the same prompt, written into each record: over the
    whole text (`judge`), and over its new material only (`new_judge`) -
    every character inside an 8-gram already seen earlier left out, so that
    a text cannot score as predictable by repeating itself."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = a.judge if "/" in a.judge else f"openai-community/{a.judge}"
    dev = a.device or "cuda"
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        name, local_files_only=True,
        dtype=torch.float16 if dev.startswith("cuda") else torch.float32
    ).to(dev).eval()
    ctx = lm.config.n_positions

    @torch.no_grad()
    def bits(prompt, cont):
        """Bits a character of `cont` after `prompt`, whole and new. The pair
        is tokenised whole, the tokens that start inside `cont` are the ones
        scored, and a token's bits are shared out over its characters."""
        if not cont:
            return None, None
        enc = tok(prompt + cont, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        first = next((j for j, (s, _) in enumerate(offs) if s >= len(prompt)),
                     len(ids))
        lo = max(0, len(ids) - ctx)
        if first >= len(ids) or first <= lo:
            return None, None
        x = torch.tensor([ids[lo:]], device=dev)
        lp = torch.log_softmax(lm(x).logits[0, :-1].float(), -1)
        nll = (-lp.gather(1, x[0, 1:, None])[:, 0]).tolist()
        span = nll[first - lo - 1:]
        whole = sum(span) / math.log(2) / max(len(prompt + cont) - offs[first][0], 1)
        mark, P = repeated(prompt, cont), len(prompt)
        new, chars = 0.0, 0
        for t in range(first, len(ids)):
            at = [c - P for c in range(max(offs[t][0], P), offs[t][1])]
            keep = [c for c in at if not mark[c]]
            if keep:
                new += nll[t - lo - 1] * len(keep) / len(at)
                chars += len(keep)
        return whole, (new / math.log(2) / chars if chars else None)

    real_cache = {}
    for fn in sorted(os.listdir(a.out)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(a.out, fn)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        t0 = time.time()
        for r in recs:
            m = r["measures"]
            m["judge"], m["new_judge"] = bits(r["prompt"], r["written"])
            if r.get("real"):
                key = r["index"]
                if key not in real_cache:
                    real_cache[key] = bits(r["prompt"], r["real"])
                m["real_judge"], m["real_new_judge"] = real_cache[key]
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        say(f"judged {fn} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------- report
def report(a):
    rows = {}
    for fn in sorted(os.listdir(a.out)):
        if fn.endswith(".jsonl"):
            recs = [json.loads(l) for l in open(os.path.join(a.out, fn),
                                                encoding="utf-8")]
            if recs:
                rows[recs[0]["setting"]] = recs
    if not rows:
        raise SystemExit("nothing written yet")

    def mean(v):
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else float("nan")

    prose = ("chat", "chat_hermes", "code", "reasoning", "stories",
             "wikipedia")
    print(f"{'setting':<22}{'loops':>7}{'rep8':>7}{'real':>6}{'nonw':>7}"
          f"{'judge':>7}{'real':>6}{'gap':>7}{'new gap':>8}{'sums':>9}"
          f"{'moves':>7}{'smp loops':>10}{'smp rep8':>9}")
    for name, recs in rows.items():
        op = [r for r in recs if r["kind"] == "opening"]
        sm = [r for r in recs if r["kind"] == "sample"]
        pr = [r for r in op if r["subject"] in prose]
        M = lambda r: r["measures"]                         # noqa: E731
        js = [M(r).get("judge") for r in pr]
        jr = [M(r).get("real_judge") for r in pr]
        gap = mean([w - r for w, r in zip(js, jr)
                    if w is not None and r is not None])
        new_gap = mean([M(r).get("new_judge") - M(r).get("real_new_judge")
                        for r in pr if M(r).get("new_judge") is not None
                        and M(r).get("real_new_judge") is not None])
        s_done = sum(M(r)["sums"][0] for r in op if "sums" in M(r))
        s_right = sum(M(r)["sums"][1] for r in op if "sums" in M(r))
        mv = [M(r)["moves"][0] for r in op if M(r).get("moves")]
        print(f"{name:<22}"
              f"{sum(M(r)['loop'] for r in op):>4}/{len(op):<2}"
              f"{mean([M(r)['rep8'] for r in pr]):>7.1%}"
              f"{mean([M(r)['real_rep8'] for r in pr]):>6.1%}"
              f"{mean([M(r).get('nonwords') for r in pr]):>7.1%}"
              f"{mean(js):>7.3f}{mean(jr):>6.3f}{gap:>+7.3f}{new_gap:>+8.3f}"
              f"{s_right:>5}/{s_done:<3}"
              f"{mean(mv):>7.1f}"
              f"{sum(M(r)['loop'] for r in sm):>7}/{len(sm):<2}"
              f"{mean([M(r)['rep8'] for r in sm]):>9.1%}")
    print("\nloops: openings ending in a loop; rep8, nonw, judge: prose "
          "openings (chat, chat_hermes, code, reasoning, stories, "
          "wikipedia), written beside real; judge in bits a character, gap "
          "written minus real, new gap the same over new material only "
          "(characters in an 8-gram seen before left out, in both); sums: "
          "arithmetic lines right / answered; "
          "moves: chess legal moves; smp: the run's sample prompts")
    if a.against:
        # Paired: the same prompts and random numbers, so the difference per
        # opening carries no luck of the draw, and its spread is the error.
        base = {(r["kind"], r["index"]): r for r in rows[a.against]}
        print(f"\nagainst {a.against}, paired over prose openings: judge and "
              f"new_judge differences, +/- standard error; wins: openings "
              f"where new_judge is lower")
        for name, recs in rows.items():
            out = [f"{name:<22}"]
            for key in ("judge", "new_judge"):
                d = [r["measures"].get(key) - base[k]["measures"].get(key)
                     for r in recs if r["kind"] == "opening"
                     and r["subject"] in prose
                     for k in [(r["kind"], r["index"])]
                     if r["measures"].get(key) is not None
                     and base[k]["measures"].get(key) is not None]
                se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
                out.append(f"{key} {float(np.mean(d)) if d else float('nan'):+.3f} +/-{se:.3f}")
            out.append(f"wins {sum(x < 0 for x in d)}/{len(d)}")
            print("   ".join(out))
    if a.by_subject:
        subs = sorted({r["subject"] for recs in rows.values() for r in recs
                       if r["kind"] == "opening"})
        for key in ("judge", "new_judge", "rep8", "nonwords"):
            print(f"\n{key} by subject (openings)")
            print(f"{'setting':<22}" + "".join(f"{s[:11]:>12}" for s in subs))
            for name, recs in rows.items():
                print(f"{name:<22}" + "".join(
                    f"{mean([r['measures'].get(key) for r in recs if r['kind'] == 'opening' and r['subject'] == s]):>12.3f}"
                    for s in subs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated names from SETTINGS to write")
    ap.add_argument("--device", default=None,
                    help="where to write, or with --judge where to judge "
                         "(cuda by default; cpu works while the card is busy)")
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--openings", type=int, default=4,
                    help="real openings per subject")
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--chars", type=int, default=256)
    ap.add_argument("--ram", type=int, default=96,
                    help="experts kept in RAM")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--judge", default=None,
                    help="an outside model to judge with, e.g. gpt2-large")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--by-subject", action="store_true")
    ap.add_argument("--against", default=None,
                    help="with --report: every setting paired against this one")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.judge:
        return judge(a) or 0
    if a.report:
        return report(a) or 0
    if not a.settings:
        raise SystemExit("--settings is needed to write")
    return write_all(a) or 0


if __name__ == "__main__":
    sys.exit(main())
