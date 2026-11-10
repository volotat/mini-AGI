#!/usr/bin/env python3
"""
Rewrite the parts of README.md that are facts about the run.

The numbers in a README go stale the moment the run moves, and a stale
number is worse than no number: it is a claim, it looks maintained, and
nobody re-checks it. So the regions that are facts are generated from the
run's own artefacts rather than typed - the benchmark tables from
runs/samples.txt, the scaling paragraph from the fit that drew the chart,
the two collapsible blocks from the sample log and the progress graph, and
the parameter count from the weights themselves.

Everything between a pair of markers is replaced; everything else in the
README is left exactly as written. The markers stay in the README, which
costs nothing - they are HTML comments - and means the next refresh can
find its own regions in a file that has been edited by hand.

    <!-- auto:NAME -->   ...generated...   <!-- /auto:NAME -->

Run by release.sh before a push, and by measure_benchmarks.sh after it
measures, so what is pushed is current by construction.
"""
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN2 = 0.6931471805599453
BAR = "=" * 78


def read(p, default=None):
    try:
        with open(os.path.join(ROOT, p), encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        if default is None:
            raise
        return default


def entries():
    """Every sample round in the log, as (header, body) pairs.

    The log is written as a header between two rules, then the samples, then
    the next header. Splitting on the rule gives alternating header and body
    from index 1 - index 0 is the session banner.
    """
    parts = read("runs/samples.txt").split(BAR)
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


# every thousands separator, so step 1,234,567 reads as one number
NUM = r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
PACE_HOURS = 3


def running():
    """How fast the run reads on the wall clock, and how long it has run.

    Returns (pace, window, ran): characters a second over the latest
    sessions, the seconds of running that pace was measured over, and the
    seconds the run has spent running at all.

    Each session's banner says how much had been read when it started, and
    each round's header how much has been read by then and how many minutes
    the session has been running - so a session's characters over its
    minutes is its pace with everything the reading waits for included:
    held-out checks, rounds of samples, saves. The reading speed in the
    headers leaves those out, and a projection made from it runs fast.

    The latest session runs the current code, so its pace is the one to
    project with. One too short to measure well - the headers round to a
    tenth of a million characters - borrows the sessions before it until
    PACE_HOURS of running are behind the number. A session stopped before
    its first round logged no time, and is left out of both.
    """
    sessions = []
    for part in re.split(r"^# session started ", read("runs/samples.txt"),
                         flags=re.M)[1:]:
        m = re.search(rf"at step {NUM}, {NUM}M of", part)
        heads = re.findall(rf"^step {NUM}\s+{NUM}M of {NUM}M characters "
                           rf"\([^)]*\)\s+{NUM} min", part, flags=re.M)
        if m and heads and fnum(heads[-1][3]) > 0:
            sessions.append(((fnum(heads[-1][1]) - fnum(m.group(2))) * 1e6,
                             fnum(heads[-1][3]) * 60))
    chars = secs = 0.0
    for c, s in reversed(sessions):
        chars, secs = chars + c, secs + s
        if secs >= PACE_HOURS * 3600:
            break
    return (chars / secs if secs else None, secs,
            sum(s for _, s in sessions))



def ran_until(step):
    """Seconds the run had spent running when it first reached `step` -
    running()'s count, stopped there: every session before it, and the part
    of the one it was reached in. What a measurement of older weights is
    costed at."""
    secs = 0.0
    for part in re.split(r"^# session started ", read("runs/samples.txt"),
                         flags=re.M)[1:]:
        m = re.search(rf"at step {NUM}, {NUM}M of", part)
        if m and fnum(m.group(1)) >= step:
            break
        heads = re.findall(rf"^step {NUM}\s+{NUM}M of {NUM}M characters "
                           rf"\([^)]*\)\s+{NUM} min", part, flags=re.M)
        for h in heads:
            if fnum(h[0]) >= step:
                return secs + fnum(h[3]) * 60
        if heads:
            secs += fnum(heads[-1][3]) * 60
    return secs

def duration(secs):
    if secs >= 2 * 86400:
        return f"{secs / 86400:.0f} days"
    if secs >= 2 * 3600:
        return f"{secs / 3600:.0f} hours"
    return f"{secs / 60:.0f} minutes"


def scale(days):
    """The unit a stretch of reading is naturally counted in."""
    for limit, word in ((1, "hours"), (14, "days"), (60, "weeks"),
                        (365, "months")):
        if days < limit:
            return word
    return "years"


def fnum(s):
    return float(s.replace(",", ""))


def parse(header):
    """The figures in one header, or None if it carries no held-out score."""
    m = re.search(rf"held-out loss {NUM} \+/-{NUM} nats\s+{NUM} bits/char", header)
    if not m:
        return None
    d = {"val": fnum(m.group(1)), "se": fnum(m.group(2)), "bpb": fnum(m.group(3))}
    m = re.search(rf"step {NUM}\s+{NUM}M of {NUM}M characters", header)
    if m:
        d["step"], d["chars"], d["total"] = fnum(m.group(1)), fnum(m.group(2)), fnum(m.group(3))
    m = re.search(rf"{NUM} experts", header)
    if m:
        d["experts"] = int(fnum(m.group(1)))
    m = re.search(rf"train loss {NUM}", header)
    if m:
        d["train"] = fnum(m.group(1))
    dom = {}
    for line in header.splitlines():
        if line.startswith("  ") and "  " in line.strip():
            for k, v in re.findall(rf"([a-z_]+) {NUM}", line):
                dom[k] = fnum(v)
    d["dom"] = dom
    return d


def checks():
    """Every scored round with its figures, held-out on the level checks
    report at now: [(header, body, figures)], oldest first.

    Checks moved to blocks spread over the whole held-out set partway through
    the run (minagi/heldout.py). A check from before read the top of each
    subject's files, and is moved onto the new level by the difference the
    first spread session start measured, subject by subject, with "shifted"
    set in its figures. The larger evaluation a session ended on before then
    was neither measurement and is left out: only a check writes the rate
    controller's evidence line. After the change that evaluation is the level
    itself, and counts. Everything before held-out counted whole blocks only
    moves again, by what the session that switched measured.
    memory/corrections.md says all this in words, with the numbers.
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from minagi import heldout
    text = read("runs/samples.txt")
    first = heldout.first_shift(text)
    whole = heldout.part_shift(text)
    parts = text.split(BAR)
    out, had_check, at = [], False, len(parts[0]) + len(BAR)
    for i in range(1, len(parts) - 1, 2):
        h, b, start = parts[i], parts[i + 1], at
        at += len(h) + len(b) + 2 * len(BAR)
        d = parse(h)
        if first is None or start < first[0]:
            check = heldout.is_check(h)
            if had_check and not check:
                continue
            had_check = had_check or check
            if d and first is not None:
                d["val"], dom = heldout.moved(d["val"], d["dom"], first[1])
                d["dom"] = dom or d["dom"]
                d["bpb"] = d["val"] / LN2
                d["shifted"] = True
        if d and whole is not None and start < whole[0]:
            # and before held-out counted whole blocks only, by what that
            # change measured
            d["val"], dom = heldout.moved(d["val"], d["dom"], whole[1])
            d["dom"] = dom or d["dom"]
            d["bpb"] = d["val"] / LN2
            d["shifted"] = True
        if d:
            out.append((h, b, d))
    return out


def measured(latest):
    """Where the model is, as tools/measure_heldout.py measured it on
    weights/ as it is now: the latest round's figures with held-out replaced
    by that measurement. None when it was taken on an earlier state of the
    weights, and the table carries the latest check instead."""
    try:
        m = json.loads(read("runs/heldout_measured.json"))
        step = json.loads(read("weights/manifest.json")).get("step")
    except (FileNotFoundError, ValueError):
        return None
    if m.get("step") != step:
        print(f"  runs/heldout_measured.json is from step {m.get('step')}, "
              f"weights/ from step {step} - the table carries the latest check")
        return None
    if not m.get("whole"):
        print("  runs/heldout_measured.json counted each file's leftover "
              "part-block - the table carries the latest check")
        return None
    return dict(latest, val=m["val"], se=m["se"], bpb=m["val"] / LN2,
                dom=dict(m["per_subject"]),
                chars=(m.get("chars") or latest["chars"] * 1e6) / 1e6,
                experts=m.get("experts") or latest.get("experts"), measured=m)


# Published enwik9 results for models of about this one's size. The papers
# give a raw compression rate - compressed size / original size - and bits
# per byte is that rate times 8; each entry keeps the figure as printed and
# where it is printed, and the README shows both. `data` and `cost` are what
# each model was trained on and roughly what that cost when it came out -
# None where its makers never said; the README says where each comes from.
# `seen` is the bytes it was trained on where its makers published that - at
# 4 bytes a token, as the scaling chart converts every subword count - and
# only those are drawn on the scaling chart: an estimate would put a point
# where nobody knows the model was.
ENWIK9 = [
    {"name": "Llama-3.2-1B", "params": "1.23B", "published": "Sep 2024",
     "data": "9T tokens (~36 TB)", "cost": 925_000, "cost_text": "~$0.9M",
     "seen": 9e12 * 4, "window": "2,048 tokens, about 8 KB,",
     "cite": "Tsai, *Revisiting Data Compression with Language Modeling*, "
             "Jan 2026",
     "short": "Tsai", "url": "https://arxiv.org/abs/2601.02875",
     "where": 'Table 2, row "Llama-3.2-1B", quantization "None", raw '
              'compression rate',
     "printed": "0.085", "rate": 0.085},
    {"name": "Chinchilla 1B", "params": "1B", "published": "Mar 2022",
     "data": "~1.4T tokens (~5.6 TB, est.)", "cost": 68_000,
     "cost_text": "~$68k (est.)",
     "cite": "Delétang et al., *Language Modeling Is Compression*, Sep 2023",
     "short": "Delétang et al.", "url": "https://arxiv.org/abs/2309.10668",
     "where": 'Table 1, row "Chinchilla 1B", column enwik9 under "Raw '
              'Compression Rate (%)"',
     "printed": "11.3%", "rate": 0.113},
]

# Models measured here the same way as this one - the GPT-2 paper reports
# enwik8, not enwik9: tools/measure_enwik9_gpt2.py, on the pieces
# tools/measure_enwik9.py reads.
ENWIK9_MEASURED = [
    {"name": "GPT-2 XL", "params": "1.56B", "published": "Feb 2019",
     "data": "40 GB, passes not published", "cost": 43_000,
     "cost_text": "~$43k",
     "url": "https://huggingface.co/openai-community/gpt2-xl",
     "date": "2026-10-17", "bpb": 1.0730146214983058,
     "se": 0.004740405603357087, "pieces": 2000},
    {"name": "GPT-2 Large", "params": "0.77B", "published": "Feb 2019",
     "data": "40 GB, passes not published", "cost": 21_000,
     "cost_text": "~$21k (est.)",
     "url": "https://huggingface.co/openai-community/gpt2-large",
     "date": "2026-10-17", "bpb": 1.1188700949495516,
     "se": 0.005009525758987226, "pieces": 2000},
    {"name": "GPT-2 Medium", "params": "0.35B", "published": "Feb 2019",
     "data": "40 GB, passes not published", "cost": 9_800,
     "cost_text": "~$10k (est.)",
     "url": "https://huggingface.co/openai-community/gpt2-medium",
     "date": "2026-10-17", "bpb": 1.174256766370911,
     "se": 0.005083727800548771, "pieces": 2000},
    {"name": "GPT-2 Small", "params": "0.12B", "published": "Feb 2019",
     "data": "40 GB, passes not published", "cost": 3_400,
     "cost_text": "~$3k (est.)",
     "url": "https://huggingface.co/openai-community/gpt2",
     "date": "2026-10-17", "bpb": 1.328245863141902,
     "se": 0.005759209552959587, "pieces": 2000},
]

# What an hour of this model's training is costed at: the cheapest rental of
# an RTX 3070 that no other job can interrupt - Vast.ai's, on
# computeprices.com on 17 Oct 2026 - a desktop card, faster than the RTX 3070
# Laptop it trains on. Salad Cloud's $0.04 is its lowest priority tier, which
# any better-paying job takes the card from.
GPU_HOUR = 0.081

# Published bits per byte on PG19's test split, as the papers print them.
# Yu et al.'s three are from their Table 2; the Transformer and PerceiverAR
# there are baselines they trained for the MEGABYTE paper. MambaByte is shown
# by its best model, the 972M: Wang et al.'s Table 5 gives it 0.883 after 75B
# of the 150B bytes it was trained on (Appendix C); their compute-matched
# 353M scores 0.930 (Table 2). Each was trained on PG19's own training books,
# `bytes` of them, and scored in windows of `context` bytes. The scaling
# chart draws the same four (tools/plot_scaling.py's BYTE); a test holds the
# two lists to each other.
#
# Neither paper says what its training cost, so the README makes a
# best-effort estimate of what it cost its authors: 6 x `per_byte`, the
# parameters each byte runs through, x the bytes trained - MegaByte's 262M
# local model on every byte and its 758M global model once per 8, which Yu et
# al. gave all three of their models as one compute budget, and every one of
# MambaByte's 972M - on A100s at 40% utilisation, as the enwik9 notes assume
# for Chinchilla, at what AWS charged on demand for an A100 when the papers
# came out: $32.77 an hour for a p4d.24xlarge, eight A100s, the same on its
# pricing page in May 2023, December 2023 and April 2024 (AWS_PRICED, in the
# Internet Archive). Lambda rented an A100 for $1.10 in May 2023 (LAMBDA),
# about a quarter of that, and the notes say so.
A100_FLOPS, UTILISATION = 312e12, 0.4               # bf16, dense
AWS_A100 = 32.77 / 8
AWS_PRICED = [
    ("May 2023", "https://web.archive.org/web/20230518023128/"
                 "https://aws.amazon.com/ec2/instance-types/p4/"),
    ("December 2023", "https://web.archive.org/web/20231221084854/"
                      "https://aws.amazon.com/ec2/instance-types/p4/"),
    ("April 2024", "https://web.archive.org/web/20240402175315/"
                   "https://aws.amazon.com/ec2/instance-types/p4/")]
LAMBDA = (1.10, "https://web.archive.org/web/20230512094749/"
                "https://lambdalabs.com/service/gpu-cloud")
MEGABYTE_PER_BYTE = 262e6 + 758e6 / 8
PG19 = [
    {"name": "MambaByte-972M", "params": "0.97B",
     "published": "Jan 2024", "bytes": 75e9, "context": 8192,
     "bpb": 0.883, "per_byte": 972e6,
     "cite": "Wang et al., *MambaByte*, Jan 2024", "short": "Wang et al.",
     "url": "https://arxiv.org/abs/2401.13660"},
    {"name": "MegaByte-758M+262M", "params": "1.02B",
     "published": "May 2023", "bytes": 80e9, "context": 8192,
     "bpb": 1.000, "per_byte": MEGABYTE_PER_BYTE,
     "cite": "Yu et al., *MEGABYTE*, May 2023", "short": "Yu et al.",
     "url": "https://arxiv.org/abs/2305.07185"},
    {"name": "Transformer-320M", "params": "0.32B",
     "published": "May 2023", "bytes": 80e9, "context": 1024,
     "bpb": 1.057, "per_byte": MEGABYTE_PER_BYTE,
     "cite": "Yu et al., *MEGABYTE*, May 2023", "short": "Yu et al.",
     "url": "https://arxiv.org/abs/2305.07185"},
    {"name": "PerceiverAR-248M", "params": "0.25B",
     "published": "May 2023", "bytes": 80e9, "context": 8192,
     "bpb": 1.104, "per_byte": MEGABYTE_PER_BYTE,
     "cite": "Yu et al., *MEGABYTE*, May 2023", "short": "Yu et al.",
     "url": "https://arxiv.org/abs/2305.07185"},
]


def a100_hours(e):
    """The A100-hours a published PG19 model's training is estimated at."""
    return 6 * e["per_byte"] * e["bytes"] / (A100_FLOPS * UTILISATION) / 3600


def enwik9_known():
    """The published enwik9 scores whose models' training data is known, as
    (name, bytes trained on, bits per byte, how much it was scored on at a
    time or None) - what the scaling chart can place."""
    return [(e["name"], e["seen"], e.get("bpb") or e["rate"] * 8,
             e.get("window"))
            for e in ENWIK9 + ENWIK9_MEASURED if e.get("seen")]


def dollars(x):
    """A rough cost, two significant figures."""
    x = float(f"{x:.2g}")
    if x >= 1e6:
        return f"~${x / 1e6:g}M"
    if x >= 1e3:
        return f"~${x / 1e3:g}k"
    return f"~${x:,.0f}"


def enwik9_table(m, total, hours, dyn=None, natural=None):
    """The enwik9 table and the notes under it, from this model's measurement
    `m` (runs/enwik9_measured.json), its parameter count and the hours it
    has trained: best first, the best score and the cheapest training in
    bold, every published score traced back to the figure its paper prints,
    and the notes folded away under the table. `dyn` is a measurement of it
    learning as it reads (tools/measure_enwik9_adapt.py), and `natural` of it
    reading whole pages in its own window as it reads anything
    (tools/measure_enwik9_adapt.py --natural): each its own row, marked, and
    what the mark means straight under the table."""
    import time as _t

    def day(d):
        return _t.strftime("%-d %b %Y", _t.strptime(d[:10], "%Y-%m-%d"))
    when = day(m["date"])
    cost = hours * GPU_HOUR
    rows = [dict(e, model=e["name"], src=f"[{e['cite']}]({e['url']})",
                 bpb=e["rate"] * 8, paper=e) for e in ENWIK9]
    rows += [dict(e, model=f"[{e['name']}]({e['url']})", src="measured",
                  paper=None) for e in ENWIK9_MEASURED]
    # frozen is not how this model runs - it learns from everything it
    # reads - so its frozen score says so
    rows.append({"model": "mini-AGI (frozen)",
                 "params": f"{total / 1e9:.2f}B",
                 "published": when, "data": f"{m['chars'] / 1e9:.1f} GB",
                 "cost": cost, "cost_text": dollars(cost),
                 "src": "measured", "bpb": m["bpb"], "paper": None})
    if dyn:
        rows.append({"model": "mini-AGI (dynamic evaluation)\\*",
                     "params": f"{total / 1e9:.2f}B",
                     "published": day(dyn["date"]),
                     "data": f"{dyn['chars'] / 1e9:.1f} GB", "cost": cost,
                     "cost_text": dollars(cost), "src": "measured",
                     "bpb": dyn["adapted_bpb"], "paper": None})
    if natural:
        rows.append({"model": "mini-AGI (best effort)\\*\\*",
                     "params": f"{total / 1e9:.2f}B",
                     "published": day(natural["date"]),
                     "data": f"{natural['chars'] / 1e9:.1f} GB",
                     "cost": cost, "cost_text": dollars(cost),
                     "src": "measured", "bpb": natural["adapted_bpb"],
                     "paper": None})
    rows.sort(key=lambda r: r["bpb"])
    best = rows[0]["bpb"]
    cheapest = min(r["cost"] for r in rows if r["cost"] is not None)
    bold = lambda text, yes: f"**{text}**" if yes else text
    lines = ["| model | params | date | trained on | training cost ↓ | "
             "scored in | enwik9 bits/byte ↓ |",
             "|---|---|---|---|---|---|---|"]

    def row(r):
        cost = (bold(r["cost_text"], r["cost"] == cheapest)
                if r["cost"] is not None else "not published")
        score = bold(f"{r['bpb']:.2f}", r["bpb"] == best)
        return (f"| {r['model']} | {r['params']} | {r['published']} | "
                f"{r['data'] or 'not published'} | {cost} | {r['src']} | "
                f"{score} |")
    lines += [row(r) for r in rows]
    notes = [
        "**Date** is when each model came out: Llama-3.2-1B's release, the "
        "Chinchilla paper (its weights were never released), and the GPT-2 "
        "paper (the Small weights came with it; Medium, Large and XL "
        "followed in May, Aug and Nov 2019); mini-AGI's is the "
        "day these weights were measured, as it is still training.", "",
        "**Trained on** is how much text each model learned from, and "
        "**training cost** roughly what that training cost when it came "
        "out:", "",
        "- Llama-3.2-1B: up to 9 trillion tokens and 370k H100 GPU-hours, "
        "both from [its model card]"
        "(https://huggingface.co/meta-llama/Llama-3.2-1B). 9 trillion tokens "
        "is about 36 TB of text at roughly 4 bytes a token, and the cost is "
        "those hours at about $2.5 each, a typical cloud price in late 2024. "
        "It leaves out the Llama 3.1 8B and 70B models whose outputs it was "
        "also trained on.",
        "- Chinchilla 1B: DeepMind has not published how much it was trained "
        "on or what that cost, so both are a best-effort estimate, which "
        "assumes it read as much as Chinchilla 70B did ([Hoffmann et al.]"
        "(https://arxiv.org/abs/2203.15556)): 1.4T tokens, about 5.6 TB of "
        "text, and 8.4×10<sup>21</sup> FLOPs for 1B parameters - about "
        "21,000 chip-hours of TPU v4 at 40% utilisation, at $3.22 a "
        "chip-hour, its price in 2022. For scale, EleutherAI's Pythia-1B "
        "took 4,830 A100-hours to read 300B tokens ([Biderman et al.]"
        "(https://arxiv.org/abs/2304.01373), Table 5), about 22,500 for "
        "1.4T. It may have read less: the Chinchilla paper's compute-optimal "
        "amount for 1B parameters is 20.2B tokens (Table 3), which would "
        "have cost about 70 times less.",
        "- GPT-2 XL: [WebText](https://en.wikipedia.org/wiki/GPT-2), 40 GB of "
        "text from 8 million web pages; OpenAI has not said how many times "
        "it went over it. The cost is Andrej Karpathy's estimate, quoted "
        "there: 32 TPU v3 chips for 7 days at about $8 per chip-hour.",
        "- GPT-2 Large, Medium and Small: the same WebText. Their costs are "
        "best-effort estimates: Karpathy's figure for GPT-2 XL scaled by "
        "parameter count (0.77B, 0.35B and 0.12B ÷ 1.56B), which assumes "
        "the same data and the same hardware.",
        f"- mini-AGI: everything it has read so far, and its {hours:,.0f} "
        f"hours on one RTX 3070 Laptop GPU at ${GPU_HOUR:.3f} an hour - the "
        "cheapest rental of an RTX 3070 that no other job can interrupt, "
        "Vast.ai's on [computeprices.com]"
        "(https://computeprices.com/providers/vast/gpus/rtx3070) on 17 Oct "
        "2026, for a desktop card faster than this laptop's. Salad Cloud "
        "rents one for $0.04, but only in its lowest priority tier, which "
        "any better-paying job can take the card from.", "",
        "**Bits per byte** - lower is better - is how many bits a model "
        "needs, on average, to encode each byte of the text: the total "
        "-log<sub>2</sub> of the probabilities it gave the text, divided by "
        "the text's length in bytes. The papers report a **compression "
        "rate** instead - compressed size ÷ original size - and bits per "
        "byte is that rate × 8:", ""]
    notes += [f"- {r['name']}: [{r['short']}]({r['url']}), {r['where']}: "
              f"**{r['printed']}**, and {r['rate']:g} × 8 = {r['bpb']:.2f}."
              for r in rows if r["paper"] is not None]
    ms = ENWIK9_MEASURED
    names = [e["name"] for e in ms]
    names = (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1
             else names[0])
    same = all(e["pieces"] == m["pieces"] for e in ms)
    ses = [f"{e['se']:.3f}" for e in ms]
    off = (f"± {ses[0]}" + (" for each" if len(ms) > 1 else "")
           if len(set(ses)) == 1 else
           "± " + ", ".join(f"{s} for {e['name']}" for s, e in zip(ses, ms)))
    notes += [
        f"- {names} " + ("are" if len(ms) > 1 else "is") + " measured here "
        "the same way as mini-AGI - the GPT-2 paper reports enwik8, not "
        "enwik9 - on " + ("the same " if same else "")
        + f"{ms[0]['pieces']:,} pieces of the same slice: each cut back to "
        "whole UTF-8 characters at its two ends, tokenized, and started from "
        "`<|endoftext|>`, at full precision. Bits per byte is the total "
        "-log<sub>2</sub> probability of the tokens ÷ the pieces' bytes, "
        + off + "."]
    notes += [
        "- mini-AGI (frozen) is measured directly, the same quantity without "
        "running an arithmetic coder: its loss on each byte, converted from "
        "nats to bits (÷ ln 2) and averaged, with its weights frozen - it "
        "learns nothing from the pieces while they are scored. It is "
        "measured the way Chinchilla 1B was - enwik9 cut into pieces of "
        "2,048 bytes, each predicted with no memory of the others - on "
        f"{m['pieces']:,} pieces spread over a "
        f"{m['slice_bytes'] / 1e6:.0f} MB slice of enwik9 that it never "
        "trained on (its Wikipedia comes from the same 2006 dump, without "
        f"these pages), ± {m['se']:.3f}. Llama-3.2-1B was given 2,048 "
        "tokens, about 8 KB, at a time."]
    star = ""
    if dyn:
        q = dyn["gain_by_quarter"]
        notes += [
            f"- mini-AGI (dynamic evaluation) reads the same "
            f"{dyn['pieces']:,} pieces in the same order, as one stream it "
            "goes on learning from: each piece is scored alone, exactly as "
            "in the frozen measurement, and then the model takes one "
            f"optimiser step on it {rate_words(dyn)}, with "
            "Adam's moments as training left them. Learning is the only "
            "difference: read frozen, the same weights score "
            f"{dyn['frozen_bpb']:.3f}, and learning took "
            f"{dyn['gain_bpb']:.3f} ± {dyn['gain_se']:.4f} off - "
            f"{q[0]:.3f} over the first quarter of the pieces and "
            f"{q[-1]:.3f} over the last, so it was still gaining at the "
            "end."]
        star = (
            "\n\n\\* **Learning while it reads**, known as dynamic "
            "evaluation: each piece is scored first and then learned from, "
            "so no byte is scored by weights that have already seen it. "
            f"Read frozen, the same weights score {dyn['frozen_bpb']:.2f}. "
            "The learning is done on a copy of the weights, which is thrown "
            "away after the measurement, so nothing learned from enwik9 ever "
            "reaches the weights the run goes on training, or any later "
            "score. "
            "For mini-AGI this is simply how it runs: it is a continual "
            "learning model, and this is the learning it does in training "
            "and in every conversation, on the same code path and at its "
            "own rate" + (", which its controller sets as it does in "
                          "training" if dyn.get("checks") is not None
                          else "") + ". The frozen models above would need a separate "
            "fine-tuning loop added to be scored this way ([Krause et al., "
            "2017](https://arxiv.org/abs/1709.07432)), so this row is closer "
            "to online compressors that learn as they compress, such as "
            "NNCP and cmix, than to the frozen rows.")
    if natural:
        every = whole(natural)
        notes += [
            f"- mini-AGI (best effort) reads whole pages, as it reads "
            "anything: "
            + (f"every one of the slice's {natural['pages']:,} pages, in "
               "the order the slice holds them," if every else
               f"{natural['pages']:,} of the slice's pages, spread evenly "
               "over it and")
            + " framed as the Wikipedia files it trains on "
            "are - one page from `<page>` to `</page>` - each from its top "
            f"in its {natural['context']:,}-byte window sliding by "
            f"{natural['chunk']:,} characters, {natural['bytes'] / 1e6:.2f}M "
            "characters in all, every chunk scored first and then learned "
            f"from {rate_words(natural)}"
            + (", so what it learns from one page carries into every page "
               "after it" if every else "")
            + ". Read frozen in the same windows, the same "
            f"pages score {natural['frozen_bpb']:.3f}, and chunk by chunk "
            f"alone {natural['alone_bpb']:.3f}."]
        star += (
            "\n\n\\*\\* **Best effort**: the model used exactly as it is "
            "used in training and in conversation - "
            + ("every page of the slice, whole, " if every else
               "whole pages ")
            + f"in its native {natural['context']:,}-byte window, sliding "
            f"{natural['chunk']:,} characters at a time, learning as it goes "
            "on a copy that is thrown away afterwards, as above"
            + (" - every chunk scored before it is learned from, and what it "
               "learns carried into everything it reads after" if every
               else "")
            + ". This is its "
            "most natural score, but it sees more of each page at once than "
            "the 2,048-byte pieces the other mini-AGI rows, GPT-2 and "
            "Chinchilla 1B are scored on, so it is not on the same footing "
            "as them.")
    return ("\n".join(lines) + star + "\n\n<details>\n<summary>How these "
            "numbers were worked out</summary>\n\n" + "\n".join(notes)
            + "\n\n</details>")


def enwik9_dynamic():
    """The latest measurement of this model learning as it reads enwik9 -
    tools/measure_enwik9_adapt.py on the pieces tools/measure_enwik9.py
    scores - or None."""
    try:
        lines = read("runs/enwik9_adapt.jsonl").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if (d.get("piece_bytes") == 2048 and d.get("pieces", 0) >= 2000
                and d.get("weights_unchanged")):
            return d
    return None


def enwik9_natural():
    """The latest measurement of this model reading whole enwik9 pages in its
    own window, learning as it reads - tools/measure_enwik9_adapt.py
    --natural, on as many bytes of pages as the frozen measurement's pieces
    hold - or None."""
    return natural_latest("runs/enwik9_natural.jsonl")


def pg19_natural():
    """The same for whole PG19 books (measure_benchmarks.sh), or None."""
    return natural_latest("runs/scaling/pg19_natural.jsonl")


def whole(natural):
    """Whether a --natural measurement read every page of its slice - every
    book, on PG19 - rather than an even spread of them."""
    return bool(natural.get("pages_in_slice")
                and natural.get("pages") == natural.get("pages_in_slice"))


def rate_words(d):
    """The rate a measurement that learned stepped at, in the notes' words:
    where the run had brought it, and - where its controller went on working
    as it does in training (tools/measure_enwik9_adapt.py records `checks`)
    - where the held-out checks took it."""
    def sci(x):
        mant, ex = f"{x:.1e}".split("e")
        return f"{mant}×10<sup>{int(ex)}</sup>"
    if d.get("lr") is None:
        return "at the rate its training had come down to"
    said = (f"at {sci(d['lr'])} - its base rate times the factor its "
            f"training had come down to")
    if d.get("checks") is None:
        return said
    if not d["checks"]:
        return (said + f", with its controller working as in training - no "
                f"held-out check, one every {d['check_every']:g} minutes, "
                f"came due")
    return (said + f", then set by its controller as in training, from a "
            f"held-out check every {d['check_every']:g} minutes - "
            f"{d['checks']:,} of them, ending at {sci(d['lr_end'])}")


def natural_latest(path):
    """The latest --natural measurement in `path`, of the standard size and
    with the weights left unchanged, or None."""
    try:
        lines = read(path).splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if (d.get("regime") == "natural" and d.get("weights_unchanged")
                and d.get("budget", 0) >= 2000 * 2048):
            return d
    return None


def enwik9(out):
    """The README's enwik9 region, rebuilt from tools/measure_enwik9.py's
    measurement when it was taken on weights/ as it is; left as it was
    otherwise."""
    try:
        m = json.loads(read("runs/enwik9_measured.json"))
        man = json.loads(read("weights/manifest.json"))
    except (FileNotFoundError, ValueError):
        print("  no enwik9 measurement - the enwik9 table is left as it was")
        return out, True
    if m.get("step") != man.get("step"):
        print(f"  runs/enwik9_measured.json is from step {m.get('step')}, "
              f"weights/ from step {man.get('step')} - the enwik9 table is "
              f"left as it was")
        return out, True
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from minagi.store import count_params
        total = count_params(os.path.join(ROOT, "weights"))["total"]
    except Exception as e:                          # noqa: BLE001
        print(f"  weights/ could not be counted ({e}) - the enwik9 table is "
              f"left as it was")
        return out, False
    pace, window, ran = running()                  # every session's hours
    return region(out, "enwik9", enwik9_table(m, total, ran / 3600,
                                              enwik9_dynamic(),
                                              enwik9_natural()))


def size_at(p, experts):
    """The parameter count the model had with `experts` experts, from
    count_params's `p` of it now: each expert gained since taken off with its
    row of the routers. Nothing else in it changes size."""
    each = p["per_expert"] + p["routers"] / p["n_experts"]
    return p["total"] - (p["n_experts"] - experts) * each


def pg19_table(pg, natural, size, hours):
    """The PG19 table and the notes under it: the published byte-level models
    beside this model read frozen over the whole test split
    (tools/eval_pg19.py's `pg`) and, once it has been measured, its best
    effort - whole books in its own window, learning as it reads
    (tools/measure_enwik9_adapt.py --natural, `natural`). Best first, the
    best score and the cheapest training in bold. A mini-AGI row is of the
    weights it was measured on: its date and reading are its measurement's,
    its size `size(experts)` at the pool it had then, and its cost the
    `hours(step)` it had trained for by then."""
    import time as _t

    def day(d):
        return _t.strftime("%-d %b %Y", _t.strptime(d[:10], "%Y-%m-%d"))

    def sci(x):
        mant, ex = f"{x:.1e}".split("e")
        return f"{mant}×10<sup>{int(ex)}</sup>"

    def mine(name, d, chars, ctx, bpb):
        h = hours(d["step"])
        return {"model": name, "params": f"{size(d['experts']) / 1e9:.2f}B",
                "published": day(d["date"]),
                "data": f"{chars / 1e9:.1f} GB, none of it PG19",
                "cost": h * GPU_HOUR, "cost_text": dollars(h * GPU_HOUR),
                "hours": h, "ctx": f"{ctx:,}", "src": "measured", "bpb": bpb}
    rows = [dict(e, model=e["name"], src=f"[{e['cite']}]({e['url']})",
                 data=f"{e['bytes'] / 1e9:.0f} GB of PG19",
                 cost=a100_hours(e) * AWS_A100,
                 cost_text=dollars(a100_hours(e) * AWS_A100) + " (est.)",
                 ctx=f"{e['context']:,}") for e in PG19]
    rows.append(mine("mini-AGI (frozen)", pg, pg["chars_read"], pg["window"],
                     pg["bpb"]))
    if natural:
        rows.append(mine("mini-AGI (best effort)\\*", natural,
                         natural["chars"], natural["context"],
                         natural["adapted_bpb"]))
    rows.sort(key=lambda r: r["bpb"])
    best = rows[0]["bpb"]
    cheapest = min(r["cost"] for r in rows)
    bold = lambda text, yes: f"**{text}**" if yes else text  # noqa: E731
    lines = ["| model | params | date | trained on | training cost ↓ | "
             "context, bytes | scored in | PG19 bits/byte ↓ |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['params']} | {r['published']} | "
              f"{r['data']} | {bold(r['cost_text'], r['cost'] == cheapest)} | "
              f"{r['ctx']} | {r['src']} | "
              f"{bold(format(r['bpb'], '.3f'), r['bpb'] == best)} |"
              for r in rows]
    gb = sorted({e["bytes"] / 1e9 for e in PG19})
    under = (
        f"The published models were trained on PG19's own training books, "
        f"{gb[0]:.0f} to {gb[-1]:.0f} GB of them, and are scored on more "
        f"books like those. mini-AGI has never been trained on PG19 and has "
        f"read no Victorian novels, so for it these books are out of its "
        f"domain.")
    if natural:
        under += (
            f"\n\n\\* **Best effort**: the model used exactly as it is used "
            f"in training and in conversation - whole books, each from its "
            f"top, in its native {natural['context']:,}-byte window sliding "
            f"{natural['chunk']:,} characters at a time, and learning as it "
            f"reads, as in the enwik9 table above: every chunk is scored "
            f"first and then learned from, on a copy of the weights that is "
            f"thrown away afterwards. "
            + (f"It reads all {natural['pages']} books of the split, the "
               f"frozen row's books, one after another, so what it learns "
               f"from one book carries into every book after it."
               if whole(natural) else
               f"It reads {natural['pages']} whole books spread over the "
               f"split, where the frozen row reads all {pg['books']}."))
    wang, yu = PG19[0], PG19[1]
    mine_rows = [r for r in rows if r["src"] == "measured"]
    mine_rows.sort(key=lambda r: r["model"] != "mini-AGI (frozen)")
    notes = [
        "**Date** is when each model came out: the MEGABYTE paper for the "
        "three trained there, and the MambaByte paper; mini-AGI's is the day "
        "those weights were measured, and its size the pool it had then, as "
        "it is still training and growing.", "",
        f"**Trained on** is how much text each model learned from, and "
        f"**training cost** roughly what that training cost when it came out. "
        f"Neither paper says what its training cost or what it ran on, so the "
        f"published costs are best-effort estimates of what their authors "
        f"paid: the training compute by the 6ND rule - 6 × the parameters "
        f"each byte runs through × the bytes trained on - on A100s at "
        f"{UTILISATION:.0%} utilisation, as the enwik9 notes assume for "
        f"Chinchilla, at what AWS charged on demand for an A100 when the "
        f"papers came out: ${AWS_A100 * 8:.2f} an hour for a machine with "
        f"eight, ${AWS_A100:.2f} each, the same on its pricing page in "
        + ", ".join(f"[{when}]({url})" for when, url in AWS_PRICED[:-1])
        + f" and [{AWS_PRICED[-1][0]}]({AWS_PRICED[-1][1]}). Smaller clouds "
        f"rented A100s for less - [Lambda]({LAMBDA[1]}) for "
        f"${LAMBDA[0]:.2f} in May 2023, about a quarter as much - so the "
        f"real cost may have been lower:", "",
        f"- MegaByte-758M+262M, Transformer-320M and PerceiverAR-248M: "
        f"{yu['bytes'] / 1e9:.0f} billion bytes of PG19's training books each "
        f"([{yu['short']}]({yu['url']}), §5) - about eight passes over PG19, "
        f"which is 10.1 GB in all (their Table 1) - on one compute budget the "
        f"three share. MegaByte runs about {yu['per_byte'] / 1e6:.0f}M "
        f"parameters a byte - its 262M local model on every byte and its 758M "
        f"global model once per 8 - so that is "
        f"{sci(6 * yu['per_byte'] * yu['bytes'])} FLOPs, about "
        f"{float(f'{a100_hours(yu):.2g}'):,.0f} A100-hours.",
        f"- MambaByte-972M: {wang['bytes'] / 1e9:.0f} billion bytes when it "
        f"scored this, of the 150 billion it went on to - 380,000 steps of "
        f"48 × 8,192 ([{wang['short']}]({wang['url']}), Appendix C). All "
        f"{wang['per_byte'] / 1e6:.0f}M of its parameters run on every "
        f"byte, so that is {sci(6 * wang['per_byte'] * wang['bytes'])} "
        f"FLOPs, about {float(f'{a100_hours(wang):.2g}'):,.0f} A100-hours.",
        f"- mini-AGI: everything it had read when measured, from its own "
        f"corpus with no PG19 in it, and the hours it had trained for by then "
        f"on one RTX 3070 Laptop GPU at ${GPU_HOUR:.3f} an hour, priced as in "
        f"the enwik9 table above - the cheapest rental of that card, which no "
        f"big cloud offers: "
        + " and ".join(f"{r['hours']:,.0f} hours for the "
                       + ("frozen row" if r["model"] == "mini-AGI (frozen)"
                          else "best effort") for r in mine_rows) + ".", "",
        f"**Context** is how many bytes each model reads at once: 1,024 for "
        f"the Transformer, 8,192 for the other published models, and "
        f"{pg['window']:,} for mini-AGI, the window it was built with.", "",
        f"**Bits per byte** - lower is better - is printed by both papers. "
        f"[{yu['short']}]({yu['url']}) give 1.057 for the Transformer, 1.104 "
        f"for PerceiverAR and 1.000 for MegaByte, in the PG-19 column of "
        f"their Table 2. [{wang['short']}]({wang['url']}) give "
        f"{wang['bpb']:.3f} for MambaByte-972M in their Table 5, where they "
        f"time how fast it writes, and 0.863 for the same model scored with "
        f"a sliding window; the table shows the first. MambaByte is shown by "
        f"its best model only: the smaller MambaByte-353M, which Wang et al. "
        f"train on the same compute per byte as MegaByte-758M+262M, scores "
        f"0.930 after 30 billion bytes (their Table 2).", "",
        "mini-AGI's " + ("two rows are" if natural else "row is")
        + " measured here:", "",
        f"- mini-AGI (frozen) is measured by tools/eval_pg19.py over every "
        f"book of the test split - {pg['books']} books, "
        f"{pg['bytes_scored']:,} bytes - in consecutive windows of "
        f"{pg['window']:,} bytes, each read with no memory of the one before "
        f"and with the weights frozen. Its loss on each byte is converted "
        f"from nats to bits (÷ ln 2) and pooled by byte: total bits ÷ total "
        f"bytes."]
    if natural:
        notes += [
            f"- mini-AGI (best effort) is measured by "
            f"tools/measure_enwik9_adapt.py --natural on "
            + (f"all {natural['pages']} books of the split, in order, "
               if whole(natural) else
               f"{natural['pages']} whole books spread evenly over the "
               f"split, ")
            + f"{natural['bytes'] / 1e6:.2f}M bytes, learning "
            f"{rate_words(natural)}. Read frozen in the same windows, "
            f"the same books score {natural['frozen_bpb']:.3f}, and chunk by "
            f"chunk alone {natural['alone_bpb']:.3f}."]
    return ("\n".join(lines) + "\n\n" + under + "\n\n<details>\n<summary>"
            "How these numbers were worked out</summary>\n\n"
            + "\n".join(notes) + "\n\n</details>")


def pg19(out):
    """The README's PG19 region, from tools/eval_pg19.py's measurement over
    the whole test split and the best effort's, when there is one. A row
    carries the date, size and training cost of the weights it was measured
    on, so an older measurement is shown as what it is rather than held back;
    with no measurement of the whole split the region is left as it was."""
    try:
        pg = json.loads(read("runs/scaling/pg19.json"))
    except (FileNotFoundError, ValueError):
        print("  no PG19 measurement - the PG19 table is left as it was")
        return out, True
    if pg.get("fraction", 0) < 1 or not pg.get("date") or not pg.get("experts"):
        print("  runs/scaling/pg19.json is not a dated measurement of the "
              "whole split - the PG19 table is left as it was")
        return out, True
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from minagi.store import count_params
        p = count_params(os.path.join(ROOT, "weights"))
    except Exception as e:                          # noqa: BLE001
        print(f"  weights/ could not be counted ({e}) - the PG19 table is "
              f"left as it was")
        return out, False
    return region(out, "pg19", pg19_table(pg, pg19_natural(),
                                          lambda n: size_at(p, n),
                                          lambda st: ran_until(st) / 3600))


def region(text, name, body):
    """Replace what is between the markers, or say the markers are missing."""
    a, b = f"<!-- auto:{name} -->", f"<!-- /auto:{name} -->"
    pat = re.compile(re.escape(a) + r".*?" + re.escape(b), re.S)
    if not pat.search(text):
        print(f"  MISSING marker pair in README.md: {a} ... {b}")
        return text, False
    return pat.sub(lambda _: f"{a}\n{body}\n{b}", text, count=1), True


def parameters(out):
    """The parameter table and the paragraph under it, from the weights."""
    # ---- the parameter count ------------------------------------------------
    # Counted from the files the run writes (minagi.store.count_params), the
    # same count the scaling chart reads, never typed: the pool grows and
    # prunes itself, and a table typed at 175 experts was still claiming them
    # when there were 150. The run writes its files atomically, so reading
    # them while it trains is safe.
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from minagi.store import count_params, param_table
        p = count_params(os.path.join(ROOT, "weights"))
    except Exception as e:                          # noqa: BLE001
        print(f"  weights/ could not be counted ({e}) - the parameter table "
              f"is left as it was")
        return out, False
    else:
        tot, res = p["total"] / 1e6, p["resident_params"] / 1e6
        act = p["dense_equiv"] / 1e6
        body = (
            "The parameter count moves, because the pool grows and prunes "
            "itself while training. `python3 -m minagi.store weights` prints "
            f"what it is now. At step {p['step']:,}:\n\n"
            "```\n" + "\n".join(param_table(p)) + "\n```\n\n"
            f"**VRAM is set by the card's {p['resident']} slots, not by the "
            f"pool.** Only {p['resident']} experts are resident at a time - "
            f"about {res:.0f}M parameters of the {tot:.0f}M - which is why the "
            f"pool can keep growing on an 8 GB card. Per byte the model "
            f"activates about **{act:.0f}M** parameters - two prelude blocks, "
            f"then attention and top-{p['top_k']} of the resident experts on "
            f"each recurrent step, counted at the {p['steps']:g} steps training "
            f"samples its depth around (a character may take up to "
            f"{p['max_steps']}) - so by the 6ND rule it costs the same per byte "
            f"as a dense {act:.0f}M byte-level transformer, not a {tot:.0f}M "
            f"one. That is the figure the scaling chart in Benchmarks is drawn "
            f"against."
        )
        return region(out, "params", body)


def scaling(out, latest, trend, pg, timing=None):
    """The README's scaling region: what the chart shows, the fit it was
    drawn with, and the projection read off that fit - from runs/scaling,
    which tools/plot_scaling.py writes as it draws. Left as it was when the
    chart has not been drawn, or no session logged its time; `timing` is
    running()'s (pace, window, ran), read from the log unless given."""
    agg = trend.get("AGGREGATE", {})
    if not (agg and pg):
        return out, True
    pace, window, ran = timing or running()
    if not pace:
        print("  no session in runs/samples.txt logged its time - the "
              "projection is left as it was")
        return out, False
    al, A = agg["alpha"], agg["A"]
    D0, L0 = latest["chars"] * 1e6, latest["bpb"]

    def days_to(D):
        return (D - D0) / pace / 86400

    def row(label, D, bold=False):
        days = days_to(D)
        when = (f"~{days:,.0f}" if days >= 1
                else f"{days * 24:.0f} hours")
        if bold:
            when = f"**{when}**"
        return (f"| {label} | {D / 1e9:.2f}B | "
                f"+{(D - D0) / 1e9:.2f}B | {when} |")

    proj, ahead = [], []
    for target in (1.10, 1.00, 0.93, 0.80):
        if target >= L0:
            continue
        D = D0 * (L0 / target) ** (1 / al)
        # THREE COLUMNS, because two were ambiguous: "bytes needed" is
        # cumulative and "days" counts from here, so 0.49B beside ~1 day
        # read as "0.49B in a day" when the model has already spent a week
        # reading 0.46B of it.
        proj.append(row(f"{target:.2f} BPB", D))
        ahead.append(D)
    # ...and where one full pass over the corpus lands, which is the
    # milestone the run is actually walking toward. The BPB here is read
    # OFF the trend rather than chosen, so it is the one number in the
    # table that is a prediction rather than a target.
    Dend = latest["total"] * 1e6
    proj.append(row(f"**{L0 * (Dend / D0) ** (-al):.2f} BPB**", Dend, bold=True))
    # only the subjects the run actually scores. trend.json carries keys
    # the held-out mixture does not, and naming one of those here would
    # put a subject in the README that the tables above never mention.
    steep = sorted(((v.get("alpha", 0), k) for k, v in trend.items()
                    if k in latest["dom"]), reverse=True)
    # the sentence under the table describes the rows above it, so it is
    # worked out from them: a slower pace turns weeks into months, and a
    # target the model has passed drops out of the count
    said = ""
    if ahead:
        n = len(ahead)
        lo, hi = scale(days_to(ahead[0])), scale(days_to(ahead[-1]))
        span = lo if lo == hi else f"{lo} to {hi}"
        said = (f"The first {('one', 'two', 'three', 'four')[n - 1]} "
                f"{'is' if n == 1 else 'are'} {span} of reading on one "
                f"laptop GPU")
        if ahead[-1] <= Dend:
            said += (f", and {'it sits' if n == 1 else 'every one of them sits'}"
                     f" inside a single pass of the "
                     f"{latest['total'] / 1000:.2f}B-character corpus")
        said += ".\n\n"
    # enwik9's squares, when the chart drew them: what they are and how far
    # the dotted line carries them, in the numbers the chart was drawn with
    e = trend.get("enwik9") or {}
    e9 = ""
    if e.get("bpb") and e.get("alpha") and e.get("end_bpb"):
        mine = e.get("how") == "measured"
        how = (f"the rate enwik9 itself improved at between its {e['n']} "
               f"measurements (α {e['alpha']:.3f})" if mine else
               f"the rate this model's Wikipedia held-out fell at over its "
               f"last billion characters (α {e['alpha']:.3f})")
        e9 = (f"The green squares are this model on **enwik9**, the set the "
              f"enwik9 table above compares it with published models on: "
              f"Wikipedia, which it trains on, though never on these pages. "
              f"It scores **{e['bpb']:.3f} BPB** there after "
              f"{e['D'] / 1e9:.2f}B characters"
              + (f". A square is drawn only once it stands clear of the one "
                 f"before it, so "
                 + ("the one measured since is" if e["since"] == 1 else
                    f"the {e['since']} measured since are")
                 + f" not drawn yet; the "
                 f"latest is {e['latest_bpb']:.3f} after "
                 f"{e['latest_D'] / 1e9:.2f}B characters"
                 if e.get("since") else "")
              + f". The dotted line carries it "
              f"on to one pass of the corpus at {how}, where it reaches "
              f"**{e['end_bpb']:.2f}**"
              + ("" if mine else
                 f" - an assumption until enwik9's own measurements are "
                 f"{e.get('own_span', 1.25) - 1:.0%} apart in what the model "
                 f"has read"
                 + (f"; its {e['n']} so far are "
                    f"{100 * (e['span'] - 1):.1f}% apart"
                    if e.get("n", 1) >= 2 and e.get("span") else ""))
              + ".")
        known = enwik9_known()
        if known:
            names = " and ".join(k[0] for k in known)
            def times(seen):                 # two significant figures
                return f"{float(f'{seen / e9_read:.2g}'):,.0f}"
            e9_read = e["D"]
            far = "; ".join(
                (f"{k[0]} " if len(known) > 1 else "")
                + f"{k[2]:.2f} after about {k[1] / 1e12:,.0f} TB - about "
                f"{times(k[1])} times what this model has read"
                + (f", scored on {k[3]} at a time" if k[3] else "")
                for k in known)
            many = len(known) > 1
            e9 += (f" The hollow green circle{'s' if many else ''} far to the "
                   f"right {'are' if many else 'is'} {names}, the only "
                   f"model{'s' if many else ''} in that table whose training "
                   f"data is published: {far}. The rest are not drawn, "
                   f"because how much they read was never published.")
        e9 += "\n\n"
    body = (
        f"![Data scaling on PG19{' and enwik9' if e9 else ''}]"
        f"(assets/scaling.png)\n\n"
        + f"Everything purple on this chart is **bits-per-byte on the PG19 "
          f"test split** - one held-out set, so it compares directly: the "
          f"diamonds are this model, the hollow circles the published "
          f"byte-level models in the PG19 table above. "
        + f"This "
        f"model scores **{pg['bpb']:.3f} BPB** over the whole split "
        f"({pg['books']} books, {pg['bytes_scored']:,} bytes) at a context "
        f"of {pg['window']:,}, against its own mixture's "
        f"{pg['own_bpb']:.2f}. PG19 is out of distribution for it: it was "
        f"trained on a corpus assembled for this project and has read no "
        f"Victorian novels, so much of that gap is subject matter rather "
        f"than capability.\n\n"
        + e9 +
        f"The red line is the fitted "
        f"power law on this model's own held-out, `L ∝ D^-{al:.3f}` with "
        f"R² {agg['r2']:.2f} over every point past the warmup - between "
        f"Kaplan's 0.095 and Chinchilla's 0.28, and it has held for more "
        f"than a decade of data. How steep it looks depends on where the "
        f"fit starts, and the band on the chart spans that range.\n\n"
        f"Read straight off that trend, on this model's own mixture. It "
        f"has read {D0 / 1e9:.2f}B characters so far, in about "
        f"{duration(ran)} of running. The days below assume the pace of "
        f"the last {duration(window)} of it: {pace:,.0f} characters a "
        f"second on the wall clock, held-out checks and rounds of samples "
        f"included, because the reading waits for them.\n\n"
        f"| held-out | total data read | further reading | days from "
        f"here at ~{pace:,.0f} char/s |\n|---|---|---|---|\n"
        + "\n".join(proj) + "\n\n" + said +
        f"The right panel shows which subjects are still moving. "
        f"{', '.join(n for _, n in steep[:4])} are the steep ones; "
        f"{steep[-1][1]} and {steep[-2][1]} have the shallowest slopes."
    )
    return region(out, "scaling", body)


def main():
    scored = checks()
    if not scored:
        sys.exit("no scored entries in runs/samples.txt")
    latest = measured(scored[-1][2]) or scored[-1][2]
    best_h, best_b, best = min(scored, key=lambda t: t[2]["val"])

    trend = json.loads(read("runs/scaling/trend.json", "{}") or "{}")
    pg = json.loads(read("runs/scaling/pg19.json", "{}") or "{}")

    out = read("README.md")
    ok = True

    # ---- the two collapsible blocks ---------------------------------------
    graph = os.path.join(ROOT, "runs/training_progress.png")
    if os.path.exists(graph):
        shutil.copyfile(graph, os.path.join(ROOT, "assets/training_progress.png"))
        print("  assets/training_progress.png <- runs/training_progress.png")
    body = (
        "<details>\n<summary><b>Graph of the whole run so far</b></summary>\n\n"
        "![training progress](assets/training_progress.png)\n\n"
        f"*Every sample round of the run to date: {latest['chars']:,.1f}M "
        f"characters over {len(scored):,} evaluations"
        + ("; held-out from before the [in-flight corrections]"
           "(memory/corrections.md) is drawn moved onto today's level"
           if any(d.get("shifted") for _, _, d in scored) else "")
        + ".*\n\n</details>\n\n"
        "<details>\n<summary><b>Current quality of samples the model "
        "generates</b></summary>\n\n"
        f"*The round with the lowest held-out loss so far - "
        f"{best['val']:.4f} nats at {best['chars']:,.1f}M characters"
        + (" (from before an [in-flight correction](memory/corrections.md) "
           "to how held-out is read, moved onto today's level; its header "
           "below is as it was written)" if best.get("shifted") else "")
        + ". "
        + ("Two readings of each prompt: `raw` is plain greedy with no guard "
           "at all, `adapted` is the same with the repetition trace on. "
           if "[adapted]" in best_b else
           "Each prompt is written the way chat writes, by the rule the "
           "header names"
           + (", and `raw` beside it is plain greedy with no guard at all. "
              if "[raw]" in best_b else ". ")
           if "[written]" in best_b else
           "Each prompt is written plain greedy, with no guard at all. ")
        + f"The whole history is in [runs/samples.txt](runs/samples.txt).*\n\n"
        "```\n" + BAR + best_h + BAR + best_b.rstrip() + "\n```\n\n</details>"
    )
    out, k = region(out, "run-blocks", body); ok = ok and k

    # ---- the benchmark tables ---------------------------------------------
    names = sorted(latest["dom"], key=lambda k: latest["dom"][k])
    rows = "\n".join(f"| `{n}` | {latest['dom'][n]:.3f} | "
                     f"{latest['dom'][n] / LN2:.3f} |" for n in names)
    m = latest.get("measured")
    how = "" if not m else (
        "; held-out measured on the model as it stands, on "
        + ("every block of the held-out set" if m.get("all") else
           f"{min(m['blocks'].values())} blocks of {m['block_chars']:,} "
           "characters spread over all of each subject"))
    body = (
        f"**Where the model is** ({latest['chars']:,.1f}M characters read, "
        f"{latest.get('experts', '?')} experts{how}):\n\n"
        "| | nats/char | bits/byte |\n|---|---|---|\n"
        f"| **held-out, all {len(names)} subjects** | **{latest['val']:.4f}** "
        f"± {latest['se']:.4f} | **{latest['bpb']:.4f}** |\n"
        f"| train | {latest.get('train', float('nan')):.4f} | "
        f"{latest.get('train', float('nan')) / LN2:.4f} |\n\n"
        "**Held-out loss per subject:**\n\n"
        "| Subject | nats/char | bits/byte |\n|---|---|---|\n" + rows
    )
    out, k = region(out, "benchmarks", body); ok = ok and k

    # ---- enwik9 against published models -----------------------------------
    out, k = enwik9(out); ok = ok and k

    # ---- PG19 against published models -------------------------------------
    out, k = pg19(out); ok = ok and k

    # ---- the scaling paragraph and its projection --------------------------
    out, k = scaling(out, latest, trend, pg); ok = ok and k

    out, k = parameters(out); ok = ok and k

    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print(f"  README.md refreshed from {len(scored):,} sample rounds "
          f"(latest {latest['val']:.4f}, best {best['val']:.4f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
