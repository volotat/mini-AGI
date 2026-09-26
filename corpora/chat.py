#!/usr/bin/env python3
"""
Build a conversational corpus for mini-AGI.

The models so far have seen Python source and arithmetic and nothing else, so
they cannot hold a conversation at all - not because of the interface, but
because they have never seen a dialogue turn.

Rather than download instruction data of unknown relevance, most of this is
mined from the code already on the machine: a function's docstring is an
instruction and its body is the response, which is real instruction-following
data grounded in what the model was actually trained on. Arithmetic questions
are phrased naturally over the same generator as math_data. A small hand-written
seed covers greetings, capability questions and refusals, because none of that
exists in stdlib source.

FORMAT
    <user>
    Write a function that merges two sorted lists.
    </user>
    <bot>
    def merge_sorted(a, b):
        ...
    </bot>

    python3 -m corpora chat
"""

import os
import re
import ast
import sys
import json
import time
import random
import argparse
import textwrap

import numpy as np

import corpora.arithmetic as math_data
from corpora.code import collect_roots, scan

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"

# Ways to ask for an implementation. Variety here is what stops the model from
# only responding to one exact phrasing.
ASK_CODE = [
    "Write a function that {d}",
    "Write {name}: {d}",
    "Implement a function called {name}. {d}",
    "How do I write a function that {d}?",
    "Can you write {name}? It should {d}",
    "I need a Python function that {d}",
    "{d} Write it as a function named {name}.",
]
ASK_EXPLAIN = [
    "What does this do?\n{code}",
    "Explain this function:\n{code}",
    "I don't understand this code:\n{code}",
]
ASK_MATH = [
    "What is {q}?", "Compute {q}", "{q} = ?", "Can you work out {q}?",
    "What's {q}?", "Please calculate {q}.",
]

# ---------------------------------------------------------------------------
# What the model knows about itself.
#
# Every number below is read from the model that exists, not typed in: facts.py
# reads weights/manifest.json and the training log. If the architecture changes
# and the corpus is rebuilt, the answers change with it, so the model is never
# taught a description of itself that has gone stale.
#
# The answers describe mechanism. A model that can say how it routes, how deep
# it went, and where its experts live is giving a checkable account of itself;
# one that says it is small and unreliable is only apologising.
# ---------------------------------------------------------------------------

def _self_facts():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "video"))
    try:
        from facts import FACTS
    except Exception:
        FACTS = {}
    f = dict(FACTS)
    f.setdefault("block", 4096)
    f.setdefault("vocab", 265)
    f.setdefault("n_experts", 74)
    f.setdefault("top_k", 16)
    f.setdefault("max_steps", 6)
    f.setdefault("pool_max", 512)
    f.setdefault("block_applications", 26)
    # the vocabulary is the 256 byte values plus the structural markers
    f["n_markers"] = f.get("vocab", 265) - 256
    f.setdefault("unique_blocks", 6)
    f.setdefault("weights_mb", 236)

    # What the model is now, read from the model itself and from the settings
    # it runs under, so the answers cannot drift from the thing they describe.
    import json
    try:
        from minagi.config import load as _lc, get as _g
        c = _lc()
    except Exception:
        c, _g = {}, lambda c, k, d=None: d
    # WHICH manifest. This read "weights/manifest.json" while the run has been
    # writing to weights_rows/ for months, so every man.get() below fell
    # through to a hardcoded default and the model was told it had 74 experts
    # and a top-16 router - the exact drift this file exists to prevent. The
    # configured directory first, then any weights* that has one, newest.
    import glob
    man = {}
    # MINAGI_FACTS_WEIGHTS lets a rebuild describe a directory other than
    # whichever one happens to be newest - a fresh run has no manifest yet, and
    # the honest source for it is the config, not the previous run's pool.
    forced = os.environ.get("MINAGI_FACTS_WEIGHTS")
    if forced == "config":
        cands = []
    elif forced:
        cands = [os.path.join(forced, "manifest.json")]
    else:
        cands = [os.path.join(str(_g(c, "data.weights", "weights")),
                              "manifest.json")]
    if forced is None:
        cands += sorted(glob.glob("weights*/manifest.json"),
                        key=os.path.getmtime, reverse=True)
    for q in cands:
        try:
            man = json.load(open(q))
            break
        except Exception:
            continue
    cfg = man.get("cfg", {})
    f["d_model"] = cfg.get("d_model") or _g(c, "model.d_model", 512)
    f["expert_width"] = cfg.get("pool_d_ff") or _g(c, "pool.width", 2048)
    f["expert_depth"] = cfg.get("pool_depth") or _g(c, "pool.depth", 1)
    f["expert_params"] = (f["expert_depth"] * 3 * f["d_model"]
                          * f["expert_width"])
    f["resident"] = cfg.get("pool_resident") or _g(c, "pool.resident", 32)
    f["ram_cache"] = _g(c, "pool.ram_cache", 96)
    # a fresh run has no manifest yet, so config is the authority
    f["n_experts"] = man.get("n_experts") or _g(c, "pool.experts",
                                                f["n_experts"])
    f["top_k"] = cfg.get("pool_top_k") or _g(c, "pool.top_k", f["top_k"])
    f["n_prelude"] = cfg.get("n_prelude") or _g(c, "model.n_prelude", 2)
    f["n_recur"] = cfg.get("n_recur") or _g(c, "model.n_recur", 3)
    f["n_coda"] = cfg.get("n_coda") or _g(c, "model.n_coda", 1)
    f["max_steps"] = cfg.get("max_steps") or _g(c, "model.max_steps",
                                                f["max_steps"])
    # only the recurrent blocks and the coda route through the pool; the
    # prelude blocks keep a dense feed-forward of their own
    f["pooled_depths"] = f["max_steps"] * (f["n_recur"] + f["n_coda"])
    f["block_applications"] = f["n_prelude"] + f["pooled_depths"]
    f["context"] = man.get("context_now") or _g(c, "model.context_start",
                                                f["block"])
    f["context_ceiling"] = cfg.get("block") or _g(c, "model.context_end", _g(c, "model.context", 24576))
    f["margin_pct"] = int(100 * _g(c, "pool.margin", 0.10))
    f["dwell"] = _g(c, "pool.dwell_chars", _g(c, "pool.dwell", 2048))
    f["precision"] = _g(c, "training.precision", "bf16")
    f["n_head"] = cfg.get("n_head") or _g(c, "model.n_head", 8)
    f["head_dim"] = f["d_model"] // max(f["n_head"], 1)
    f["trunk_d_ff"] = cfg.get("d_ff") or _g(c, "model.d_ff", 1408)
    f["min_steps"] = cfg.get("min_steps") or _g(c, "model.min_steps", 1)
    f["bptt_window"] = cfg.get("bptt_window") or _g(
        c, "model.bptt_window", 16)
    f["halt_thresh"] = cfg.get("halt_thresh") or _g(c, "model.halt_thresh", 0.9)
    f["halt_prior"] = cfg.get("halt_prior") or _g(c, "model.halt_prior", 0.4)
    f["mean_depth"] = 1.0 / max(f["halt_prior"], 1e-6)
    f["chunk"] = _g(c, "training.chunk", 512)
    f["lr"] = _g(c, "training.lr", 3e-4)
    f["trunk_lr_mult"] = _g(c, "training.trunk_lr_mult", 0.2)
    f["weight_decay"] = _g(c, "training.weight_decay", 0.1)
    f["clip"] = _g(c, "training.clip", 1.0)
    f["grow_every"] = _g(c, "growth.every_chars", _g(c, "growth.every", 200_000))
    f["grow_k"] = _g(c, "growth.k", 4)
    f["max_in_flight"] = _g(c, "growth.max_in_flight", 32)
    # Deadness is staleness now, measured against this window, and the gate is
    # read by neither the pruner nor the growth brake. `prune.min_gate` and
    # `prune.min_age` are gone from the config entirely.
    f["survival_chars"] = int(str(_g(c, "prune.survival_chars",
                                     100_000_000)).replace("_", ""))
    f["survival_chars_m"] = f["survival_chars"] / 1e6
    f["dying_at"] = _g(c, "prune.dying_at", 0.65)
    f["min_visit_chunks"] = _g(c, "data.min_visit_chunks", 16)
    f["context_step"] = _g(c, "model.context_step", 1)
    f["visit_chars"] = f["min_visit_chunks"] * f["chunk"]
    f["pool_params_m"] = f["n_experts"] * f["expert_params"] / 1e6
    f["resident_params_m"] = f["resident"] * f["expert_params"] / 1e6
    f["expert_params_m"] = f["expert_params"] / 1e6
    f["expert_mb"] = f["expert_params"] * 4 / 1e6
    f["expert_mb_paged"] = f["expert_params"] * 4 * 3 / 1e6
    return f


_F = _self_facts()

def _load_self_knowledge():
    """
    What the model says about itself, kept as text rather than as code.

    It lives in self_knowledge.yaml so it can be read, checked and corrected
    without touching a program - which matters, because a description that
    lives in source goes stale quietly. Every number in it is a placeholder
    filled from the model that exists, so an answer cannot claim a size or a
    setting the model does not have.
    """
    import os
    import yaml
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "self_knowledge.yaml")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        doc = yaml.safe_load(fh) or []
    out = []
    for item in doc:
        ask = item.get("ask") or []
        say = (item.get("say") or "").strip()
        if ask and say:
            out.append((ask, " ".join(say.split())))
    return out


SELF_FROM_CODE = [
    (["who are you", "what are you", "introduce yourself", "tell me about yourself"],
     "I am mini-AGI. I read and write one character at a time. My alphabet is "
     "the 256 byte values, plus {n_markers} markers for structure, so any text "
     "is already spelled in it. I am assembled from a pool of small experts "
     "rather than a fixed "
     "stack, and the set of them running changes for every character I "
     "produce."),

    (["how do you work", "explain how you work", "how are you built",
      "describe your architecture"],
     "I hold a pool of {n_experts} experts. For each character a router scores "
     "them and the top {top_k} run; the rest stay idle. The same blocks are "
     "applied repeatedly rather than stacked once, so {unique_blocks} distinct "
     "blocks become {block_applications} applications in a single pass."),

    (["how big is your context", "how much can you remember",
      "what is your context window", "how far back can you see"],
     "I attend directly over the last {block} characters. Past that, what I "
     "read is not held as text — training steps fold it into my weights while "
     "I read, so it is still present, in a form I cannot quote back."),

    (["how do you think", "do you think before answering",
      "what happens before you answer"],
     "Two things. I can run the same blocks over my own hidden state up to "
     "{max_steps} times before committing to a character, which is thinking "
     "that never becomes text. I can also write working out between think "
     "tags and read it back, which is thinking you can inspect."),

    (["how many experts do you have", "how many parameters do you have",
      "how big are you"],
     "I currently hold {n_experts} experts, with room for {pool_max}. Only "
     "{top_k} of them are on the card at any moment. On disk I am "
     "{weights_mb:.0f} megabytes, one file per expert."),

    (["where are your weights", "where do you store your experts",
      "how are you stored"],
     "In a directory, one file per expert. That directory is not a copy of me "
     "— it is me. When I grow an expert a new file appears; when one is pruned "
     "its file is deleted."),

    (["can you learn", "do you learn", "are you still learning",
      "can you change"],
     "Yes. Reading is training: each stretch of text I read produces a "
     "gradient step, and my weights are different afterwards. New experts are "
     "added on speculation, start at a gate of zero so they change nothing, "
     "and are kept only if training raises that gate."),

    (["how do you choose your answer", "are you random", "do you sample randomly",
      "why do you give different answers"],
     "I do not draw from a random number generator. The character I emit is "
     "the highest scoring one, adjusted so I do not repeat myself. What varies "
     "my output is the state I am in — which experts are loaded, how many "
     "passes I took, what I have just read."),

    (["what were you trained on", "what data have you seen",
      "what do you know about"],
     "Python source, arithmetic, conversation, and chess notation, mixed "
     "together rather than taken in turn. All of it as raw characters."),

    (["how deep are you", "how many layers do you have",
      "how much computation do you do"],
     "It depends on the character. I decide per character how many passes to "
     "take, up to {max_steps}, and stop when another pass would not change the "
     "answer. At full depth that is {block_applications} block applications "
     "from {unique_blocks} distinct blocks."),

    (["are you sure", "how confident are you", "can you be wrong"],
     "I can be wrong, and I have a measurable sense of when: I take more "
     "passes over characters I find harder. If you can run what I give you, "
     "run it."),

    (["do you remember our last conversation", "do you remember me"],
     "Not as a transcript. I see the current conversation directly. Earlier "
     "conversations, if I trained on them, are in my weights rather than in "
     "front of me."),

    (["can you browse the internet", "can you look things up"],
     "I answer from my weights and from what is in front of me right now."),

    (["what is an expert", "what do you mean by expert",
      "explain your experts"],
     "A small feed-forward network in my pool. None of them is assigned a "
     "subject. What each one is for is settled by training, and any one "
     "capability is spread across many of them at once."),

    (["hello", "hi", "hey"],
     "Hello. I can write Python, do arithmetic, talk, and read chess "
     "notation. What do you need?"),

    (["thanks", "thank you"], "You're welcome."),
]

YAML_SELF = _load_self_knowledge()
SELF = YAML_SELF or SELF_FROM_CODE

SEED = [(q, a.format(**_F)) for qs, a in SELF for q in qs]
SELF_SEED = [(q, a.format(**_F)) for qs, a in YAML_SELF for q in qs]


def clean_doc(doc):
    """First sentence of a docstring, lowercased into an instruction."""
    doc = textwrap.dedent(doc or "").strip()
    if not doc:
        return None
    doc = doc.split("\n\n")[0].replace("\n", " ")
    doc = re.sub(r"\s+", " ", doc).strip()
    m = re.match(r"^(.{15,200}?[.!?])(\s|$)", doc)
    body = (m.group(1) if m else doc)[:200].strip()
    if len(body) < 15:
        return None
    if any(t in body.lower() for t in
           ("todo", "fixme", "deprecated", "internal use", ">>>", "http")):
        return None
    if not re.match(r"^[A-Za-z]", body):
        return None
    return body


def to_instruction(body):
    """'Returns the index of x.' -> 'returns the index of x'"""
    s = body[0].lower() + body[1:]
    return s.rstrip(".")


def harvest(limit_files=None, max_pairs=200000, min_lines=2, max_lines=30):
    """Mine (docstring, implementation) pairs from local Python."""
    roots = collect_roots([])
    pairs, seen = [], set()
    files = 0
    for path, text in scan(roots, verbose=False):
        files += 1
        if limit_files and files > limit_files:
            break
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            continue
        lines = text.split("\n")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            body = clean_doc(doc)
            if not body or node.name.startswith("_"):
                continue
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if not end or end - start < min_lines or end - start > max_lines:
                continue
            src = textwrap.dedent("\n".join(lines[start:end])).rstrip()
            if not src.startswith(("def ", "async def ")):
                continue
            # drop the docstring from the body so the answer is code, not prose
            key = (node.name, body[:60])
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"name": node.name, "doc": body, "code": src})
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def render_code_turn(p, rng):
    tmpl = rng.choice(ASK_CODE)
    q = tmpl.format(d=to_instruction(p["doc"]), name=p["name"])
    return f"{U0}\n{q}\n{U1}\n{B0}\n{p['code']}\n{B1}\n"


def render_explain_turn(p, rng):
    q = rng.choice(ASK_EXPLAIN).format(code=p["code"])
    return f"{U0}\n{q}\n{U1}\n{B0}\n{p['doc']}\n{B1}\n"


def render_math_turn(rng):
    task = rng.choice(["add", "sub", "mul", "mod"])
    fn, cap, _ = math_data.TASKS[task]
    line = fn(rng, rng.randint(1, min(cap, 6)), rng.random() < 0.25)
    head, _, ans = line.rpartition("=")
    expr = head.split(" ", 1)[1].strip()
    q = rng.choice(ASK_MATH).format(q=expr)
    return f"{U0}\n{q}\n{U1}\n{B0}\n{ans.strip()}\n{B1}\n"


def render_seed_turn(rng, seed=None):
    q, a = rng.choice(SEED if seed is None else seed)
    if rng.random() < 0.5:
        q = q.capitalize() + ("?" if q.split()[0] in
                              ("who", "what", "can", "do", "are", "how",
                               "where", "why", "when", "is") else "")
    return f"{U0}\n{q}\n{U1}\n{B0}\n{a}\n{B1}\n"


def build_conversation(pairs, rng, max_turns=4):
    """Stack several turns so the model learns multi-turn structure."""
    n = rng.randint(1, max_turns)
    out = []
    for _ in range(n):
        r = rng.random()
        # the self-knowledge turns carry the model's account of its own
        # architecture, so they get a fifth of the conversation rather than the
        # eighth a stock persona would need
        if r < 0.46 and pairs:
            out.append(render_code_turn(rng.choice(pairs), rng))
        elif r < 0.64 and pairs:
            out.append(render_explain_turn(rng.choice(pairs), rng))
        elif r < 0.80:
            out.append(render_math_turn(rng))
        else:
            out.append(render_seed_turn(rng))
    return "".join(out) + "\n"


def build_self_conversation(rng, max_turns=4):
    """Stack only the model-description turns loaded from YAML."""
    return "".join(render_seed_turn(rng, SELF_SEED)
                   for _ in range(rng.randint(1, max_turns))) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--conversations", type=int, default=400000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--max-pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-only", action="store_true",
                    help="generate only YAML-backed model-description turns")
    args = ap.parse_args()
    args.out = args.out or ("data_self_knowledge_char" if args.self_only
                            else "data_chat_char")

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    if args.self_only:
        if not SELF_SEED:
            print("self_knowledge.yaml has no usable question/answer pairs",
                  file=sys.stderr)
            return 1
        pairs = []
        def make_conversation():
            return build_self_conversation(rng)
        print(f"generating from {len(SELF_SEED):,} self-knowledge "
              f"question/answer pairs")
    else:
        print("harvesting docstring/implementation pairs from local Python ...")
        t0 = time.time()
        pairs = harvest(max_pairs=args.max_pairs)
        print(f"  {len(pairs):,} pairs in {time.time()-t0:.1f}s")
        if not pairs:
            print("no pairs harvested", file=sys.stderr)
            return 1
        print("\nexamples:")
        for p in pairs[:3]:
            print(f"  {p['name']}: {p['doc'][:70]}")
        def make_conversation():
            return build_conversation(pairs, rng)

    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()

    print("\nsample conversation:")
    print(textwrap.indent(make_conversation()[:600], "  "))

    for split, count in (("train", args.conversations), ("val", args.val)):
        path = os.path.join(args.out, f"{split}.bin")
        total = 0
        with open(path, "wb") as f:
            batch = []
            for i in range(count):
                batch.append(make_conversation())
                if len(batch) >= 2048:
                    for enc in tok.encode_batch(batch):
                        a = np.array(enc.ids, dtype=np.uint16)
                        f.write(a.tobytes())
                        total += len(a)
                    batch.clear()
                    if split == "train" and i % 50000 < 2048:
                        print(f"\r  {split}: {total/1e6:.1f}M tokens",
                              end="", file=sys.stderr)
            if batch:
                for enc in tok.encode_batch(batch):
                    a = np.array(enc.ids, dtype=np.uint16)
                    f.write(a.tobytes())
                    total += len(a)
        print(f"\r  {split}: {total:,} tokens -> {path}")
        if split == "train":
            tr = total
        else:
            va = total

    meta = {"vocab_size": tok.get_vocab_size(), "train_tokens": tr,
            "val_tokens": va,
            "pairs": len(SELF_SEED) if args.self_only else len(pairs),
            "self_only": args.self_only,
            "tokenizer": "byte",
            "format": f"{U0}...{U1}{B0}...{B1}"}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
