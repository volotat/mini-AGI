#!/usr/bin/env python3
"""
A chat window for talking to the model.

    python3 serve.py                 then open http://127.0.0.1:8080

Nothing but the conversation. Generation streams a character at a time, so
what you watch is the model writing rather than a spinner followed by a wall
of text - at this size that difference is most of the information.

Decoding is greedy and deterministic: the same conversation gives the same
reply twice. The loops greedy decoding falls into are held off by the
adaptation trace in minagi/decode.py, at the strength config.yaml sets.

IT LEARNS WHILE YOU TALK TO IT, and this CHANGES THE WEIGHTS ON DISK. The
exchange joins one continuous stream and the model takes an optimiser step
every `training.chunk` characters of it - the same path minagi/stream.py
takes through a corpus, which is why there is no second mechanism to reason
about. Read minagi/live.py for what it costs; it is not free, and on short
chat it is a far denser diet than documents. `--no-learn` serves read-only.

Binds to localhost.
"""

import argparse
import json
import os
import sys
import threading
import time

# see train.py: the allocator reads this once at CUDA init, so it has to be
# set before torch loads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from flask import Flask, Response, jsonify, request

from minagi.recur import load_any
from minagi.tokenizer import ByteTokenizer

app = Flask(__name__)

# One model, one conversation at a time. The lock is what makes that true:
# two overlapping requests would interleave their forward passes through the
# same KV caches and produce two corrupted replies instead of one good one.
LOCK = threading.Lock()
STATE = {"model": None, "tok": None, "weights": None, "learner": None}

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"

from minagi import tools as tools_mod
from minagi import maat as maat_mod

REGISTRY = tools_mod.ToolRegistry()
maat_mod.register_maat_tools(REGISTRY)

from minagi import ledger as ledger_mod
from minagi import http_tool as http_tool_mod

os.makedirs("runs", exist_ok=True)
RECEIPTS = ledger_mod.ReceiptLedger(os.path.join("runs", "receipts.jsonl"))

TOOLS_ON = True
MAX_TOOL_CALLS = 4

# A passage of the corpus the conversation opens with. Empty when priming is
# off or no corpus is on disk.
PRIME = ""


def load_prime(chars, root="data/train/self-knowledge"):
    """
    Corpus text to open the conversation with, or "" if there is none.

    TWO THINGS IT BUYS, and they are separate. The reply has context to stand
    on: "hi" alone is two characters, and every character the model writes
    attends over what came before it and routes among the experts that whole
    context has voted for. And the learning stream starts part-way to its
    first step rather than twenty exchanges short of one.

    Taken from the self-knowledge lane because it is already in the register
    the chat is in - <user>/<bot> turns about what the model is - so this is
    text the model has read, not a preamble invented here and trained on.
    The `self-` files are the short question-and-answer ones; `auto-src-` are
    whole source files quoted back, and priming with one of those biases the
    router toward code and teaches the model to open by dumping a module.

    CUT WHERE THE MARKERS BALANCE, not at the last closing tag. This corpus
    documents its own format, so a file contains a code block reading
    "<bot>   </bot>" as prose - and those are real marker tokens, not
    spellings. Trimming to the last </bot> landed inside that block and
    returned a passage with an unclosed turn in it, which hands the model an
    open <bot> to answer inside before the conversation has started.

    Deterministic: the first file that yields a usable passage, every time. A
    passage picked at random would make the same conversation give different
    answers across restarts, and repeating a reply is worth more than variety.
    """
    if chars <= 0 or not os.path.isdir(root):
        return ""
    names = sorted(f for f in os.listdir(root) if f.endswith(".txt"))
    # conversational turns first, source dumps only if there are none
    names = ([n for n in names if n.startswith("self-")]
             + [n for n in names if not n.startswith("self-")])
    for name in names[:20]:
        with open(os.path.join(root, name), errors="replace") as f:
            text = f.read(chars * 3)
        best = ""
        at = text.find(f"{B1}\n")
        while 0 <= at < chars:
            end = at + len(B1) + 1
            head = text[:end]
            if (head.count(U0) == head.count(U1) == head.count(B0)
                    == head.count(B1)):
                best = head
            at = text.find(f"{B1}\n", end)
        if len(best) > chars // 3:
            return best
    return ""


def _manifest(path):
    """As loaded, so a save does not overwrite it with a blank one."""
    try:
        with open(os.path.join(path, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(weights, device=None, learn=True, lr=3e-4, save_every=8,
         chunk=None):
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # read_only marks nothing dirty, so an expert paged in is never written
    # back. That is right for serving and wrong for learning - what the model
    # learned would live in VRAM until the slot was reused and then be gone.
    ro = not learn
    try:
        model, _ = load_any(weights, dev, read_only=ro)
    except torch.cuda.OutOfMemoryError:
        # training may be holding most of the card; serve slowly rather than
        # not at all
        torch.cuda.empty_cache()
        dev = torch.device("cpu")
        model, _ = load_any(weights, dev, read_only=ro)
        print("[warn] CUDA is full, serving on CPU", file=sys.stderr)
    model.eval()
    STATE.update(model=model, tok=ByteTokenizer(), weights=weights)
    print(f"[loaded] {weights} on {dev}", file=sys.stderr)

    # A paged model loads with an EMPTY card - every slot -1. A reply is
    # admitted by its own text's vote, so this does not change a reply; what
    # it changes is that the page can show experts on the card before the
    # first question instead of nothing. It is one forward over the priming
    # text, which admits the experts that text asks for, as any forward does.
    if PRIME:
        ids = STATE["tok"].encode(PRIME).ids[-model.cfg.block:]
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))

    if learn:
        from minagi.config import get as _g, load as _lc
        from minagi.live import LiveLearner
        c = _lc()
        STATE["learner"] = LiveLearner(
            model, lr=lr,
            trunk_lr_mult=_g(c, "training.trunk_lr_mult", 0.1),
            wd=_g(c, "training.weight_decay", 0.1),
            clip=_g(c, "training.clip", 1.0),
            # Defaults to the granularity the corpus is read at, so talking
            # to it and reading a book are one mechanism. That is the right
            # default and a poor fit for a short session: at 2048 an exchange
            # of ninety characters needs twenty-odd turns to reach one step.
            # --learn-chunk trades that against the denser diet live.py warns
            # about.
            chunk=chunk or _g(c, "training.chunk", 2048),
            save_every=save_every,
            weights_dir=weights,
            manifest=_manifest(weights))
        # The prime goes into the stream too, so the first step arrives
        # sooner. Kept SHORTER than a chunk on purpose: long enough to
        # advance the counter, short enough that it never takes a step by
        # itself, so restarting cannot train on the same passage repeatedly.
        if PRIME:
            STATE["learner"].feed(PRIME[-(STATE["learner"].chunk - 1):],
                                  STATE["tok"], note="prime")
        ch = STATE["learner"].chunk
        left = max(0, ch - STATE["learner"].pending)
        print(f"[learning] on - one step per {ch} characters, "
              f"{left} to the first (about {max(1, round(left / 93))} "
              f"exchanges), saved to {weights}/ every {save_every} steps",
              file=sys.stderr)
    return model


def resident_experts():
    """
    Which experts are on the card right now, by stable id.

    `slots` is slot -> expert index, and the index is not an identity:
    pruning renumbers the pool, so the same index means different experts at
    different times. `uid` is what survives that, which is why it is what the
    page shows.
    """
    pool = getattr(STATE.get("model"), "pool", None)
    if pool is None or not hasattr(pool, "slots"):
        return None
    gate = pool.gate.detach().abs()
    top = float(gate.max().clamp_min(1e-9))
    out = []
    for j, e in enumerate(pool.slots):
        e = int(e)
        if e < 0:
            continue
        out.append({"slot": j, "uid": int(pool.uid[e]),
                    "gate": round(float(gate[e]) / top, 3)})
    return {"resident": out, "n_experts": int(pool.n_experts()),
            "n_slots": len(pool.slots),
            "admitted": len(getattr(pool, "_admitted", ()) or ())}


def learn_state():
    """How far the stream is from its next optimiser step."""
    lr_ = STATE.get("learner")
    if lr_ is None:
        return None
    return {"pending": int(lr_.pending), "chunk": int(lr_.chunk),
            "steps": int(lr_.steps)}


def build_prompt(messages, budget, prime=""):
    """
    The conversation as the model reads it, newest turns kept.

    The trailing open `<bot>` tag is what asks for a reply, so the trim is
    from the FRONT. Cutting the end to fit would remove the question.
    """
    parts = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if m.get("role") == "user":
            parts.append(f"{U0}\n{text}\n{U1}\n")
        else:
            parts.append(f"{B0}\n{text}\n{B1}\n")
    prompt = prime + "".join(parts) + f"{B0}\n"
    return prompt[-budget:] if len(prompt) > budget else prompt


def where(caches):
    """
    The position the next character sits at: however much history the
    cache still holds after trimming.

    NOT a running count. Rotary tables are built for positions 0 to
    block-1, so a counter that saturates at `block` asks for position
    `block` on the very next character and the model refuses. Reading it
    back off the cache cannot drift, because the cache is the thing the
    positions have to agree with.
    """
    for c in caches:
        if c.get("k") is not None:
            return c["k"].shape[-2]
    return 0


def _prefill_text(out, caches, tok, text, model):
    from minagi.stream import trim_caches
    ids = tok.encode(text).ids[-model.cfg.block:]
    chunk = torch.tensor([ids], device=out.device)
    CHUNK = 512
    logits = None
    for i in range(0, chunk.shape[1], CHUNK):
        part = chunk[:, i:i + CHUNK]
        trim_caches(caches, model.cfg.block - part.shape[1])
        logits = model(part, caches=caches, pos_offset=where(caches))[0]
    out = torch.cat([out, chunk], dim=1)
    return out, caches, logits


@torch.no_grad()
def stream(prompt, max_new):
    from minagi.config import get as _g, load as _lc
    from minagi.decode import pick_next
    from minagi.stream import trim_caches

    c = _lc()
    strength = _g(c, "decoding.adapt_strength", 2.5)
    decay = _g(c, "decoding.adapt_decay", 0.88)

    model, tok = STATE["model"], STATE["tok"]
    device = next(model.parameters()).device
    ids = tok.encode(prompt).ids[-model.cfg.block:]
    out = torch.tensor([ids or [10]], device=device)

    # THE TEXT CHOOSES. The prompt is read in chunks, and each chunk adds its
    # characters' requests to the text's vote and is admitted the most-voted
    # experts; then every character of the reply is a forward of its own that
    # adds its requests and routes among what the whole text has voted for,
    # loading whatever is not already on the card. Nothing here chooses
    # experts - the forward does.
    pool = getattr(model, "pool", None)
    caches = model.empty_caches()

    # Prefill in chunks, the way training reads a corpus. Feeding a long
    # prompt in one pass materialises activations for every position across
    # every block application at once, which is what puts a long context out
    # of reach; the cache carries the reach instead.
    CHUNK = 512
    logits = None
    for i in range(0, out.shape[1], CHUNK):
        part = out[:, i:i + CHUNK]
        trim_caches(caches, model.cfg.block - part.shape[1])
        logits = model(part, caches=caches, pos_offset=where(caches))[0]

    cur = out[:, -1:]
    produced = []
    # The prompt's forwards have admitted its experts. Report them, so the
    # page starts the reply showing what is actually answering it.
    yield {"swap": {"at": 0, "moved": None, "pool": resident_experts()}}
    n_tool = 0
    for i in range(max_new):
        loads = getattr(pool, "loads", 0)
        if logits is None:
            trim_caches(caches, model.cfg.block - cur.shape[1])
            logits = model(cur, caches=caches, pos_offset=where(caches))[0]
            moved = getattr(pool, "loads", 0) - loads
            if moved:
                # this character asked for experts that were not on the card;
                # `moved` counts the ones copied onto it
                yield {"swap": {"at": i, "moved": int(moved),
                                "pool": resident_experts()}}
        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
        logits = None
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
        calls = tools_mod.parse_tool_calls(text) if TOOLS_ON else []
        if calls and n_tool < MAX_TOOL_CALLS:
            n_tool += 1
            call = calls[0]
            yield {"tool": {"name": call.name, "args": call.args}}
            result = tools_mod.execute(REGISTRY, call)
            RECEIPTS.append("tool", {"name": call.name, "args": call.args,
                                     "ok": result.get("ok", False)})
            res_text = tools_mod.result_text(result)
            out, caches, logits = _prefill_text(out, caches, tok, res_text, model)
            cur = out[:, -1:]
            produced = []
            yield {"t": res_text}
            continue
        if text.endswith(B1):
            break
        yield {"t": tok.decode([produced[-1]])}


def remember(user_text, bot_text):
    """
    Read the finished exchange, the way the corpus reader reads a file.

    ONLY THIS TURN goes in. The prompt above carries the whole conversation,
    so re-feeding it every time would put the earliest turns through the
    stream once per exchange - re-reading rather than reading on, which is
    the dense regime minagi/live.py warns about.

    The caller already holds LOCK.
    """
    learner = STATE.get("learner")
    if learner is None:
        return None
    from minagi.live import exchange_text
    text = exchange_text(user_text, bot_text.replace(B1, ""))
    if not text:
        return None
    try:
        recs = learner.feed(text, STATE["tok"], note="chat")
        # Reported even when no step was taken. A step needs `chunk`
        # characters and an exchange is about ninety, so the first twenty or
        # so take none at all - and silence there is indistinguishable from
        # learning being broken, which is exactly the wrong thing to leave
        # ambiguous.
        out = {"steps": len(recs),
               "pending": learner.pending,
               "chunk": learner.chunk,
               "total": learner.steps}
        if recs:
            out["loss"] = round(float(recs[-1].get("loss", 0.0)), 4)
            if learner.due_to_save():
                learner.save()
                out["saved"] = True
        return out
    except Exception as e:                                    # noqa: BLE001
        # A failed step must not cost the reply that was already written.
        print(f"[learn] failed: {e}", file=sys.stderr)
        return {"error": str(e)}


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    model = STATE["model"]
    msgs = body.get("messages", [])
    prompt = build_prompt(msgs, int(model.cfg.block * 0.9), PRIME)
    max_new = int(body.get("max_new", 400))
    # the half of the exchange the model did not predict, which is where the
    # signal in a conversation is
    last_user = next((m.get("content") for m in reversed(msgs)
                      if m.get("role") == "user"), "")

    def events():
        t0, n = time.time(), 0
        reply = []
        learned = None
        try:
            with LOCK:
                for ev in stream(prompt, max_new):
                    if "t" in ev:
                        n += 1
                        reply.append(ev["t"])
                    yield f"data: {json.dumps(ev)}\n\n"
                learned = remember(last_user, "".join(reply))
        except torch.cuda.OutOfMemoryError:
            # Almost always a training run holding the card. Say so and stay
            # up: dying here closes the socket, and all the browser can tell
            # you then is that the fetch failed.
            torch.cuda.empty_cache()
            yield "data: " + json.dumps({"error":
                "the GPU is out of memory - something else is probably using "
                "it. Restart with --device cpu, or stop the other process."
                }) + "\n\n"
        except Exception as e:                            # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        dt = max(time.time() - t0, 1e-6)
        done = {"done": True, "n": n, "cps": round(n / dt, 1),
                "learn": learn_state(), "pool": resident_experts()}
        if learned:
            done["learned"] = learned
        yield "data: " + json.dumps(done) + "\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/state")
def api_state():
    """
    What the two mechanisms are doing between replies.

    The page asks once on load and after every exchange, so the counters are
    right even when nothing is being generated - otherwise "how far from the
    next update" would only exist while the model was talking.
    """
    return jsonify({"learn": learn_state(),
                    "pool": resident_experts()})


@app.route("/api/prime")
def api_prime():
    """
    What the conversation is opened with.

    The prime is prepended to every prompt and never appears in the replies,
    so without this the model is answering from context the person talking to
    it cannot see. That is the kind of thing that should be inspectable rather
    than trusted.
    """
    return jsonify({"chars": len(PRIME),
                    "turns": PRIME.count(U0),
                    "text": PRIME})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-AGI</title>
<style>
  :root{
    --bg:#ffffff; --fg:#1f2328; --dim:#6b7280; --line:#e5e7eb;
    --user:#f3f4f6; --accent:#2563eb; --err:#b91c1c;
    --ok:#15803d; --glow:rgba(37,99,235,.28);
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0d1117; --fg:#e6edf3; --dim:#8b949e; --line:#21262d;
           --user:#1c2128; --accent:#58a6ff; --err:#f85149;
           --ok:#3fb950; --glow:rgba(88,166,255,.30); }
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column}
  header{border-bottom:1px solid var(--line);padding:10px 16px;
         display:flex;align-items:center;gap:12px;flex:0 0 auto}
  header b{font-weight:600}
  header .sp{flex:1}
  button{font:inherit;color:var(--fg);background:none;cursor:pointer;
         border:1px solid var(--line);border-radius:6px;padding:5px 11px}
  button:hover{border-color:var(--accent);color:var(--accent)}
  #log{flex:1 1 auto;overflow-y:auto;padding:24px 16px}
  .wrap{max-width:720px;margin:0 auto}
  .msg{margin:0 0 20px}
  .who{font-size:12px;color:var(--dim);margin-bottom:5px}
  .body{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
  .user .body{background:var(--user);padding:10px 13px;border-radius:10px}
  .err{color:var(--err)}
  .caret::after{content:"\258B";color:var(--accent);
                animation:blink 1s step-end infinite}
  @keyframes blink{50%{opacity:0}}
  details.prime{border:1px solid var(--line);border-radius:8px;
                margin:0 0 22px;background:var(--user)}
  details.prime summary{cursor:pointer;padding:8px 12px;font-size:12px;
                        color:var(--dim);list-style:none}
  details.prime summary::-webkit-details-marker{display:none}
  details.prime summary::before{content:"\25B8  ";display:inline-block;
                                transition:transform .15s}
  details.prime[open] summary::before{transform:rotate(90deg)}
  details.prime pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;
                    word-wrap:break-word;overflow-wrap:anywhere;
                    font-size:12px;color:var(--dim);max-height:320px;
                    overflow-y:auto}
  /* the two mechanisms, always on screen */
  #mech{border-bottom:1px solid var(--line);padding:9px 16px;flex:0 0 auto;
        background:var(--bg)}
  .mrow{max-width:720px;margin:0 auto;display:flex;gap:22px;
        align-items:center;flex-wrap:wrap}
  .meter{flex:1 1 210px;min-width:190px}
  .mlab{font-size:11px;color:var(--dim);display:flex;
        justify-content:space-between;margin-bottom:4px}
  .mlab b{font-weight:600;color:var(--fg)}
  .bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden}
  .fill{height:100%;width:0;border-radius:3px;background:var(--accent);
        transition:width .18s linear}
  .fill.mem{background:var(--ok)}
  /* the moment it fires */
  @keyframes flare{0%{transform:scale(1);box-shadow:0 0 0 0 var(--glow)}
                   40%{transform:scale(1.03);box-shadow:0 0 0 7px transparent}
                   100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
  .fired{animation:flare .75s ease-out}
  .note{font-size:11px;color:var(--ok);opacity:0;transition:opacity .2s}
  .note.on{opacity:1}
  .note.swap{color:var(--accent)}
  /* which experts are on the card */
  #chips{max-width:720px;margin:9px auto 0;display:flex;flex-wrap:wrap;gap:3px}
  .chip{font-size:10px;font-variant-numeric:tabular-nums;padding:1px 5px;
        border-radius:3px;border:1px solid var(--line);color:var(--dim);
        transition:background .3s,color .3s,border-color .3s}
  .chip.new{background:var(--accent);border-color:var(--accent);color:#fff}
  .empty{color:var(--dim);text-align:center;margin-top:18vh;line-height:1.9}
  .empty code{background:var(--user);padding:2px 6px;border-radius:4px;
              font-size:13px}
  footer{border-top:1px solid var(--line);padding:12px 16px;flex:0 0 auto}
  form{max-width:720px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
  textarea{flex:1;resize:none;font:inherit;color:var(--fg);
           background:var(--bg);border:1px solid var(--line);
           border-radius:9px;padding:10px 12px;max-height:180px;
           overflow-y:auto}
  textarea:focus{outline:none;border-color:var(--accent)}
  .stat{max-width:720px;margin:7px auto 0;font-size:12px;color:var(--dim);
        height:16px;display:flex;justify-content:space-between}
  .hint{color:var(--dim)}
</style></head>
<body>
<header>
  <b>mini-AGI</b>
  <span class="sp"></span>
  <button id="reset">New chat</button>
</header>

<div id="mech">
  <div class="mrow">
    <div class="meter" id="m-mem">
      <div class="mlab"><span>long-term memory</span><b id="mem-n">-</b></div>
      <div class="bar"><div class="fill mem" id="mem-f"></div></div>
    </div>
    <div class="meter" id="m-exp">
      <div class="mlab"><span>experts the last character admitted</span><b id="exp-n">-</b></div>
      <div class="bar"><div class="fill" id="exp-f"></div></div>
    </div>
    <span class="note" id="note"></span>
  </div>
  <div id="chips"></div>
</div>

<div id="log"><div class="wrap"><div id="primebox"></div><div id="thread"></div></div></div>

<footer>
  <form id="f">
    <textarea id="q" rows="1" placeholder="Say something"></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="stat"><span id="stat"></span><span class="hint">Enter to send, Shift+Enter for a newline</span></div>
</footer>

<script>
const thread = document.getElementById('thread');
const log    = document.getElementById('log');
const q      = document.getElementById('q');
const stat   = document.getElementById('stat');
const send   = document.getElementById('send');
let messages = [];
let busy = false;

// ------------------------------------------------ the two mechanisms
// Both are counters the server keeps; the page mirrors them so that what
// the model does to itself is visible while it is happening, rather than
// inferred afterwards from the fact that nothing changed.
let MECH = {pending:0, chunk:0, steps:0, admitted:0, slots:0, uids:[], loaded:0};
const $ = id => document.getElementById(id);

function note(text, kind){
  const n = $('note');
  n.textContent = text;
  n.className = 'note on' + (kind ? ' ' + kind : '');
  clearTimeout(note._t);
  note._t = setTimeout(() => { n.className = 'note'; }, 2600);
}
function flare(el){ el.classList.remove('fired');
                    void el.offsetWidth;            // restart the animation
                    el.classList.add('fired'); }

function drawMem(){
  const {pending, chunk} = MECH;
  if (!chunk){ $('mem-n').textContent = 'off'; $('mem-f').style.width = '0';
               return; }
  const left = Math.max(0, chunk - pending);
  // The step is taken after the reply finishes, not mid-sentence, so once
  // the stream is full "0 characters to go" is true but says the wrong
  // thing - it reads as though nothing is going to happen.
  $('mem-n').textContent = left
      ? left + ' characters to the next update'
      : (busy ? 'updates when this reply finishes' : 'ready to update');
  $('mem-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}
function drawExp(){
  // A character may use at most as many experts as the card has slots: the
  // ones its text - the prompt and the reply so far - has voted for most.
  const {admitted, slots} = MECH;
  if (!slots){ $('exp-n').textContent = '-'; return; }
  $('exp-n').textContent = admitted + ' of ' + slots;
  $('exp-f').style.width = (100 * Math.min(admitted, slots) / slots) + '%';
}
// Chips are keyed by uid, not by slot. A slot number says where an expert
// sits; the uid says WHICH expert it is, and that is what survives pruning.
function drawChips(pool, changed){
  if (!pool) return;
  const box = $('chips');
  box.innerHTML = '';
  for (const e of pool.resident){
    const c = document.createElement('span');
    c.className = 'chip' + (changed && changed.has(e.uid) ? ' new' : '');
    c.textContent = e.uid;
    c.title = 'expert ' + e.uid + ' - gate ' + e.gate + ' of the largest';
    box.appendChild(c);
  }
  if (changed && changed.size) setTimeout(() => {
    box.querySelectorAll('.chip.new').forEach(c => c.classList.remove('new'));
  }, 1400);
}

function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk;
                MECH.steps = d.learn.steps; }
  if (d.pool){
    MECH.admitted = d.pool.admitted || 0; MECH.slots = d.pool.n_slots || 0;
    drawExp();
    const seen = new Set(MECH.uids);
    const now = d.pool.resident.map(e => e.uid);
    const changed = MECH.uids.length ? new Set(now.filter(u => !seen.has(u)))
                                     : null;
    MECH.uids = now;
    drawChips(d.pool, changed);
    $('m-exp').title = d.pool.resident.length + ' of ' + d.pool.n_experts
                     + ' experts on the card';
  }
  drawMem(); drawExp();
}
fetch('/api/state').then(r => r.json()).then(applyState).catch(() => {});

// What the model is answering from, besides what you typed. It is prepended
// to every prompt server-side and never appears in a reply, so it is shown
// here rather than left to be taken on trust.
fetch('/api/prime').then(r => r.json()).then(p => {
  if (!p.chars) return;
  const d = document.createElement('details');
  d.className = 'prime';
  d.innerHTML = '<summary></summary><pre></pre>';
  d.querySelector('summary').textContent =
    'primed with ' + p.chars + ' characters of the corpus'
    + (p.turns ? ' (' + p.turns + ' turns)' : '')
    + ' - in front of every prompt';
  d.querySelector('pre').textContent = p.text;
  document.getElementById('primebox').appendChild(d);
}).catch(() => {});

function empty(){
  thread.innerHTML = '<div class="empty">Ask it something.<br>'
    + 'It is small, it is mid-training, and it will show.</div>';
}
function add(role, text){
  if (thread.querySelector('.empty')) thread.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = '<div class="who"></div><div class="body"></div>';
  d.querySelector('.who').textContent = role === 'user' ? 'You' : 'mini-AGI';
  d.querySelector('.body').textContent = text || '';
  thread.appendChild(d);
  return d.querySelector('.body');
}
// Only follow the text down while the reader is already at the bottom, so
// scrolling up to re-read something is not undone by the next character.
function atBottom(){ return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }
function follow(was){ if (was) log.scrollTop = log.scrollHeight; }

// Collapse to zero before measuring. 'auto' does not shrink a textarea
// inside a flex row, so scrollHeight reports the height it already has and
// the box only ever grows - an empty input opened at the 180px maximum.
function grow(){ q.style.height = '0px';
                 q.style.height = Math.min(q.scrollHeight, 180) + 'px'; }
q.addEventListener('input', grow);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault();
                                         document.getElementById('f').requestSubmit(); }
});

document.getElementById('reset').onclick = () => {
  if (busy) return;
  messages = []; empty(); stat.textContent = ''; q.focus();
};

document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true;
  q.value = ''; grow();

  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  MECH.loaded = 0;
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages, max_new: 400})
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for(;;){
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      // SSE frames are separated by a blank line; a frame can arrive split
      // across two reads, so anything after the last separator stays in buf.
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const f of frames){
        if (!f.startsWith('data: ')) continue;
        const d = JSON.parse(f.slice(6));
        if (d.t){ const was = atBottom(); reply += d.t;
                  body.textContent = reply; follow(was);
                  // every character the model writes goes into the stream
                  // it learns from, so both counters advance as it speaks
                  MECH.pending++;
                  if (MECH.chunk && MECH.pending >= MECH.chunk)
                    MECH.pending = MECH.chunk;
                  drawMem(); }
        else if (d.swap){
          applyState(d);
          if (d.swap.moved){
            // a load comes when the text's vote shifts, now and then; the
            // note keeps the reply's running total rather than flashing one
            // message per load
            MECH.loaded += d.swap.moved;
            flare($('m-exp'));
            note(MECH.loaded + (MECH.loaded === 1 ? ' expert loaded'
                                                  : ' experts loaded')
                 + ' for this reply', 'swap');
          }
        }
        else if (d.error){ body.classList.add('err');
                           body.textContent = reply + '\n\n[' + d.error + ']'; }
        else if (d.done){
          // The server's counters are authoritative: the page has been
          // guessing per character, and a step resets pending to something
          // it cannot work out on its own.
          const stepped = d.learned && d.learned.steps;
          applyState(d);
          if (stepped){
            flare($('m-mem'));
            note(stepped + (stepped === 1 ? ' weight update' : ' weight updates')
                 + (d.learned.saved ? ', saved to disk' : ''));
          }
          let line = d.n + ' characters, ' + d.cps + '/s';
          const L = d.learned;
          if (!L) line += '  ·  not learning';
          else if (L.error) line += '  ·  learning failed: ' + L.error;
          else if (L.steps) line += '  ·  ' + L.steps
                    + (L.steps === 1 ? ' step' : ' steps')
                    + ', loss ' + L.loss + (L.saved ? ', saved' : '')
                    + '  ·  ' + L.total + ' total';
          // No step yet is the normal state for the first several turns, so
          // say how far off one is rather than nothing at all.
          else line += '  ·  learning: ' + L.pending + '/' + L.chunk
                    + ' characters to the next step';
          stat.textContent = line;
        }
      }
    }
  } catch (err) {
    // A dropped connection is the server going away, not the model failing.
    // The browser reports both as "Failed to fetch", so say which is likely.
    const gone = String(err).includes('Failed to fetch');
    body.classList.add('err');
    body.textContent = reply + '\n\n[' + (gone
      ? 'lost the server - it stopped while answering. Check the terminal '
        + 'it is running in; if the GPU was busy, restart with --device cpu.'
      : err) + ']';
  }
  body.classList.remove('caret');
  // Keep the turn even when it came back empty, so the conversation the
  // model sees next time matches the one on screen.
  messages.push({role:'bot', content:reply});
  busy = false; send.disabled = false; q.focus();
  drawExp();
};

empty(); q.focus();
// After layout, not during it. At the end of the script the flex row has no
// resolved width yet, so the placeholder wraps and scrollHeight reports a
// tall box - the input opened at its maximum every time.
requestAnimationFrame(grow);
</script>
</body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights",
                    help="the weights directory to talk to")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default auto, falling back to CPU if "
                         "the card is full")
    ap.add_argument("--no-learn", dest="learn", action="store_false",
                    help="serve without learning. The weights are then opened "
                         "read-only and nothing is written back")
    ap.add_argument("--learn-lr", type=float, default=3e-4,
                    help="learning rate for the live stream")
    ap.add_argument("--prime-chars", type=int, default=1024,
                    help="characters of corpus to open the conversation "
                         "with, so the router has something to choose "
                         "experts from and the learning stream starts part "
                         "of the way to its first step. 0 disables it")
    ap.add_argument("--learn-chunk", type=int, default=None,
                    help="characters of conversation per optimiser step. "
                         "Defaults to training.chunk, which is tuned for "
                         "reading a corpus; try 256 for a short session, at "
                         "the cost of a denser diet than documents give")
    ap.add_argument("--save-every", type=int, default=8,
                    help="optimiser steps between writing the weights out")
    ap.add_argument("--precision", default=None,
                    choices=["bf16", "fp16", "fp32"],
                    help="what the forward computes in; defaults to whatever "
                         "config.yaml trains with")
    ap.add_argument("--no-tools", dest="tools", action="store_false",
                    help="disable the tool-call loop")
    args = ap.parse_args()

    from minagi.config import get as _g, load as _lc
    from minagi.precision import set_compute_dtype
    set_compute_dtype(args.precision
                      or _g(_lc(), "training.precision", "bf16"))

    global TOOLS_ON, MAX_TOOL_CALLS
    c0 = _lc()
    TOOLS_ON = args.tools
    MAX_TOOL_CALLS = int(_g(c0, "interop.max_tool_calls", 4) or 0)

    http_tool_mod.register_http_tools(REGISTRY, _g(c0, "interop", {}) or {})

    if not os.path.exists(os.path.join(args.weights, "core.npz")):
        raise SystemExit(
            f"no model in {args.weights}. Train one first:\n"
            f"    python3 train.py read data/train --save")

    global PRIME
    PRIME = load_prime(args.prime_chars)
    if args.prime_chars and not PRIME:
        print("  no corpus to prime from - the router will choose experts "
              "from the prompt alone", file=sys.stderr)

    if args.learn:
        print(f"\n  LEARNING IS ON. Talking to this model CHANGES "
              f"{args.weights}/ on disk.\n  Use --no-learn to serve it "
              f"read-only.\n")
    model = load(args.weights, args.device, learn=args.learn,
                 lr=args.learn_lr, save_every=args.save_every,
                 chunk=args.learn_chunk)
    if PRIME:
        print(f"  primed with {len(PRIME)} characters of self-knowledge")
    pool = getattr(model, "pool", None)
    if pool is not None:
        # Both numbers are the POOL. Comparing model.n_params(), which counts
        # the trunk plus whatever is resident, against the pool total reads as
        # the card holding more than the model has.
        print(f"  {pool.n_experts()} experts, "
              f"{pool.vram_params() / 1e6:.1f}M of {pool.n_params() / 1e6:.1f}M "
              f"on the card, context {model.cfg.block:,}")
    else:
        print(f"  {model.n_params() / 1e6:.1f}M parameters, "
              f"context {model.cfg.block:,}")
    print(f"\n  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
