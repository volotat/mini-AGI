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

# A passage of the corpus the conversation opens with. Empty when priming is
# off or no corpus is on disk.
PRIME = ""


def load_prime(chars, root="data/train/self-knowledge"):
    """
    Corpus text to open the conversation with, or "" if there is none.

    TWO THINGS IT BUYS, and they are separate. The router gets something to
    choose from: choose_for on "hi" has twenty characters to work with, and a
    paged model that picks its working set badly answers out of the wrong
    quarter of itself. And the learning stream starts part-way to its first
    step rather than twenty exchanges short of one.

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

    # A paged model loads with an EMPTY card - every slot -1. Generation
    # chooses a working set from the prompt before writing anything, so this
    # does not change a reply; what it changes is that the page can show
    # which experts are loaded before the first question instead of nothing.
    if PRIME:
        ids = STATE["tok"].encode(PRIME).ids[-model.cfg.block:]
        model.choose_for(torch.tensor([ids], device=dev), free=True)

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
            "n_slots": len(pool.slots)}


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


def _prefill_cache(model, tokens, chunk=512):
    """Encode one self-contained window and return its cache and last logits."""
    if tokens.shape[1] > model.cfg.block:
        raise ValueError("prefill exceeds the model context")
    caches = model.empty_caches()
    logits = None
    offset = 0
    chunk = max(1, int(chunk))
    for i in range(0, tokens.shape[1], chunk):
        part = tokens[:, i:i + chunk]
        logits = model(part, caches=caches, pos_offset=offset)[0]
        offset += part.shape[1]
    return caches, logits, offset


@torch.no_grad()
def stream(prompt, max_new):
    from minagi.config import get as _g, load as _lc
    from minagi.decode import pick_next

    c = _lc()
    strength = _g(c, "decoding.adapt_strength", 2.5)
    decay = _g(c, "decoding.adapt_decay", 0.88)
    # Characters between re-decisions of the working set. What the reply wants
    # after a hundred characters is not what the prompt alone asked for, and
    # it is a far shorter span than pool.segment_chars, which is for reading.
    reselect = _g(c, "pool.reselect_chars", 64)

    model, tok = STATE["model"], STATE["tok"]
    device = next(model.parameters()).device
    ids = tok.encode(prompt).ids[-model.cfg.block:]
    out = torch.tensor([ids or [10]], device=device)

    # THE PROMPT CHOOSES THE EXPERTS. A paged model loads with an empty card -
    # every slot -1 and every expert weight zero - so generating without
    # asking for experts first runs on the trunk alone. That is not a degraded
    # model, it is a different and much smaller one: 8M parameters of 468M,
    # and it reads as fluent-shaped nonsense. `free` drops the hysteresis that
    # keeps a working set steady while reading a stream, because a prompt is a
    # deliberate change of subject.
    model.choose_for(out, free=True)
    # Prefill in chunks, the way training reads a corpus. Feeding a long
    # prompt in one pass materialises activations for every position across
    # every block application at once, which is what puts a long context out
    # of reach; the cache carries the reach instead.
    caches, logits, offset = _prefill_cache(model, out)

    cur = out[:, -1:]
    produced = []
    # The prompt has already chosen the working set above. Report it, so the
    # page starts the reply showing what is actually answering it.
    yield {"swap": {"at": 0, "moved": None, "pool": resident_experts()}}
    for i in range(max_new):
        if reselect and i and i % reselect == 0:
            moved = model.choose_for(cur)
            if moved:
                # Only when something actually moved. choose_by_demand
                # refuses to swap unless a candidate beats a resident by
                # `margin`, so most re-checks change nothing, and animating
                # those would say a thing happened when it did not.
                yield {"swap": {"at": i, "moved": int(moved),
                                "pool": resident_experts()}}
        if logits is None:
            if offset + cur.shape[1] > model.cfg.block:
                # Cached keys have already been rotated at their old positions.
                # Trimming them and deriving a new offset from the shorter cache
                # reuses positions and corrupts every relative distance. Start
                # a self-contained recent window instead, as RecurCoder.generate
                # does, leaving half the context free before the next rebuild.
                keep = max(1, model.cfg.block // 2)
                recent = out[:, -keep:]
                del caches                 # let the replacement reuse its VRAM
                caches, logits, offset = _prefill_cache(
                    model, recent)
            else:
                logits = model(cur, caches=caches, pos_offset=offset)[0]
                offset += cur.shape[1]
        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
        logits = None                       # spent; the next pass recomputes
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
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
    from minagi.config import get as _g, load as _lc
    return jsonify({"learn": learn_state(),
                    "pool": resident_experts(),
                    "reselect": _g(_lc(), "pool.reselect_chars", 64)})


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
      <div class="mlab"><span>working set</span><b id="exp-n">-</b></div>
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
let MECH = {pending:0, chunk:0, steps:0, reselect:64, since:0, uids:[]};
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
  const r = MECH.reselect;
  if (!r){ $('exp-n').textContent = 'fixed'; return; }
  const left = r - (MECH.since % r);
  $('exp-n').textContent = busy ? (left + ' characters to the next re-check')
                                : ('re-checked every ' + r + ' characters');
  $('exp-f').style.width = (100 * (MECH.since % r) / r) + '%';
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
  if (d.reselect !== undefined) MECH.reselect = d.reselect;
  if (d.pool){
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
  // Generation re-chooses the working set from the prompt before writing a
  // character, so the countdown to the next re-check restarts every turn.
  MECH.since = 0; drawExp();
  const body = add('bot', '');
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
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
                  MECH.pending++; MECH.since++;
                  if (MECH.chunk && MECH.pending >= MECH.chunk)
                    MECH.pending = MECH.chunk;
                  drawMem(); drawExp(); }
        else if (d.swap){
          applyState(d);
          if (d.swap.moved){
            flare($('m-exp'));
            note(d.swap.moved + (d.swap.moved === 1 ? ' expert swapped'
                                                    : ' experts swapped'),
                 'swap');
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
    args = ap.parse_args()

    from minagi.config import get as _g, load as _lc
    from minagi.precision import set_compute_dtype
    set_compute_dtype(args.precision
                      or _g(_lc(), "training.precision", "bf16"))

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
