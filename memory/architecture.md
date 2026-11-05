# Architecture

**Status (2026-10-28).** Moved here from `ARCHITECTURE.md` at the repository root. Written 2026-09-02,
last changed 2026-09-27. It describes the early-September model and is NOT current: the
README's sections on the architecture, paging, growth and pruning, and the model, plus config.yaml and
state.md, describe the model as it runs. Superseded below:
- Depth: 2 prelude blocks and one weight-shared block applied up to 24 rows (09-06), not
  prelude(2) -> [recur(3) -> coda(1)] x 6. Still at most 26 block-applications; training samples depth
  around 12.8 rows (the 4.51 of 6 below is the old design).
- Pool: 397 experts of 3.15M parameters (SwiGLU 512 -> 2048 -> 512), top-8 among 32 resident, not 74 of
  98,304 at top-16. Growth by five brakes (config.yaml `growth:`), births at a small gate recombined from
  parents' hidden units; pruning by staleness since last admission, never by the gate.
- Selection: the text's vote admits each forward's 32; the exploration bonus under "Three tiers"
  was replaced by the balance term. From 11-04 (next start) the routing bias's floor rule lifts newborns on
  trial into the vote (`pool.route_floor`, E52).
- Context: grows one character at a time while the far end pays, not `ramp_context`; the cache
  costs ~52 KB a character (experiments.md E15b).
- Choosing a character: greedy with the adaptation trace (adaptation.md, `decoding.adapt_*`);
  `rep_penalty` 1.0 (off), no n-gram block. `contrastive_generate` exists; no commit measures it.
- Storage: a RAM tier of 384 experts (288 until 11-03) with bf16 moments - the momentum alone under Muon; routers.npz holds the gate and one router row per
  expert per site; core.npz includes the adapter.
- "Numbers are from the current weights" no longer holds: they are from early September.


How the model is put together, and why each part is the way it is. Numbers are
from the current weights; a script read them out of
`weights/manifest.json` and the run logs so nothing here had to be kept in sync
by hand.

---

## Characters, not tokens

The vocabulary is the 256 byte values plus 9 structural markers — 265 in total.

```
<think> </think>    a scratchpad the model writes and then reads back
<user>  </user>     conversation turns
<bot>   </bot>
<g>     </g>        a game
<|endoftext|>
```

Anything is already spelled in this alphabet: source code, arithmetic, chess
notation, dialogue. A new kind of data needs no new vocabulary, and only the
markers ever have to be agreed on.

This is not free. A character is worth less than a token, so the same text costs
more positions. What buys it back is that PonderNet's per-pass logits cost
`[B,T,vocab]` — about 1 GB at BPE vocab 8192 against 33 MB at 265. Per-character
adaptive depth only fits on an 8 GB card at character level.

---

## Depth by recurrence, not by stacking

Six distinct blocks: 2 prelude, 3 recurrent, 1 coda.

```
prelude(2)  ->  [ recur(3) -> coda(1) ] x up to 6 passes
```

which is `2 + 6 x (3 + 1) = 26` block applications from 6 blocks — **4.3x
compute per parameter**. `RecurConfig.n_layer_effective` is the authority on
that arithmetic; anything quoting it reads it rather than recomputing it.

The state carried between passes is *latent*. It is never turned back into
characters, so it is not a chain of thought and nobody — including the model —
can read it. What it is, is a point in the model's own representation space
being refined. Measured over one generation, successive passes move it less and
less:

```
pass 1 -> 2   1.280
pass 2 -> 3   0.563
pass 3 -> 4   0.357
pass 4 -> 5   0.261
pass 5 -> 6   0.209
```

Each character decides for itself when to stop (PonderNet halting, with a KL
term toward a geometric prior). Measured mean depth 4.51 of 6, range 1 to 6.

---

## The expert pool

74 experts, each a small SwiGLU network of 98,304 parameters. Every call site
routes into the *same* pool — there is no per-layer set of experts — and picks
the top 16 for each character.

Nothing assigns a speciality. Which expert handles what is settled entirely by
training, and a capability is spread across many experts at once. Measured:
each expert is used at a mean of 5.8 different call sites, and 64 of them at
more than one depth. That is the point of the design — capabilities compose out
of shared fragments rather than living in dedicated modules.

The architecture is therefore not a fixed object. Measured across one
conversational reply, **12.6 experts differ between one character and the next**
(range 7–19). Two adjacent characters of the same word are produced by different
arrangements of the pool.

### Growth and pruning

New experts are added on speculation, gated to exactly zero so they change
nothing on arrival, and kept only if training raises that gate.

```
reachable   hold while experts sit at a dead gate
earning     back off when the previous batch was pruned away
headroom    refuse when VRAM is nearly spent
```

Every one of those reads the **gate**, or memory, never routing traffic. The
reason is in `findings.md` §1: the load-balancing auxiliary loss manufactures
uniform routing on purpose, so any brake reading routing share is reading its
own objective.

That is one of two mistakes of the same shape. The other is in §2 — a
validation set so small that the differences being compared were smaller than
its own sampling noise. Both are cases of trusting a number without asking what
produces it.

---

## Reading is training

There is one way text moves through the model. `minagi/stream.py` walks a
character stream behind a KV cache, consuming it a chunk at a time. With
`learn=True` each chunk takes a gradient step; with `learn=False` nothing is
updated. Both cost the same, because both keep the same cache and the same
length of graph.

```
--chunk      characters inside one autograd graph. Sets peak activation memory.
--context    how far attention reaches. Sets the cache: linear, ~0.107 MB/char.
```

Because the graph is bounded by chunk rather than by context, context is nearly
free to extend. This is what makes batch 1 worth it: giving up gradient
averaging buys roughly an order of magnitude of reach on the same card.

**What does not cross a chunk boundary is gradient.** The cache carries the
model's view of everything it has read; backpropagation reaches only to the
start of the current chunk. The model can attend to something 30,000 characters
back and cannot learn a weight update through it.

### One live cache

A cache is cheap once and ruinous four times over. Holding one reader per corpus
at full context multiplies the largest term by the number of corpora — measured,
that turned a 5.8 GB run into an OOM. Since every window starts at a fresh random
offset anyway, there is nothing in a cache worth preserving across a switch, so
`StreamSet` keeps one active reader, reads a whole window, then picks the next
source by weight. Sources interleave at window granularity, and the memory is
that of a single stream.

### Extending the window

Positions are rotary and carry no learned parameters, so a longer context costs
nothing to represent. But the model has only ever been trained at the length it
was trained at, and degrades beyond it:

| context | held-out loss, same weights |
|---|---|
| 4,096 | 1.158 |
| 8,192 | 1.384 |
| 16,384 | 1.879 |
| 32,768 | 2.091 |

So `ramp_context` walks the window out over the first third of a run rather than
jumping to it.

A reader never reads past its own context, either — it starts a new window
instead. Sliding the window while holding the position offset looks equivalent
and is not: the cached keys carry the rotary encoding of the positions they
were computed at, so the relative distances stop matching. `findings.md` §3.

---

## Choosing a character

Nothing draws from a random number generator. At temperature zero the highest
scoring character wins, with a repetition penalty and an n-gram block so it
cannot fall into a loop. Contrastive search is available, scoring candidates
against the context already in the cache (Su & Collier 2022).

Output still varies, because the *state* varies: which experts are loaded, how
many passes were taken, what has just been read. Variety comes from the machine
being different, not from noise being added.

---

## Three tiers of storage

```
disk    every expert the model has, one file each with its Adam moments.
        Bounded by free space.
RAM     the most recently used experts, as CPU tensors. An expert leaving
        the card goes here first, and is written to disk when it is evicted.
VRAM    the experts the current forward admitted - at most `resident` of them.
```

Which experts are on the card is one rule, and every forward applies it
afresh. While a forward has room on the card, every character at every pass
ranks the whole pool with the router and asks for its top-k; the forward admits
experts in order of requested probability until the card is full. Every
character then routes among the admitted experts, so one whose request was not
admitted gets its best admitted experts instead. A forward is a training
window, a chunk of held-out text, a prompt, or one character being written.
An expert leaving the card is written back only if it was trained there.
While training, an expert used less than its fair share gets a small bonus
when experts are chosen, so rarely used experts still get tried; the bonus
never changes how much a chosen expert contributes.

Growing the pool moves the disk number. It does not move the VRAM number,
because a forward admits at most `resident` experts however large the pool
becomes. That is the arithmetic that lets the pool keep growing on a card that
never gets any bigger.

---

## The weights directory

`weights/` is not a serialisation of the model. It is the model.

```
manifest.json   what exists, its shape, where it came from, every gate value
core.npz        embeddings, attention, norms, halting head
routers.npz     one entry per call site
optim.npz       Adam moments, so a session resumes rather than restarts
experts/        one file per expert - the unit that is paged, grown and pruned
```

Only the experts are split out, because an expert is the thing that gets added
and deleted; everything else is always resident together and gains nothing from
being separate. Writes are atomic — every file goes to a `.tmp` and is renamed —
so an interrupted save cannot leave half a weight behind.
