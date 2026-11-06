# Adaptation, and what could replace it

**Status (2026-10-28).** Moved here from `docs/adaptation.md`. Written 2026-09-05,
last changed 2026-09-28; measured on step 72,979 (158 experts, 37.1M characters). Still
current: the adaptation trace is the decoder's loop guard (`decoding.adapt_strength` 2.5,
`decoding.adapt_decay` 0.88; the `adapted` samples in runs/samples.txt, against `raw`); chat learns once
per finished exchange (route C's verdict), at the run's own rate since 2026-10-19 (minagi/governor.py). Superseded: the machinery of section 5 (noted in place). The order of work in section
6 was not acted on: no commit measures `contrastive_generate`, a floor on the rate of genuine choices,
depth perturbation or expert rotation. The `runs/*comparison.txt` files it cites are not tracked (of `runs/`, git keeps only samples.txt).
2026-11-06: section 1's "no amount of training removes it" does not hold at the lengths that matter (section 7,
E56): the loops in the samples are learned, and published training objectives remove most of them under plain
greedy decoding. Built the same day: loop practice (`training.loop_practice_every` 8, E57), which removes the
prose and code loops in the weights; the trace stays for the arithmetic lane's loop, a tie it does not touch.


How the model is currently stopped from repeating itself, why that works,
where it fails, and whether the expert pool could do the job instead.

Everything measured here is on the model at step 72,979 — 158 experts, 32
resident, 37.1M characters read. Held-out error bar is ±0.06 nats throughout.

---

## 1. Why a deterministic decoder must repeat

Generation is a map. Let $h_t$ be everything the model carries at step $t$ —
the context, the KV cache, which experts are resident, how many latent passes
the halting head took. Let $f$ be the network. Then

$$
c_{t+1} \;=\; \arg\max_v \; f(h_t)_v ,
\qquad
h_{t+1} \;=\; g(h_t, c_{t+1})
$$

Both $f$ and $g$ are fixed functions, and there is no random draw anywhere.
So the trajectory $h_0, h_1, h_2, \dots$ is a deterministic orbit.

The context window is finite: $L = 32{,}768$ characters over a vocabulary of
$V = 265$ symbols. The reachable state space is therefore finite, and a
deterministic orbit in a finite space **must** eventually revisit a state.
Once $h_{t+p} = h_t$, the sequence is periodic with period $p$ forever after.

This is not a flaw in the weights and no amount of training removes it. It is
a property of iterating a fixed map. Extra capacity moves *where* the cycle
is, not *whether* there is one.

There are exactly three ways out:

| | change what? | status |
|---|---|---|
| **A** | the scores $f(h_t)$ | adaptation — what is implemented |
| **B** | the function $f$ itself | expert rotation — §5, unexplored |
| **C** | the state carried in weights | measured, harmful — §4 |

All three work by making the map **non-stationary**: the same $h$ must not
always produce the same $c$. Sampling does this too, by adding noise; the
point of adaptation is to do it without giving up reproducibility.

---

## 2. The mechanism

One counter per symbol, $\tau \in \mathbb{R}^{V}$, $V = 265$. At every
character:

$$
\boxed{
\begin{aligned}
\tilde{\ell}_t &= f(h_t) - \lambda \, \tau_t &&\text{suppress}\\
c_{t+1} &= \arg\max_v \tilde{\ell}_{t,v} &&\text{choose}\\
\tau_{t+1} &= \gamma \, \tau_t + e_{c_{t+1}} &&\text{decay, then mark}
\end{aligned}}
$$

with $e_v$ the one-hot vector for symbol $v$, and

$$
\lambda = 2.5 \quad(\texttt{adapt\_strength}), \qquad
\gamma = 0.88 \quad(\texttt{adapt\_decay})
$$

Unrolled, the trace is a geometrically weighted count of the recent past:

$$
\tau_{t,v} \;=\; \sum_{k=1}^{t} \gamma^{\,t-k}\,\bigl[\,c_k = v\,\bigr]
$$

so a symbol used $k$ characters ago still costs $\lambda\gamma^{k}$ logits.
That is the whole of it — no state is carried between calls, because the trace
is a *function of the history*. `pick_next` recomputes it from `prev_ids` each
time, which is why every existing caller kept working unchanged.

### Half-life

$$
t_{1/2} = \frac{\ln 0.5}{\ln \gamma} = \frac{\ln 0.5}{\ln 0.88} = 5.42 \text{ characters}
$$

| characters since used | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|
| logits of suppression | 2.50 | 1.94 | 1.50 | 1.16 | 0.90 | 0.70 | 0.54 | 0.42 |

### Why a loop cannot survive it

Suppose the decoder enters a cycle of period $p$ in which symbol $v$ appears
once per cycle. Its trace obeys $\tau \leftarrow \gamma^{p}\tau + 1$ each
cycle, whose fixed point is

$$
\tau_v^{\ast} = \frac{1}{1-\gamma^{p}}
\qquad\Longrightarrow\qquad
\text{penalty} = \frac{\lambda}{1-\gamma^{p}}
$$

Evaluated for real loop periods:

| period $p$ | steady penalty (logits) | example |
|---|---|---|
| 1 | **20.8** | `aaaaaa` |
| 2 | 11.1 | `abababab` |
| 7 | 4.2 | `, axis2` |
| 20 | 2.7 | `of the construction ` |
| 45 | 2.5 | `This content-like and showed that consiled.` |
| ∞ (used once) | 2.5 | ordinary text |

The escape condition is a **margin** condition. Writing $\Delta$ for the logit
gap between the looping symbol and the best alternative at that position, the
loop breaks as soon as

$$
\frac{\lambda}{1-\gamma^{p}} \;>\; \Delta
$$

**This is the key asymmetry.** A short loop is crushed — at $p = 1$ the model
would need a 20.8-logit preference to persist, which is $e^{20.8} \approx
10^{9}$ to one. But as $p$ grows, $\gamma^p \to 0$ and the penalty collapses
toward the baseline $\lambda = 2.5$, which is what any character pays for
being used once. **A long loop is charged almost nothing.**

### Why it does not damage ordinary text

A symbol of natural frequency $f_v$ has expected trace

$$
\mathbb{E}[\tau_v] = \sum_{k\ge 1}\gamma^{k}f_v = \frac{f_v}{1-\gamma}
$$

so it pays a standing cost of $\lambda f_v/(1-\gamma)$ simply for how common
it is. For a symbol appearing once in a hundred characters that is 0.2 logits —
nothing. For space, which is **20.6%** of all text, it is

$$
\frac{2.5 \times 0.206}{1-0.88} = 4.3 \text{ logits}
$$

which is *not* nothing, and §3 records what it costs. Note the asymmetry this
creates: the rule taxes a symbol in proportion to how ordinary it is, which is
backwards on the face of it — and yet §3 shows that correcting it makes things
worse.

---

## 3. What it achieved, and what it cost

Mean repeated 8-grams over six prompts, 300 characters each:

| decoding rule | repeats | held-out |
|---|---|---|
| greedy alone | 73.5% | — |
| greedy + `rep_penalty` 1.12 (the old rule) | 59.0% | — |
| **greedy + adaptation** $\lambda=2.5,\gamma=0.88$ | **11.6%** | unchanged |

### Why the rule it replaced failed

`rep_penalty` divided the logit of every symbol that had *ever* appeared:

$$
\ell_v \leftarrow \ell_v / \rho \quad \text{if } v \in \{c_1,\dots,c_t\}
$$

The set only grows. Measured on real samples, the fraction of the alphabet
already flagged:

| after | 20 chars | 40 | 80 | 140 |
|---|---|---|---|---|
| already penalised | 52.6% | 68.3% | 81.1% | **93.4%** |

A rule that penalises 93% of the options equally does not rank them. It was
not switched off — it had **saturated**. Adaptation cannot saturate, because
$\tau$ decays: the trace only ever describes the last ~10 characters, no
matter how long the text runs.

### Two refinements that were measured and rejected

**Frequency normalisation.** Charge a symbol only for use above its natural
rate, $\text{penalty}_v = \lambda\max(0, \tau_v - f_v/(1-\gamma))$, so space is
not billed 4.3 logits for being space. Principled, and **worse: 42.5% against
11.6%.** Loops are built out of common characters — `the construction of the`
is nothing but common characters — so exempting them is precisely what lets a
loop through. The over-punishment of space is doing necessary work.

**`no_repeat_ngram`.** Ban any $n$-gram that already occurred. Unusable at
character level, because real text repeats long spans:

| corpus | repeats a 12-gram | 24 | 32 | 48 |
|---|---|---|---|---|
| code | 100% | 100% | **100%** | 40% |
| chat | 100% | 100% | 95% | 81% |
| wikipedia | 100% | 0% | 0% | 0% |

Any $n$ strict enough to stop a loop also forbids ordinary code. Tried anyway,
it barely helped: the loop steps around the ban and resumes —
`of the construction of the construction OFF the construction`.

### The cost, stated plainly

Instrumented character by character, adaptation **overruled the model on 28 of
72 choices**, and what it fights hardest is the space character:

```
#31  chose _   penalty 2.53
#32  chose _   penalty 4.72
#33  chose _   penalty 6.66
#34  wanted _  penalty 8.35  ->  OVERRULED, forced to 'n'
```

Which is why the output contains `"   now,:\n"` and `""It's`. Adaptation is a
**patch on the scores, not a cure**. It buys coherence-at-a-distance by
spending well-formedness locally.

---

## 4. Why not weight updates (route C)

The obvious alternative: let the model read what it just said, so the state
that changes is the weights. Measured directly.

| what it learned from | repeats | held-out change | σ |
|---|---|---|---|
| 54 steps, own output only | 49.6% | +0.5855 | 9.8 |
| 54 steps, whole window per chunk | 42.7% | +0.6507 | 10.9 |
| 6 steps, one per finished exchange | 25.2% | +0.1018 | 1.7 |
| 5 steps, **real documents** (control) | — | +0.0197 | 0.3 |

Two distinct effects, which the control separates:

1. **Over-training.** Most of the top two rows is not self-training at all —
   it is 9× as many gradient steps landing on a few hundred characters.
2. **Self-confirmation.** What survives the control is real: the model's own
   output costs ~5× what the same quantity of real text costs.

The second has a clean explanation. The model learns from prediction error,
and its own greedy output is *by construction* what it already predicted best.
The gradient can only sharpen a peak that is already there:

$$
c_{t+1} = \arg\max_v p_\theta(v \mid h_t)
\quad\Longrightarrow\quad
-\nabla_\theta \log p_\theta(c_{t+1} \mid h_t) \ \text{ raises an already-maximal term}
$$

Confirmed by the diagnostic: loss on its own output fell 0.9408 → 0.4065 while
held-out rose. **A smaller learning rate does not fix this** — descent has a
direction, and a smaller step confirms the loop more slowly, not less.

It also *creates* adaptation's blind spot. Learning burned a 45-character
phrase into the weights, producing a loop with period far beyond the trace's
5.4-character memory.

---

## 5. Route B: could the expert pool do this instead?

> **The machinery this section builds on is gone (2026-09-27).** It was written
> against the selection rule of the time: a working set scored by demand before
> every chunk, a `margin` a candidate had to beat, a `dwell` period, the
> `since` buffer, and re-selection every 64 characters while generating. All of
> that was replaced by one rule, which every forward applies afresh: while it
> has room on the card, every character at every pass ranks the whole pool with
> the router and asks for its top 8; the forward admits experts in order of
> requested probability until the card's 32 slots are full, and every
> character then routes among the admitted experts. While the model writes,
> each character is a forward of its own, so the experts can change at every
> character; which ones is decided by the router's ranking alone, with no
> fatigue term. The analysis is kept as it was written.

The idea: rather than patching the scores, make $f$ itself non-stationary by
rotating which experts compute it. Penalise an expert that has been resident
too long so it becomes easier to displace.

### What the machinery already provides

| | |
|---|---|
| pool | 158 experts, 32 resident, top-8 per site |
| pooled call sites | 24 block applications |
| swap rule | candidate must beat the weakest resident by `margin` = 10% |
| hysteresis | a new arrival holds its slot `dwell` = 4 chunks |
| already tracked | `since[e]` — the segment an expert became resident |
| re-selection during generation | every **64 characters** |

`since` is the buffer a fatigue term needs, and it already exists.

### The proposal, concretely

Add a residency-fatigue term to the demand score, mirroring §2 exactly but at
expert granularity. With $r_e$ the number of segments expert $e$ has been
resident:

$$
\widetilde{\text{want}}_e \;=\; \text{want}_e \;-\; \mu \cdot \max(0,\; r_e - r_0)
$$

An expert past $r_0$ segments becomes progressively cheaper to evict, so the
resident set turns over even when demand is flat — which is exactly the
condition under which a loop persists.

### My honest assessment: it cannot replace adaptation, but it fits its blind spot

**Against it — the granularity is wrong by an order of magnitude.**

This is the decisive objection. Expert re-selection happens every 64
characters. The loops adaptation actually catches have periods of 1–20
characters:

$$
p_{\text{loop}} \approx 7 \text{ chars} \quad\ll\quad p_{\text{swap}} = 64 \text{ chars}
$$

Within one re-selection interval the function $f$ is *constant by
construction*, so `, axis2` repeats nine times before the experts are even
consulted. Rotating experts cannot break a loop shorter than the rotation
period. To fix that you would re-select every few characters — 24 sites × a
demand scoring over 158 experts, per character, plus the paging traffic of
swapping 32 experts in and out of an 8 GB card. That is not a decoding tweak;
it is a rewrite of the inner loop, and it would cost far more than the ~5% the
trace costs now.

**For it — it covers precisely what adaptation cannot.**

From §2, the penalty at period $p$ collapses to $\lambda$ as $p$ grows. A
45-character phrase pays 2.7 logits, essentially the same as a first use.
Those are exactly the loops that survive today:

```
This content-like and showed that consiled.
""
This content-like and showed that consiled.
```

And a 45-character period sits *comfortably inside* a 64-character
re-selection interval. **The two mechanisms are complementary, not
alternatives** — one covers $p \lesssim 20$, the other covers $p \gtrsim 40$.

**The real risk: forced rotation trades coherence for variety.**

Eviction currently requires a candidate to be *wanted* 10% more. A fatigue
term evicts an expert that is still the best available, replacing it with a
worse-matched one. Text quality would drop for the same reason adaptation's
overruling of space drops it — you are overriding the model's judgment. The
difference is one of blast radius: adaptation vetoes one character, a swap
changes the function computing the next 64.

There is also a quieter failure. `dwell` and `margin` were tuned so the pool
*settles* on unchanged text; the commit history records a bug where the working
set churned on noise. A fatigue term deliberately reintroduces churn, and
churn was measured to cost 2.09 nats when the resident set was wrong.

### A better version of the same instinct: make it event-driven

Rotation as a *constant* tax is what makes it costly. Rotation as a *response*
is nearly free. Loops are detectable — §3's detector fires on 0.6% of real
documents and 0% of code, chess, wikipedia and stories:

> the tail is a block repeated 3+ times, the block contains an alphanumeric,
> and it is not a single repeated character

So:

```
if a long-period loop is detected:
      force re-selection, with the current residents' demand
      scaled by (1 - mu) so the set genuinely turns over
otherwise:
      change nothing
```

This keeps every property that makes the pool work — settles on stable text,
no randomness, no permanent tax — and spends the disruption only where a loop
has actually been found. It is deterministic, so samples still diff.

**My estimate:** worth building, expect a *modest* win. It attacks the residual
that adaptation leaves, which is roughly the difference between 11.6% and the
~5% floor real text sits at. It will not replace adaptation, and I would not
expect it to beat it on the short loops that dominate today.

### Two other levers, ranked

**Vary the latent depth (promising).** PonderNet already chooses how many
passes to take, averaging ~2.28 of a possible 6. In a loop that count is
almost certainly pinned. Forcing $n_{\text{steps}} \to n_{\text{steps}} \pm 1$
on detection changes the computation at *character* granularity — no paging,
no swap traffic, and the model already trains at every depth, so the output
stays in-distribution. **This is the cheapest of all the ideas here and the
one I would try first.**

**Contrastive search (already in the codebase, unmeasured).**
`minagi/decode.py` has `contrastive_generate`, which penalises a candidate for
resembling the context in *hidden-state* space:

$$
c_{t+1} = \arg\max_{v \in \text{top-}k}\;
\Bigl[(1-\alpha)\,p(v \mid h_t) \;-\; \alpha \max_{j<t} \cos\bigl(e_v, h_j\bigr)\Bigr]
$$

This is adaptation's idea done in representation space rather than symbol
space, which means it is **period-blind** — a 45-character phrase has a similar
hidden state whenever it recurs, however long the gap. It targets the blind
spot directly, costs one extra forward per candidate, and it is already
written. Measuring it is an afternoon, and it may make the whole discussion
moot.

---

## 5a. Route D: a floor on surprise

Proposed: rather than tracking which symbols were used, track how *surprised*
the model is, and refuse to let it fall below a threshold. A loop, the
reasoning goes, is the model saying something it already knew it would say.

Surprise of the chosen symbol is its information content:

$$
I_t = -\log p_\theta(c_{t+1} \mid h_t)
$$

This is attractive for one reason adaptation cannot match: **it is
period-blind**. A loop is uninformative whether it repeats every 7 characters
or every 45, so route D attacks precisely the blind spot §2 predicts and §5
could not close.

### What the model actually does inside a loop

Measured during generation, real context, correct experts resident — 90
characters from the story prompt, which degenerated into `state the start the
start of the start the start of the`:

| | mean $I$ | median $I$ | $p_{\text{top}}>0.9$ | $p_{\text{top}}<0.2$ |
|---|---|---|---|---|
| **loop (generated)** | **0.916** | 0.866 | 20.0% | **18.9%** |
| real chat | 1.305 | 0.410 | 35.2% | 30.7% |
| real reasoning | 1.314 | 0.539 | 29.2% | 36.0% |
| real stories | 1.386 | 0.969 | 23.6% | 34.5% |
| real wikipedia | 1.781 | 1.096 | 20.2% | 40.4% |

Three things fall out, and only the third supports the idea.

**A per-character floor is impossible.** Between 20% and 35% of characters in
*real* text are predicted with $p>0.9$ — the `n` at the end of `constructio_`
is not a failure of imagination, it is English. Forcing $I_t > \tau$ at every
position would forbid finishing any word. Whatever this becomes, it must be a
statistic over a window.

**The obvious windowed statistics do not separate cleanly.** By mean, the loop
sits 0.53 nats below real text — a real gap, but real text itself spans
1.305 to 1.781 across domains, a spread of comparable size. A single global
threshold would sit inside the range it is trying to discriminate. Worse, by
*median* the loop (0.866) is **more** surprising than real chat (0.410) or
reasoning (0.539), so a median-based rule fires backwards.

**But the distributions differ in shape, and that does separate.** Look at the
last column. Real text is bimodal: it alternates between near-certainty inside
words and genuine choice at boundaries, so 30–40% of its characters are
positions where the model is really uncertain. The loop has 18.9%, and there
is **no overlap** with any real domain.

$$
\Pr[p_{\text{top}} < 0.2] \;=\;
\begin{cases}
0.189 & \text{loop}\\
0.307 \ldots 0.404 & \text{real text}
\end{cases}
$$

Which says something more precise than "a loop is confident". A loop is not
especially confident — mean $p$ of the chosen character is 0.500, and the
distribution is flat enough that the runner-up is often close. **A loop is
text that has stopped making choices.** It sits in a middling groove, neither
committing nor branching, and it is the *absence of genuine decision points*
that marks it, not high probability.

### Whether it could replace adaptation

**As stated — a floor on surprise — no.** The mean overlaps the range of real
text, the median points the wrong way, and per-character it is unusable.

**As a floor on the rate of genuine choices — plausibly yes, and it would be
strictly better than adaptation where it works**, because it is period-blind
and because it never overrules the model on a character the model is right
about. Adaptation's cost is exactly that it does: it forces the decoder off
space at 8.35 logits of penalty because space is common. A choice-rate rule
does not care what the symbol is.

A deterministic form, requiring no sampling:

$$
\rho_t = \frac{1}{W}\sum_{k=t-W+1}^{t}\bigl[\,p^{(k)}_{\text{top}} < \theta\,\bigr],
\qquad \theta = 0.2,\; W \approx 64
$$

While $\rho_t$ holds above ~0.28, change nothing at all — the model is behaving
like text. When it falls below, the decoder has stopped branching, and the
response is to take the runner-up at the next position where the margin
$p_{(1)}-p_{(2)}$ is smallest, which is the cheapest possible place to differ.

That last detail matters. Adaptation intervenes wherever the *trace* says to,
regardless of how confident the model is there. A margin-ranked intervention
spends its disruption at the position where the model was most nearly
indifferent — the least damaging character in the window to change.

### My estimate

Worth building, and I would rank it **above** expert rotation and alongside
`contrastive_generate`. It is cheap — one softmax per character, already
computed — deterministic, period-blind, and its threshold has a meaning that
can be read off the corpus rather than tuned by hand.

Two honest caveats. The numbers above rest on 3 documents per domain and a
single 90-character loop; the gap in the last column is wide, but it should be
re-measured properly before anything is built on it. And a rule that fires on
"too few hard choices" will also fire on genuinely predictable *real* text —
a licence header, a table of contents, a long stretch of boilerplate — where
the correct behaviour is to keep going. `p<0.2` at 20.2% for wikipedia is
already uncomfortably close to the loop's 18.9%, and wikipedia is the domain
this model reads worst.

---

## 6. Summary

Adaptation works because the decay makes it non-saturating, and it is strong
against exactly the short loops that dominate — 59.0% → 11.6%. It fails on long
periods for a reason that falls straight out of the algebra: $\lambda/(1-\gamma^p)
\to \lambda$.

Weight updates during generation are the wrong instrument, measured, and the
learning rate is not the reason.

Expert fatigue cannot replace adaptation — the swap period is 64 characters
and the loops are 7 — but as an *event-driven* response it fits the blind spot
that remains. (Written under an old selection rule; under the current one
every character of a reply routes among what its whole text has voted for —
see §5.)

A floor on surprise does not work as stated: the loop's mean sits inside the
range real text spans, and by median it is *more* surprising than real chat.
What does separate is the shape — a loop makes genuine choices at 18.9% of
positions against 30.7–40.4% for real text, with no overlap. A loop is not
confident text; it is text that has stopped choosing. Before building it, measure `contrastive_generate`, which is
already written and attacks the same residual without any paging cost at all.

Order I would work in:

1. Measure `contrastive_generate` — free, already implemented, period-blind
2. A floor on the *rate of genuine choices* (§5a) — cheap, deterministic,
   period-blind, and its threshold can be read off the corpus. Not a floor on
   surprise itself, which does not separate.
3. Ponder-depth perturbation on loop detection — cheap, character granularity
4. Event-driven expert rotation — real work, modest expected win
5. Constant expert-fatigue tax — I would not; it fights the hysteresis that
   was tuned to stop churn

*Numbers throughout are from runs recorded in `runs/decoding_comparison.txt`
and `runs/combined_comparison.txt`.*

---

## 7. Training the argmax instead of patching the scores (2026-11-06)

Research only; nothing here is built. E56 measured where the samples' loops come from, and the literature says
what removes them in the weights, with the decoder left as plain argmax.

**Two mechanisms (E56).** (1) A tie at a genuinely open choice broken toward the copy: the arithmetic digits
win 0.110 to 0.109 and repeat at 0.109/0.109 in every later line. The lane's generator has the same tie, and its
own character-by-character argmax is one line (`add 6 + 2 = 8`) drawn afresh each time, so a perfect predictor
of the lane loops too. (2) A copy, once made, raising its own odds: a repeated real sentence goes from 0.78 to
0.91 mean probability over six repetitions (none in arithmetic or chess), and the code loop's least likely choice
from 0.26 to 0.89. These are the two causes the literature names: argmax amplifying small biases at hard or open
steps (Pipis et al. 2025, on reasoning models), and the self-reinforcement effect (Xu et al. 2022).

**Section 1, revisited.** A deterministic map on a finite space must cycle, but the bound is the number of
possible windows, 265^4096, not 140 characters. The loops seen are learned behaviour. What a better predictor
cannot remove is mechanism (1): the argmax of a calibrated near-uniform choice is decided by noise. So the
writing has to be trained to depart from calibration at those choices. At true ties (the arithmetic digits)
that is free. At the wider near-ties of prose and code the copy is right about half the time, and demoting it
bluntly costs +0.0033 nats per character of held-out (E56). There the departure belongs in what the model writes
rather than in what it predicts, or must be kept small enough to act only on true ties.

**What has removed loops under plain greedy decoding, in the weights:**
- Sequence-level unlikelihood (Welleck et al. 2020): the model's own greedy continuations, tokens of repeated
  4-grams pushed down. Repeated 4-grams 0.442 -> 0.137 with perplexity 25.64 -> 25.42; 0.058 with the
  token-level term added (human text 0.006).
- DITTO (Xu et al. 2022): a real sentence repeated to fill the window; each repetition's probability trained
  toward lambda = 0.5 of the previous one's. Greedy sentence repetition 14.5% -> 2.85%, perplexity
  25.68 -> 24.33, preferred by people 84% to 16% over plain likelihood. It flattens the self-reinforcement curve;
  phrase-level repetition only halves (44% -> 22%, human 1.1%).
- Loop-onset preference (FTPO, Paech et al. 2025; applied to loops by Liquid AI in 2026): in the model's own
  low-temperature output, the first token of a loop is the rejected token, and its plausible top alternatives
  are the chosen ones. They are trained above it by a margin, with all other logits anchored to the starting
  model. Greedy loop rate 22.9% -> 1% on a 4B model, 10.2% -> 1.4% on a 2.6B model; rounds repeat because fixed
  loops expose new ones.
- Also: SelfCont (Guan and Huang 2023) factors an early checkpoint's repetition predictions out of training;
  repetition dropout (Li et al. 2023) masks attention to repeated n-grams in training data and costs perplexity
  (21.98 -> 28.26); MixCE (Zhang et al. 2023) helps sampling but leaves greedy loops.

**Boring against intended.** What separates them is the real text in the same context. Every term above that
works without damage is relative to it. It penalizes repetition only where the real continuation does not
repeat, and plain likelihood keeps refrains, code and tags.

**Proposed for this model (not built).** First a loop meter: per subject, self-reinforcement curves, the greedy
loop rate on real prefixes against their real continuations, and a probe of repetition the real text intends.
Then two cheap teacher-forced terms: a margin against the copy candidate at near-ties the real text does not
copy, and DITTO-style repeated spans of real text at character level. Then rounds of loop-onset pairs from the
model's own greedy writing. The trace retires when raw greedy matches the real text's repetition per subject.

Section 4's harm came from learning the model's own output as a target, which confirms the loop. These terms
learn against the loop instead: the same signal with the opposite sign.

**Built (2026-11-06, E57).** Loop practice, as proposed, at character level: the span's whole log-probability
held to at most half the previous copy's, one-sided, so the push lands on the open choices and spelling is left
alone. From the same weights and 3M characters, every 8th step cost +0.0003 held-out (noise) and took the sample
prompts from 32.5% to 18.1% repeated 8-grams, every 2nd step +0.003 and 12.5%. The tie preference (a margin
against the copy at ties, from real text) did nothing measurable, even on arithmetic alone, and was removed: in
real text the line before a tie is random, in writing it is the model's own mode line. Freeing the first copy
changed nothing. Arithmetic's free-running loop remains; the untested candidates for it are a generator whose
next line never repeats a recent line's start, and loop-onset preference rounds on the model's own writing.
