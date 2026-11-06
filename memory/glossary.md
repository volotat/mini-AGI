# glossary

The project's own words, as the code, the run's log and the other files here use them, each with where it is defined.
Grouped by topic. Where a word changed meaning, the old meaning is given with its date.

## Units and clocks

- **character (char)**: one token of `minagi/tokenizer.py` ByteTokenizer, vocabulary 265: a UTF-8 byte, or one of 9
  markers that are single tokens (`<think>`, `</think>`, `<user>`, `</user>`, `<bot>`, `</bot>`, `<g>`, `</g>`,
  `<|endoftext|>`). Any other tag is spelled out byte by byte; the next stage's `<tools>`, `<call>`, `<result>` are
  ordinary text, because a new marker would change the vocabulary, which is fixed in the manifest (state.md, next
  stage). "Characters read" (`read_chars` in the manifest) is the run's clock; "3,321M" = 3.321B.
- **nats/char**: held-out and train loss, natural log, per character. bits/char = nats / ln 2 (the log prints both);
  perplexity = e^nats. Held-out overall = the mean over the 8 subjects. enwik9 and PG19 are reported in **bits per
  byte**; they hold no markers, so a character is a byte there.
- Every held-out, enwik9 and PG19 figure is the model's PonderNet objective (the mixture of per-row cross-entropies
  plus 0.01 x the halting KL), so an upper bound on code length (conventions.md, Measuring).
- **step**: one optimiser step, taken on one chunk of 2,048 characters (`training.chunk`); the manifest's `step`.
- **session**: the run between one start and the next stop; each start writes a banner into `runs/samples.txt`.
  A resume is a pause: never called a restart on graphs. timeline.md maps sessions to dates and characters.
- **the run**: run 2, begun 2026-09-16 from a fresh model and never started over. Run 1 (2026-09-03..09-16) is
  archived and left out of everything (timeline.md).
- **first pass**: the run having read as many characters as the corpus holds (8,037M); its percentage is the log's
  "(43.29%)". It does not mean every file read once: a visit draws a file at random with replacement, and lanes take
  different shares, so by then some lanes are read several times over and others less than once (state.md, Lanes).

## The model

- **prelude**: the 2 dense blocks every character goes through once, before the recurrence.
- **row**: one application of the weight-shared recurrent block (`n_recur` 1, `n_coda` 0), at most 24 (`max_steps`),
  at least 1. **block-applications** = 2 + rows, at most 26; the KV cache keeps one K,V per block-application (E15b).
  Before 2026-09-06 the unit was a "pass" of recur(3) + coda(1), up to 6 (architecture.md).
- **halting / ponder**: PonderNet. A halt head gives each character a probability of stopping after each row.
  Training samples the depth as poisson(`train_steps_mean` 12.8) + 1, clipped; inference stops at the first row past
  `halt_thresh` 0.9; the loss adds `ponder_beta` 0.01 x KL to a geometric prior of mean 1/`halt_prior` rows.
- **freeze-and-carry** (`halt_freeze`): a halted character stops being computed; later characters read its frozen
  state at the deeper rows (E11).
- **adapter**: merges the latent state with the embedded input at every row (`core.npz` `adapter.weight`), so the
  loop stays tied to the text; the latent is never decoded.
- **trunk**: everything outside the pool: embeddings, attention, norms, adapter, halting head (`core.npz`, 8.27M).
  It learns at x0.1 of the experts' rate (`training.trunk_lr_mult`, E9). The router rows, depth embeddings and gates
  belong to the pool and learn at its rate (`minagi/pool.py` `split_trunk_pool`).
- **expert**: a SwiGLU MLP 512 -> 2048 -> 512 (w1, w3, w2; 3.15M parameters), one file
  `weights/experts/eNNNNN.npz` with its own optimiser moments in bf16: Muon's momentum, ~19 MB (AdamW's two, ~25 MB).
  Known by a stable id (`uid`), not a position.
- **pool**: every expert there is: 397 since 1,553.9M characters, the ceiling `growth.max_disk_gb` 10 sets.
  **site**: a layer that calls the pool (a PooledMLP); each site has its own router.
- **router / router row**: per site, a linear map with one row per expert, scoring every expert for each character.
  A newborn's row is the share-weighted mean of its parents' rows.
- **gate**: one learned scale per expert multiplying its output (`pool.gate`, `routers.npz`). A newborn starts at
  `growth.birth_gate` 0.001. "Gated to rest": a gate near 0 (`python3 -m minagi.store weights` counts them). Pruning
  never reads it (since 2026-09-13).
- **top-k**: each character routes through its 8 best-scored experts among the admitted (`pool.top_k`).
- **forward**: one pass of text through the model: a training window, a held-out chunk, a prompt, or one character
  being written.
- **admit / the vote**: a forward's first pass adds each character's top-8 requests to its text's vote (everything
  since position 0); the forward is admitted the vote's top 32, and characters route only among those.
- **resident / slot / the card**: the 32 experts on the GPU at once (`pool.resident`), with gradient and moments.
  **loads**: experts copied onto the card.
- **RAM tier** (`pool.ram_cache` 384, 288 until 11-03): experts kept in system memory, moments in bf16. **hit rate**: the share of
  loads served from RAM rather than disk. Disk holds every expert.
- **paging / eviction / write-back / dirty**: moving experts disk -> RAM -> card and back. An expert leaving RAM is
  written back only if something trained it. Paging an expert in marks it dirty, so a tool that opens
  `weights/` without `read_only=True` writes expert files (conventions.md).
- **balance term** (`pool.balance`): a Switch-style loss on the router's probabilities over the whole pool, each
  expert charged by its recent share of admissions (`recent`, an EMA over ~1,000 training forwards); only router rows
  learn from it. It replaced the **exploration bonus** (2026-09-27 .. 10-02). Under Muon it barely moves
  the router (E52, E55).
- **routing bias / floor rule** (`pool.route_bias`, `pool.route_floor`): an offset per expert added to the router's
  scores where experts are chosen (each character's requests, the card's vote, its top 8), never to the mixing
  weights or the loss, stepped by `route_bias` after every optimiser step. The **curve** rule (route_floor 0)
  steers every expert toward a rank curve and was rejected (E35, +0.31). The **floor** rule (route_floor > 0,
  from 11-04) lifts only experts on trial whose share of training choices is below route_floor of an even share,
  falls back to 0 as they earn it, and is 0 off trial; lifted requests count lifted in the vote.
- **growth / newborn / birth**: every 2M characters (`growth.every_chars`) one expert (`growth.k`) is born if five
  brakes allow it: ROOM, USED, EARNING, FITS, HONEST (config.yaml `growth:`). A newborn is recombined from whole
  hidden units of several pool members.
- **trial**: an expert's first `prune.survival_chars` (100M characters), during which it cannot be pruned; at most 50
  at once (`growth.max_in_flight`).
- **staleness / the prune clock / last_seen**: characters since an expert was last admitted, as a share of
  `survival_chars`; every admission resets it (before, only the router's own choices did). **dying**: past
  `prune.dying_at` 0.65; **pruned**: at 1.0, file and all.
- **originals / grown**: the 31 experts left of the 64 a fresh model starts with (gates ~1.2, ~54% of routing) and
  the 366 grown since (median gate ~0.04) (E34).
- **loop practice** (`training.loop_practice_every`, `_lambda`, `_reps`; `minagi/practice.py`): after every Nth
  reading step, one more step on a real span of the window written out again; each copy is held to at most
  lambda times as likely as the one before (E57). **loop meter** (`tools/loop_meter.py`): greedy writing against
  real text on fixed openings, the sample prompts, the self-reinforcement curve and copying at open choices.
- **adaptation trace** (`decoding.adapt_strength` 2.5, `decoding.adapt_decay` 0.88): the decoder's loop guard; what
  was just written is suppressed, and the suppression fades (adaptation.md). Decoding is greedy: no random numbers.
- **the controller / plasticity / the governor**: the learning-rate controller (`minagi/plasticity.py`), run through
  `minagi/governor.py` by everything that learns. **scale** ("rate x0.012"): its multiplier on the base rate 3e-4,
  between `FLOOR` 0.001 (0.01 until 2026-10-29) and `CEIL` 1; printed to three figures since then (x0.00951).

## Reading and the corpus

- **lane**: a top-level folder of `data/train`, read as one stream (8 in the first pass). **subject**: a folder of
  `data/val`, scored as one; a lane feeds one or more subjects (chat feeds chat and chat_hermes).
- **visit**: one passage from one lane: a file drawn uniformly at random, a random offset, then up to 32,768
  characters (`data.passage`, 16 chunks), fewer when the file is shorter (a short file is read whole from its start).
  **window / context**: the 4,096 characters attention reaches back over (`context_now`); within a visit it moves a
  chunk at a time, and it is empty at each visit's start.
- **share**: a lane's share of the CHARACTERS read, set by `minagi/lanes.py`. worth = the sum over its subjects of
  |alpha| L / D, shares within [1/4, 3] of an even share, moving at most 5% a check, visits by smooth weighted
  round-robin in characters (**credit**): a lane of long files is visited less often. Shares were of visits until
  11-03 (E47).
- **self-knowledge**: `data/train/self-knowledge`, the repository's own source and docs plus generated answers about
  the model (`tools/rebuild_self_knowledge.py`, `corpora/self_knowledge.yaml`); rebuilt by hand, with no run training.
- **held-out**: `data/val`, never trained on. **eval**: `data/eval` (the PG19 test split; the next stage's ARC
  evaluation sets), never trained on. **the enwik9 slice**: `data/enwik9`, 50 MB never trained on.

## Held-out as the run measures it

- **check**: the held-out reading of each sample round, every 10 minutes of training (`checks.every` in the
  manifest): 8 whole blocks of 4,096 characters per subject, the same blocks every time.
- **start evaluation**: at each session start, 30 blocks per subject spread over the whole set; the level the
  session reports at.
- **level**: per subject, the start evaluation minus what the checks' own blocks read at the start; added to every
  check that session (`minagi/heldout.py`, `minagi/governor.py`). The log's `level -0.0111` is the mean.
- **banner**: the `# session started` block, with one `# held-out at the start, ...` line per evaluation the start
  ran (corrections.md quotes two).
- **moved / shifted**: a value from before a measurement fix, drawn moved by the difference measured at the fix,
  subject by subject; the raw log keeps it as measured (corrections.md).
- **frozen / dynamic evaluation (\*) / best effort (\*\*)**: a benchmark read without learning; read piece by piece,
  each scored and then learned from; whole pages or books in order, learning as it goes. The last two always on a
  **scrapped copy** of the weights, deleted after scoring.
- **arm / control / paired A/B**: runs from copies of the same weights that read identical text
  (`tools/ab_pool.py`); noise floor in E4.
- **alpha**: the exponent of held-out against characters read, L ~ D^-alpha, overall or per subject (E44).
- **6ND (est.)**: training compute as 6 x active parameters x data, priced as conventions.md says.

## A check in runs/samples.txt, line by line

```
step 1,699,345   3479.5M of 8,037M characters (43.29%)   1706 min   397 experts
context 4,096 characters of 4,096   reading 1,769 char/s   writing 26.7 char/s   still gaining -0.0102 deep into it
grad norm 2.64 against a clip of 1   clipping
train loss 0.5241   lr 3.67e-06   evidence t +1.50 over 65.7 (effect +0.0334)   rate x0.012
attention logit 17.88 at most   prelude.0 12.81   prelude.1 16.62   recur.0 17.88
held-out loss 0.5714 +/-0.0236 nats   0.8243 bits/char   perplexity 1.77   gap +0.0472   level -0.0111
  arithmetic 0.610   chat 0.472   chat_hermes 0.767   chess 0.379   code 0.476   reasoning 0.534   stories 0.450   wikipedia 0.884
repeats 26% of 8-grams, greedy with no guard
```

1. step; characters read, of the corpus size; minutes since the session started; the pool's size.
2. the window and its ceiling; reading (training) speed and the samples' writing speed; the window gain: the mean
   loss of a visit's first chunk minus that of its chunks from the third on (`train.py` `_window_gain`). With a
   4,096 window and 2,048 chunks it compares a passage's opening with its later text, so it is not a test of a longer
   window. The window grows on it only while below its ceiling; it has sat at 4,096 since ~17M characters.
3. the gradient norm before clipping, mean of the last 400 steps; "clipping" when above the clip (every check lately).
4. train loss, mean of the last 400 steps; the experts' learning rate in effect (the trunk takes x0.1); the
   controller's evidence: t of its slow fit's slope, the fit's effective number of checks, and the effect size
   (improvement per check in units of scatter) that actually moves the rate; the scale.
5. the largest attention logit before the softmax, per block, on the check's blocks: Muon's known failure
   mode is that this grows.
6. held-out at the start evaluation's level, +/- its standard error, in bits and as perplexity; gap = held-out minus
   the train loss above; level = the mean per-subject level.
7. each subject, at the same level.
8. the share of repeated 8-grams in the `[raw]` samples.

Then the samples: for each prompt, `[raw]` (greedy, no guard) and `[adapted]` (with the adaptation trace), each with
its own repeated-8-gram share; `samples.raw_only` (config.yaml, since 2026-11-06) writes `[raw]` alone, and
`samples.chars` sets the length, 140 by default. Older checks lack some of these fields; `minagi/heldout.py` `parse_log` reads every
format the log has had.

## Repository words

- **workshop**: the private repository the project was developed in until 2026-10-31; **published copy**: what
  `publish.sh` copied from it into this public repository until then. One repository since; the workshop's
  commits survive only as changes.jsonl, and "workshop only" in an older entry means a file that never reached
  this repository.
- **auto regions**: README blocks between `auto:` markers that `tools/refresh_readme.py` regenerates (run-blocks,
  weights-date, benchmarks, enwik, pg, scaling, params).
- **the author's hunks**: the author's uncommitted hand edits; agents never revert or rework them (conventions.md).
- **E-numbers**: entries of experiments.md.
