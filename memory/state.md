# state

Snapshot as of 2026-10-31, at step 1,772,472. Everything here goes stale; refresh it when picking the project up
(`weights/manifest.json`, the tail of `runs/samples.txt`, `python3 -m minagi.store weights`, the README's
auto regions). timeline.md has the run's history session by session.

## The run

- Live: `python3 train.py read data/train --save --weights-dir weights --sample-every 10 --sample-log
  runs/samples.txt`, started by the author (agents never start or stop it). The run began 2026-09-16 from a fresh
  model (run 2; timeline.md); 90 sessions so far; this one started 2026-10-30 at step 1,756,345 (3,596.2M chars).
- Latest check (2026-10-31): step 1,772,472, 3,629.3M of 8,037M chars (45.16%), 397 experts, context 4,096 of
  4,096, reading 1,796 char/s, writing 23.5 char/s, train loss 0.5396, held-out 0.5689 +/-0.0234 (level -0.0119),
  grad norm 2.81 (clip 1, so every step is clipped), largest attention logit 18.12 (recur.0), lr 3.28e-06
  (base 3e-4 x0.0109, the controller's scale), repeats 23% of 8-grams greedy.
- Per subject: arithmetic 0.610, chat 0.472, chat_hermes 0.762, chess 0.377, code 0.475, reasoning 0.528,
  stories 0.448, wikipedia 0.879. Lowest held-out of any check so far: 0.5683 (step 1,755,293, 3,594.1M chars).
- `weights/` is the run's latest state, not its best: `read` saves at every check (the manifest's `val` is that
  check's held-out), every 5 minutes and when it stops (`val` null). There is no automatic fallback; older states
  exist only as the author's backups (`weights_*_bkp/`). Saving only on improvement, and `--revert-factor`, belong
  to `train.py stream`; the README and `minagi/store.py` `best_val` say so since 2026-10-29.
- Optimiser: Muon since 3,194.4M chars (the session of 2026-10-27); AdamW before.
- Pace on the wall clock ~127M chars/day lately, stops included (3,321.4M -> 3,629.3M in 58.1 hours); the first
  pass ends after ~4,410M more, about five weeks at that pace (early December 2026), if nothing changes.

## The model as configured (config.yaml working copy)

- 2026-11-06 (agent, uncommitted): loop practice on, `training.loop_practice_every` 8 (E57), from the run's next
  start; chat learning takes the same steps. The run was stopped at step 2,163,711 for the A/B.
- 2026-11-09: loop practice's held-out cost grows with the reading. The start-of-session level rose 0.5632 ->
  0.5661 over the 147M characters since it began (wikipedia +0.0185), ~+0.004 against the course before it.
  The training line's two steps (2,329.0M, 4,067.6M) are changes of the reading mix, not of the model (E61).
  The author turned practice off (`training.loop_practice_every` 0) from the next start.
- 2026-11-09 (agent, in f2839b6): the model writes at temperature 0.5 (`decoding`, E62), in chat from serve's
  next reply and in the sample log since the run's start at 16:41 that day (step 2,239,774), which writes
  `[written]` alone (`samples.readings`). Its first 75 blocks against greedy's last 75: the end of E62.
- 2026-11-10 (agent, uncommitted): a visit reads one subject as one stream, its files joined by `<|endoftext|>`
  until the passage is full (train.py `_Lane.open`), from the run's next start; before, a visit ended with its
  file. The loss log lists a visit's files at its first step. The run was stopped at step 2,320,858 for E63,
  which ended 2026-11-11.
  Expect wikipedia's held-out to drop about 0.011 in the first hours after that start: its held-out file is pages
  back to back, the old reader never showed a page after a page, and on the new one the held-out chunks that cross
  a page boundary fell 0.070 in an hour while chunks inside a page fell 0.0009 (E63). The automatic shares read
  the drop as a steeper slope: simulated from the saved schedule, wikipedia's share goes 16% -> about 21% over
  two to four days and is still about 19% after eight.

- d_model 512, 8 heads, 2 dense prelude blocks (SwiGLU d_ff 1408), 1 weight-shared recurrent block applied
  up to 24 rows (min 1), PonderNet halting (halt_prior 0.072, ponder_beta 0.01, halt_thresh 0.9),
  halt_freeze on, bptt_window 24, train_steps_mean 12.8. Vocabulary 265.
- Context: start 2,048, end 4,096 (reached at ~17M chars and held since); grows one character at a time, at most
  every 100,000 chars, while the far end of the window still gains >= 0.015 (asked every 65,536 chars). Chunk
  2,048; passage 32,768.
- Pool: experts SwiGLU 512 -> 2048 -> 512 (3.15M parameters each), top_k 8, resident 32 on the card, RAM tier
  384 from 11-03 (288 before; one moment per expert makes 384 the same RAM), capacity_factor 0, balance 2e-5 (1e-5 until 11-05; under Muon it barely moves the router, E55), select_temperature 0, gate_after false, route_bias 2e-4 with route_floor 0.1 (the floor rule, since the restart at step 2,075,068 on 11-04). A fresh model
  starts with 64 experts; 31 of the originals remain and carry ~54% of routing. 331 experts carry weight and 66
  are gated to rest (`python3 -m minagi.store weights`, 10-29).
- Growth every 8M chars (k 1, birth_gate 0.001), max_disk_gb 10 (the ceiling: 397 experts since ~1,538M while
  an expert carried AdamW's two moments, 25.2 MB; with Muon's one it is 18.9 MB and the ceiling ~529);
  prune survival 100M chars, dying_at 0.65.
- Newborns are not chosen (E52, 11-04): every expert born since births resumed on 11-03 (22 by step 2,045,378)
  was admitted to no forward and routed no characters, and those past their trial were pruned unused. Growth adds
  nothing until that is fixed. Most likely cause: Muon on the router - a row no text admits gets only the balance
  term's gradient, far below Muon's resolution, so it barely moves (E52). The fix runs since the restart at step
  2,075,068 (11-04): the routing bias's floor rule (`pool.route_floor` 0.1, `route_bias` 2e-4) lifts newborns on
  trial into the vote (E53). The first two newborns reached the card 1,669 steps after the restart, at an offset
  of 0.333. By 2,237 steps after it, 11 of the twelve had been admitted (1.32M characters), each held at its own
  offset by the floor's feedback, and held-out was flat. The first to end its trial was not admitted once in
  the ~3,000 steps after it, and in two hours no held offset fell: the rule brings newborns onto the card, but
  their own rows are not yet learning to be wanted. Held-out shows nothing beyond what a restart alone does
  (E54).
- Training: bf16, lr 3e-4 base, trunk x0.1, weight decay 0.1, clip 1.0, optimizer muon (all matrices).
- Parameters at step 1,622,136: core 8.27M, routers 0.20M, experts 1,248.9M (397 x 3.15M), total 1,257.3M;
  ~109M resident on the card; ~345M active per byte (two prelude blocks + attention + top-8 at 12.8 rows),
  which is the size the scaling chart's 6ND uses.
- Held-out (data/val): arithmetic, chat, chat_hermes, chess, code, reasoning, stories, wikipedia. Shares automatic
  since 2,329.0M (minagi/lanes.py), of characters since 11-03 (of visits before, E47).

## Lanes

From `runs/corpus_index.json` (files and bytes) and the manifest's `lanes` (10-29). Until 11-03 the schedule
allocated visits, and a visit was one file's passage of up to 32,768 characters (until 11-10), so a lane's share of the characters
read followed its visit share times its mean visit length (E47). Shares are of characters from the first start with
that change: it carries the visit shares it finds over as the characters they gave, then moves toward the targets. Characters read are measured per lane only since the automatic
schedule began at 2,329.0M; before that the run took equal turns of visits, and the "read" column estimates that
stretch from the mean visit lengths (the manifest instead records 291.1M each for it, an even split of 2,329.0M).

| lane | MB | files | mean visit (chars) | share of visits | share of chars since 2,329M | read, est. (M) | passes, est. |
|---|---|---|---|---|---|---|---|
| arithmetic | 149.6 | 749 | 32,768 | 0.031 | 0.069 | ~600 | ~4.0 |
| chat | 1,903.4 | 113,080 | 16,827 | 0.186 | 0.262 | ~570 | ~0.30 |
| chess | 327.3 | 1,640 | 32,754 | 0.060 | 0.138 | ~680 | ~2.1 |
| code | 429.8 | 41,786 | 9,692 | 0.188 | 0.123 | ~300 | ~0.7 |
| reasoning | 2,463.5 | 113,557 | 18,185 | 0.119 | 0.155 | ~470 | ~0.19 |
| self-knowledge | 7.1 | 497 | 13,794 | 0.031 | 0.030 | ~255 | ~36 |
| stories | 1,934.1 | 116,755 | 16,565 | 0.091 | 0.129 | ~410 | ~0.21 |
| wikipedia | 822.5 | 147,545 | 5,324 | 0.294 | 0.094 | ~190 | ~0.23 |

The code lane was rebuilt at 1,901.4M (issue #30), so its estimate before then is rougher. A visit draws a file
uniformly whatever its size and reads a short file whole, so "passes" is characters read over the lane's size, not
the share of its text seen.

## Benchmarks (latest measured)

- enwik9 (2,000 pieces of 2,048 bytes, 50 MB slice never trained on), weights of 22 Oct 2026 at ~2.8B
  chars: frozen 1.29, dynamic evaluation 1.23; best effort (all 17,426 pages, 23 Oct) 1.14. Reference:
  GPT-2 XL 1.07, Large 1.12, Medium 1.17, Small 1.33 (measured here), Chinchilla 1B 0.90, Llama-3.2-1B 0.68
  (papers). The 10-17 forecast of when GPT-2 Medium, Large and XL are reached: E46.
- PG19 test split (100 books, 41 MB), 23 Oct weights: frozen 1.995, best effort 1.447. Published byte
  models trained on PG19: MambaByte-972M 0.883, MegaByte 1.000.
- Held-out scaling exponent over the run 0.188 (R2 0.96); late windows flatter (~0.1).
  Start-of-session series (E64, 11-11): 0.119 over 2,386-3,194M, 0.081 over 3,194-4,430M, 0.052 over
  4,055-4,430M.
- Training cost to date at $0.081/h (RTX 3070 rental price): ~$56 at the 22-23 Oct measurement (~692 h).
- Re-measure with `./measure_benchmarks.sh` (about a day of the card; only when the author frees it).

## Open questions and next steps

- First pass: keep reading; the author publishes final weights after it. Undertrained snapshot on
  Hugging Face from 12 Oct 2026.
- The controller's floor is x0.001 (the author's request, 2026-10-29; it was x0.01), in force since session 89
  (2026-10-29, 3,492.6M), and the log prints the scale to three figures (x0.00951, not x0.010). Scale x0.0121
  at that session's first check, x0.0109 on 10-31. On a plateau the rate drifts to the floor, so the floor sets
  the plateau rate (decisions.md). Scale x0.00727 on 11-11 (4,799M). The open question is now whether the
  rate is too LOW (E64): the rule cannot see a rate that is too low, at this rate the weights move like a
  walk with no signal, and the short tests (E29, E30) could only show the noise floor falling. Proposed, not
  run: on copies, 60M chars at x0.0071 against x0.028, then two cool tails; and the enwik9 adaptation gain
  at x1, x4, x16 of the present rate (both in E64).
- After the first pass: `python3 -m corpora agent --check`, then the build (~114 GB of disk and ~5 GB of
  RAM by the 2026-10-28 estimate, more disk than was free then). Before running: accept xLAM's terms and
  `hf auth login`; `pip install alfworld`; uv provides Python 3.12 for ARC-AGI-3 (`arc-agi` 0.9.9); set
  ARC_API_KEY (else 3 games). TextWorld, ALFWorld and the ARC-AGI-3 player were never run against the real
  packages: expect to fix corpora/worlds.py and corpora/arc3_play.py on first contact.
- Next stage architecture (author, 2026-10-28): context 8K or more and latent state carried across
  iterations. Read experiments.md E15 (a broadcast latent carry failed; attention-readable memory slots and
  position-aligned carry are untested) and E15b (the KV cache costs 52 KB a character) first.
- What the next stage can change on these weights: the run-time settings (resident, ram_cache, capacity_factor,
  balance, select_temperature, shares, rate) and the window's ceiling (`model.context_end`: RoPE has no
  parameters, so a longer window costs a bigger table and the cache; E8 is why it grows gradually). The manifest's
  `cfg` fixes the shape (d_model, heads, d_ff, prelude/recurrent/coda blocks, expert width and depth, top_k,
  vocabulary; config.yaml says so above its `model:` block): changing any of it means a fresh model, the end of
  the one continuous run (decisions.md). A new marker would change the vocabulary, which is why the tool tags are
  spelled as text (corpora/agent.py). Whether latent state can be added to the trained weights without starting
  over depends on whether it needs new parameters; not worked out yet.
- The log's window gain has been near zero or below since ~1B chars (-0.010 lately) but compares a passage's first
  chunk with its later ones (glossary.md), so it does not say whether a window past 4,096 would pay; measure that
  on a copy (E8-style, loss by position) before relying on it.
- Continual-learning benchmarks: planned, not designed; pair with a fine-tuned standard transformer.
- Forgetting under the benchmarks' learning (dynamic evaluation, best effort) was never measured: the next step
  named on 10-19 was held-out before and after, on the adapted copy.
- ARC scores: only on data/eval copies; ARC Prize 2026 runs Kaggle offline tracks (check deadlines before
  relying on them).
- Proposed, not run: the learning rate under Muon (x0.04 vs x0.014 over 30M chars, on a copy).
- Proposed, not built (11-03): training among several participants from one checkpoint on their own data -
  rounds that merge the trunk and the shared experts by averaging deltas, and exchange only experts born on
  another participant's machine, chosen on the receiver's dev text and admitted as newborns in trial. A pilot
  first: two branches on disjoint subjects, against one model trained on the union.
- GitHub: 8 open issues and 11 open pull requests on 2026-10-31, not yet triaged. Issue #33 reports that the
  experts a forward admits are chosen from the whole forward's text, so a character's score can depend on
  characters after it. Still present on 11-07: scored causally (each character among the top 32 of the vote
  up to it, as writing does), held-out is +0.0084 +/-0.0014 nats/char above the published way over all 8
  subjects, nearly all of it in each block's first 2,048 characters, and cold enwik9 pieces +0.022 bits/byte
  (E59; E48 before it). Writing is not affected. Not to be fixed (decided 11-07): one path for training and
  evaluation is kept, which a per-position mask used only in scoring would split. The README says so instead: a
  note at the top of Benchmarks and a line under the enwik9 table, both outside the regenerated regions.
- Issue #17: serve.py stops at its startup banner on a weights directory no paged pool has saved yet - a fresh
  one from create(), as a training run without --save leaves it. Its manifest has no `paged` flag, so
  `_load_dir` builds the in-memory pool, which has no `vram_params` (nor `slots`, so no expert chips). Still
  present on 11-07, reproduced on the CPU. Directories the run has saved carry the flag and are unaffected.
  Proposed, not built: `_load_dir` reads every weights directory through the paging path, as training does,
  unless a caller passes paged=False; only `_load_dir` reads the flag.
- Held-out as measured is the model's training loss: 0.5658 at step 1,983,939 is halting-weighted
  cross-entropy 0.5588 plus the 0.01 KL halting term 0.0069; the returned logits score 0.5572 (E50). Reporting
  cross-entropy would be a measurement correction (conventions.md), not a gain; lowering `ponder_beta` or
  moving `halt_prior` would only shrink the KL part.
- Weight edits found nothing to take (E50): read-out, depth, temperature, gate and router edits are all at or
  above the unedited model on held-out. Inference at a 12-row ceiling scores the same as 24 and evaluates ~18%
  faster (not adopted; needs an inference ceiling apart from training's `max_steps`).
- Found in the weights (E49), not yet acted on: grown experts' router rows are near-twins (a child's row is
  the average of its 16 parents'); the gate vector's weight decay shrinks every expert's gate at every step,
  used or not; 153 of 265 byte embeddings have collapsed onto one direction (tied embeddings), so non-ASCII
  text is hardly readable; ~70 experts have gates below 0.01, and zeroing their gates costs nothing, but
  removing them costs +0.0014 held-out cross-entropy (E51): kept.
- Deferred by the author: pruning ~150 experts; chat_hermes dedup; bigger steps; training on the
  RX 6800 XT.
- If grad-norm spikes recur: skip steps far above the recent mean, or confirm a passage's loss on a copy
  (E40).
- The passage (`data.passage`, 32,768): read side by side at it, three subjects ended level with or ahead of an
  hour of each in turn - 0.0011 ahead at equal characters, level once a 15-minute shake followed the hours -
  at the same speed (E63, at the present rate). Shorter passages, and specialize-then-shake cycles shorter than
  an hour, are untested.
- Known, unfixed (reported 2026-09-27, still in the code 2026-10-28): `PooledMLP.aux` is overwritten at
  every row, so `pool_aux()` (weight 0.01) sees only the last row's value; a gate tensor replaced by growth
  or pruning is re-adopted by `train.py _resync_opt` into a group with weight decay 0 (1-D tensors).

## The working tree

- Everything uncommitted waits for the author, who commits it after review: agents' finished work and the
  author's hand edits (README prose, some config.yaml values, the odd test, table header or deletion) alike.
  Agents never stage or commit (conventions.md, Commits). What the running program writes (runs/samples.txt,
  weights/manifest.json, generated figures, the README's auto regions) goes in commits of its own.
- Tracked since the workshop's files moved here (2026-10-31): `tools/` (43 scripts), `tests/` (70 test
  scripts), `benchmarks/` (exact-match arithmetic, chess Elo, code; last changed 2026-09-20 and not part of any
  current measurement) and `experiments/` (last changed 2026-09-21); `replication/` was public already. Never
  tracked: `data/`, the weights' files, `.env`; of `runs/`, only `samples.txt`.

## The machine

- One RTX 3070 Laptop GPU, 8 GB, which the run keeps busy; system RAM is shared with the run (its RSS is
  ~12 GB at a 288-expert RAM tier of two-moment experts; 384 one-moment experts hold the same bytes). The AMD measurements were made on an RX 6800 XT, 16 GB.
- This repository is public (github.com/volotat/mini-AGI) and, since 2026-10-31, the only one; the run reads
  and writes `data/`, `weights/` and `runs/` inside its folder. The author commits and pushes. The
  weights' snapshots are on Hugging Face (`Volotat/mini-AGI-undertrained`, publish_weights.sh).
