# decisions

Settled choices, reversals and rejected ideas, with the reason and the evidence. Status tags:
`SETTLED` (in force), `REVERSED` (was in force, undone), `REJECTED` (tried or proposed, not adopted),
`DEFERRED` (open, the author postponed it). Re-propose a SETTLED or REJECTED item only with new evidence,
and say what is new (conventions.md). E-numbers point into experiments.md.

## Goals and framing

- SETTLED: the thesis is adaptation (continual learning), not data efficiency or static scores
  (2026-10-17). Static benchmarks are context. Continual-learning benchmarks are planned; pair them with a
  standard transformer fine-tuned on the same new material (optionally with replay), trade-off reported.
- SETTLED: target ~1B parameters (2026-09-18). Parameters are the solved part (paging); corpus size and
  throughput bind (8,037M chars is ~10x short of a 1B model at Chinchilla/MambaByte ratios). When more
  experts are proposed, say what the data supports first.
- SETTLED: one continuous run, never restarted from scratch since it began on 2026-09-16; resumes are pauses
  (the word "restart" is not used on graphs). The run before it (2026-09-03..09-16) is archived and left out:
  most of what it counted was the same text re-read after every restart (timeline.md). Hardware:
  one RTX 3070 Laptop 8 GB; learning faster is to come from the code, not from a bigger card (2026-10-18).
- DEFERRED: bigger steps / chunk 4,096 (put off, 10-18); training on the RX 6800 XT 16 GB (put off, kept in
  mind for later).

## Model

- SETTLED: byte level, vocabulary 265 (256 bytes + 9 markers); the BPE path was removed (09-20 fixes found
  corpora still on it).
- SETTLED: one weight-shared block per row, up to 24 rows, PonderNet halting per character (09-06); replaced
  prelude(2) -> [recur(3) -> coda(1)] x 6 (architecture.md describes the old design).
- SETTLED: freeze-and-carry (`halt_freeze`, E11). REJECTED: skipping halted characters' attention (E12,
  2-4% at best).
- SETTLED: context grows only when the far end of the window still pays; ceiling 4,096 for the
  first pass. The next stage: 8K+ window and latent state carried across iterations (2026-10-28), so its
  corpus is not trimmed for length.
- REJECTED (as built): a latent vector carried between chunks, plain or gated (E15, 09-26; reverted
  entirely). That does not rule out latent state: only the broadcast mechanism was refuted;
  position-aligned carry or attention-readable memory slots remain untested.
- SETTLED: compute bf16; fp16 never (E13); fp32 on a GPU without bf16 when it has >= 14 GiB, else emulated
  bf16 (`tools/device_check.py`). Adam moments stored bf16 on disk and in RAM.
- REJECTED: gradient accumulation (E7, no gain; flag stays 1).
- Writing without loops (11-06, an agent's change, uncommitted until the author decides): loop practice
  (`training.loop_practice_every` 8, `minagi/practice.py`) teaches the weights that a repetition must not feed
  itself; from the same weights and 3M characters every 8th step cost +0.0003 held-out and took the sample
  prompts from 32.5% to 18.1% repeated 8-grams, every 2nd step +0.003 and 12.5% (E57). REJECTED: a tie
  preference against the copy at free choices (no measurable effect, E57); leaving the first copy free (no
  change, E57). Arithmetic's free-running loop remains: the lane's own argmax line repeated (E56). The
  `[adapted]` reading was to stay until raw greedy was nearly free of repetition. REJECTED (11-06): self-play in
  its place, the model's own continuation against the real one (E58: little effect on loops, a quarter of the
  reading speed); loop practice's held-out cost measured at ~0.0004 (E58). REVERSED (11-09): loop practice, off
  from the next start (`training.loop_practice_every` 0, the author's change): its held-out cost grows with the
  reading, ~+0.004 after 147M characters, wikipedia first (E61).
- SETTLED (11-09): the model writes by sampling, temperature 0.5, in chat and in the sample log
  (`decoding` in config.yaml, minagi/decode.py `Sampler`). Always-greedy writing was given up as a failed bet and
  the established rules measured against it: 0.5 wrote text as predictable to GPT-2 Large as the real
  continuation, half greedy's repetition, no loops; greedy + DRY 8 tied it on prose and lost on chess, spelling,
  the sample prompts and copying (E62). REVERSED: greedy as the only way of writing, with the adaptation trace as
  its guard. The sample log writes `[written]` (by `decoding`, random numbers fixed per prompt); `[raw]` plain
  greedy only when `samples.readings` adds it. REJECTED: typical sampling as defined (+0.25 bits on new
  material, arithmetic falls apart), anything drawn at temperature 1, DRY on top of sampling (E62).

## Learning rate and optimiser

- SETTLED: the rate is governed by held-out, never scheduled (minagi/plasticity.py). The
  controller in use is the 19 September one restored with spike and restart guards, at most 2% a
  check, 0.16 of the original gains (10-10). The author's aim: a controller that runs a continual learner
  on its own, with nothing adjusted by hand along the way.
- REVERSED: upward probing while held-out improves (10-06): it raised x0.058 -> x0.067 while held-out worsened
  (E29). REJECTED: a plateau controller cutting 10%: too much; 30% at once: too aggressive. REVERSED: a cosine
  schedule restarting per regime, then a four-day anneal (09-06..08), superseded by the held-out controller.
- SETTLED: the controller is part of the model (10-23): training, chat and every learning
  measurement run minagi/governor.py identically.
- SETTLED: trunk at x0.1 of the experts' rate (`training.trunk_lr_mult`, E9): it is what prevents
  forgetting.
- SETTLED: one learning rate for every expert (10-27). REVERSED: the newborn rate (10-16; neutral at 2 x 20M
  chars and no effect in E35), removed for simplicity.
- SETTLED: Muon for every weight matrix including the router (10-27; from 3,194.4M chars; E35).
  REJECTED: AdEMAMix (AdamW did better at the paper's settings; removed 09-18).
- SETTLED: the controller's floor is x0.001 (2026-10-29, the author's request); it was x0.05 until
  10-09 and x0.01 until 10-29, lowered each time because the rate had come down to it on a plateau.
  Easing is the resting posture (`T_MID` sits above the middle), so on a plateau the rate settles at the floor:
  the floor is the plateau rate, not just a bound. The way back up is unchanged, one doubling per confirmed
  jump in held-out, so from x0.001 it takes ~3 more doublings to reach x0.01 than it did.
- OPEN: the base rate. The 10-18 A/B (E30) favoured x1/3 (`--lr 1e-4`); the controller has since eased to
  x0.0122 on its own (10-29). A rate test under Muon (x0.04 vs x0.014, 30M chars) is proposed, not run;
  so is a test of the plateau rate itself (state.md).

## Expert pool and selection

- SETTLED: one selection rule (09-27): the router admits what each text asks for. Each forward's
  first pass adds its characters' top-8 requests to the vote of its whole text since position 0; the
  forward is admitted the vote's top 32; characters route only among the admitted, top-8 each.
  Removed with it (deleted, not switched off): segment router, expert keys, demand election, margin, dwell,
  auditions, key_floor, explore. The 32 is a VRAM limit per forward; generation may load experts as needed
  and the author accepted slower writing for that freedom.
- REJECTED: per-character admission while writing (E19, garbled words). REJECTED: selection temperature
  > 0 (E20; T=0.2 costs quality and speed and does not reduce repetition). Kept at 0.
- REVERSED: the exploration bonus (09-27; beta 0.35, then 0.65 by the author), replaced on 10-02
  by the balance term (E21) because reading and chat chose differently under a training-only
  bias. Use resets the prune clock.
- SETTLED: `pool.balance` is the author's: 0.003 committed, then lowered by hand in steps to 1e-5 by 10-23,
  raised to 2e-5 by 11-05 (timeline.md has each value; uncommitted in config.yaml; YAML reads `1e-5` as a
  string, train.py converts it with float()). Under Muon, the router's optimizer since 10-27, it barely moves
  the router at these weights (E52, E55).
- SETTLED: `pool.capacity_factor: 0` (no dispatch cap), the author's (E28; uncommitted in config.yaml).
- REJECTED: DeepSeek-style per-expert routing offsets / `pool.route_bias` (E21 at 0.001 0.7019; E35 +0.31).
  ADOPTED (11-04, the author's go-ahead; from the next start): the same offsets under the FLOOR rule
  (`pool.route_floor` 0.1, `route_bias` 2e-4) - only experts on trial, only upward, 0 off trial, lifted requests
  counted lifted in the card's vote - because no newborn was admitted under Muon (E52). The router stays on Muon
  (one optimizer family; the author dislikes mixing more). The curve rule stays rejected.
  The author was ready to switch it back on proof that it is much better; it was far worse. Usage
  follows usefulness and cannot be imposed.
- REJECTED: mixing after gates (`pool.gate_after`, plain and scaled, E35). REJECTED before testing: log
  gates (an agent's claim that they help small gates was wrong).
- SETTLED: a newborn is recombined from whole hidden units of several pool members; its router
  row is the share-weighted mean of its parents'. UNTESTED: `growth.birth_gate_parents` (the pool is at its
  ceiling; needs forced births on a copy).
- SETTLED: prune for atrophy (staleness since last use), not for a low gate.
- DEFERRED: pruning to free capacity. ~150 of 397 experts are removable at no measurable cost (E33), but
  the author postponed it (10-07): the capacity is wanted for what the model will read, and the problem to
  solve is getting data into it efficiently, not its size. On 10-16, after E33, the author had the 50 quietest removed
  together with the newborn rate, and 50 more went on 10-17; growth refilled the pool to 397 by 10-19.
- REJECTED: masking young experts (E32; they carry knowledge).
- STANDING CONSTRAINT: 10 GB of expert files on disk = 397 experts, reached ~1,538M chars; the pool has
  stayed at 397 since (growth and pruning still swap members).

## Paging and speed

- SETTLED: the RAM tier is the paging lever (E25, E26): 288 experts with bf16 moments. Direct `.npz` reader
  kept. REJECTED: write-behind paging thread with lazy moments (no gain, dropped at the author's
  choice); evicted experts written on their own thread were kept for when the pool outgrows RAM.
- SETTLED: an expert nothing trained is not written back; a load copies only what the card uses.

## Data and lanes

- SETTLED: a file is one document, a subject is a top-level folder.
- CHANGED (11-10, the author's call): a visit reads one subject as one stream - files joined by
  `<|endoftext|>` until the passage (32,768) is full - so the subject changes once a passage, as intended, keeping its
  experts on the card. Before, a visit ended with its file and most files are shorter (E63).
- SETTLED: reading shares are automatic (10-18; the author wanted them to set themselves, with no flag to
  turn on): worth = sum(|alpha| L)/D per lane, shares in [1/4, 3] of even, <= 5% a check. `--lane-schedule equal` and
  `--lane-weights` exist as overrides. The shares are of characters (11-03, the author's call; they had been
  of visits, so the characters each lane got did not follow its worth, E47).
- SETTLED: arithmetic in standard notation, scratchpads that derive every step (09-20): a scratchpad holds
  the working that leads to the answer, not an asserted result. Its floor (E39) keeps it at the 1/32 share.
- SETTLED: self-knowledge is a lane holding the model's own source and docs, without README
  samples or fenced sample logs; `memory/` excluded (2026-10-28).
- DEFERRED: dedup of chat_hermes held-out answers found verbatim in train (E37, ~9%).
- SETTLED: corpora build everything from public sources; weights are created on the first run. Until
  2026-10-31 the published copy also stood apart from the workshop's `tools/`; since then everything is one
  repository (Publishing).

## Measurement and reporting

- SETTLED: held-out = whole 4,096-character blocks spread over each subject, 30 per subject at a session
  start, 8 per check, checks reported at the start's level (corrections.md). History moved across fixes,
  no markers (the author rejected starred lines and footnotes).
- SETTLED: ~0.03 is the threshold for a real difference between separate runs; stderr printed with every
  loss; paired A/Bs (`tools/ab_pool.py`) for small effects.
- SETTLED: README numbers are generated; no slogans; references say only how each was used;
  labels scientific ("not then/now", no rhetorical questions); no "power law breaks here" claims (an
  episode retracted 09-19).
- SETTLED: enwik9 table: ~1B models only, none quantised, papers' own numbers, GPT-2 measured here on the
  same 2,000 pieces of 2,048 bytes of a 50 MB slice never trained on; rows frozen, dynamic evaluation (*),
  best effort (** whole pages, learning, all 17,426 pages). REVERSED by the author: the whole-pages
  "natural" regime and "read as" columns (10-20; judged too involved for the table), brought back as
  the best-effort row (10-21). The author reverted an agent's change that kept enwik9 to the 2,000
  pieces only; an agent re-applied it and reverted it again: the best effort over all
  pages stays.
- SETTLED: PG19 table against published byte-level models (MambaByte, MEGABYTE), whole test split pooled by
  byte; best effort over all 100 books.
- SETTLED: training-cost column in every comparison table (2026-10-21; method in conventions.md);
  mini-AGI at $0.081/h; published models at what their authors paid (AWS on-demand A100 $4.10/h for
  2023-24 papers), not the cheapest rental.
- SETTLED: the scaling chart (`tools/plot_scaling.py`) puts the run's bits per byte against data read, one
  held-out set per colour: PG19 purple (the model's diamonds, published byte-level models' circles), enwik9
  green (the model's squares, published scores whose training data is known, e.g. Llama-3.2-1B at 36 TB),
  plus per-subject exponents. A model without a published BPB on the same set is left off, never converted
  (09-18).
- SETTLED: long measurements (`./measure_benchmarks.sh`, ~a day of the card) are separate from the refresh
  before a push; `./release.sh` measures held-out only.

## Publishing

- SETTLED (2026-10-31): one public repository; what is tracked is published. REVERSED: the filtered copy
  (09-20 .. 10-31: publish.sh copied an INCLUDE list of tracked workshop files into this repository),
  retired with the commit-log generator it ran; `./release.sh` keeps its refresh and checks.
  `runs/samples.txt` and `memory/` (2026-10-28) are tracked; the notifier ships (2026-10-29) and stays
  silent without credentials (conventions.md). The author's own material is not tracked and not named.
- SETTLED (2026-10-31): the author examines every change and makes every commit and push; agents never
  stage or commit. REVERSED: agents committing freely (2026-09-27 .. 10-31, in the workshop).
- SETTLED: commit hashes in memory/ only of this repository's `main` (2026-10-31); never the workshop's,
  whose history is not public (10-29: none at all, while the workshop's was the only history).
- SETTLED: weights are never in git; undertrained snapshots go to Hugging Face
  (`Volotat/mini-AGI-undertrained`, publish_weights.sh) and may be deleted after the first pass. Final
  weights after the first pass.
- SETTLED: config.yaml comments say only what each knob does, no history or comparisons; code
  comments describe the mechanism, not how it was found; removals need no explanation in the
  code.
- SETTLED: one layer figure in the README (it was split in two, then went back to one: two took too much
  space).

## Next stage (after the first pass)

- SETTLED: synthetic executable tool lane first, then function-calling sets, then text environments
  (2026-10-28). One tool protocol: `<tools>` JSON Schema lines, `<call>{json}</call>`, `<result>` inside the
  bot turn.
- SETTLED: every dataset whole, no trimmed subsets; the builder states the disk and memory it needs and
  asks before starting (`python3 -m corpora agent`, `--check`, `--yes`).
- SETTLED: ARC-AGI-1/2/3 evaluation sets never trained on (data/eval copies; guard by id and by content
  under symmetries and recolouring); ARC-AGI-2 training tasks that are ARC-AGI-1 evaluation tasks (376)
  dropped unless `--arc1-eval-for-training`, which forfeits a clean ARC-AGI-1 score.
- SETTLED: licence-less sources are opt-in only (agentinstruct, NVARC sets, arc-community-unlicensed).
  REJECTED: sets built from evaluation tasks (pathwaycom/arc-task-gen, ARC-TGI), deliberate counter-example
  sets (arc-dataset-bad).
