# history

Chronological record of the project, oldest first. Each entry: what changed, why, measured result. The full
message of a workshop commit (until 2026-10-31) is in `changes.jsonl`, found by its date and subject; later
ones are in git. Numbers are as measured at the time;
a later entry supersedes an earlier one. Session boundaries are from the working
transcripts of the author's agent sessions (2026-09-18 onward); before that the commits are the only record.

Names used throughout:
- **run**: the one continuous training run (`python3 train.py read data/train --save ...`), begun from a
  fresh model on 2026-09-16 and never started over since; stopped and resumed many times ("sessions"). History
  carries across resumes; restarts are pauses, not new runs. An earlier run (2026-09-03..09-16) is archived and
  left out of everything (timeline.md).
- **lane / subject**: one top-level folder of `data/train` (8 of them during the first pass: arithmetic,
  chat, chess, code, reasoning, self-knowledge, stories, wikipedia). Held-out subjects are the folders of
  `data/val` (arithmetic, chat, chat_hermes, chess, code, reasoning, stories, wikipedia).
- **chars**: characters (bytes) read by the run; the run's clock. "3,321M" = 3.321B characters.

---

## 2026-09-02 .. 09-05 - restructure, honest measurement, first corpora

- 38 flat scripts -> package `minagi/` (model, no CLIs) + `train.py`, `serve.py`, `benchmarks/`,
  `corpora/`, `experiments/`, `tools/`. Also the **KV-cache attention-mask fix**: `is_causal` aligns
  its triangle top-left; with P cached positions a query at P+i saw keys 0..i, not 0..P+i. Chunked loss
  4.0326 -> 1.2022 (= one full pass). Single-token decode had hidden it (1xN needs no mask). findings.md section 6.
- every run advances the model safely; only the best weights are kept; no `.pt` duplicates.
- **the evaluator measured its own noise**: 6 chunks/subject = 12,288 chars, sd 0.0275 across draws
  on fixed weights; a whole sweep compared differences of 0.0008-0.0114 inside that noise. Retracted the
  "streaming loses to batch" result (it was the instrument). `Evaluator` returns stderr with every loss.
  Run-to-run recipe variance ~0.014 (CUDA dispatch nondeterminism) -> "~0.03 is the threshold for a real
  difference". findings.md section 2.
- expert slots matched by identity (not position); dropped a VRAM-cache experiment.
- subjects interleaved by window; compute in bfloat16.
- 09-04 cluster (pool correctness): prune must not reshape a router under a live graph; a dry read
  could destroy the pool; pruning renumbered experts in the order that destroys them;
  per-expert records; demand elected experts that had earned nothing; held-out was
  measured with the wrong experts on the card; generation chose experts at random and the window
  could not grow; benchmarks scored the trunk alone; benchmarks pointed at existing
  corpora + enwik8.
- **the context window grows when the model earns it, one character at a time** (window extends
  only while the far end of the current window still helps); growth amount and cadence became separate settings.
- a subject is read in whole windows, with a floor. **A file is one document, a subject is
  a top-level folder** (the layout every later corpus follows).
- gradient norm shown, "so the learning rate can be argued about with a number".
- 09-05: self-knowledge becomes data and true (answers filled from the real model;
  stale answers stripped by question); reasoning traces in the existing markers (OpenThoughts);
  text-file cache + conversation and stories corpora.
- a grown expert could never earn its place (so none ever did) - fixed.
- two thirds of an expert file was Adam state at full width -> moments stored narrow (bf16).
- the loop guard marked a character forever (saturated, see adaptation.md section 3) -> decaying
  **adaptation trace** (lambda 2.5, gamma 0.88): repeats 59.0% -> 11.6%.
- generation-time learning measured (own output costs ~5x real text; adaptation.md
  section 4) -> learn from a conversation once per finished exchange only.
- `docs/adaptation.md` written (now `memory/adaptation.md`); surprise as a loop
  detector measured: mostly cannot replace adaptation.
- Growth gating: on the train/held-out gap itself, not a proxy; capacity paid for with
  information, not characters (later removed); judged on the network as it stands;
  routers sized by the pool.

## 2026-09-06 .. 09-13 - depth per row, the rate controller, prune for atrophy

- graphs redrawn whenever the sample log gains an entry (the plot hook).
- **one block per row**, so halting can stop after any row (the earlier design was
  prelude(2) -> [recur(3) -> coda(1)] x up to 6 passes = 26 applications; architecture.md describes that
  design); config.yaml decides how much recurrence depth training pays for.
- cosine schedule restarting per regime, then annealing over four days, then
  **the learning rate governed by held-out, not by a horizon** (minagi/plasticity.py). 09-09 fixes: the
  controller waited on a modulo, eases continuously on a windowless fit, measured
  patience not progress, seeding the fast fit from the slow made it not fast, down and
  up are different questions; 09-12: it raised where raising measurably hurt.
- after measuring what a newborn computes: **a newborn expert is recombined** from whole
  hidden units of several pool members, not cloned from one.
- one definition of a dead gate, stable expert ids, moments survive.
- **prune for atrophy, not for a low gate**; on 09-13 prune stops reading the gate, because the gate was
  pointed the wrong way; **dead means unaddressed** (staleness since last use) for the growth brake as well
  as the pruner; a cap on experts in flight (trial); "dying" replaces "dead gate" in the graphs.
- the read path saved Adam's moments and never read them back (resume lost momentum) - fixed.
- 09-10 memory/VRAM: dispatch had no capacity, only a comment; gradients reach the whole window,
  not the last 512 chars; serving reads one stream at training granularity;
  `--context` binds on resume; a VRAM probe tool; 09-11: RMSNorm kept an fp32 copy of
  every activation; a read could pass the rotary tables; the window grew past its
  ceiling (looked like a leak); `model.context` renamed `model.context_end`.
- **every cadence is configured in characters** (not chunks or steps).
- **self-knowledge is its own lane and contains the model's own source and every .md**
  (tools/rebuild_self_knowledge.py; see conventions.md "repo text is training data").
- measured how much of the 9.9 GB of experts is distinct.
- 09-14: every restart replayed the same text from the beginning -> lane RNG seeded with
  `[shuffle_seed, lane index, read_chars at session start]` (the replay recipe in experiments.md relies on it).

## 2026-09-14 .. 09-20 - forgetting measured, the public repo, arithmetic notation

- 09-16: **the run begins** from a fresh model (64 experts, corpus 7,874M chars), the graphs first cleared of
  the previous run's special cases. That previous run (2026-09-03..09-16, 609M chars by its counter, 112
  experts at the end) had re-read the same text after every restart until 09-14, so only ~66M of it was
  distinct; it is archived (`runs/archive/run-1`, not tracked) and left out of every fit (timeline.md).
- one window to watch the run in (the dashboard, `tools/plot_dashboard.py`).
- a sample re-picked its experts from one character every 64 characters (fixed later by the
  selection rewrites).
- 09-18: **the forgetting claim measured instead of asserted**: the trunk
  learning rate (x0.1 of the experts') is what prevents forgetting, not the expert pool (pool alone -12.5%
  spread; freezing the trunk removes 99.7% of forgetting; findings.md section 5); "the damage is
  displacement, not destruction".
- AdEMAMix dropped from results, then removed (AdamW did better at the paper's settings).
- 09-19: comments describe the mechanism, not how it was found. **The "19 September"
  rate controller** referred to later is this commit's plasticity.py.
- First transcript session (from 09-18), requests and outcomes:
  - Held-out accelerated around 147.4M chars (a resume); settled; single power-law fit alpha 0.221 (R2 0.94)
    at 236M. Contamination ruled out: 0 exact hash overlap val/train, 0.03-2.8% 64-gram overlap.
  - Scaling graph built (tools/plot_scaling.py), GPT-2/Llama points first added, then removed for
    comparability (the author: models without a published BPB are left off); the word "restart" removed
    from graphs (resumes are pauses, not restarts); a "power law breaks here" reading was retracted the
    next day and is not mentioned anywhere.
  - Chess: per-move legality p~0.906, ~9.65 legal moves before an illegal one; not Elo-ready (needs
    p>0.983 for a 40-move game). Samples now log legal moves, and the chess prompt was moved into the
    training distribution.
  - Depth by character class measured: space 7.00 rows, first letter 7.65, inside word 8.90, punctuation
    10.12 (the opposite of the first guess).
  - 09-20: **arithmetic in standard notation** (most significant digit first) and scratchpads that
    derive every step (no asserted results), per the author: consistency with how numbers appear elsewhere
    beats the reversed-digit trick, and a scratchpad holds the working that leads to the answer. Corpus
    regenerated (4,000,000 lines; checked 300,000 examples, all correct).
- 09-20 public repo: the published subset stands alone (nothing rescued from tools/, corpora build
  everything, weights created on first run, `publish.sh` at root, tracked files only); `runs/samples.txt`
  ships; the surface is leaned; config.yaml comments say only what each knob does, without the project's
  history. On 09-21 the fixed-window batch trainer is removed.
- **serve.py rewritten as a chat window that learns from the conversation** (on by
  default), opens with a corpus passage (prime), shows it, and shows the two mechanisms (meters, expert
  chips). Generation had been running on the trunk alone (8M of 468M) until `choose_for` fixed it - "What
  are you?" went from garbage to coherent self-knowledge.
- two externally reported bugs (chat and code corpora still on the removed BPE path;
  `stream` crashed on its first prune - the two pools now answer the same calls).

## 2026-09-22 .. 09-26 - PG19 scaling point, boundaries, continual-learning replication, auditions

- scaling graph on PG19 alone (strictly comparable byte-level models only); the
  model's PG19 point is the whole test split, 2.4496 BPB at that time. corpora/pg19.py pulls only the test
  split; tools/eval_pg19.py pools bits by byte; `--fraction` samples every n-th window (a prefix read
  title pages, 2.55 BPB).
- a boundary (new passage, prompt, reply) chooses its experts from its
  own text (the "canon" peek), not the last subject's; samples had been jumping because of it.
- README figures and numbers are generated, never typed (tools/refresh_readme.py auto regions).
- 09-23..24: **continual-learning replication shipped** (`replication/`, both
  runs, the weights on Hugging Face `Volotat/mini-AGI-cl-replication-weights`): PG19 read massed for
  1,048,576 chars at the operating LR; trunk x0.1 retains 97.30% vs 74% unmitigated; interleaved as a 9th
  lane, 100.13% (nothing forgotten). Measured at 428.2M chars, 174 experts. The README table comes from it.
- every README number audited; projection labelled so it cannot be misread.
- `pool.key_floor` to config, 0.25.
- every dashboard panel on the same last-100M-chars window.
- 09-26: **auditions** for idle experts: v1 (stalest) leaked (a kept
  incumbent reset its prune clock, 0 dying), turned off; v2 auditions the expert the text wants but the
  earned-discount buries, for one chunk, without resetting its clock. A resume drops what an abandoned
  branch wrote (`_truncate_history`).
- Session notes: generation `choose_for` fix; latent-carry experiments tried and reverted at the author's call,
  back to the state before them; measured the faithful-context memory: dispatch buffers x24 dominate, KV
  cache ~52 KB/char.

## 2026-09-27 .. 09-28 - the single selection rule

The author asked for one selection rule on every character, any expert reachable, minimal RAM/disk
traffic, no chunk-boundary reload, at most 32 on the card, without breaking the trained network.

- **one rule - the router admits what each text asks for**: every character at every
  row ranks the whole pool with the router and asks for its top 8; requested probability mass is summed;
  the text admits experts by mass until 32 are admitted; characters route only among the admitted (others
  masked to -inf). Removed: segment router, expert keys (power-iteration on w1), demand/choose_by_demand,
  margin, dwell, auditions, key_floor, explore. paged.py 1211 -> 850 lines. A/B: held-out 0.7036 vs 0.717,
  reading speed +6.3%.
- every forward admits its own experts (training window, held-out chunk, prompt, each written
  character); **an expert nothing trained is not written back**.
- a load copies only what the card will use.
- exploration bonus while training (explore_bias 0.35, later 0.65 by the author) and a
  prune clock that counts only the router's own choice.
- reading and writing speeds reported apart; the sample clock starts when a round ends (rounds had
  run back to back: ~60% of wall time was writing samples). The README projection uses the wall-clock
  pace. The author said they do not touch the code, so pending code edits are the agents' own.
- **moments travel only to a step, and every expert is stepped with its own** (C1; a slot's Adam
  state never belongs to the previous occupant); replication and live chat optimisers resume Adam's
  moments too.
- **freeze-and-carry** (`halt_freeze`): a halted character stops being computed; its
  K/V at deeper rows come from the adapter of its frozen state. A/B: ~0.001 emitted-loss cost, +15%
  training speed, -1.8 GB VRAM, +60% writing speed. On by default.
- **the text chooses**: per-character admission made writing incoherent (writing loss 1.3490 vs
  1.1111 on the same held-out text); now every forward's first pass adds its requests to the vote of its
  whole text since position 0, and the forward is admitted that vote's top 32.

## 2026-10-02 .. 10-05 - selection temperature, balance term, AMD/CPU

- 09-29 .. 10-02: the pool shrank 224 -> 118 (751M -> 980M chars) as experts grown in the days before reached
  the prune line unchosen: the clock counted only the router's own choices, not the bonus's. The
  author's account: they had been made by mistake while the dying clock was being reset. The author asked for
  the A/Bs below to be re-run on the smaller pool; the balance term then made every admission reset the clock.
- `pool.select_temperature` (Gumbel top-k draws of experts from the text's tally; tokens stay
  greedy). A/B at 15 min (from step 478,910): T=1 0.9264 garbled, T=0.5 0.8329, T=0.2 0.7124 (top-8 scored
  0.6656), vs 0.6538 baseline; sampling did not reduce repetition. **Kept at 0.**
- **a balance term replaces the exploration bonus; use resets the prune clock**. A/B/C of current
  rule vs per-expert offsets vs pool-wide balance: balance chosen. Term:
  `balance * (E * sum(share * P) - 1)`, share = EMA of admissions over ~1000 training forwards, input
  detached so only router rows learn. Author chose 0.003, then lowered it by hand in steps to 1e-5 by 10-23
  (uncommitted in config.yaml; each value in timeline.md). Pool then grew ~1 expert per 2M chars (156-161).
- README parameter count counted from the weights at every publish (auto:params).
- every PG19 measurement kept and drawn; PG19 point 2.091 BPB, alpha_pg19 0.153 (at that time).
- The README's layer diagram, drawn from the weights in the Transformer paper's style, was split in two
  and went back to one figure at the author's request: two took too much space.
- Measured: Spearman |gate| vs routing use +0.86 (contradicted a README claim about gates).
- 10-05 performance/portability: writing and held-out cast weights once and run only used experts
  (x1.57 writing on CPU); bf16 on CPUs with it in hardware; **attention in blocks where
  no fused kernel exists** + AOTriton on ROCm; lower-right causal bias for cached chunks; fewer host syncs
  per row; tools/device_check.py (shipped); serve.py past the window re-reads from position 0; q/k cast to
  v's dtype (fixed a crash at the first held-out); **a GPU without bf16 arithmetic computes fp32 when it has
  >= 14 GiB** (fp16 is impossible here, see conventions.md); README "GPU support" with measured speeds.

## 2026-10-06 .. 10-10 - rate-controller saga, reading speed, self-knowledge leak

- 10-06: rate probes upward while held-out improves; unconfirmed readings stop counting.
- each slot multiplies exactly its own characters (exact dispatch, no padded rectangle);
  blocked attention with its own backward. AMD RX 6800 XT fp32 reading 1,242 -> 1,388 char/s.
- expert files read straight into arrays (direct `.npz` reader `read_npz`: zip offsets +
  readinto, bit-exact, 3.6x faster). A write-behind paging thread was built, measured, and dropped at the
  author's choice (kept the reader).
- RAM tier measured as the paging lever (96 -> 200 experts: +37..42% whole-session throughput, hit rate
  0.63 -> 0.78); see experiments.md.
- Limiter analysis (10-07): ~30 strong early experts carry most routing; ~310 weak experts take ~40% of
  card slots; growth without pruning. The author postponed pruning: the model has to be big to store
  enough, and the problem to solve is getting the data into it efficiently.
- 10-08: **self-knowledge leak**: the model answered "What are you?" with its own old
  sample ("...up to 6 passes") quoted in the README and copied into the lane. Fixed: README samples and
  fenced sample logs stripped from the copy; answers in several wordings; then a test fixture quoting the
  same sample leaked it back within the hour (fixed with placeholder text). After ~22M chars the answer
  was right again.
- Disk cap: growth blocked at 10 GB = 397 experts (pool size has stayed 397 since). Expert-masking test:
  young experts carry real knowledge; 88 idle experts (gate < 0.03, past trial).
- `capacity_factor` set to 0 by the author (no dispatch cap) after measurement.
- Controller saga (the author's aim: a controller that runs a continual learner on its own, with nothing
  adjusted by hand along the way):
  - 10-09: floor x0.01; gentle settling on long plateaus; restarts keep the trend, not the level.
  - GradSNR meter crash on a step with one sampled pass (halt head got no gradient) - fixed.
  - Paired anneal probe (author stopped the run): x0.02 beat x0.058 on the same text at every check, all
    8 subjects (0.6123 vs 0.6191); the upward-probing rule had raised the rate while held-out worsened.
  - 10-10: a plateau controller (down-only rotation means); its first cut was 10%, which the
    author judged too much; the author then judged the original controller better than every change since,
    its one flaw being that sudden spikes moved it.
  - **back to the 19 September controller, at most 2% a check, blind to spikes and restart steps**;
    then 0.16 of the 19 September gains (a plateau takes a tenth in six hours, not half). This is the
    controller the run uses (minagi/plasticity.py; README "The learning rate is not scheduled").

## 2026-10-12 .. 10-16 - weights on Hugging Face, issue #30, the newborn rate

- `publish_weights.sh` pushes a snapshot of the weights to Hugging Face
  (`Volotat/mini-AGI-undertrained`): checks the run is stopped, mirrors with `cp -al` (safe: every save is a
  `.tmp` + rename), updates the model-card date and the README's auto:weights-date, commits with a live
  percentage, `git lfs prune`. README line: undertrained snapshot, 12 Oct 2026.
- the self-knowledge rebuild refuses while a trainer runs (it rewrites files a reader may open).
- README and generated text without slogans; references say only how each
  paper was used.
- **GitHub issue #30**: `looks_like_text` decoded a 4,096-byte sample strictly, so a file cut
  mid-UTF-8 character looked binary and was dropped silently - 1,663 files of this corpus (9% of one
  reader's). Fixed with incremental decoding (`final=not cut`, probe+1 bytes read), a verdict cache keyed
  on size+mtime (`_CACHE_VERSION` 2) so the fix reaches existing caches, and skips are now reported. The
  code lane was rebuilt (NUL separators had dropped 31% of it): corpus 7,880M -> 8,037M chars.
- Pruning capacity sweep (bottom-|gate| removal, nothing trained): 50 -0.0002, 100 -0.0003, 150 +0.0002,
  200 +0.0105, 250 +0.0144, 300 +0.0397 -> ~150 of 397 removable without measurable loss.
- 10-16: **newborn learning rate** (author's rule): a new expert's weights,
  router row and gate start at the full base rate and converge to the main rate as it is trained
  (the log-rate halves every 15,000 of its own trained steps, so it converges over weeks, not days);
  `--drop-quietest` removed the
  50 quietest experts (twice, 50 + 50) protecting those inside their trial; dashboard: thick line = main
  rate, thin lines per newborn. A/B 2 x 20M chars was neutral (0.5974 -> 0.5982); adopted
  anyway. **Removed again on 10-27**.

## 2026-10-17 - held-out read honestly, publish measures, enwik9 table

- The 0.5982 "dip" was a session-end evaluation (60 chunks/subject) appended with a test arm's log; such
  values are no longer drawn.
- **held-out checks read blocks spread over the whole set**, reported at one level everywhere
  (correction 1 in corrections.md; switch at 2,290.1M chars). After history drawn as logged with a starred
  line, the author settled on **earlier values moved onto the measured
  level, unmarked** (see conventions.md "a measurement fix adjusts history").
- `--drop-quietest` asks for the number back before deleting anything.
- publish.sh measures held-out on the weights as they are, refuses
  while a run is training, stops if held-out cannot be measured, reuses a measurement of the same weights
  (including a cleanly stopped run's end evaluation). It publishes code and progress, not weights.
- **enwik9 table** in the README, re-measured at every publish: ~1B models only, no
  quantised, scores from the papers' own tables (compression rate x 8), GPT-2 Small..XL measured here on the
  same 2,000 pieces of 2,048 bytes from a 50 MB slice the model never trained on; a training-cost column
  (the author wanted it in every comparison, estimated where nothing was published), costs at release time;
  mini-AGI's hours at $0.081/h (cheapest uninterruptible RTX 3070 rental; the author rejected $60 as too
  high, and $0.04 as an interruptible tier). Chinchilla 1B at ~$68k. First measurements:
  mini-AGI 1.3277 (step 1,132,995, 2,319.6M chars, 345 experts), GPT-2 XL 1.0730, Large 1.1189.
- enwik9 squares on the scaling chart, carried on to one corpus pass; Llama-3.2-1B at its
  36 TB as the only point with published data.
- Running time then: 610 hours on the RTX 3070 Laptop (~115 W).

## 2026-10-18 - paging lever, automatic lane shares, whole blocks

- **Adam's moments stay bf16 in RAM** (~25 MB an expert instead of 38): at the same RAM, hit
  rate 0.72 -> 0.87, reading 1,468 -> 1,901 char/s (+29%). RAM tier 288 experts. Evicted experts written
  on their own thread (no measurable gain yet; matters when the pool outgrows RAM).
- LR A/B (3M chars, checks off): x1/3 of the base rate won in all 8 subjects (experiments.md). Recommended
  `--lr 1e-4`; not adopted as such - the controller kept easing on its own.
- **each subject's share of the reading follows where it still pays** (minagi/lanes.py; the author wanted the
  shares to set themselves, with no flag to turn on). Worth = |alpha| L / D summed over the subjects a lane
  feeds; fit window TAU 0.15 in log characters; shares in [1/4, 3] of even, <= 5% a check; re-read discount R_STAR
  15.4 for unmeasured lanes; smooth weighted round-robin. Targets at 2,329M: chat 28.9%, wikipedia 16.7%,
  reasoning 12.9%, code 12.0%, stories 11.7%, chess 11.5%, arithmetic and self-knowledge 3.1% (floor).
- chat_hermes investigation (the author noticed its curve looked wrong): not reliable since the spread switch -
  short leftover blocks (73, 68, 32 chars) were 21% of its check; sd per check 0.0018 -> 0.0238. Also
  found: ~9% of chat_hermes held-out answers appear verbatim in data/train/chat/hermes (leakage; dedup
  deferred).
- **whole blocks only**, history moved onto that level (correction 2; switch at 2,386.0M).
- corrections moved into a README section of their own (since 2026-10-28 they live in
  memory/corrections.md).
- enwik9's own rate drawn only from measurements far enough apart (OWN_SPAN 1.25); PG19 all purple.

## 2026-10-19 .. 10-23 - dynamic evaluation, best efforts, the governor

- **enwik9 with dynamic evaluation** (score each piece, then learn from it, on a
  copy of the weights that is deleted afterwards; weights/ verified unchanged): frozen 1.3091 -> adapting
  1.2515 (gain 0.058 +/- 0.0015, 4.4%, still growing by quarter) at step 1,200,798.
- chat learns at the run's own rate (`LiveLearner.for_run`) and its saves keep the run's
  state - fixed a manifest wipe that would have restarted training at full LR, and a name-based parameter
  split that put routers at a tenth of their rate.
- a whole-pages "natural" regime and "read as" columns were tried and reverted at the
  author's request (judged too involved for the table; back to the 2,048-byte pieces).
- 10-21: brought back as a third row, **"best effort"**: the model as it runs (whole pages, its
  4,096 window sliding 2,048, learning as it goes on a scrapped copy), marked as not on the same footing.
  Best efforts read everything they are scored on - all 17,426 pages of the enwik9 slice (~13 h)
  and all 100 PG19 books (~10 h). The intent: show the model as it actually runs, learning from what it has
  just been scored on.
- long measurements move to `./measure_benchmarks.sh` (~a day); publish.sh keeps only held-out.
- **PG19 table** against published byte-level models (MambaByte-972M 0.883 as its
  best model; MEGABYTE's three), costs by 6ND at 40% utilisation priced at AWS's on-demand A100 ($4.10,
  May 2023..Apr 2024): the author wanted each cost to reflect what the model's makers would have paid,
  not the cheapest rental.
- the window, visit and growth-gap rules extracted from cmd_read into functions their tests test
  (a walkthrough of all 60 tests had found tests testing themselves).
- 2026-10-21: a one-line `open("README.md","w").write(...open("README.md").read())` truncated the README to
  0 bytes; restored from the published copy (see conventions.md, README is the author's).
- 10-23: **an agent re-applied a change the author had reverted on purpose** (enwik9
  on the 2,000 pieces only); reverted again. Rule in conventions.md.
- measure_benchmarks.sh ships with every script it runs; `python3 -m corpora
  enwik9` builds the 50 MB slice.
- **the learning rate's controller runs the same wherever the model learns** - minagi/governor.py
  shared by training, chat and every learning measurement (the author treats the controller as part of the
  model: it governs how fast the model adapts).
- Latest table values (22-23 Oct weights, 2.8 GB read): enwik9 frozen 1.29, dynamic evaluation 1.23, best
  effort 1.14 (between GPT-2 Medium 1.17 and Large 1.12); PG19 frozen 1.995, best effort 1.447.

## 2026-10-26 .. 10-27 - the alpha tests, Muon

The author's goal: a better power-law exponent alpha, so that training stays efficient, above all in the late
stage the run was entering; budget 12-24 h of tests. Switch list: weight after gates, log gates, routing bias,
remove the newborn rate, newborn gate as a share of parents', and Muon (Liu et al. 2025).

- A/B switches for choosing and mixing experts + `tools/ab_pool.py` (paired arms on copies of
  weights/, identical text, checks off, fixed rate and shares; 960 held-out blocks per arm; control twice).
- Muon for the weight matrices (`training.optimizer`), router included.
- Results (experiments.md): routing bias catastrophic (+0.31); after-gates mixing hurts (+0.15 probe,
  +0.008 scaled after 12M); newborn rate no effect; Muon -0.0028 at 12M, -0.0039 at 30M and growing.
  Log gates dropped before testing (an earlier claim that they help small gates was wrong).
- Findings: usage follows usefulness and cannot be imposed; the expert-usage "cliff" at ~30 is the 31
  original experts (gate ~1.2) against 366 grown ones (median gate ~0.036); originals carried 44% of
  routing at 2.0B chars and 54% at 3.1B; the pool is at the 10 GB disk ceiling (397) since ~1,538M chars.
  Late-alpha suspects: the controller's rate ratchet (x0.014 now) and the pool ceiling.
- 10-27: the author chose Muon and one learning rate for all experts, for simplicity - newborn-rate code
  removed; the run steps with Muon from the
  2026-10-27 14:34 session (3,194.4M chars). The largest attention logit is read at every check
  (Muon's known failure mode; 18.5 at 2026-10-28 13:59).
- A restart on 10-27 read slowly (~1,400 vs ~1,900 char/s): CPU contention from a test suite and another
  process, plus swap; recovered.

## 2026-10-28 .. 10-29 - arithmetic floor, the spike, the next stage's corpus, memory/

- Arithmetic's flat held-out traced to an exact floor (experiments.md): data clean, ~97.6% of its loss
  irreducible; the scheduler correctly holds it at 1/32.
- Grad-norm spike 9.25 at 3,300.8M chars: one passage (most likely a half-Japanese chat file), harmless
  (every step is clipped; Muon is scale-free; held-out recovered within ~6M chars).
- **the next stage's corpus builder**, `python3 -m corpora agent` (corpora/agent.py,
  toolbox.py, worlds.py, arc.py, arc3_play.py; not run yet - it waits for the first pass to end). Seven
  lanes: tools (generated, every call executed), function-calls, trajectories, arc, arc-synthetic,
  arc-traces, arc-games; one tool protocol; evaluation sets kept out; asks for ~114 GB before starting.
  Author's rules for it: take every dataset whole; state the space needed and ask; the next stage has an
  8K+ window and latent state across iterations, so nothing is trimmed for length. Details in state.md
  "next stage".
- `memory/` created (this folder): FINDINGS.md, ARCHITECTURE.md and docs/adaptation.md moved here; the
  README's in-flight corrections moved to corrections.md.
- 10-29: glossary.md and timeline.md added. Found while writing them (state.md): `weights/` is the run's latest
  state, not its best (the README's Layout and Initialization describe `train.py stream`); the lane schedule
  allocates visits, not characters (E47); the controller's scale (x0.0122) is close to its floor x0.01.
- 10-29: **the controller's floor x0.01 -> x0.001** at the author's request, so the rate can keep
  easing past x0.01 on the plateau; the log, the notes and the startup line print the scale to three figures
  (x0.00951 had printed as x0.010); README, config.yaml's comment and the self-knowledge answers say 0.1%. It
  reaches the run at its next start.
- 10-29: the notifier is ordinary code: publish.sh no longer leaves out `minagi/notify.py` or strips its five
  lines from train.py, and the commit log no longer hides its commits. It stays silent without its two
  settings; the README does not mention it (conventions.md, The model as it runs). Its test now uses a `.env`
  of its own, so it passes with or without the author's.
- 10-29: memory/ and the commit log lose every commit hash (the workshop's git is not published) and every
  trace of the author's wording; a hash a commit message cites becomes the subject of the commit it names.
- 10-29: the README's Layout lists governor.py, heldout.py, lanes.py and muon.py; it and `minagi/store.py` now
  say what `read` saves (its latest state) and that nothing rolls back on its own; "Adam moments" became the
  optimiser's. The self-knowledge answers name the optimiser from config.yaml (Muon), not Adam.

## 2026-10-29 .. 10-31 - one public repository

- 10-29: `./release.sh` takes over publish.sh's refresh and checks without copying anything: it refuses while
  a run trains, measures held-out, redraws the scaling chart and the architecture figure, refreshes the
  README's auto regions, then checks that every tracked `.py` compiles, that no `.env` is tracked and that
  the README's links lead somewhere. It never commits or pushes.
- 10-29: session 89 starts with the controller's floor at x0.001 (3,492.6M chars); the scale reads x0.0121 at
  its first check and x0.0109 on 10-31.
- 10-31: **one public repository.** publish.sh is retired, and with it the commit-log generator it ran:
  changes.jsonl is frozen as the workshop's log. The workshop's tools, tests, benchmarks and experiments are
  tracked from now on; its git history stays behind. The tests that read publish.sh hold release.sh to the
  same things (it redraws the architecture figure, runs none of the day-long measurements), and everything
  measure_benchmarks.sh runs and imports must be tracked.
- 10-31: the code no longer names the author's own material or writes into it. publish_weights.sh looks for
  the Hugging Face clone beside the repository's folder (`DEST` overrides) instead of at one machine's path.
- 10-31: ten inspection tools defaulted to `weights_rows/`, a directory the run left months ago; they default
  to `weights/`. Five that read `weights/` opened it writable (animate_routing, plot_routing, expert_grid,
  latent_flow, depth_proposal), so paging could write expert files back under the run; all open it
  read-only now.
- 10-31: memory/ for one repository: hashes of this repository's own commits on `main` may be cited, the
  workshop's never; the conventions on commits, the README and publishing follow release.sh.
- 10-31: agents no longer commit: the author examines every change and makes every commit. An agent hands
  over finished work uncommitted, with a proposed commit for each group of changes (conventions.md, Commits).

## 2026-11-03 - one moment under Muon, shares of characters

- 11-03: Muon keeps no second moment for the matrices it steps. AdamW's exp_avg_sq had been carried for them
  since the switch, unread: on the card (fp32, ~0.4 GB for the 32 expert slots), in the RAM tier, in optim.npz
  and in every expert file, a quarter of its 25.2 MB. Files, RAM entries and checkpoints now hold the momentum
  alone where Muon steps (`PagedPool.keep_second_moment`, set before the start's held-out check), and an
  expert costs 18.9 MB, so `growth.max_disk_gb` 10 holds ~529 instead of 397. A run that moves back to AdamW
  starts the missing second moment from the momentum squared, which holds its first steps near the rate.
  Training is the same step for step. Weights saved this way need this code to resume with an optimiser.
- 11-03: the reading shares are of characters, as the rule that sets them values characters (E47). The
  round-robin's credit is paid by the characters each visit read, not one a visit, so a lane of long files
  (chess, arithmetic: ~32,768 a visit) is visited less often than one of short files (wikipedia: ~5,300) and reads
  its share. A visit that reads nothing is charged as a full one. The first start carries the saved visit shares
  over as the characters they gave (each times its lane's mean visit, from the files' sizes), so the reading does
  not jump, then the shares move toward the targets at most 5% a check: wikipedia ~11% -> ~30% of characters,
  chess ~16% -> ~7%, arithmetic ~7% -> ~3% at the targets of 11-03. `equal` and `--lane-weights` are of characters too.
- 11-03: `pool.ram_cache` 288 -> 384: with one moment an expert takes 18.9 MB, so 384 hold what 288 did.
- 11-03: the weights read directly (E49) and 27 edits to them tested on a copy (E50): none improves held-out.
  What they found - near-twin router rows, gates decayed whether used or not, collapsed byte embeddings, the KL
  term inside the held-out figure - is listed in state.md, not yet acted on.
- 11-03: dropping the 70 quietest experts was measured before doing it (E51): +0.0014 held-out cross-entropy,
  where zeroing their gates had cost nothing (E50). Not dropped.

## 2026-11-04 - newborns that are never chosen

- 11-04: the run's own history shows that no expert born since births resumed on 11-03 has been chosen (E52): 22
  births, no admissions, the ones past their trial pruned unused. Earlier newborns were chosen from their first
  steps. Not acted on yet; the cause is untested.
- 11-04: **the routing bias's floor rule** (`pool.route_floor`, for the next start): an expert on trial taking less
  than a tenth of an even share of training choices is lifted 2e-4 a step where experts are chosen, falls back to 0
  as its own row earns the share, and has no offset off trial; a lifted request counts lifted in the card's vote.
  The DeepSeek-style offsets rejected in E21/E35 steered every expert toward a curve; this one lifts only newborns.
  README's growth paragraph and the balance paragraph's Adam clause updated to match (E52, E53).
- 11-04: the run restarted with the floor rule at step 2,075,068. After 1,110 steps the twelve experts on trial
  carried 0.222 each and every other expert 0; none admitted yet. Measured on that checkpoint's first pass, a
  newborn needs about 0.5 to reach most windows' cards; the first two were admitted earlier, at 0.333, 1,669
  steps after the restart. The twelve rise together, so the rule's feedback is what keeps them from taking the
  card. Within an hour 11 were admitted, each held at its own offset, with held-out unchanged. The first to end
  its trial was not admitted again on its own row (E54).

## 2026-11-05 - the balance term under Muon

- 11-05: `pool.balance` was 2e-5 at that day's restart (1e-5 from 10-23). Measured on the checkpoint of step
  2,115,355, the term's gradient is 0.12% of the router's momentum. Its part of Muon's step raises an idle or
  newborn expert's score ~0.0001 an hour. Carried on in one direction for a trial, it would change under half of
  a window's 32 card slots. Use is spread by the texts' votes, not by the term (E55).
