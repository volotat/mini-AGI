# experiments

Registry of tests, probes and A/Bs: setup, numbers, verdict. Oldest first. "held-out" is nats/char on
`data/val` unless said; lower is better; "arm - control" negative is better. Raw results live in `runs/`
(not tracked, so only on the machine that made them) where a path is given. Before re-running anything here, read its verdict; the
noise rules are in conventions.md "Measuring and reporting".

## Measurement instruments

**E1 evaluator noise (2026-09-03).** Repeated draws on fixed weights: 6 chunks 12,288 chars sd
0.0275; 24 chunks sd 0.0170; 96 sd 0.0093; 240 sd 0.0064. The sample is the window, not the chunk (chunks
correlate; per-chunk spread overstates the error ~2.5x). Same recipe run twice: 0.6594 vs 0.6735 (CUDA
dispatch nondeterminism). Verdict: stderr shown with every loss; ~0.03 threshold between separate runs.
Details findings.md section 2.

**E2 held-out read from the top vs spread (2026-10-17).** 30 blocks spread over each subject vs the first
30,720 chars: per-subject offsets -0.115 (chat_hermes) .. +0.094 (stories), all 8 -0.011. Verdict: spread
blocks; history moved (corrections.md item 1).

**E3 leftover part-blocks (2026-10-18).** Short leftovers (73, 68, 32 chars) were 21% of chat_hermes's
check, which read ~1.0 against 0.74 with noise 0.025/check (0.002 before). Whole blocks only; offsets
chat_hermes +0.086, wikipedia -0.040 (mostly re-sampling: 12 of its 30 picks moved), all 8 +0.004.
Verdict: whole blocks; history moved (corrections.md item 2). A subject's 30-block figure is good to a few
hundredths.

**E4 paired A/B noise floor (2026-10-26, tools/ab_pool.py).** Control run twice, 12M chars each on copies,
identical text, checks off, 960 held-out blocks: held-out +/-0.0005, paired training loss +/-0.00003.

## Architecture and training

**E5 KV-cache mask (2026-09-02).** is_causal top-left with a cache: chunked loss 4.0326, max logit diff
34.85 vs a full pass; lower-right mask: 1.2022, 0.177. findings.md section 6.

**E6 sliding the KV cache (early Sept).** Reading past the window by sliding while holding the position
offset: 2.60 vs 0.68 within the window (RoPE distances mismatch). Readers start a new window instead.
findings.md section 3.

**E7 gradient accumulation (early Sept).** 3.0M chars fixed: accum 1 0.6594 +/-0.0093, 8 0.6615, 64 0.6701.
Nothing; flag kept at 1. findings.md section 4.

**E8 context beyond trained length (Sept, old weights).** Same weights: 4,096 1.158; 8,192 1.384; 16,384
1.879; 32,768 2.091. Hence the window grows gradually (`ramp_context`, then earned growth). A ramp
to 8,192 once diverged (0.6524 -> 5.4490).

**E9 where forgetting lives (2026-09-18).** Four corpora read in sequence. Pool alone -12.5%
spread; freezing the trunk removes 99.7% of forgetting; trunk LR sweep x1 3.519/2.331 (mean/spread), x0.3
3.475/1.307, **x0.1 3.370/0.878**, x0.03 4.004/0.723, frozen 4.922/0.559. On mixed batches x0.3 also helps
(2.472 -> 2.380 mean). Verdict: trunk at x0.1 of the experts' rate (`training.trunk_lr_mult`). findings.md 5.

**E10 continual-learning replication (2026-09-23..24, replication/).** 428.2M-char weights, 174 experts.
PG19 read massed, 1,048,576 chars at the operating LR, batch 1:

| arm | PG19 | the 8 known | learned per nat forgotten | retained |
|---|---|---|---|---|
| experts frozen, trunk LR = expert LR | -0.2909 | +1.2663 | 0.23 | 74.12% |
| swapping, trunk LR = expert LR | -0.3106 | +1.2628 | 0.25 | 74.25% |
| swapping, trunk x0.1 (the run) | -0.4216 | +0.1297 | 3.25 | 97.30% |
| interleaved as a 9th lane | -0.3124 | -0.0065 | nothing forgotten | 100.13% |

Over the massed read only 44 of 174 experts received gradient. Freezing experts changed forgetting by
0.0035. Measured before the current selection rule (experts re-chosen per chunk then). Weights:
huggingface.co/Volotat/mini-AGI-cl-replication-weights.

**E11 freeze-and-carry (2026-09-28).** Halted characters stop being computed: ~0.001 loss
cost, +15% training speed, -1.8 GB VRAM, +60% writing speed. On.

**E12 halted characters already cheap (2026-10-05, weights_0.6695_bkp).** Characters still computed per row
(2048-char chunk): arithmetic 2048, 2048, 2047, 2040, 2030, 1993, 1948, 1890, 1798, 1637, 1316, 629, 140, 19,
5, 3; chess 2048 x6, then 1966 at row 10, 1110, 534, 237, 86, 33, 10, 1. Halted character-rows 15-25% of a
step; skipping their attention would save ~2-4%. Dropped.

**E13 fp16 (2026-10-05).** Recurrent state h peaks ~9,920 on arithmetic text (line starts after
"</think> N"), 664 on chess, <100 elsewhere; RMSNorm squares reach 98.5M; 3.7M values over fp16's 65,504.
fp16 + GradScaler for AMD RDNA2 dropped.

**E14 depth by character class (2026-09-19).** Mean rows: space 7.00, first letter of a word 7.65, inside a
word 8.90, punctuation 10.12.

**E15 latent carry between chunks (2026-09-26).** One 512-vector (the last latent) added at step 0 of the
recurrence at every position of the next chunk (~6x a character embedding), meant to travel across restarts
too. On the trained checkpoint, ungated: +0.24 nats, paired gap narrowing +0.56 -> +0.10. From one fresh
initialisation, 8M chars each (held-out / final-quarter train): control 2.3206 / 1.9868; ungated carry
3.5921 / 3.5453, gap widening +1.07 -> +1.56 by quarter, better on 3% of 3,912 chunks; gated (alpha * carry,
alpha learned from 0, trunk rate): alpha stayed within +/-0.001 (a consistent gradient could have reached
0.117), train 2.0006, i.e. the control within run-to-run noise. Reading: a vector broadcast to every
position is a position-independent bias that changes every chunk; attention cannot use it, so forced in it
costs capacity and offered it is declined. Not tested (structurally different): position-aligned carry
(last N latents seed the first N positions), memory as extra positions attention reads (RMT /
Block-Recurrent Transformer style), a learned projection, other injection points. All code reverted at the
author's call, back to the state before the experiment; per-chunk loss logs were kept only in that
session's scratch. Relevant to the next stage's latent state (state.md).

**E15b context cost (2026-09-26).** The KV cache holds 26 slots per character (every block-application):
26 x K,V x 512 x 2 bytes = 52 KB a character; 4,096 -> 0.22 GB, 8,192 -> 0.44 GB, 16,384 -> 0.87 GB.
Sparse long-range attention (NSA, Compressive Transformer, LongNet) cuts what is read, not what is stored;
memory for a long window is the cache plus dispatch buffers (x24 rows), which dominate.

**E16 gate vs use (2026-10-05).** Spearman |gate| vs routing use +0.86 (contradicted a README claim).

## Expert selection

**E17 single rule (2026-09-27, step 295,101).** Router admits what each text asks for vs the old
working-set rule, 15 min each: held-out 0.7170 -> 0.7036 zero-shot, 0.7111 -> 0.7011 after 15 min; reading
1,695 -> 1,802 char/s (+6.3%); loads/step 6.39 -> 2.43; VRAM same.

**E18 exploration bonus (2026-09-27).** beta 0.35 vs 0, 15 min: 128 vs 92 experts reached the card, young
39 vs 22, loads/step 3.67 vs 2.56, speed -5%, held-out +0.0007 vs +0.0013 (stderr 0.015). Adopted, later
replaced (E21).

**E19 per-character vs text vote while writing (2026-09-28).** Writing loss on the same held-out text: 1.3490
per-character admission vs 1.1111 with the text's vote. Per-character admission garbled words.

**E20 selection temperature (2026-10-02, 15 min from step 478,910).** Held-out (own rule / top-8-scored),
speed, loads/step, experts carrying 99%:
- A top-8 + bonus 0.65: 0.6538, 1,968 c/s, 3.9, 89
- D top-8, no bonus: 0.6547, 2,034 c/s, 2.4, 53
- F T=1: 0.9264, 1,031 c/s, 25, 115 - writing garbled, 3.0 c/s
- G T=0.5: 0.8329, 1,186 c/s
- H T=0.2: 0.7124 / 0.6656, 1,371 c/s, 16.3, 104 - writing repeats 30.9%, non-words 10.1%
Sampling experts did not reduce repetition. Temperature stays 0.

**E21 bonus vs balance vs per-expert offsets (2026-10-02/03, 15 min from 478,910).** Held-out, experts admitted
across held-out (59 at start): bonus 0.65 0.6541, 58; balance 0.01 0.6793, 114; DeepSeek-style offset 0.001
0.7019; offset 0.0002 0.6518, 73. The author chose the balance term (0.003 committed; later lowered by hand in
steps to 1e-5, timeline.md).

**E22 generation per-character loads (2026-09-27, step 298,919).** Writing admits per character: ~19.6
loads/char (9-30), 0 writes. Reply speed 188 ms/char warm RAM without an optimiser, 308 ms with learning on;
the old per-text rule 92 ms. Author accepted slower generation for freedom (writing later used
the text's vote too).

**E23 late-born experts (2026-09-21).** Routing received by experts born late vs originals: 3.4x less.

**E24 auditions (2026-09-26).** v1 (stalest + last_try) leaked: a kept incumbent reset its prune clock, 0
dying. v2 (best undiscounted fit among the suppressed, one chunk, no clock reset) worked but the whole
mechanism was replaced by E17 the next day.

## Paging and speed

**E25 RAM tier (2026-10-07, 360 experts, 10-min isolated runs).** 96 -> 200 experts: hit 0.63 -> 0.78,
reading ~1,280 -> ~1,560 char/s, whole session 792 -> 1,125 (+42%). Write-behind + lazy moments: no gain
(792 vs 823 at 96), dropped. Direct `.npz` reader: reads 3.6x faster, disk-read share 20% -> 6%,
end-to-end within noise (committed). RSS ~2 GB + ~47 MB/tier expert + 1.4 GB training. One 8-min run
carries +/-5-8% noise.

**E26 moments bf16 in RAM (2026-10-18, 350 experts, 9-min arms).** py-spy: paging 28.5% of the main thread
(eviction writes 13.5%, card->RAM copies 8.8%, disk reads 3.9%, moments to card 3.9%). bf16 moments in RAM
(~25 MB/expert vs 38): same RAM 288 vs 192 entries, hit 0.72 -> 0.87, reading 1,468 -> 1,901 char/s (+29%).
GPU kernel time ~0.9 s/step: matmuls 43%, add_ 13%, casts ~8%, pageable memcpys ~11%, AdamW 6%. ~9.6 of 32
slots swapped per chunk. Sample rounds write ~30 char/s (~11% of wall time at a 10-min cadence).

**E27 CPU and AMD (2026-10-04..07).** A laptop CPU alone, throttled: reading 25 c/s, writing 3.4 c/s, RSS
13.4 GB. Writing x1.57 on CPU after casting weights once and running only used experts. RX 6800 XT fp32
reading 1,242 -> 1,388 char/s after exact dispatch + blocked attention's own backward; bf16 there is 4.6x
slower than fp32. README "GPU support" table holds the measured rows.

**E28 capacity factor (2026-10-10).** The dispatch cap dropped ~7% of assignments; the author set
`capacity_factor: 0` (no cap).

## Learning rate

**E29 paired anneal probe (2026-10-10).** Same text, rate x0.02 vs x0.058: x0.02 better at every check and in
all 8 subjects; full held-out 0.6123 vs 0.6191. The upward-probing rule had raised x0.058 ->
x0.067 while held-out went 0.6255 -> 0.6290. Verdict: the 19 September controller with guards, gentle.

**E30 LR A/B (2026-10-18, step 1,137,561, 2,329M chars, controller at x0.032).** Four arms, same 3M chars,
checks off, saved, 60 spread blocks/subject:

| base lr | held-out change | subjects better | online train loss |
|---|---|---|---|
| 9e-4 (x3) | +0.0141 | 0/8 | 0.4996 |
| 3e-4 (x1) | -0.0010 | 4/8 | 0.4903 |
| 1e-4 (x1/3) | -0.0033 | 8/8 | 0.4902 |
| 3e-5 (x1/10) | -0.0026 | 8/8 | 0.4910 |

x1/3 best; a 3M test is mostly an annealing drop. Recommended `--lr 1e-4`; the controller reached lower on
its own (x0.014 by 10-26, x0.0133 on 10-28).

**E31 newborn learning rate (2026-10-16).** Rule on vs off, 2 x 20M chars: full evaluation 0.5974 -> 0.5982
(neutral). Adopted, then removed 10-27 (E35 showed no effect).

## Pool composition

**E32 expert masking (2026-10-10).** Masking young experts hurts (they carry knowledge); 88 idle experts
(gate < 0.03, all past trial); growth blocked at the 10 GB disk cap (397 experts).

**E33 pruning capacity (2026-10-16).** Removing the bottom-|gate| experts, nothing trained: 50 -0.0002, 100
-0.0003, 150 +0.0002, 200 +0.0105 (chess +0.045, stories +0.039), 250 +0.0144, 300 +0.0397. ~150 of 397
removable. The author postponed pruning (decisions.md).

**E34 usage anatomy (2026-10-26).** The usage cliff at ~30 experts is the 31 originals (median gate ~1.21) vs
366 grown (median 0.037). Originals' share of routing 44% at 2.0B chars -> 54% at 3.1B. Untouched weights at
0.5719: 394 chosen on held-out, top-31 share 0.559, n90 183, n99 332, effective number 140.5. Grown experts
get ~46% of picks; 52 gates above 0.1.

**E35 pool A/B rounds 1-2 (2026-10-26/27, tools/ab_pool.py, runs/ab_pool/).** Arms minus control:
- `pool.route_bias` 1e-3 (target 1/(rank+25), centered sign step): +0.31 train and held-out (0.8836);
  usage forced flat (effective 384) onto weak grown experts that dilute the originals. Usage follows
  usefulness and cannot be imposed.
- `pool.gate_after` plain: probe +0.152 +/-0.008 (nothing trained); "scaled" variant: probe +0.058, after 12M
  +0.008, stalled ~+0.007, usage narrower.
- `training.newborn_lr` off: -0.0004 +/-0.0003 (no effect).
- **Muon** (minagi/muon.py, every matrix incl. the router): 12M: -0.0028 +/-0.0003, better in all 8
  subjects, reading -6.4%. 30M: -0.0039 +/-0.0004; Muon improved 0.0024 from the start while AdamW drifted
  0.0015 worse (chat, stories, chess); training gap per 5M chars -0.0002, -0.0008, -0.0014, -0.0017,
  -0.0016, -0.0021 (still widening); reading -8.5%; usage narrower (n90 146 vs 186; 374 vs 395 used on
  held-out). Adopted 10-27.
- Not tested: `growth.birth_gate_parents` (the pool is at its ceiling; needs forced births on a copy).
- Proposed, not run: the rate under Muon (x0.04 vs x0.014, 30M chars).

## Corpus and data

**E36 contamination (2026-09-18).** 0 exact hash overlap of 548 val vs 535,596 train files; 64-gram overlap
0.03-2.8%. Held-out set: 240 chunks/subject cap, ~3.9M chars total then.

**E37 chat_hermes leakage (2026-10-18).** ~9% of chat_hermes held-out answers appear verbatim in
`data/train/chat/hermes`; other subjects' span overlaps are generic phrasing (chat 35/150 spans, stories
18/150, code 9/23, reasoning 4/150, wikipedia 1/150). Dedup deferred.

**E38 issue #30 (2026-10-12).** 1,663 files of this corpus were skipped as binary because a 4,096-byte sample
cut a UTF-8 character; the code lane had lost 31% to NUL separators. After the fix 0 regressions; corpus
7,880M -> 8,037M chars.

**E39 arithmetic floor (2026-10-28).** The generator's randomness (operands, digit counts, task, notes) sets
an exact minimum: -log P(line) / tokens = 0.5919 nats/token over val, 0.5931 over train; 0.594-0.597 on the
30 start-evaluation blocks, 0.614-0.615 on the 8 check blocks; both leave the same ~0.014-0.015 gap. All
4,000,000 train and 20,000 val lines are exactly what the generator writes. Learnable part (moved history
minus 0.596): 0.149 in the first 250M, 0.025 at 600M, 0.017 at 1.5B, ~0.015 at 2.3B, 0.0141 at 3.3B. The lane
sits at the 1/32 floor since ~2.39B (smallest worth; 1/32 of visits, ~7% of characters, E47). The floor adds a
constant 0.0745 to the 8-subject average, so alpha fitted on the total understates the learnable exponent by >=
15%. Method: parse each line, -log P with sum's digit cap and round's d=1/2 marginalised, tokens counted with
ByteTokenizer (think markers are one token), block-start partial lines as a range.

**E40 grad-norm spike (2026-10-28).** 9.25 at the check of step 1,612,069 (3,300.8M chars) vs ~2.6 normally
(the only other check above 5: 5.02 at 810M); train loss 0.674 vs ~0.53 in the same 400-step window.
Held-out +0.0019 (stories +0.010), back 5.6M chars later. Replay of the session's reading (each lane draws file
then offset from `np.random.default_rng([shuffle_seed, lane index, read_chars at session start])`, offset only
if the file's tokens > span+1, span 16 x 2048; interleaved by the schedule's round-robin at the manifest's
shares - landed within 7 of 62,000 steps) named the most likely passage: a half-Japanese chat file
(`chat/hermes/0039/part-079367.txt`, ~3,700 kanji). Harmless: every step is clipped (norm always > 1) and
Muon normalises before orthogonalising. If spikes recur: skip steps far above the recent mean, or confirm a
passage's loss on a copy.

**E41 chess (2026-09-20).** Per-move legality p ~0.906, ~9.65 legal moves before an illegal one; Elo needs
p > 0.983 for a 40-move game. Samples log legal moves; prompt fixed to the training distribution.

## Benchmarks

**E42 enwik9 (2026-10-17..23).** 2,000 pieces of 2,048 bytes from a 50 MB slice never trained on
(`python3 -m corpora enwik9`): mini-AGI 1.3277 (2,319.6M chars), later 1.3091 (2,458M) and 1.29 (2.8B);
GPT-2 XL 1.0730, Large 1.1189, Medium 1.17, Small 1.33 (measured here, same pieces). Dynamic evaluation
1.3091 -> 1.2515 (-0.058 +/-0.0015, gain growing by quarter: +0.042, +0.060, +0.061, +0.068; 38 min; 3.6 GB
VRAM). Whole pages in the 4,096 window ("natural"): 1.19 frozen-in-window, learning's own share 0.024; GPT-2
Small/Medium on pages 1.314/1.144 (left out of the README). Best effort over all 17,426 pages: 1.14 (2.8B).

**E43 PG19 (2026-09-22 .. 10-23).** Whole test split (100 books, 41,289,001 bytes), windows of 4,096 pooled
by byte: 2.4496 (09-22), 2.091 (10-03, alpha_pg19 0.153), 1.995 (10-23); best effort (all books in order,
learning) 1.447. Out of distribution: no PG19 or Victorian novels in the corpus.

**E44 scaling fits.** L proportional to D^-alpha on the run's held-out: alpha 0.221 (R2 0.94) at 236M; 0.188
(R2 0.96) over everything past warmup (README, 10-28); recent windows flatter: whole run 0.195, last 140M
0.155, last 100M 0.095 (10-27). Wikipedia held-out over its last billion chars alpha 0.107 (enwik9 projection).

**E46 GPT-2 catch-up forecast (2026-10-17).** At 2,319.6M chars enwik9 read 1.3277, level with GPT-2 Small (1.33,
measured here on the same pieces). Forecast for the larger GPT-2s, with the Wikipedia held-out subject as the proxy
(enwik9 assumed to move with it) and its local power-law slope, which kept flattening (0.156 over the run, 0.08 over
the last 1B chars, 0.06 over the last 300M). At 0.08: Medium (1.17) at ~10.8B chars read, Large (1.12) at ~19.7B,
XL (1.07) at ~33B, i.e. 1.3, 2.5 and 4.1 passes over the corpus; at the pace then (~100M chars a day) about January,
April and August 2027. Since: 1.3091 at 2,458M and 1.29 at ~2.8B, where that slope gives 1.31, so the run is a
little ahead of it; the pace is now ~134M a day. The README's chart carries enwik9 on at its own measured rate
instead.

## Next-stage corpus (2026-10-28)

**E45 converters on real rows.** Each corpora/agent.py converter run on ~200 rows per dataset read from the
Hugging Face dataset viewer (in memory, nothing saved): ~100% kept (2,789 of 2,790; the dropped Toucan row
has a malformed call). Toucan's three big sets write calls as `function_call` + `function` results (a
first converter would have lost every call); Nemotron-SFT writes grids with spaces. When2Call's SFT set has no
calls by design (its calls are in the preference set). Generated tools lane: 5M chars in 19 s, every call
replays identically.

## Reading

**E47 lane shares are shares of visits (2026-10-29, nothing run).** `minagi/lanes.py` `pick` pays one credit per
visit, and a visit (`train.py` `_Lane.open`) draws a file uniformly, takes a random offset if the file is longer
than the 32,768-character passage, else reads it whole. A lane's mean visit is the mean over its files of
min(size, 32,768) (`runs/corpus_index.json`). Characters each lane read since the schedule began at 2,329.0M (the
manifest's `lanes.read` minus the 291.1M each was seeded with), against the visit shares of 10-29 (shares moved
over the period, so the ratios are approximate):

| lane | share of visits | mean visit (chars) | share of chars since 2,329M | chars per visit share |
|---|---|---|---|---|
| wikipedia | 0.294 | 5,324 | 0.094 | x0.32 |
| code | 0.188 | 9,692 | 0.123 | x0.65 |
| chat | 0.186 | 16,827 | 0.262 | x1.41 |
| reasoning | 0.119 | 18,185 | 0.155 | x1.30 |
| stories | 0.091 | 16,565 | 0.129 | x1.42 |
| chess | 0.060 | 32,754 | 0.138 | x2.30 |
| arithmetic | 0.031 | 32,768 | 0.069 | x2.21 |
| self-knowledge | 0.031 | 13,794 | 0.030 | x0.97 |

Visit share x mean visit, normalised, matches the measured character shares to within ~0.03. So the rule values a
character (|alpha| L / D) but allocates visits: lanes of long files get more characters than their worth asks for,
lanes of short files fewer, and the floors and ceilings bind in visits (arithmetic's 1/32 is ~7% of characters).
By the same arithmetic the equal turns before 2,329.0M gave characters as arithmetic 22.5%, chess 22.4%, reasoning
12.5%, chat 11.5%, stories 11.4%, self-knowledge 9.5%, code 6.6%, wikipedia 3.6%, not the eighth each that the
schedule's seed assumes (and that lanes.py's docstring says self-knowledge got). Within a lane a short file is read
whole at every draw and a long one only a window, so coverage per byte favours short files. Nothing was changed:
paying credit by characters read, or drawing files by size, changes the run's reading and is the author's call.
Changed 11-03 at the author's call: credit is paid by characters read (history.md). Estimated then over the last 500M
characters (visit share x mean visit, as above): chess 16.4% of characters against a 7.0% share, arithmetic 7.4%
against 3.1%, wikipedia 11.3% against 29.6%, code 11.3% against 16.2%. If each subject learned only from its own
lane, reading by characters would have taken the 8-subject average from -0.0008 to -0.0012 a 100M characters - a
bound, since what one subject teaches another is left out. Files are still drawn uniformly within a lane.

## The weights themselves (2026-10-31 .. 11-03)

**E48 issue #33: admission sees the whole forward (2026-10-31/11-01, partial).** On a copy of the weights at step
1,773,530: a forward admits the experts its first pass asks for, tallied over all of its characters, so an
earlier character's prediction can use experts a later character asked for. The same spread blocks scored the
published way and causally (each position admitted from the characters up to it), causal minus published in
nats/char: arithmetic +0.0049 +/-0.0005, chat +0.0113 +/-0.0020, chat_hermes +0.0111 +/-0.0013, chess +0.0037
+/-0.0006, code +0.0112 +/-0.0022 (23 blocks); 143 of 240 blocks, reasoning, stories, wikipedia and 7 code blocks
not yet measured. Published twice: +/-0.0002. Nearly all of it is in each block's first 2,048 characters. A
cheaper approximation overshoots by +0.0026 to +0.0103. Per block: published 0.33 s, the approximation 1.14 s,
exactly causal ~11 s. Not a correction yet: the remaining blocks and the fix come first (corrections.md).

**E49 what the weights hold (2026-11-03, step 1,983,054, 400 experts).** Read-only, on the CPU, after the
"Weight-Space Programming" post (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026-11-02): weight-matrix products,
decoding through the tied embedding, spectra; then forward probes on held-out text.
- Embeddings: 153 of 265 byte vectors lie within cosine 0.9 of one direction - every non-ASCII and control
  byte, and `~`; the 13 bytes UTF-8 never uses are identical, the CJK lead bytes E4-E9 0.997, the Cyrillic D0/D1
  0.995, the continuation bytes 0.955 pairwise; norm against log frequency -0.91. The same at 2,329M (154). The
  output side pushes every rarely-correct byte the same way, and with tied embeddings that is their input too.
  Non-ASCII bytes cost 2.0-2.4 nats against 0.84 for ASCII in the same windows; they are at most 0.63% of a
  held-out subject (wikipedia). Digits are ordered (7's nearest: 8, 6, 9, 5); newline sits with `<user>`,
  `</user>`, `<|endoftext|>`.
- Router: 256 of 400 rows have a twin above cosine 0.9 even with the mean row removed (grown experts' nearest
  median 0.95, originals' 0.37); grown rows' own part is half the originals' (0.45 against 0.93); the
  differences take ~33 directions for 90% of their energy. The experts are distinct: cosine of w2 @ w1, which
  ignores how an expert orders its units, has median nearest 0.12 and nothing above 0.5 but e18's family.
- Gates: 72 below 0.01 (1.3% of use; momentum ~100x below the loud experts'), 348 below 0.1; the 20 above 1
  carry 44.7% of use. Grown gates' median 0.040 at 1.5B, 0.034 at 4.06B; originals' 1.02 -> 1.31. The gate is
  one tensor, and its decoupled decay shrinks every expert's gate at every step.
- Specialists: chess sends 58% of its requests to e6, arithmetic 25% to e18 and 13% to e184; their w2 stable
  ranks are 3.3, 5.9 and 9.0 against a median of 43. Most other subjects share originals (e2, e58, e9, e35, e33).
- Attention: positive copying scores (eigenvalues of the direct path through the tied embedding) for prelude.0
  heads 1, 4, 5, 7 and recur head 3, negative for most others; recur head 5's keys compose with prelude.0 heads
  4-7 at ~3x random. A repeated random string of letters: 3.96 nats, then 1.96 on the repeat.
- Since 2,329M: the median expert moved 4.3%, trunk matrices 5-10%, embeddings 22%; stable ranks unchanged; 14
  dead hidden units of 819,200; spectral tail exponents 3-4.
- Read out after each row on 512-character snippets, the prediction stops improving by row ~8 (0.0065 better
  than the halted output); on full 4,096-character windows it does not (E50).

**E50 edits to the weights and the read-out (2026-11-03, step 1,983,939, 401 experts).** Every change made in
memory on a copy; chosen on a dev set of training files (8 subjects x 15 blocks), the finalists measured once
on held-out, where the unedited model reproduces the run's own 0.5658. Cross-entropy change on dev:
- rows capped at 10 / 12 / 16 / 20: -0.0002 / -0.0001 / -0.0001 / 0.0000 (8: +0.0033 on a 2-block check);
  allowed 28 / 32: +0.0001 / 0.0000; halt_thresh 0.8 / 0.95 / 0.99: 0.0000 / +0.0002 / +0.0002
- logit temperature 0.95 / 1.05: +0.0002 / +0.0015; mixing the rows' probabilities instead of their logits:
  +0.0002
- gates below 0.01 zeroed: +0.0001 (below 0.03: +0.0002); grown experts' gates x0.8 / 1.25 / 1.5 / 2: +0.0013 /
  +0.0022 / +0.0077 / +0.0314; all gates x0.9 / 1.1: +0.0029 / +0.0017; a floor of 0.05 / 0.1: +0.0029 / +0.0133
- grown router rows' own part x1.5 / 0.7: +0.0204 / +0.0718
Held-out: the 12-row ceiling -0.0001 (evaluation ~18% faster on dev), gates below 0.01 zeroed 0.0000. The
held-out figure is halting-weighted cross-entropy 0.5588 plus the 0.01 KL term 0.0069; the returned logits
(the halting-weighted mixture) score 0.5572. Verdict: no edit improves held-out; the weights sit at a sharp
optimum for edits like these, and the near-twin router rows' small differences carry real information.

**E51 removing the quietest experts is not zeroing them (2026-11-03, step 1,986,517, 404 experts).** On a copy,
`PagedPool.drop_quietest(70)` - what `train.py read --drop-quietest 70` runs - removed the 70 experts with |gate|
0.0040-0.0101 that were off the card and past their newborn window (77 had gates below 0.01; the 7 newborns were
spared; 2 of the 70 were above 0.01 because 2 below were on the card). Held-out 0.5658 -> 0.5674, cross-entropy
0.5572 -> 0.5587 (+0.0014; chat_hermes +0.0039, wikipedia +0.0036, chess 0.0000); dev cross-entropy +0.0010.
Zeroing the same experts' gates had cost 0.0000 (E50): a zero-gate expert is still chosen and adds nothing, while a
removed one sends its characters to the next expert in line, which adds something. The quiet experts work as the
router's "add nothing" choice - the pool's own prune() rule warns that a low gate can mark a busy expert. Verdict:
not dropped; +0.0014 is about 175M characters of the current progress.

**E52 newborns since births resumed are never chosen (2026-11-04, step 2,045,378, 410 experts).** Births stopped
at the disk ceiling (397 experts) from 10-19 (2,452M) until one moment per expert raised it to ~529 on 11-03;
growth then added one expert every 3,906 steps (8M characters). From `runs/expert_history.jsonl` (every check's
gates, use, admissions and births):
- All 22 experts born since (from step 1,980,379) were admitted to no forward and routed no characters; their
  gates sit at birth_gate, 0.0010. The 9 past their 48,828-step trial were pruned unused.
- Earlier newborns were chosen from the start. In their first 50,000 steps, median 1,822 admissions and 20.3M
  characters routed (born before 2,222.9M, AdamW, one rate), and 1,364 and 16.0M (born under the newborn rate,
  2,222.9-3,194.4M).
- Median gate by age. Born 1,500-2,222.9M (AdamW, one rate; 19 experts): 0.0011 at 5k steps, 0.0105 at 25k,
  0.0275 at 100k, 0.041 at 400k. Born under the newborn rate (100): 0.0133 at 5k, falling to 0.0073 at 50k and
  0.0068 at 100k, 0.0092 at 400k - a fast start that reversed and settled 4-5x lower.
- The newborns' router rows sit on the pool's average direction (cosine to the established experts' mean row 0.963,
  against 0.699 for the established) and are shorter (norm 0.50 against 0.69): averaged from 16 random parents,
  they are asked for nowhere in particular.
- A row that is never admitted gets only the balance term's gradient; the language-model gradient reaches only
  the admitted rows (`PooledMLP._route`). In the router's saved momentum (bf16, unpacked) the newborn rows have norm
  ~2e-8 against an established median of 6.7e-8, while the rows in use put the top singular values at ~2e-3: five
  orders of magnitude. After Muon's orthogonalisation (5 Newton-Schulz steps) only ~104 of the 410 directions get
  a full-size update (singular value above 0.5; median 0.0014). A newborn row gets ~8e-4 of a unit row update, an
  established one a median 3.4e-3. At x0.0087 that moves a newborn's row ~1e-8 a step, ~0.01 of a logit in 50,000
  steps. Under AdamW each element is scaled by its own history, so a row with a small but steady gradient moves
  at the full rate whatever the gradient's size; that is what the balance term was built on
  (`PagedPool.note_balance`).
- Not caused by the 11-03 changes, which touch no routing, admission, balance or growth code; they only let births
  resume. Two things changed in the 1.6B characters without births: Muon took over the router (10-27, 3,194.4M),
  and the controller eased the rate from ~x0.03 to x0.0087. The momentum above points at Muon on the router as the
  main cause, with the averaged newborn row adding to it. Raising `pool.balance` would not get past it: the idle
  rows' balance gradient sits about five orders of magnitude below the rows in use, so the weight would have to
  rise ~1,000-100,000x, which flattens usage for every expert (E21, E35). Untested on a copy. A copy with forced
  births would say, with arms for the router under AdamW and a newborn row taken from its largest parent.
Verdict: growth adds nothing while this holds; each newborn lives through its trial unused and is pruned. Harmless
to the loss, since an expert that is never admitted is never in a forward.

**E53 the routing bias's floor rule on a tiny model (2026-11-04, CPU).** A paged model of 32 experts (resident 4,
top_k 2, d_model 32) was trained 400 steps as chat learning trains (LiveLearner, Muon, lr 3e-4). One expert was
then added by recombination, and training went on from that state twice on the same text: routing bias off,
and the floor rule on (floor 0.1 of an even share). The newborn's router row started at cosine 0.695 to the
pool's mean row.
- Trial of 200 steps, offset step 0.05 (trial x step ~10, as 48,828 x 2e-4 is): off, never admitted in 400 steps
  (gate 0.0010; row moved 0.008). Floor: admitted 12 times in its trial, 1,459 characters routed, gate 0.0010 ->
  0.016, row moved 0.035.
- Trial of 1,500 steps, step 0.01: off, admitted 5 times, 432 characters, gate peaked 0.0053 and fell to 0.0030.
  Floor: admitted 70 times, 8,226 characters (19x), gate 0.0010 -> 0.088, row moved 0.07; the offset peaked at
  1.13 and was near 0 at times while admissions went on, so its own row was earning part of its share.
- In both runs, once the trial ended (and so the offset), no fresh window admitted it on its own (0 of 20).
Verdict: the rule does what it is for. A newborn whose row no text asks for gets onto the card, is trained, and
its gate climbs while it is chosen. Whether a newborn then stands on its own row is not decided by the rule; in
this tiny model the one tried did not. In the run, the first lifted newborns finish their trial ~100M
characters (~18 hours) after the restart: check their admissions after it (as E52 did).

**E54 the floor rule in the run, its first hour (2026-11-04).** The run restarted at step 2,075,068 with
route_bias 2e-4 and route_floor 0.1. Twelve experts were on trial, born 2,031,120 to 2,074,086, one every 3,906
steps. After 1,110 steps each of the twelve carried an offset of 0.222 (1,110 x 2e-4), every other expert 0, as
the rule says, and none had been admitted yet. How large an offset does a newborn need? Admission is decided by a
forward's first pass, whose router input depends on the trunk alone, so it was measured on that checkpoint with
no expert loaded: 6 windows of 2,048 characters per subject, the offsets put on the twelve by hand.
- Router scores lie close together. Per character they spread 0.46-0.65 (standard deviation across the 410
  experts), and a newborn sits a median 0.68-0.82 below the character's 8th choice (chess 0.94, arithmetic
  1.28; the nearest characters 0.07-0.27).
- At 0.222, at most 0.7% of characters ask for a newborn, and no window admits one. At 0.5, 33-42% of the
  characters in chat, code, reasoning, self-knowledge and wikipedia ask, and every window admits 4-8 newborns;
  stories admit in 83% of windows. Arithmetic and chess need 0.75.
- The twelve rise together and their rows are nearly the same (all near the pool's mean row). Above 1.0 they
  would take 10-12 of the card's 32 slots in every window, and at 2.0 nearly every character's choices. What
  stops that is the rule's own feedback: an expert whose 50-step share of choices reaches the floor starts
  falling, a few to 50 steps (at most 0.01 of offset) after it crosses.
- The first admissions came before the median window's edge. By step 2,076,737 (offset 0.333, 1,669 steps
  after the restart), two of the twelve had been loaded onto the card, 3 times and once, and had routed 1,012
  and 35 characters. They are the first newborns admitted since births resumed on 11-03. The tails of the
  training text admit before the median window does. Held-out stayed at 0.5641-0.5642 (0.5644 at the last
  check before the restart).
- By step 2,077,305 (2,237 steps after the restart), 11 of the twelve had been admitted: 1-9 loads each,
  9,504-300,674 characters, 1.32M in all. The feedback had held every admitted one below the 0.447 the rule
  alone gives (0.326-0.437); the one never admitted was still at 0.447. Their 50-step shares were 0.01-2.8x
  an even share, gates 0.0010-0.0011, held-out 0.56425. The twelve did not flood the card: each found its own
  level. Gates move slowly at the controller's rate (x0.0086 of 3e-4, ~2.6e-6 a step).
- The first trial to end under the rule (born 2,031,120, trial over at step 2,079,948) had been loaded onto the
  card 21 times and routed 908,296 characters while lifted, its gate 0.0010 -> 0.0012. In the 2,968 steps after,
  its offset 0, it was not admitted once. By step 2,082,916 the others had been loaded 17-80 times and routed
  0.5-3.0M characters each, gates 0.0009-0.0015. Their held offsets stayed within about 0.02 of where they were
  two hours earlier (0.31-0.45): an offset held by the floor measures how far the newborn's own row falls short
  of the card's edge, and in those two hours no row closed its gap. Held-out averaged 0.56444 over the 11
  checks after the newborns entered, against 0.56454 over the 8 checks before the restart: no cost seen.
Expected: the run makes ~2,800 steps an hour, so the offsets climb ~0.56 an hour and reach 0.5 about an hour
after the restart. From then most windows admit newborns, and each offset should hover near its own edge,
where its newborn takes a tenth of an even share. To check: held-out over the next hours (the twelve enter together);
each newborn's offset and share; and admissions after the trial ends. The oldest, born 2,031,120, leaves its
trial at step 2,079,948 after at most ~2,400 steps on the card. The first newborn lifted for a whole trial is
one born after the restart. Past its trial, prune keeps an expert until it has gone a whole trial (48,828
steps) without an admission, so a newborn that is never chosen on its own row fades out ~17 hours after its
last lifted admission.
Verdict at 3.5 hours (step 2,084,590): the rule fixed admission, and nothing more is shown yet. All 14 newborns
lifted since the restart were admitted (0.5-3.5M characters each). Held-out since the restart sits 0.0002 below
the line of the 12 hours before. But that gap was there at the first checks, before any newborn was admitted, and
the 8 restarts of 10-28 to 11-03 moved held-out by -0.0013 to +0.0012 against their own lines, so a comparison
across a restart cannot see an effect this size. Both newborns whose trials had ended (4,390 and 557 steps
earlier) routed nothing after it. The ten still held on trial lowered their offsets by 0.009 on average in 2.6
hours (0.370 to 0.361): about 100 hours' worth to close the gap, where a trial leaves at most 17. A likely reason
is in `PooledMLP._route`. Mixed before the gates (gate_after false), a newborn at gate 0.001 takes weight from the
strong experts beside it, so the language-model gradient moves its row away from the characters it is lifted to.
Its gate grows at the controller's rate (0.0010 to 0.0012-0.0015 in 3 hours). Untested lever:
`growth.birth_gate_parents` (a newborn's gate as a share of its parents'), which gives its output a voice while
the floor bounds its share. A same-checkpoint A/B with forced births is the way to decide.

**E55 the balance term under Muon, in the run (2026-11-05, step 2,115,355, 419 experts, CPU).** `pool.balance`
was 2e-5 by 11-05 (1e-5 from 10-23). The term reaches only the router's rows and is computed from a forward's
first pass, whose input depends on the trunk alone, so its gradient was computed exactly on a copy of that
checkpoint with no expert loaded: 6 windows of 2,048 characters per subject, weighted by the subjects' shares.
The router's saved momentum (Muon's one moment) holds the rest.
- The term is 2e-5 x 0.134 (E x sum(u x P) - 1, 0 when use is even; 0.03-0.29 across windows), ~3e-6 nats on
  the training loss. Its gradient is 1.0e-5 (Frobenius), 0.12% of the router's momentum (8.5e-3).
- On the 50 rows that no forward had admitted in the last 400 texts, the momentum is the term's gradient
  (cosine 0.86) at 0.31 of its size. That is the clip: the run's gradient norm is ~3.3 against a clip of 1.0. A
  weight of 1e-5 would give 0.15, so the run uses 2e-5. On rows in use the term is a median ~4% of the row's momentum.
- Muon's step moves a row in use a median 0.031 of a full row, the same without the term; the term's part is
  0.002. Rows not admitted move 0.001-0.002. At this step size (1.13e-5 per full row) the term raises an idle or
  newborn expert's mean score ~0.0001 an hour, ~0.002 over a trial. That compares with a score spread of 0.53
  across experts and the ~0.36 a newborn needed to reach the card (E54). The floor rule moves an offset 0.56 an
  hour.
- Carried on in a constant direction (an upper bound), the term's part of the step would change 0.44 of a
  window's 32 card slots after a trial's worth of steps (17.4 h), 2.3 after a week and 6.4 after 30 days. The
  whole step, under the same bound, changes 12.9 in 17.4 h.
- Use is spread by the texts' votes instead. 307 of 419 experts hold a share of recent admissions above 1e-4.
  Every one of the 409 experts present at both steps 2,076,178 and 2,115,355 was admitted between them. Router rows
  moved a median 1.22% of their norm in those 14 hours, 0.97% of it weight decay.
Verdict: at 1e-5 to 2e-5 under Muon the term barely touches the network, and doubling it doubled a negligible
effect. Under AdamW each element's step was scaled by its own history, so a row with only this gradient moved
at the full rate whatever the weight; E21's 0.01 took the experts in use from 59 to 114 in 15 minutes. Muon's
five Newton-Schulz steps cannot lift directions this far below the strongest (E52). Turning it off should change
nothing measurable. To matter it would need ~1,000x or more, which flattens every expert's use (E21, E35). The
weight is the author's. The comments on it in config.yaml and `PagedPool.note_balance` still describe AdamW.

**E56 how the greedy loops start (2026-11-06, step 2,162,417, 424 experts, CPU).** The raw greedy samples the run
logged at this step were replayed on a copy of its weights, one forward per character as `sample_now` writes them
(the CPU's argmax was the logged character at 1,259 of 1,260 positions). Every choice was recorded with its
distribution and its copy candidate: the character that continued the longest earlier occurrence of the text's
current suffix.
- Arithmetic (after `5205`, every line `add 3 + 1 = 4`): the digits are exact ties. The first operand's '3' won
  0.110 to 0.109, the second's '1' 0.103 to 0.102, and every later line repeats them at 0.109/0.109 and
  0.102/0.101: the repeat wins each tie by under 0.001 and its probability never rises. Only choices that were
  already the argmax drift toward the repeat ('a' starting a line 0.477 -> 0.821; the space ending a one-digit
  operand 0.107 -> 0.220). The generator itself gives every first digit 1-9 0.110 (2M lines of
  `corpora/arithmetic.py`), and its own character-by-character argmax is `add 6 + 2 = 8`: one-digit addition, the
  shape the model writes. Lines are drawn independently, so the next line has the same argmax: a perfect model of
  the lane loops under greedy decoding.
- Code (`Merge a sorted array of sorted array ...`, period 16), wikipedia (period 74) and chat_hermes (a line
  repeated, period 39) start at open choices the copy wins narrowly (code: 'array' 0.258 against 0.150, 'of'
  0.279 against 0.221; wikipedia '[[R' 0.116 against 0.067) and then feed themselves. In the code loop the mean
  probability of the chosen characters per repetition is 0.831, 0.868, 0.881, 0.905, 0.923, 0.956, 0.988, the
  least likely choice in a repetition rises from 0.258 to 0.890, and entropy falls from 0.64 to 0.08 nats.
- Xu et al. 2022's self-reinforcement measurement on held-out text (a real line or sentence after 200 characters
  of its own preceding text, repeated 6 times; 37 spans): the mean probability of its characters per repetition
  is 0.776, 0.851, 0.881, 0.894, 0.903, 0.911, and the share that are the argmax 0.824 -> 0.936. Per subject,
  first to sixth: wikipedia 0.664 -> 0.990 (1 span), chat_hermes 0.712 -> 0.982, chat 0.823 -> 0.957, code
  0.839 -> 0.966, reasoning 0.790 -> 0.932, stories 0.765 -> 0.877. None in arithmetic (0.788 -> 0.769, 4
  spans) or chess (0.824 -> 0.814, 2 spans): there a repeated line gains nothing.
- Real held-out text, teacher-forced (3 windows of 1,536 characters per subject; a copy candidate needs an
  earlier match of 4+ characters). At open choices (entropy above 1 nat) the argmax is the copy candidate more
  often than the real text copies, in every subject: arithmetic 4.1% against 1.8%, chat 10.0/6.0, chat_hermes
  6.5/4.9, chess 10.4/7.5, code 14.6/10.3, reasoning 20.2/13.4, stories 10.4/9.4, wikipedia 16.3/11.4. Near-tie
  copies (the argmax is the copy candidate, less than 1 logit ahead) are 1.1-3.4% of characters, 2.2% equally
  weighted, and right 40-55% of the time (arithmetic 15%). Lowering every one of them to 0.5 logit below the
  runner-up would change the loss on the real character by -0.0004 nats per character (arithmetic) to +0.0071
  (reasoning), +0.0033 equally weighted.
Verdict: the two mechanisms the literature names. Ties at genuinely open choices are broken toward the copy
(arithmetic), which a perfect predictor does as well, so predicting better cannot remove it; and a copy, once
made, raises the odds of the next (prose and code), Xu et al.'s self-reinforcement at about their size. Both
are what training-time objectives against repetition target (adaptation.md section 7). A margin against the
copy is not free on the reading distribution outside arithmetic: at a near tie the copy is right about half
the time, so demoting it there bluntly costs ~6x the run's ±0.0005 noise in held-out.

**E57 loop practice against greedy loops (2026-11-06, step 2,163,711, 424 experts).** The run was stopped for
this. `tools/loop_meter.py` (new) measures greedy writing against real text on a weights directory, read only:
the 9 sample prompts as the raw reading writes them, 48 real held-out openings (6 per subject, 512 characters
of real text, 256 written) against their real continuations, the self-reinforcement curve (12 real spans per
subject repeated 6 times after 200 characters of their own text), and copying at open choices. A loop is a
block repeated 3+ times at a text's end over 24+ characters, with a letter or digit, not one character.
- Baseline, the run's weights: samples 31.2% repeated 8-grams, 2 of 9 end in a loop (arithmetic, code;
  chat_hermes 64% and wikipedia 51% repeat without a clean loop). Openings: 10 of 48 written continuations end
  in a loop, none of the real ones; repeated 8-grams 39.5% written against 6.7% real. A repeated span's mean
  probability by repetition 0.748, 0.705, 0.771, 0.801, 0.817, 0.829. At open choices the argmax copies 12.8%,
  the real text 9.7%; where the real text continues an earlier passage of 8+ characters the argmax does too
  96.2% of the time (repetition the writer intended).
- Loop practice (`minagi/practice.py`, `training.loop_practice_*`): after a reading step, one more optimizer step
  on a real span of the window just read (8-200 characters, from a word start to a space or newline, never one
  its own text repeats) written out 6 times after up to 512 characters of its real preceding text, with the
  one-sided term mean over n of relu(log P_n - sg(log P_(n-1)) - log 0.5) / L - each copy at most half as likely
  as the one before, spelling left alone because a span's log-probability pushes hardest where the choice was
  open. Speed probe, not controlled: a copy read 1M characters at the run's rate with a practice step after
  every reading step (488 of them; 1.6 s a step against ~1.07). The term stayed ~0.145; its gradient norm is
  ~43, so every practice step is clipped to 1 and weighs as much as a reading step. Samples 31.2% -> 15.0%
  repeated 8-grams (code 77% -> 11%, its loop gone; chat_hermes 64% -> 2%; reasoning 14% -> 2%; wikipedia
  51% -> 32%; arithmetic 75% -> 86%, still a loop). Openings: loops 10 -> 7 of 48, repeated 8-grams
  39.5% -> 30.8%. The repeated span's probability by repetition 0.742, 0.673, 0.711, 0.732, 0.748, 0.762
  (baseline 0.829 at the sixth). Intended repetition kept 96.2% -> 95.1%. Arithmetic is untouched as expected:
  there a repeated line already loses probability (0.650 -> 0.539 by the sixth), so the term never fires; its
  loops are ties (mechanism 1), which is what `training.tie_preference` is for.
- A/B (`tools/ab_pool.py --meter`, every arm a copy of the stopped run's weights reading the same 3M characters
  at the run's rate; held-out of the weights as they were 0.5587 over 960 blocks). Control, as the run reads:
  held-out 0.5587; the feeding curve unchanged (0.829 at the sixth repetition); samples 32.5% repeated 8-grams,
  2 loops; openings 8 of 48 loops, 36.2% repeated 8-grams; intended repetition kept 96.4%. Reading alone does
  not move repetition over 3M characters, so what the probe changed was the practice.
- both2 (practice after every 2nd reading step, lambda 0.5, every copy scored; tie preference weight 1, eps and
  margin 0.1 logit, 5.6 positions a step): held-out 0.5618, +0.0031 on control (wikipedia +0.0065, code
  +0.0044, reasoning +0.0043, chat_hermes +0.0040, chat +0.0036, stories +0.0019, arithmetic and chess +0.0002).
  Samples 32.5% -> 13.4% repeated 8-grams (code 77% -> 8%, chat_hermes 64% -> 6%, wikipedia 51% -> 19%,
  reasoning 14% -> 2%; arithmetic still a loop). Openings: loops 8 -> 8 of 48 (arithmetic 4 -> 5), repeated
  8-grams 36.2% -> 31.8%. Sixth-repetition probability 0.829 -> 0.745 (prose subjects 0.95 -> 0.82-0.86).
  Intended repetition kept 96.4% -> 95.4%, lower in every subject (wikipedia 95.2% -> 92.0%). The cost sits
  where the term fires: scoring the first copy against the original pushes against copying a passage at all,
  which real text does often. Next arm: the first copy left free (`training.loop_practice_free` 1).
- practice2 (practice after every 2nd step, every copy scored, no tie term): held-out 0.5617, +0.0030 - the
  whole cost of both2 is the practice; the tie term at weight 1 cost nothing measurable and did nothing
  measurable in 3M characters (samples 12.5%, openings 7 of 48 loops, the same feeding curve and intended
  repetition 95.4% as both2). With arithmetic 3% of reading, a few hundred of its ties met the term.
- free2 (practice after every 2nd step with the first copy left free, `loop_practice_free` 1): held-out 0.5617,
  the same +0.003; the same feeding curve (0.742 at the sixth repetition) and intended repetition (95.3%);
  openings 12 of 48 loops. Leaving the first copy free changes nothing, so the cost is not what the term asks
  but how often practice takes a step: at every 2nd step a third of all updates are practice, and under Muon each
  moves the weights a full step away from where reading had settled. The option was removed again.
- The tie preference cannot break arithmetic's ties from real text. Arithmetic-only probes, 330k characters (160
  steps) at the run's rate, weight 5, margin 0.3 logit: counting the continuation of ANY earlier occurrence as a
  copy fired 88 times a step but in a long arithmetic window every digit has followed some earlier `add `, so it
  only shuffled the top of every tie (openings 6 -> 4 of 6 looping, the sample still a loop); counting only the
  LATEST occurrence fired 26 times a step and changed nothing (6 of 6). In real text the line before is random;
  in writing it is the model's own mode line, so the term never practises the situation the loop forms in.
  Removed. Arithmetic's free-running loop is mechanism 1 and stays: the lane's own argmax line repeated.
- practice8 (practice after every 8th step, 184 practice steps): held-out 0.5590, +0.0003 on control - inside
  the run's noise. Samples 32.5% -> 18.1% repeated 8-grams (code 77% -> 8%, its loop gone; chat_hermes 64% ->
  8%; wikipedia 51% -> 45% and reasoning 14% not yet); openings 7 of 48 loops, 36.3% repeated 8-grams; sixth
  repetition 0.829 -> 0.814; intended repetition kept 96.2% (control 96.4%). The cost falls much faster than the
  rate: a quarter of the practice for a tenth of the held-out.
Verdict: loop practice is what works here, in the weights, with greedy decoding left plain. Set to every 8th
step (`training.loop_practice_every` 8, from the run's next start, chat learning included): no measurable cost,
the code and word-problem loops gone at once, and the rest gaining as the steps add up; every 2nd step does
more sooner for +0.003. The tie preference and freeing the first copy were dropped. Arithmetic's free-running
loop stays, so the `[adapted]` reading stays until raw greedy is nearly free of repetition. Untested next
steps for arithmetic: a generator whose next line never repeats a recent line's start (Pipis et al.'s margin
in the data), or loop-onset preference rounds on the model's own writing.

**E58 self-play against loop practice (2026-11-06, step 2,170,135).** Proposed by the author: instead of synthetic
repetitions, let the model write from a starting point and learn away from what it wrote. Pushing its own text
down alone cannot work: over text sampled from the model itself the push is zero in expectation (the sum of
p times the gradient of log p is the gradient of 1), and over greedy text it flattens the model around
everything it writes, right or wrong. With the real continuation of the same prefix as the positive it is
self-play in the manner of SPIN (Chen et al. 2024). Built as `training.self_play_every` (`minagi/practice.py`
`self_play_term`, `FileReader.self_play`): after every Nth reading step the model writes 64 characters on from
a word start in the window just read (greedy by default, `self_play_sample` draws them), then one training
forward of two rows - the prefix with the real continuation, the prefix with its own - and one step on
-log sigmoid((lp_real - lp_own) / 64). Where the two agree the pushes cancel. With the current model as the
reference and drawn continuations the update is, in expectation, plain likelihood on the real text; only greedy
continuations add a push of their own, against the model's mode in the states it writes itself.
A/B from the same weights, 3M characters each, with held-out and the loop meter: control (the run as it is,
loop practice every 8th step), selfplay8 (self-play every 8th step instead), none (neither).
- Held-out of the weights as they were 0.5592 (960 blocks). control (loop practice every 8th step, as the run):
  0.5593; samples 16.6% repeated 8-grams (the morning's baseline 31.2%: the run's own loop practice since its
  restart shows), openings 7 of 48 loops, sixth-repetition probability 0.770, intended repetition 96.1%.
  selfplay8 (self-play every 8th step, no loop practice): 0.5590, -0.0003 on control and lower in seven of
  eight subjects (wikipedia -0.0010; within the noise, but one-sided); samples 20.7%, openings 8 of 48, sixth
  repetition 0.796, intended 96.1%; reading 34 min against 28 (+21%). Its own continuation agreed with the real
  one at 14 of 64 characters and stayed 0.35 nats a character likelier than the real one: the term never
  settled.
- none (neither): held-out 0.5589. Paired on the same blocks against control, selfplay8 and none are both
  -0.0004 +/-0.0001: loop practice every 8th step costs ~0.0004 in held-out (E57's +0.0003 was the same cost,
  then within the noise), and self-play costs nothing over doing nothing. On loops self-play is close to doing
  nothing: samples 20.7% against 23.2% (none) and 16.6% (control), sixth-repetition probability 0.796 against
  0.799 and 0.770; its one clear effect is the code sample (19% against none's 53%). Closeness of written text to
  the real continuation unchanged (8-character overlap 0.090 against 0.091), non-words 1.5% against 1.7%.
  Reading speed 1,485 characters a second against 2,009 for none and 1,840 for control: self-play costs a
  quarter of the reading, loop practice an eleventh.
Verdict: not worth pursuing instead of loop practice. It costs three times as much reading, does a fraction as
much against loops, and showed no gain in closeness to real text over 3M characters. As expected: with the
current model as reference its only new push is against the greedy mode in its own states, spread over all of
its choices rather than the ones that loop, and the real continuation's push is plain likelihood. Loop practice
stays at every 8th step; its held-out cost is now measured at ~0.0004. Self-play's code is off by default; its
pair loss is what correction feedback would use (the feedback-learning direction).

**E59 issue #33 again, every subject (2026-11-07, step 2,184,784, 423 experts, CPU).** E48's method on a
hard-link snapshot of the weights, taken while the run trained: each held-out block (4,096 characters, two
forwards of 2,048 through the cache) scored the published way and causally - position i routes among the top
32 of the vote over positions 0..i, which is what writing one character at a time computes. Runs of positions
with the same causal set are one forward with that set forced; a position whose vote holds fewer than 32
experts is a single-character forward with the vote preset, as in writing. The per-position requests follow
`minagi.pool.requests` with the floor rule's lifted probability and summed to what `admit` was given within
1.1e-4. fp32, 3 spread blocks a subject, 49 s a block.
- Causal minus published, nats/char: arithmetic +0.0044, chat +0.0157 (one block +0.0315), chat_hermes +0.0077,
  chess +0.0021, code +0.0090, reasoning +0.0119, stories +0.0089, wikipedia +0.0076 (one block -0.0019). All 24
  blocks +0.0084 +/-0.0014 (+0.012 bits/byte); E48 had +0.004 to +0.011 at step 1,773,530.
- Each block's first chunk +0.0158, its second +0.0010: the first chunk's vote starts empty, and by the second
  the 2,048 characters already read outweigh the chunk's own.
- The causal sets of a first chunk's positions span 77-110 experts between them (subject means), a second
  chunk's 34-47, so a 32-slot card holds neither in one forward.
- The issue's repro prints its four lines unchanged on the working tree. Writing is not affected: every written
  character is a forward of its own, and a prompt's last position sees only what came before it.
- enwik9: 30 pieces of 2,048 bytes spread over the slice, each read cold as the frozen row reads them: published
  1.257 bits/byte, causal 1.279, +0.0221 +/-0.0023 (+0.0153 nats/char), higher causally in all 30 (+0.003 to
  +0.042 nats/char). The frozen and dynamic rows score every piece cold. The best-effort row reads pages from
  their tops and was not measured.
- Forwards an exact causal reading needs with a per-position mask, cutting a forward wherever its positions'
  sets together outgrow the card. 32 slots: 123 a held-out first chunk (61-275), 30 a second, 97 an enwik9
  piece - a character needs all 32, so every change of set is a new forward, as slow as this method. 48 slots:
  5.0, 1.0 and 4.3. 64 slots: 2.5, 1.0 and 2.1. A vote first holds 32 experts at the 6th character on average
  (4-18). These forwards split one chunk's characters through the cache, so the work stays about the same: in
  these CPU runs an extra forward cost 0.11-0.14 s against ~6 s for a whole 2,048-character one, ~3% for a
  64-slot reading of a cold chunk. The GPU was not timed.
Verdict: the issue is real and the published scores are optimistic, by ~0.008 nats/char on held-out and ~0.022
bits/byte on cold enwik9 pieces; writing is unaffected. The fix is scoring, not training: a scoring forward
routes each character among its causal set on a card of 48-64 slots of inference copies (0.3-0.4 GB in bf16).
Not built: on 11-07 the author chose to keep one path for training and evaluation, which a per-position mask
used only in scoring would split.

**E60 a learned per-group rate, CPU testbed (2026-11-08).** A tiny RecurCoder (d_model 96, 8 -> 12 -> 16
experts born at steps 1,000 and 2,000) read 3,000 steps of 8 x 256 characters from the training subjects, with
a rate for every trunk tensor and every expert (its matrices, its router rows, its gate), Muon for matrices.
A small net predicted each group's log-rate from the group's own state (age, update cosine, update/weight
ratio, gradient norm, use, last rate) and was trained toward its own value plus 0.5 x the mean cosine between
the group's move and the next 8 gradients (each on text not yet trained on). Held-out at step 3,000: one rate
for everything 1.618 / 1.560 / 1.614 at 1e-3 / 3e-3 / 1e-2; trunk at 0.3x 1.677 / 1.641 at 3e-3 / 1e-2; learned
2.008 from 3e-3, 1.946 from 1e-3. The controller sent every group to the same rate and drove it down - to
1.5e-4 from one start, to its 1e-5 floor from the other - newborns included: the next gradients point against
the last move more often than with it, so this signal only says "lower", everywhere. Verdict: next-gradient
alignment is not a usable reward for per-group rates; it is short-horizon bias in its plainest form.
- Rounds 2-4, same testbed. (2) The controller also read weight statistics (RMS, top singular value over
  RMS·√dim, mean|w|/RMS, gradient and update over weight) and was trained by a paired perturbation: two copies
  read the same batches, one with each group's log-rate +0.5 and the other -0.5 at random, compared on held-out
  after 25-50 steps, the better copy kept. Every rate sank again (to 4e-5 and 3.7e-4); 1.593 and 1.574 at step
  3,000. (3) The copies compared only after a 50-step tail at equal rates: 1.498 on seed 0 against 1.560, but
  one shared rate replaying its trajectory scored 1.506 - the gain was a learned anneal - and seed 1 sank to
  3e-5 (1.585 against 1.586). (4) Only the differences learned, the log-mean held at 3e-3: step 3,000 against
  the constant rate, seeds 0-3: 1.561/1.560, 1.547/1.586, 1.555/1.590, 1.519/1.608 - three wins and a tie;
  averaged over steps 2,000-3,000 +0.012 +/-0.015, not yet clear of noise. Where it differentiated (seeds 1, 3)
  the trunk ran 1.3-5x the experts' rate and younger experts ran faster than the founders. Each step costs two
  forwards and backwards; equal-compute comparisons not made.
- Round 5, the follow-ups, 4 seeds each, against the constant 3e-3 (averaged over steps 2,000-3,000:
  1.531/1.557/1.552/1.567). (a) A three-number rule in place of the net - trunk at k x the experts, a newborn
  at (1 + m·e^(-age/tau)) x the founders - fitted by the paired copies: 1.579/1.558/1.595/1.524, the trunk
  driven to its 0.05x bound. (b) Each expert scored by the held-out loss of the characters routed to it:
  1.568/1.797/1.516/1.541, one seed sent an expert to 0.057 and broke. (c) The level from the run's own
  controller (Plasticity, fed every 50 steps): rule 1.641/1.625/1.530/1.581, one rate 1.568/1.570/1.559/1.566 -
  on 3,000 steps it only adds its warm-up and drift. (d) The rule paired one episode in four (1.25x compute):
  1.503/1.515/1.518/1.503, all four ahead by ~0.04. But replaying (d)'s rule trajectory with nothing learned
  scored 1.511/1.515/1.509/1.512 - the same - and the rule's final values held from step 0 scored ~2.0. What
  it found is a schedule: the trunk eased from 1x to ~0.08x of the experts over the run (log k 0 -> -2.0 to
  -2.7, steady), the experts held at 3e-3. Its newborn factor differed in sign between seeds (m from e^-1.3 to
  e^2.3): no stable newborn signal. Equal compute by more steps was not a usable baseline: the constant rate
  rises after ~2,500 steps (1.59 -> 1.97 by 6,000 in every seed).
Verdict: a learned per-group rate adds nothing a schedule does not; the one result worth carrying is that on
this testbed easing the trunk while the experts keep their rate beat one constant rate by ~0.04, which is an
anneal confined to the trunk, untested against an anneal of everything and untested at scale.
- Round 6, that test, 4 seeds, log-linear easing from 1x to a final factor over the 3,000 steps, averaged over
  steps 2,000-3,000 (constant 1.552 mean): trunk only to 0.08x 1.519; experts only to 0.08x 1.539; every rate
  to 0.08x 1.509; every rate to 0.3x 1.501 (step 3,000: 1.536 / 1.530 / 1.500 / 1.497). Easing everything
  beats easing the trunk alone in every seed. Verdict: round 5's gain was an ordinary anneal; nothing on this
  testbed favours a rate per part over one rate that eases.

**E61 the training line's steps and the held-out rise under loop practice (2026-11-09, step 2,238,291, CPU, from
runs/samples.txt and the E57/E58 A/B folders; nothing run on the card).**
- The logged train loss steps at the two changes of what is read and nowhere else: within 60M characters either
  side, 0.4990 -> 0.5270 at 2,329.0M (automatic shares begin) and 0.5261 -> 0.5585 at 4,067.6M (shares of
  characters), while held-out held its course (0.6033 -> 0.5985; 0.5655 -> 0.5658). It is the mean over what the
  steps read, and both changes moved the reading from low-loss or much re-read text (arithmetic and chess ~22% of
  the characters each under equal visits, now 3% and 9%; self-knowledge ~9%, now 3%) towards wikipedia (~4%, now
  22%; held-out ~0.87), chat and code. The line does not compare across those two points.
- The logged held-out's local exponent was 0.08-0.135 from 650M to 3.2B, and the automatic shares left it there
  (1,700-2,329M 0.102, 2,329-2,800M 0.107). Shares of characters about doubled wikipedia's reading, and its
  start-of-session reading (the same 30 blocks every session) then fell 0.8731 -> 0.8593 in 363M characters,
  against 0.8748 -> 0.8731 in the 263M before.
- From the session of 11-06 13:50 (4,430.5M: loop practice every 8th step; reading 1,970-1,990 -> 1,780-1,860
  char/s, as E57's arm) the start-of-session level rose at every start: 0.5632, 0.5640, 0.5651, 0.5661 (4,430.5M,
  4,443.7M, 4,480.6M, 4,577.9M). Wikipedia 0.8593, 0.8639, 0.8698, 0.8778 (+0.0185, ~80% of the rise), chat
  +0.003, the other six within 0.0015. Before it the level fell ~0.0007 per 100M characters; against that course
  the cost by 4,577.9M is ~+0.004. The checks' own 8 blocks of each subject moved far less (wikipedia
  0.8296 -> 0.8313), so the rise shows as steps at session starts, where the line is levelled to the 30 blocks,
  and the rate controller, which reads only the checks, has barely seen it.
- Both A/Bs show the same pattern, per subject over 120 blocks each: E57 practice8 against control, wikipedia
  +0.0007 and chat +0.0007 of a +0.0003 mean; E58 control (practice) against none, wikipedia +0.0012 of +0.0004.
  E57's training-loss gap was widening, +0.0069 +/-0.0036 per 10M characters (practice2 +0.0213 +/-0.0096): the
  +0.0003 after 3M characters was the start of a cost that grows with the steps practised, not its size. Within
  wikipedia the cost sits on blocks whose own text repeats: it correlates with a block's share of repeated
  character 8-grams at +0.44 (practice8; +0.56 practice2), the more repetitive half carries +0.0013 against
  +0.0001, and the worst blocks are lists (a disambiguation page's people, a dated list of treaties). Reading:
  the practice teaches that a recurring span grows less likely, and the weights apply that to text whose
  structure legitimately recurs.
- Not separated from it: `pool.balance` 5e-5 -> 1.5e-4 from the 11-07 02:46 session, one session after the rise
  began. The slower slope from ~3.5B predates loop practice (3,500-3,800M 0.076, 3,800-4,067M 0.021,
  4,067-4,430M 0.062) and has no configuration change at its start; the controller's scale has eased from x0.0121
  (10-29, when its floor went to x0.001) to x0.00705, and whether that costs is state.md's open question.
Verdict: the shares are not the cause; they moved the training line and helped held-out. Loop practice is, by its
timing, its subject pattern and its own two A/Bs. Proposed: `training.loop_practice_every` 0 from the next start,
with the start-of-session levels showing whether and how fast the reading undoes it. A later attempt at loops
needs a target relative to the real text (a copy held down only where the real text does not repeat) and an A/B
long enough to show a trend (tens of millions of characters), not 3M.

**E62 how the model should write: the established samplers measured (2026-11-09, step 2,239,774, 423 experts,
on the card while the run was stopped; tools/write_meter.py, runs/sampling/).** Asked for after always-greedy
writing was judged a failed bet: compare the established ways of choosing a character and keep the most coherent.
- Method: held-out openings (512 characters of real text, then 256 written beside the real continuation; an
  arithmetic opening ends at a problem's "= ", a chess one a few moves into a game) and the run's 9 sample
  prompts; every setting on the same prompts with the same random numbers. Measured: repeated 8-grams, loops,
  words found nowhere in the subject's text, arithmetic lines right, legal chess moves, and GPT-2 Large's bits a
  character on the written text beside the real continuation of the same prompt - whole, and over new material
  only (characters inside an 8-gram already seen left out, in both texts), because a repeated span is easy to
  predict and the whole-text score rewards repeating. (GPT-2 XL is not in the local cache; Large is the usual
  judge in the decoding papers.) One forward per character at batch 1 writes ~35 characters a second on the card
  however many processes share it; batching streams would change which experts a forward admits.
- Round 1, 16 settings, 4 openings a subject (seed 20261109; real prose 1.029 bits, 7.9% repeated), whole / new
  material against real: greedy -0.013 / +0.250, 24.8% repeated, loops in 5 of 32 openings; greedy + DRY
  (0.8 x 1.15^(length - 8), the token defaults in characters) +0.052 / +0.132, 12.7%, none; greedy + the
  adaptation trace (2.5, 0.88) +0.185 / +0.327, 19.0%, 7 of 9 sums right; temperature 0.7 +0.170 / +0.237;
  min-p 0.1 +0.180 / +0.217; mirostat 0.8 +0.200 / +0.320; top-p 0.9 +0.281; every rule drawing at temperature 1
  +0.33 to +0.64 (top-k 40 the worst, 8.1% non-words). Typical 0.2 as the paper defines it (the most typical
  character kept, not the likeliest) +0.341 / +0.377, 19 of 31 sums: where the model is unsure it passes over
  the likeliest character every time.
- Round 2, 8 finalists on fresh openings (seed 20261110, 6 a subject; real prose 0.999 bits, 7.5% repeated), and
  temperatures 0.4 and 0.6 on the same prompts after it. Whole / new material against real, repeated 8-grams,
  loops of 48 openings, legal chess moves, non-words:
  greedy -0.155 / +0.007, 27.4%, 6, 12.2, 2.0%; greedy + DRY 32 -0.132 / +0.084, 25.6%, 0, 12.2, 3.8%;
  greedy + DRY 16 -0.099 / -0.099, 20.7%, 0, 12.2, 2.6%; greedy + DRY 8 -0.070 / -0.043, 15.2%, 0, 12.2, 2.2%;
  temperature 0.4 -0.033 / +0.071, 15.5%, 1, 16.8, 1.4%; temperature 0.5 -0.006 / +0.038, 13.9%, 0, 16.3, 1.8%;
  temperature 0.6 +0.038 / +0.112, 14.1%, 1, 13.3, 2.3%; temperature 0.5 + DRY 16 +0.044 / +0.062, 12.9%, 0;
  typical 0.2 +0.269 / +0.250; typical 0.2 + DRY 16 +0.343 / +0.283, 4.8% non-words. Every setting answered the
  problem an arithmetic opening posed right 5 times of 6. DRY on top of sampling costs: temperature 0.5 + DRY 16
  against 0.5 alone +0.050 +/-0.022 whole. The loops at 0.4 and 0.6 are one page's closing tags (`</page>`
  again and again) in the same wikipedia opening; 0.5 passed it.
- Greedy + DRY 8 against temperature 0.5, paired on the same prompts: new material -0.081 +/-0.060 bits (DRY
  ahead, not significant); an agent's blind reading of the 31 pairs whose texts differ (letters shuffled, the
  key read after the verdicts were written) 14 to 9 for DRY with 8 even, sign test p 0.40 - DRY ahead on code
  and stories, temperature on assistant chat and wikipedia. Temperature 0.5 ahead on chess (16.3 against 12.2
  legal moves), spelling, and the sample prompts: asked what it is, it describes itself as a model that keeps
  learning from what it reads, where greedy and DRY write the same garble; its arithmetic lines are varied and
  mostly right, where DRY walks one digit through the same sum. DRY's cost is copying: it charges any repeat of
  8 characters or more, the prompt's included, so a name or a code line a reply ought to repeat comes out bent
  (a "callback" written "callbacK").
Verdict: temperature 0.5 is `decoding` from 11-09 (config.yaml), for chat and for the sample log, which writes
it with its random numbers fixed per prompt (`samples.readings`; `raw` adds plain greedy). DRY and the other
rules stay in minagi/decode.py's Sampler; the adaptation trace is no longer used for writing.
- The sample log after the switch (read 2026-11-10): its first 75 blocks at temperature 0.5 (4,587.5M-4,675.3M,
  loop practice off) against the last 75 greedy blocks before them (4,504.8M-4,586.3M, loop practice on). The
  weights differ, so the rule and the training are mixed here; the paired rounds above are what isolate the rule.
  Repeated 8-grams a prompt 10.9% against 23.4% (arithmetic 3.8% against 70.2%, wikipedia 14.5% against 33.8%,
  the train problem 26.9% against 44.3%, stories 5.4% against 11.7%, code 31.5% against 29.3%); loops 2 of 675
  texts against 11 (temperature's both in code, a list of numbers cycling; greedy's all arithmetic); legal chess
  moves 34.6 against 28.3, the fewest 23 against 6; code past its docstring 35 of 75 against 16; arithmetic
  lines right 89% against 87%. What the model does not know is the same under both: the train problem is never
  answered 80 km/h (0 of 150), the reasoning prompt never reaches its function within 300 characters, wikipedia
  invents its dates and places. Asked what it is, the answer's third word is close to a tie between "model" and
  "metric"; greedy wrote the true self-description in 24 of its 75 blocks, temperature 0.5 in 15 - a threshold
  on a slowly moving probability, since each prompt's draws reuse one sequence of random numbers, not a
  frequency. The prompt's own sum is right in all 75 against 46 before (5205 in the other 29): the weights'
  carry, which no rule could fix.

**E63 order of reading: subjects side by side, or an hour each (2026-11-10, step 2,320,858, 4,752.4M, 424
experts, on the card while the run was stopped; tools/ab_lanes.py, runs/ab_lanes/2026-11-10/).** Asked whether
three hours of three subjects learn more read interleaved, as training reads them, or one subject an hour at a
time - and if one at a time loses, whether 15 minutes interleaved after it wins the difference back. The measure
is held-out at the end, the arms having read the same characters.
- How interleaved the reading already is: a visit reads one file, up to `data.passage` (32,768) characters, and
  most files are shorter, so the subject changes after nearly every document. Mean characters a visit, from the
  file sizes: wikipedia 5.3K (1.5% of files reach 32K), code 9.7K (2.6%), self-knowledge 13.8K, stories 16.6K
  (none), chat 16.8K (none), reasoning 18.2K (20.6%), arithmetic and chess 32.8K (all). The passage binds on
  arithmetic, chess and a fifth of reasoning; a longer one would not keep the reader in one subject longer
  anywhere else. One subject for long stretches needs reading that stays in a subject across files - phases of
  their own here.
- Arms, each a copy of the weights read through train.py in sessions (phases) with the checks off, so the rate
  stays at the run's x0.00706 (2.1e-6) and the shares are even: chat, code and wikipedia - the three the run reads
  most (24%, 18%, 16% of its reading) - 7.0M characters each, an hour at the run's 1,950 characters a second.
  mixed: the three side by side, three phases of 7.0M. blocked: code, then chat, then wikipedia, 7.0M each (chat
  feeds two of the four held-out subjects, so it sits in the middle). Each then reads 1.75M (15 minutes) of the
  three side by side. mixed2: mixed again with `data.shuffle_seed` 1, other documents in the same order - how far
  the documents alone move held-out. Held-out after every phase on the same 120 blocks of every subject; the four
  the three lanes feed (chat, chat_hermes, code, wikipedia) and the four left unread are reported apart.
- The reader changed first (11-10, the author's call - a visit ending with its file was not the intended
  design): a visit is now one subject's stream, files joined by `<|endoftext|>` until the 32,768 are read (train.py
  `_Lane.open`; a file already ending in the marker gets no second one; the first file's draws are unchanged, so a
  visit that starts in a long file reads what it always read). On real data every visit then read its 16 chunks,
  over 2.2 chat files, 4.8 code and 4.5 wikipedia on average (8 at most). The first launch, on the old reader, was
  stopped 4.2M characters into its first phase; the arms above run on the new one, from 23:02.
- mixed, first 7.01M (63 min, 1,864 characters a second on these three subjects): held-out on the four subjects
  the lanes feed -0.0030 +- 0.0005 (paired over the blocks), on the four left unread +0.0002 +- 0.0001, all eight
  -0.0014 +- 0.0003. Nearly all of it is wikipedia, -0.0111 +- 0.0020 (0.8688 -> 0.8578); chat -0.0008, code
  -0.0008, chat_hermes +0.0007; of the unread, stories +0.0005 and reasoning +0.0003, arithmetic and chess flat.
  Experts chosen at least once on held-out: 403 of 424 (406 before).
- mixed, second 7.01M (1,838 characters a second): the four fed subjects -0.0005 +- 0.0001 more (-0.0035 in all),
  the unread +0.0002 more (+0.0004), all eight -0.0016 +- 0.0002 in all. chat -0.0012, code -0.0006, chat_hermes
  -0.0003; wikipedia +0.0001 - none of its first hour came back or went further. Experts chosen 405 of 424.
- Where wikipedia's first-hour drop sits: its held-out file is pages back to back (one 3 MB stream of the dump's
  XML), and the old reader never showed a page after a page - a visit ended with its file. Of the 240 held-out
  chunks, the 35 holding a page boundary fell 0.070 +- 0.006 in the first hour and rose 0.002 in the second; the
  205 inside a page fell 0.0009 +- 0.0004, then 0.0003. So the drop is the new reader, once, and the order of
  reading is judged with it in every arm; the reports split wikipedia into the two kinds of chunk.
- mixed, third 7.01M (1,776 characters a second): the four fed subjects -0.0037 +- 0.0005
  in all, -0.0013 +- 0.0002 with wikipedia counted inside pages only (block means of the in-page chunks: -0.0006,
  -0.0012, -0.0013 after each hour); the unread +0.0006 +- 0.0001 (reasoning +0.0008 this hour); all eight
  -0.0016 +- 0.0002, as after the second hour. chat rose 0.0008 this hour, chat_hermes -0.0007, code -0.0003.
  At this rate (x0.00706) the third hour of the same three subjects bought almost nothing.
- mixed, its last 1.75M (the same three side by side, 1,818 characters a second): +0.0002 on the fed subjects.
  mixed at the end, 22.81M: the fed four -0.0035 +- 0.0005, -0.0012 +- 0.0002 with wikipedia inside pages only
  (its boundary chunks -0.067); the unread four +0.0007 +- 0.0001; all eight -0.0014 +- 0.0002. blocked started
  02:56 from the same weights (code first).
- blocked, code alone for 7.01M (1,940 characters a second): code -0.0021 +- 0.0002 - more than mixed's whole
  three hours did for it (-0.0017 over the same 7.0M of code) - chat -0.0004, chat_hermes +0.0023, wikipedia
  +0.0012 inside pages and +0.0032 on its boundaries; the unread four +0.0001 (stories +0.0006). The fed four
  +0.0003 +- 0.0001, all eight +0.0002 +- 0.0001.
- Speed: code alone read at 1,939 characters a second against 1,893 for code inside mixed's first hour, which
  looked like the cost of switching subjects - it was not: mixed2 (below) read the same mix at 1,928, its steps
  as fast as the blocks'. Changing subject every 32,768 characters costs no measurable speed.
- blocked, then chat alone for 7.01M (1,925 characters a second):
  chat -0.0010, chat_hermes -0.0045, code +0.0004 of its gain back, wikipedia -0.0004 inside pages, reasoning
  +0.0011, stories -0.0002. After 14.02M: the fed four -0.0011 +- 0.0001 - as far as mixed got at the same point
  with wikipedia inside pages only (-0.0012), though blocked has not read wikipedia yet - the unread four
  +0.0004, all eight -0.0004. blocked minus mixed at 14.02M, paired: chat_hermes -0.0026 +- 0.0003, code -0.0003,
  stories -0.0003, chat +0.0007, reasoning +0.0004, wikipedia +0.0119 (its boundaries, still unread).
- blocked, then wikipedia alone for 7.01M (1,923 characters a second): wikipedia -0.0033
  inside pages (mixed's three hours: -0.0019) and -0.067 on its boundaries, as mixed; but chat_hermes +0.0048,
  code +0.0030 - code now 0.0013 worse than before its own hour of code - reasoning +0.0022, stories +0.0011.
  The last subject read is the one held; the hour of wikipedia took back what the earlier hours taught.
- Equal characters, 21.04M (7.0M of each subject in both): blocked minus mixed, paired - the fed four +0.0015 +-
  0.0002, the unread four +0.0006 +- 0.0001, all eight +0.0011 +- 0.0001; by subject chat_hermes +0.0030, code
  +0.0030, reasoning +0.0018, stories +0.0007, chat -0.0001, wikipedia +0.0001. Read side by side wins; the
  15-minute shake after blocked comes next.
- The shake, 1.75M of the three side by side after blocked (1,878 characters a second): code -0.0025, chat_hermes
  -0.0024, reasoning -0.0011 though no arm reads it, stories -0.0003; all eight -0.0008 +- 0.0001 in a quarter
  of an hour. The hours of one subject had not erased what came before - a short mixed stretch brought it back.
- At the end, 22.81M each: blocked-then-shake minus mixed, paired - the fed four -0.0000 +- 0.0001, the unread
  four +0.0002 +- 0.0001, all eight +0.0001 +- 0.0001; by subject chat -0.0005, wikipedia -0.0004, chess -0.0003,
  code +0.0003, chat_hermes +0.0005, reasoning +0.0007, stories +0.0002. A tie on final held-out.
  mixed2 (shuffle_seed 1) started 06:35 from the same weights: how far other documents alone move these numbers.
- mixed2, first 7.01M (1,928 characters a second): the fed four -0.0033, all eight -0.0016; mixed2 minus mixed
  after the same hour, paired: all eight -0.0002 +- 0.0001, but chat -0.0009 +- 0.0002 and code -0.0004 - other
  documents move a single subject further than the blocks' standard error says. Its steps past a visit's first
  took 1,100 / 1,070 / 1,080 ms (chat / code / wikipedia), against 1,086 / 1,079 / 1,087 read alone and 1,141 /
  1,101 / 1,116 in mixed's first hour, which slowed further each hour (1,147 / 1,138 / 1,171 in its third) - mixed
  read slowly for reasons of its own hours, not its order.
- mixed2, second 7.01M (1,929 characters a second; 1,087 / 1,071 / 1,088 ms a step): all eight -0.0015, the fed
  four -0.0032. mixed2 minus mixed at 14.02M: all eight +0.0001 +- 0.0001, the fed four +0.0003, chat now +0.0005
  (it was -0.0009 an hour before). Two runs of one order differ by at most 0.0002 on all eight and 0.0003 on the
  fed four so far; a single subject wanders by up to 0.0009.
- mixed2, third 7.01M (1,922 characters a second; 1,084 / 1,082 / 1,094 ms a step): all eight -0.0016, the fed
  four -0.0039, -0.0015 with wikipedia inside pages only, the unread four +0.0007. mixed2 minus mixed at 21.04M:
  all eight -0.0000 +- 0.0001, the fed four -0.0001, the unread +0.0000; chat -0.0013 and chat_hermes +0.0011,
  the two subjects the chat lane feeds moving opposite ways. blocked minus mixed2 at 21.04M: all eight +0.0011 +-
  0.0001, the fed four +0.0016, the unread +0.0006 - what it was against mixed. Over three points the two runs of
  one order differ by at most 0.0002 on all eight; blocked before its shake is 0.0011 behind either.
- mixed2, its last 1.75M (1,895 characters a second): all eight +0.0000. The quarter hour side by side that
  brought blocked back 0.0008 bought nothing after side by side - in blocked it was recovery, not new learning.
  mixed2 at the end, 22.81M: the fed four -0.0039 +- 0.0005, -0.0016 with wikipedia inside pages only, the unread
  four +0.0007, all eight -0.0016 +- 0.0003. mixed2 minus mixed at the end: all eight -0.0002 +- 0.0001, the fed
  four -0.0004, the unread +0.0001 (chat -0.0016, chat_hermes +0.0009). At the four points the two runs of one
  order differed on all eight by -0.0002, +0.0001, -0.0000, -0.0002.
- Final held-out, 22.81M each: blocked-then-shake minus mixed2, all eight +0.0003 +- 0.0001 (the fed four +0.0004,
  the unread +0.0001); minus the block-wise mean of mixed and mixed2, +0.0002 +- 0.0001 (the fed four +0.0002,
  the unread +0.0002; code +0.0003, reasoning +0.0005, wikipedia's page boundaries +0.0010).
- Result: the three side by side, the subject changing every 32,768 characters, gave the best final held-out of
  what was tried. An hour of each subject in turn ended 0.0011 +- 0.0001 behind at equal characters, against
  either side-by-side run; 15 minutes side by side after it brought it to +0.0002 of their mean, inside the 0.0002
  the two side-by-side runs differ by - level, not ahead. What the hours of one subject lost was forgetting: the
  last subject read held, the others slipped, the unread reasoning and stories too, and a quarter hour side by
  side brought it back; specializing and then shaking gained nothing over never specializing. Steps took the same
  time either way (1,070-1,100 ms side by side, 1,079-1,087 alone), so longer stretches of one subject buy no
  speed. Limits: the run's present rate (x0.00706, held), three subjects, 3.25 hours, even shares; passages
  shorter than 32,768 and cycles shorter than an hour were not tried.

**E64 what limits the recent exponent (2026-11-11, step 2,343,668, 4,799M, 424 experts, CPU, nothing trained).**
Read from the check log, the code and saved weights. alpha = -d ln(held-out) / d ln(characters).
- Start-of-session series: 2,386 -> 3,194M alpha 0.119; 3,194 -> 4,430M 0.081; 4,055 -> 4,430M 0.052; 3,194 ->
  4,752M 0.069 (seven subjects without arithmetic 0.081). With the easing's own immediate gain taken out
  (0.15-0.22 nats per unit of scale), the check series gives 0.085-0.09 over 1.25-3.2B and 0.044-0.046 over
  3.4-4.43B. Wikipedia and code keep their exponents, the rest flatten (shares of characters since 4,067.6M
  are part of that).
- The controller eases by construction: resting move -0.08 to -0.14% a check, at most +0.08% up, one doubling
  per confirmed 0.25-nat jump. The logged verdict put through the rule reproduces the rate's path since ~2.0B.
  The rate fell as D^-2.1 over 2.0-4.76B; at this pace it reaches x0.001 near 7.6B.
- The deciding number is blind to the rate: over 16 windows (1.0-4.77B) slope/scatter stayed 0.015-0.042 while
  the rate fell 10.7x.
- Travel (sum of scale x M chars): whole run 644; 0 -> 2,329M 610; 2,329 -> 4,770M 34 (5.2% for 51% of the
  characters). Left to the corpus end at the present easing 13; held at x0.0071 23; at x0.028 93.
- Weights, session start (step 2,320,858) against live (22,298 steps): nothing stuck on fp32 rounding; trunk
  matrices moved 0.1-0.4%, net/path 0.022-0.039 against 0.038 for a no-signal walk at momentum 0.95 (Nesterov);
  twelve experts across the admission range within ~10% of the walk; busiest founders' gates +0.005-0.008,
  newer gates still. Against the 2,329M copy (1.2M steps) the trunk's net/path is 0.008-0.014 against 0.006:
  consistent drift ~1% of each step (a lower bound).
- Telemetry: `admits` counts loads onto the card; `recent` is the share of training forwards an expert is
  admitted in (sums to 32). Admission share by rank: 1.0 busiest, 0.72 at the 5th percentile, 0.12 at the
  10th, 0.011 median.
- A visit's first step is a half window that the second step repeats at the same positions (a new FileReader
  per visit): 16 steps, 63,488 scored positions for 32,768 fresh characters.
- Checks: 12 min apart for 10 min of reading; 9 written samples (~2,950 chars at ~31.5 char/s, ~94 s) are
  ~13% of the wall-clock.
- Reading: the low rate is a likely, unproven part of the fall; saturation, the shares, loop practice, Muon's
  early gain and unused growth (127 experts born after step 1M carry 1.5% of the choice) share it. Short tests
  cannot show a higher rate's gain (floor moves in ~2M chars, learning shows after 130-480M).
- Proposed, not run: (A) enwik9 adaptation at x1, x4, x16 of the present rate; (B) on copies, 60M chars at
  x0.0071 against x0.028, then two 14M tails at x0.0071 with a 120-block evaluation after each; expected
  difference 0 (data-limited), ~-0.0003 (crossover), ~-0.0010 (movement-limited); optional arms momentum 0.99
  against a x2.26 rate arm, and trunk x0.3. Needs a per-phase setting in tools/ab_lanes.py.
- Ideas beyond the rate, none run (11-11): score the check's samples as capability measures; held-out at
  top-k 4/6/8/12 (forward only); screen passages by first-chunk loss; `growth.birth_gate_parents` is 0.0 so a
  newborn starts silent and cannot grow at this rate; spaced replay in place of the immediate second look
  (chunk 4,096 + delayed re-reading by loss); multi-byte prediction heads; grow the trunk rather than the
  pool; an episodic store beyond the window at use (PG19, frozen). 7.3 expert loads a step this session.
