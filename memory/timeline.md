# timeline

What was in force at each point of the run, on the run's own clock (characters read), so that a curve, a check or an
old number can be read against the code and the settings that produced it. history.md says why each change was made;
this file says where on the curve it landed. Dates are days of 2026.

How the points were found. The trainer loads its code once, at start, and reads config.yaml at start too, so a
change acts from the first session that starts after it. A commit is placed at the first session that began after
it was committed: the latest point it can have started acting (an edit the author ran before it was committed acted
earlier). The author's own config.yaml values were never committed; each is dated by the first agent record that
shows it in the working copy, so those dates are upper bounds as well. `grep '^# session started' runs/samples.txt`
lists every session with its time, step, characters and experts.

## Two runs

- **Run 1**, 2026-09-03 .. 09-16: 72 sessions, 609M characters by its counter (step 714,530), 112 experts of today's
  size at the end. Every restart re-read the same text from the beginning (the lane generator was seeded without the
  characters read; fixed on 09-14), so only about 66M of those characters were distinct text, and a power
  law fitted it with R2 0.56 against 0.99 for run 2. Archived in `runs/archive/run-1` (not tracked); left out of
  every fit, graph and table.
- **Run 2, "the run"**: started 2026-09-16 from a fresh model (64 experts, corpus 7,874M characters) and never
  started over. Everything in this folder is about it unless it says otherwise.

## Where each change landed

| chars (M) | session | date | what changed | refs |
|---|---|---|---|---|
| 0 | 1 | 09-16 | fresh model, 64 experts; the rate governed by held-out; pruning by staleness; lanes seeded with the characters read | |
| ~17 | 2 | 09-16 | the window reaches its 4,096 ceiling, and has stayed there since | log |
| 245.0 | 7 | 09-20 | the "19 September" controller | |
| 249.3 | 8 | 09-20 | arithmetic in standard notation, lane regenerated; corpus 7,879M | |
| 550.1 | 29-30 | 09-26 | auditions for idle experts, v1 then v2 | E24 |
| 605.8 | 31 | 09-27 | corpus 7,880M | |
| 611.4 | 32 | 09-27 | one selection rule; every forward admits its own experts; exploration bonus 0.35, raised by the author to 0.65 by 10-02 | E17, E18 |
| 630.9 | 35 | 09-28 | freeze-and-carry; every expert stepped with its own moments | E11 |
| 649.7 | 36 | 09-28 | the text's vote admits each forward's 32 | E19 |
| 751 -> 980 | 43-48 | 09-29 .. 10-02 | the pool shrinks 224 -> 118 as experts grown in the days before reach the prune line unchosen: the clock counted only the router's own choices, not the bonus's | history.md |
| 980.1 | 48 | 10-02 | the balance term, 0.003, replaces the bonus; every admission resets the prune clock; selection temperature 0 | E20, E21 |
| 1,262.8 | 55 | 10-05 | writing and held-out cast the experts once and run only those used | |
| 1,342.4 | 57 | 10-06 | the controller probes upward while held-out improves | E29 |
| 1,454.6 | 58 | 10-07 | exact dispatch, blocked attention with its own backward | |
| 1,465.5 | 59 | 10-07 | expert files read straight into arrays | E25 |
| 1,553.9 | 61 | 10-08 | the pool reaches its 397 ceiling (10 GB on disk) | |
| 1,564.5 | 62 | 10-08 | `pool.capacity_factor` 0, the author's (uncommitted): no dispatch cap | E28 |
| 1,593.7 | 63 | 10-08 | the self-knowledge lane rebuilt without the model's own samples | |
| 1,665.4 | 64 | 10-09 | the controller: floor x0.01, gentle settling on long plateaus | |
| 1,745.8 | 66 | 10-10 | a plateau controller, for one session | |
| 1,755.9 | 67 | 10-11 | the 19 September controller again, at most 2% a check, blind to spikes, 0.16 of its gains: the controller since | |
| 1,901.4 | 70 | 10-13 | binary detection fixed, code lane rebuilt: corpus 8,037M (issue #30) | E38 |
| 2,222.9 | 73 | 10-16 | the newborn learning rate | E31 |
| 2,222.9 -> 2,290.1 | 73-75 | 10-16 .. 10-17 | 50 of the quietest experts removed, then 50 more (`--drop-quietest`): 397 -> 330, with growth between | |
| 2,290.1 | 75 | 10-17 | checks read blocks spread over the whole set: correction 1 | corrections.md |
| 2,329.0 | 79 | 10-18 | reading shares automatic; Adam moments bf16 in RAM; RAM tier 288 (96 committed before, 192 in the working copy by 10-17); evictions written on their own thread | E26 |
| 2,386.0 | 80 | 10-18 | whole blocks only: correction 2 | corrections.md |
| 2,452.4 | 81 | 10-19 | the pool back at 397 | |
| 2,458.5 | 82 | 10-19 | chat learns at the run's own rate | |
| 2,800.9 | 86 | 10-23 | one code path for the controller wherever the model learns (training unchanged) | |
| 3,194.4 | 87 | 10-27 | Muon for every weight matrix; one learning rate for every expert, the newborn rate removed | E35 |
| 3,321.4 | 88 | 10-28 | the largest attention logit logged at every check | E40 |
| 3,492.6 | 89 | 10-29 | the controller's floor x0.001 (x0.01 before); the scale printed to three figures | |

The author's `pool.balance`, lowered by hand after the balance term came in at 0.003, as the working copy showed
it: 0.001 later on 10-02, 2.5e-4 by 10-07, 6.5e-5 by 10-09, 3.5e-5 by 10-17, 1.5e-5 by 10-21, 1e-5 by 10-23
(unchanged on 10-31), 2e-5 by 11-05 (the router's momentum shows the run uses it, E55). The run logs none of
these values, so which session ran with which is known only to these bounds.

Not dated here: rebuilds of the self-knowledge lane other than 10-08 (made by hand, with no run training), and the
benchmark sweeps (on copies; they change nothing in the run).

## The controller's scale

The scale at a session's first check (table below): x1 for the first ~250M characters; x0.5 at ~545M; x0.27 at
606M; x0.11 at 980M; x0.056 from 1.45B to 1.6B; up to x0.067 at 1,746M under the upward probe; x0.052 at 1.87B;
x0.032 at 2.33B; x0.019 at 2.80B; x0.014 at 3.19B; x0.0121 at 3.49B (10-29); x0.0109 at 3.63B (10-31). Its floor
(`minagi/plasticity.py` `FLOOR`) was x0.05 until 1,665.4M, x0.01 until 3,492.6M, and x0.001 since.

## Sessions

Held-out is the session's first check, moved onto today's level (`minagi/heldout.py` `parse_log`); scale and reading
speed are from the same check. "-" is a session that stopped before its first check.

| # | date | step | chars (M) | experts | held-out | scale | reading char/s |
|---|---|---|---|---|---|---|---|
| 1 | 09-16 | -1 | 0.0 | 64 | 2.5857 | x1 | 3,347 |
| 2 | 09-16 | 3,515 | 7.2 | 97 | 2.0179 | x1 | 1,982 |
| 3 | 09-16 | 21,316 | 43.7 | 97 | 1.3221 | x1 | 869 |
| 4 | 09-18 | 71,763 | 147.0 | 136 | 1.0800 | x1 | 760 |
| 5 | 09-19 | 92,135 | 188.7 | 146 | - | - | - |
| 6 | 09-19 | 92,135 | 188.7 | 146 | 0.9258 | x1 | 980 |
| 7 | 09-20 | 119,626 | 245.0 | 146 | 0.8796 | x0.986 | 1,067 |
| 8 | 09-20 | 121,716 | 249.3 | 147 | 0.9007 | x0.987 | 1,094 |
| 9 | 09-20 | 125,349 | 256.7 | 147 | 0.8831 | x0.958 | 1,098 |
| 10 | 09-20 | 126,296 | 258.7 | 148 | - | - | - |
| 11 | 09-20 | 126,296 | 258.7 | 148 | 0.9173 | x0.949 | 554 |
| 12 | 09-20 | 127,028 | 259.4 | 148 | 0.8731 | x0.917 | 1,136 |
| 13 | 09-21 | 151,941 | 310.4 | 166 | 0.8270 | x0.845 | 1,055 |
| 14 | 09-21 | 156,320 | 319.4 | 170 | - | - | - |
| 15 | 09-21 | 156,320 | 319.4 | 170 | 0.8497 | x0.862 | 1,096 |
| 16 | 09-21 | 159,139 | 325.2 | 172 | 0.8274 | x0.872 | 1,048 |
| 17 | 09-22 | 184,026 | 376.1 | 176 | 0.7931 | x0.784 | 1,076 |
| 18 | 09-22 | 190,980 | 390.4 | 176 | 0.7943 | x0.753 | 1,046 |
| 19 | 09-22 | 192,971 | 394.5 | 176 | 0.7918 | x0.699 | 987 |
| 20 | 09-23 | 203,610 | 416.2 | 175 | 0.7847 | x0.683 | 1,059 |
| 21 | 09-23 | 217,368 | 444.4 | 170 | 0.7747 | x0.62 | 1,033 |
| 22 | 09-23 | 219,206 | 448.2 | 172 | 0.7745 | x0.612 | 1,060 |
| 23 | 09-23 | 223,099 | 456.2 | 174 | - | - | - |
| 24 | 09-24 | 223,506 | 457.0 | 174 | 0.7719 | x0.596 | 1,079 |
| 25 | 09-24 | 237,080 | 484.8 | 173 | 0.7607 | x0.553 | 1,046 |
| 26 | 09-24 | 238,401 | 487.5 | 172 | 0.7602 | x0.561 | 1,094 |
| 27 | 09-24 | 240,467 | 491.7 | 171 | 0.7554 | x0.549 | 1,083 |
| 28 | 09-25 | 244,265 | 499.5 | 166 | 0.7529 | x0.543 | 1,046 |
| 29 | 09-26 | 268,957 | 550.1 | 168 | - | - | - |
| 30 | 09-26 | 268,957 | 550.1 | 168 | 0.7400 | x0.456 | 986 |
| 31 | 09-27 | 296,153 | 605.8 | 171 | 0.7030 | x0.272 | 1,176 |
| 32 | 09-27 | 298,919 | 611.4 | 170 | 0.7018 | x0.253 | 516 |
| 33 | 09-27 | 299,421 | 612.5 | 171 | 0.6989 | x0.226 | 585 |
| 34 | 09-27 | 299,932 | 613.5 | 171 | 0.6988 | x0.218 | 1,792 |
| 35 | 09-28 | 308,412 | 630.9 | 180 | 0.6933 | x0.205 | 2,015 |
| 36 | 09-28 | 317,620 | 649.7 | 190 | 0.6877 | x0.197 | 2,051 |
| 37 | 09-28 | 331,517 | 678.2 | 204 | 0.6815 | x0.192 | 1,971 |
| 38 | 09-28 | 334,408 | 684.1 | 207 | 0.6852 | x0.192 | 1,904 |
| 39 | 09-28 | 335,522 | 686.4 | 208 | 0.6836 | x0.192 | 1,717 |
| 40 | 09-28 | 345,827 | 707.5 | 218 | 0.6957 | x0.185 | 1,487 |
| 41 | 09-29 | 362,280 | 741.2 | 222 | 0.6755 | x0.144 | 1,890 |
| 42 | 09-29 | 363,629 | 744.0 | 223 | 0.6739 | x0.143 | 1,933 |
| 43 | 09-29 | 367,145 | 751.2 | 224 | 0.6787 | x0.141 | 1,919 |
| 44 | 09-30 | 414,220 | 847.6 | 139 | 0.6698 | x0.12 | 1,941 |
| 45 | 10-01 | 417,232 | 853.7 | 133 | 0.6670 | x0.121 | 1,861 |
| 46 | 10-01 | 433,352 | 886.8 | 127 | 0.6648 | x0.119 | 1,901 |
| 47 | 10-01 | 443,301 | 907.1 | 131 | 0.6657 | x0.114 | 2,009 |
| 48 | 10-02 | 478,910 | 980.1 | 118 | 0.6641 | x0.111 | 2,003 |
| 49 | 10-02 | 480,774 | 983.9 | 120 | 0.6671 | x0.104 | 1,811 |
| 50 | 10-02 | 501,627 | 1,026.6 | 141 | 0.6501 | x0.073 | 1,640 |
| 51 | 10-03 | 512,072 | 1,048.0 | 152 | 0.6484 | x0.075 | 1,568 |
| 52 | 10-03 | 512,869 | 1,049.6 | 152 | 0.6493 | x0.075 | 1,490 |
| 53 | 10-03 | 516,696 | 1,057.4 | 156 | 0.6476 | x0.078 | 1,570 |
| 54 | 10-04 | 578,777 | 1,184.6 | 220 | 0.6410 | x0.075 | 1,520 |
| 55 | 10-05 | 616,974 | 1,262.8 | 259 | 0.6350 | x0.058 | 1,528 |
| 56 | 10-05 | 639,026 | 1,308.0 | 282 | 0.6322 | x0.053 | 1,447 |
| 57 | 10-06 | 655,833 | 1,342.4 | 299 | 0.6325 | x0.052 | 1,456 |
| 58 | 10-07 | 710,614 | 1,454.6 | 355 | 0.6303 | x0.056 | 1,328 |
| 59 | 10-07 | 715,933 | 1,465.5 | 360 | 0.6299 | x0.056 | 1,342 |
| 60 | 10-07 | 732,331 | 1,499.1 | 377 | 0.6295 | x0.056 | 1,469 |
| 61 | 10-08 | 759,111 | 1,553.9 | 397 | 0.6325 | x0.056 | 1,281 |
| 62 | 10-08 | 764,290 | 1,564.5 | 397 | 0.6248 | x0.056 | 1,250 |
| 63 | 10-08 | 778,551 | 1,593.7 | 397 | 0.6231 | x0.056 | 1,319 |
| 64 | 10-09 | 813,563 | 1,665.4 | 397 | 0.6219 | x0.058 | 1,235 |
| 65 | 10-10 | 824,882 | 1,688.6 | 397 | 0.6365 | x0.063 | 1,381 |
| 66 | 10-10 | 852,790 | 1,745.8 | 397 | 0.6223 | x0.067 | 1,373 |
| 67 | 10-11 | 857,743 | 1,755.9 | 397 | 0.6208 | x0.06 | 1,327 |
| 68 | 10-12 | 914,508 | 1,872.2 | 397 | 0.6132 | x0.052 | 1,347 |
| 69 | 10-12 | 922,230 | 1,888.0 | 397 | 0.6126 | x0.051 | 1,344 |
| 70 | 10-13 | 928,793 | 1,901.4 | 397 | 0.6143 | x0.051 | 1,411 |
| 71 | 10-14 | 984,273 | 2,015.0 | 397 | 0.6072 | x0.045 | 1,370 |
| 72 | 10-15 | 1,032,423 | 2,113.7 | 397 | 0.6067 | x0.041 | 1,245 |
| 73 | 10-16 | 1,085,757 | 2,222.9 | 397 | 0.6045 | x0.036 | 1,516 |
| 74 | 10-16 | 1,091,149 | 2,233.9 | 352 | 0.6045 | x0.036 | 1,552 |
| 75 | 10-17 | 1,118,555 | 2,290.1 | 330 | - | - | - |
| 76 | 10-17 | 1,119,266 | 2,291.5 | 331 | 0.6064 | x0.033 | 1,582 |
| 77 | 10-17 | 1,128,086 | 2,309.6 | 340 | 0.6034 | x0.033 | 1,472 |
| 78 | 10-17 | 1,132,995 | 2,319.6 | 345 | 0.6031 | x0.032 | 1,298 |
| 79 | 10-18 | 1,137,561 | 2,329.0 | 350 | 0.6009 | x0.032 | 1,928 |
| 80 | 10-18 | 1,165,384 | 2,386.0 | 378 | 0.5991 | x0.03 | 1,845 |
| 81 | 10-19 | 1,197,835 | 2,452.4 | 397 | 0.5961 | x0.028 | 1,989 |
| 82 | 10-19 | 1,200,798 | 2,458.5 | 397 | 0.5963 | x0.028 | 1,953 |
| 83 | 10-20 | 1,221,768 | 2,501.4 | 397 | 0.5969 | x0.027 | 1,964 |
| 84 | 10-21 | 1,258,101 | 2,575.8 | 397 | 0.5919 | x0.024 | 1,913 |
| 85 | 10-22 | 1,306,750 | 2,675.5 | 397 | 0.5927 | x0.023 | 1,945 |
| 86 | 10-23 | 1,368,001 | 2,800.9 | 397 | 0.5884 | x0.019 | 1,910 |
| 87 | 10-27 | 1,560,143 | 3,194.4 | 397 | 0.5777 | x0.014 | 1,442 |
| 88 | 10-28 | 1,622,136 | 3,321.4 | 397 | 0.5730 | x0.014 | 1,830 |
| 89 | 10-29 | 1,705,727 | 3,492.6 | 397 | 0.5696 | x0.0121 | 1,867 |
| 90 | 10-30 | 1,756,345 | 3,596.2 | 397 | 0.5693 | x0.0112 | 1,874 |
