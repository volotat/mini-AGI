# mini-AGI

mini-AGI - is a **continual learning** byte-level language model that assembles its own architecture, trains from scratch on a single 8 GB VRAM GPU, and keeps learning from everything it reads.
It stores its weights as ordinary files on disk and pages them onto the card as it needs them, so the parameter count is bounded by free disk space rather than by VRAM. It grows new capacity while training when it runs short, prunes what nothing asks for, and reads through exactly the same code path it serves on. Targeted at a PC or laptop with at least an 8 GB VRAM GPU on the board. 

**NOTE: as of now this is a small toy-level model.** Do not expect a frontier level capabilities. This is rather a small experiment to show, that continual learning from the single stream of data without catastrophic forgetting is possible. Furthermore it is possible on a modest hardware. Which means that almost everyone could train their own version of the model (or simply continue training this one) exactly as they see it fit. And the capabilities would be bounded by the actual hardware, scale and quality of the data available and the amount of time one willing to spend on training the model.

*The name is a half-joke and not a statement of the current capabilities of the model, but rather the potential and traits it has. Continual single-stream learning and natural size and processing adaptation to the available resources exactly what I myself expect from an AGI to have. It has just a small toy-level memory footprint and hence it is “mini-AGI”.*

![dashboard](assets/dashboard.png)
*Here is how min-run dashboard looks like. The model is pointed to the corpus to constantly read and learn from.*

[History](runs/samples.txt) - here is the samples from the whole training run history so far. You can inspect them yourself to see how the model improved over the course of training/reading the corpus. 

The "final" weights are **not published yet**. The run is still reading its first pass over the corpus, the weights go up once it has been through all of it, which is several weeks away at the current rate. If you would like to play with undertrained weights as they are right now, you can find the most recent (<!-- auto:weights-date -->9 Nov 2026<!-- /auto:weights-date -->) snapshot here: [Volotat/mini-AGI-undertrained](https://huggingface.co/Volotat/mini-AGI-undertrained/tree/main) 

<!-- auto:run-blocks -->
<details>
<summary><b>Graph of the whole run so far</b></summary>

![training progress](assets/training_progress.png)

*Every sample round of the run to date: 5,048.3M characters over 5,563 evaluations; held-out from before the [in-flight corrections](memory/corrections.md) is drawn moved onto today's level.*

</details>

<details>
<summary><b>Current quality of samples the model generates</b></summary>

*The round with the lowest held-out loss so far - 0.5582 nats at 5,037.0M characters. Each prompt is written the way chat writes, by the rule the header names. The whole history is in [runs/samples.txt](runs/samples.txt).*

```
==============================================================================
step 2,459,842   5037.0M of 8,037M characters (62.67%)   13 min   424 experts
context 4,096 characters of 4,096   reading 1,764 char/s   writing 23.8 char/s   still gaining -0.0516 deep into it
grad norm 3.51 against a clip of 1   clipping
train loss 0.5476   lr 2.16e-06   evidence t +1.81 over 65.7 (effect +0.0845)   rate x0.00719
attention logit 17.38 at most   prelude.0 12.56   prelude.1 16.38   recur.0 17.38
held-out loss 0.5582 +/-0.0225 nats   0.8054 bits/char   perplexity 1.75   gap +0.0106   level -0.0106
  arithmetic 0.612   chat 0.459   chat_hermes 0.749   chess 0.374   code 0.461   reasoning 0.524   stories 0.442   wikipedia 0.846
written temperature 0.5: repeats 11% of 8-grams
==============================================================================

--- stories ---
prompt: 'Once upon a time, there was a little boy named Tom. One day he '
[written]  repeated 8-grams 11%
was playing with his toy car when he saw a big dog. The dog had a long tail and sharp claws. Tom wanted to play with the dog and his car, but the dog was square and had a long tail. 

Tom didn't want to give up his claws. He said to the dog, "No, I don't want to give up my car. It's mine." The dog was sad and started t

--- code ---
prompt: 'def merge_sorted(a, b):\n    '
[written]  repeated 8-grams 35%
"""
    Merge a boolean array of sorted arrays of shape (n, n).

    Parameters
    ----------
    a : array_like
        Array of shape (n,) which is equal to the size of the original array of shape
        (n,).
    b : array_like
        Array of shape (n,) which is equal to the size of the original array of
       

--- arithmetic ---
prompt: 'add 4917 + 388 = '
[written]  repeated 8-grams 3%
5305
add 5 + 4 = 9
add 627 + 9 = 6 6
add 8000 + 0 = 8000
add 9798 + 5 = 9 94
add 233 + 77 = 300
add 1876 + 2 = 1878
add 1 + 6 = 7
add 7199 + 47 = 7 106
add 33 + 98 = 131
add 56332 + 7 = 56339
add 9483 + 2 = <think> 3+2+0=5c0 8+0+0=8c0 4+0+0=4c0 9+0+0=9c0 </think> 9485
mul 1 * 1 = 1
mul 30 * 3 = <think> 30*3=90 </think> 90
sub 596253 - 83 = 5961

--- chat ---
prompt: '<user>\nWhat are you?\n</user>\n<bot>\n'
[written]  repeated 8-grams 0%
mini-AGI: a model that keeps learning from what it reads, one character at a time. I am a pool of 397 experts on disk and a small trunk that is always loaded, and for each stretch of text the text itself decides which 32 of them and on the graphics card.
</bot>
<user>
how do you decide how many passes to take
</user>
<bot>
After every pas

--- chat_hermes ---
prompt: '<user>\nA train travels 60 km in 45 minutes. What is its speed in km/h?\n</user>\n<bot>\n'
[written]  repeated 8-grams 22%
The speed of the train is 25 km/h * 60 km/h = 120 km/h
The speed of the train is 25 km/h * 120 km/h = 30 km/h
Therefore, it is 30 km/h * 30 km/h = 120 km/h
</bot>
<user>
A train travels 10 kilometers per hour. If it takes 15 minutes to reach the same train, how long will it take for the train to reach the same train?
</user>
<bot>
The tra

--- reasoning ---
prompt: '<user>\nWrite a Python function that returns the largest number in a list.\n</user>\n<bot>\n<think>\n'
[written]  repeated 8-grams 16%
Okay, so I need to find the largest number in a list where all the numbers in the list are larger than the number itself. Hmm, let's start by understanding the problem.

First, the numbers are larger than the numbers. So the numbers are lists of numbers. Now, the task is to check if a number has a larger number than it

--- wikipedia ---
prompt: '== History ==\nThe [[Roman Empire]] was '
[written]  repeated 8-grams 16%
founded in [[1692]] and [[Roman Niville]] was laid out in [[Greenville]]. The [[Constitutional Revolution]] is now based on the [[Roman Empire]] and [[Spain]].

==Economy==
The department of the [[Roman Empire]] founded a number of [[surveyor]]s including [[Battle of Hammond]], [[Battle of Hammond]], [[Battle of San Fr

--- chess ---
prompt: '<g>1700 1-0 1. e4 e5 2. '
[written]  repeated 8-grams 0%   38 legal moves, then Bc2
Bc4 Nf6 3. d3 Bc5 4. Bg5 h6 5. Bxf6 Qxf6 6. Nf3 d6 7. Nc3 c6 8. Bb3 Be6 9. Nd2 Nd7 10. Nc4 Bxc4 11. dxc4 O-O 12. O-O Qg6 13. Kh1 Nf6 14. f3 Qh5 15. Qe1 Nh7 16. Qg3 f5 17. Rad1 f4 18. Qxg6 Nxg5 19. Rd2 Nh7 20. Rfd1 Rad8 21. Bc2 Ng5 22. Bb3 Kh8 23. Ne2 Ne6 24. b3 Rf6 25. c5 dxc5 26. Bxe6 Rd6 27. Bb1 g6 28. Rxd6 cxd6 29. 

--- self-knowledge ---
prompt: '<user>\nhow do you decide which experts to use?\n</user>\n<bot>\n'
[written]  repeated 8-grams 1%
Three matrices. Two read a vector of 512 into 2048 hidden units, one gating the other through a smooth curve, and the third writes the result back down to 512. That is 3.15 million numbers per expert.
</bot>
<user>
what is the router
</user>
<bot>
At each depth a small matrix, one row per expert, scores every expert I have against the sta
```

</details>
<!-- /auto:run-blocks -->


## Motivation

Every language model you can actually own today is a model somebody else trained and then froze. You can fine-tune around the edges of it, but you cannot train one from scratch on your own hardware, and you cannot keep training it on what you do day to day - the moment you try, it forgets what it knew before. The result is that a personal model is always somebody else's model with a thin layer of you on top, and it stops learning the day it ships.

**mini-AGI** model has small enough GPU footprint that it is possible to train end-to-end on one consumer card, and it is built so that training never has to stop. It reads a stream of characters one chunk at a time, takes a gradient step on each, and the same path serves generation. There is no separate fine-tuning regime and no frozen base: reading and being trained are the same event.

Three constraints shape everything else in the design:

- **It has to fit on 8 GB.** Not with quantisation - training needs gradients and optimiser state, which is roughly three times the weights again. So the weights live on disk and only the experts the current forward uses are on the card.
- **It has to avoid catastrophic forgetting.** It learns from everything it reads, so new material must not overwrite what it already knows completely.
- **It has to be able to read anything.** The alphabet is the 256 byte values, so there is no tokenizer to fit and no data type that needs a new vocabulary.

The model is genuinely yours: trained on your hardware, on your data, that keeps learning from every conversation you have with it, and that nobody else can take it away or switch it off. 

<!--**Watch this video for a detailed explanation**
[here will be the link to the video when its ready]()-->

## How the architecture works

Characters (bytes) does not pass through a fixed stack of layers as it would be in a traditional LLM. Instead, it passes through **two dense prelude blocks** and then through **one recurrent block applied up to 24 times**, each application choosing its own experts from a shared pool. The latent state between applications is never decoded - it is merged with the embedded input by an adapter each time round, so the loop cannot drift away from the text it is reading.

Three distinct blocks, up to 26 block-applications per character.

![mini-AGI's layers: an input embedding; two prelude layers of attention and feed-forward; one recurrent layer - the previous state h joined to the encoded text x, a linear adapter, attention, and a mixture of experts behind a router - applied up to 24 times with the same weights; a norm and two heads, the next character and the halt probability. Beside it, inside each attention layer: 8 heads of 64 numbers, rotary positions on queries and keys, the cache of keys and values they read, and the concat and projection back into the character's state](assets/layers.svg)

*Every box is a layer, every ⊕ a residual add, and the shaded groups are what repeats: the prelude layer twice, the recurrent layer up to 24 times with the same weights. On the right, the inside of each attention layer - its 8 heads, the cache they read, and where their output goes.*

- **Adaptive depth.** A halting head scores every character at every row, and the character stops as soon as another row would not change the answer. Easy characters take one row, hard ones take many. A character that has stopped is finished: its experts stop running, and the characters after it read its final state at the deeper rows. Training does exactly the same, with the PonderNet recipe inside it: every row a character runs is weighted by its halting probability, so the halting head learns through those weights.
- **Routing per block-application, not per character.** Each recurrent application a character runs, up to 24, picks its own top-8 experts from the 32 on the card, so one character touches far more of the pool than "top-8" suggests, and the same expert can be selected several times at different depths. What varies is *which* eight at each point.
- **No expert is assigned a subject.** There are no labels anywhere. Soft top-k routing distributes capability across the pool by itself, and a character can combine fragments from several experts. The cost is that capabilities share parameters and so *can* interfere.

![how the model processes one character](assets/shape.gif)

**This is the architecture assembling itself, one character at a time**, captured from the live model.

Each tile on the left is one expert; colour is expert identity and stays the same for the whole clip. A **row** is one application of the recurrent block, and the eight tiles in it are the eight experts that row actually ran. The stack grows downward as the model keeps going, and the amber line is where halting stopped it - **the grey rows below are computation the model declined to spend.**

The trace on the right is how many rows each character took. It moves constantly between 9 and 15 against a ceiling of 24, and the caret under the text shows which character is being read. The text is a held-out story, read one character at a time after its first 2,048, and every character is a forward of its own: it adds its requests to the story's vote and routes among the 32 experts the story so far has asked for most, each row picking its eight from those. Over these 160 characters the card does not change once - the 2,048 characters before them have already voted, and the story goes on asking for the same experts.

Positions are rotary and carry no learned parameters, which is why the context window can be extended by continued training rather than by re-initialising anything.

### ...and the same thing while it writes

![how the model generates text](assets/generate.gif)

The clip above is the model **reading** - every character is held-out text it is being shown. This one is the model **writing**: it was primed with 2,495 characters of held-out stories and then continued on its own, so the grey text is what it was given and **the green text is entirely its own**. Greedy decoding with the repetition guard - the `adapted` reading in the sample log - and no sampling anywhere: run it twice and you get the same sentence.

Two things are worth watching. The stack behaves the same way, because generating and reading are the same forward pass in this model - the only difference is whether the next character comes from a file or from the model's own argmax - and it costs about the same: 11.8 rows a character over the 140 characters written, against 12.1 over the 160 read from the same stories. And **the whole text chooses the experts**, the way it does in training. The prompt's forward admitted the 32 its characters asked for most; after that, every character added its own requests to the vote and routed among what the prompt and the reply so far had asked for. This reply never moved the vote far enough to change the card, so not one expert was loaded after the prompt; a reply that wanders from its prompt takes the card with it. Letting each character choose alone instead, as an earlier version did, made its predictions on the same held-out text a fifth worse, 1.35 nats a character against 1.11, and garbled its words. Nothing was written back: nothing is trained while it writes.

What it produced, continuing the story about a cherry tree that the prompt ends on:

> They would stay inside, but only for any more.
>
> One day, the cherry tree saw another cherry. It was very big, and it had long, shiny wings.

Spelled right and on the story it was given, but "only for any more" means nothing, and the other cherry has wings - a fair picture of where the model was at 648.6M characters, when the clip was recorded.

## How paging works

Every expert is a file on disk holding its weights and its optimiser moments. Above disk sit a RAM cache and the card:

| | key | what it is |
|---|---|---|
| disk | — | every expert the model has; bounded by free space |
| RAM | `ram_cache` | the experts used most recently, least-recently-used evicted |
| VRAM | `resident` | the experts the current forward admitted - what a character may route through |

Selection is one rule: **the text chooses**. Every character ranks the **whole pool** with the router - one row per expert, always on the card, so an expert on disk is scored exactly like one in VRAM - and asks for its top 8, each request carrying the probability the router gave it. Every forward's first pass adds its characters' requests to the **vote of its text** - everything read since position 0 - and the forward is admitted the experts its text has voted for most, until the card's 32 slots are full. Every character then routes with the same router over the admitted experts only and takes its top 8 of those, so a character whose request was not admitted gets its best admitted experts instead.

The 32 is what VRAM can hold for training: everything a training forward used has to stay on the card through its backward pass and the optimiser step. In training a window is a text of its own, and the first pass of its 4,096 characters already asks for well over a hundred experts, so the window's 32 are its first pass's most requested. **When the model writes, every character is a forward of its own** that adds its requests to the vote the prompt and the reply so far have cast, and routes among the 32 that whole text has asked for most - the choice a training window over that text would make, cut off at the character being written. The set follows the text as it goes and moves only when the vote does, so a reply loads a new expert now and then rather than at every character, from RAM or from disk.

Three rules the project holds to:

- **Every expert is stepped with its own moments.** The optimiser's moments belong to the expert, not to the slot of VRAM it happens to occupy. Stepping it with whatever its slot held would hand it the momentum of the expert before it, and training would carry on looking healthy while every swapped expert inherited a stranger's history. Only a step reads them, so they travel only to a step: an expert comes to the card with its weights alone, and just before each optimiser step every expert on the card that is not already holding its own moments gets them from RAM or disk. A reply loads experts at almost every character and steps none of them, so writing moves no moments at all.
- **An expert already on the card stays in the slot it is in.** Admission is a set, not a ranking. An expert the new forward also admits is never moved; a newcomer takes the slot of an expert this forward did not admit - an empty slot first, then the one admitted longest ago. The number of loads is exactly the number of admitted experts that were not already there, and consecutive windows of one subject ask for nearly the same experts: over fifteen minutes of real training a step loaded 2.4 experts on average.
- **An expert nothing trained is not written back.** An expert leaving the card is copied back to RAM - and later to disk - only if the optimiser stepped it while it was there. A reply loads experts at almost every character and trains none of them, so each is still exactly its copy, and writing it back would cost a disk write per load for nothing.

**What was on the card before changes the cost, never the choice.** Admission reads only the text and the weights, so the same text admits the same experts whatever was read before it; history decides only how many of them have to be loaded. And because the rows that admit experts are the rows that route every character, admission is learned by the ordinary gradient. When a character mixes its experts, the loss raises the router score of each one whose output helped more than the mixture as a whole and lowers the score of each that helped less. Only the rows of experts the window admitted receive that signal, and the next text whose states look like those asks more strongly for the experts that helped.

**While it trains, the router is taught to use the whole pool.** That gradient only reaches experts that get chosen, and left alone the router keeps choosing the same ones: reading held-out text on its own, it put just 59 of its 118 experts on a card across every subject. So the loss carries a balance term - the [Switch Transformer's](https://arxiv.org/abs/2101.03961), with one change. Every expert keeps a running share of the recent training forwards that admitted it, and every forward that trains pays, on the router's probability for each expert over the whole pool, in proportion to that share: `balance × (experts × Σ share × probability − 1)`, zero when use is even. Probability on a busy expert costs more than on an idle one, so the router's rows move toward the idle. Switch charges each expert by its share of the current batch; here the share is taken over the last thousand or so training forwards, because a training window is one text and a text should be free to want few experts - what has to be even is the use across texts. Only the router's rows learn from it, since the characters' states are detached, so it cannot bend what the trunk computes. And because it changes the router itself, reading and writing choose the way training does. In a 15-minute test at weight 0.01, the experts the router put on a card across held-out rose from 59 to 114, at a cost of 0.023 in held-out loss over those minutes. The exploration bonus it replaced acted only while training: over the same 15 minutes it spread the training work across 116 experts, while the model reading on its own still used 58. The weight sets how hard the term leans against learning the text on the experts in use, and the default is gentler, 0.003. An expert no text admits gets no other gradient. Under Adam that was enough: Adam scales each weight's step by that weight's own history, so it moved an idle row at the usual pace whatever the weight. Muon spends each step on the router's largest directions, and an idle row's balance gradient is too small to move it, so newborns are brought in by the routing bias instead (below).

## How growth and pruning work

The pool grows when it is short of capacity and shrinks when parts of it stop being asked for.

New experts are added on speculation, at a small gate so they change almost nothing, and kept only if something goes on asking for them. A new expert is built by **recombination** - whole hidden units taken from several existing experts - because a clone of one parent is not novel enough to be worth routing to, and a random expert computes nothing worth routing to. Its router row is its parents' rows averaged, weighted by how many units each gave, so it starts out scoring every character with the same weighted average of its parents' scores. With parents drawn at random, that average sits near the pool's average row, and no text in particular asks for it: none of the first 22 experts born under Muon was ever admitted. So while a newborn is on trial, the **routing bias** lifts it. The bias is an offset added to its score where experts are chosen, never to the weights it is mixed with. It rises a little after every training step in which characters chose the newborn less than a tenth of an even share, and falls back to zero as the newborn's own row earns that share. A lifted request also counts lifted in the card's vote, so the newborn reaches the card and gets chosen, and its row, gate and weights learn from what it computes. When its trial ends the offset goes, and it lives or dies by its own use.

Growth is refused unless every brake agrees:

- **room** - disk and VRAM can take it
- **used** - the capacity already added is being asked for
- **earning** - the previous cohort survived its trial
- **fits** - not too many experts are already inside their trial
- **honest** - train and held-out have not separated

**Dead means unaddressed.** Both the growth brake and the pruner read how long it has been since a forward last admitted an expert, and never its gate. The gate is not merely uninformative here, it is anti-predictive: the smallest gates belong to the *busiest* experts - one that behaves as a sink, chosen constantly and contributing little per character, reads as dead on a gate test, while a high-gate expert nothing has asked for in a long time reads as alive.

A new expert is safe for a full survival window no matter what, so it cannot be judged before it has had a chance to be chosen. When the model grows an expert a new file appears; when it prunes one, that file is deleted. 

## How continual learning works

Training on a single stream, one subject at a time, is the classic recipe for catastrophic forgetting. The test here is deliberately the worst case: the model is switched cold onto **PG19** - 19th-century novels, a domain it has never read - and made to read **1,048,576 consecutive characters of it and nothing else**, at batch 1.

**The trunk learning rate is the mechanism.** The trunk - embeddings, attention, routers, the halting head - is the part every character passes through, and it carries **98% of the squared gradient norm**. Running it at 0.1x the experts' rate is the difference between a model that absorbs a new domain and one that is wrecked by it.

| configuration | PG19, the new domain | the 8 it already knew | learned per nat forgotten | retained vs chance |
|---|---|---|---|---|
| experts frozen, trunk LR = expert LR | -0.2909 | +1.2663 | 0.23 | 74.12% |
| swapping, trunk LR = expert LR | -0.3106 | +1.2628 | 0.25 | 74.25% |
| **swapping, trunk at 0.1x - what the run uses** | **-0.4216** | **+0.1297** | **3.25** | **97.30%** |
| *interleaved: PG19 added as a 9th lane* | *-0.3124* | *-0.0065* | *nothing forgotten* | *100.13%* |

*These arms were measured before expert selection moved to the rule under [How paging works](#how-paging-works), when the experts on the card were re-chosen before every chunk; the probe now runs the current rule.*

The fourth column is the exchange rate: nats gained on the new domain for every nat lost across the eight. **The mitigated configuration is 13x better at that trade than either unmitigated one** - and interleaved there is no trade at all.

![Reading a new domain under three configurations](assets/mitigations.png)

Panel A is what happened to the **eight subjects the model did not read** - zero means nothing was forgotten. Two configurations climb to +1.27 nats, which is the model losing most of what it knew. The third, at a trunk learning rate one tenth of the experts', reaches **+0.13 after more than a million characters of a single unfamiliar domain**.

Panel B: **what it learned while it was there**. The mitigated configuration is not trading plasticity for retention - it learns PG19 *faster* than either unmitigated arm, **-0.4216 nats against -0.3106 and -0.2909**, while forgetting ten times less. Slowing the trunk does not slow learning. It accelerates it, because the trunk stops being dragged around by every passage and the experts are free to specialise.

### This is a worst case, not a use case

A million consecutive characters of one subject is **32 passages back to back**. The run never does this: it reads a passage of 32,768 characters, moves to another subject, and comes back to each one within about a million characters at most - sooner to the subjects still learning fastest. It is the analogous to a person who does one thing for a solid week and a person who changes activity through the day. 

Nothing in normal use looks like the massed arm either. A conversation wanders, and a model reading your files reads whatever is there. Such regime only arises deliberately - a bot specialised on one subject and fed nothing else for a long stretch.

So, the last row here is the one that describes the actual system:

```
pg19         1.6773 -> 1.3649   -0.3124   (read)
reasoning    0.7076 -> 0.6846   -0.0230   (read)
code         0.6552 -> 0.6404   -0.0148   (read)
chat         0.7402 -> 0.7289   -0.0113   (read)
stories      0.5599 -> 0.5537   -0.0063   (read)
wikipedia    1.2549 -> 1.2493   -0.0056   (read)
arithmetic   0.6419 -> 0.6433   +0.0014   (read)
chess        0.4955 -> 0.4989   +0.0035   (read)
chat_hermes  1.1043 -> 1.1085   +0.0042   (withheld)
```

![PG19 read as one of eight interleaved subjects](assets/probe_interleaved.png)

**Add a new domain as a ninth lane and six of the nine improve.** PG19 falls by 0.31 nats and the eight the model already knew improve by 0.0065 on average - nothing moves more than +0.004 in the wrong direction, and the subject it was best at is untouched. Adding a domain to this model costs nothing: 100.13% retained is the eight coming out very slightly ahead of where they started. In the figure the eight are the flat band at zero and PG19 is the line leaving it - the same read that costs 0.13 nats when it is massed costs nothing when it is interleaved.

Two readings matter here:

- **The expert pool is not what prevents forgetting.** Freezing the experts on the card - removing the one property that makes the pool a pool - changes forgetting by 0.0035 nats, which is nothing, and it *learns the new domain slowest of all three*. In that arm 137 of 174 experts received no gradient at all and the model still collapsed. Preserving most of the weights is not sufficient; the trunk is where the damage happens.
- **The cost of learning is real, and it is small.** On genuinely new material the massed arm pays **0.13 nats across eight domains to gain 0.42 on a ninth** - a real exchange rate, and a favourable one. Interleaved, the exchange disappears.

![Every subject during a massed read, and how much of the pool was touched](assets/probe_massed.png)

**What that same read looks like from the inside.** This is the working configuration - trunk at 0.1x - during the 1,048,576-character PG19 read. On the left, every subject against where it started. PG19 drops away from the pack; the eight withheld subjects drift up together, and the ones that drift most are **chat, stories and reasoning** - the prose-like lanes, nearest to Victorian novels. Chess and arithmetic barely move, at +0.009 and +0.043. The damage lands where the representations overlap, which is what the routing story predicts.

The right panel is why the damage is bounded at all. Over the whole read only **44 of 174 experts received any gradient** - 75% of the model was structurally untouched, because routing never selected it: the pool confines an update to the part of the model that the text actually addressed.

**Which subject comes next follows the evidence too.** Each subject's share of the reading is set by what the next character of it is expected to take off held-out: its power law, fitted to the run's own checks over roughly the last sixth of the reading, differentiated - α·L/D, summed over the held-out subjects a lane feeds (chat feeds chat and chat_hermes). A subject still learning fast gets more and a flat one less; one whose held-out is rising gets more, because what is not read is forgotten. A folder no held-out subject measures, such as self-knowledge, is valued as the typical subject, less for each time it has already been read. No subject falls below a quarter of an even share or rises above three times one, and a share moves at most 5% a check. The shares are of characters: a visit reads 32,768 characters of one subject - part of a long file, or short files joined end to end by `<|endoftext|>` - so the subject changes once a passage and each gets exactly its share. `--lane-schedule equal` gives every subject an even share; `--lane-weights` pins the shares by hand (minagi/lanes.py).

**The learning rate is not scheduled.** A cosine schedule asserts that the run ends, which for a model that reads continually is false. Instead a controller nudges the rate at every held-out evaluation, by an amount set by how clearly held-out is still improving - read off two exponentially weighted fits of held-out against time, one remembering about 66 evaluations and one about 12. Unless the improvement is clear, the rate eases down a fraction of a percent at a time: on a plateau about 0.2% an evaluation, a tenth or less in six hours, never more than 0.4% at once and never below 0.1% of the base rate. Clear improvement holds it up and can raise it by at most 0.08% an evaluation, and a confirmed jump in held-out, which means the material changed, doubles it. Easing is the resting posture because on this run a lower rate was measured to help on a plateau: from the same weights, the same 3M characters read at a third of the rate came out ahead on every held-out check and in every subject. Three things are not evidence: a reading that does not hold at the next check (this run's spikes are single readings 0.01 to 0.02 above their neighbours), an evaluation with nothing read since the last, and the step a restart makes in held-out, which is measured on the first five readings after it and taken out, so a changed setting is neither progress nor harm.

<details>
<summary><b>Replicate this measurement yourself</b> - the probe, the results, and the weights it was measured on</summary>

The probe, all four arms and the figures ship in [`replication/`](replication/).

**Weights for the checkpoint every number above was measured on:**

| data read | held-out | experts | download |
|---|---|---|---|
| 428.2M characters | 0.7702 nats / 1.1112 bits/byte | 174 | [Volotat/mini-AGI-cl-replication-weights](https://huggingface.co/Volotat/mini-AGI-cl-replication-weights/tree/main/weights) |

Download the `weights/` folder into the repository root. Held-out there is the probe's own baseline - the eight training domains under `data/val`, 16 chunks each - which is what every figure above is measured against, and is evaluated on fewer chunks than the training run's own log.

**To run it:**

```bash
python3 -m corpora all                             # data/train and data/val
python3 -m corpora pg19 --split validation \
        --out data/cl/pg19                         # the new domain, 50 books

cp -a weights /tmp/w                               # a COPY - paging marks experts dirty
python3 replication/forgetting_probe.py --weights /tmp/w \
    --domain pg19 --read-root data/cl --steps 512 \
    --at 0,64,128,256,384,512 --eval-chunks 16 --chunk 2048 \
    --lr 2.08e-4 --held-out data/cl_val \
    --trunk-lr-mult 0.1 --out replication/results/mine.json

python3 replication/cl_summary.py                  # the table, control verdict first
python3 replication/plot_figures.py                # redraws the figures above
```

`--read-root` keeps the new domain outside `data/train` on purpose: put PG19 in the corpus and the training run would start reading it too. `data/cl_val` holds the eight existing held-out sets plus PG19 books that are never read, so "PG19 improved" cannot be memorisation. The PG19 **validation** split is used rather than the test split, so the PG19 benchmark further down this page stays untouched.

Set `--trunk-lr-mult 1.0` for the unmitigated arm, add `--no-swap` to freeze the experts the first window admits, and pass `--rotate pg19,chess,code,stories,arithmetic,wikipedia,chat,reasoning` for the interleaved control.

**Read the control first.** `cl_summary.py` prints a verdict on it before anything else: every lane is read there, so forgetting is impossible by construction and any degradation is the instrument rather than the model. 

</details>


## Reading your own files

```bash
python3 train.py read ~/notes                     # a dry read - nothing kept
python3 train.py read ~/src ~/docs --passes 3 --save
```

Point it at files or directories. There is nothing to prepare: the alphabet is the 256 byte values, so a file is already written in the only vocabulary the model has. Directories are walked, binaries are skipped by sampling their contents rather than trusting the extension, and each file is read from its beginning to its end because a document has an order.

It is the same path training uses: same chunking, same cache, same gradient step.

| flag | |
|---|---|
| `--passes N` | read the whole set N times |
| `--save` | keep what it learned; without it `weights/` is untouched |
| `--lr` | default 5e-5, below a training run: reading should adjust the model, not overwrite it |
| `--mix ""` | skip the before/after scoring |

Two defaults worth knowing. **Nothing is saved without `--save`**, so a read is a dry run until you decide otherwise. And it scores the held-out mixture before and after, then says plainly if reading your files cost the model ground elsewhere.

## Benchmarks

The numbers below are for tracking purposes and move as the run continues. Held-out loss is reported with its standard error, and the size of the evaluation is what sets that error - a difference smaller than it is noise.

Held-out is read as whole blocks of 4,096 characters spread evenly over everything each subject holds in `data/val`, the same blocks every time, and a block scores exactly what it would if its file were read from the top. Each round's check reads 8 blocks of each subject, and a few blocks sit a little above or below the whole set, so every session starts with a 30-block evaluation and reports each of its checks at that level: what the check reads, plus the difference the two had at the start. The run has read held-out this way since 18 Oct 2026, and the checks from before are shown moved onto it - [memory/corrections.md](memory/corrections.md) says how.

There is a second variance underneath these figures. The same configuration run twice lands about 0.014 apart, because the expert dispatch is not deterministic on CUDA. **Treat about 0.03 as the threshold for a real difference**, not the error bar printed beside one score.

<!-- auto:benchmarks -->
**Where the model is** (5,048.3M characters read, 424 experts; held-out measured on the model as it stands, on 30 blocks of 4,096 characters spread over all of each subject):

| | nats/char | bits/byte |
|---|---|---|
| **held-out, all 8 subjects** | **0.5582** ± 0.0120 | **0.8053** |
| train | 0.5276 | 0.7612 |

**Held-out loss per subject:**

| Subject | nats/char | bits/byte |
|---|---|---|
| `chess` | 0.374 | 0.540 |
| `stories` | 0.443 | 0.639 |
| `chat` | 0.459 | 0.662 |
| `code` | 0.460 | 0.664 |
| `reasoning` | 0.523 | 0.755 |
| `arithmetic` | 0.612 | 0.883 |
| `chat_hermes` | 0.749 | 1.081 |
| `wikipedia` | 0.846 | 1.221 |
<!-- /auto:benchmarks -->

### enwik9 against published models

enwik9 is the first billion bytes of English Wikipedia from March 2006, the text compressors and language models are most often compared on. Published models of up to 1.6B parameters, in bits per byte:

<!-- auto:enwik9 -->
| model | params | date | trained on | training cost ↓ | scored in | enwik9 bits/byte ↓ |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1.23B | Sep 2024 | 9T tokens (~36 TB) | ~$0.9M | [Tsai, *Revisiting Data Compression with Language Modeling*, Jan 2026](https://arxiv.org/abs/2601.02875) | **0.68** |
| Chinchilla 1B | 1B | Mar 2022 | ~1.4T tokens (~5.6 TB, est.) | ~$68k (est.) | [Delétang et al., *Language Modeling Is Compression*, Sep 2023](https://arxiv.org/abs/2309.10668) | 0.90 |
| [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl) | 1.56B | Feb 2019 | 40 GB, passes not published | ~$43k | measured | 1.07 |
| mini-AGI (best effort)\*\* | 1.34B | 8 Nov 2026 | 4.6 GB | **~$81** | measured | 1.09 |
| [GPT-2 Large](https://huggingface.co/openai-community/gpt2-large) | 0.77B | Feb 2019 | 40 GB, passes not published | ~$21k (est.) | measured | 1.12 |
| [GPT-2 Medium](https://huggingface.co/openai-community/gpt2-medium) | 0.35B | Feb 2019 | 40 GB, passes not published | ~$10k (est.) | measured | 1.17 |
| mini-AGI (dynamic evaluation)\* | 1.34B | 7 Nov 2026 | 4.6 GB | **~$81** | measured | 1.18 |
| mini-AGI (frozen) | 1.34B | 7 Nov 2026 | 4.6 GB | **~$81** | measured | 1.26 |
| [GPT-2 Small](https://huggingface.co/openai-community/gpt2) | 0.12B | Feb 2019 | 40 GB, passes not published | ~$3k (est.) | measured | 1.33 |

\* **Learning while it reads**, known as dynamic evaluation: each piece is scored first and then learned from, so no byte is scored by weights that have already seen it. Read frozen, the same weights score 1.26. The learning is done on a copy of the weights, which is thrown away after the measurement, so nothing learned from enwik9 ever reaches the weights the run goes on training, or any later score. For mini-AGI this is simply how it runs: it is a continual learning model, and this is the learning it does in training and in every conversation, on the same code path and at its own rate, which its controller sets as it does in training. The frozen models above would need a separate fine-tuning loop added to be scored this way ([Krause et al., 2017](https://arxiv.org/abs/1709.07432)), so this row is closer to online compressors that learn as they compress, such as NNCP and cmix, than to the frozen rows.

\*\* **Best effort**: the model used exactly as it is used in training and in conversation - every page of the slice, whole, in its native 4,096-byte window, sliding 2,048 characters at a time, learning as it goes on a copy that is thrown away afterwards, as above - every chunk scored before it is learned from, and what it learns carried into everything it reads after. This is its most natural score, but it sees more of each page at once than the 2,048-byte pieces the other mini-AGI rows, GPT-2 and Chinchilla 1B are scored on, so it is not on the same footing as them.

<details>
<summary>How these numbers were worked out</summary>

**Date** is when each model came out: Llama-3.2-1B's release, the Chinchilla paper (its weights were never released), and the GPT-2 paper (the Small weights came with it; Medium, Large and XL followed in May, Aug and Nov 2019); mini-AGI's is the day these weights were measured, as it is still training.

**Trained on** is how much text each model learned from, and **training cost** roughly what that training cost when it came out:

- Llama-3.2-1B: up to 9 trillion tokens and 370k H100 GPU-hours, both from [its model card](https://huggingface.co/meta-llama/Llama-3.2-1B). 9 trillion tokens is about 36 TB of text at roughly 4 bytes a token, and the cost is those hours at about $2.5 each, a typical cloud price in late 2024. It leaves out the Llama 3.1 8B and 70B models whose outputs it was also trained on.
- Chinchilla 1B: DeepMind has not published how much it was trained on or what that cost, so both are a best-effort estimate, which assumes it read as much as Chinchilla 70B did ([Hoffmann et al.](https://arxiv.org/abs/2203.15556)): 1.4T tokens, about 5.6 TB of text, and 8.4×10<sup>21</sup> FLOPs for 1B parameters - about 21,000 chip-hours of TPU v4 at 40% utilisation, at $3.22 a chip-hour, its price in 2022. For scale, EleutherAI's Pythia-1B took 4,830 A100-hours to read 300B tokens ([Biderman et al.](https://arxiv.org/abs/2304.01373), Table 5), about 22,500 for 1.4T. It may have read less: the Chinchilla paper's compute-optimal amount for 1B parameters is 20.2B tokens (Table 3), which would have cost about 70 times less.
- GPT-2 XL: [WebText](https://en.wikipedia.org/wiki/GPT-2), 40 GB of text from 8 million web pages; OpenAI has not said how many times it went over it. The cost is Andrej Karpathy's estimate, quoted there: 32 TPU v3 chips for 7 days at about $8 per chip-hour.
- GPT-2 Large, Medium and Small: the same WebText. Their costs are best-effort estimates: Karpathy's figure for GPT-2 XL scaled by parameter count (0.77B, 0.35B and 0.12B ÷ 1.56B), which assumes the same data and the same hardware.
- mini-AGI: everything it has read so far, and its 1,001 hours on one RTX 3070 Laptop GPU at $0.081 an hour - the cheapest rental of an RTX 3070 that no other job can interrupt, Vast.ai's on [computeprices.com](https://computeprices.com/providers/vast/gpus/rtx3070) on 17 Oct 2026, for a desktop card faster than this laptop's. Salad Cloud rents one for $0.04, but only in its lowest priority tier, which any better-paying job can take the card from.

**Bits per byte** - lower is better - is how many bits a model needs, on average, to encode each byte of the text: the total -log<sub>2</sub> of the probabilities it gave the text, divided by the text's length in bytes. The papers report a **compression rate** instead - compressed size ÷ original size - and bits per byte is that rate × 8:

- Llama-3.2-1B: [Tsai](https://arxiv.org/abs/2601.02875), Table 2, row "Llama-3.2-1B", quantization "None", raw compression rate: **0.085**, and 0.085 × 8 = 0.68.
- Chinchilla 1B: [Delétang et al.](https://arxiv.org/abs/2309.10668), Table 1, row "Chinchilla 1B", column enwik9 under "Raw Compression Rate (%)": **11.3%**, and 0.113 × 8 = 0.90.
- GPT-2 XL, GPT-2 Large, GPT-2 Medium and GPT-2 Small are measured here the same way as mini-AGI - the GPT-2 paper reports enwik8, not enwik9 - on the same 2,000 pieces of the same slice: each cut back to whole UTF-8 characters at its two ends, tokenized, and started from `<|endoftext|>`, at full precision. Bits per byte is the total -log<sub>2</sub> probability of the tokens ÷ the pieces' bytes, ± 0.005 for GPT-2 XL, 0.005 for GPT-2 Large, 0.005 for GPT-2 Medium, 0.006 for GPT-2 Small.
- mini-AGI (frozen) is measured directly, the same quantity without running an arithmetic coder: its loss on each byte, converted from nats to bits (÷ ln 2) and averaged, with its weights frozen - it learns nothing from the pieces while they are scored. It is measured the way Chinchilla 1B was - enwik9 cut into pieces of 2,048 bytes, each predicted with no memory of the others - on 2,000 pieces spread over a 50 MB slice of enwik9 that it never trained on (its Wikipedia comes from the same 2006 dump, without these pages), ± 0.005. Llama-3.2-1B was given 2,048 tokens, about 8 KB, at a time.
- mini-AGI (dynamic evaluation) reads the same 2,000 pieces in the same order, as one stream it goes on learning from: each piece is scored alone, exactly as in the frozen measurement, and then the model takes one optimiser step on it at 2.1×10<sup>-6</sup> - its base rate times the factor its training had come down to, then set by its controller as in training, from a held-out check every 10 minutes - 3 of them, ending at 2.1×10<sup>-6</sup>, with Adam's moments as training left them. Learning is the only difference: read frozen, the same weights score 1.259, and learning took 0.076 ± 0.0026 off - 0.047 over the first quarter of the pieces and 0.090 over the last, so it was still gaining at the end.
- mini-AGI (best effort) reads whole pages, as it reads anything: every one of the slice's 17,426 pages, in the order the slice holds them, framed as the Wikipedia files it trains on are - one page from `<page>` to `</page>` - each from its top in its 4,096-byte window sliding by 2,048 characters, 49.92M characters in all, every chunk scored first and then learned from at 2.1×10<sup>-6</sup> - its base rate times the factor its training had come down to, then set by its controller as in training, from a held-out check every 10 minutes - 71 of them, ending at 1.6×10<sup>-6</sup>, so what it learns from one page carries into every page after it. Read frozen in the same windows, the same pages score 1.122, and chunk by chunk alone 1.181.
</details>
<!-- /auto:enwik9 -->

These mini-AGI rows are slightly optimistic, as the note at the top of [Benchmarks](#benchmarks) explains: scored the way it writes, the frozen row comes out about 0.02 bits per byte higher. The other two were not re-measured.

### PG19 against published models

PG19 is a set of books from Project Gutenberg published before 1919, gathered by [Rae et al., 2019](https://arxiv.org/abs/1911.05507) for long-range language modelling. Its test split is 100 whole books, 41 MB, and it is the set the byte-level models on the scaling chart below report their scores on. In bits per byte:

<!-- auto:pg19 -->
| model | params | date | trained on | training cost ↓ | context, bytes | scored in | PG19 bits/byte ↓ |
|---|---|---|---|---|---|---|---|
| MambaByte-972M | 0.97B | Jan 2024 | 75 GB of PG19 | ~$4k (est.) | 8,192 | [Wang et al., *MambaByte*, Jan 2024](https://arxiv.org/abs/2401.13660) | **0.883** |
| MegaByte-758M+262M | 1.02B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 8,192 | [Yu et al., *MEGABYTE*, May 2023](https://arxiv.org/abs/2305.07185) | 1.000 |
| Transformer-320M | 0.32B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 1,024 | [Yu et al., *MEGABYTE*, May 2023](https://arxiv.org/abs/2305.07185) | 1.057 |
| PerceiverAR-248M | 0.25B | May 2023 | 80 GB of PG19 | ~$1.6k (est.) | 8,192 | [Yu et al., *MEGABYTE*, May 2023](https://arxiv.org/abs/2305.07185) | 1.104 |
| mini-AGI (best effort)\* | 1.34B | 9 Nov 2026 | 4.6 GB, none of it PG19 | **~$81** | 4,096 | measured | 1.557 |
| mini-AGI (frozen) | 1.34B | 8 Nov 2026 | 4.6 GB, none of it PG19 | **~$81** | 4,096 | measured | 2.010 |

The published models were trained on PG19's own training books, 75 to 80 GB of them, and are scored on more books like those. mini-AGI has never been trained on PG19 and has read no Victorian novels, so for it these books are out of its domain.

\* **Best effort**: the model used exactly as it is used in training and in conversation - whole books, each from its top, in its native 4,096-byte window sliding 2,048 characters at a time, and learning as it reads, as in the enwik9 table above: every chunk is scored first and then learned from, on a copy of the weights that is thrown away afterwards. It reads all 100 books of the split, the frozen row's books, one after another, so what it learns from one book carries into every book after it.

<details>
<summary>How these numbers were worked out</summary>

**Date** is when each model came out: the MEGABYTE paper for the three trained there, and the MambaByte paper; mini-AGI's is the day those weights were measured, and its size the pool it had then, as it is still training and growing.

**Trained on** is how much text each model learned from, and **training cost** roughly what that training cost when it came out. Neither paper says what its training cost or what it ran on, so the published costs are best-effort estimates of what their authors paid: the training compute by the 6ND rule - 6 × the parameters each byte runs through × the bytes trained on - on A100s at 40% utilisation, as the enwik9 notes assume for Chinchilla, at what AWS charged on demand for an A100 when the papers came out: $32.77 an hour for a machine with eight, $4.10 each, the same on its pricing page in [May 2023](https://web.archive.org/web/20230518023128/https://aws.amazon.com/ec2/instance-types/p4/), [December 2023](https://web.archive.org/web/20231221084854/https://aws.amazon.com/ec2/instance-types/p4/) and [April 2024](https://web.archive.org/web/20240402175315/https://aws.amazon.com/ec2/instance-types/p4/). Smaller clouds rented A100s for less - [Lambda](https://web.archive.org/web/20230512094749/https://lambdalabs.com/service/gpu-cloud) for $1.10 in May 2023, about a quarter as much - so the real cost may have been lower:

- MegaByte-758M+262M, Transformer-320M and PerceiverAR-248M: 80 billion bytes of PG19's training books each ([Yu et al.](https://arxiv.org/abs/2305.07185), §5) - about eight passes over PG19, which is 10.1 GB in all (their Table 1) - on one compute budget the three share. MegaByte runs about 357M parameters a byte - its 262M local model on every byte and its 758M global model once per 8 - so that is 1.7×10<sup>20</sup> FLOPs, about 380 A100-hours.
- MambaByte-972M: 75 billion bytes when it scored this, of the 150 billion it went on to - 380,000 steps of 48 × 8,192 ([Wang et al.](https://arxiv.org/abs/2401.13660), Appendix C). All 972M of its parameters run on every byte, so that is 4.4×10<sup>20</sup> FLOPs, about 970 A100-hours.
- mini-AGI: everything it had read when measured, from its own corpus with no PG19 in it, and the hours it had trained for by then on one RTX 3070 Laptop GPU at $0.081 an hour, priced as in the enwik9 table above - the cheapest rental of that card, which no big cloud offers: 1,001 hours for the frozen row and 1,001 hours for the best effort.

**Context** is how many bytes each model reads at once: 1,024 for the Transformer, 8,192 for the other published models, and 4,096 for mini-AGI, the window it was built with.

**Bits per byte** - lower is better - is printed by both papers. [Yu et al.](https://arxiv.org/abs/2305.07185) give 1.057 for the Transformer, 1.104 for PerceiverAR and 1.000 for MegaByte, in the PG-19 column of their Table 2. [Wang et al.](https://arxiv.org/abs/2401.13660) give 0.883 for MambaByte-972M in their Table 5, where they time how fast it writes, and 0.863 for the same model scored with a sliding window; the table shows the first. MambaByte is shown by its best model only: the smaller MambaByte-353M, which Wang et al. train on the same compute per byte as MegaByte-758M+262M, scores 0.930 after 30 billion bytes (their Table 2).

mini-AGI's two rows are measured here:

- mini-AGI (frozen) is measured by tools/eval_pg19.py over every book of the test split - 100 books, 41,289,001 bytes - in consecutive windows of 4,096 bytes, each read with no memory of the one before and with the weights frozen. Its loss on each byte is converted from nats to bits (÷ ln 2) and pooled by byte: total bits ÷ total bytes.
- mini-AGI (best effort) is measured by tools/measure_enwik9_adapt.py --natural on all 100 books of the split, in order, 41.29M bytes, learning at 2.1×10<sup>-6</sup> - its base rate times the factor its training had come down to, then set by its controller as in training, from a held-out check every 10 minutes - 55 of them, ending at 1.7×10<sup>-6</sup>. Read frozen in the same windows, the same books score 1.963, and chunk by chunk alone 2.048.

</details>
<!-- /auto:pg19 -->

<details>
<summary>Every mini-AGI score in this section is slightly optimistic</summary>
[#33](https://github.com/volotat/mini-AGI/issues/33). When text is scored, many characters go through the model in one forward, and that forward's 32 experts are chosen by a vote that includes all of them, so a character near the start of a text is predicted with experts that the characters after it helped choose. Writing is not affected: every character the model writes is a forward of its own. Scored the way it writes instead - each character among the experts chosen by the text up to it - held-out comes out 0.0084 ± 0.0014 nats a character higher, 0.012 bits per byte or about 1.5%, and cold 2,048-byte enwik9 pieces 0.022 ± 0.002 bits per byte higher, nearly all of it in each text's first 2,048 characters (7 Nov 2026, on 3 blocks of each subject and 30 pieces). The scoring stays as it is because it reads text exactly the way training does.
</details>

### Data Scaling

<!-- auto:scaling -->
![Data scaling on PG19 and enwik9](assets/scaling.png)

Everything purple on this chart is **bits-per-byte on the PG19 test split** - one held-out set, so it compares directly: the diamonds are this model, the hollow circles the published byte-level models in the PG19 table above. This model scores **2.010 BPB** over the whole split (100 books, 41,289,001 bytes) at a context of 4,096, against its own mixture's 0.82. PG19 is out of distribution for it: it was trained on a corpus assembled for this project and has read no Victorian novels, so much of that gap is subject matter rather than capability.

The green squares are this model on **enwik9**, the set the enwik9 table above compares it with published models on: Wikipedia, which it trains on, though never on these pages. It scores **1.259 BPB** there after 4.58B characters. The dotted line carries it on to one pass of the corpus at the rate enwik9 itself improved at between its 9 measurements (α 0.070), where it reaches **1.21**. The hollow green circle far to the right is Llama-3.2-1B, the only model in that table whose training data is published: 0.68 after about 36 TB - about 7,900 times what this model has read, scored on 2,048 tokens, about 8 KB, at a time. The rest are not drawn, because how much they read was never published.

The red line is the fitted power law on this model's own held-out, `L ∝ D^-0.171` with R² 0.95 over every point past the warmup - between Kaplan's 0.095 and Chinchilla's 0.28, and it has held for more than a decade of data. How steep it looks depends on where the fit starts, and the band on the chart spans that range.

Read straight off that trend, on this model's own mixture. It has read 5.05B characters so far, in about 45 days of running. The days below assume the pace of the last 2 days of it: 1,513 characters a second on the wall clock, held-out checks and rounds of samples included, because the reading waits for them.

| held-out | total data read | further reading | days from here at ~1,513 char/s |
|---|---|---|---|
| 0.80 BPB | 5.25B | +0.20B | ~2 |
| **0.74 BPB** | 8.04B | +2.99B | **~23** |

The first one is days of reading on one laptop GPU, and it sits inside a single pass of the 8.04B-character corpus.

The right panel shows which subjects are still moving. chat, code, reasoning, stories are the steep ones; arithmetic and wikipedia have the shallowest slopes.
<!-- /auto:scaling -->

### Reproducing the benchmarks

Every mini-AGI row of the enwik9 and PG19 tables, and the scaling chart, comes from one script, which measures whatever model is in `weights/` - the [undertrained snapshot](https://huggingface.co/Volotat/mini-AGI-undertrained/tree/main) above, say, or one you trained yourself:

```bash
python3 -m corpora enwik9          # the 50 MB enwik9 slice, checked against the one measured here
python3 -m corpora pg19            # PG19's 100 test books
./measure_benchmarks.sh            # about a day on an RTX 3070 Laptop GPU
```

It needs the GPU to itself, so stop any training run first. Every measurement that learns also runs the learning rate's controller exactly as training does - the controller is part of the model, and sets how fast it adapts - so it reads the held-out set in `data/val`, which `python3 -m corpora all` builds along with the corpus; without it the rate stays where the run left it. Every measurement that learns does it on a copy of the weights that is deleted afterwards, so `weights/` is never changed, and the script then rewrites this README's tables and chart with what it measured. The GPT-2 rows of the enwik9 table were measured on the same pieces by `tools/measure_enwik9_gpt2.py`.

## GPU support

NVIDIA and AMD GPUs both run it, and so does a CPU on its own. Speeds at the settings this repository trains with - a 4,096-character window, read 2,048 characters at a time, 32 experts on the card:

| Hardware | Computes in | Reading | Writing | VRAM used |
|---|---|---|---|---|
| NVIDIA RTX 3070 Laptop, 8 GB | bf16 | 2,042 char/s | 28.4 char/s | 5.1 GiB |
| AMD Radeon RX 6800 XT, 16 GB | fp32 (its default) | 1,388 char/s | 34.7 char/s | 10.1 GiB |
| AMD Radeon RX 6800 XT, 16 GB | bf16 | 301 char/s | 15.1 char/s | 7.1 GiB |
| AMD Ryzen 9 9900X, CPU only | bf16 | 186 char/s | 27.6 char/s | - |

Reading is learning: characters trained on per second. Writing is a reply being generated. The speeds were measured on early snapshots of the weights, with 127-133 experts. Only 32 experts are on the card at a time, so the pool's size barely moves VRAM, but a larger pool still reads more slowly, because less of it fits in RAM and more comes from disk: the current model, 355 experts, reads about 1,310 characters a second on the 3070.

- **NVIDIA** - tested on the RTX 3070 Laptop. Every card from the RTX 30 series on computes bf16 in hardware.
- **AMD** - needs PyTorch's ROCm build (tested: torch 2.9.1 on ROCm 6.4). The RX 6000 series (RDNA2) has no bf16 arithmetic, so on a card with 14 GB or more the model computes in fp32 by itself: 4.6 times faster than bf16 on the 6800 XT. A card with less stays in bf16, the third row: it peaks at 7.1 GiB, so it fits an 8 GB card, at about a fifth of fp32's speed. On a bigger card `MINAGI_GPU_BF16=1` chooses it, to leave memory for something else. PyTorch ships no fused attention kernels for these cards, so attention is computed in blocks - without the memory blow-up of PyTorch's own fallback.
- **CPU** - bf16 where the CPU has it in hardware (AMD Zen 4 and later, Intel with AVX-512 BF16 or AMX), fp32 otherwise.

`python3 tools/device_check.py` measures what a particular machine does with the model: which attention kernels it has, bf16 against fp32, and one learning step.

## Running it

1. Make sure you have an NVIDIA or AMD GPU with at least 8 GB of VRAM - or just a CPU, see [GPU support](#gpu-support) - and Python 3.10 or newer. The reference machine is an RTX 3070 Laptop GPU with 8 GB.
2. Clone the repository:
    ```bash
    git clone <repository-url>
    cd mini-AGI
    ```
3. Install the dependencies:
    ```bash
    pip install torch numpy pyyaml matplotlib      # the model, and its graphs
    pip install flask                              # serve.py
    pip install chess zstandard datasets           # building corpora
    pip install scipy                              # a few of the analysis tools
    ```
    PyTorch has to match your CUDA version, or be the ROCm build on an AMD card - see [the PyTorch install page](https://pytorch.org/get-started/locally/). The reference environment is torch 2.6.0+cu124 with numpy 1.24.4. Only the first line is needed to train.
4. Build the corpus. One command downloads the four public datasets and generates the other four lanes:
    ```bash
    python3 -m corpora all                  # all eight subjects, a few GB
    python3 -m corpora all --limit 5000     # a small slice first, to try it
    python3 -m corpora all --full           # entire datasets: tens of GB, hours
    ```
    Lanes already on disk are left alone, so an interrupted build can simply be run again. Individual lanes are available too - `python3 -m corpora` lists them - or skip this entirely and point the model at your own files.
5. Start reading. The weights directory is created from `config.yaml` the first time, so there is nothing to set up:
    ```bash
    python3 train.py read data/train --save --weights-dir weights \
        --held-out data/val --sample-every 10
    ```
6. Serve it:
    ```bash
    python3 serve.py --port 8080            # then open http://127.0.0.1:8080
    ```

The run writes a sample log, redraws its graphs as it goes, and checkpoints every few minutes. It is meant to be left alone for days.

### Everything else

```bash
python3 -m minagi.store weights                    # what the model is right now
python3 -m corpora                                 # every corpus target
python3 -m corpora all --only wikipedia stories    # rebuild particular lanes
python3 -m corpora expand                          # .bin -> the text files read

python3 train.py read --help                       # every knob the reader has
python3 train.py stream --steps 140000 --lr 2e-4   # the packed-corpus path
python3 train.py ponder-probe --ckpt weights       # depth against difficulty
```

Every tool takes `--ckpt weights` - the directory is the model, and there are no `.pt` files to keep track of.

## Initialization

A fresh model starts small and grows into its shape. The context window begins at `model.context_start` and extends one character at a time, but only when the model is still getting something out of the far end of the window it already has. The expert pool begins at `pool.experts` and grows from there.

This means the first hours of a run look nothing like the rest of it. Loss falls fast, the pool churns, the window is short, and the learning-rate controller has not gathered enough evaluations to act. None of that is a problem to fix.

Nothing rolls back on its own. A run saves to `weights/` every five minutes, at every held-out check and when it stops, so even a crash costs at most a few minutes of reading. To keep an earlier state, copy the directory.

## Layout

```
minagi/          the model. no command lines here.
  config.py        reading config.yaml, which building and training both use
  precision.py     what the model computes in, and how moments are stored
  tokenizer.py     bytes in, bytes out - 256 values plus structural markers
  model.py         the transformer: RMSNorm, rotary positions, SwiGLU, flash attention
  decode.py        how a character is chosen, without a random number generator
  ingest.py        turning a pile of files into something to read
  pool.py          the expert pool, and the rules by which it grows and shrinks
  paged.py         the same pool spread over disk, RAM and VRAM
  recur.py         latent recurrence with adaptive depth
  stream.py        reading a corpus behind a KV cache, one chunk at a time
  lanes.py         how the reading is shared between subjects
  store.py         the weights directory, which IS the model
  optim.py         how much of a gradient is signal
  muon.py          the optimiser: Muon for weight matrices, AdamW for the rest
  plasticity.py    the learning rate, governed by held-out loss
  governor.py      the same controller, wherever the model learns
  heldout.py       the level a held-out check reports at
  live.py          serving a model that is being trained underneath
  report.py        the model reading statistics off its own weights
  create.py        writing a fresh weights directory from config.yaml

train.py         read | stream | ponder-probe
serve.py         local web UI
config.yaml      the settings worth changing
corpora/         python3 -m corpora all - the whole corpus, downloaded and made
memory/          the project's history, kept for agents
weights/         one file per expert. this directory is the model.
```

`weights/` is written on the first run and `data/` by `corpora`; neither is in
the repository. Everything else above is.

`memory/` is the project's history: every change, the tests that were run and what
they found, what worked and what did not, and the corrections made to how the run
was measured while it trained. It is kept mostly to preserve that history, for an
agent picking the project up at any point, so it is written for an agent to read
rather than for a person.

### The weights directory is the model

```
weights/
  manifest.json     what exists, its shape, and where it came from
  core.npz          embeddings, attention, norms, adapter, halting head
  routers.npz       the gate, and the router: one row per expert per site
  optim.npz         the optimiser's state for the trunk and the routers
  experts/          one file per expert: w1, w3, w2 and its own optimiser state
    e00000.npz ...
```

Training resumes from it - weights, optimiser state and step count - and saves into it every five minutes and at every held-out check, so the directory is the model as it last stood. Writes are atomic: every file is written to a `.tmp` and renamed, so an interrupted save cannot leave a half-written weight behind.

The directory is written on the first run.

## The model

Byte level - vocabulary 265: the 256 byte values plus 9 structural markers (`<think>…</think>` scratchpad, `<user>/<bot>` turns, `<g>` for games, and end-of-text). Context 4,096.

| | |
|---|---|
| body | RMSNorm, RoPE, SwiGLU, flash attention via `scaled_dot_product_attention` |
| depth | 3 distinct blocks, up to 26 block-applications per character |
| recurrence | one weight-shared block applied up to 24 times; the latent is never decoded |
| halting | PonderNet - each character halts independently, so hard ones get more depth |
| routing | top-8 experts per block-application, chosen per character |
| paging | 32 experts resident on the card; the rest live on disk |

<!-- auto:params -->
The parameter count moves, because the pool grows and prunes itself while training. `python3 -m minagi.store weights` prints what it is now. At step 2,465,331:

```
core        8.27M  embeddings, attention, norms, adapter, halting head
routers     0.22M  one row per expert at each call site, depth embedding, gates
experts  1333.8M   424 x 3.15M each  (3 x 512 x 2048)
----------------------
total    1342.3M
```

**VRAM is set by the card's 32 slots, not by the pool.** Only 32 experts are resident at a time - about 109M parameters of the 1342M - which is why the pool can keep growing on an 8 GB card. Per byte the model activates about **345M** parameters - two prelude blocks, then attention and top-8 of the resident experts on each recurrent step, counted at the 12.8 steps training samples its depth around (a character may take up to 24) - so by the 6ND rule it costs the same per byte as a dense 345M byte-level transformer, not a 1342M one. That is the figure the scaling chart in Benchmarks is drawn against.
<!-- /auto:params -->

## AI usage

This project was assisted by "Claude Opus" 5 and 5.5 models. The model did implemented most of code of this project, verified and debugged it when it was necessary. The model was searching for published papers related to the problems that the project were trying to solve, build tests and experiments, and help with brainstorming the complex problems that arose along the way. The animations, graphs and other media you see here are all done by Claude as well form the real data traces. While I myself provided main ideas, steering, intuition, rejections when thing went in a wrong direction, code monitoring and verification, as well as decisions and strong opinions of how everything should be wired together and work in principle. Documentation was written in tandem.  

## Contributions

Everyone is welcome to contribute to this project in whatever way suits them, with or without AI. I only ask that you show you care about what you contribute enough by writing the PR description yourself first, in your own words, and only then add AI-written technical details if they are needed. I want to see your intent and what you are trying to achieve. As the maintainer, I reserve the right to decline a contribution, including when I cannot understand what you are trying to do or I do not think the contribution fits the project. Obviously, ill intended PRs are going to be refused as well. For anything large, please open a discussion first, so we can agree on the direction before you put the work in. Big PRs without a clear justifications for a large changes are going to be refused as well. 

You are always welcome to fork the project, and that may even be the better way to adapt it to your own needs. I keep an eye on active forks, and when one has an idea worth borrowing, this project may adopt it, with credit, even without a direct merge, creating some sort of soft cross-pollination.

## Acknowledgments

[PyTorch](https://pytorch.org/) does the arithmetic, [NumPy](https://numpy.org/) holds the weights on disk, and [Matplotlib](https://matplotlib.org/) draws every graph.

The parts the model is built out of:

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017. The entire expert pool, and the load-balancing auxiliary loss.  
[Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021. The capacity-based batched dispatch, which is what lets the pool run as three matrix multiplies.  
[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) - Banino et al., 2021. The adaptive depth mechanism.  
[RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021. Why the context window can grow by continued training.  
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020. SwiGLU.  
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019.  
[FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022. Reached through PyTorch's `scaled_dot_product_attention`.  
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017. AdamW.  
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - Chen et al., 2016. Gradient checkpointing, which on 8 GB is not optional.  
[Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) - Hutchins et al., 2022. Carrying a recurrent state across blocks.

[ZeRO-Offload](https://arxiv.org/abs/2101.06840) - Ren et al., 2021, and [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) - Rajbhandari et al., 2021. Training a model larger than the GPU's memory, by offloading to CPU memory and disk.  
[Dynamic Mixture of Experts Against Severe Distribution Shifts](https://arxiv.org/abs/2511.18987) - Kim et al., 2025. Adds experts to a mixture of experts while it trains, which is how this model's pool grows.

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022. Chinchilla: the compute-optimal balance of parameters and training data.  
[The Pile](https://arxiv.org/abs/2101.00027) - Gao et al., 2020. Bits per UTF-8 byte, chosen there for invariance to tokenisation.  
[Transformer-XL](https://arxiv.org/abs/1901.02860) - Dai et al., 2019, and [Compressive Transformers](https://arxiv.org/abs/1911.05507) - Rae et al., 2019. The character-level benchmarks to aim at.  
[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - McCandlish et al., 2018. The gradient noise scale.  
[The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137) - Pagliardini et al., 2024. Tried as the trunk's optimiser; at the paper's settings AdamW did better here, so it was removed.  
[A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417) — Su et al., 2022. Contrastive search, implemented as an alternative decoder. The default path does not use it, but the idea that degeneration is a property of the decoding rule rather than of the weights is what led to the adaptation trace that replaced it.  
[Muon](https://kellerjordan.github.io/posts/muon/) — Keller Jordan, 2024. Orthogonalised momentum. The run's optimiser for every weight matrix since 3.19B characters, applied as in [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (Liu et al., 2025): from the same weights and 30M characters of the same text, it ended 0.0039 nats lower on held-out than AdamW.  

[TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) — Zhang et al., 2024. 1.1B parameters on three trillion tokens, sixteen A100s for ninety days. The closest published analogue to a small team training a small model from scratch.  
[MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) — Yu et al., 2023. 1.000 bits per byte on PG19 after reading 80 billion bytes, and one of the published byte-level points on the scaling chart.  
[MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660) — Wang et al., 2024. 0.883 bits per byte on PG19 from its 972M model, after 75 billion bytes - the best of the published points on the chart.  
[General-purpose, long-context autoregressive modeling with Perceiver AR](https://arxiv.org/abs/2202.07765) — Hawthorne et al., 2022. 1.104 bits per byte on PG19, the third published point.  
[Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Gemma Team, 2025. The 1B model reads two trillion tokens and holds a 32k context.  
[SmolLM2: When Smol Goes Big](https://arxiv.org/abs/2502.02737) — Allal et al., 2025. 1.7B on eleven trillion tokens, with a detailed account of the data work.  
[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) — Biderman et al., 2023. Publishes the GPU-hours per model size: 4,830 A100-hours for the 1B.  
[Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) — Meta, 2024. 370k H100-hours for the 1B, though it is pruned and distilled from larger models rather than trained from scratch, so it is not a like-for-like number.  
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — Qwen Team, 2026. Smaller than this model and holds a 262k context, by making three quarters of its layers linear-attention.  


[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Dai et al., 2019. 0.99 bits per character on enwik8 and 1.08 on text8, from 277M parameters.  
[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507) — Rae et al., 2019. 0.97 on enwik8; the next marker after Transformer-XL. It is also the paper that introduced PG19, the out-of-distribution test used here.  
[Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — Anthony, Biderman & Schoelkopf, 2023. C = 6ND, and the bytes-per-parameter arithmetic that says what a training checkpoint must weigh.  
[The Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/) — Hugging Face, 2025. Reports 161,280 GPU-hours of ablations against 276,480 for the main run.  
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI. The standard suite, not used here yet: at this scale HellaSwag, ARC and the rest score at chance.  
[Why Does the Effective Context Length of LLMs Fall Short?](https://arxiv.org/abs/2410.18745) — An et al., 2024. Models use less of their context than they are given, which is why the 4,096-character window here is reported as a setting rather than as a capability.  

The corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [the Lichess open database](https://database.lichess.org/). Wikipedia and the source-code portion come from public dumps and public repositories.

<details>
<summary><b>Some additional sources that might affected the decisions in this project</b></summary>

[Infinite test-time scaling | Talk at MIT](https://www.youtube.com/watch?v=CKdlVoMhHwU) - mostly about prefix-sliding approach for "infinite" context. While interesting, I am pretty sure it is vastly inferior to "context with full attention for immediate memory + carry latent for short-term memory (basically LSTM) + weights update for long-term memory" approach.  
[SGD for batch-1 training](https://www.reddit.com/r/LocalLLaMA/comments/1wm1gab/comment/pbr1kpf/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) - proposal for using SGD for batch-1 training, as it might be just as affective as AdamW at this bach size. This will allow to cut VRAM utilization almost in half giving spare memory for something else. Definitely going to be tested in the future.  

</details>


## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{Borsky_mini_AGI_2026,
  author = {Borsky, Alexey},
  month = {9},
  title = {{mini-AGI: A Continually Learning Byte-Level Language Model}},
  url = {https://github.com/volotat/mini-AGI},
  version = {1.0.0},
  year = {2026}
}
```
