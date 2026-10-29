#!/usr/bin/env python3
"""
One way of moving text through the model, used for training and for serving.

Here there is one primitive, `read`. It walks a character stream, keeps a KV
cache, and consumes the stream a chunk at a time. With `learn=True` each chunk
takes a gradient step; with `learn=False` nothing is updated. Both cost the
same, because both keep the same cache and the same length of graph.

Two knobs, and they are independent:

    CHUNK     how much text is inside one autograd graph. This sets the peak
              activation cost and nothing else. It is not a batch: the chunks
              are consecutive, and the cache carries between them.

    CONTEXT   how far back attention reaches. This sets the cache size, which
              is linear in context and cheap - about 0.107 MB per character
              across every cache slot the depth uses.

Because the graph is bounded by CHUNK rather than by CONTEXT, context is nearly
free to extend. That is what makes the trade worth making: batch 1 gives up
gradient averaging and buys back an order of magnitude of context.

What does NOT cross a chunk boundary is gradient. The cache carries the model's
view of everything it has read; backpropagation reaches only to the start of
the current chunk. The model can attend to something 30,000 characters back and
cannot learn a weight update through it. That is the cost of the design, and it
is stated here rather than discovered later.
"""

import os

import numpy as np
import torch

from .precision import amp


def detach_caches(caches):
    """Carry the cache forward without carrying the graph with it."""
    for c in caches:
        if c["k"] is not None:
            c["k"] = c["k"].detach()
            c["v"] = c["v"].detach()
    return caches


class Reader:
    """
    A cursor into one character stream, with its own cache.

    One of these per source. Round-robin between several and every source stays
    a continuous stream while the gradient sequence stays mixed. That mixing is
    what stops one corpus read end-to-end from displacing the last; interleaving
    keeps the memory shape of a stream and the mixing behaviour of a batch.
    """

    def __init__(self, model, data, name="", chunk=512, context=None,
                 device=None, rng=None):
        self.model = model
        self.data = data                     # a uint16 memmap of the corpus
        self.name = name
        self.chunk = chunk
        self.context = context or model.cfg.block
        self.device = device or next(model.parameters()).device
        self.rng = rng or np.random.default_rng(0)
        self.caches = model.empty_caches()
        self.pos = int(self.rng.integers(0, max(1, len(data) - self.context - 1)))
        self.seen = 0                        # characters read since the reset
        self.resets = 0

    def reset(self, jump=True):
        """Start a fresh window somewhere else in the corpus."""
        self.caches = self.model.empty_caches()
        self.seen = 0
        self.resets += 1
        if jump:
            self.pos = int(self.rng.integers(
                0, max(1, len(self.data) - self.context - 1)))

    def next_chunk(self):
        n = self.chunk
        if self.pos + n + 1 >= len(self.data):
            self.reset()
        a = self.data[self.pos:self.pos + n + 1].astype(np.int64)
        self.pos += n
        x = torch.from_numpy(a[:-1]).to(self.device).unsqueeze(0)
        y = torch.from_numpy(a[1:]).to(self.device).unsqueeze(0)
        return x, y

    def step(self, learn=True, aux_weight=0.0):
        """
        Read the next chunk. Returns the loss on it.

        A reader never reads past its own context. When the window is full it
        starts a new one somewhere else, which is what the training loop was
        already doing from the outside.

        It has to. Sliding the window instead - trimming the cache and holding
        the offset at its last value - looks reasonable and is wrong: the keys
        left in the cache carry the rotary encoding of the positions they were
        computed at, while every new query would claim a position near the end
        of the window. The relative distances RoPE encodes would stop matching
        and stay mismatched for the rest of the read, which costs whole nats of
        held-out loss without touching a weight.
        """
        # `seen + chunk`, not `seen`: the test has to cover the chunk about to
        # be read, because the read starts at `seen` and the rotary tables end
        # at the context. Testing `seen` alone is only equivalent when the
        # chunk divides the context exactly, and reads past the tables when it
        # does not.
        if self.seen + self.chunk > self.context:
            self.reset()
        x, y = self.next_chunk()
        offset = self.seen
        with amp(self.device):
            logits, loss = self.model(x, y, caches=self.caches,
                                      pos_offset=offset)
        if aux_weight and getattr(self.model, "pool", None) is not None:
            loss = loss + aux_weight * self.model.pool_aux()
        if hasattr(self.model, "pool_balance"):
            loss = loss + self.model.pool_balance()
        self.seen += x.shape[1]
        if not learn:
            detach_caches(self.caches)
        return loss


class StreamSet:
    """
    Several sources, but only ONE cache live at a time.

    A cache is linear in context - about 0.107 MB per character across every
    cache slot - which is cheap once and ruinous several times over. Holding one
    reader per source at full context multiplies the largest term in the memory
    budget by the number of sources, which is the difference between fitting on
    the card and not.

    Since every window starts at a fresh random offset anyway, there is nothing
    in a cache worth preserving across a switch. So the set keeps one active
    reader, reads a whole window from it, and then picks the next source by
    weight and starts a new window there. Sources still interleave; they
    interleave at window granularity rather than chunk granularity, and the
    memory is that of a single stream.
    """

    def __init__(self, model, sources, chunk, context, device, rng,
                 replay=0.0, replay_burst=8, replay_marks=512):
        self.model = model
        self.chunk = chunk
        self.context = context
        self.device = device
        self.rng = rng
        # Replay, at a batch of one, cannot mean mixing old rows into a batch -
        # there is only one row. So it means revisiting a POSITION: the set of
        # places the stream has already been is remembered, and now and then
        # the reader is sent back to one of them instead of pressing on.
        #
        # It goes back for a burst of consecutive chunks rather than a single
        # one, because a lone chunk read with an empty cache has no context and
        # is not the thing that was read the first time.
        self.replay = replay
        self.replay_burst = replay_burst
        self.marks = []                      # (source index, position)
        self.max_marks = replay_marks
        self.replaying = 0                   # chunks left in this burst
        self.replays = 0
        self.names = [n for n, _ in sources]
        self.weights = np.array([w for _, w in sources], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.data = [np.memmap(os.path.join(n, "train.bin"), dtype=np.uint16,
                               mode="r") for n in self.names]
        self.active = None
        self.switches = 0
        self._pick()

    def _pick(self, at=None):
        """Start a window. `at` re-opens a remembered one instead of a new one."""
        if at is None:
            i = int(self.rng.choice(len(self.names), p=self.weights))
            pos = None
        else:
            i, pos = at
        r = Reader(self.model, self.data[i], self.names[i],
                   self.chunk, self.context, self.device, self.rng)
        r.source_index = i
        if pos is not None:
            r.pos = pos
        elif len(self.marks) < self.max_marks:
            self.marks.append((i, r.pos))
        else:
            # a reservoir, so the buffer stays a sample of the whole run
            # rather than of its beginning
            j = int(self.rng.integers(0, self.switches + 1))
            if j < self.max_marks:
                self.marks[j] = (i, r.pos)
        self.active = r
        self.switches += 1
        return r

    def step(self, learn=True, aux_weight=0.0):
        r = self.active
        if self.replaying > 0:
            self.replaying -= 1
        elif r.seen + self.chunk > self.context:   # see Reader.step
            if (self.replay > 0 and self.marks
                    and self.rng.random() < self.replay):
                mark = self.marks[int(self.rng.integers(0, len(self.marks)))]
                r = self._pick(at=mark)
                self.replaying = self.replay_burst - 1
                self.replays += 1
            else:
                r = self._pick()
        loss = r.step(learn=learn, aux_weight=aux_weight)
        return loss, r.name

    def set_context(self, context):
        """Extend the window. Takes effect at the next switch."""
        self.context = context


class FileReader:
    """
    A cursor through one file, read start to finish.

    Unlike Reader, which samples random windows out of a large corpus, this
    reads a file in order and stops at the end. That is what reading something
    means: a document has a beginning, and the model should see it before it
    sees the middle.

    GRADIENTS REACH THE WHOLE WINDOW. Each learning step re-forwards the whole
    window ending at the cursor, scores ALL of it, and backpropagates through
    all of it. There is no cache on this path and nothing to detach - a
    detached cache is exactly what would let the model attend over tens of
    thousands of characters while being taught by only the newest chunk, which
    is a model attending to context it can never be corrected by.

    Scoring the whole window rather than only the new chunk also means the last
    window's worth of text is re-learned at every step, which for a model that
    never stops training is rehearsal rather than waste.

    THE COST IS ARITHMETIC. With reach R and a step every S characters, every
    character is forwarded R/S times, so reading is that many times slower than
    attending alone. A character also contributes to about R/S consecutive
    updates, which makes those updates highly correlated and the newest chunk a
    small share of each. That is the price of learning from what you read.
    """

    def __init__(self, model, data, name, chunk, context, device):
        self.model, self.data, self.name = model, data, name
        self.chunk, self.context, self.device = chunk, context, device
        self.caches = None            # only the measuring path carries one
        self.pos = 0
        self.seen = 0

    def done(self):
        return self.pos + 1 >= len(self.data)

    def peek(self):
        """The next chunk's ids, without consuming it.

        Lets the model be asked what it wants before it reads, so the working
        set is chosen for the text about to arrive rather than the text just
        gone.
        """
        n = min(self.chunk, len(self.data) - self.pos - 1)
        if n < 2:
            return None
        a = self.data[self.pos:self.pos + n].astype(np.int64)
        return torch.from_numpy(a).to(self.device).unsqueeze(0)

    def step(self, learn=True, aux_weight=0.0):
        n = min(self.chunk, len(self.data) - self.pos - 1)
        if n < 2:
            return None
        if not learn:
            return self._measure(n)
        end = self.pos + n
        # the window is the last `context` characters ending at the cursor, so
        # early in a file it is short and grows - which is the warm-up, and
        # needs no special case
        start = max(0, end - self.context)
        a = self.data[start:end + 1].astype(np.int64)
        x = torch.from_numpy(a[:-1]).to(self.device).unsqueeze(0)
        y = torch.from_numpy(a[1:]).to(self.device).unsqueeze(0)
        with amp(self.device):
            # pos_offset 0: the window is its own sequence. Rotary positions
            # are relative, so a character sliding from index 16,383 to 15,871
            # keeps every offset that matters.
            logits, loss = self.model(x, y, caches=None, pos_offset=0)
        # the model's own loss, as held-out reads it - what is added below
        # only steers the router
        self.model_loss = loss.detach()
        if aux_weight and getattr(self.model, "pool", None) is not None:
            loss = loss + aux_weight * self.model.pool_aux()
        if hasattr(self.model, "pool_balance"):
            loss = loss + self.model.pool_balance()
        self.pos = end
        self.seen += n
        return loss

    def _measure(self, n):
        """
        The same number, without paying for gradient reach nobody uses.

        MEASURING IS NOT LEARNING. Re-forwarding the whole window every chunk
        exists so the gradient can reach all of it; under no_grad there is no
        gradient, and a cached chunk sees exactly the same attention context,
        so it computes the identical loss for 1/(context/chunk) of the work.

        That factor is the difference between an evaluation that runs between
        samples and one that costs more than the training it is measuring.

        Cap each read by the remaining rotary table. Training already windows
        to `context`; evaluation used to feed a full `chunk` from position 0,
        which crashes when chunk > context (#18).
        """
        if self.caches is None or self.seen + n > self.context:
            self.caches = self.model.empty_caches()
            self.seen = 0
        n = min(n, self.context - self.seen)
        if n < 2:
            return None
        a = self.data[self.pos:self.pos + n + 1].astype(np.int64)
        x = torch.from_numpy(a[:-1]).to(self.device).unsqueeze(0)
        y = torch.from_numpy(a[1:]).to(self.device).unsqueeze(0)
        with amp(self.device):
            logits, loss = self.model(x, y, caches=self.caches,
                                      pos_offset=self.seen)
        self.pos += n
        self.seen += n
        detach_caches(self.caches)
        return loss


class FolderEvaluator:
    """
    Held-out loss over a folder of text files.

    The same thing Evaluator does, for text on disk rather than a packed .bin.
    Each sub-folder is scored separately, so a mixture still reports per
    domain, and the files are read in order the way `read` reads them.

    A FILE IS SCORED AS INDEPENDENT BLOCKS. Measuring clears the attention
    cache whenever the next chunk would take it past `context` characters,
    and a forward from position 0 starts its text's expert vote afresh, so a
    file is scored as blocks of `self.block` characters from its first, and
    what a block scores depends on its own characters and nothing read before
    them. Reading one block alone gives exactly what reading the file from
    the top gives for that block - which is what lets an evaluation choose
    WHICH blocks it reads.

    WHICH BLOCKS. `run` reads each subject's files from the top until it has
    its chunks: the first documents and nothing else. With spread=True it
    reads the same number of chunks as whole blocks spread evenly over
    everything the subject holds - block j of n is the one (j + 0.5) / n of
    the way through the subject's blocks, file after file - and the same
    blocks every time, so two evaluations differ only by the model. On this
    run's held-out, reading from the top meant the opening of one 2.9 MB
    wikipedia file and about one reasoning trace of 400, and the first 30,720
    characters of a subject and the next 92,160 differed by 0.12 in one
    subject and by 0.10 the other way in another.

    WHOLE BLOCKS ONLY. A file seldom ends on a block boundary, and what is
    left after its last whole block used to be a block of its own - however
    short. A chunk's loss is a mean over its characters and a subject's is a
    mean over its chunks, so a 68-character leftover counted as much as 2,048
    characters, all of them predicted from almost no context. Three short
    chunks (73, 68 and 32 characters) were 21% of chat_hermes's check, which
    read ~1.0 nats against 0.74 for the subject, with a noise of 0.025 a check
    against 0.002 before. With every block whole, every chunk is the same
    length. `whole=False` keeps the old blocks, for measuring what that
    change moved; a subject with no whole block at all keeps its pieces.
    """

    def __init__(self, model, root, chunk, context, device, per_domain=True,
                 whole=True):
        from minagi.ingest import collect, as_stream
        self._as_stream = as_stream
        self.model, self.chunk, self.context = model, chunk, context
        self.device = device
        self.per = max(1, context // chunk)       # chunks between cache clears
        self.block = chunk * self.per
        self.whole = whole
        self._blocks = {}            # subject -> every (file, start), on demand
        self._picked = {}            # (subject, n) -> the blocks' characters
        self.groups = {}
        subs = [d for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))] if per_domain else []
        if subs:
            for d in subs:
                self.groups[d] = collect([os.path.join(root, d)])
        else:
            self.groups["all"] = collect([root])

    def blocks(self, name):
        """Every whole block of one subject in reading order, as (file,
        start): the blocks reading it from the top would score, from the same
        files, leaving out what is left of each file after its last whole
        block (see WHOLE BLOCKS ONLY)."""
        if name not in self._blocks:
            out, whole = [], []
            for f in self.groups[name]:
                n = len(self._as_stream(f))
                if n < 8:                # reading from the top skips these too
                    continue
                # a block needs a first chunk of two characters to score
                out.extend((f, s) for s in range(0, n - 2, self.block))
                # a whole block is `per` full chunks: its characters and
                # one past them, the target of its last
                whole.extend((f, s) for s in range(0, n - self.block,
                                                   self.block))
            self._blocks[name] = whole if (self.whole and whole) else out
        return self._blocks[name]

    def spread(self, name, n):
        """n of a subject's blocks spread evenly over all of them; every block
        when there are no more than n."""
        every = self.blocks(name)
        if n >= len(every):
            return list(every)
        return [every[int((j + 0.5) * len(every) / n)] for j in range(n)]

    def _texts(self, name, n):
        """The characters of the spread blocks, read from disk once and kept,
        so the text every evaluation scores is the text the first one did."""
        key = (name, n)
        if key not in self._picked:
            texts, data, at = [], None, None
            for f, s in self.spread(name, n):
                if f != at:
                    data, at = self._as_stream(f), f
                # one character past the block: the target of its last one
                texts.append(data[s:s + self.block + 1].copy())
            self._picked[key] = texts
        return self._picked[key]

    def score_block(self, data, name):
        """The chunk losses of one block, read alone."""
        r = FileReader(self.model, data, name, self.chunk, self.context,
                       self.device)
        out = []
        while len(out) < self.per and not r.done():
            l = r.step(learn=False)
            if l is None:
                break
            out.append(float(l))
        return out

    @torch.no_grad()
    def run(self, chunks=240, spread=False):
        self.model.eval()
        out, every = {}, []
        per = self.per
        if spread:
            means = []
            n = max(1, -(-int(chunks) // per))    # whole blocks, rounding up
            for name in self.groups:
                losses = []
                for data in self._texts(name, n):
                    got = self.score_block(data, name)
                    losses.extend(got)
                    if got:
                        means.append(float(np.mean(got)))
                if losses:
                    out[name] = float(np.mean(losses))
            # block means, as below, where a window of `per` chunks is a block
            out["stderr"] = (float(np.std(means, ddof=1) / np.sqrt(len(means)))
                             if len(means) >= 2 else 0.0)
            self.model.train()
            return out
        for name, files in self.groups.items():
            losses = []
            fi = 0
            while len(losses) < chunks and fi < len(files):
                data = self._as_stream(files[fi]); fi += 1
                if len(data) < 8:
                    continue
                r = FileReader(self.model, data, name, self.chunk,
                               self.context, self.device)
                # Each file is read the way reading measures it: a chunk at a
                # time through the attention cache. The experts come from the
                # forward itself - every chunk admits the ones its characters
                # ask for, and the cache carries the context between them.
                while len(losses) < chunks and not r.done():
                    l = r.step(learn=False)
                    if l is None:
                        break
                    losses.append(float(l))
                del r
            if losses:
                out[name] = float(np.mean(losses))
                every.extend(losses)
        a = np.asarray(every)
        usable = (len(a) // per) * per
        if usable >= 2 * per:
            wm = a[:usable].reshape(-1, per).mean(1)
            out["stderr"] = float(wm.std(ddof=1) / np.sqrt(len(wm)))
        else:
            out["stderr"] = float(a.std() / np.sqrt(max(1, len(a))))
        self.model.train()
        return out


class Evaluator:
    """
    Held-out loss per source, read the same way training reads.

    IT REPORTS HOW UNCERTAIN IT IS, and the size of the evaluation is what
    sets that. Repeated draws on unchanged weights:

        chunks   characters   sd across draws
             6       12,288            0.0275
            24       49,152            0.0170
            96      196,608            0.0093
           240      491,520            0.0064

    A difference smaller than the sd at the size used is not a result, it is
    the instrument. `stderr` in the returned dict estimates the same quantity
    from a single evaluation.

    It is not the whole story. This is the spread of the SCORE with the weights
    held fixed. Training itself is non-deterministic on CUDA - the same recipe
    run twice lands about 0.014 apart - so the honest threshold for a real
    difference is around 0.03, well above anything printed here.
    """

    def __init__(self, model, names, chunk, context, device):
        self.model, self.chunk, self.context = model, chunk, context
        self.device = device
        self.names = names
        self.data = [np.memmap(os.path.join(n, "val.bin"), dtype=np.uint16,
                               mode="r") for n in names]

    @torch.no_grad()
    def run(self, chunks=240):
        self.model.eval()
        out, every = {}, []
        for name, d in zip(self.names, self.data):
            r = Reader(self.model, d, name, self.chunk, self.context,
                       self.device, np.random.default_rng(7))
            losses = [float(r.step(learn=False)) for _ in range(chunks)]
            out[name] = float(np.mean(losses))
            every.extend(losses)
            del r
        self.model.train()
        # Chunks inside one window are correlated, so the sample here is the
        # WINDOW, not the chunk: average each window first, then take the
        # spread of those. Per-chunk spread overstates the error by about two
        # and a half times, because most of a chunk's variance is which passage
        # it landed on and that averages out within the window.
        per_window = max(1, self.context // self.chunk)
        a = np.asarray(every)
        usable = (len(a) // per_window) * per_window
        if usable >= 2 * per_window:
            wm = a[:usable].reshape(-1, per_window).mean(1)
            out["stderr"] = float(wm.std(ddof=1) / np.sqrt(len(wm)))
        else:
            out["stderr"] = float(a.std() / np.sqrt(max(1, len(a))))
        return out


def ramp_context(step, total, start, end, warm=0.35, granularity=256):
    """
    Grow the window over the run instead of jumping to it.

    The model attends with RoPE, which has no learned parameters, so a longer
    context costs nothing to represent - but the model has only been trained at
    the lengths it has seen, and loss degrades well beyond them. Extending
    gradually keeps it near a length it can already handle while it learns the
    next one.

    GRADUALLY MEANS GRADUALLY. Rounding the window to powers of two turns this
    into a staircase, and a window that doubles in one step is a shock the run
    may not recover from - loss and gradient norm both jump, and the clip stops
    being able to hold it. Nothing here needs a power of two: the cache and the
    rotary tables take any length, so `granularity` keeps each change small.
    """
    if end <= start:
        return end
    f = min(1.0, step / max(total * warm, 1.0))
    ctx = start * (end / start) ** f
    return int(max(start, min(end, round(ctx / granularity) * granularity)))
