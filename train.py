#!/usr/bin/env python3
"""
Training mini-AGI.

    read      point it at files and let it read them, continually. This is the
              one that matters, and the one the run uses.

    stream    the same mechanism over a packed corpus: batch 1, a KV cache,
              one chunk of characters at a time. Peak memory is set by
              --chunk and nothing else, so --context is nearly free to extend.

    probe     what depth the model chooses, per character.

Everything here runs at batch 1 behind a cache. The fixed-window regime that
preceded it - several windows per step, one autograd graph over each whole
window - cost about three times as much at an eighth of the reach: 6857 MB at
4k characters against 3090 MB at 32k, because the retained activation graph
was bounded by the window rather than by the chunk.
"""

import os
import re
import sys
import json
import collections
import math
import time
import argparse
from dataclasses import asdict

# BEFORE torch, because the allocator reads this once at CUDA init and ignores
# it afterwards. Without it PyTorch's caching allocator keeps its free blocks in
# fixed segments, and a run that alternates between a few large short-lived
# tensors - which is exactly what a checkpointed expert dispatch does, three
# batched matmuls recomputed in the backward - strands memory it cannot hand
# back. Measured on an 8,192 window: the backward asked for 384 MiB with 394
# MiB free and 861 MiB reserved but unallocated. There was plenty of memory;
# there was no contiguous piece of it. Set it here rather than in a shell so it
# is a property of the program, not of how it happened to be launched.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn

from minagi.recur import RecurConfig, RecurCoder, load_recur
from minagi.pool import PooledMLP, AutoGrow
from minagi.stream import StreamSet, Evaluator, detach_caches, ramp_context
from minagi.plasticity import Plasticity
from minagi.optim import GradSNR
from minagi import store as weights_store


def lr_at(step, total, base, warmup, floor_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))




def _resync_opt(opt, model, args):
    """Drop parameters the model no longer has, adopt the ones it gained.

    Growth appends experts and replaces the gate tensor; pruning removes them
    and replaces it again. Either way the optimiser is left holding tensors the
    model does not have - Adam would keep stepping them and keep their moments
    alive, a leak on a pool that grows and shrinks all run.
    """
    live = {id(p) for p in model.parameters()}
    for g in opt.param_groups:
        for p_ in g["params"]:
            if id(p_) not in live:
                opt.state.pop(p_, None)
        g["params"] = [p for p in g["params"] if id(p) in live]
    known = {id(p) for g in opt.param_groups for p in g["params"]}
    fresh = [p for p in model.parameters() if id(p) not in known]
    mats = [p for p in fresh if p.dim() >= 2]
    vecs = [p for p in fresh if p.dim() < 2]
    # a group added mid-run needs the anchor the schedule scales from, or it
    # would sit at whatever rate it was born with while everything else decays
    base = args.lr
    if mats:
        opt.add_param_group({"params": mats, "name": "pool",
                             "weight_decay": args.wd,
                             "lr": base, "base_lr": base})
    if vecs:
        opt.add_param_group({"params": vecs, "name": "pool",
                             "weight_decay": 0.0,
                             "lr": base, "base_lr": base})


def _auto_device():
    """The accelerator to use when the caller did not name one.

    CUDA first: it is the only backend that turns on fused AdamW, bf16
    autocast and the VRAM telemetry, so a box that has it should train there.
    MPS next: Apple Silicon can train, but those three CUDA-only paths stay
    off, so it is a deliberate second choice rather than an accidental one.
    CPU last: every machine has one, which is why it was the old fallback and
    why it stays the safety net.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cmd_stream(args):
    """
    Train the way the model runs: batch 1, a KV cache, one chunk at a time.

    The old loop built one autograd graph over the whole window. That cost
    about four times what reading the same window costs - 6857 MB against 1704
    - and the difference was entirely activations retained across 26 block
    applications. So the model could be served at a context it could not be
    trained at, and training kept hitting OOM on a card with room to spare.

    Here the graph is bounded by --chunk and the reach is bounded by the cache,
    which is linear and cheap. Batch 1 gives up gradient averaging; what it
    buys is roughly an order of magnitude of context on the same card.
    """

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sources = []
    for part in args.mix.split(","):
        name, _, w = part.partition(":")
        sources.append((name.strip(), float(w) if w else 1.0))

    wdir = args.weights_dir
    man = None
    if os.path.exists(os.path.join(wdir, "manifest.json")):
        with open(os.path.join(wdir, "manifest.json")) as f:
            man = json.load(f)

    cfgd = dict(man["cfg"]) if man and man.get("cfg") else {}
    cfgd.update(vocab_size=265, block=args.context, use_pool=True)
    cfgd["pool_max"] = args.pool_max
    cfg = RecurConfig(**{k: v for k, v in cfgd.items()
                         if k in RecurConfig.__dataclass_fields__})
    # The capacity bound comes from config.yaml rather than from the
    # checkpoint, because it is a property of the machine the model is running
    # on - how much VRAM the dispatch may use - not of the model. A directory
    # written before this existed carries no value for it and would otherwise
    # get the dataclass default, which is right but silent; taking it from the
    # config every load means the number in the file is the number in effect.
    try:
        from minagi.config import load as _load_cfg, get as _get_cfg
        cfg.pool_capacity_factor = float(
            _get_cfg(_load_cfg(), "pool.capacity_factor",
                     cfg.pool_capacity_factor))
    except Exception:
        pass
    model = RecurCoder(cfg).to(device)

    if man and int(man["n_experts"]) != model.pool.n_experts():
        have, want = model.pool.n_experts(), int(man["n_experts"])
        if want > have:
            model.pool.add_experts(want - have, device=device)
        print(f"  pool resized {have} -> {want} to match the directory")

    print(f"params {model.n_params()/1e6:.2f}M | {model.pool.n_experts()} "
          f"experts | block-applications {cfg.n_layer_effective}")
    print(f"batch 1 | chunk {args.chunk} | context up to {cfg.block:,}")
    print(f"  the graph spans {args.chunk} characters; the cache spans the "
          f"context. only the first costs activations.")

    pool_ids = set()
    sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
    pool_ids = {id(q) for e in model.pool.experts for q in e.parameters()}
    pool_ids |= {id(st.router.weight) for st in sites}
    pool_ids |= {id(st.depth_emb) for st in sites}
    pool_ids.add(id(model.pool.gate))
    trunk = [q for q in model.parameters() if id(q) not in pool_ids]
    pool = [q for q in model.parameters() if id(q) in pool_ids]
    opt = torch.optim.AdamW(
        [{"params": [q for q in trunk if q.dim() >= 2], "name": "trunk",
          "weight_decay": args.wd},
         {"params": [q for q in trunk if q.dim() < 2], "name": "trunk",
          "weight_decay": 0.0},
         {"params": [q for q in pool if q.dim() >= 2], "name": "pool",
          "weight_decay": args.wd},
         {"params": [q for q in pool if q.dim() < 2], "name": "pool",
          "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))
    print(f"trunk learns at {args.trunk_lr_mult:g}x the pool's rate "
          f"({sum(q.numel() for q in trunk)/1e6:.2f}M trunk, "
          f"{sum(q.numel() for q in pool)/1e6:.2f}M pool)")

    start_step = 0
    if man:
        weights_store.load(model, wdir, opt=opt, device=device, verbose=True)
        start_step = int(man.get("step", -1)) + 1
        print(f"resuming from {wdir} at step {start_step}")

    grower = None
    if args.grow_k > 0:
        # args.pool_max, not cfg.pool_max - the latter is the router width
        grower = AutoGrow(grow_k=args.grow_k, max_experts=args.pool_max,
                          mem_frac_max=args.grow_mem_frac)
        model.pool.born.fill_(float(start_step))
        model.pool.gate_seen.copy_(model.pool.gate.data.abs())
        print(f"auto-growth armed: {model.pool.n_experts()} experts, ceiling "
              f"{args.pool_max}, VRAM brake at {100*args.grow_mem_frac:.0f}%")

    streams = StreamSet(model, sources, args.chunk, args.ctx_start, device,
                        rng, replay=args.replay,
                        replay_burst=args.replay_burst)
    evaluator = Evaluator(model, [n for n, _ in sources], args.chunk,
                          args.ctx_start, device)
    tot = sum(w for _, w in sources)
    for name, w in sources:
        n_ch = len(np.memmap(os.path.join(name, "train.bin"),
                             dtype=np.uint16, mode="r"))
        print(f"  {name:<22}{100*w/tot:>4.0f}%   {n_ch/1e6:>7.1f}M characters")
    if args.ctx_start < args.context:
        print(f"context ramps {args.ctx_start:,} -> {args.context:,} over the "
              f"first third of the run")

    os.makedirs(args.out, exist_ok=True)

    # A run keeps its own history rather than leaving it in a log. A log is one
    # shell redirect away from being overwritten by the next run, and two
    # processes writing one log here once produced a number that looked like a
    # model bug and was not. Appended a line at a time, so an interrupted run
    # keeps everything up to the interruption.
    hist_path = os.path.join(args.out, "history.jsonl")
    hist = open(hist_path, "a", buffering=1)

    def record(kind, step=None, **kw):
        hist.write(json.dumps({"kind": kind, "step": step, **kw}) + "\n")

    record("start", weights_dir=wdir, context=cfg.block, chunk=args.chunk,
           accum=args.accum, lr=args.lr, trunk_lr_mult=args.trunk_lr_mult,
           steps=args.steps, start_step=start_step,
           experts=model.pool.n_experts(), params=model.n_params(),
           mix=[{"name": n, "weight": w} for n, w in sources])

    # the directory's own score, not infinity - resuming must not let a worse
    # state call itself the best merely because this process just started
    best = weights_store.best_val(wdir)
    if best < float("inf"):
        print(f"best on disk: val {best:.4f} - weights/ advances only past it")
    reverts = 0
    t0, seen = time.time(), 0
    model.train()

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr * (args.trunk_lr_mult if g["name"] == "trunk" else 1.0)
        ctx_now = ramp_context(step - start_step, max(1, args.steps - start_step),
                               args.ctx_start, args.context)
        if ctx_now != streams.context:
            streams.set_context(ctx_now)
            evaluator.context = ctx_now
            print(f"  context -> {ctx_now:,} characters", flush=True)
        # Accumulate over consecutive chunks before stepping. Each chunk's
        # graph is freed by its own backward, so this costs no memory beyond
        # one chunk - but the gradient the optimiser sees averages
        # accum x chunk characters instead of chunk. That is the whole gap
        # between this and the batch trainer: at chunk 512 a step saw 512
        # characters where a batch step saw 32768, and with accumulation the
        # two can be made identical without giving up the memory profile.
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            loss, src = streams.step(learn=True, aux_weight=cfg.pool_aux)
            (loss / args.accum).backward()
            detach_caches(streams.active.caches)
            seen += args.chunk
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % args.log_every == 0:
            el = max(time.time() - t0, 1e-9)
            vram = (f" vram {torch.cuda.max_memory_allocated()/1e6:.0f}MB"
                    if device.type == "cuda" else "")
            record("step", step=step, loss=float(loss), lr=lr,
                   grad_norm=float(gn), chars=seen, chars_per_s=seen / el,
                   context=streams.context, experts=model.pool.n_experts(),
                   vram_mb=(torch.cuda.max_memory_allocated() / 1e6
                            if device.type == "cuda" else None))
            rp = (f" replay {streams.replays}" if args.replay > 0 else "")
            print(f"step {step:>6}/{args.steps} loss {float(loss):.4f} "
                  f"lr {lr:.2e} gn {float(gn):.2f} ctx {streams.context//1024}k "
                  f"{seen/el/1e3:.1f}k char/s{vram}{rp}", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            v = evaluator.run(args.eval_chunks)
            se = v.pop("stderr", 0.0)
            val = float(np.mean(list(v.values())))
            record("val", step=step, val=val, stderr=se, ppl=math.exp(val),
                   per_domain=dict(v), experts=model.pool.n_experts())
            print(f"  val {val:.4f} +/-{se:.4f} ppl {math.exp(val):.2f}   "
                  + "  ".join(f"{k.replace('data_','').replace('_char','')} "
                              f"{x:.3f}" for k, x in v.items()), flush=True)
            if grower is not None:
                gone = model.pool.prune(step, survival=args.grow_min_age)
                if gone:
                    _resync_opt(opt, model, args)
                    record("pruned", step=step, n=gone,
                           to=model.pool.n_experts())
                    print(f"  pruned {gone} -> {model.pool.n_experts()} experts",
                          flush=True)
                rec = grower.step(val, model.pool, step)
                if rec["grew"]:
                    _resync_opt(opt, model, args)
                    record("grew", step=step, n=rec["grew"], to=rec["experts"],
                           reason=rec.get("reason"))
                    print(f"  GREW +{rec['grew']} -> {rec['experts']} experts "
                          f"| {rec['reason']}", flush=True)
                elif rec.get("reason"):
                    record("held", step=step, reason=rec["reason"],
                           experts=model.pool.n_experts())
                    print(f"  [pool] {rec['reason']}", flush=True)
            # The directory holds the best state the model has reached, so it
            # is written when a run improves on it and left alone otherwise.
            # Keeping a second copy aside as a snapshot would double 236 MB
            # that grows with the pool, to hold a copy of what is already here.
            if val < best:
                best = val
                weights_store.save(model, wdir, step=step, val=val, opt=opt,
                                   cfg=asdict(cfg), verbose=True)
                record("saved", step=step, val=val)
                print(f"  improved - weights/ advanced to val {val:.4f}",
                      flush=True)
            elif args.revert_factor > 0 and val > best * args.revert_factor:
                # A run can diverge: extending the context too fast once took
                # validation from 0.7375 to 5.4490 in a thousand steps. Without
                # this the live directory would carry that away with it.
                print(f"  DIVERGED: val {val:.4f} is {val/best:.1f}x the best "
                      f"{best:.4f} - reloading weights/ and halving the "
                      f"learning rate", flush=True)
                with open(os.path.join(wdir, "manifest.json")) as _f:
                    man_b = json.load(_f)
                n_back = int(man_b["n_experts"])
                if n_back != model.pool.n_experts():
                    have = model.pool.n_experts()
                    if n_back > have:
                        model.pool.add_experts(n_back - have, device=device)
                    else:
                        model.pool.experts = nn.ModuleList(
                            list(model.pool.experts)[:n_back])
                        model.pool.gate = nn.Parameter(
                            model.pool.gate.data[:n_back].clone())
                        for b in ("use", "age", "born", "gate_seen"):
                            setattr(model.pool, b,
                                    getattr(model.pool, b)[:n_back].clone())
                        model.pool.invalidate()
                    _resync_opt(opt, model, args)
                weights_store.load(model, wdir, opt=opt, device=device)
                args.lr *= 0.5
                streams.set_context(max(args.ctx_start, streams.context // 2))
                evaluator.context = streams.context
                reverts += 1
                print(f"  back to val {man_b['val']:.4f} at "
                      f"{model.pool.n_experts()} experts, lr now "
                      f"{args.lr:.2e}, context back to {streams.context:,}",
                      flush=True)
                if reverts >= args.max_reverts:
                    print(f"  {reverts} reverts - stopping rather than "
                          f"thrashing", flush=True)
                    break
            model.train()

    record("done", minutes=(time.time() - t0) / 60, best=best,
           experts=model.pool.n_experts())
    hist.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f}m best val {best:.4f}")
    print(f"history: {hist_path}")
    return 0



class _Tracer:
    """
    Record what the real training forward did, for the first few chunks.

    Not a reconstruction: this sits inside the training loop, around the same
    forward whose loss is about to be backpropagated, with the same KV cache,
    the same precision, the same chunk and the same working set. Whatever it
    writes down is what actually happened.

    The capture context is exited before backward on purpose. Gradient
    checkpointing replays the forward during backward, and a capture still
    open would record every call site twice.
    """

    def __init__(self, path, chunks, swap_chunks=160):
        self.path, self.want = path, chunks
        self.want_swaps = max(swap_chunks, chunks)
        self.ids, self.wts, self.meta, self.text = [], [], [], []
        self.swaps = []

    def full(self):
        """Routes are heavy, so only the first few chunks keep them."""
        return len(self.ids) >= self.want

    def done(self):
        """Swap counts are cheap and run on well past the routes."""
        return self.full() and len(self.swaps) >= self.want_swaps

    def add_swap(self, lane, moved, first, demand_moved=None):
        """
        One line per working-set decision, whether or not routes are traced.

        `first` marks the chunk that opens a lane's window - the only place
        the subject actually changes. If swaps have a reason, they belong
        here and nowhere else; if they are on a timer, this column will make
        no difference to the count.
        """
        self.swaps.append({"subject": lane.name, "moved": int(moved),
                           "first_of_window": bool(first)})

    def add(self, got, pool, lane, moved, step, loss, seen, ids):
        import numpy as _np
        if self.ids and ids.shape[1] != self.text[0].shape[0]:
            return                      # a short tail chunk would not stack
        self.ids.append(_np.stack([g[0].numpy() for g in got]).astype("int16"))
        self.wts.append(_np.stack([g[1].numpy() for g in got]).astype("float16"))
        self.text.append(ids[0].detach().cpu().numpy().astype("int16"))
        self.meta.append({
            "subject": lane.name, "moved": int(moved), "step": int(step),
            "loss": float(loss), "seen": int(seen),
            "resident": [int(e) for e in pool.slots if e >= 0],
            "experts": int(pool.n_experts()),
        })

    written = False

    def write(self, cfg, precision, chunk):
        import json as _json
        import numpy as _np
        self.written = True
        _np.savez_compressed(
            self.path,
            ids=_np.stack(self.ids), weights=_np.stack(self.wts),
            text=_np.stack(self.text),
            meta=_json.dumps({"chunks": self.meta, "swaps": self.swaps,
                              "chunk": chunk,
                              "top_k": cfg.pool_top_k,
                              "max_steps": cfg.max_steps,
                              "n_recur": cfg.n_recur, "n_coda": cfg.n_coda,
                              "n_prelude": cfg.n_prelude,
                              "context": cfg.block,
                              "precision": precision}))
        print(f"    routing trace -> {self.path} "
              f"({len(self.ids)} chunks of real training)", flush=True)


class _Lane:
    """
    One subject, sampled a window at a time.

    Each visit opens a random file and a random offset inside it, then reads
    one context window forward from there. The window is contiguous on
    purpose: a lone chunk carries no history, so attention would have nothing
    to reach back over and the context window would be decoration.
    """

    def __init__(self, name, paths, rng):
        self.name, self.paths, self.rng = name, paths, rng
        self.r = None
        self.visits = 0

    def open(self, model, chunk, block, device, span=None):
        """
        Open a file for this visit.

        `block` is the context window; `span` is how much will be read before
        the reader moves to another subject, which may be several windows. The
        offset has to leave room for the whole visit, not just the first
        window, or the file runs out part way through and the visit is cut
        short.
        """
        from minagi.stream import FileReader
        from minagi.ingest import as_stream
        span = max(span or block, block)
        for _ in range(8):                       # a few tries for short files
            path = self.paths[int(self.rng.integers(0, len(self.paths)))]
            try:
                data = as_stream(path)
            except OSError:
                # The corpus can change under a run that lasts days. The
                # self-knowledge lane in particular is REGENERATED from the
                # source, which deletes and rewrites every file in it, and the
                # path list was taken at startup. A vanished file is a reason
                # to pick another one, not to end a week of reading.
                continue
            if len(data) < 8:
                continue
            r = FileReader(model, data, path, chunk, block, device)
            if len(data) > span + 1:
                r.pos = int(self.rng.integers(0, len(data) - span - 1))
            self.r = r
            self.visits += 1
            return r
        return None

    def rest(self, model):
        """
        Drop the KV cache between visits.

        A visit is exactly one context window, which is where FileReader
        throws its cache away anyway - so this costs nothing, and it is what
        keeps a single window of keys and values resident rather than one per
        lane. At 26 block-applications a window is 1.7 GB, so four lanes
        holding their own would not fit on the card at all.
        """
        if self.r is not None:
            self.r.seen = 0
            self.r = None


def _lanes(files, seed, roots=None, resume=0):
    """
    One lane per subject, where a subject is a TOP-LEVEL folder of the corpus.

    Not the immediate parent: a corpus of any size grows subdirectories - a
    hundred and fifty thousand articles cannot sit in one directory - and
    grouping by the parent turns every one of those into its own subject. It
    also matches what a reader means by their own files: pointing at ~/notes
    with folders inside should give one subject per folder, not per nested
    directory.
    """
    roots = [os.path.abspath(r) for r in (roots or [])]

    def subject(path):
        p = os.path.abspath(path)
        for r in roots:
            if p.startswith(r + os.sep):
                rel = os.path.relpath(p, r).split(os.sep)
                # the first component under the root, or the root itself for
                # a file lying directly in it
                return rel[0] if len(rel) > 1 else os.path.basename(r)
        return os.path.basename(os.path.dirname(p)) or p

    by = {}
    for f in files:
        by.setdefault(subject(f), []).append(f)
    # `resume` is how many characters the model has already read, and leaving
    # it out was the single most expensive bug in this project.
    #
    # A lane picks a random file and a random offset inside it, from its own
    # generator. Seeded by `seed + j` alone, that generator is rebuilt
    # identically at every start, so every session replayed the SAME files at
    # the SAME offsets in the SAME order, from the beginning. Measured over 43
    # sessions: the median one read 3.2M characters, the longest 66M, and the
    # counter reported 500M - but the UNIQUE text was about the length of the
    # longest session, because the short ones were all re-reading the opening
    # of the same sequence.
    #
    # It showed up as a restart spike nobody could place. Train loss fell hard
    # (0.675 -> 0.446 at the last one) because the model was being handed text
    # it had already memorised, and held-out rose at the same moment (0.925 ->
    # 1.105) because it was being pushed to overfit that slice further. The
    # learning rate was identical across the seam, nothing was pruned, and
    # nothing was added - which is what made it look like lost state rather
    # than repeated data.
    #
    # Mixing the read count into the seed means a restart continues into text
    # the model has not seen instead of starting the same tape again. It stays
    # reproducible from a given checkpoint, and needs nothing persisted: the
    # character count is already in the manifest.
    return [_Lane(k, sorted(by[k]), np.random.default_rng([seed, j, resume]))
            for j, k in enumerate(sorted(by))]


def _truncate_history(path, chars):
    """Drop history rows past `chars` - they belong to a branch that was
    abandoned, and every reader of this file assumes it moves forward."""
    if not path or not os.path.exists(path) or chars <= 0:
        return
    try:
        with open(path) as f:
            rows = f.read().splitlines()
        keep, dropped = [], 0
        for line in rows:
            try:
                if json.loads(line).get("chars", 0) > chars:
                    dropped += 1
                    continue
            except json.JSONDecodeError:
                pass                       # keep anything unparseable
            keep.append(line)
        if dropped:
            with open(path, "w") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))
            print(f"  history: dropped {dropped} row(s) past "
                  f"{chars/1e6:.1f}M - superseded by this resume", flush=True)
    except OSError as e:
        print(f"  (history not trimmed: {e})", flush=True)


def cmd_read(args):
    """
    Point the model at files and let it read them.

    This is the whole interface for learning on your own data. The alphabet is
    the 256 byte values, so a file needs no preparation - no tokenizer to fit,
    no corpus to build, no step to re-run when the data changes. Give it a
    directory and it reads what is there, in order, once or several times, and
    is different afterwards.

        python3 train.py read ~/notes
        python3 train.py read src/ docs/ --passes 3

    It is the same path `stream` uses - same chunking, same cache, same
    gradient step. What differs is where the characters come from, and that a
    file is read from its beginning to its end rather than sampled at random,
    because a document has an order and the model should see it.
    """
    # Retired spellings, checked before anything loads. `--dwell` is a strict
    # prefix of `--dwell-chars` and argparse matches unambiguous prefixes, so
    # it would be accepted silently and read as characters - an alias that
    # keeps the name and changes the unit is the exact failure this change
    # exists to remove.
    if getattr(args, "dwell_retired", None) is not None:
        raise SystemExit(
            "--dwell counted SEGMENTS and is gone. Use --dwell-chars, in "
            "characters: the old default of 4 is --dwell-chars 2048.")

    # EVERY CADENCE IS CONFIGURED IN CHARACTERS AND USED IN STEPS.
    #
    # A step is `--chunk` characters, so anything counted in steps silently
    # changes meaning the moment the chunk changes - `growth.every: 400` was
    # 204,800 characters at chunk 512 and would have been 1.6 million at 4,096
    # without a line of the config moving. Characters are what the model
    # actually reads; steps are an implementation detail of how often we stop
    # to update. Convert once, here, where the chunk is known.
    def _in_steps(chars, least=1):
        return max(least, int(round(int(chars) / max(1, args.chunk))))

    grow_every_steps = _in_steps(args.grow_every)
    survival_steps = _in_steps(args.prune_survival)
    ctx_every_steps = _in_steps(args.context_every)
    ctx_grow_every_steps = _in_steps(args.context_grow_every)
    from minagi.ingest import collect, summarise
    from minagi.stream import FolderEvaluator
    from minagi.precision import set_compute_dtype

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    set_compute_dtype(args.precision)

    files = collect(args.paths)
    if not files:
        print(f"no readable text under {', '.join(args.paths)}", file=sys.stderr)
        return 1
    info = summarise(files)
    print(f"{info['files']} files, {info['characters']/1e6:.2f}M characters"
          + (f", {args.passes} passes" if args.passes > 1 else ""))

    wdir = args.weights_dir
    # No model here yet is a model that has not been made yet, not an error.
    # Everything about its shape is in config.yaml, so there is nothing to ask.
    #
    # The test is core.npz, NOT whether the directory exists. The repository
    # ships weights/manifest.json - it records what the released model is - so
    # weights/ is already there on a fresh clone, and testing the directory
    # sent every new reader straight into a missing core.npz.
    if not os.path.exists(os.path.join(wdir, "core.npz")):
        from minagi.create import create
        leftovers = ([f for f in os.listdir(wdir) if f != "manifest.json"]
                     if os.path.isdir(wdir) else [])
        if leftovers:
            # Something is in there, but not a model. Wiping it is not this
            # function's decision to make.
            raise SystemExit(
                f"{wdir} has no core.npz but is not empty: "
                f"{', '.join(sorted(leftovers)[:6])}. That is a partial or "
                f"foreign weights directory - move it aside, or point "
                f"--weights-dir somewhere else, and it will be created.")
        if os.path.isdir(wdir):
            print(f"{wdir} holds only a manifest - creating a fresh model "
                  f"from config.yaml (the manifest describes the released "
                  f"model, not this one, and is replaced)")
        else:
            print(f"{wdir} does not exist - creating a fresh model from "
                  f"config.yaml")
        create(wdir, force=True)
        print()
    model, cfg, pool, man = build_paged(wdir, device, args.resident,
                                        args.ram_capacity, args.context)
    # the best held-out this run has seen, for the optional notifier below.
    # Reads the manifest, so it cannot move above build_paged.
    best_val = float(man.get("val") or float("inf"))
    print(f"  {pool.n_experts()} experts on disk, {pool.n_resident()} resident, "
          f"{pool.vram_params()/1e6:.1f}M in VRAM of "
          f"{pool.n_params()/1e6:.1f}M total")
    print(f"  computing in {args.precision}; weights stay fp32 everywhere, "
          f"Adam's moments store as bf16")
    # dwell is configured in characters and counted in segments, one per
    # working-set decision, which is one per chunk
    pool.margin = args.margin
    pool.dwell = max(1, int(args.dwell) // max(1, args.chunk))
    print(f"  experts move only on demand: a candidate must be wanted "
          f"{args.margin:.0%} more than the resident it displaces, and a "
          f"newcomer holds its slot for {args.dwell:,} characters ({pool.dwell} chunks)")

    if args.no_pool_checkpoint:
        n_off = 0
        for _s in model.modules():
            if isinstance(_s, PooledMLP):
                _s.grad_checkpoint = False
                n_off += 1
        print(f"  pool activations kept, not recomputed ({n_off} call sites) "
              f"- faster, and it needs the memory depth was using")
    trunk, pool_ps = _split_trunk_pool(model)

    tg = {"params": trunk, "name": "trunk", "weight_decay": args.wd,
          "lr": args.lr * args.trunk_lr_mult,
          "base_lr": args.lr * args.trunk_lr_mult}
    pg = {"params": pool_ps, "name": "pool", "weight_decay": args.wd,
          "lr": args.lr, "base_lr": args.lr}
    opt = torch.optim.AdamW([tg, pg], lr=args.lr, betas=(0.9, 0.95),
                            fused=(device.type == "cuda"))
    snr = GradSNR()
    print(f"  trunk learns at {args.trunk_lr_mult:g}x the pool's rate "
          f"({sum(q.numel() for q in trunk)/1e6:.1f}M trunk, "
          f"{sum(q.numel() for q in pool_ps)/1e6:.1f}M pool)")

    # Scored at the window the model is actually reading at. A fixed
    # reference window would be steadier, but it would stop describing the
    # model as it grows; and because the window moves one character at a time,
    # two consecutive scores are effectively at the same length anyway. The
    # sample log states the window with every entry so no comparison is made
    # blind.
    # held-out is configured in CHARACTERS per domain and read in chunks
    eval_steps = max(1, int(args.eval_chunks) // max(1, args.chunk))
    sample_eval_steps = max(1, int(args.sample_eval_chunks)
                            // max(1, args.chunk))
    ev = (FolderEvaluator(model, args.held_out, args.chunk, cfg.block, device,
                          segment_chunks=max(1, args.segment_chunks
                                             // max(1, args.chunk)))
          if args.held_out and os.path.isdir(args.held_out) else None)
    before = None
    if ev is not None:
        v = ev.run(eval_steps); se = v.pop("stderr", 0.0)
        before = float(np.mean(list(v.values())))
        print(f"  before: {before:.4f} +/-{se:.4f} on {args.held_out}")

    pool.attach_optimiser(opt)
    # ADAM'S MOMENTS. They are written to optim.npz at every checkpoint and
    # were never read back here: this path builds a fresh optimiser and only
    # ever SAVED it. So every restart threw away 108.9M moment values, 99.9% of
    # them non-zero, and Adam re-estimated its second moments from a single
    # gradient - which is a large, badly scaled step in a direction inferred
    # from one batch, decaying over a few hundred steps. That is the held-out
    # jump after a restart that then settles, on a model whose whole point is
    # that it never stops training.
    #
    # The tell was in the file: Adam's step counter t read 2,381 at step
    # 458,000. It counts optimiser steps since the last restart, not since the
    # run began.
    #
    # The pool's w1/w2/w3 moments are slot-indexed and belong to whichever
    # experts were resident when they were written, but restoring them is safe:
    # swap_to overwrites a slot's moments from the expert's own file the moment
    # it is paged in, and zeroes them if the file has none. `pool.gate` is
    # per-expert and full width, so it restores exactly; a pool that changed
    # size is caught by the shape guard in _load_optim.
    try:
        weights_store._load_optim(opt, model, wdir)
        t_seen = max((float(st.get("step", 0))
                      for st in opt.state.values() if "step" in st), default=0)
        print(f"  optimiser moments restored ({t_seen:,.0f} steps of history)")
    except Exception as e:                     # never let this stop a run
        print(f"  [note] optimiser moments not restored: {e}")
    grower = AutoGrow(grow_k=args.grow_k, max_experts=10_000_000,
                      mem_frac_max=args.grow_mem_frac,
                      dying_frac_max=args.grow_dying_frac,
                      max_in_flight=args.max_in_flight,
                      keep_ratio_min=args.grow_keep_ratio,
                      max_disk_gb=args.max_disk_gb,
                      birth_gate=args.birth_gate,
                      recent_mult=args.recent_mult) \
        if args.grow_k else None
    # The checkpoint carries whatever depth policy it was trained under, but
    # this is a knob about how to spend compute now, not a property of the
    # weights - so config.yaml wins over the manifest every time.
    cfg.train_steps_mean = float(args.train_steps_mean)
    cfg.min_steps = max(1, min(int(args.min_steps), cfg.max_steps))
    if args.bptt_window:
        cfg.bptt_window = int(args.bptt_window)
    if args.ponder_beta:
        cfg.ponder_beta = float(args.ponder_beta)
    if args.halt_prior:
        cfg.halt_prior = float(args.halt_prior)
    if args.halt_thresh:
        cfg.halt_thresh = float(args.halt_thresh)
    # The settings above are counted in ROWS, and the CHECKPOINT decides how
    # many rows a step is - n_recur + n_coda. So a config written for one
    # architecture is silently wrong on another, and the failure is not always
    # loud: bptt_window 16 against max_steps 6 makes the detach condition
    # n < -10, which never fires, so NOTHING is detached and the graph holds
    # every block instead of the last few. That is how a config meant for a
    # 24-row model put a 6-row model out of memory.
    warn = []
    if cfg.bptt_window > cfg.max_steps:
        warn.append(f"bptt_window {cfg.bptt_window} exceeds max_steps "
                    f"{cfg.max_steps}, so nothing would be detached and the "
                    f"whole chain would be held. Clamping to {cfg.max_steps}.")
        cfg.bptt_window = cfg.max_steps
    if cfg.train_steps_mean and cfg.train_steps_mean >= cfg.max_steps:
        warn.append(f"train_steps_mean {cfg.train_steps_mean:g} is at or above "
                    f"max_steps {cfg.max_steps}, so depth sampling saves "
                    f"nothing - every step runs the ceiling.")
    if cfg.halt_prior and 1.0 / cfg.halt_prior > cfg.max_steps * 1.5:
        warn.append(f"halt_prior {cfg.halt_prior:g} pulls toward "
                    f"{1/cfg.halt_prior:.1f} rows but the model only has "
                    f"{cfg.max_steps}.")
    if warn:
        print(f"  [config] this checkpoint has {cfg.n_recur + cfg.n_coda} "
              f"block(s) per row and {cfg.max_steps} rows; config.yaml looks "
              f"written for a different shape:", flush=True)
        for w in warn:
            print(f"    - {w}", flush=True)
    print(f"  gradient reaches back {cfg.bptt_window} of {cfg.max_steps} rows")
    print(f"  depth: up to {cfg.max_steps} rows, halting prior pulls toward "
          f"{1/max(cfg.halt_prior,1e-9):.1f} rows, inference stops at "
          f"{cfg.halt_thresh:g} cumulative")
    if cfg.train_steps_mean > 0:
        import torch as _t
        _n = (_t.poisson(_t.full((20000,), cfg.train_steps_mean)).int() + 1
              ).clamp(cfg.min_steps, cfg.max_steps).float().mean()
        print(f"  recurrence depth is SAMPLED while training: mean {_n:.2f} of "
              f"{cfg.max_steps} ({100*_n/cfg.max_steps:.0f}% of the recurrent "
              f"compute). Inference still runs all {cfg.max_steps} and lets "
              f"halting decide.")
    pool.dying_at = args.dying_at
    _PLOTS.update(on=bool(args.plots), since=args.plot_since,
                  log=args.sample_log, weights=args.weights_dir)
    if args.plots:
        print(f"  graphs redrawn with every sample entry -> "
              f"runs/training_progress.png"
              + (f" and runs/dashboard.png (the last "
                 f"{args.plot_since:g}M)" if args.plot_since else ""))
    pool.explore = args.explore
    # A new expert competes without the handicap until it is old enough to be
    # pruned, so the moment it is judged is the moment its fair turn ends.
    # One number governs both, which is the point: there is no second window
    # to reason about.
    pool.trial = survival_steps
    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    model.train()
    t0 = time.time()
    last_sample = time.time()
    # Steps and characters are cumulative over the model's whole life, not
    # per session. `born[i]` records the step an expert was grown and prune
    # asks whether `step - born[i]` has passed min_age, so restarting the
    # count at zero made that difference negative and quietly exempted every
    # expert grown in an earlier session from ever being pruned.
    step = int(man.get("step", 0) or 0)
    # The clock the expert trial is read against. Set only inside the loop it
    # is still 0 for the first forward of a session, and `now - born` is then
    # hugely NEGATIVE for every expert grown in an earlier session - so all of
    # them pass `< trial` at once and the whole pool gets the newborn's waiver,
    # routing on key fit alone with the gate ignored. One batch, but it is the
    # first batch after every restart.
    pool.now = step
    base_chars = int(man.get("read_chars", 0) or 0)
    seen = 0
    did = 0                                    # steps taken in THIS session
    losses = []
    if args.sample_every:
        # Append. A resumed run continues the same model, so wiping the file
        # would throw away the only record of how it got here; the banner is
        # what separates one session from the next.
        with open(args.sample_log, "a") as f:
            f.write(f"\n\n{'#' * 78}\n")
            f.write(f"# session started {time.strftime('%Y-%m-%d %H:%M')}  "
                    f"at step {step:,}, {base_chars/1e6:.1f}M of "
                    f"{info['characters']/1e6:,.0f}M characters read "
                    f"({100.0 * base_chars / max(info['characters'], 1):.2f}%), "
                    f"{pool.n_experts()} experts\n")
            f.write(f"{'#' * 78}\n")

    # The window starts where the model left it, not at the ceiling. RoPE
    # carries no learned parameters, so the same weights read at any length;
    # what a model cannot do is jump to a length it has never seen, which is
    # why this moves a character at a time.
    ctx_now = int(man.get("context_now") or args.context_start)
    # --context is a CEILING that binds on resume, not only on a fresh model.
    # It used to be clamped to cfg.block alone, and both cfg.block and
    # context_now come out of the checkpoint - so lowering model.context in
    # config.yaml changed nothing at all on a resumed run. Measured the hard
    # way: two runs at 16,384 and 8,192 in the config both read 32,768 from
    # the manifest and died with byte-identical out-of-memory errors.
    #
    # The window is the one setting that has to be lowerable from outside,
    # because it is what decides whether the model fits on this card, and the
    # card is not a property of the checkpoint.
    # ONE ceiling, used everywhere. cfg.block is how far the rotary tables
    # were built; --context is how far this card can afford. Clamping only at
    # startup was worse than not clamping at all: the window came down to 4,096
    # and the growth path below, which clamped to cfg.block alone, immediately
    # started walking it back toward 32,768 a character at a time. Measured -
    # context 4,151 -> 4,492 over two hours while VRAM went 7,271 -> 7,790 MB,
    # and it died on the way. That looks exactly like a leak from the outside,
    # which is what it was reported as.
    ctx_max = max(64, min(int(cfg.block), int(args.context)))
    ctx_now = max(64, min(ctx_now, ctx_max))
    if ctx_now < int(man.get("context_now") or 0):
        print(f"  window held at {ctx_now:,} by --context-end "
              f"(the checkpoint had grown to "
              f"{int(man.get('context_now')):,})")
    # loss by distance into the window: the chunk index IS the distance, so
    # this costs nothing to collect
    by_pos = collections.defaultdict(list)

    def context_gain():
        """
        How much better the model predicts deep into the window than early in
        it. Positive means it is using the distance it has - and would use
        more. Near zero means the far end of the window is being carried for
        nothing.
        """
        n_buckets = max(2, ctx_now // args.chunk)
        early = [l for p, ls in by_pos.items() if p < n_buckets * 0.5
                 for l in ls]
        late = [l for p, ls in by_pos.items() if p >= n_buckets * 0.75
                for l in ls]
        if len(early) < 8 or len(late) < 8:
            return None
        return float(np.mean(early) - np.mean(late))

    # A RESUME SUPERSEDES WHATEVER THE OLD BRANCH WROTE. The history log is
    # append-only and its character counter is not monotonic across a restart:
    # reading on from a checkpoint leaves rows describing a model that no
    # longer exists, sitting after the point being resumed from. Every reader
    # then has to cope - the dashboard drew a seam, window_use reached back to
    # the wrong row - and the file needs cleaning by hand every time an
    # experiment is abandoned.
    #
    # So drop them here, once, at the point the run knows where it is
    # resuming. Anything already past that mark describes a branch this run is
    # not on.
    _truncate_history(args.history, base_chars)

    # One lane per subject, rotated a window at a time.
    lanes = _lanes(files, args.shuffle_seed, args.paths,
                   resume=int(man.get("read_chars", 0) or 0))
    # How long a subject is read for before the next one. It is a whole
    # number of context windows - a window never spans two subjects, because
    # attention across the seam where chess becomes Python teaches nothing -
    # but a subject may take several windows in a row.
    #
    # The floor matters at small windows. Tied to the window alone, a 2,048
    # context would change subject every four chunks, and the first chunk
    # after a change picks its experts from the states of the subject just
    # left. At four chunks a visit that is a quarter of all reading.
    def visit_chunks(ctx):
        """How many chunks make up one visit to a file.

        `--passage` is in characters and says how far the reader walks through
        one file before moving to another subject. The window slides along it
        a chunk at a time, so a visit is passage/chunk steps - and it must be
        at least one window, or the window never fills and the model is asked
        to read with less context than it has.

        This used to be `min_visit_chunks`, in chunks, rounded up to whole
        context windows - so the number in the config was not the number in
        effect. At 16 with a 32,768 window it silently became 64.
        """
        least = max(1, -(-ctx // args.chunk))          # one full window
        want = max(1, -(-args.passage // args.chunk))
        return max(least, want)

    turn = visit_chunks(ctx_now)
    wants_more = False          # whether the evidence says to keep growing
    target = info["characters"] * max(1, args.passes)
    print(f"  {len(lanes)} subjects, reading {turn * args.chunk:,} characters "
          f"from one before moving to the next"
          + (f" ({turn * args.chunk // ctx_now} windows of {ctx_now:,})"
             if turn * args.chunk > ctx_now else "")
          + ": " + ", ".join(l.name for l in lanes))
    if ctx_now < ctx_max:
        print(f"  the window grows by {args.context_step} character(s) "
              f"whenever the model is still gaining more than "
              f"{args.context_gain_min} deep into it, up to {ctx_max:,}")
    if base_chars:
        print(f"  {base_chars/1e6:.1f}M characters read in earlier sessions")

    def log_history(val=None):
        """
        Append what the pool looks like now, so a run leaves a trail.

        The manifest only ever holds the latest state, which cannot answer
        whether an expert earned its gate early and coasted or climbed all
        along. One line per checkpoint, and a run of any length stays
        readable afterwards.
        """
        if not args.history:
            return
        try:
            t = pool.telemetry()
            row = {"t": time.time(), "step": step,
                   "chars": base_chars + seen, "val": val,
                   "experts": pool.n_experts(), "segments": t["segments"],
                   "gate": t["gate"], "use": t["use"], "admits": t["admits"],
                   "since": t["since"], "born": t["born"],
                   "last_seen": t.get("last_seen"), "uid": t.get("uid"),
                   "trial": getattr(model.pool, "trial", 0)}
            os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
            with open(args.history, "a") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as e:                  # telemetry must never stop a run
            print(f"    (history not written: {e})", flush=True)

    def checkpoint(val=None):
        # There is no reading position to record any more: windows are drawn
        # at random, so what carries forward is how much has been read, not
        # where the reader had got to.
        weights_store.save(model, wdir, step=step, val=val, opt=opt,
                           cfg=asdict(cfg),
                           extra={"read_chars": base_chars + seen,
                                  "read_nats": nats,
                                  "plasticity": plast.state(),
                                  "context_now": ctx_now})
        log_history(val)

    stop = False
    swapped = 0
    recent = collections.deque(maxlen=400)     # the training loss lately
    last_gap = None                            # held-out minus train, when known
    # How much information has been read, as against how many characters were
    # passed over: the loss IS the information content of what was read, so
    # integrating it says what the model actually had to account for. A
    # million characters of chess carries 0.744M nats and a million of
    # wikipedia 1.686M - the same reading, worth 2.27x as much.
    #
    # Recorded because it is the honest way to describe a run, and NOT used to
    # decide anything. Growth is judged on the network as it stands.
    # The learning rate is governed by held-out loss, not by a horizon.
    # There is no lr_origin any more and nothing counts characters: see
    # minagi/plasticity.py for the two rules and why they are symmetric.
    plast = Plasticity.restore((man or {}).get("plasticity"))
    if args.lr_reset:
        plast.scale = 1.0
        print("  plasticity RESET: full learning rate restored", flush=True)
    print(f"  {plast.describe()}", flush=True)
    nats = float(man.get("read_nats") or 0.0) if man else 0.0
    if not nats and base_chars:
        # A checkpoint written before this was tracked has the characters but
        # not the information. Starting the counter at zero would make the
        # ratio enormous and silently re-block growth for hours after every
        # upgrade, so seed it from what those characters were most likely
        # worth: the held-out score, which is the model's own estimate of nats
        # per character on text like the text it read.
        nats = base_chars * float(before or 1.2)
        print(f"  no information count in the checkpoint; seeding it at "
              f"{nats/1e6:.1f}M nats from {base_chars/1e6:.1f}M characters "
              f"at {float(before or 1.2):.3f} nats each", flush=True)
    grads = collections.deque(maxlen=400)      # and the gradient norms
    last_save = time.time()
    mark_t, mark_c = time.time(), 0            # for the reading rate
    tracer = (_Tracer(args.trace_routes, args.trace_chunks)
              if args.trace_routes else None)
    if tracer is not None:
        print(f"  tracing the routing of the first {args.trace_chunks} chunks "
              f"into {args.trace_routes}")

    # Ctrl-C finishes the chunk in flight, writes a checkpoint, and exits, so
    # a run can always be stopped without losing the trunk trained since the
    # last one. A second Ctrl-C is the usual immediate kill.
    asked_stop = []
    import signal
    def _on_int(sig, frm):
        if asked_stop:
            raise KeyboardInterrupt
        asked_stop.append(True)
        print("\n  stopping after this chunk; checkpoint on the way",
              flush=True)
    try:
        signal.signal(signal.SIGINT, _on_int)
        signal.signal(signal.SIGTERM, _on_int)
    except ValueError:
        pass                                   # not the main thread
    # Read a window from one subject, then a window from the next. Reading a
    # file end to end instead meant 200,000 characters - seventeen minutes at
    # this speed - of one subject with nothing to balance it, and the held-out
    # score swung by two nats depending on which subject the reader happened
    # to be inside.
    while not stop and seen < target:
        for lane in lanes:
            r = lane.open(model, args.chunk, ctx_now, device,
                          turn * args.chunk)
            if r is None:
                continue
            path, data = r.name, r.data
            fl = []
            for j in range(turn):
                if r.done():
                    break
                # Ask what THIS text wants, before reading it. The experts
                # that compute the forward are the ones the backward updates -
                # swapping between the two would apply one expert's gradient
                # to whichever expert took its slot.
                nxt = r.peek()
                moved = 0
                if nxt is not None:
                    # j == 0 opens this lane's window: a new passage, whose
                    # first chunk has no previous chunk of its own. The buffer
                    # at that moment holds the states of the LAST SUBJECT -
                    # scoring on those chooses a working set for text the
                    # model has stopped reading. Read the chunk once and score
                    # on its own states instead; one forward per --passage
                    # characters, 512 in 32,768.
                    #
                    # Every chunk after that keeps the cheap path: its
                    # predecessor is the same passage, and locally coherent
                    # text is what demand() was built to score.
                    moved = (model.peek_experts(nxt, free=True) if j == 0
                             else model.want_experts(nxt))
                    swapped += moved
                    if tracer is not None and not tracer.done():
                        # j == 0 is the chunk that opens this lane's window -
                        # the only point where the subject actually changes
                        tracer.add_swap(lane, moved, j == 0)
                # One rate, scaled by the plasticity controller. It moves
                # only when held-out says something has changed - down on a
                # plateau, back up when the ground moves - so there is no
                # horizon to be wrong about and no count of characters
                # anywhere in it.
                scale = plast.factor()
                for gg in opt.param_groups:
                    gg["lr"] = gg.get("base_lr", gg["lr"]) * scale
                opt.zero_grad(set_to_none=True)
                # the reader wraps its own forward in the process-wide
                # precision; backward runs outside it on purpose, replaying
                # the dtypes the forward recorded into the parameters' fp32
                if tracer is not None and not tracer.full():
                    from minagi.pool import capture_routes
                    with capture_routes() as got:
                        loss = r.step(learn=True, aux_weight=cfg.pool_aux)
                    if loss is not None and got:
                        tracer.add(got, pool, lane, moved, step, float(loss),
                                   seen, nxt)
                else:
                    loss = r.step(learn=True, aux_weight=cfg.pool_aux)
                # writing is a side errand - it must never stand in for the
                # forward, or backward would run on the previous chunk's
                # freed graph
                if tracer is not None and tracer.done() and not tracer.written:
                    tracer.write(cfg, args.precision, args.chunk)
                if loss is None:
                    break
                loss.backward()
                grads.append(float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip)))
                if step % 8 == 0:
                    # how much of the trunk gradient is signal. Reported only:
                    # a signal is watched before it is trusted with anything.
                    snr.observe(trunk)
                opt.step()
                # nothing to detach: the reader re-forwards its whole window
                # every step so the gradient can reach all of it, and keeps no
                # cache between steps. Detaching is what used to cut the
                # gradient at the chunk boundary.
                fl.append(float(loss))
                recent.append(float(loss))
                by_pos[j].append(float(loss))
                seen += args.chunk
                nats += float(loss) * args.chunk
                step += 1
                pool.now = step          # the clock the expert trial reads
                plast.tick()
                did += 1
                if args.sample_every and (
                        time.time() - last_sample >= args.sample_every * 60):
                    last_sample = time.time()
                    v_ = se_ = None
                    dom = None
                    if ev is not None:
                        d = ev.run(sample_eval_steps)
                        se_ = d.pop("stderr", None)
                        v_ = float(np.mean(list(d.values())))
                        # The generalisation gap, kept for the growth
                        # decision. It is the thing the density ceiling was
                        # always a proxy for, and unlike density it is
                        # measured rather than assumed.
                        if recent:
                            last_gap = v_ - float(np.mean(recent))
                        dom = dict(d)
                        moved_lr = plast.observe(v_, se_)
                        if moved_lr:
                            print(f"    {moved_lr}", flush=True)
                    s = sample_now(model, tok, device, args.sample_chars,
                                   variants=_sample_variants(args))
                    dt = max(time.time() - mark_t, 1e-6)
                    rate = (seen - mark_c) / dt
                    mark_t, mark_c = time.time(), seen
                    tr = float(np.mean(recent)) if recent else None
                    write_samples(args.sample_log, step, base_chars + seen,
                                  (time.time() - t0) / 60, s, v_, se_, dom,
                                  pool.n_experts(), tr,
                                  opt.param_groups[-1]["lr"],
                                  info["characters"], ctx_now, ctx_max,
                                  context_gain(), rate,
                                  float(np.mean(grads)) if grads else None,
                                  args.clip, plast.state())
                    rp = pool.report()
                    print(f"    samples -> {args.sample_log}"
                          + (f"  train {float(np.mean(recent)):.4f}"
                             if recent else "")
                          + (f"  held-out {v_:.4f}" if v_ is not None else "")
                          + (f"  signal {snr.ratio():.3f}"
                             if snr.ratio() is not None else "")
                          + f"  | {swapped/max(did,1):.1f} experts loaded "
                            f"onto the card per chunk, RAM hit rate "
                            f"{rp['hit_rate']:.2f}"
                          + f"  | context {ctx_now:,}  {rate:,.0f} char/s"
                          + _dropped_note(model)
                          + f"  ({(time.time()-t0)/60:.0f} min)", flush=True)
                    if args.save:
                        checkpoint(v_)
                        last_save = time.time()
                        print(f"    weights/ checkpointed", flush=True)
                    elif args.history:
                        # A dry read still trains in memory, and the gate
                        # trajectory is the whole point of an experiment that
                        # is not meant to keep its weights. Writing the
                        # history costs nothing; writing the model costs
                        # minutes and would dominate a short run.
                        log_history(v_)
                # Gathering the evidence and acting on it are separate
                # things. Tied together, the window grew one character per
                # evidence batch - measured, 29 characters in 6 million read,
                # so reaching 16k would have needed more text than the corpus
                # holds. The evidence is re-gathered every context_every
                # chunks; while it says the model is still gaining deep into
                # the window, the window grows a character per chunk.
                if ctx_now < ctx_max and ctx_every_steps:
                    if step % ctx_every_steps == 0:
                        g_ = context_gain()
                        if g_ is not None:
                            wants_more = g_ > args.context_gain_min
                            by_pos.clear()
                    # One character at a time, and not on every chunk. The
                    # window is small early, so a character per chunk is a
                    # quarter of the window per sample interval - fast in the
                    # terms that matter, since what decides whether the model
                    # keeps up is how much the window grows relative to what
                    # it already handles.
                    if wants_more and step % ctx_grow_every_steps == 0:
                        ctx_now = min(ctx_max, ctx_now + args.context_step)
                        turn = visit_chunks(ctx_now)
                if (args.save and args.save_every
                        and time.time() - last_save >= args.save_every * 60):
                    last_save = time.time()
                    checkpoint()
                    print(f"    weights/ checkpointed at step {step:,} "
                          f"({seen/1e6:.1f}M characters)", flush=True)
                if (grower is not None and grow_every_steps
                        and step % grow_every_steps == 0):
                    # Only a run that keeps what it learns may change what is
                    # on disk. Growing writes new expert files and pruning
                    # deletes and renumbers them, both immediately, while the
                    # manifest that indexes them is only written at a
                    # checkpoint - so a dry read that pruned and then stopped
                    # left a directory naming experts that were no longer
                    # there, and the next load died on the first missing file.
                    gone = pool.prune(step, survival=survival_steps) \
                        if args.save else 0
                    # Whether the model may grow is asked of the model as
                    # it is now, not of anything accumulated over the run.
                    #
                    # The one remaining question the pool cannot answer about
                    # itself is whether it has started memorising, because
                    # that needs a held-out set. Everything else - is capacity
                    # idle, are recent additions earning, is there room - is a
                    # property of the network at this instant and is asked
                    # inside AutoGrow.
                    #
                    # What was here before was a ceiling on parameters per
                    # character read, integrated over the whole run. It was
                    # wrong twice over. Its unit ignored that a character of
                    # chess costs 0.744 nats to predict and one of wikipedia
                    # 1.686, so a long run of chess bought as much capacity as
                    # a run of encyclopaedia. And measured against 117
                    # checkpoints of this run it did not even track the thing
                    # it stood in for: correlation with the gap +0.20, not
                    # monotonic. It held growth off for 14M characters on
                    # evidence that did not exist.
                    gap_ok = (last_gap is None or args.max_gap <= 0
                              or last_gap < args.max_gap)
                    may_grow = gap_ok
                    if not may_grow and step % (args.grow_every * 10) == 0:
                        print(f"    growth held: train and held-out have "
                              f"separated by {last_gap:.3f}, over "
                              f"{args.max_gap:.2f} - that is memorising, and "
                              f"more capacity would make it worse", flush=True)
                    rec = (grower.step(float(loss), pool, step)
                           if (args.save and may_grow) else {"grew": 0})
                    if gone or rec["grew"]:
                        _resync_opt(opt, model, args)
                        pool.attach_optimiser(opt)
                        # The expert files have already moved. Write the index
                        # that names them NOW rather than waiting for the next
                        # checkpoint: between those two moments the directory
                        # describes a pool that no longer exists, and anything
                        # that stops in the gap cannot be loaded again.
                        if args.save:
                            checkpoint()
                            last_save = time.time()
                        print(f"    pool {pool.n_experts()} experts "
                              f"({'+%d' % rec['grew'] if rec['grew'] else ''}"
                              f"{'-%d' % gone if gone else ''})  "
                              f"{pool.vram_params()/1e6:.1f}M in VRAM  "
                              f"vram {torch.cuda.max_memory_allocated()/1e6:.0f}MB"
                              if device.type == "cuda" else "", flush=True)
                if asked_stop or (args.minutes
                                  and (time.time() - t0) / 60 >= args.minutes):
                    break
            if args.minutes and (time.time() - t0) / 60 >= args.minutes:
                print(f"  reached {args.minutes:g} minutes", flush=True)
                if args.save:
                    checkpoint()
                    print(f"    weights/ checkpointed at step {step:,} "
                          f"({seen/1e6:.1f}M characters)", flush=True)
                stop = True
            elif asked_stop:
                if args.save:
                    checkpoint()
                    print(f"    weights/ checkpointed at step {step:,} "
                          f"({seen/1e6:.1f}M characters)", flush=True)
                stop = True
            lane.rest(model)               # the window is done; free its cache
            if fl:
                losses.append(float(np.mean(fl)))
                if args.verbose:
                    print(f"  {lane.name:<12} {os.path.relpath(path):<44} "
                          f"{len(data)/1e3:>7.1f}k chars  loss {losses[-1]:.4f}",
                          flush=True)
            if stop or seen >= target:
                break
    el = time.time() - t0
    print(f"\nread {seen/1e6:.2f}M characters in {el/60:.1f}m "
          f"({seen/max(el,1e-9)/1e3:.1f}k char/s)")
    if losses:
        print(f"  mean loss over the files: {np.mean(losses):.4f}")

    if ev is not None:
        v = ev.run(eval_steps); se = v.pop("stderr", 0.0)
        after = float(np.mean(list(v.values())))
        d = after - before
        print(f"  after:  {after:.4f} +/-{se:.4f}   {d:+.4f}")
        if d > 2 * se:
            print(f"  reading this cost ground on the held-out set - that is "
                  f"what forgetting looks like, measured rather than assumed")

    if args.sample_every:
        s = sample_now(model, tok, device, args.sample_chars,
                                   variants=_sample_variants(args))
        write_samples(args.sample_log, step, base_chars + seen,
                      (time.time() - t0) / 60, s,
                      after if ev is not None else None,
                      se if ev is not None else None,
                      dict(v) if ev is not None else None,
                      pool.n_experts(),
                      float(np.mean(recent)) if recent else None,
                      opt.param_groups[-1]["lr"], info["characters"],
                      ctx_now, ctx_max, context_gain(),
                      seen / max(time.time() - t0, 1e-6),
                      float(np.mean(grads)) if grads else None, args.clip)
        print(f"  final samples in {args.sample_log}")

    if args.save:
        # Carry the reading position and the counters through. Saving without
        # them dropped read_pass/read_file from the manifest, so a run that
        # finished normally sent the next one back to the first file - only a
        # run killed between checkpoints resumed where it had got to.
        weights_store.save(model, wdir, step=step, val=None,
                           opt=opt, cfg=asdict(cfg), verbose=True,
                           extra={"read_chars": base_chars + seen,
                                  "read_nats": nats,
                                  "plasticity": plast.state(),
                                  "context_now": ctx_now})
        print(f"  weights/ updated - the model has read this and kept it")
    else:
        print(f"  not saved (pass --save to keep what it learned)")
    return 0


# One prompt per corpus in data/train, in that corpus's OWN format. A prompt
# the training data never looks like tells you nothing: the model is off
# distribution before it has generated a character.
#
# `stories` was missing for a long time and it is the one worth watching. It is
# the easiest domain by a distance - simple vocabulary, short sentences - and
# it reads 1.24 bits per character against wikipedia's 2.29, so coherence
# appears there first. Judging the model on the chat prompt alone understates
# it.
SAMPLE_PROMPTS = [
    ("stories", "Once upon a time, there was a little boy named Tom. One day he "),
    ("code", "def merge_sorted(a, b):\n    "),
    ("arithmetic", "add 4917 + 388 = "),
    ("chat", "<user>\nWhat are you?\n</user>\n<bot>\n"),
    # same markers as `chat`; what differs is that the answer needs working out
    ("chat_hermes",
     "<user>\nA train travels 60 km in 45 minutes. "
     "What is its speed in km/h?\n</user>\n<bot>\n"),
    # the reasoning corpus is coding tasks, and the bot opens with <think>
    ("reasoning",
     "<user>\nWrite a Python function that returns the largest number in a "
     "list.\n</user>\n<bot>\n<think>\n"),
    # raw MediaWiki markup, links and headings included - that is what it read
    ("wikipedia", "== History ==\nThe [[Roman Empire]] was "),
    ("chess", "<g>1700 1-0 1. e4 e5 2. "),
    # the model's account of itself. Same markers as chat, because that is the
    # shape the self-knowledge corpus is written in - and this is the prompt
    # worth watching, since the lane went from 0.25% of reading to 12.5%
    ("self-knowledge",
     "<user>\nhow do you decide which experts to use?\n</user>\n<bot>\n"),
]


@torch.no_grad()
def _sample_variants(args):
    """
    Which readings of each prompt to write down.

    RAW first, always. It is the only one that says what the model predicts,
    and it is the one whose repeat rate is a fact about the model rather than
    about the decoding rule.

    --sample-raw-only drops the guarded reading, for when the raw text is good
    enough that the guard is no longer telling you anything.
    """
    v = [("raw", dict(rep_penalty=1.0, adapt_strength=0.0))]
    if not getattr(args, "sample_raw_only", False):
        v.append(("adapted", dict(rep_penalty=args.rep_penalty,
                                  adapt_strength=args.adapt_strength,
                                  adapt_decay=args.adapt_decay)))
    return v


def repeat_rate(s, n=8):
    """
    The share of this text's n-grams that have been seen before in it.

    Kept as a NUMBER rather than an impression, because it is the thing a
    better model should drive down on its own. Measured on the RAW reading it
    describes the model; measured on the guarded one it describes the guard.
    """
    g = [s[i:i + n] for i in range(max(len(s) - n + 1, 0))]
    if not g:
        return 0.0
    seen, rep = set(), 0
    for x in g:
        if x in seen:
            rep += 1
        seen.add(x)
    return rep / len(g)


_SAN = re.compile(r"^(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
                  r"[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][1-8](?:=[QRBN])?|"
                  r"O-O(?:-O)?)[+#]?$")
_NUMBER = re.compile(r"^\d+\.+$")
_RESULT = ("1-0", "0-1", "1/2-1/2", "*")


def chess_legality(prompt, text):
    """
    How many moves into its own game the model gets before an impossible one.

    Free, because the chess prompt is generated at every checkpoint anyway -
    this only replays the text that is already in the log. Returns
    (legal_moves, first_bad_token) with first_bad_token None when the sample
    ran out of characters before making a mistake, or None if the position
    cannot be set up at all.

    THIS IS THE OPTIMISTIC NUMBER. The model plays both sides here, so it can
    steer toward lines it knows; in a real game the opponent's moves arrive
    unbidden. `tools/chess_legality.py` measures the honest version on
    held-out positions, at the cost of a forward pass per position.

    If p is the chance one move is legal, a game where the model plays m moves
    finishes cleanly with probability p**m, and p is estimated from a run of
    n legal moves as n/(1+n). A 40-move game needs p > 0.983.
    """
    try:
        import chess                                  # noqa: PLC0415
    except ImportError:
        return None                                   # never a reason to fail
    try:
        board = chess.Board()
        for t in prompt.replace("\\n", " ").split(">")[-1].split():
            if _NUMBER.match(t) or t in _RESULT or t.isdigit():
                continue                              # move number, result, elo
            board.push_san(t)
        n = 0
        for t in text.replace("\n", " ").split():
            if _NUMBER.match(t):
                continue
            if t in _RESULT:
                break
            if not _SAN.match(t):
                return n, t                           # not even move-shaped
            try:
                board.push_san(t)
            except ValueError:
                return n, t                  # real notation, impossible move
            n += 1
        return n, None
    except Exception:
        # A malformed prompt or a python-chess quirk is not worth a run for.
        return None


def sample_now(model, tok, device, n_new=140, variants=None):
    """
    What the model says right now, on one prompt from each domain.

    TWO readings of every prompt, because neither alone is honest:

      raw       plain greedy, no guard of any kind. This is what the model
                actually predicts - the true argmax trajectory - and the only
                text here that can be reasoned from about the MODEL.
      adapted   the same, with the adaptation trace holding off loops. Easier
                to read, and partly written by the decoding rule: instrumented
                character by character, adaptation overruled the model on 28 of
                72 choices, so more than a third of that text is the guard
                talking rather than the model.

    Where the two agree, the guard is doing nothing and the model stands on its
    own. Where they diverge is exactly where it is falling into a hole - a
    signal that was invisible while only the guarded text was kept.

    Nothing draws a random number, and not for reproducibility: CUDA does not
    give that anyway. Temperature would make the text look better by rolling
    dice against the model's own prediction, which is a different thing from
    the model being right. The repeat rate on the RAW reading is the number to
    drive down, and only a better model can drive it.

    Each variant re-chooses its experts from the prompt, so the second is not
    reading with the working set the first left behind.
    """
    from minagi.decode import pick_next
    if variants is None:
        variants = [("raw", dict(rep_penalty=1.0, adapt_strength=0.0)),
                    ("adapted", dict(rep_penalty=1.0, adapt_strength=2.5,
                                     adapt_decay=0.88))]
    was_training = model.training
    model.eval()
    # Sampling must not change what the next training chunk reads with. The
    # pool's observation buffer is consumed by the next want_experts, and a
    # round of sampling left it holding the tail of the last prompt's
    # generation - so one training chunk every ten minutes chose its working
    # set from sample text. Put back whatever was there.
    _pool = getattr(model, "pool", None)
    _held = getattr(_pool, "_h_keep", None)
    _held = list(_held) if _held is not None else None
    out = []
    for name, prompt in SAMPLE_PROMPTS:
        texts = []
        for label, kw in variants:
            ids = list(tok.encode(prompt).ids)[-model.cfg.block:]
            cur = torch.tensor([ids], device=device)
            caches = model.empty_caches()
            # Choose the experts for THIS prompt, by reading it first.
            #
            # want_experts alone scored the buffer of states left by the
            # PREVIOUS prompt's generation, and ignored this one entirely -
            # so every prompt in a round was answered with a working set
            # chosen from the text before it, and the first with one chosen
            # from the last training chunk. peek_experts reads the prompt
            # once and scores on its own states.
            #
            # `free` releases the hysteresis that keeps the working set steady
            # while reading a continuous stream. A prompt is the opposite - a
            # deliberate change of subject - and the model should be free to
            # re-choose at once.
            if hasattr(model, "peek_experts"):
                model.peek_experts(cur, free=True)
            else:
                model.begin_segment()
            off = 0
            got = []
            # NO GRAPH. model.eval() only changes dropout; without this every
            # generated character is retained in an autograd graph, chained to
            # the last through `cur`, across all max_steps rows of expert
            # dispatch. Nothing here is ever backpropagated, and at 24 rows
            # and two readings per prompt it is what put sampling out of
            # memory while training itself sat at 3.8 GB.
            for i in torch.arange(n_new).tolist():
                # Ask again as the answer develops: what the text wants after
                # a hundred characters of chess is not what the prompt alone
                # asked for.
                #
                # It has to be asked of the TEXT. `cur` is one token from the
                # second step onward - `cur = nxt` at the bottom of this loop
                # - so this used to re-pick the whole working set from a single
                # character's embedding, every 64 characters. That is not
                # adaptation but a re-roll on whichever character landed on the
                # boundary, and it can evict experts the prompt chose
                # correctly. `ids + got` is the text so far and is already
                # being built for the decoder below.
                if i and i % 64 == 0 and hasattr(model, "want_experts"):
                    recent = (ids + got)[-model.cfg.block:]
                    model.want_experts(torch.tensor([recent], device=device))
                with torch.no_grad():
                    logits, _ = model(cur, caches=caches, pos_offset=off)
                off += cur.shape[1]
                nxt = pick_next(logits[:, -1, :].float(),
                                torch.tensor([ids + got], device=device),
                                temperature=0.0, **kw)
                t = int(nxt[0, 0])
                got.append(t)
                cur = nxt
            texts.append((label, tok.decode(got)))
            del caches
        out.append((name, prompt, texts))
    if _pool is not None:
        _pool._h_keep = _held
    if was_training:
        model.train()
    return out

# The training graphs, redrawn whenever the sample log gains an entry.
#
# Kept beside write_samples rather than run by hand, because a graph that has
# to be remembered is a graph that is out of date when it is looked at.
#
# Drawn in a SEPARATE PROCESS, and not waited for. It takes about two seconds,
# which is nothing against a ten-minute sample interval, but a training step
# should not be held up by matplotlib, and a plotting bug must not be able to
# stop a run that has been going for days.
def _dropped_note(model):
    """What the capacity bound threw away since the last sample, or nothing.

    Silent when it is zero, which is the common case and the case that needs
    no words. `self.dropped` was counted in the dispatch and read nowhere for
    as long as it existed, so the one trade `pool.capacity_factor` makes had
    no number attached to it.
    """
    fn = getattr(model, "pool_dropped", None)
    if fn is None:
        return ""
    try:
        share, dropped, routed = fn()
    except Exception:
        return ""
    if not routed or not dropped:
        return ""
    return (f"  | capacity dropped {share:.2%} of routing "
            f"({dropped:,} of {routed:,})")


_PLOTS = {"proc": None, "on": True, "since": None, "log": None,
          "weights": None}


def refresh_plots():
    """Redraw the graphs from the log just written. Never raises, never waits."""
    if not _PLOTS["on"]:
        return
    p = _PLOTS["proc"]
    if p is not None and p.poll() is None:
        return                          # the last one has not finished
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    tool = os.path.join(here, "tools", "plot_progress.py")
    if not os.path.exists(tool):
        _PLOTS["on"] = False
        return
    log = _PLOTS["log"] or "runs/samples.txt"
    # log-log on the whole history: it spans decades, and a power law is a
    # straight line there, so the fitted trend becomes a ruler. The zoomed
    # graph stays linear - inside a trailing window the axis covers a factor
    # of one and a bit, where a log scale says nothing.
    # runs/dashboard.png is THE one to watch. It was two graphs - the recent
    # training curve and the pool - which meant switching between them to
    # answer one question, and being shown different steps when one had been
    # redrawn and the other had not.
    #
    # training_progress.png stays because it answers a different question: the
    # whole history on log-log, where a power law is a straight line and the
    # fitted trend becomes a ruler. Useless inside a trailing window, which is
    # exactly where the dashboard lives.
    cmds = [[sys.executable, tool, "--log", log, "--logx",
             "--out", "runs/training_progress.png"]]
    dash = os.path.join(here, "tools", "plot_dashboard.py")
    if os.path.exists(dash):
        c = [sys.executable, dash, "--log", log,
             "--weights", _PLOTS.get("weights") or "weights",
             "--out", "runs/dashboard.png"]
        if _PLOTS["since"] is not None:
            c += ["--last", str(_PLOTS["since"])]
        cmds.append(c)
    try:
        script = " && ".join(" ".join(f'"{x}"' for x in c) for c in cmds)
        _PLOTS["proc"] = subprocess.Popen(
            ["bash", "-c", script], cwd=here,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        _PLOTS["on"] = False            # no python, no bash - stop trying


def write_samples(path, step, chars, minutes, samples, val=None, se=None,
                  per_domain=None, experts=None, train=None, lr=None,
                  corpus=None, context=None, ceiling=None, gain=None,
                  rate=None, gnorm=None, clip=None, plast=None):
    with open(path, "a") as f:
        f.write(f"\n{'=' * 78}\n")
        # How much has been read, against how much there is. A bare count of
        # characters read says nothing on its own - the same 4M is most of a
        # small corpus and a rounding error of a large one.
        f.write(f"step {step:,}   {chars/1e6:.1f}M")
        if corpus:
            f.write(f" of {corpus/1e6:,.0f}M characters "
                    f"({100.0 * chars / corpus:.2f}%)")
        else:
            f.write(" characters")
        f.write(f"   {minutes:.0f} min")
        if experts is not None:
            f.write(f"   {experts} experts")
        f.write("\n")
        if context is not None:
            # The window this was read and scored at. It moves during a run,
            # so a score means little without it.
            f.write(f"context {context:,} characters")
            if ceiling:
                f.write(f" of {ceiling:,}")
            if rate:
                # Measured over the interval since the last entry, not over
                # the whole run, so it describes the window in force now
                f.write(f"   reading {rate:,.0f} char/s")
            if gain is not None:
                f.write(f"   still gaining {gain:+.4f} deep into it")
            f.write("\n")
        if gnorm is not None:
            # Whether the clip is what limits the step. A norm far above the
            # clip means most of the learning rate is being thrown away, and
            # raising the rate only makes the direction noisier.
            f.write(f"grad norm {gnorm:.2f}")
            if clip:
                f.write(f" against a clip of {clip:g}   "
                        f"{'clipping' if gnorm > clip else 'under the clip'}")
            f.write("\n")
        if train is not None:
            # The loss on the text being read, beside the loss on text held
            # out. Reading draws fresh windows from a corpus far larger than
            # any run gets through, so the training number is already a
            # measure of unseen text - and a large gap between the two is
            # therefore not overfitting but a sign the two are being measured
            # differently.
            f.write(f"train loss {train:.4f}")
            if lr:
                f.write(f"   lr {lr:.2e}")
            if plast:
                # What the rate controller can currently prove. t is the
                # verdict: it eases the rate UP above T_MID (1.25) and down
                # below it, so the number says which way it is leaning before
                # the rate has moved. `effect` is the number the verdict is
                # now built from - improvement per evaluation in units of the
                # residual scatter - and it is printed because t alone hid the
                # bug: t grows as evidence accumulates whether or not the
                # model is still improving, and effect does not.
                f.write(f"   evidence t {plast.get('t', 0):+.2f}"
                        f" over {plast.get('n', 0)}"
                        f" (effect {plast.get('e', 0):+.4f})"
                        f"   rate x{plast.get('scale', 1):.3f}")
            f.write("\n")
        if val is not None:
            f.write(f"held-out loss {val:.4f}")
            if se:
                f.write(f" +/-{se:.4f}")
            # Say the unit. This is a natural log; the benchmark reports the
            # same quantity in bits, and 1.30 nats against 1.88 bits looks
            # like a difference when it is the same number twice.
            f.write(f" nats   {val / math.log(2):.4f} bits/char"
                    f"   perplexity {math.exp(val):.2f}")
            if train is not None:
                f.write(f"   gap {val - train:+.4f}")
            f.write("\n")
            if per_domain:
                f.write("  " + "   ".join(
                    f"{k} {v:.3f}" for k, v in sorted(per_domain.items())) + "\n")
        # The repeat rate on the UNGUARDED reading, averaged over the
        # prompts. This is a fact about the model - how often it falls into
        # saying what it has just said - and the number a better model should
        # drive down without any decoding rule holding it down from outside.
        raw = [t for _, _, ts in samples if not isinstance(ts, str)
               for lab, t in ts if lab == "raw"]
        if raw:
            f.write(f"repeats {sum(repeat_rate(t) for t in raw)/len(raw):.0%} "
                    f"of 8-grams, greedy with no guard\n")
        f.write(f"{'=' * 78}\n")
        for name, prompt, texts in samples:
            f.write(f"\n--- {name} ---\n")
            f.write(f"prompt: {prompt!r}\n")
            # A bare string is still accepted, so anything that calls this with
            # one reading keeps working.
            if isinstance(texts, str):
                texts = [("", texts)]
            for label, text in texts:
                r = repeat_rate(text)
                head = f"[{label}]" if label else ""
                if head:
                    f.write(f"{head}  repeated 8-grams {r:.0%}")
                    # Chess is the one prompt whose output can be CHECKED, so
                    # it is, on the text already generated. How far the model
                    # gets before an impossible move is the number that
                    # decides whether a whole game is possible at all.
                    legal = (chess_legality(prompt, text)
                             if name == "chess" else None)
                    if legal is not None:
                        n, bad = legal
                        f.write(f"   {n} legal moves"
                                + (f", then {bad}" if bad else
                                   ", none illegal"))
                    f.write("\n")
                f.write(f"{text}\n")
    # the file is closed and complete before the plotter reads it
    refresh_plots()
    return path

def _split_trunk_pool(model):
    """
    Which parameters are the pool's, and which are the shared body.

    A resident pool keeps its experts as modules; a paged pool keeps three
    stacked slot tensors instead, so the set has to be found by asking the
    pool rather than by walking a ModuleList that is not there.
    """
    p = model.pool
    ids = {id(p.gate)}
    if hasattr(p, "experts") and p.experts is not None:
        ids |= {id(q) for e in p.experts for q in e.parameters()}
    for nm in ("w1", "w3", "w2", "segment_router"):
        obj = getattr(p, nm, None)
        if obj is None:
            continue
        ids |= ({id(obj)} if torch.is_tensor(obj)
                else {id(q) for q in obj.parameters()})
    for s in model.modules():
        if isinstance(s, PooledMLP):
            ids.add(id(s.router.weight))
            ids.add(id(s.depth_emb))
    trunk = [q for q in model.parameters() if id(q) not in ids]
    pool = [q for q in model.parameters() if id(q) in ids]
    return trunk, pool


def build_paged(wdir, device, resident=None, ram_capacity=256, ceiling=None,
                read_only=False):
    """
    Load the model with its pool on disk rather than in VRAM.

    The three tiers the project describes: every expert is a file, RAM keeps
    the recently wanted ones, and only the working set is on the card. What
    makes it possible is that the working set is chosen once per segment
    rather than per token - per-token routing wanted nearly the whole pool
    within microseconds and nothing could be left out.

    read_only makes the pool incapable of writing to `wdir`. Pass it from any
    tool that inspects a directory a training run may own - paging an expert in
    marks it dirty whether or not anything touched it, so a plain read would
    otherwise write expert files back under the run.
    """
    from minagi.paged import PagedPool
    with open(os.path.join(wdir, "manifest.json")) as f:
        man = json.load(f)
    cfgd = dict(man["cfg"])
    resident = resident or cfgd.get("pool_resident") or cfgd["pool_top_k"]
    cfg = RecurConfig(**{k: v for k, v in cfgd.items()
                         if k in RecurConfig.__dataclass_fields__})
    # The capacity bound comes from config.yaml rather than from the
    # checkpoint, because it is a property of the machine the model is running
    # on - how much VRAM the dispatch may use - not of the model. A directory
    # written before this existed carries no value for it and would otherwise
    # get the dataclass default, which is right but silent; taking it from the
    # config every load means the number in the file is the number in effect.
    _key_floor = _audition = None
    try:
        from minagi.config import load as _load_cfg, get as _get_cfg
        _c = _load_cfg()
        cfg.pool_capacity_factor = float(
            _get_cfg(_c, "pool.capacity_factor", cfg.pool_capacity_factor))
        # Same reasoning as the capacity factor: how the pool is SCHEDULED is
        # a property of the run, not of the checkpoint. Applied here rather
        # than in cmd_read so that everything opening a paged model - serving,
        # the probes - schedules it the way the run does.
        _key_floor = _get_cfg(_c, "pool.key_floor", None)
        _audition = _get_cfg(_c, "pool.audition_slots", None)
    except Exception:
        pass
    # the per-token router keeps one row per EXPERT, not per VRAM slot, so it
    # is sized by the pool and grows with it - see PooledMLP.__init__
    # EXACTLY what the pool holds. pool_max sizes the routers, which keep one
    # row per expert, so a checkpoint's router weights only fit a model built
    # at the same count - one row too many and load_state_dict refuses the
    # whole model.
    #
    # It was briefly max(cfg.pool_max, n_experts) to stop `stream` latching
    # its growth ceiling to the pool's current size. That conflated two jobs
    # in one number: sizing the routers, and capping growth. The manifest's
    # pool_max ratchets up and never comes down, so after a prune took the
    # pool from 158 to 157 the max() still read 158 and the directory would
    # not load at all. The ceiling now lives in `stream` where it belongs.
    cfg.pool_max = int(man["n_experts"])
    if ceiling and ceiling > cfg.block:
        # RoPE tables are built to cfg.block and carry no learned parameters,
        # so raising the ceiling on an existing model costs a bigger table and
        # nothing else. The window the reader actually uses is separate, and
        # grows a character at a time.
        cfg.block = int(ceiling)
    model = RecurCoder(cfg).to(device)
    pool = PagedPool(os.path.join(wdir, "experts"), cfg.d_model, cfg.pool_d_ff,
                     int(man["n_experts"]), resident=resident,
                     ram_capacity=ram_capacity, device=device,
                     read_only=read_only).to(device)
    for m in model.modules():
        if isinstance(m, PooledMLP):
            m._pool[0] = pool
    model.pool = pool
    pool.attach_sites(model)
    if _key_floor is not None:
        pool.key_floor = float(_key_floor)
    if _audition is not None:
        pool.audition_slots = int(_audition)
    pool.load_telemetry(man.get("telemetry"))
    ever = cfgd.get("pool_ever")
    if ever:
        n = min(len(ever), pool.ever.numel())
        pool.ever[:n] = torch.tensor(ever[:n], dtype=torch.bool,
                                     device=pool.ever.device)
    since = cfgd.get("pool_since")
    if since:
        n = min(len(since), pool.since.numel())
        pool.since[:n] = torch.tensor(since[:n], dtype=pool.since.dtype,
                                      device=pool.since.device)
    # WHAT EACH EXPERT RESPONDS TO. `keys` is a non-persistent buffer, so a
    # resumed run started with all of them zero, and `want` then skipped the
    # key term entirely - see PagedPool.route. That term is half the routing
    # mass (key_weight 1.0 against a router normalised to the same total), and
    # it is the half that can speak for an expert which is NOT resident. The
    # router alone speaks only for the ~32 on the card, so the other ~190 were
    # ranked by their gate and nothing else: content-blind, the same experts
    # chosen for every input. That is the held-out jump at every restart, and
    # the experts left unchosen went flat, went silent, and were pruned in
    # bulk once they were old enough. A key is a function of w1, so rebuilding
    # is exact - nothing needs to be stored.
    if hasattr(pool, "build_keys"):
        pool.build_keys(verbose=not read_only)
    # the trunk still comes from the bundles; the experts come from their files
    core = np.load(os.path.join(wdir, "core.npz"))
    rout = np.load(os.path.join(wdir, "routers.npz"))
    sd = {k: torch.from_numpy(core[k]) for k in core.files}
    # Router rows belong to EXPERTS, not to VRAM slots, so they are loaded
    # whole. Slicing them to the resident count - which this did, left over
    # from when rows were slots - silently discarded every expert past the
    # working set and made any resume after growth fail to load.
    for k in rout.files:
        sd[k] = torch.from_numpy(rout[k])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in unexpected if "router" in k or "gate" in k]
    if bad:
        raise RuntimeError(f"router/gate tensors did not load: {bad}")
    return model, cfg, pool, man


def cmd_ponder_probe(args):
    """
    Does the model actually spend more computation on harder problems?

    This is the falsifiable claim behind adaptive depth. If mean halting steps
    are flat across difficulty, the halting head learned nothing useful and the
    mechanism is decoration. Reported per digit count, which is the cleanest
    difficulty axis available.
    """
    import corpora.arithmetic as math_data
    from minagi.tokenizer import load_tokenizer
    device = torch.device(args.device)
    model, ck = load_recur(args.ckpt, device)
    tok = load_tokenizer(args.data)
    import random
    rng = random.Random(0)
    print(f"{'digits':>7} {'mean steps':>11} {'max':>5}  {'example':<34}")
    rows = []
    for d in range(1, args.max_digits + 1):
        steps = []
        example = ""
        for _ in range(args.n):
            fn, cap, _ = math_data.TASKS[args.task]
            line = fn(rng, min(d, cap), False)
            prompt = line.rpartition("=")[0] + "="
            example = example or prompt
            ids = torch.tensor([tok.encode(prompt).ids], device=device)
            with torch.no_grad():
                _, extra = model(ids, collect=True)
            steps.append(float(extra["steps"][0, -1]))
        rows.append((d, float(np.mean(steps)), max(steps)))
        print(f"{d:>7} {np.mean(steps):>11.2f} {max(steps):>5.0f}  {example:<34}")
    lo = rows[0][1]
    hi = rows[-1][1]
    print(f"\n1-digit {lo:.2f} steps -> {args.max_digits}-digit {hi:.2f} steps "
          f"({hi-lo:+.2f})")
    print("adaptive compute is working" if hi - lo > 0.15 else
          "FLAT - the halting head is not responding to difficulty")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="train mini-AGI: read files continually, or stream a "
                    "packed corpus. Batch 1, cached, chunked, either way")
    ap.add_argument("--device", default=_auto_device())
    sub = ap.add_subparsers(dest="cmd", required=True)

    rd = sub.add_parser("read",
                        help="point the model at files and let it read them")
    from minagi.config import load as _load_cfg, get as _cfg
    _c = _load_cfg()
    # The learning rate is no longer scheduled. A config still carrying the
    # old keys would be silently ignored, and the run would quietly train at a
    # rate the file does not describe - so say so instead.
    _dead = [k for k in ("lr_horizon", "lr_warmup", "lr_floor")
             if isinstance(_c.get("training"), dict) and k in _c["training"]]
    if _dead:
        raise SystemExit(
            f"config.yaml still sets training.{', training.'.join(_dead)}. "
            f"The learning rate is governed by held-out loss now, not by a "
            f"horizon - see minagi/plasticity.py. Remove those keys; `lr` and "
            f"`trunk_lr_mult` are the only two that remain.")
    rd.add_argument("paths", nargs="*",
                    default=[_cfg(_c, "data.train", "data/train")],
                    help="files or directories; binaries are skipped")
    rd.add_argument("--weights-dir", default=_cfg(_c, "data.weights", "weights"))
    rd.add_argument("--passes", type=int, default=1,
                    help="how many times to read the whole set")
    rd.add_argument("--chunk", type=int,
                    default=_cfg(_c, "training.chunk", 512))
    rd.add_argument("--lr", type=float,
                    default=_cfg(_c, "training.lr", 5e-5),
                    help="lower than a training run by default: reading your "
                         "files should adjust the model, not overwrite it")
    rd.add_argument("--wd", type=float,
                    default=_cfg(_c, "training.weight_decay", 0.1))
    rd.add_argument("--clip", type=float,
                    default=_cfg(_c, "training.clip", 1.0))
    rd.add_argument("--seed", type=int, default=0)
    rd.add_argument("--save", action="store_true",
                    help="keep what it learned; without this the run is a dry "
                         "read and weights/ is untouched")
    rd.add_argument("--held-out", default=_cfg(_c, "data.held_out", "data/val"),
                    help="a folder of text to score before and after, so the "
                         "cost of reading is visible; each sub-folder is "
                         "reported separately")
    rd.add_argument("--eval-chars", dest="eval_chunks", type=int, default=_cfg(_c, "data.eval_chars", 240 * 512),
                    help="CHARACTERS of held-out read per domain per "
                         "evaluation. A chunk count here meant the evaluation "
                         "grew with the chunk: 60 chunks is 30,720 characters "
                         "at chunk 512 and 122,880 at 2,048, so the same "
                         "config would have quadrupled how long every sample "
                         "takes")
    rd.add_argument("--verbose", action="store_true")
    rd.add_argument("--shuffle-seed", type=int,
                    default=_cfg(_c, "data.shuffle_seed", 0),
                    help="seed for which file and which offset each window is "
                         "drawn from")
    rd.add_argument("--sample-every", type=float, default=0,
                    help="minutes between writing sample generations")
    rd.add_argument("--sample-log", default="runs/samples.txt")
    rd.add_argument("--sample-chars", type=int, default=140)
    rd.add_argument("--save-every", type=float,
                    default=_cfg(_c, "training.save_every", 5),
                    help="minutes between checkpoints; 0 to save only at the "
                         "end")
    rd.add_argument("--sample-eval-chars", dest="sample_eval_chunks",
                    type=int, default=60 * 512,
                    help="CHARACTERS of held-out scored alongside each "
                         "sample; smaller than the full evaluation so it is "
                         "quick")
    rd.add_argument("--minutes", type=float, default=0,
                    help="stop after this long")
    rd.add_argument("--trace-routes", default=None,
                    help="write the routing of the first chunks of real "
                         "training to this .npz, for tools/plot_routing.py")
    rd.add_argument("--trace-chunks", type=int, default=8,
                    help="how many chunks --trace-routes records")
    rd.add_argument("--precision", default=_cfg(_c, "training.precision",
                                                "bf16"),
                    choices=["bf16", "fp16", "fp32"],
                    help="what the forward computes in; weights and "
                         "gradients stay fp32 either way, and so do the "
                         "weights inside the expert files - only Adam's "
                         "moments are stored narrower")
    rd.add_argument("--resident", type=int,
                    default=_cfg(_c, "pool.resident", None),
                    help="experts held in VRAM; the rest stay on disk")
    rd.add_argument("--passage", type=int,
                    default=_cfg(_c, "data.passage", 65536),
                    help="contiguous characters read from one file before "
                         "moving to another subject; the window slides along "
                         "it a chunk at a time. Distinct from model.context_end, "
                         "which is how far the model sees and how far the "
                         "gradient reaches")
    rd.add_argument("--context-end", "--context", dest="context", type=int,
                    default=_cfg(_c, "model.context_end",
                                 _cfg(_c, "model.context", 24576)),
                    help="the furthest the window may ever reach. Binds at "
                         "load AND at every growth step, so lowering it on a "
                         "resumed run is the supported way to make a model "
                         "fit a smaller card. The rotary tables are a second "
                         "ceiling and the smaller of the two wins")
    rd.add_argument("--context-start", type=int,
                    default=_cfg(_c, "model.context_start", 1024),
                    help="where a fresh model's window begins")
    rd.add_argument("--context-step", type=int,
                    default=_cfg(_c, "model.context_step", 1),
                    help="characters the window grows by when the model asks")
    rd.add_argument("--context-gain-min", type=float,
                    default=_cfg(_c, "model.context_gain_min", 0.01),
                    help="how much better the model must predict deep into "
                         "the window than early in it before the window grows")
    rd.add_argument("--context-grow-every", type=int,
                    default=_cfg(_c, "model.context_grow_every_chars",
                                 _cfg(_c, "model.context_grow_every", 8) * 512),
                    help="chunks between single-character growths, while the "
                         "evidence says the model is still gaining. Higher is "
                         "slower: 8 gives about 200 characters of window per "
                         "million characters read")
    rd.add_argument("--context-every", type=int,
                    default=_cfg(_c, "model.context_every_chars",
                                 _cfg(_c, "model.context_every", 200) * 512),
                    help="chunks between asking whether to grow")
    rd.add_argument("--history", default=_cfg(_c, "data.history",
                                              "runs/expert_history.jsonl"),
                    help="append one line per checkpoint recording every "
                         "expert's gate, usage and admissions, so a long run "
                         "can be analysed afterwards; empty to disable")
    rd.add_argument("--margin", type=float,
                    default=_cfg(_c, "pool.margin", 0.10),
                    help="how much more a candidate must be wanted before it "
                         "displaces a resident expert")
    # `--dwell` is a strict prefix of `--dwell-chars`, and argparse matches
    # unambiguous prefixes - so the old spelling would be accepted silently and
    # read as characters. An alias that keeps the name and changes the unit is
    # the exact failure this whole change exists to remove, so it is rejected
    # explicitly rather than left to be discovered.
    rd.add_argument("--dwell", dest="dwell_retired", type=int, default=None,
                    help=argparse.SUPPRESS)
    rd.add_argument("--dwell-chars", dest="dwell", type=int,
                    default=_cfg(_c, "pool.dwell_chars",
                                 _cfg(_c, "pool.dwell", 4) * 512),
                    help="CHARACTERS a newly admitted expert is safe from being "
                         "displaced again")
    rd.add_argument("--ram-capacity", type=int,
                    default=_cfg(_c, "pool.ram_cache", 128),
                    help="experts kept in system memory; when full the one "
                         "untouched longest is dropped")
    rd.add_argument("--grow-every", type=int,
                    default=_cfg(_c, "growth.every_chars",
                                 _cfg(_c, "growth.every", 2000) * 512),
                    help="CHARACTERS between growth decisions")
    rd.add_argument("--eval-every", type=int, default=4000)
    rd.add_argument("--grow-k", type=int, default=_cfg(_c, "growth.k", 8),
                    help="experts added per growth decision; 0 disables")
    rd.add_argument("--prune-survival", type=int,
                    default=_cfg(_c, "prune.survival_chars",
                                 _cfg(_c, "prune.survival", 8600) * 512),
                    help="CHARACTERS an expert lives untouched from birth, "
                         "and how long it may then go unchosen before it "
                         "counts as atrophied - one number governs both")
    rd.add_argument("--grow-dying-frac", type=float,
                    default=_cfg(_c, "growth.dying_frac_max", 0.25),
                    help="stop growing once this share of the pool has gone "
                         "`dying_at` of the way to the prune line unaddressed. "
                         "Capacity nothing asks for is not a reason to add more")
    rd.add_argument("--max-in-flight", type=int,
                    default=_cfg(_c, "growth.max_in_flight", 0),
                    help="how many experts may be inside their trial at once; "
                         "0 disables. Neither usage brake can see a newborn, "
                         "so without this there is a whole trial in which "
                         "nothing on that side can refuse")
    rd.add_argument("--max-gap", type=float,
                    default=_cfg(_c, "growth.max_gap", 0.4),
                    help="hold growth when held-out minus train exceeds this; "
                         "0 disables. This is the real guard - a model that "
                         "generalises has earned more capacity")
    rd.add_argument("--max-disk-gb", type=float,
                    default=_cfg(_c, "growth.max_disk_gb", 16.0),
                    help="the pool may not grow past this many GB of expert "
                         "files; 0 means no limit")
    rd.add_argument("--adapt-strength", type=float,
                    default=_cfg(_c, "decoding.adapt_strength", 2.5),
                    help="logits of suppression for a character just used; "
                         "0 disables the adaptation trace")
    rd.add_argument("--adapt-decay", type=float,
                    default=_cfg(_c, "decoding.adapt_decay", 0.88))
    rd.add_argument("--rep-penalty", type=float,
                    default=_cfg(_c, "decoding.rep_penalty", 1.0),
                    help="the older, weaker loop guard; off by default")
    rd.add_argument("--plots", dest="plots", action="store_true",
                    default=_cfg(_c, "plots.enabled", True),
                    help="redraw runs/training_progress.png every time the "
                         "sample log gains an entry (default)")
    rd.add_argument("--no-plots", dest="plots", action="store_false",
                    help="do not redraw the graphs while training")
    rd.add_argument("--plot-last", "--plot-since", type=float,
                    dest="plot_since",
                    default=_cfg(_c, "plots.last",
                                 _cfg(_c, "plots.since", None)),
                    help="how much of the run runs/dashboard.png covers, "
                         "many million characters, counted back from wherever "
                         "the run has got to. A TRAILING span, not a fixed "
                         "start: `since 40` was set at 60M read and by 189M "
                         "it was showing the whole history again")
    rd.add_argument("--sample-raw-only", action="store_true",
                    help="write only the unguarded greedy reading of each "
                         "prompt, not the adapted one as well")
    rd.add_argument("--lr-reset", action="store_true",
                    help="restore the full learning rate by hand. The "
                         "plasticity controller does this on its own when "
                         "held-out says the ground moved; this is the manual "
                         "override for when you know it did and it has not "
                         "noticed yet")
    rd.add_argument("--bptt-window", type=int,
                    default=_cfg(_c, "model.bptt_window", None),
                    help="how many of the last ROWS carry gradient back "
                         "through the recurrence. Counted in rows, so it must "
                         "be rescaled whenever a row changes size")
    rd.add_argument("--ponder-beta", type=float,
                    default=_cfg(_c, "model.ponder_beta", None),
                    help="weight on the KL that holds halting to its prior")
    rd.add_argument("--halt-prior", type=float,
                    default=_cfg(_c, "model.halt_prior", None),
                    help="geometric prior on depth: the KL pulls the halting "
                         "head toward a mean of 1/this ROWS. Must be rescaled "
                         "whenever a row changes size")
    rd.add_argument("--halt-thresh", type=float,
                    default=_cfg(_c, "model.halt_thresh", None),
                    help="at inference, stop at the first row whose cumulative "
                         "halting mass passes this")
    rd.add_argument("--min-steps", type=int,
                    default=_cfg(_c, "model.min_steps", 1),
                    help="rows a character must run before halting may stop "
                         "it; config.yaml had this but only the old `train` "
                         "command ever read it")
    rd.add_argument("--no-pool-checkpoint", action="store_true",
                    help="stop recomputing the expert pass during backward. "
                         "It trades about 30%% more compute for most of the "
                         "pool's activation memory, which is only worth it "
                         "while depth makes that memory the binding "
                         "constraint")
    rd.add_argument("--train-steps-mean", type=float,
                    default=_cfg(_c, "model.train_steps_mean", 0.0),
                    help="sample the recurrence depth while training instead "
                         "of always running max_steps; 0 keeps the old "
                         "behaviour. NOT the mean depth - the draw is "
                         "poisson(this)+1, so 2.0 averages 3.0 of 6")
    rd.add_argument("--dying-at", type=float,
                    default=_cfg(_c, "prune.dying_at", 0.75),
                    help="share of prune.survival_chars an expert may go "
                         "unaddressed before the growth brake counts it as "
                         "dying. One definition of dead, two thresholds on it; "
                         "the gate is read by neither")
    rd.add_argument("--recent-mult", type=float,
                    default=_cfg(_c, "growth.recent_mult", 4.0),
                    help="how far past its trial an expert still counts as "
                         "recent when asking whether additions are earning")
    rd.add_argument("--birth-gate", type=float,
                    default=_cfg(_c, "growth.birth_gate", 0.001),
                    help="gate a new expert is born at; below the growth "
                         "brake's idle line so additions have to earn their way")
    rd.add_argument("--grow-keep-ratio", type=float,
                    default=_cfg(_c, "growth.keep_ratio_min", 0.35))
    rd.add_argument("--segment-chars", dest="segment_chunks", type=int,
                    default=_cfg(_c, "pool.segment_chars",
                                 _cfg(_c, "pool.segment_chunks", 16) * 512),
                    help="CHARACTERS one working set covers")
    rd.add_argument("--explore", type=float,
                    default=_cfg(_c, "pool.explore", 0.15),
                    help="share of the working set reserved for experts that "
                         "have gone longest without being resident")
    rd.add_argument("--trunk-lr-mult", type=float,
                    default=_cfg(_c, "training.trunk_lr_mult", 0.3),
                    help="the trunk is shared by every domain and is where "
                         "forgetting happens; it learns slower on purpose")
    rd.add_argument("--grow-mem-frac", type=float,
                    default=_cfg(_c, "growth.mem_frac", 0.85))
    rd.set_defaults(fn=cmd_read)

    st = sub.add_parser(
        "stream",
        help="batch 1, cached, chunked, over a PACKED corpus. `read` is the "
             "one the project runs on; this is the same mechanism pointed at "
             "a data_* directory instead of a tree of files")
    st.add_argument("--mix", default="data_char:0.25,data_math_char:0.15,"
                                     "data_chat_char:0.25,data_chess_char:0.35")
    st.add_argument("--weights-dir", default="weights")
    st.add_argument("--out", default="runs/agi")
    st.add_argument("--context", type=int, default=32768,
                    help="how far attention reaches; sets the cache, which is "
                         "linear and cheap")
    st.add_argument("--ctx-start", type=int, default=4096,
                    help="context to begin at; ramps to --context, since the "
                         "model degrades past the length it last trained at")
    st.add_argument("--chunk", type=int, default=512,
                    help="characters inside one autograd graph - the only "
                         "thing that sets peak activation memory")
    st.add_argument("--accum", type=int, default=1,
                    help="chunks accumulated before an optimiser step. Costs "
                         "no memory; multiplies the characters each step "
                         "averages over. accum x chunk is the effective batch")
    st.add_argument("--steps", type=int, default=20000)
    st.add_argument("--lr", type=float, default=1e-4)
    st.add_argument("--trunk-lr-mult", type=float, default=0.3)
    st.add_argument("--warmup", type=int, default=200)
    st.add_argument("--wd", type=float, default=0.1)
    st.add_argument("--clip", type=float, default=1.0)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--log-every", type=int, default=100)
    st.add_argument("--eval-every", type=int, default=1000)
    st.add_argument("--eval-chunks", type=int, default=240,
                    help="chunks of held-out text per domain. At 6 the score "
                         "has a standard deviation of 0.0275 across draws, "
                         "which is larger than most differences worth seeing")
    st.add_argument("--grow-k", type=int, default=8)
    st.add_argument("--grow-min-age", type=int, default=3000)
    st.add_argument("--grow-mem-frac", type=float, default=0.80)
    st.add_argument("--pool-max", type=int, default=512)
    st.add_argument("--replay", type=float, default=0.0,
                    help="fraction of windows that revisit a position already "
                         "read instead of opening a new one; batch stays 1")
    st.add_argument("--replay-burst", type=int, default=8,
                    help="consecutive chunks read on each revisit, so the "
                         "replayed text has context rather than an empty cache")
    st.add_argument("--revert-factor", type=float, default=1.5,
                    help="reload weights/ if validation exceeds the best by "
                         "this factor; 0 disables the guard")
    st.add_argument("--max-reverts", type=int, default=3,
                    help="give up after this many reverts rather than thrash")
    st.set_defaults(fn=cmd_stream)

    p = sub.add_parser("ponder-probe")
    p.add_argument("--ckpt", default="weights",
                    help="the weights directory, or a .pt checkpoint")
    p.add_argument("--data", default="data_math_char")
    p.add_argument("--task", default="add")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-digits", type=int, default=8)
    p.set_defaults(fn=cmd_ponder_probe)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()

