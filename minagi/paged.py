"""
A pool larger than the card it runs on.

The claim this exists to make true: disk holds every expert, RAM caches the
ones recently wanted, and VRAM holds only the ones being worked with now. Three
tiers, and only the last is scarce.

Paging is a consequence of how the routing is arranged, not of storage. Routing
per character alone cannot support it: every call site at every depth picks its
own top-k, so a single chunk makes sites x depth x top_k x characters
selections and their union is essentially the whole pool. Nothing could be left
off the card, because everything is wanted within microseconds of everything
else.

So the routing is split into two questions at two timescales, and the paging
follows from the slower one:

    SEGMENT     before each stretch of text, one working set of `resident`
                experts is chosen for the whole model, from a summary of the
                segment just finished. This is the set that lives in VRAM.
    TOKEN       within the segment, each call site still picks its own top-k,
                but only from the experts that are resident.

The choice is made from the PREVIOUS segment, so it cannot see the text it is
about to predict.

WHAT MAKES THIS HARD is not the weights. It is the optimiser. Adam keeps two
moments per parameter, and those belong to the EXPERT, not to the slot of VRAM
it happens to occupy. Swapping an expert out without its moments would hand its
history to whichever expert took its place - the model would keep training, and
every swapped expert would carry a stranger's momentum. So `swap_to` moves
moments with weights, and the optimiser is told about the slots rather than
about the experts.
"""

import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .precision import is_moment, pack_bf16, unpack_bf16


class Tiers:
    """
    Disk, RAM and VRAM, with an LRU between them.

    Disk is the whole pool, one .npy per expert - the same files the weights
    directory already holds, so nothing new is stored. RAM keeps the most
    recently used `ram_capacity` of them as CPU tensors, so an expert wanted
    again soon costs a copy rather than a read. Least recently used means
    exactly that: when the cache is full, the expert untouched longest is the
    one dropped.

    An expert that was trained while resident is marked dirty and written back
    to disk on eviction. Otherwise the file on disk is still the truth.
    """

    def __init__(self, path, d_model, d_ff, ram_capacity=256, device="cpu",
                 read_only=False, write_path=None):
        self.path = path
        self.write_path = write_path or path
        self.d_model, self.d_ff = d_model, d_ff
        self.ram = OrderedDict()             # id -> dict of CPU tensors
        self.ram_capacity = ram_capacity
        self.device = device
        # Read-only means exactly that: nothing is ever marked dirty and
        # nothing is ever written. Needed because paging an expert IN marks it
        # dirty regardless of whether anything changed it, so a visualisation
        # reading a live training directory would otherwise write expert files
        # back underneath the run that owns them.
        self.read_only = read_only
        self.dirty = set()
        self.reads = self.hits = self.evictions = self.writebacks = 0

    def _file(self, i):
        name = "e%05d.npz" % i
        if self.write_path != self.path:
            changed = os.path.join(self.write_path, name)
            if os.path.exists(changed):
                return changed
        return os.path.join(self.path, name)

    def _write_file(self, i):
        return os.path.join(self.write_path, "e%05d.npz" % i)

    def delete(self, i):
        """Delete an expert only from the writable layer."""
        if self.read_only:
            raise RuntimeError("read-only pool tried to delete e%05d.npz" % i)
        f = self._write_file(i)
        if os.path.exists(f):
            os.remove(f)
        self.ram.pop(i, None)
        self.dirty.discard(i)

    def _from_disk(self, i):
        """
        One expert, complete: its weights and its optimiser moments.

        The moments live in the same file as the weights because they belong
        to the same thing. Adam's history for an expert is as much a part of
        that expert as its weights are - an expert paged out and back without
        its moments resumes with a stranger's momentum, and one written to disk
        without them cannot be resumed at all. Keeping them together makes the
        file the whole of what that expert is, which is what the weights
        directory claims about itself.
        """
        z = np.load(self._file(i))
        out = {k: torch.from_numpy(z[k]).clone() for k in ("w1", "w3", "w2")}
        for k in ("w1_m", "w3_m", "w2_m", "w1_v", "w3_v", "w2_v"):
            if k in z.files:
                # bf16 if this file has been written since the moments were
                # narrowed, fp32 if it has not; unpack_bf16 reads both
                out[k] = unpack_bf16(z[k]).clone()
        return out

    def fetch(self, i):
        """The expert's CPU tensors, from RAM if it is there."""
        if i in self.ram:
            self.hits += 1
            self.ram.move_to_end(i)
            return self.ram[i]
        self.reads += 1
        e = self._from_disk(i)
        self.ram[i] = e
        self.ram.move_to_end(i)
        self._trim()
        return e

    def put(self, i, tensors, dirty=True):
        """Hand an expert back after it has been resident."""
        self.ram[i] = tensors
        self.ram.move_to_end(i)
        if dirty and not self.read_only:
            self.dirty.add(i)
        self._trim()

    def _trim(self):
        while len(self.ram) > self.ram_capacity:
            j, ent = self.ram.popitem(last=False)      # least recently used
            self.evictions += 1
            if j in self.dirty:
                self._to_disk(j, ent)
                self.dirty.discard(j)

    def _to_disk(self, i, ent):
        if self.read_only:
            raise RuntimeError("read-only pool tried to write e%05d.npz" % i)
        # Weights fp32, Adam's moments bf16. The moments are two thirds of the
        # file and cost a thousandth of their own magnitude to narrow, because
        # what they need is exponent range and bf16 keeps all of fp32's. The
        # weights cannot go with them - see minagi/precision.py, which has the
        # measurement for both.
        arrays = {k: (pack_bf16(v) if is_moment(k) else v.to(torch.float32).numpy())
                  for k, v in ent.items() if torch.is_tensor(v)}
        os.makedirs(self.write_path, exist_ok=True)
        target = self._write_file(i)
        tmp = target + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, target)
        self.writebacks += 1

    def flush(self):
        if self.read_only:
            return
        for i in list(self.dirty):
            self._to_disk(i, self.ram[i])
        self.dirty.clear()

    def report(self):
        tot = self.reads + self.hits
        return {"ram_held": len(self.ram), "ram_capacity": self.ram_capacity,
                "disk_reads": self.reads, "ram_hits": self.hits,
                "hit_rate": self.hits / max(tot, 1),
                "evictions": self.evictions, "writebacks": self.writebacks}


class PagedPool(nn.Module):
    """
    A pool whose VRAM cost is set by the working set, not by its size.

    Presents enough of SharedPool's surface that PooledMLP routes into it
    unchanged: `gate`, `n_experts()`, `stacked()`, and the usage buffers. What
    differs is that only `resident` experts exist as CUDA parameters at any
    moment, and `stacked()` returns those.

    Routing indices are into the RESIDENT set, so a call site picking expert 3
    means the third of the experts currently held, not the third on disk.
    `self.slots` maps back.
    """

    def __init__(self, path, d_model, d_ff, n_experts, resident=16,
                 ram_capacity=256, device="cpu", max_experts=1_000_000,
                 read_only=False, write_path=None):
        super().__init__()
        self.path = path
        self.d_model, self.d_ff = d_model, d_ff
        self.resident = min(resident, n_experts)
        self.max_experts = max_experts
        self._n = n_experts
        self.read_only = read_only
        self.tiers = Tiers(path, d_model, d_ff, ram_capacity, device,
                           read_only=read_only, write_path=write_path)

        # the VRAM slots - fixed in number, their contents swapped
        self.w1 = nn.Parameter(torch.zeros(self.resident, d_ff, d_model))
        self.w3 = nn.Parameter(torch.zeros(self.resident, d_ff, d_model))
        self.w2 = nn.Parameter(torch.zeros(self.resident, d_model, d_ff))
        self.slots = [-1] * self.resident          # slot -> expert id

        # gates are one float per expert, so the whole pool's worth is nothing
        self.gate = nn.Parameter(torch.ones(n_experts))
        for name in ("use", "age", "born", "gate_seen"):
            self.register_buffer(name, torch.zeros(n_experts),
                                 persistent=False)

        # the segment router: which experts to hold for the next stretch
        self.segment_router = nn.Linear(d_model, max(n_experts, 1), bias=False)
        nn.init.normal_(self.segment_router.weight, 0.0, 0.02)
        self.register_buffer("summary", torch.zeros(d_model), persistent=False)

        self.grow_events = []
        self.pressure = 0.0
        self.want_k = 0.0
        self.swaps = 0
        self._opt = None
        self._sites = []          # the call sites routing into this pool
        # how long ago each expert was last resident, counted in segments. An
        # expert that is never chosen never trains, never earns a gate, and is
        # eventually pruned for it - so without this the pool would collapse to
        # whichever experts the untrained router happened to favour first. It
        # is what the exploration reserve in choose_by_demand() sorts on.
        self.register_buffer("last_seen", torch.zeros(n_experts),
                             persistent=False)
        self.segments = 0
        self.explore = 0.25              # kept for choose(); see choose_by_demand
        # how much better a candidate must be before it displaces a resident,
        # and how long a newcomer is safe from being displaced itself. Together
        # they are what stops the working set churning on noise.
        self.margin = 0.10
        self.dwell = 4
        self.register_buffer("ever", torch.zeros(n_experts, dtype=torch.bool),
                             persistent=False)
        # What each expert responds to, derived from its own weights.
        #
        # A router row only receives gradient while its expert is resident, so
        # asking a learned router "which experts do you want" can only ever
        # name the experts you already have - the answer is self-reinforcing.
        # A key is not like that: it is a property of the expert, defined
        # whether or not it has ever been loaded, and it moves when the expert
        # learns rather than when the scheduler happens to pick it.
        #
        # The key is the dominant right singular vector of w1 - the input
        # direction the expert's first layer reads most strongly, which is as
        # close to "what this expert is for" as its own weights can say.
        self.register_buffer("keys", torch.zeros(n_experts, d_model),
                             persistent=False)
        # when an expert was ADMITTED, which is not when it was last seen:
        # a resident is seen every chunk it stays, so measuring dwell against
        # last_seen makes every resident permanently too young to evict and
        # the working set can never move at all
        self.register_buffer("since", torch.zeros(n_experts),
                             persistent=False)
        # How many times each expert has been brought onto the card, over the
        # model's whole life rather than this session. `use` counts routing
        # hits and `admits` counts admissions: an expert can sit resident for
        # a long stretch and be heavily routed while being admitted once.
        self.register_buffer("admits", torch.zeros(n_experts),
                             persistent=False)
        # An expert's NAME is its uid, not its position. Position changes
        # every time something is pruned; a uid never does. That is what keeps
        # a file where it is across a prune, and what lets a record written
        # a million characters ago still mean the same expert.
        self.register_buffer("uid", torch.arange(n_experts, dtype=torch.long),
                             persistent=False)
        self.next_uid = int(n_experts)
        self.key_weight = 1.0
        # how much an expert that has earned nothing may still be wanted for
        # its fit alone - low, but not zero, or growth could never take
        self.key_floor = 0.1
        # A new expert cannot lift its gate without being chosen, and would not
        # be chosen because its gate is low - a loop that deletes every new
        # expert and leaves only the originals. So a new expert competes
        # without the handicap for its first `trial` steps, which is the same
        # window prune gives it before it may be deleted, and is then judged on
        # what it did with a fair turn rather than on never having had one.
        # `now` is the training step, set by the reader; without it nothing is
        # ever on trial, which is the safe default.
        self.trial = 0
        self.now = 0
        # share of the survival window unaddressed that counts as dying
        self.dying_at = 0.75
        self._keys_built = False
        self._h_keep = None

    def _f(self, i):
        """Position in the pool's arrays -> the id its file is named by."""
        return int(self.uid[int(i)])

    def dying(self):
        """
        How close each expert is to being deleted, as a share of the window.

        0.0 means something asked for it just now. 1.0 means it has gone
        exactly as long unaddressed as `survival`, which is the line prune
        deletes on. Above 1.0 it is only alive because prune has not run yet.

        DEAD MEANS UNADDRESSED, and this is the one definition prune acts on.
        It is deliberately not a gate question. The gate says how loudly an
        expert speaks when chosen, and the smallest gates belong to the
        BUSIEST experts - a sink that is picked constantly and contributes
        little per character reads as dead on any gate test, while a rarely
        chosen expert with a large gate reads as alive. Time since anything
        last wanted it is the quantity that actually predicts being wanted
        again.

        A newborn inside its trial returns 0. It has not failed to be chosen;
        it has not finished being offered.
        """
        n = self._n
        z = torch.zeros(n, device=self.gate.device)
        survival = float(getattr(self, "trial", 0) or 0)
        step = float(getattr(self, "now", 0) or 0)
        if n == 0 or survival <= 0 or step <= 0:
            return z
        # the same steps-to-segments conversion prune uses, for the same reason
        per_step = self.segments / max(step, 1.0)
        window = survival * per_step
        if window <= 0:
            return z
        # An expert cannot have gone unaddressed for longer than it has
        # existed. `last_seen` starts at 0 for a newborn - deliberately, so it
        # sorts to the front of the exploration queue - but read here as "how
        # long since anything wanted it" that same 0 would mean "since the
        # beginning of the run". Clamping to the expert's own age is what stops
        # a newborn scoring past the prune line from the moment it exists.
        born_seg = (step - self.born[:n].to(z.dtype)).clamp_min(0.0) * per_step
        idle = (self.segments - self.last_seen[:n].to(z.dtype)).clamp_min(0.0)
        frac = torch.minimum(idle, born_seg) / window
        young = (step - self.born[:n].to(z.dtype)) < survival
        return torch.where(young, z, frac)


    def _carry(self, opt, old_p, new_p, idx=None, grow=0):
        """
        Move Adam's moments from a replaced parameter onto its replacement.

        Growing or pruning the pool builds a NEW gate tensor and NEW router
        rows. Without this the optimiser would drop the state attached to the
        old ones, and the machinery that decides WHICH expert to use would
        restart its moment estimates at every growth decision. Pruning slices
        rows out, growth appends zeroed ones, and the moments follow the same
        shape either way.
        """
        if opt is None:
            return
        st = opt.state.pop(old_p, None)
        if not st:
            return
        out = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() and v.shape[0] == old_p.shape[0]:
                if idx is not None:
                    v = v[idx].clone()
                elif grow:
                    pad = torch.zeros((grow,) + tuple(v.shape[1:]),
                                      dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad])
            out[k] = v
        opt.state[new_p] = out

    # -- the surface PooledMLP expects ------------------------------------
    def n_experts(self):
        return self._n

    def n_routable(self):
        """A token chooses only between the experts in VRAM."""
        return self.resident

    def router_rows(self):
        """One row per expert on disk; growth adds rows."""
        return self._n

    def resident_rows(self):
        """Which router rows the resident experts own, in slot order."""
        return torch.tensor([max(s, 0) for s in self.slots],
                            device=self.gate.device)

    def n_resident(self):
        return self.resident

    def routable_gate(self):
        """Gates of the resident experts, in slot order."""
        idx = torch.tensor([max(s, 0) for s in self.slots],
                           device=self.gate.device)
        return self.gate[idx]

    def note_use(self, hit):
        """Routing counts arrive per SLOT; usage is kept per EXPERT."""
        idx = torch.tensor([max(s, 0) for s in self.slots],
                           device=self.use.device)
        self.use.index_add_(0, idx, hit.to(self.use.dtype))
        self.age += 1

    def n_params(self):
        per = 3 * self.d_model * self.d_ff
        return self._n * per + self.gate.numel() + self.segment_router.weight.numel()

    def disk_bytes(self, extra=0):
        """
        What the pool costs on disk, and what `extra` more experts would cost.

        EIGHT bytes per parameter, which is what the files actually hold: one
        fp32 weight (4) plus Adam's two moments as bf16 (2 + 2). At d_model 512
        and d_ff 2048 that is 25.2 MB an expert.

        `growth.max_disk_gb` is enforced against this number, so it has to
        match the files rather than the widest possible layout - an estimate
        that assumed fp32 moments would run 1.5x over and stop growth well
        short of the budget it was given.
        """
        per = 3 * self.d_model * self.d_ff * 8
        return (self._n + int(extra)) * per

    def vram_params(self):
        return (3 * self.resident * self.d_model * self.d_ff
                + self.gate.numel() + self.segment_router.weight.numel())


    def stacked(self):
        return [(self.w1, self.w3, self.w2)]

    def invalidate(self):
        pass

    # -- choosing and swapping the working set -----------------------------
    @staticmethod
    @torch.no_grad()
    def _key_of(w1, iters=4):
        """Dominant input direction of an expert, by power iteration."""
        v = w1.sum(0)
        if not torch.isfinite(v).all() or float(v.norm()) < 1e-9:
            v = torch.randn(w1.shape[1], device=w1.device, dtype=w1.dtype)
        v = v / v.norm().clamp_min(1e-9)
        for _ in range(iters):
            v = w1.t() @ (w1 @ v)
            v = v / v.norm().clamp_min(1e-9)
        return v

    @torch.no_grad()
    def build_keys(self, verbose=False):
        """
        Describe every expert on disk, once, at startup.

        Only w1 is read from each file - the rest of the expert is not needed
        to say what it responds to - so this costs a fraction of loading the
        pool, and it is what lets an expert that has never been resident be
        asked for at all.
        """
        import numpy as _np
        done = 0
        for i in range(self._n):
            f = self.tiers._file(self._f(i))
            if not os.path.exists(f):
                continue
            ent = self.tiers.ram.get(self._f(i))
            if ent is not None:
                w1 = ent["w1"]
                self.keys[i] = self._key_of(w1.float().to(self.keys.device))
            else:
                # close the archive on every iteration. Left to the garbage
                # collector, one open member per expert is held at once and
                # describing the pool costs more memory than loading it.
                with _np.load(f) as z:
                    w1 = torch.from_numpy(_np.ascontiguousarray(z["w1"]))
                self.keys[i] = self._key_of(w1.float().to(self.keys.device))
                del w1
            done += 1
        self._keys_built = True
        if verbose:
            print(f"  described {done} experts by what they respond to")
        return done

    @torch.no_grad()
    def refresh_keys(self):
        """Recompute keys for whichever experts are on the card right now."""
        for slot, e in enumerate(self.slots):
            if 0 <= e < self.keys.shape[0]:
                self.keys[e] = self._key_of(self.w1.data[slot].float())

    def arm_observation(self):
        """Start collecting the states the call sites route on."""
        self._h_keep = []

    def demand(self, x, sites, top_k):
        """
        Which experts the text wants, scored over the WHOLE pool.

        The states matter more than anything else here. Scoring the routers on
        raw character embeddings looks reasonable and is useless: an embedding
        carries no context, so every subject asks for very nearly the same
        experts. What a character means depends on what surrounds it, and only
        the hidden state knows that.

        So demand is scored on the states the call sites really routed on
        while reading the previous chunk, sampled as they went. They are one
        chunk stale, which is the price of knowing anything at all: what the
        next chunk wants cannot be known without computing it, and text is
        locally coherent enough that what the last few hundred characters
        needed is a fair guess at what the next few hundred will.
        """
        if not self._keys_built:
            # built on first use, not at load: a tool that never routes - the
            # film's captures, an inspector - should not pay for reading every
            # expert on disk just to open the model
            self.build_keys()
        want = torch.zeros(self._n, device=self.gate.device)
        k = min(top_k, self._n)
        seen = getattr(self, "_h_keep", None)
        if seen:
            pairs = seen
        else:
            # nothing has been read yet, so the embedding is all there is
            flat = x.reshape(-1, x.shape[-1])
            pairs = [(s, flat) for s in sites]
        for s, h in pairs:
            w = s.router.weight[:self._n]
            lg = F.linear(h.to(w.dtype) + s.depth_emb.to(w.dtype), w).float()
            top = torch.topk(torch.softmax(lg, -1), k, dim=-1)
            want.index_add_(0, top.indices.reshape(-1),
                            top.values.reshape(-1).to(want.dtype))
        self._h_keep = []                       # consumed

        # The router can only speak for the experts it has been trained on,
        # which are the ones already resident. The keys speak for all of them,
        # because a key belongs to the expert rather than to the scheduler.
        if self.key_weight and float(self.keys.abs().sum()) > 0:
            hs = torch.cat([h.float() for _, h in pairs]) if pairs else None
            if hs is not None and hs.numel():
                hs = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                kk = self.keys[:self._n]
                kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                sim = torch.softmax((hs @ kk.t()).abs() * 8.0, dim=-1)
                # What an expert has earned scales the similarity BEFORE the
                # per-character choice, not the tally afterwards: discounting
                # the tally would still let an untrained expert win the choice
                # and only then be marked down, and a single strong random
                # match would outweigh the penalty.
                g = self.gate.detach().abs()[:self._n]
                earned = (g / g.max().clamp_min(1e-9)).clamp_min(self.key_floor)
                # ...except while an expert is on trial. See __init__: an
                # expert born into this handicap can never escape it, because
                # the gate it is judged on only moves when it is chosen. Born
                # experts carry born > 0; the originals carry 0 and are never
                # young, so a fresh pool waives nothing.
                if self.trial > 0:
                    born = self.born[:self._n]
                    young = (born > 0) & ((self.now - born) < self.trial)
                    earned = torch.where(young, torch.ones_like(earned), earned)
                sim = sim * earned
                top = torch.topk(sim, k, dim=-1)
                fit = torch.zeros_like(want)
                fit.index_add_(0, top.indices.reshape(-1),
                               top.values.reshape(-1).to(fit.dtype))
                # Fit is not enough on its own. A brand new expert's key comes
                # from its random w1, which matches text about as well as
                # anything does, so fit alone elects the untrained - and an
                # expert gated to nothing contributes nothing while still
                # taking one of the top-k slots at that depth.
                #
                # So an expert is wanted in proportion to how well it fits AND
                # how much it has shown it can contribute. `key_floor` is what
                # keeps a new expert reachable at all; the cold-start sweep in
                # choose_by_demand() is what gives it its first turn.
                #
                # Each term is normalised on its own, so that where the router
                # has no opinion the key term still speaks.
                want = (want / want.sum().clamp_min(1e-9)
                        + self.key_weight * fit / fit.sum().clamp_min(1e-9))
        # an expert that has earned a gate is worth keeping resident over one
        # that has not, when demand is otherwise equal
        return want + 0.05 * self.gate.detach().abs()

    @torch.no_grad()
    def choose_by_demand(self, want):
        """
        Which experts should be resident, decided by need rather than a clock.

        Nothing moves unless something asks for it. A candidate displaces a
        resident only when it is wanted enough to beat it by `margin`, and a
        resident that has only just arrived cannot be thrown out again before
        it has had `dwell` chunks to be useful. On text that has not changed,
        this settles to no movement at all; when the subject turns over, a
        burst of admissions happens because the demand really did change.

        The one thing that is not demand-driven is the cold start, and it is
        deliberately finite. An expert that has never been resident has never
        trained, so its router row is noise and it can never be wanted - the
        pool would collapse to whatever an untrained router liked first, and
        most of the experts on disk would never be loaded even once.
        So while any expert has never been resident, exactly one is
        admitted per chunk, in index order. Once every expert has had its
        turn the sweep stops for good, and growth re-arms it only for the
        experts it just created. No randomness, and no permanent tax.
        """
        k = min(self.n_routable(), self._n)
        if self.ever.numel() < self._n:                 # grown since last time
            self.ever = torch.cat([
                self.ever, torch.zeros(self._n - self.ever.numel(),
                                       dtype=torch.bool, device=self.ever.device)])
        cur = [e for e in self.slots if e >= 0]
        if len(cur) < k:                                # cold card: fill it
            order = torch.argsort(want, descending=True).tolist()
            cur = (cur + [e for e in order if e not in cur])[:k]
            return cur

        held = set(cur)
        score = want.clone()
        # a resident that has just arrived is not up for eviction yet
        young = [e for e in cur
                 if self.segments - float(self.since[e]) < self.dwell]
        evictable = [e for e in cur if e not in young]
        cands = [int(e) for e in torch.argsort(want, descending=True).tolist()
                 if e not in held]

        plan = list(cur)
        for cand in cands:
            if not evictable:
                break
            weakest = min(evictable, key=lambda e: float(score[e]))
            if float(score[cand]) <= float(score[weakest]) * (1.0 + self.margin):
                break                                   # nothing wants in
            plan[plan.index(weakest)] = cand
            evictable.remove(weakest)

        # the terminating cold-start sweep: one never-resident expert, in
        # index order, into the least wanted slot that may be evicted
        never = (~self.ever).nonzero().flatten().tolist()
        never = [e for e in never if e not in plan]
        if never and evictable:
            weakest = min(evictable, key=lambda e: float(score[e]))
            plan[plan.index(weakest)] = never[0]
        return plan[:k]

    @torch.no_grad()
    def choose(self, summary=None):
        """
        Which experts to hold next, scored on the segment just finished.

        Most of the working set is what the router asks for. A share of it -
        `explore` - goes to the experts that have gone longest without being
        resident, whatever the router thinks of them.

        That reservation is not a nicety. Selection here is self-reinforcing:
        an expert that is not chosen does not train, so its gate does not
        rise, so it is chosen even less, and the age rule eventually deletes
        it. Without exploration the pool shrinks to whichever experts an
        untrained router happened to prefer on the first segment, and the rest
        of the disk is dead weight.
        """
        s = self.summary if summary is None else summary
        logits = self.segment_router(s)[:self._n].clone()
        logits = logits + 0.5 * self.gate.detach().abs().log1p()
        k = min(self.n_routable(), self._n)
        n_explore = min(int(k * self.explore), max(self._n - k, 0))
        chosen = torch.topk(logits, k - n_explore).indices.tolist()
        if n_explore:
            stale = self.last_seen.clone()
            stale[torch.tensor(chosen, device=stale.device)] = float("inf")
            # longest unseen first; ties broken by index, which is stable
            order = torch.argsort(stale)[:n_explore].tolist()
            chosen = chosen + order
        return chosen[:k]

    @torch.no_grad()
    def swap_to(self, ids):
        """
        Make `ids` the resident set, moving optimiser state with the weights.

        Adam's moments belong to the expert. Leaving them behind would hand an
        expert's momentum to whatever took its slot, and training would carry
        on looking healthy while every swapped expert inherited a stranger's
        history.
        """
        ids = list(dict.fromkeys(int(i) for i in ids))[:self.resident]
        while len(ids) < self.resident:
            for cand in range(self._n):
                if cand not in ids:
                    ids.append(cand)
                    break
            else:
                break
        self.segments += 1
        for i in ids:                       # newly admitted start their dwell
            if 0 <= i < self.since.numel() and i not in self.slots:
                self.since[i] = self.segments
                self.admits[i] += 1
        if self.segments % 8 == 0:
            # a resident expert is still learning, so its description drifts
            self.refresh_keys()
        for i in ids:
            if 0 <= i < self.last_seen.numel():
                self.last_seen[i] = self.segments
                self.ever[i] = True

        # An expert already on the card stays in the slot it is in, and only
        # what is genuinely new is fetched from the host.
        #
        # Assigning by position instead - slot j simply gets ids[j] - would
        # make the same resident set arriving in a different order count as a
        # full reload. Demand comes back sorted, so the order churns while the
        # set itself barely moves; matching by identity is what keeps the load
        # count equal to how much the set really changed.
        here = {e: s for s, e in enumerate(self.slots) if e >= 0}
        plan = [-1] * self.resident
        kept = set()
        for e in ids:
            if e in here:
                plan[here[e]] = e                     # stays where it is
                kept.add(here[e])
        free = [s for s in range(self.resident) if s not in kept]
        for e in ids:
            if e not in here:
                plan[free.pop(0)] = e                 # fills a vacated slot
        if plan == self.slots:
            return 0
        loads = sum(1 for e in plan if e >= 0 and e not in here)
        self._rearrange(plan, here)
        self.swaps += 1
        return loads

    @torch.no_grad()
    def _rearrange(self, plan, here):
        """
        Put the card into the state `plan` describes: park what is leaving,
        fetch what is arriving, and leave everything else where it is.

        Adam's moments travel with the expert in both directions, because they
        belong to the expert and not to the slot it occupied.
        """
        st = (self._opt.state if self._opt is not None else {})
        dev = self.w1.device
        tensors = ((self.w1, "w1"), (self.w3, "w3"), (self.w2, "w2"))
        old = list(self.slots)

        for e in old:
            if e >= 0 and e not in plan:
                s = here[e]
                if e < self.keys.shape[0]:
                    # its weights are final for this stay, so this is the
                    # moment its description is most accurate
                    self.keys[e] = self._key_of(self.w1.data[s].float())
                ent = {nm: p.data[s].detach().to("cpu").clone()
                       for p, nm in tensors}
                for p, nm in tensors:
                    o = st.get(p)
                    if o and "exp_avg" in o:
                        ent[nm + "_m"] = o["exp_avg"][s].to("cpu").clone()
                        ent[nm + "_v"] = o["exp_avg_sq"][s].to("cpu").clone()
                self.tiers.put(self._f(e), ent, dirty=True)

        for s, e in enumerate(plan):
            if e < 0 or old[s] == e:
                continue
            src = self.tiers.fetch(self._f(e))
            src = {k: (v.to(dev) if torch.is_tensor(v) else v)
                   for k, v in src.items()}
            for p, nm in tensors:
                p.data[s] = src[nm]
                o = st.get(p)
                if o and "exp_avg" in o:
                    if nm + "_m" in src:
                        o["exp_avg"][s] = src[nm + "_m"]
                        o["exp_avg_sq"][s] = src[nm + "_v"]
                    else:
                        o["exp_avg"][s].zero_()
                        o["exp_avg_sq"][s].zero_()
        self.slots = list(plan)

    # -- growing and pruning ----------------------------------------------
    @torch.no_grad()
    def add_experts(self, k, seed_from=None, device=None, step=0,
                    birth_gate=0.001, make=None, recombine=16):
        """
        Write k new experts to disk and make them part of the pool.

        They are not loaded. A new expert only reaches VRAM if the segment
        router chooses it. Growth is therefore cheap in a way it never was
        while every expert had to be resident: adding a hundred experts costs
        a hundred files and not one megabyte of card.

        HOW ONE IS BUILT is RECOMBINATION, by default from `recombine` other
        experts. A hidden unit is the triple (w1[u], w3[u], w2[:, u]) and units
        are interchangeable, so a child assembled from whole units taken from
        different parents keeps every unit's learned feature intact while
        computing a function nothing in the pool computes.

        The two obvious alternatives both fail, in opposite directions. A clone
        plus noise is not novel - its output sits at ~0.98 cosine to its parent
        where two established experts sit at ~0.04 - so it only double-counts
        what the pool already has. A fresh random expert is novel but computes
        nothing worth routing to. What works is novelty built from TRAINED
        parts. The comparison is in `runs/results/birth_schemes.json`.

        A new expert is born at `birth_gate`, a small starting scale rather
        than a verdict: prune reads staleness, not the gate, so what keeps a
        newborn alive is its trial window and then being chosen.

        The per-character router does not change size - it addresses slots,
        not experts. Only the segment router gains a row each.
        """
        dev = self.gate.device
        src = None
        if seed_from is not None and 0 <= int(seed_from) < self._n:
            src = self.tiers.fetch(self._f(int(seed_from)))

        # The sources recombination draws from: a sample of the pool, with the
        # expert the caller asked to split kept among them. Fetched once and
        # reused for every child in this batch.
        srcs = None
        if make is None and recombine and recombine >= 2 and self._n >= 2:
            m = min(int(recombine), self._n)
            pick = torch.randperm(self._n)[:m].tolist()
            if (seed_from is not None and 0 <= int(seed_from) < self._n
                    and int(seed_from) not in pick):
                pick[0] = int(seed_from)
            got = [self.tiers.fetch(self._f(int(i))) for i in pick]
            srcs = tuple(torch.stack([g[nm].float() for g in got])
                         for nm in ("w1", "w3", "w2"))
        new_uids = []
        for _ in range(k):
            i = self._n
            nu = self.next_uid
            self.next_uid += 1
            if make is not None:
                # a birth scheme supplied by the caller, for comparing what a
                # newborn should BE against the default. See
                # tools/birth_probe.py.
                ent = {nm: t.detach().clone()
                       for nm, t in zip(("w1", "w3", "w2"), make(i, src))}
            elif srcs is not None:
                # RECOMBINATION. A hidden unit is (w1[u], w3[u], w2[:, u]) and
                # units are interchangeable, so taking whole units from
                # different experts keeps every unit's learned feature intact
                # while composing a function nothing in the pool computes.
                who = torch.randint(0, srcs[0].shape[0], (self.d_ff,))
                unit = torch.arange(self.d_ff)
                ent = {"w1": srcs[0][who, unit].clone(),
                       "w3": srcs[1][who, unit].clone(),
                       "w2": srcs[2][who, :, unit].t().contiguous().clone()}
            elif src is not None:
                ent = {nm: (src[nm] + 0.02 * torch.randn_like(src[nm])).clone()
                       for nm in ("w1", "w3", "w2")}
            else:
                ent = {"w1": torch.randn(self.d_ff, self.d_model) * 0.02,
                       "w3": torch.randn(self.d_ff, self.d_model) * 0.02,
                       "w2": torch.randn(self.d_model, self.d_ff) * 0.02}
            self.tiers.put(nu, ent, dirty=True)
            new_uids.append(nu)
            self._n += 1
        self.tiers.flush()
        self.uid = torch.cat([self.uid,
                              torch.tensor(new_uids, dtype=torch.long,
                                           device=self.uid.device)])

        def grow_vec(t, fill=0.0):
            return torch.cat([t, torch.full((k,), float(fill),
                                            device=t.device, dtype=t.dtype)])
        opt = getattr(self, "_opt", None)
        old_gate = self.gate
        self.gate = nn.Parameter(grow_vec(self.gate.data, birth_gate))
        self._carry(opt, old_gate, self.gate, grow=k)
        # a new expert has never been seen, so it goes to the front of the
        # exploration queue rather than the back
        self.last_seen = grow_vec(self.last_seen, 0.0)
        self.use = grow_vec(self.use)
        self.age = grow_vec(self.age)
        self.born = grow_vec(self.born, step)
        self.gate_seen = grow_vec(self.gate_seen)
        # a new expert has never been resident, which re-arms the cold-start
        # sweep for it alone - it will get its turn on the card, in order,
        # and then the sweep goes quiet again
        self.ever = torch.cat([self.ever, torch.zeros(k, dtype=torch.bool,
                                                      device=self.ever.device)])
        self.keys = torch.cat([self.keys, torch.zeros(k, self.d_model,
                                                      device=self.keys.device,
                                                      dtype=self.keys.dtype)])
        self.since = grow_vec(self.since, 0.0)
        self.admits = grow_vec(self.admits, 0.0)

        w = self.segment_router.weight.data
        if w.shape[0] < self._n:
            extra = torch.randn(self._n - w.shape[0], self.d_model,
                                device=w.device) * 0.02
            old_sr = self.segment_router.weight
            self.segment_router = nn.Linear(self.d_model, self._n,
                                            bias=False).to(dev)
            self.segment_router.weight.data = torch.cat([w, extra])
            self._carry(opt, old_sr, self.segment_router.weight,
                        grow=self._n - w.shape[0])
        # the per-token routers also keep one row per expert, so they grow too
        for site in self._sites:
            rw = site.router.weight.data
            if rw.shape[0] < self._n:
                extra = torch.randn(self._n - rw.shape[0], self.d_model,
                                    device=rw.device) * 0.01
                old_r = site.router.weight
                site.router = nn.Linear(self.d_model, self._n,
                                        bias=False).to(rw.device)
                site.router.weight.data = torch.cat([rw, extra])
                self._carry(opt, old_r, site.router.weight,
                            grow=self._n - rw.shape[0])
        return self._n

    @torch.no_grad()
    def prune(self, step, survival=8600, protect=0):
        """
        Delete experts that have atrophied. Their files go with them.

        USE IT OR LOSE IT, and USE IS THE ONLY TEST. An expert is removed when
        it has not been admitted to the card once in THE LAST `survival` steps,
        and it is never touched at all in its first `survival` steps.

          stale    a trailing window, not a one-off check at birth: an expert
                   that worked for 200,000 steps and has since gone quiet dies
                   too. Age earns nothing permanent. Not "its gate is low",
                   not "its gate stopped rising" - it was not chosen.

        THERE IS NO GATE TERM HERE, deliberately. The gate is not merely
        uninformative about whether an expert will be wanted again, it is
        anti-predictive: the smallest gates belong to the busiest experts. One
        that behaves as a sink - chosen constantly, contributing little per
        character - reads as dead on a gate test, while a high-gate expert
        nothing has asked for in hundreds of thousands of segments reads as
        alive. A gate test would delete the first and spare the second.

        The growth brake reads the SAME staleness, through dying(): an expert
        is dying once it has gone `dying_at` of the way to this window without
        being addressed. One definition, two thresholds - one that stops
        growth, one that deletes.

        A newborn is safe for `survival` steps no matter what, so it cannot be
        judged before it has had a chance to be chosen; the exploration reserve
        in choose() reaches every expert well inside that window.

        THE COST OF THIS TRADE is that an expert which is genuinely rare rather
        than dead is deleted, and deletion is permanent. `survival` is the only
        thing holding that, so it should be set wide.
        """
        # `last_seen` counts SEGMENTS and `survival` is in steps, so the window
        # is converted with the pool's own cumulative ratio rather than a
        # constant - it is self-calibrating and needs nothing stored.
        per_step = self.segments / max(float(step), 1.0)
        window = survival * per_step if per_step > 0 else float("inf")
        keep = []
        for i in range(self._n):
            if i < protect or i in self.slots:
                keep.append(i)                 # never drop what is resident
                continue
            young = (step - float(self.born[i])) < survival
            seen = (self.segments - float(self.last_seen[i])) <= window
            if young or seen:
                keep.append(i)
        if len(keep) == self._n:
            return 0
        if not keep:
            # Everything qualified and nothing was resident to protect it. A
            # pool of zero experts cannot route, so one stays whatever the
            # thresholds say - a rule that never fires in a healthy run and
            # stops a collapsed one from destroying itself. It is the most
            # recently wanted one, not the best-gated one: the gate is no
            # longer what this rule is about, and the last expert anything
            # asked for is the least bad thing to be left holding.
            keep = [int(self.last_seen[:self._n].argmax())]
        gone = self._n - len(keep)
        kept = set(keep)

        # Files are named by uid, so NOTHING is renamed. A prune that
        # renumbered the directory would cost up to n renames for one deletion,
        # and every record ever written by position - expert_history.jsonl
        # included - would silently stop meaning what it said.
        for i in range(self._n):
            if i not in kept:
                u = self._f(i)
                self.tiers.delete(u)

        idx = torch.tensor(keep, dtype=torch.long, device=self.gate.device)
        opt = getattr(self, "_opt", None)
        old_gate = self.gate
        self.gate = nn.Parameter(self.gate.data[idx].clone())
        self._carry(opt, old_gate, self.gate, idx=idx)
        for nm in ("use", "age", "born", "gate_seen", "last_seen", "ever",
                   "since", "admits", "uid"):
            setattr(self, nm, getattr(self, nm)[idx].clone())
        self.keys = self.keys[idx].clone()
        # Give every router a NEW parameter rather than reshaping the one it
        # has. Autograd sizes a gradient from the tensor it saved, so shrinking
        # a Parameter in place under a graph that still references it makes the
        # backward return the new number of rows where the graph recorded the
        # old one - which is a crash, and only in a run that prunes. Growth
        # never hit it because add_experts already allocates a fresh Linear.
        old_sr = self.segment_router.weight
        sr = nn.Linear(self.d_model, len(keep), bias=False).to(
            old_sr.device)
        sr.weight.data = old_sr.data[idx].clone()
        self.segment_router = sr
        self._carry(opt, old_sr, sr.weight, idx=idx)
        for site in self._sites:
            w = site.router.weight
            fresh = nn.Linear(w.shape[1], len(keep), bias=False).to(w.device)
            fresh.weight.data = w.data[idx].clone()
            site.router = fresh
            self._carry(opt, w, fresh.weight, idx=idx)
        remap = {old_i: new_i for new_i, old_i in enumerate(keep)}
        self.slots = [remap.get(s, -1) for s in self.slots]
        self._n = len(keep)
        return gone

    def saturation(self):
        """
        What the growth brakes read. Idle is a STALENESS question - see
        dying(). It is deliberately not measured on routing traffic: the
        load-balancing loss makes routing near-uniform by design, so anything
        counted from traffic describes that objective rather than the pool.
        """
        u = self.use / self.use.sum().clamp_min(1)
        n = max(u.numel(), 1)
        ideal = 1.0 / n
        ent = float(-(u.clamp_min(1e-9) * u.clamp_min(1e-9).log()).sum())
        # Idle is a STALENESS question, not a gate one. See dying().
        d = self.dying()
        return {"experts": self._n,
                "idle": int((d >= self.dying_at).sum()),
                "dying_at": self.dying_at,
                "idle_by_routing": int((u < 0.1 * ideal).sum()),
                "entropy_frac": ent / max(math.log(n), 1e-9),
                "peak_over_ideal": float(u.max()) / ideal if n else 0.0,
                "pressure": float(self.pressure),
                "want_k": float(self.want_k)}

    def telemetry(self):
        """
        Per-expert history, for a checkpoint to carry and a tool to read back.

        None of it is reconstructable afterwards. The usage counters are
        rebuilt every run, and an expert's file records only when the cache
        last wrote it, which is a fact about the cache and not about use.
        """
        return {"gate": [round(float(v), 6) for v in self.gate.data],
                # What the gate was at the last prune check. Carried so a
                # resumed run can still tell a gate that is small but climbing
                # from one that has never moved; this pool's own prune reads
                # staleness instead, so nothing here depends on it.
                "gate_seen": [round(float(v), 6) for v in self.gate_seen],
                "use": [float(v) for v in self.use],
                "admits": [float(v) for v in self.admits],
                "born": [float(v) for v in self.born],
                "since": [float(v) for v in self.since],
                "last_seen": [float(v) for v in self.last_seen],
                "ever": [bool(v) for v in self.ever],
                "uid": [int(v) for v in self.uid],
                "next_uid": int(self.next_uid),
                "segments": int(self.segments)}

    def load_telemetry(self, t):
        """Put back what a checkpoint carried, ignoring anything resized."""
        if not t:
            return
        for nm in ("use", "admits", "born", "since", "last_seen",
                   "gate_seen"):
            v = t.get(nm)
            if not v:
                continue
            b = getattr(self, nm)
            k = min(len(v), b.numel())
            b[:k] = torch.tensor(v[:k], dtype=b.dtype, device=b.device)
        ev = t.get("ever")
        if ev:
            k = min(len(ev), self.ever.numel())
            self.ever[:k] = torch.tensor(ev[:k], dtype=torch.bool,
                                         device=self.ever.device)
        u = t.get("uid")
        if u:
            k = min(len(u), self.uid.numel())
            self.uid[:k] = torch.tensor(u[:k], dtype=torch.long,
                                        device=self.uid.device)
        # A directory written before ids existed has files named by position,
        # which is exactly what uid = arange gives, so it loads unchanged.
        self.next_uid = int(t.get("next_uid")
                            or (int(self.uid.max()) + 1 if self.uid.numel()
                                else 0))
        self.segments = int(t.get("segments") or self.segments)

    def attach_optimiser(self, opt):
        """swap_to needs the optimiser to carry moments with experts."""
        self._opt = opt

    def attach_sites(self, model):
        """
        Remember the call sites, so growth can extend their routers.

        Their rows belong to experts, not to VRAM slots, so a pool that gains
        an expert must give every router a row for it - otherwise the new
        expert is unaddressable by the thing that decides what runs.
        """
        from minagi.pool import PooledMLP
        self._sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
        return len(self._sites)

    @torch.no_grad()
    def observe(self, h):
        """Remember what this segment looked like, for the next choice."""
        self.summary.mul_(0.0).add_(h.detach().float().mean(dim=(0, 1)))

    def flush(self):
        """Park every resident expert, then write everything dirty to disk."""
        st = (self._opt.state if self._opt is not None else {})
        for slot, i in enumerate(self.slots):
            if i < 0:
                continue
            ent = {"w1": self.w1.data[slot].detach().to("cpu").clone(),
                   "w3": self.w3.data[slot].detach().to("cpu").clone(),
                   "w2": self.w2.data[slot].detach().to("cpu").clone()}
            for p, nm in ((self.w1, "w1"), (self.w3, "w3"), (self.w2, "w2")):
                s = st.get(p)
                if s and "exp_avg" in s:
                    ent[nm + "_m"] = s["exp_avg"][slot].to("cpu").clone()
                    ent[nm + "_v"] = s["exp_avg_sq"][slot].to("cpu").clone()
            self.tiers.put(self._f(i), ent, dirty=True)
        self.tiers.flush()

    def report(self):
        r = self.tiers.report()
        r.update(experts=self._n, resident=self.resident, swaps=self.swaps,
                 vram_params=self.vram_params(), total_params=self.n_params())
        return r
