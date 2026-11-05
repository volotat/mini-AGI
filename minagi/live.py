"""
Learning from a conversation while it happens.

ONE STREAM. The model reads everything it is given and everything it writes as
a single continuous stream of characters, and takes an optimiser step every
`chunk` of them - the same mechanism minagi/stream.py uses on a corpus, with
the interleaving between subjects removed, because a conversation is one
subject. There is no per-exchange boundary and nothing is re-read: the cursor
only moves forward. What the model sees at every step is the last `context`
characters, and the gradient reaches all of them.

That symmetry is the point. The thing that learns from a book and the thing
that learns from you are the same code path at the same granularity, so there
is no second mechanism to reason about and no regime the model was never
trained in.

WHAT THIS COSTS, AND IT IS NOT MEASURED IN THE CURRENT REGIME. Learning from a
conversation is learning partly from the model's own output, and reading a few
hundred characters of chat under many gradient steps is a far denser diet than
a corpus. In an earlier, denser variant - short turns re-read many times - that
cost around +0.65 nats on held-out against +0.02 for real documents of the same
length, so most of the damage came from the step density rather than from
self-training as such.

Streaming continuously is a lighter regime: the cursor only advances, text
leaves the window, and nothing is read twice at the same position. Whether the
cost is still material here is UNKNOWN and would have to be measured. It is
stated rather than assumed away.

A smaller learning rate is not the fix. Descent has a direction, and on the
model's own argmax that direction is to confirm it; a lower rate confirms it
more slowly.

THE RATE IS THE RUN'S, AND SO IS ITS CONTROLLER. Whatever the stream is - a
conversation, or a test set read while learning - it continues the stream
training reads, so it continues training's rate as well: the base rate times
the plasticity controller's factor (minagi/plasticity.py), both restored from
the weights, as in training. And the controller goes on working as it does in
training, because it is part of the model - it is what sets how fast the
model adapts: the session starts with training's held-out evaluation, and
every few minutes of learning a check reads held-out at that level and the
controller observes it (minagi/governor.py, the one path training takes as
well). How often, and on which held-out set, is what the run wrote
into the weights. Without that held-out folder the controller cannot read,
and the rate stays where training left it. A flat 3e-4, as this once used,
was ~35 times the rate training had come down to.
And a save writes back everything the weights directory carried that this
does not own - the controller's state, the reading shares - where it once
rebuilt the manifest without them, and the next training session started at
the full rate.
"""

import time

import torch

from .precision import amp


class LiveLearner:
    """One optimiser, and one step per `chunk` characters of the stream, at
    the rate training would take its next step at. `lr` is the base rate the
    controller scales, as train.py's --lr is."""

    # what minagi/store.py writes into a manifest itself; everything else a
    # manifest holds is carried through a save untouched
    OWN = ("step", "val", "cfg", "telemetry", "n_experts", "d_model", "d_ff",
           "paged", "core_tensors", "router_tensors", "experts",
           "total_bytes")

    def __init__(self, model, lr=3e-4, trunk_lr_mult=0.2, wd=0.1, clip=1.0,
                 aux_weight=0.0, save_every=8, weights_dir=None, manifest=None,
                 chunk=512, context=None, optimizer="adamw",
                 muon_params="all"):
        self.model = model
        # What the model was loaded as. save() writes the manifest that names
        # every tensor's shape, and without this it writes an empty one: the
        # directory then claims the default vocabulary of 8192 while holding
        # 265, and will not load again. Caught by reloading after a save,
        # which is the only way this kind of damage shows up.
        self.manifest = dict(manifest or {})
        self.clip = clip
        self.aux_weight = aux_weight
        self.save_every = save_every
        self.weights_dir = weights_dir
        self.steps = 0
        self.unsaved = 0
        self.log = []
        # the held-out check and the controller's response to it - set by
        # for_run (minagi/governor.py); None leaves the rate where it is
        self.gov = None
        self.checked = []
        # the controller, where training left it, and how far the stream had
        # got: a save carries both on by what was learned here
        from .plasticity import Plasticity
        self.plast = Plasticity.restore(self.manifest.get("plasticity"))
        self.step0 = int(self.manifest.get("step") or 0)
        self.chars0 = float(self.manifest.get("read_chars") or 0)
        self.nats0 = float(self.manifest.get("read_nats") or 0.0)
        self.chars, self.nats = 0, 0.0
        # the stream. `buf` holds the last `context` character ids and nothing
        # older; `pending` counts characters read since the last step.
        self.chunk = int(chunk)
        self.context = int(context or getattr(model.cfg, "block", 16384))
        self.buf = []
        self.pending = 0
        dev = next(model.parameters()).device
        if getattr(model, "pool", None) is not None:
            # training's own split: the router rows and the depth embedding
            # are the pool's, at its rate - split by name they went to the
            # trunk and learned at its tenth
            from .pool import split_trunk_pool
            trunk, pool_ps = split_trunk_pool(model)
        else:
            trunk = [p for n, p in model.named_parameters()
                     if not n.startswith("pool.")]
            pool_ps = [p for n, p in model.named_parameters()
                       if n.startswith("pool.")]
        # the optimiser training steps with (training.optimizer)
        from .muon import build
        self.opt = build(model, trunk, pool_ps, lr, lr * trunk_lr_mult, wd,
                         kind=optimizer, fused=(dev.type == "cuda"),
                         which=muon_params)
        pool = getattr(model, "pool", None)
        if pool is not None and hasattr(pool, "attach_optimiser"):
            pool.attach_optimiser(self.opt)
        # ADAM'S MOMENTS carry on from where the weights directory left them,
        # as they do when the trainer resumes. A fresh optimiser would start
        # every tensor from zero moments at the first step - the trunk's and
        # every expert on the card, instead of their own - and its first
        # save would write that short history over the trainer's in
        # optim.npz, step counters included. The experts' own moments are
        # put in place by the pool before each step (PagedPool._own_moments).
        if weights_dir:
            from . import store as weights_store
            try:
                weights_store._load_optim(self.opt, model, weights_dir)
            except Exception as e:             # never stop the server for it
                print(f"  [note] optimiser moments not restored: {e}")

    def feed(self, text, tok, note=""):
        """
        Add text to the stream and take a step for every `chunk` characters.

        Everything goes in - what you typed and what the model wrote - because
        the stream is what the model is reading, and it does not get to skip
        the parts it produced. Returns one record per step taken, which may be
        none for a short message and several for a long one. The caller holds
        whatever lock guards the model.
        """
        ids = list(tok.encode(text).ids)
        recs, i = [], 0
        while i < len(ids):
            take = min(self.chunk - self.pending, len(ids) - i)
            self.buf.extend(ids[i:i + take])
            self.pending += take
            i += take
            if self.pending >= self.chunk:
                # trim to the window BEFORE the step: what the model sees is
                # the last `context` characters, exactly as the corpus reader
                # sees them, and the gradient reaches all of it
                if len(self.buf) > self.context:
                    del self.buf[:-self.context]
                rec = self._step(note)
                if rec is not None:
                    recs.append(rec)
                self.pending = 0
        if len(self.buf) > self.context:
            del self.buf[:-self.context]
        return recs

    def learn(self, text, tok, note=""):
        """Compatibility: feed the text and return the last step, if any."""
        recs = self.feed(text, tok, note=note)
        return recs[-1] if recs else None

    def factor(self):
        """The controller's factor on the base rate, as training applies it."""
        return self.plast.factor()

    def step_on(self, ids, note="", new=None):
        """One optimiser step on exactly `ids`, outside the stream's window:
        for text read in pieces, each learned from alone, or a window a
        reader has laid out itself. `new` is how many of its characters were
        not read before - all of them unless said."""
        return self._step(note, ids=list(ids),
                          new=len(ids) - 1 if new is None else new)

    def _step(self, note="", ids=None, new=None):
        """One optimiser step on the window as it stands. `new` is how many
        of its characters the stream had not read before - a chunk."""
        ids = self.buf if ids is None else ids
        new = self.chunk if new is None else int(new)
        if len(ids) < 8:
            return None
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        # the rate training would step at now: base times the controller's
        # factor
        scale = self.plast.factor()
        for g in self.opt.param_groups:
            g["lr"] = g.get("base_lr", g["lr"]) * scale
        self.model.train()
        try:
            with amp(dev):
                _, loss = self.model(
                    torch.tensor([ids[:-1]], device=dev),
                    torch.tensor([ids[1:]], device=dev))
                if self.aux_weight and getattr(self.model, "pool", None):
                    loss = loss + self.aux_weight * self.model.pool_aux()
                if hasattr(self.model, "pool_balance"):
                    loss = loss + self.model.pool_balance()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip))
            self.opt.step()
        finally:
            self.opt.zero_grad(set_to_none=True)
            if not was_training:
                self.model.eval()
        self.steps += 1
        self.unsaved += 1
        self.plast.tick()
        self.check_if_due()
        self.chars += new
        self.nats += float(loss) * new
        pool = getattr(self.model, "pool", None)
        if pool is not None and hasattr(pool, "now"):
            pool.now = self.step0 + self.steps   # the clock training keeps
        rec = {"step": self.steps, "chars": len(ids), "loss": float(loss),
               "grad_norm": gn, "lr": self.opt.param_groups[-1]["lr"],
               "scale": scale, "note": note, "at": time.time()}
        self.log.append(rec)
        return rec

    def check_if_due(self):
        """Training's held-out check, when it is due: read held-out at the
        session's level and let the controller and the experts' rates respond
        (minagi/governor.py). The model is left in the mode it was in."""
        if self.gov is None or not self.gov.due():
            return None
        was_training = self.model.training
        try:
            got = self.gov.check()
        finally:
            self.model.train(was_training)
        if got is None:
            return None
        d, v, se, note = got
        rec = {"step": self.steps, "val": v, "se": se,
               "scale": self.plast.factor(), "note": note,
               "at": time.time()}
        self.checked.append(rec)
        return rec

    def due_to_save(self):
        return (self.weights_dir and self.save_every
                and self.unsaved >= self.save_every)

    def save(self, extra=None):
        """Persist what has been learned. Without this it dies with the process."""
        if not self.weights_dir:
            return False
        from . import store as weights_store
        from dataclasses import asdict, is_dataclass
        cfg = self.manifest.get("cfg")
        if not cfg:
            mc = getattr(self.model, "cfg", None)
            cfg = asdict(mc) if is_dataclass(mc) else (dict(mc.__dict__)
                                                       if mc else None)
        # everything the directory carried that the store does not write
        # itself - the controller, the reading shares, the base rate - and
        # the stream carried on by what was read here
        keep = {k: v for k, v in self.manifest.items() if k not in self.OWN}
        keep.update(self.progress())
        keep.update(extra or {})
        weights_store.save(self.model, self.weights_dir, opt=self.opt,
                           step=self.step0 + self.steps,
                           val=self.manifest.get("val"),
                           cfg=cfg, extra=keep)
        self.unsaved = 0
        return True

    def progress(self):
        """Where the stream is now, as the manifest keeps it."""
        return {"read_chars": int(self.chars0 + self.chars),
                "read_nats": self.nats0 + self.nats,
                "plasticity": self.plast.state()}

    @classmethod
    def for_run(cls, model, weights_dir=None, manifest=None, lr=None,
                chunk=None, context=None, save_every=8, config=None,
                checks=True, held_out=None):
        """A learner set up as training is: config.yaml's training settings,
        the base rate the run itself used (the manifest's `lr`, which train.py
        writes) unless `lr` says otherwise, and the controller where the run
        left it - working as it does in training: the session's held-out
        evaluation now, and a check every few minutes of learning, as the run
        took them (the manifest's `checks`; minagi/governor.py). `held_out`
        names another folder; `checks=False` leaves the rate where the run
        left it, for tests."""
        from .config import get, load
        c = config if config is not None else load()
        man = dict(manifest or {})
        base = (float(lr) if lr is not None
                else float(man.get("lr") or get(c, "training.lr", 3e-4)))
        me = cls._for_run(model, base, man, c, weights_dir, chunk, context,
                          save_every)
        if checks:
            me.govern(man.get("checks"), held_out,
                      chunk=int(get(c, "training.chunk", 2048)))
        return me

    def govern(self, settings=None, held_out=None, chunk=None):
        """Run the controller as training does: the session's held-out
        evaluation now, then a check whenever one is due. Held-out is read
        in training's chunks, `chunk`, whatever this stream steps by - the
        readings are training's. Returns the evaluation, or None when there
        is no held-out folder to read - the rate then stays where the run
        left it, and this says so."""
        import os
        from .governor import Governor
        s = dict(settings or {})
        if held_out:
            s["held_out"] = held_out
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dev = next(self.model.parameters()).device
        self.gov = Governor.for_learner(self.model, self.plast,
                                        int(chunk or self.chunk), dev, s,
                                        root=root)
        if self.gov is None:
            print(f"  [note] no held-out folder at "
                  f"{s.get('held_out') or 'data/val'} - the learning rate's "
                  f"controller cannot read, so the rate stays where the run "
                  f"left it", flush=True)
            return None
        print(f"  reading held-out as a training session starts, "
              f"{-(-self.gov.eval_steps // self.gov.ev.per)} blocks of each "
              f"subject, for the level its checks report at", flush=True)
        was_training = self.model.training
        try:
            v, se, c = self.gov.start()
        finally:
            self.model.train(was_training)
        import numpy as np
        print(f"  held-out at the start: {np.mean(list(v.values())):.4f} "
              f"+/-{se:.4f}; a check every {self.gov.every:g} minutes of "
              f"learning moves the rate as training's do", flush=True)
        return v

    @classmethod
    def _for_run(cls, model, base, man, c, weights_dir, chunk, context,
                 save_every):
        from .config import get
        learner = cls(model, lr=base,
                      aux_weight=float(getattr(getattr(model, "cfg", None),
                                               "pool_aux", 0.0) or 0.0),
                      trunk_lr_mult=get(c, "training.trunk_lr_mult", 0.1),
                      wd=get(c, "training.weight_decay", 0.1),
                      clip=get(c, "training.clip", 1.0),
                      chunk=chunk or get(c, "training.chunk", 2048),
                      context=context, save_every=save_every,
                      weights_dir=weights_dir, manifest=man,
                      optimizer=str(get(c, "training.optimizer", "adamw")),
                      muon_params=str(get(c, "training.muon_params", "all")))
        # a newborn's trial, in the steps the run counts, as train.py sets it:
        # the routing bias's floor rule reads it after every step taken here
        # too, and without it would take every newborn's lift away
        pool = getattr(model, "pool", None)
        if pool is not None and hasattr(pool, "trial") and not pool.trial:
            surv = float(get(c, "prune.survival_chars", 0) or 0)
            run_chunk = max(1, int(get(c, "training.chunk", 2048) or 2048))
            pool.trial = max(1, int(round(surv / run_chunk))) if surv else 0
        return learner


def exchange_text(user_text, bot_text, u0="<user>", u1="</user>",
                  b0="<bot>", b1="</bot>"):
    """
    One turn, marked up the way the chat corpus is.

    The user's half is the part that carries information: the model did not
    predict it, and that surprise is the whole signal in the exchange. Its own
    half is the argmax and mostly confirms what it already believed - kept
    because what it said is part of what happened, not because it teaches
    much.
    """
    user_text = (user_text or "").strip()
    bot_text = (bot_text or "").strip()
    if not user_text and not bot_text:
        return ""
    return f"{u0}\n{user_text}\n{u1}\n{b0}\n{bot_text}\n{b1}\n"
