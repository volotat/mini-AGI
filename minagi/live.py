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
"""

import time

import torch

from .precision import amp


class LiveLearner:
    """One optimiser, and one step per `chunk` characters of the stream."""

    def __init__(self, model, lr=3e-4, trunk_lr_mult=0.2, wd=0.1, clip=1.0,
                 aux_weight=0.0, save_every=8, weights_dir=None, manifest=None,
                 chunk=512, context=None):
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
        # the stream. `buf` holds the last `context` character ids and nothing
        # older; `pending` counts characters read since the last step.
        self.chunk = int(chunk)
        self.context = int(context or getattr(model.cfg, "block", 16384))
        self.buf = []
        self.pending = 0
        dev = next(model.parameters()).device
        trunk = [p for n, p in model.named_parameters()
                 if not n.startswith("pool.")]
        pool_ps = [p for n, p in model.named_parameters()
                   if n.startswith("pool.")]
        self.opt = torch.optim.AdamW(
            [{"params": trunk, "name": "trunk", "weight_decay": wd,
              "lr": lr * trunk_lr_mult, "base_lr": lr * trunk_lr_mult},
             {"params": pool_ps, "name": "pool", "weight_decay": wd,
              "lr": lr, "base_lr": lr}],
            lr=lr, betas=(0.9, 0.95), fused=(dev.type == "cuda"))
        pool = getattr(model, "pool", None)
        if pool is not None and hasattr(pool, "attach_optimiser"):
            pool.attach_optimiser(self.opt)

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

    def _step(self, note=""):
        """One optimiser step on the window as it stands."""
        ids = self.buf
        if len(ids) < 8:
            return None
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.train()
        try:
            with amp(dev):
                _, loss = self.model(
                    torch.tensor([ids[:-1]], device=dev),
                    torch.tensor([ids[1:]], device=dev))
                if self.aux_weight and getattr(self.model, "pool", None):
                    loss = loss + self.aux_weight * self.model.pool_aux()
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
        rec = {"step": self.steps, "chars": len(ids), "loss": float(loss),
               "grad_norm": gn, "note": note, "at": time.time()}
        self.log.append(rec)
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
        keep = {k: self.manifest[k] for k in ("read_chars", "context_now")
                if k in self.manifest}
        keep.update(extra or {})
        # A step is CUMULATIVE over the model's whole life, the way train.py
        # counts it, and this process only ever took the newest ones. Writing
        # `self.steps` alone would restart the counter at zero, and an expert's
        # age is `step - born`, so the next train.py resume would read a step
        # younger than the experts and exempt them all from pruning.
        weights_store.save(self.model, self.weights_dir, opt=self.opt,
                           step=(self.manifest.get("step") or 0) + self.steps,
                           val=self.manifest.get("val"),
                           cfg=cfg, extra=keep)
        self.unsaved = 0
        return True


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
