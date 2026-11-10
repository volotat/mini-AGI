"""Choosing the next character.

`Sampler` holds the established ways of choosing from the model's prediction,
one stream at a time, and `sampler_from` builds the one config.yaml's
`decoding` sets - how chat (serve.py) and the training sample log write.
tools/write_meter.py measures them against each other on held-out openings.

`pick_next` is the older rule: greedy, holding off loops with an adaptation
trace - what was just said is suppressed, and the suppression decays - and it
is what model.generate uses unless given a Sampler. `contrastive_generate`
scores candidates against the context already in the cache (Su & Collier
2022); it asks the forward for hidden states, which the forward no longer
returns, so it does not run against the current model.
"""


import math

import numpy as np
import torch
import torch.nn.functional as F

from .precision import amp


class Sampler:
    """
    How one stream of writing chooses each next character, by the established
    methods. One Sampler per stream: it holds the stream's random numbers and
    what mirostat carries from one character to the next.

    In this order, each step working on what the one before left:

      adapt_strength  the adaptation trace below (pick_next): what was just
                      written is suppressed, and the suppression decays
      dry             DRY, "don't repeat yourself": a character that would
                      continue a repeat of `dry_allowed` characters or more
                      loses dry * dry_base ** (length - dry_allowed) logits,
                      so short repeats - words, markup, a name - are free and
                      a long one is priced out as it grows. The length is the
                      longest stretch of the text, ending here, that occurred
                      earlier and was followed there by that character, looked
                      for over the last `dry_range` characters. Its defaults
                      are the token ones in characters: a token is about four,
                      so allowed 2 tokens is 8 characters and base 1.75 a
                      token is 1.15 a character.
      temperature     divides the logits; 0 takes the highest scoring
                      character and draws nothing
      top_k           the k likeliest characters (Fan et al. 2018)
      top_p           nucleus: the fewest likeliest characters holding p of
                      the probability (Holtzman et al. 2020)
      min_p           those at least min_p times as likely as the likeliest
                      (Nguyen et al. 2024)
      typical_p       locally typical: the characters whose surprise is
                      nearest the distribution's entropy, until they hold p
                      (Meister et al. 2023)
      eta             eta sampling: those likelier than min(eta,
                      sqrt(eta) * exp(-entropy)) (Hewitt et al. 2022)
      mirostat        mirostat 2.0, a target surprise in bits a character:
                      characters more surprising than a threshold mu are cut,
                      and mu moves by mirostat_lr times each drawn
                      character's surprise above the target (Basu et al.
                      2021)

    Each cut keeps the likeliest character - typical_p its most typical -
    renormalises what is left, and the draw is from that. Logits come in as
    one row, the history as the character ids of the text so far, prompt
    included.
    """

    def __init__(self, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0,
                 typical_p=1.0, eta=0.0, mirostat=0.0, mirostat_lr=0.1,
                 dry=0.0, dry_base=1.15, dry_allowed=8, dry_range=4096,
                 adapt_strength=0.0, adapt_decay=0.88, seed=None):
        self.temperature = float(temperature or 0.0)
        self.top_k, self.top_p = int(top_k or 0), float(top_p)
        self.min_p, self.typical_p = float(min_p), float(typical_p)
        self.eta = float(eta)
        self.mirostat, self.mirostat_lr = float(mirostat), float(mirostat_lr)
        self.mu = 2.0 * self.mirostat
        self.dry, self.dry_base = float(dry), float(dry_base)
        self.dry_allowed, self.dry_range = int(dry_allowed), int(dry_range)
        self.adapt_strength = float(adapt_strength)
        self.adapt_decay = float(adapt_decay)
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))
        else:
            self.gen.seed()

    def describe(self):
        """The rule in a line, for a log: only what is switched on."""
        p = ["greedy" if self.temperature <= 0
             else f"temperature {self.temperature:g}"]
        if self.temperature > 0:
            for name, v, off in (("top_k", self.top_k, 0),
                                 ("top_p", self.top_p, 1.0),
                                 ("min_p", self.min_p, 0.0),
                                 ("typical_p", self.typical_p, 1.0),
                                 ("eta", self.eta, 0.0),
                                 ("mirostat", self.mirostat, 0.0)):
                if v != off:
                    p.append(f"{name} {v:g}")
        if self.dry:
            p.append(f"DRY {self.dry:g} x {self.dry_base:g}^(repeat - "
                     f"{self.dry_allowed})")
        if self.adapt_strength:
            p.append(f"adaptation {self.adapt_strength:g} decaying "
                     f"{self.adapt_decay:g}")
        return ", ".join(p)

    def __call__(self, logits, history):
        z = logits.detach().float().reshape(-1).cpu()
        if self.adapt_strength:
            z = adapted(z, history, self.adapt_strength, self.adapt_decay)
        if self.dry:
            z = z - dry_penalty(history, z.numel(), self.dry, self.dry_base,
                                self.dry_allowed, self.dry_range)
        if self.temperature <= 0:
            return int(z.argmax())
        z = z / self.temperature
        top = int(z.argmax())
        if self.top_k and self.top_k < z.numel():
            kth = torch.topk(z, self.top_k).values[-1]
            z = _keep(z, z >= kth, top)
        if self.top_p < 1.0:
            p = F.softmax(z, -1)
            srt, order = torch.sort(p, descending=True)
            before = srt.cumsum(0) - srt          # mass of the likelier ones
            keep = torch.zeros_like(p, dtype=torch.bool)
            keep[order[before < self.top_p]] = True
            z = _keep(z, keep, top)
        if self.min_p > 0:
            p = F.softmax(z, -1)
            z = _keep(z, p >= self.min_p * p.max(), top)
        if self.typical_p < 1.0:
            lp = F.log_softmax(z, -1)
            p = lp.exp()
            ent = -(p[p > 0] * lp[p > 0]).sum()
            order = torch.argsort((-lp - ent).abs())
            before = p[order].cumsum(0) - p[order]
            keep = torch.zeros_like(p, dtype=torch.bool)
            keep[order[before < self.typical_p]] = True
            # the most typical character stays, not the likeliest: leaving
            # out a character too predictable for its place is the method
            z = _keep(z, keep, int(order[0]))
        if self.eta > 0:
            lp = F.log_softmax(z, -1)
            p = lp.exp()
            ent = float(-(p[p > 0] * lp[p > 0]).sum())
            cut = min(self.eta, math.sqrt(self.eta) * math.exp(-ent))
            z = _keep(z, p > cut, top)
        if self.mirostat > 0:
            bits = -F.log_softmax(z, -1) / math.log(2)
            z = _keep(z, bits <= self.mu, top)
        p = F.softmax(z, -1)
        nxt = int(torch.multinomial(p, 1, generator=self.gen))
        if self.mirostat > 0:
            surprise = -math.log2(max(float(p[nxt]), 1e-30))
            self.mu -= self.mirostat_lr * (surprise - self.mirostat)
        return nxt


def _keep(z, keep, top):
    """The logits with everything outside `keep` cut; the likeliest stays."""
    keep = keep.clone()
    keep[top] = True
    return z.masked_fill(~keep, float("-inf"))


def adapted(z, history, strength, decay=0.88, window=64):
    """One row of logits less the adaptation trace of the history (see
    pick_next)."""
    n = min(len(history), window)
    if n == 0:
        return z
    tail = torch.as_tensor(list(history[-n:]), dtype=torch.long)
    w = decay ** torch.arange(n - 1, -1, -1, dtype=z.dtype)
    trace = torch.zeros_like(z).scatter_add_(0, tail, w)
    return z - strength * trace


def dry_penalty(history, vocab, multiplier, base, allowed, window=4096,
                longest=512):
    """DRY's logits to take off each next character: for every character, the
    longest repeat it would continue - the longest stretch ending the text
    that also occurred earlier, followed there by that character - and, from
    `allowed` characters up, multiplier * base ** (length - allowed)."""
    seq = np.asarray(list(history[-window:]), dtype=np.int64)
    n = len(seq)
    pen = torch.zeros(vocab)
    if n < 2:
        return pen
    ends = np.nonzero(seq[:-1] == seq[-1])[0]   # earlier places of the last
    length = np.ones(len(ends), dtype=np.int64)
    alive = np.ones(len(ends), dtype=bool)
    for k in range(1, min(longest, n - 1)):
        j = ends - k
        ok = alive & (j >= 0)
        ok[ok] = seq[j[ok]] == seq[n - 1 - k]
        if not ok.any():
            break
        length[ok] += 1
        alive = ok
    best = np.zeros(vocab, dtype=np.int64)
    np.maximum.at(best, seq[ends + 1], length)
    over = best >= allowed
    if over.any():
        pen[torch.as_tensor(over)] = torch.as_tensor(
            multiplier * base ** np.minimum(best[over] - allowed, 200),
            dtype=pen.dtype)
    return pen


def settings_from(cfg=None):
    """config.yaml's `decoding`: the Sampler settings chat and the sample log
    write with. Every key there is one of Sampler's settings, and a name
    Sampler does not have is an error rather than a setting silently
    ignored."""
    import inspect
    if cfg is None:
        from .config import load
        cfg = load()
    dec = dict(cfg.get("decoding") or {})
    known = set(inspect.signature(Sampler.__init__).parameters) - {"self",
                                                                   "seed"}
    unknown = sorted(set(dec) - known)
    if unknown:
        raise ValueError(f"config.yaml decoding: {', '.join(unknown)} - "
                         f"not a Sampler setting (minagi/decode.py)")
    return dec


def sampler_from(cfg=None, seed=None):
    """The Sampler config.yaml's `decoding` sets. `seed` None draws fresh
    random numbers each time; a number repeats them."""
    return Sampler(seed=seed, **settings_from(cfg))


def pick_next(logits, prev_ids, temperature=0.0, top_k=0, top_p=1.0,
              rep_penalty=1.0, no_repeat_ngram=0,
              adapt_strength=0.0, adapt_decay=0.88, adapt_window=64):
    """
    Choose the next token. Deterministic when temperature <= 0.

    Determinism is worth having: a model you can test and diff needs to give
    the same answer twice. But greedy decoding is also the classic cause of
    degenerate repetition - argmax on a distribution whose mode reinforces
    itself will re-enter the same state forever. Extra capacity does not fix
    that, because the loop lives in the decoding rule, not the parameters.

    Three knobs suppress loops without giving up determinism:

      adapt_strength   ADAPTATION, and the one that works. A decaying trace
                       per character: what was just said is suppressed, and
                       the suppression fades. Measured over six prompts, mean
                       repeated 8-grams:

                           greedy alone                       73.5%
                           greedy + rep_penalty 1.12          59.0%
                           greedy + adaptation 2.5            11.6%

      rep_penalty      divides the logits of tokens already emitted. Weak,
                       because it marks a character FOREVER the first time it
                       appears - after 140 characters that is 93% of the
                       alphabet, penalised equally, which is no signal. Not
                       useless (59% against 73.5% without it), but adaptation
                       does the same job properly and they were not measured
                       together.

      no_repeat_ngram  hard-bans any n-gram that has already occurred. Keep it
                       large (12+) at character level, or better, leave it off:
                       100% of real code files repeat a 32-gram and 40% repeat
                       a 48-gram, so any value strict enough to stop a loop
                       also forbids ordinary code. Measured, it barely helped -
                       the loop simply steps around the ban and resumes.

    WHY ADAPTATION AND NOT A LONGER MEMORY. The trace is deliberately short:
    at decay 0.88 its half-life is 5.4 characters, so one use of a character
    costs 2.5 logits now and 0.9 eight characters later - invisible to ordinary
    text. A character stuck in a loop never gets to decay, and converges to
    strength/(1-decay) = 20.8 logits, which no loop survives.

    Its blind spot is a loop whose period exceeds that memory. Nothing here
    catches a phrase that returns every forty characters; only the model
    getting better does.

    A frequency-normalised variant - charging a character only for use above
    its natural rate, so space is not billed 4.3 logits for being 20% of all
    text - was measured and is WORSE, 42.5% against 11.6%. Loops are built out
    of common characters, so exempting them is what lets the loop through.
    """
    logits = logits.clone()
    B = logits.shape[0]
    if adapt_strength and prev_ids is not None and prev_ids.numel():
        # The trace is a function of the history, not state carried between
        # calls, so this stays pure and every existing caller keeps working.
        # Only the last `adapt_window` characters matter: at decay 0.88,
        # 0.88^64 is 3e-4, and everything older is already nothing.
        n = min(prev_ids.shape[1], adapt_window)
        tail = prev_ids[:, -n:]
        w = (adapt_decay ** torch.arange(n - 1, -1, -1, device=logits.device,
                                         dtype=logits.dtype))
        trace = torch.zeros_like(logits)
        trace.scatter_add_(1, tail, w.expand(B, n))
        logits = logits - adapt_strength * trace
    if rep_penalty and rep_penalty != 1.0 and prev_ids is not None:
        for b in range(B):
            uniq = torch.unique(prev_ids[b])
            l = logits[b, uniq]
            logits[b, uniq] = torch.where(l > 0, l / rep_penalty,
                                          l * rep_penalty)
    if no_repeat_ngram and prev_ids is not None:
        n = no_repeat_ngram
        for b in range(B):
            seq = prev_ids[b].tolist()
            if len(seq) >= n:
                prefix = tuple(seq[-(n - 1):])
                banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                          if tuple(seq[i:i + n - 1]) == prefix}
                if banned:
                    logits[b, list(banned)] = float("-inf")
    if temperature is None or temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    z = logits / temperature
    if top_k:
        kth = torch.topk(z, min(top_k, z.size(-1)))[0][..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    probs = F.softmax(z, dim=-1)
    if top_p and top_p < 1.0:
        srt, order = torch.sort(probs, descending=True, dim=-1)
        cums = srt.cumsum(-1)
        srt[cums - srt > top_p] = 0.0
        srt = srt / srt.sum(-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, order, srt)
    return torch.multinomial(probs, 1)




@torch.no_grad()
def contrastive_generate(model, idx, max_new_tokens, top_k=8, alpha=0.6,
                         stop_ids=None):
    """
    Contrastive search: deterministic, and anti-degeneration by construction.

        score(v) = (1 - alpha) * p(v)  -  alpha * max_j cos(h_v, h_j)

    The first term is the model's own confidence; the second penalises a
    candidate whose hidden representation looks like something already in the
    context. Loops are suppressed because a looping token is, by definition,
    representationally close to what came before.

    This matters because greedy degeneration is a symptom of maximum-likelihood
    training - MLE piles probability onto locally-safe continuations, so argmax
    walks into a cycle. Temperature does not fix that; it injects noise so the
    walk falls out of the cycle by luck. This addresses the mechanism instead,
    and the output stays a deterministic function of the input.

    Cost: one extra batched forward step of width top_k per token, sharing the
    context's KV cache. (Su & Collier 2022)
    """
    model.eval()
    device = idx.device
    n_layer = model.cfg.n_layer
    caches = [{"k": None, "v": None} for _ in range(n_layer)]
    with amp(idx.device):
        logits, _, hidden = model(idx, caches=caches, pos_offset=0,
                                  return_hidden=True)
    ctx = F.normalize(hidden[0].float(), dim=-1)          # [T, D]
    offset = idx.shape[1]
    out = idx

    for _ in range(max_new_tokens):
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        top_p_vals, top_i = probs.topk(min(top_k, probs.numel()))
        k = top_i.numel()
        # replay the same context for every candidate in one batched step
        exp = [{"k": c["k"].repeat(k, 1, 1, 1),
                "v": c["v"].repeat(k, 1, 1, 1)} for c in caches]
        cand = top_i.view(k, 1)
        with amp(cand.device):
            lg, _, h = model(cand, caches=exp, pos_offset=offset,
                             return_hidden=True)
        hv = F.normalize(h[:, -1].float(), dim=-1)        # [k, D]
        penalty = (hv @ ctx.T).max(dim=-1).values         # [k]
        score = (1 - alpha) * top_p_vals - alpha * penalty
        win = int(score.argmax())

        caches = [{"k": c["k"][win:win + 1], "v": c["v"][win:win + 1]}
                  for c in exp]
        nxt = top_i[win].view(1, 1)
        out = torch.cat([out, nxt], dim=1)
        ctx = torch.cat([ctx, hv[win:win + 1]], dim=0)
        logits = lg[win:win + 1]
        offset += 1
        if stop_ids is not None and int(nxt.item()) in stop_ids:
            break
        if out.shape[1] >= model.cfg.block:
            break
    return out


