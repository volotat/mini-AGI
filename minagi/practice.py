"""
Loop practice: teaching the model that a repetition must not feed itself.

Greedy writing falls into loops for two reasons (memory/experiments.md E56).
A near-tie at an open choice goes to the copy of something just written, and
once one copy is made the next becomes more likely: the model's estimate of
"this text repeats" rises with every repetition (Xu et al. 2022, the
self-reinforcement effect). Real text almost never repeats a sentence back to
back, so reading never shows the model what to do once that has happened. The
situation only arises in its own writing.

Practice shows it. Every few training steps the window just read supplies a
real span - a phrase, a line or a sentence that its own text does not repeat -
and the span is written out several times after its real preceding text. The
model is trained so that each repetition is at most `lam` times as likely as
the one before:

    loss = mean over n = 2..R of  relu(log P_n - sg(log P_{n-1}) - log lam) / L

where P_n is the probability of the n-th copy of the span (the product of its
characters' probabilities), sg is a stop-gradient and L is the span's length.
Only the repetitions are scored, and nothing teaches the model to predict
them.

WHY THE WHOLE SPAN AND NOT EACH CHARACTER. DITTO (Xu et al.) pulls each token
of the n-th repetition towards lam times its probability in the one before.
With words for tokens that is one decision per token. With characters, most
of a span is spelling that its first characters decide, and pulling every
character down would teach the model to misspell inside a repetition rather
than to leave it. The gradient of a span's log-probability reaches each
character in proportion to how open that choice still was, so the push lands
on the open choices - where a different phrase could begin - and leaves the
spelling alone.

WHY ONE-SIDED. The term never asks for repetition: a span that already decays
faster than lam costs nothing.

WHAT KEEPS INTENDED REPETITION. Spans are taken only where the real text does
not go on to repeat them, and everything read is still learned from in full -
refrains, repeated code, tags. So the model keeps learning from real text
where repeating is right; practice teaches it only what to do once a passage
that was not meant to repeat has started to.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SPACE, NEWLINE = 32, 10
ENDS = (SPACE, NEWLINE)                 # a span ends with one of these


def _alnum(i):
    return 48 <= i <= 57 or 65 <= i <= 90 or 97 <= i <= 122


def pick_span(ids, rng, min_len=8, max_len=200, min_prefix=64, tries=32):
    """
    A span of `ids` to practise on: (start, end), or None.

    It begins where a word does (after a space or a newline) and ends with a
    space or a newline, so its copies read as a phrase, line or sentence said
    again. It holds a letter or digit and no structural marker, is not one
    character repeated, has at least `min_prefix` characters of real text
    before it, and is neither repeated by the text right after it nor a repeat
    of the text right before: repetition the real text intends is never
    practised against.
    """
    ids = np.asarray(ids)
    n = len(ids)
    if n < min_prefix + 2 * min_len + 2:
        return None
    starts = np.flatnonzero(np.isin(ids[min_prefix - 1:n - 2 * min_len],
                                    ENDS)) + min_prefix
    if not len(starts):
        return None
    for _ in range(tries):
        a = int(starts[rng.integers(0, len(starts))])
        want = int(round(math.exp(rng.uniform(math.log(min_len),
                                              math.log(max_len)))))
        ends = np.flatnonzero(np.isin(ids[a + want - 1:min(n, a + max_len)],
                                      ENDS))
        if not len(ends):
            continue
        b = a + want + int(ends[0])                 # one past the boundary
        L = b - a
        if L < min_len or L > max_len or b + L > n:
            continue
        span = ids[a:b]
        if span.max() >= 256 or len(np.unique(span)) < 2:
            continue
        if not any(_alnum(int(c)) for c in span):
            continue
        if np.array_equal(ids[b:b + L], span):
            continue                    # the real text repeats it: intended
        if a >= L and np.array_equal(ids[a - L:a], span):
            continue                    # already a repetition
        return a, b
    return None


def loop_window(ids, rng, reps=6, prefix=512, **kw):
    """
    The practice text: up to `prefix` real characters, then a real span
    written out `reps` times. Returns (window, p, L) - the window's ids as a
    numpy array, where the first copy starts, and the span's length - or None
    when this text has no span to offer.
    """
    got = pick_span(ids, rng, **kw)
    if got is None:
        return None
    a, b = got
    ids = np.asarray(ids)
    head = ids[max(0, a - prefix):a]
    span = ids[a:b]
    window = np.concatenate([head] + [span] * reps).astype(np.int64)
    return window, len(head), b - a


def practice_loss(logits, window, p, L, reps, lam=0.5):
    """
    The term on one practice window. `logits` are the model's for
    window[:-1] (batch 1); repetition n (1-based) is window[p+(n-1)L : p+nL].
    Returns (loss, per-repetition log-probabilities as floats).
    """
    y = torch.as_tensor(window[1:], device=logits.device)
    lp = F.log_softmax(logits[0].float(), -1).gather(1, y[:, None])[:, 0]
    # the character at window index i is predicted at index i - 1
    logp = [lp[p + n * L - 1:p + (n + 1) * L - 1].sum() for n in range(reps)]
    floor = math.log(lam)
    terms = [F.relu(logp[n] - logp[n - 1].detach() - floor)
             for n in range(1, reps)]
    loss = torch.stack(terms).mean() / L
    return loss, [round(float(v), 3) for v in logp]


def practice_term(model, ids, context, device, lam=0.5, reps=6, rng=None):
    """
    One practice term on `ids`, the text a reader has just read: a span of it
    written out `reps` times after its real preceding text, forwarded as a
    training forward, and scored by practice_loss. Returns (loss, the copies'
    log-probabilities) or None when the text has no span to offer. Training
    (FileReader.practice) and chat learning (LiveLearner) both take it from
    here, so the model practises the same way wherever it learns.
    """
    from .precision import amp
    prefix = max(64, min(512, context // 4))
    longest = min(200, (context - prefix) // reps - 1)
    if longest < 8:
        return None
    got = loop_window(np.asarray(ids, dtype=np.int64),
                      rng if rng is not None else np.random.default_rng(),
                      reps=reps, prefix=prefix, max_len=longest)
    if got is None:
        return None
    window, p, L = got
    if len(window) - 1 > context:
        return None
    x = torch.from_numpy(window[:-1]).to(device).unsqueeze(0)
    y = torch.from_numpy(window[1:]).to(device).unsqueeze(0)
    with amp(device):
        logits, _ = model(x, y, caches=None, pos_offset=0)
    return practice_loss(logits, window, p, L, reps, lam)


def self_play_pair_loss(logits, y, prefix_len, chars):
    """
    The self-play term on one batch of two: row 0 is a prefix and its real
    continuation, row 1 the same prefix and the model's own; `logits` are the
    model's for each row's ids[:-1] and `y` the rows' ids[1:]. With lp the
    log-probability of each continuation (the sum over its characters),

        loss = -log sigmoid((lp_real - lp_own) / chars)

    so the real continuation goes up and the model's own goes down, and the
    push fades once the real one is clearly the more likely. Where the two
    agree character for character, their gradients cancel. Returns (loss,
    lp_real, lp_own).
    """
    lp = F.log_softmax(logits.float(), -1).gather(2, y[..., None])[..., 0]
    lp = lp[:, prefix_len - 1:prefix_len - 1 + chars].sum(1)
    return -F.logsigmoid((lp[0] - lp[1]) / chars), lp[0], lp[1]


def self_play_term(model, ids, context, device, chars=64, rng=None,
                   sample=False, prefix=512, min_prefix=256):
    """
    One self-play term on `ids`, the text a reader has just read.

    A place to write from is picked in it: where a word starts, with at least
    `min_prefix` characters of real text before it and `chars` after. The
    model writes `chars` characters on from up to `prefix` characters of that
    real text - greedily, as it writes samples and replies, or drawn from its
    distribution with `rng` when `sample` - and then reads both the real
    continuation and its own, as one training forward of two rows. The term
    raises the real one and lowers its own (self_play_pair_loss).

    Returns (loss, info) or None when there is no place to write from.
    """
    ids = np.asarray(ids, dtype=np.int64)
    rng = rng if rng is not None else np.random.default_rng()
    n = len(ids)
    hi = n - chars - 1
    if hi <= min_prefix:
        return None
    starts = np.flatnonzero(np.isin(ids[min_prefix - 1:hi - 1], ENDS)) \
        + min_prefix
    if not len(starts):
        return None
    cut = int(starts[rng.integers(0, len(starts))])
    head = ids[max(0, cut - prefix):cut]
    real = ids[cut:cut + chars]
    if len(head) + chars + 1 > context:
        return None
    was_training = model.training
    model.eval()
    own = []
    with torch.no_grad():
        caches = model.empty_caches()
        cur = torch.from_numpy(head).to(device).unsqueeze(0)
        off = 0
        for _ in range(chars):
            logits, _ = model(cur, caches=caches, pos_offset=off)
            off += cur.shape[1]
            z = logits[0, -1].float()
            if sample:
                pr = torch.softmax(z, -1).cpu().numpy().astype(np.float64)
                t = int(rng.choice(len(pr), p=pr / pr.sum()))
            else:
                t = int(z.argmax())
            own.append(t)
            cur = torch.tensor([[t]], device=device)
        del caches
    model.train(was_training)
    own = np.asarray(own, dtype=np.int64)
    rows = np.stack([np.concatenate([head, real]),
                     np.concatenate([head, own])])
    x = torch.from_numpy(rows[:, :-1]).to(device)
    y = torch.from_numpy(rows[:, 1:]).to(device)
    from .precision import amp
    with amp(device):
        logits, _ = model(x, y, caches=None, pos_offset=0)
    loss, lp_real, lp_own = self_play_pair_loss(logits, y, len(head), chars)
    return loss, {"lp_real": round(float(lp_real), 3),
                  "lp_own": round(float(lp_own), 3),
                  "agree": int((own == real).sum()), "own": own}


def loop_tail(text, min_chars=24, copies=3):
    """
    Whether `text` ends in a loop: a block repeated `copies` times or more at
    its end, covering at least `min_chars` characters, holding a letter or a
    digit and not one character repeated. Returns (period, characters the
    loop covers) for the shortest such block, or None.
    """
    n = len(text)
    for per in range(1, n // copies + 1):
        k = 0
        while k < n - per and text[n - 1 - k] == text[n - 1 - k - per]:
            k += 1
        if k >= (copies - 1) * per and k + per >= min_chars:
            block = text[n - per:]
            if any(c.isalnum() for c in block) and len(set(block)) > 1:
                return per, k + per
    return None
