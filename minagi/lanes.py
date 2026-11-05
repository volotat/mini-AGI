"""
How the reading is shared between subjects.

Reading goes one visit at a time - a passage from one subject's files - and
which subject comes next used to be a strict rotation: every subject the same
share, whatever its size and whatever it still had to teach. On this run that
gave the 7 MB self-knowledge folder an eighth of all reading, enough to read
it about forty times over, and arithmetic, whose held-out had not moved in a
billion characters, as much as code or chat.

THE RULE. The next visit goes where the next character read is expected to
lower held-out most. Each subject's held-out follows a power law in what its
lane has read, L = A * D^-alpha, and the drop one more character buys is the
derivative, alpha * L / D. A lane is worth that summed over the held-out
subjects it feeds - chat feeds chat and chat_hermes - and the shares follow
the worths. A lane read more is worth less per character, so the shares come
back toward each other as the reading evens out the worths: the allocation
that lowers the sum of the subjects' held-out most for what is read.

THE EVIDENCE. alpha and L are fitted to the run's own held-out checks, log L
against log D, weighted toward the recent: a check read when the run had
read D' counts (D'/D)^(1/TAU) - a window in log D, the scale a power law
lives on. A window in characters does not work: the run's first checks sit
far out on log D, and however little they weigh they set the slope. The
window has to be long, too. A check reads a few blocks of a subject, and a
subject's slope is about 1e-4 a check against 0.05 of scatter, so a few
hundred checks are what make a slope - while the whole run's slopes are two
to five times today's, so the start of the run would mislead. TAU 0.15 is
the last ~16% of the reading, 0.37B characters at 2.3B read: on this run's
log, fits from alternate checks agree to 0.002, and a slope that fell
sharply from 0.3 to 0.1, 1.3B characters back, is read as 0.105 (at 0.25
it read 0.131). A
subject moving the wrong way, its held-out rising, has as much claim as one
falling: its slope counts by size (Matiisen et al. 2017, teacher-student
curriculum learning).

NO EVIDENCE. A lane no held-out subject measures - self-knowledge - is valued
as the typical subject is, discounted for how often it has been read already:
each reading past the first is worth exp(-k / 15.4) of new text (Muennighoff
et al. 2023, scaling data-constrained language models).

LIMITS. No subject below a quarter of an even share, because what is not read
is forgotten and interleaving is what keeps it; none above three times an
even share. Shares move at most STEP a check, so a change is spread over
hours and the held-out the rate controller steers by is never jolted.

THE SHARES ARE OF CHARACTERS, as the rule values characters. A visit is one
file's passage - a short file read whole - so a visit to chess or
arithmetic holds about six times the characters of one to wikipedia. Visits
are chosen by smooth weighted round-robin in characters: whatever a visit
reads, every subject gains its share of those characters in credit and the
subject read pays them back, and the next visit goes to the subject with the
most credit. A subject of long files is visited less often and reads as much,
the proportions are exact, and no subject waits long. A visit that reads
nothing - every file it tried had gone - is charged as a full one, so the
reader moves on.
"""
import math

MODES = ("auto", "equal", "fixed")


def feeds(lanes, subjects):
    """Which held-out subjects each lane feeds: the subject of the same name,
    or one named after the lane and an underscore (chat_hermes is fed by
    chat). A subject goes to the longest lane name that fits it."""
    out = {lane: [] for lane in lanes}
    for s in subjects:
        fit = [lane for lane in lanes if s == lane or s.startswith(lane + "_")]
        if fit:
            out[max(fit, key=len)].append(s)
    return out


class LaneSchedule:
    TAU = 0.15          # how far back the slopes look, in log characters
    MIN_EFF = 30.0      # weight of checks before a subject's fit is used
    FLOOR = 0.25        # no subject below a quarter of an even share
    CEIL = 3.0          # nor above three times one
    STEP = 0.05         # most a share moves at one check, in log terms
    R_STAR = 15.4       # how fast re-reading loses value (Muennighoff et al.)

    def __init__(self, lanes, sizes, subjects=(), mode="auto", weights=None):
        if mode not in MODES:
            raise ValueError(f"lane schedule {mode!r} is not one of {MODES}")
        self.names = list(lanes)
        self.sizes = {n: float(sizes.get(n, 0) or 0) for n in self.names}
        self.feeds = feeds(self.names, subjects)
        self.mode = mode
        n = len(self.names)
        if mode == "fixed":
            w = {k: float((weights or {}).get(k, 1.0)) for k in self.names}
            tot = sum(w.values())
            self.shares = {k: w[k] / tot for k in self.names}
        else:
            self.shares = {k: 1.0 / n for k in self.names}
        self.read = {k: 0.0 for k in self.names}     # characters, lifetime
        self.fits = {}             # subject -> [w, sx, sy, sxx, sxy]
        self.last = None           # characters read at the last observation
        self.credit = {k: 0.0 for k in self.names}

    # -------------------------------------------------------------- evidence
    def observe(self, total, per_subject):
        """A held-out check, read with `total` characters behind the model:
        every subject's reading joins its fit, against what its lane has
        read, and every older reading weighs a little less."""
        if self.mode != "auto":
            return
        decay = (1.0 if self.last is None or self.last <= 0 or total <= 0
                 else math.exp(-max(0.0, math.log(total / self.last))
                               / self.TAU))
        self.last = total
        for f in self.fits.values():
            for i in range(5):
                f[i] *= decay
        for lane, subs in self.feeds.items():
            if self.read[lane] <= 0:
                continue
            x = math.log(self.read[lane])
            for s in subs:
                y = per_subject.get(s)
                if y is None or not y > 0:
                    continue
                y = math.log(y)
                f = self.fits.setdefault(s, [0.0] * 5)
                f[0] += 1.0
                f[1] += x
                f[2] += y
                f[3] += x * x
                f[4] += x * y

    def fit(self, subject):
        """(alpha, held-out now) for a subject, or None while its fit has too
        little weight or no spread in what was read."""
        f = self.fits.get(subject)
        lane = next((k for k, v in self.feeds.items() if subject in v), None)
        if not f or lane is None or f[0] < self.MIN_EFF or self.read[lane] <= 0:
            return None
        w, sx, sy, sxx, sxy = f
        var = sxx / w - (sx / w) ** 2
        if var <= 1e-12:
            return None
        b = (sxy / w - (sx / w) * (sy / w)) / var
        a = sy / w - b * sx / w
        return -b, math.exp(a + b * math.log(self.read[lane]))

    def worth(self, lane):
        """What the next character of this lane is expected to take off the
        sum of held-out, in nats - or None with no evidence."""
        got = [self.fit(s) for s in self.feeds.get(lane, [])]
        got = [g for g in got if g is not None]
        if not got or self.read[lane] <= 0:
            return None
        return sum(abs(al) * L for al, L in got) / self.read[lane]

    def reread(self, lane):
        """What reading this lane again is worth against new text."""
        size = self.sizes.get(lane, 0.0)
        if size <= 0:
            return 1.0
        return math.exp(-max(0.0, self.read[lane] / size - 1.0) / self.R_STAR)

    def targets(self):
        """The shares the evidence asks for, inside the limits."""
        n = len(self.names)
        if self.mode != "auto" or n == 0:
            return dict(self.shares)
        val = {k: self.worth(k) for k in self.names}
        known = sorted(v for v in val.values() if v is not None)
        if not known or known[-1] <= 0:
            return {k: 1.0 / n for k in self.names}
        typical = known[len(known) // 2]
        for k in self.names:
            if val[k] is None:
                val[k] = typical * self.reread(k)
        tot = sum(val.values())
        t = {k: val[k] / tot for k in self.names}
        return self._clip(t, self.FLOOR / n, self.CEIL / n)

    @staticmethod
    def _clip(t, lo, hi):
        """Shares held inside [lo, hi], the rest spread over the others in
        proportion, until nothing moves."""
        t = dict(t)
        fixed = {}
        for _ in range(len(t) + 1):
            free = {k: v for k, v in t.items() if k not in fixed}
            room = 1.0 - sum(fixed.values())
            s = sum(free.values())
            if s <= 0:
                for k in free:
                    t[k] = room / max(len(free), 1)
                break
            t.update({k: v * room / s for k, v in free.items()})
            out = {k: (lo if t[k] < lo else hi) for k in free
                   if t[k] < lo or t[k] > hi}
            if not out:
                break
            fixed.update(out)
            t.update(out)
        return t

    def step(self):
        """Move the shares toward the targets, at most STEP each in log
        terms. Returns a line naming the subjects that moved, or ''."""
        if self.mode != "auto":
            return ""
        t = self.targets()
        new = {}
        for k in self.names:
            s = max(self.shares[k], 1e-9)
            r = math.log(max(t[k], 1e-9) / s)
            new[k] = s * math.exp(max(-self.STEP, min(self.STEP, r)))
        tot = sum(new.values())
        new = {k: v / tot for k, v in new.items()}
        moved = sorted(((new[k] - self.shares[k], k) for k in self.names),
                       key=lambda m: -abs(m[0]))
        self.shares = new
        if abs(moved[0][0]) < 0.001:
            return ""
        return "reading shares: " + ", ".join(
            f"{k} {100 * new[k]:.1f}% ({100 * d:+.1f})"
            for d, k in moved[:3] if abs(d) >= 0.001)

    # --------------------------------------------------------------- reading
    def pick(self):
        """The subject to read next: the one with the most credit."""
        return max(self.names, key=lambda n: self.credit[n])

    def note_read(self, lane, chars):
        """A visit to `lane` read `chars` characters."""
        if lane in self.read:
            self.read[lane] += float(chars)
            self._pay(lane, chars)

    def missed(self, lane, chars):
        """A visit to `lane` read nothing: it is charged as `chars`, a full
        visit, and nothing is counted as read."""
        if lane in self.read:
            self._pay(lane, chars)

    def _pay(self, lane, chars):
        """Every subject gains its share of `chars` in credit; `lane`, which
        read them, pays them back. The credits keep their sum."""
        c = float(chars)
        for k in self.names:
            self.credit[k] += self.shares[k] * c
        self.credit[lane] -= c

    # ----------------------------------------------------------- start, save
    def describe(self):
        order = sorted(self.names, key=lambda k: -self.shares[k])
        what = {"auto": "where reading still pays", "equal": "evenly",
                "fixed": "by fixed weights"}[self.mode]
        return (f"reading shared {what}: "
                + ", ".join(f"{k} {100 * self.shares[k]:.1f}%" for k in order))

    def bootstrap(self, rows):
        """Fits from a sample log's checks (minagi.heldout.parse_log rows),
        for a run that has none yet. Before this schedule the subjects took
        equal turns, so each lane had read an equal part of the whole."""
        n = len(self.names)
        keep = dict(self.read)
        for r in rows:
            dom, total = r.get("dom"), (r.get("chars") or 0) * 1e6
            if not dom or total <= 0:
                continue
            for k in self.names:
                self.read[k] = total / n
            self.observe(total, dom)
        self.read = keep

    def state(self):
        return {"mode": self.mode, "unit": "characters",
                "shares": dict(self.shares),
                "read": dict(self.read), "fits": {k: list(v) for k, v
                                                  in self.fits.items()},
                "last": self.last, "credit": dict(self.credit)}

    @classmethod
    def restore(cls, d, lanes, sizes, subjects=(), mode="auto", weights=None,
                total=0.0, per_visit=None):
        """A schedule for these lanes, carrying on from saved state `d` when
        there is one and it was the same mode. A lane the state does not know
        - the corpus grew one - starts at an even share and an even part of
        what has been read.

        A state saved while the shares counted visits carries on as the
        characters those visits read: each share times what one visit to its
        lane reads, `per_visit`, so the reading does not jump. Its credits
        counted visits too, and start again from zero."""
        s = cls(lanes, sizes, subjects, mode, weights)
        n = max(len(s.names), 1)
        for k in s.names:
            s.read[k] = total / n
        if not d or d.get("mode") != mode or mode == "fixed":
            return s
        old = {k: float(v) for k, v in d.get("shares", {}).items()
               if k in s.names}
        credit = d.get("credit", {})
        if d.get("unit") != "characters":
            credit = {}
            if mode == "auto":
                old = _in_characters(old, per_visit)
        new = [k for k in s.names if k not in old]
        kept = sum(old.values())
        room = 1.0 - len(new) / n
        s.shares = {k: (old[k] * room / kept if k in old and kept > 0
                        else 1.0 / n) for k in s.names}
        s.read.update({k: float(v) for k, v in d.get("read", {}).items()
                       if k in s.read})
        s.fits = {k: [float(x) for x in v] for k, v in d.get("fits", {}).items()}
        s.last = d.get("last")
        s.credit.update({k: float(v) for k, v in credit.items()
                         if k in s.credit})
        return s


def _in_characters(shares, per_visit):
    """Shares of visits as the shares of the characters they read: each
    times what one visit to its lane reads. A lane with no figure is taken
    at the typical one."""
    known = [v for k, v in (per_visit or {}).items() if k in shares and v > 0]
    if not known:
        return shares
    typical = sum(known) / len(known)
    w = {k: v * ((per_visit or {}).get(k) or typical) for k, v in shares.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else shares
