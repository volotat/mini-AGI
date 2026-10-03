"""Self-directed learning: the model steers its own rate and growth, bounded.

The loop renders a <self> telemetry block; the model may continue it with a
<policy> directive. Everything is advisory: the LR multiplier is clamped to
[LR_MIN, LR_MAX] and plast.scale stays inside Plasticity.FLOOR/CEIL.
"""

import re

S0, S1 = "<self>", "</self>"
P0, P1 = "<policy>", "</policy>"
LR_MIN, LR_MAX = 0.5, 2.0

_LR = re.compile(r"lr\s+x([0-9.]+)")
_GROW = re.compile(r"grow\s+([01])")


class Directive:
    def __init__(self, kind, value):
        self.kind = kind          # "lr" | "grow"
        self.value = value


class DirectiveLog:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


def render_telemetry(step, chars, train_loss, held_loss, plast, pool, grower):
    sat = pool.saturation() if hasattr(pool, "saturation") else {}
    n = sat.get("experts", 0)
    idle = sat.get("idle", 0)
    kr = getattr(grower, "keep_ratio", 1.0)
    scale = getattr(plast, "scale", 1.0)
    t = getattr(plast, "last_t", 0.0)
    e = getattr(plast, "last_e", 0.0)
    return (f"{S0}\n"
            f"step {step} chars {chars} loss {train_loss:.4f} "
            f"held {held_loss:.4f}\n"
            f"lr scale x{scale:.3f} verdict t {t:+.2f} e {e:+.3f}\n"
            f"pool experts {n} idle {idle} keep {kr:.2f}\n"
            f"{S1}\n")


def parse_directive(text):
    """The first well-formed <policy> block, or None."""
    if not text:
        return None
    start = text.find(P0)
    if start < 0:
        return None
    body_start = start + len(P0)
    end = text.find(P1, body_start)
    if end < 0:
        return None
    body = text[body_start:end]
    m = _LR.search(body)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            return None
        return Directive("lr", max(LR_MIN, min(LR_MAX, v)))
    m = _GROW.search(body)
    if m:
        return Directive("grow", int(m.group(1)))
    return None


def apply_directive(plast, grower, directive):
    """Apply one directive. Returns a human-readable note."""
    if directive.kind == "lr":
        before = float(plast.scale)
        lo = getattr(plast, "FLOOR", 0.05)
        hi = getattr(plast, "CEIL", 1.0)
        plast.scale = max(lo, min(hi, before * directive.value))
        return (f"selfdir lr x{directive.value:.2f}: "
                f"{before:.3f} -> {plast.scale:.3f}")
    if directive.kind == "grow":
        grower.selfdir_veto = (directive.value == 0)
        return f"selfdir grow {directive.value}"
    return "selfdir: ignored"
