"""A complete-monotonicity health certificate for the agent's own trajectories.

Reuses the *verified* complete-monotonicity algebra from the RH work
(05_research/RH/lean4/CmRhCore/Basic.lean): exp(-mu x) is completely monotone,
cos(b x) is NOT, and finite non-negative combinations of CM functions are CM.
The discrete analogue holds for sequences: a_n = r^n (0<r<1) is discretely CM;
an oscillatory (cos-like) component breaks the alternating-sign condition.

A CM loss trajectory is a non-negative combination of geometric decays: it
monotonically improves and cannot bounce back. An oscillation flag is the
"cos-obstruction": a sign violation in the alternating finite differences. This
is a diagnostic only — it never mutates the model.
"""


def _differences(v):
    cur = [float(x) for x in v]
    levels = [cur]
    while len(cur) > 1:
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
        levels.append(cur)
    return levels


def discrete_cm(values, eps=1e-9):
    """(-1)^k * Delta^k a_n >= 0 for all k, n (alternating-sign differences)."""
    v = [float(x) for x in values]
    if len(v) <= 1:
        return {"cm": True, "first_flip": None, "depth": 0}
    levels = _differences(v)
    for k, level in enumerate(levels):
        sign = 1 if k % 2 == 0 else -1
        for n, x in enumerate(level):
            if sign * x < -eps:
                return {"cm": False, "first_flip": k,
                        "depth": len(levels) - 1}
    return {"cm": True, "first_flip": None, "depth": len(levels) - 1}


def certify(values, eps=1e-9):
    r = discrete_cm(values, eps)
    if r["cm"]:
        r["verdict"] = "no-oscillation (CM)"
    else:
        r["verdict"] = f"oscillation detected at diff {r['first_flip']}"
    return r
