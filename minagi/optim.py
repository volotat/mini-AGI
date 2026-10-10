"""
A meter for how much of a gradient is signal.
"""

import torch


class GradSNR:
    """
    How much of the gradient is signal, measured every step.

    The held-out signal this project steers the learning rate by moves 0.0014
    per evaluation against 0.018 of noise, so it takes over a hundred readings
    to say anything - twenty hours at a ten-minute checkpoint. This is the
    same question asked of a quantity that is available on every step.

    Keep an average of the gradient and an average of its squared norm. If
    successive gradients agree, the average keeps its length and the ratio
    ||mean||^2 / mean(||g||^2) approaches one. If they are independent noise
    the average shrinks toward zero and so does the ratio. It is the gradient
    noise scale of McCandlish et al. 2018, in the cheapest form that answers
    the question: two scalars, no extra tensors.

    Reported, not acted on. A signal is watched for a while before anything is
    allowed to steer on it.
    """

    def __init__(self, beta=0.98):
        self.beta = beta
        self.m = None
        self.sq = 0.0
        self.n = 0

    @torch.no_grad()
    def observe(self, params):
        params = list(params)
        if all(p.grad is None for p in params):
            return None
        # A parameter that took no part in this step contributed nothing, and
        # counts as a zero gradient rather than as a shorter vector. The
        # halting head is one: a step whose sampled depth is a single pass
        # forces that pass to stop and never reads it - a few in a million
        # steps, and one of them landed on a reading here and stopped the run
        # 346 minutes in (8,134,656 values against an average of 8,135,169).
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                          .detach().float().reshape(-1) for p in params])
        if self.m is not None and self.m.numel() != flat.numel():
            # the set itself changed shape: a meter starts again, it does not
            # stop the run it is only reporting on
            self.m, self.sq, self.n = None, 0.0, 0
        self.m = flat.clone() if self.m is None else \
            self.m.mul_(self.beta).add_(flat, alpha=1 - self.beta)
        s = float((flat * flat).sum())
        self.sq = s if self.n == 0 else self.beta * self.sq + (1 - self.beta) * s
        self.n += 1
        return self.ratio()

    def ratio(self):
        """0 = pure noise, 1 = every step pointing the same way."""
        if self.m is None or self.sq <= 0 or self.n < 8:
            return None
        # No bias correction: both averages start at the first reading, not
        # at zero, so neither is biased toward zero. Dividing by 1 - beta^n
        # here inflated the ratio by that factor - 6.7x at the eighth reading,
        # and a constant gradient read above one.
        return float(self.m.pow(2).sum() / self.sq)
