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
        # A parameter with no gradient contributed a zero gradient this step,
        # so it is counted as zeros rather than left out. Leaving it out made
        # the flattened vector change length whenever a parameter sat a step
        # out - which recurrence depth sampling makes routine: a step sampled
        # at depth 1 forces the halt, and the halting head receives nothing -
        # and the running mean then no longer matched it.
        if all(p.grad is None for p in params):
            return None
        flat = torch.cat([(p.grad if p.grad is not None
                           else torch.zeros_like(p)).detach().float().reshape(-1)
                          for p in params])
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
        c = 1 - self.beta ** self.n                  # bias correction
        return float((self.m / c).pow(2).sum() / (self.sq / c))
