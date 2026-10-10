"""GradSNR.ratio: no bias correction on averages that start at the first reading."""

import torch

from minagi.optim import GradSNR


def test_constant_gradient_reads_one():
    a = torch.nn.Parameter(torch.zeros(4))
    snr = GradSNR()
    for n in range(1, 401):
        a.grad = torch.full((4,), 0.5)
        r = snr.observe([a])
        if n >= 8:                    # from the first reading, not only late
            assert abs(r - 1.0) < 1e-5, (n, r)


def test_noise_reads_below_one_early():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.zeros(256))
    snr = GradSNR()
    for _ in range(8):
        a.grad = torch.randn(256)
        r = snr.observe([a])
    assert 0.0 <= r < 1.0
