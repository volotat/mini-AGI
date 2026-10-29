"""Opt in with MINAGI_TEST_XPU=1; only fresh tiny tensors, never checkpoints."""
import os
import unittest
from unittest.mock import patch

import torch
from minagi import precision as p


@unittest.skipUnless(os.environ.get("MINAGI_TEST_XPU") == "1", "opt-in XPU smoke test")
class XpuSmokeTests(unittest.TestCase):
    def test_bf16_forward_backward(self):
        self.assertTrue(torch.xpu.is_available(), "XPU requested but unavailable")
        original = p.compute_dtype()
        self.addCleanup(p.set_compute_dtype, original)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINAGI_GPU_BF16", None)
            p.set_compute_dtype("bf16")
            device = torch.device("xpu:0")
            model = torch.nn.Linear(16, 8, device=device)
            inputs = torch.ones(4, 16, device=device)
            before = model.weight.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            with p.amp(device):
                output = model(inputs)
                loss = output.float().square().mean()
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            for parameter in model.parameters():
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(model.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            torch.xpu.synchronize(device)
            self.assertFalse(torch.equal(before, model.weight))
            print(f"XPU smoke: torch={torch.__version__}; device={torch.xpu.get_device_name(device)}; output={output.dtype}; finite gradients and SGD update OK")


if __name__ == "__main__":
    unittest.main()
