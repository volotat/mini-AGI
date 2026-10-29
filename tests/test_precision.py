"""Precision policy tests; no accelerator or saved model is required."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from minagi import precision as p


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.dtype = p.compute_dtype()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("MINAGI_GPU_BF16", None)
        os.environ.pop("MINAGI_CPU_BF16", None)
        self.addCleanup(self.env.stop)
        self.addCleanup(p.set_compute_dtype, self.dtype)

    def test_xpu_bf16_policy_without_cuda_queries(self):
        p.set_compute_dtype("bf16")
        with patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA query")) as current, patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CUDA query")) as props:
            for device in ("xpu", "xpu:1", torch.device("xpu:1")):
                with self.subTest(device=device):
                    self.assertTrue(p.gpu_bf16_native(device))
                    self.assertTrue(p.autocast_on(device))
                    self.assertFalse(p.gpu_fp32_instead(device))
                    self.assertEqual(p.describe(device), "bf16")
                    self.assertEqual(p.dispatch_dtype(device, torch.float32), torch.bfloat16)
            current.assert_not_called()
            props.assert_not_called()

    def test_xpu_overrides_do_not_inspect_cuda(self):
        p.set_compute_dtype("bf16")
        # A colocated CUDA card must not decide XPU precision or memory policy.
        cuda_props = SimpleNamespace(major=7, total_memory=24 * 2**30)
        with patch.object(torch.cuda, "current_device", return_value=0) as current, patch.object(torch.cuda, "get_device_properties", return_value=cuda_props) as props, patch.dict(p._GPU_INFO, {}, clear=True):
            for force, native in (("0", False), ("false", False), ("1", True)):
                with self.subTest(force=force), patch.dict(os.environ, {"MINAGI_GPU_BF16": force}):
                    self.assertEqual(p.gpu_bf16_native(torch.device("xpu:0")), native)
                    self.assertIsNone(p._gpu_info(torch.device("xpu:0")))
                    self.assertTrue(p.autocast_on("xpu:0"))
                    self.assertFalse(p.gpu_fp32_instead("xpu:0"))
                    self.assertEqual(p.describe("xpu:0"), "bf16")
            current.assert_not_called()
            props.assert_not_called()

    def test_xpu_dtype_and_amp_dispatch(self):
        for name, dtype, enabled in (("fp32", torch.float32, False), ("bf16", torch.bfloat16, True), ("fp16", torch.float16, True)):
            with self.subTest(dtype=name):
                p.set_compute_dtype(name)
                self.assertEqual(p.autocast_on("xpu:0"), enabled)
                self.assertEqual(p.describe("xpu:0"), name)
                self.assertEqual(p.dispatch_dtype("xpu:0", torch.float64), dtype if enabled else torch.float64)
                with patch.object(torch, "autocast") as autocast:
                    p.amp(torch.device("xpu:0"))
                    autocast.assert_called_once_with("xpu", dtype=dtype, enabled=enabled)

    def test_cpu_policy_unchanged(self):
        for force in ("0", "1"):
            with patch.dict(os.environ, {"MINAGI_CPU_BF16": force}):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    p.set_compute_dtype(dtype)
                    self.assertEqual(p.autocast_on("cpu"), force == "1" and dtype is torch.bfloat16)

    def test_cuda_and_rocm_policy_unchanged(self):
        p.set_compute_dtype("bf16")
        for major, arch, amd, native in ((7, "", False, False), (8, "", False, True), (0, "gfx1030", True, False), (0, "gfx1100", True, True), (0, "gfx90a", True, True)):
            for gib in (8, 24):
                with self.subTest(major=major, arch=arch, gib=gib):
                    props = SimpleNamespace(major=major, minor=0, gcnArchName=arch, total_memory=gib * 2**30)
                    with patch.object(p, "_gpu_info", return_value=(props, amd)):
                        self.assertEqual(p.gpu_bf16_native("cuda"), native)
                        self.assertEqual(p.autocast_on("cuda"), native or gib < p.FP32_MIN_GIB)
                        self.assertEqual(p.autocast_on(), native or gib < p.FP32_MIN_GIB)
                        with patch.dict(os.environ, {"MINAGI_GPU_BF16": "1"}):
                            self.assertTrue(p.autocast_on("cuda"))
                        with patch.dict(os.environ, {"MINAGI_GPU_BF16": "0"}):
                            self.assertEqual(p.autocast_on("cuda"), gib < p.FP32_MIN_GIB)


if __name__ == "__main__":
    unittest.main()
