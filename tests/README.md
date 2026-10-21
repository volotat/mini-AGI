# Precision regression tests

Run from the repository root with PyTorch installed:

```sh
python -m unittest discover -s tests -v
```

The policy tests use CPU-side mocks and need no accelerator. They cover
XPU device strings/objects, bf16/fp16/fp32 selection, autocast dispatch,
GPU overrides, and unchanged CPU/CUDA/ROCm precision policy.

Optionally test a fresh 16-to-8 linear layer on `xpu:0`:

```sh
MINAGI_TEST_XPU=1 python -m unittest discover -s tests -v
```

This checks bf16 output, finite fp32 gradients/parameters, and an SGD update;
it never loads model weights or training data. Validated on Arc Pro B60
with PyTorch 2.14.1+xpu, not a claim about all Intel GPUs or training speed.

XPU now uses the configured low-precision autocast path. XPU bf16 policy
does not use CUDA properties (including on mixed CUDA/XPU systems).
`MINAGI_GPU_BF16` still overrides the native-bf16 policy predicate, but the
CUDA/ROCm VRAM-based fp32 fallback does not apply to XPU; set the compute
dtype to fp32 to disable XPU autocast. fp16 remains an explicit choice;
this change does not add gradient scaling or recommend fp16 training.
