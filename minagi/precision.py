"""
What precision the model computes in.

One setting for the whole process, because reading and generating are the same
code path and must not differ in what they cost or what they compute.

Only the arithmetic and the activations move. WEIGHTS stay fp32, on disk as
well as in memory, and that is not caution: an expert here pages out to RAM
and back several times per chunk, and every page-out in a shorter format
would be a fresh rounding. Measured on the real pool, against a typical Adam
update of 1.8e-05 (weight |w| median 1.3e-02):

    weights bfloat16   err 1.4e-05    0.80x one update
    weights float16    err 1.8e-06    0.10x one update

bfloat16 would round away most of what an expert had just learned, over and
over. float16 looks affordable and is not: its resolution at a typical weight
is 1.3e-05 against an update of 1.8e-05, so updates land barely above the
grid, and an expert is rewritten a median of 406 times.

Adam's MOMENTS are a different question with a different answer, and holding
them to the weights' standard would cost gigabytes for nothing. They are tiny
and span an enormous range - m median 3.5e-09, v median 2.0e-15 - so what they
need is exponent, which bfloat16 keeps in full, and not mantissa. Measured the
same way:

    m bfloat16   0.10% of its own magnitude    0.000x one update
    v bfloat16   0.10% of its own magnitude    0.000x one update
    m float16    86% - underflows              (v: 100%, to zero)

float16 destroys them outright: both sit far below its smallest subnormal.
bfloat16 costs a thousandth of their magnitude and two thirds of the file, so
the moments are stored bf16 and the weights fp32. Nothing re-derives a weight,
but Adam re-derives its moments continuously, which is the other reason the
error does not accumulate there.

What autocast does buy is the part that is actually large: the KV cache is 26
block-applications of keys and values across the whole context window, and it
halves. bfloat16 needs no loss scaling - it keeps fp32's exponent range and
spends its bits on mantissa instead. The places that cannot afford a short
mantissa force themselves back to fp32 on the spot: the halting accumulator,
the cross-entropy, the router logits and RMSNorm.
"""

import os

import torch

_COMPUTE = {"dtype": torch.float32}

NAMES = {"fp32": torch.float32, "float32": torch.float32,
         "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
         "fp16": torch.float16, "float16": torch.float16}


def set_compute_dtype(dtype):
    """Takes a torch dtype or one of the names above."""
    if isinstance(dtype, str):
        if dtype not in NAMES:
            raise ValueError(f"unknown precision {dtype!r}; "
                             f"expected one of {sorted(NAMES)}")
        dtype = NAMES[dtype]
    _COMPUTE["dtype"] = dtype


def compute_dtype():
    return _COMPUTE["dtype"]


def cpu_bf16_native():
    """
    Whether this CPU multiplies bfloat16 in hardware: AVX-512 BF16 (AMD Zen 4
    and later, recent Intel) or Intel AMX. Without it bf16 on the CPU is
    emulated and slower than fp32, so the CPU stays in fp32 there.
    MINAGI_CPU_BF16=1 or =0 overrides the detection, for testing.
    """
    force = os.environ.get("MINAGI_CPU_BF16")
    if force is not None:
        return force.strip().lower() not in ("0", "", "false", "no", "off")
    for name in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        f = getattr(torch.cpu, name, None)
        try:
            if f is not None and f():
                return True
        except Exception:                                  # noqa: BLE001
            pass
    return False


def _device_type(device):
    return "cuda" if device is None else (
        device.type if hasattr(device, "type") else str(device).split(":")[0])


# -- GPUs without bf16 arithmetic -----------------------------------------
#
# PyTorch runs bf16 on every GPU, converting where the hardware has no bf16
# arithmetic, and conversion is not cheap: on an RX 6800 XT (RDNA2, gfx1030)
# bf16 matmuls ran at 9 TFLOP/s against fp32's 15, and a whole learning step
# took 3.6x as long - 336 characters a second read against 1,242. Such a GPU
# computes in fp32 when it has the memory for it. fp32 holds its activations
# at twice the size - a 4,096-character reading step peaked at 13.2 GB there,
# against 7.6 in bf16 - so a smaller card keeps bf16: slower, but it fits.

FP32_MIN_GIB = 14
_GPU_INFO = {}


def _gpu_info(device=None):
    """CUDA/ROCm (properties, is it AMD); None for XPU or without a GPU."""
    if _device_type(device) == "xpu":
        return None                  # never query a colocated CUDA GPU for XPU
    try:
        idx = getattr(device, "index", None)
        if idx is None:
            idx = torch.cuda.current_device()
    except Exception:                                      # noqa: BLE001
        return None
    if idx not in _GPU_INFO:
        try:
            _GPU_INFO[idx] = (torch.cuda.get_device_properties(idx),
                              bool(torch.version.hip))
        except Exception:                                  # noqa: BLE001
            _GPU_INFO[idx] = None
    return _GPU_INFO[idx]


def _gpu_arch(info):
    props, amd = info
    return ((getattr(props, "gcnArchName", "") or "").split(":")[0] if amd
            else f"sm_{props.major}{props.minor}")


def gpu_bf16_native(device=None):
    """
    Whether this GPU multiplies bfloat16 in hardware: NVIDIA from Ampere
    (compute capability 8) on, AMD from RDNA3 (gfx11, gfx12) and the CDNA
    accelerators (gfx908, gfx90a, gfx94x, gfx95x). Not RDNA1 or 2, not Vega,
    not NVIDIA before Ampere. Intel XPU uses bf16 by default (validated on
    Arc Pro B60). MINAGI_GPU_BF16=1 or =0 overrides the detection; the
    CUDA/ROCm memory-based fp32 fallback does not apply to XPU.
    """
    force = os.environ.get("MINAGI_GPU_BF16")
    if force is not None:
        return force.strip().lower() not in ("0", "", "false", "no", "off")
    if _device_type(device) == "xpu":
        return True                  # use bf16 for Intel XPU; override above
    info = _gpu_info(device)
    if info is None:
        return True                  # nothing to ask: the rule as it always was
    if info[1]:
        arch = _gpu_arch(info)
        return (arch.startswith(("gfx11", "gfx12", "gfx94", "gfx95"))
                or arch in ("gfx908", "gfx90a"))
    return info[0].major >= 8


def gpu_fp32_instead(device=None):
    """bf16 configured, on a GPU without bf16 arithmetic and with the memory
    for fp32 (FP32_MIN_GIB): that GPU computes in fp32."""
    if _COMPUTE["dtype"] is not torch.bfloat16 or gpu_bf16_native(device):
        return False
    info = _gpu_info(device)
    return info is not None and info[0].total_memory >= FP32_MIN_GIB * 2**30


def autocast_on(device=None):
    """
    Whether forwards on `device` compute in the configured dtype. On a GPU
    whenever that dtype is not fp32, unless it is bf16 and the GPU has no bf16
    arithmetic but has the memory for fp32 (gpu_fp32_instead). On the CPU only
    for bfloat16 and only where the CPU has it natively: there it is the GPU
    path's own precision at about twice fp32's arithmetic rate, and everywhere
    else it would be slower.
    """
    dt = _COMPUTE["dtype"]
    dev = _device_type(device)
    if dt is torch.float32:
        return False
    if dev in ("cuda", "xpu"):
        return not gpu_fp32_instead(device)
    return dev == "cpu" and dt is torch.bfloat16 and cpu_bf16_native()


def describe(device=None):
    """What forwards on `device` compute in, and why when it is not what was
    configured - for the line a run starts with."""
    dt = _COMPUTE["dtype"]
    name = {torch.float32: "fp32", torch.bfloat16: "bf16",
            torch.float16: "fp16"}.get(dt, str(dt))
    dev = _device_type(device)
    info = _gpu_info(device) if dev == "cuda" else None
    gpu = f"this GPU ({_gpu_arch(info)})" if info else "this GPU"
    if autocast_on(device):
        if dev == "cuda" and dt is torch.bfloat16 and not gpu_bf16_native(device):
            return (f"bf16, which {gpu} converts rather than computes - fp32 is "
                    f"faster on it but needs a {FP32_MIN_GIB} GB card")
        return name
    if dt is torch.float32:
        return "fp32"
    if dev == "cuda":
        return f"fp32 - {gpu} has no {name} arithmetic, and fp32 is faster on it"
    return f"fp32 - this CPU has no {name} arithmetic"


def dispatch_dtype(device, fallback):
    """The dtype the expert dispatch computes in: the configured one wherever
    autocast is on, `fallback` - the residual stream's - where it is off."""
    return _COMPUTE["dtype"] if autocast_on(device) else fallback


def amp(device=None):
    """The autocast region every forward runs inside."""
    dev = _device_type(device)
    return torch.autocast(dev, dtype=_COMPUTE["dtype"], enabled=autocast_on(dev))


# -- storing moments in half the space ------------------------------------
#
# numpy has no bfloat16, so a bf16 array travels as int16 with the same bits.
# The dtype IS the marker: an array that comes back float32 was written by an
# older version and is used as-is, so old weight directories keep loading and
# convert themselves the first time each expert is written out.

def pack_bf16(t):
    """A float tensor as int16 carrying bfloat16 bits, for storage."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return t.to(torch.bfloat16).view(torch.int16).numpy()


def unpack_bf16(a):
    """Undo pack_bf16. Passes float arrays through, so old files still load."""
    t = torch.as_tensor(a)
    if t.dtype == torch.int16:
        return t.view(torch.bfloat16).to(torch.float32)
    return t.to(torch.float32)


def as_bf16(a):
    """A stored moment as a bfloat16 tensor: pack_bf16's int16 bits viewed as
    what they are, or a float array narrowed as pack_bf16 would narrow it."""
    t = torch.as_tensor(a)
    if t.dtype == torch.int16:
        return t.view(torch.bfloat16)
    return t.to(torch.bfloat16)


def is_moment(name):
    """Whether a saved array is an Adam moment rather than a weight.

    Expert files name them w1_m / w1_v; the trunk's optimiser file uses
    `<param>|m` and `<param>|v`. `|t` is a step count and stays as it is.
    """
    return name.endswith(("_m", "_v", "|m", "|v"))
