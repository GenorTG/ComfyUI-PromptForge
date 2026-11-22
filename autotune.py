"""Picks model weight settings for this machine.

`weight_dtype = auto` asks `choose_weight_dtype`. The only measured fact is whether fp8 matrix multiplies
are really faster than bf16 on this GPU; it takes a fraction of a second and is stored in `autotune.json`
beside this file, keyed by GPU name and torch version, so it runs once per setup.
"""
import json
import time
from pathlib import Path

import torch

import comfy.model_management as model_management

CACHE = Path(__file__).with_name("autotune.json")
FP8_FAST_MIN_SPEEDUP = 1.2
# quantized storage formats that ComfyUI must keep as they are
_KEEP_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E5M2"}


def _timed(fn, repeats=20):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def _measure_fp8_speedup(device):
    """bf16 matmul time / fp8 matmul time at a typical transformer shape; 0 when fp8 matmul is unavailable."""
    size = 4096
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    a8, b8 = a.to(torch.float8_e4m3fn), b.t().contiguous().to(torch.float8_e4m3fn).t()
    scale = torch.tensor(1.0, device=device)
    try:
        slow = _timed(lambda: a @ b)
        fast = _timed(lambda: torch._scaled_mm(a8, b8, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16))
    except (RuntimeError, TypeError):
        return 0.0
    return slow / fast


def fp8_speedup():
    """Cached fp8-vs-bf16 matmul speedup for the current GPU (0 on CPU or when unsupported)."""
    device = model_management.get_torch_device()
    if device.type != "cuda" or not model_management.supports_fp8_compute(device):
        return 0.0
    key = f"{torch.cuda.get_device_name(device)}|torch {torch.__version__}"
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    if key not in cache:
        cache[key] = round(_measure_fp8_speedup(device), 2)
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def choose_weight_dtype(stored_dtypes, model_bytes, optimize_for):
    """(weight_dtype, reason) for UNETLoader-style loading.

    stored_dtypes: safetensors dtype names in the file. model_bytes: file size.
    optimize_for: quality keeps the stored precision, balanced adds fp8 storage only when the model would not fit
    in VRAM, speed also uses fp8 compute when this GPU measures faster with it.
    """
    device = model_management.get_torch_device()
    if device.type != "cuda":
        return "default", "no GPU"
    if stored_dtypes & _KEEP_DTYPES:
        return "default", "file is already quantized"
    if optimize_for == "quality":
        return "default", "quality: keep stored precision"
    total = model_management.get_total_memory(device)
    if optimize_for == "speed":
        speedup = fp8_speedup()
        if speedup >= FP8_FAST_MIN_SPEEDUP:
            return "fp8_e4m3fn_fast", f"fp8 matmul measured {speedup:.1f}x faster than bf16 here"
    if model_bytes > 0.85 * total:
        return "fp8_e4m3fn", f"{model_bytes / 1024 ** 3:.1f} GB does not fit {total / 1024 ** 3:.0f} GB VRAM"
    return "default", "fits in VRAM"
