"""Buffers for whole clips that do not fail when RAM is short.

ComfyUI keeps copies of model weights pinned in RAM; those are released first. A clip that still does not fit lives in a
file in the temp folder that the system pages in and out as needed: slower, same result.
"""
import math
import os
import shutil
import uuid

import psutil
import torch

import comfy.model_management as model_management
import folder_paths

HEADROOM = 3 * 1024 ** 3  # RAM left for everything else in the run
DISK_SPARE = 2 * 1024 ** 3


def video_dtype(values, share=0.1):
    """float32 while the clip stays small next to the free RAM, else float16 (still finer than 10-bit video)."""
    return torch.float32 if values * 4 < share * psutil.virtual_memory().available else torch.float16


def empty(shape, dtype, device="cpu"):
    """torch.empty that moves a CPU buffer to disk instead of running out of RAM."""
    if torch.device(device).type != "cpu":
        return torch.empty(shape, dtype=dtype, device=device)
    count = math.prod(shape)
    size = count * model_management.dtype_size(dtype)
    shortfall = size + HEADROOM - psutil.virtual_memory().available
    if shortfall > 0:
        model_management.free_pins(shortfall, evict_active=True)
    if size + HEADROOM <= psutil.virtual_memory().available:
        return torch.empty(shape, dtype=dtype)
    folder = folder_paths.get_temp_directory()
    os.makedirs(folder, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if size + DISK_SPARE > free:
        raise ValueError(f"[PromptForge] a {size / 1024 ** 3:.1f} GB clip fits neither in free RAM nor on the disk of the temp folder ({free / 1024 ** 3:.1f} GB free).")
    path = os.path.join(folder, f"promptforge_frames_{uuid.uuid4().hex}.bin")
    with open(path, "wb") as file:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(file.fileno(), 0, size)  # a full disk fails here instead of crashing while frames are written
        else:
            file.truncate(size)
    buffer = torch.from_file(path, shared=True, size=count, dtype=dtype).view(tuple(shape))
    try:
        os.unlink(path)  # the mapping keeps the data and the space returns when the frames are dropped
    except OSError:
        pass  # Windows keeps mapped files; ComfyUI empties the temp folder when it starts
    print(f"[PromptForge] low RAM: keeping a {size / 1024 ** 3:.1f} GB clip in the temp folder instead.")
    return buffer
