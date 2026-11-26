"""Optional same-size NVIDIA RTX VFX cleanup (denoise and deblur) for decoded frames.

Runs the nvidia-vfx effects frame by frame so no second full batch exists. Both
passes keep the frame size: denoise lifts decode grain before interpolation and
deblur restores the skin and contact detail, so nothing is resampled upwards.
"""
import contextlib
import gc

import torch
import torch.nn.functional as F

import comfy.model_management as mm

from .. import frame_buffer

QUALITIES = ["Low", "Medium", "High", "Ultra"]


def _effect(kind, quality, device):
    from nvvfx import VideoSuperRes

    level = getattr(VideoSuperRes.QualityLevel, f"{kind.upper()}_{quality.upper()}")
    return VideoSuperRes(quality=level, device=device)


def aligned_width(width):
    # nvidia-vfx corrupts rows at widths that are not a multiple of 8
    return ((int(width) + 7) // 8) * 8


def _passes(denoise, denoise_quality, deblur, deblur_quality, device):
    if denoise:
        yield _effect("Denoise", denoise_quality, device)
    if deblur:
        yield _effect("Deblur", deblur_quality, device)


def clean(frames, denoise=False, denoise_quality="High", deblur=False, deblur_quality="High", device=0):
    if not denoise and not deblur:
        return frames
    if not torch.cuda.is_available():
        raise RuntimeError("[PromptForge] the RTX cleanup needs an NVIDIA RTX GPU and the nvidia-vfx runtime.")
    count, height, width = frames.shape[:3]
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    cuda = torch.device(f"cuda:{device}")
    passes = " + ".join(name for name, on in (("denoise", denoise), ("deblur", deblur)) if on)
    print(f"[PromptForge] RTX cleanup {passes} on {count} frames at {width}x{height}", flush=True)
    padded = aligned_width(width)
    out = frame_buffer.empty((count, height, width, 3), frames.dtype, frames.device)
    with contextlib.ExitStack() as stack:
        effects = [stack.enter_context(effect) for effect in
                   _passes(denoise, denoise_quality, deblur, deblur_quality, device)]
        for effect in effects:
            effect.output_width = padded
            effect.output_height = height
            effect.load()
        for index in range(count):
            frame = frames[index, :, :, :3].float().to(cuda).permute(2, 0, 1).contiguous()
            if padded != width:
                frame = F.pad(frame, (0, padded - width, 0, 0), mode="replicate").contiguous()
            for effect in effects:
                frame = torch.from_dlpack(effect.run(frame).image).clone().contiguous()
            out[index].copy_(frame[:, :, :width].permute(1, 2, 0).clamp(0.0, 1.0).to(out.device, out.dtype))
            del frame
    return out
