"""PromptForge MiniMax H3 VAE Decode: the H3 video decoder with an out-of-memory ladder.

The stock decode gives up on an OOM by switching to a generic tiler that ignores H3's blend rules. This node
calls H3's own decoder instead and, when memory runs out, steps down a ladder that keeps the same quality:

    whole frame -> 512 px tiles -> 256 px tiles -> 256 px tiles + 5 / 2 / 1 groups of frames per pass

The rung that worked for a given latent size is remembered, so the next run does not repeat the failed
attempts. Output is written into one preallocated buffer (no list-then-concatenate), non-finite pixels are
repaired and reported, and a model that has a 2x packed output projection (like the Wan 2x VAE trick) is
pixel-shuffled a few frames at a time.
"""
import gc
import math

import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.utils

CONTEXT_GROUPS = 2
# (spatial tile, 0 = whole frame; frame groups per pass, 0 = whole clip). One group = 17 frames, 5 latent steps.
LADDER = ((0, 0), (512, 0), (256, 0), (256, 5), (256, 2), (256, 1))
RUNG_NAMES = {0: "whole frame", 512: "512px tiles", 256: "256px tiles"}
SHUFFLE_FRAMES = 8
_worked = {}


def _inner(vae):
    inner = getattr(vae, "first_stage_model", None)
    if not all(hasattr(inner, name) for name in ("tiling", "tile_size", "tile_overlap_min", "decode_output_shape")):
        raise ValueError("[PromptForge] MiniMax H3 VAE Decode needs the MiniMax H3 video VAE (not the audio VAE).")
    return inner


def _video(samples):
    latent = samples["samples"]
    video = latent.unbind()[0] if latent.is_nested else latent
    if video.ndim != 5:
        raise ValueError(f"[PromptForge] expected an H3 video latent [B, C, T, H, W], got {tuple(video.shape)}.")
    return video


def _packing(inner):
    """PixelShuffle factor of a decoder whose projection emits packed RGB sub-pixels (1 for the stock H3 decoder)."""
    decoder = inner.decoder
    channels = decoder.proj_out.weight.shape[0] // (decoder.patch_size_t * decoder.patch_size ** 2)
    factor = math.isqrt(channels // 3)
    if channels % 3 or factor * factor * 3 != channels:
        raise ValueError(f"[PromptForge] the H3 decoder emits {channels} channels per pixel, which is not a packed RGB layout.")
    return channels, factor


def _expand(stat, channels):
    return stat if stat.shape[1] == channels else stat.repeat_interleave(channels // 3, dim=1)


def _shuffle(pixels, factor):
    """[B, C, T, H, W] packed channels -> [B, T, H * factor, W * factor, 3], a few frames at a time."""
    batch, channels, frames, height, width = pixels.shape
    out = torch.empty(batch, frames, height * factor, width * factor, channels // factor ** 2, device=pixels.device, dtype=pixels.dtype)
    for start in range(0, frames, SHUFFLE_FRAMES):
        part = pixels[:, :, start:start + SHUFFLE_FRAMES]
        count = part.shape[2]
        part = F.pixel_shuffle(part.movedim(1, 2).reshape(batch * count, channels, height, width), factor)
        out[:, start:start + count] = part.reshape(batch, count, -1, height * factor, width * factor).movedim(2, -1)
    return out


def _decode(vae, video, device, factor):
    """One decode pass of `video` with the tiling currently set on the model; returns [B, T, H, W, 3] on `device`."""
    vae.throw_exception_if_invalid()
    inner = vae.first_stage_model
    with model_management.cuda_device_context(vae.device):
        model_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype), force_full_load=vae.disable_offload)
        if factor > 1:  # a dynamic load can restore the 3-channel statistics, so expand them after it
            inner.pixel_mean = _expand(inner.pixel_mean, inner.decoder.out_channels)
            inner.pixel_std = _expand(inner.pixel_std, inner.decoder.out_channels)
        pixels = torch.empty(inner.decode_output_shape(video.shape), device=device, dtype=vae.vae_output_dtype())
        inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=pixels)
        vae.process_output(pixels)
        return _shuffle(pixels, factor) if factor > 1 else pixels.movedim(1, -1)


def _windows(latent_steps, groups):
    """(first latent step, end latent step, frames to drop from the front) for each pass over a 2 + 5n latent."""
    total = (latent_steps - 2) // 5
    done = 0
    while done < total:
        core = min(groups, total - done)
        context = min(CONTEXT_GROUPS, done)
        yield 5 * (done - context), 2 + 5 * (done + core), (5 + 17 * context) if done else 0
        done += core


def _run(vae, video, device, factor, rung):
    tile, groups = rung
    inner = vae.first_stage_model
    inner.tiling = tile > 0
    if tile:
        inner.tile_size, inner.tile_overlap_min = tile, max(64, tile // 4)
    if not groups or video.shape[2] < 2 or (video.shape[2] - 2) % 5 or (video.shape[2] - 2) // 5 <= groups:
        return _decode(vae, video, device, factor)

    windows = list(_windows(video.shape[2], groups))
    progress = comfy.utils.ProgressBar(len(windows))
    out = None
    position = 0
    for start, end, drop in windows:
        model_management.throw_exception_if_processing_interrupted()
        piece = _decode(vae, video[:, :, start:end], device, factor)[:, drop:]
        if out is None:
            frames = 5 + 17 * ((video.shape[2] - 2) // 5)
            out = torch.empty(piece.shape[0], frames, *piece.shape[2:], device=piece.device, dtype=piece.dtype)
        out[:, position:position + piece.shape[1]] = piece
        position += piece.shape[1]
        progress.update(1)
        del piece
    return out[:, :position]


class PROMPTFORGE_MiniMaxDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "0 = auto: start with the whole frame and step down on out-of-memory. A number forces tiles of that size (overlap = a quarter of it)."}),
                "frames_per_pass": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17, "tooltip": "0 = auto: the whole clip, then fewer frames per pass on out-of-memory. A number decodes the clip in passes of about that many frames, with causal context."}),
                "output_device": (["cpu", "gpu"], {"default": "cpu", "tooltip": "Where the finished frames are kept. gpu skips the copy to RAM but needs VRAM for the whole clip."}),
                "retry_on_oom": ("BOOLEAN", {"default": True, "tooltip": "On out-of-memory, step down to smaller tiles and shorter passes instead of failing."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "INFO")
    FUNCTION = "decode"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "MiniMax H3 video decode that steps down to smaller tiles and shorter passes on out-of-memory instead of failing, and remembers what worked."

    def decode(self, samples, vae, tile_size, frames_per_pass, output_device, retry_on_oom):
        inner = _inner(vae)
        video = _video(samples)
        channels, factor = _packing(inner)
        device = model_management.get_torch_device() if output_device == "gpu" else model_management.intermediate_device()
        key = tuple(video.shape)

        if tile_size or frames_per_pass:
            forced = (max(256, tile_size // 16 * 16) if tile_size else 0, max(1, round(frames_per_pass / 17)) if frames_per_pass else 0)
            unlimited = lambda value: value or 10 ** 9
            safer = [rung for rung in LADDER if unlimited(rung[0]) < unlimited(forced[0]) or rung[0] == forced[0] and unlimited(rung[1]) < unlimited(forced[1])]
            ladder = [forced] + (safer if retry_on_oom else [])
        else:
            ladder = list(LADDER[_worked.get(key, 0):]) if retry_on_oom else [LADDER[_worked.get(key, 0)]]

        saved = (inner.tiling, inner.tile_size, inner.tile_overlap_min, inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std)
        inner.decoder.out_channels = channels
        failures = []
        try:
            for index, rung in enumerate(ladder):
                try:
                    images = _run(vae, video, device, factor, rung)
                except model_management.OOM_EXCEPTION:
                    failures.append(rung)
                    gc.collect()
                    model_management.soft_empty_cache()
                    if index == len(ladder) - 1:
                        raise
                    continue
                break
        finally:
            inner.tiling, inner.tile_size, inner.tile_overlap_min, inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std = saved

        if not tile_size and not frames_per_pass and rung in LADDER:
            _worked[key] = LADDER.index(rung)
        repaired = False
        for start in range(0, images.shape[1], 16):
            part = images[:, start:start + 16]
            if not torch.isfinite(part).all():
                torch.nan_to_num_(part, nan=0.0, posinf=1.0, neginf=0.0)
                repaired = True

        how = RUNG_NAMES.get(rung[0], f"{rung[0]}px tiles") + (f", {17 * rung[1]} frames per pass" if rung[1] else "")
        info = f"{images.shape[1]} frames {images.shape[3]}x{images.shape[2]}, {how}" + (f" (after {len(failures)} out-of-memory retries)" if failures else "") + (", non-finite pixels were repaired" if repaired else "")
        return (images.reshape(-1, *images.shape[-3:]), info)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_MiniMaxDecode": PROMPTFORGE_MiniMaxDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_MiniMaxDecode": "PromptForge MiniMax H3 VAE Decode"}
