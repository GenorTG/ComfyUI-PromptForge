"""PromptForge MiniMax H3 VAE Decode: the H3 video decoder with an out-of-memory ladder.

The stock decode gives up on an OOM by switching to a generic tiler that ignores H3's blend rules. This node
calls H3's own decoder instead and, when memory runs out, halves the frames per pass and then the tile size,
which keeps the same quality. Whole-frame or larger tiles look fried and banded with H3, so they are never
chosen automatically.

The decoder emits 12 packed sub-pixel channels per pixel for the 2x detail VAEs, so the frames are unpacked to
RGB inside the decoder: blending and the output buffer then work on the final video instead of on four times
the sub-pixels. Decoding straight into the finished buffer also means the clip is never copied a second time.

A clip decodes in one pass straight into its output frames. Only when a pass runs out of memory is it split into
shorter passes with causal context. Every pass also decodes one group past its end, so its last frames are
decoded against the real continuation and the passes are pixel-identical to the whole clip. How far a latent
size had to be split is remembered, so the next run starts there. Non-finite pixels are repaired and reported.
"""
import gc
import math

import torch

import comfy.model_management as model_management
import comfy.utils

from . import frame_buffer

CONTEXT_GROUPS = 1  # latents kept before a pass for the causal context; one group (5) already covers the 3 the decoder trims
TILES_PER_CALL_MAX = 16
RESERVE = 512 << 20
AUTO_GROUPS = 4  # pass size the first out-of-memory drops the whole clip to; out-of-memory halves it from there
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


def _check_packing(inner):
    """The decoder must emit three RGB channels or packed sub-pixels of them, or the buffer sizes below do not mean anything."""
    decoder = inner.decoder
    channels = decoder.proj_out.weight.shape[0] // (decoder.patch_size_t * decoder.patch_size ** 2)
    factor = math.isqrt(channels // 3)
    if channels % 3 or factor * factor * 3 != channels:
        raise ValueError(f"[PromptForge] the H3 decoder emits {channels} channels per pixel, which is not a packed RGB layout.")


def _output(vae, video):
    """(dtype, bytes) of the finished frames, which is exactly the buffer the decode writes into; float16 when float32 would crowd the RAM."""
    values = math.prod(vae.first_stage_model.decode_output_shape(video.shape))
    dtype = vae.vae_output_dtype()
    if dtype == torch.float32:
        dtype = frame_buffer.video_dtype(values, 0.25)
    return dtype, values * model_management.dtype_size(dtype)


def _destination(choice, final):
    """Where the finished frames are kept: auto puts them on the GPU while the clip and its passes fit there."""
    if choice == "gpu":
        return model_management.get_torch_device()
    if choice == "cpu":
        return model_management.intermediate_device()
    gpu = model_management.get_torch_device()
    return gpu if model_management.get_free_memory(gpu) > 4 * final + RESERVE else model_management.intermediate_device()


def _decode(vae, video, dest, work, dtype, drop=0):
    """One decode pass of `video` into a buffer on `work` (the dest itself for a single pass); frames end up on `dest`."""
    inner = vae.first_stage_model
    with model_management.cuda_device_context(vae.device):
        pixels = frame_buffer.empty(inner.decode_output_shape(video.shape), dtype, work)
        inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=pixels)
        vae.process_output(pixels)
    frames = pixels.movedim(1, -1)[:, drop:]
    del pixels
    return frames if frames.device == dest else frames.to(dest)


def _windows(latent_steps, groups):
    """(first latent step, end latent step, frames to drop from the front) for each pass over a 2 + 5n latent."""
    total = (latent_steps - 2) // 5
    done = 0
    while done < total:
        core = min(groups, total - done)
        context = min(CONTEXT_GROUPS, done)
        yield 5 * (done - context), 2 + 5 * (done + core), (5 + 17 * context) if done else 0
        done += core


def _run(vae, video, dest, dtype, tile, groups, tiles_per_call):
    inner = vae.first_stage_model
    inner.tiling = tile > 0
    inner.tiles_per_call = tiles_per_call
    if tile:
        inner.tile_size, inner.tile_overlap_min = tile, max(64, tile // 4)
    total = (video.shape[2] - 2) // 5
    if not groups or video.shape[2] < 2 or (video.shape[2] - 2) % 5 or groups >= total:
        return _decode(vae, video, dest, dest, dtype)

    # a pass decodes on the GPU and leaves only its finished frames on `dest`, so a long clip never needs a
    # whole-clip buffer anywhere and the packed sub-pixels never cross the bus
    work = vae.device if vae.device.type == "cuda" else dest
    windows = list(_windows(video.shape[2], groups))
    progress = comfy.utils.ProgressBar(len(windows))
    out = None
    position = 0
    for start, end, drop in windows:
        model_management.throw_exception_if_processing_interrupted()
        # one more group of latents after the pass, decoded only for its continuation: without it the last 4
        # frames of every pass would be decoded against the clip end instead of what really follows
        tail = min(5, 2 + 5 * total - end)
        piece = _decode(vae, video[:, :, start:end + tail], dest, work, dtype, drop)
        if tail:
            piece = piece[:, :5 + 17 * ((end - start - 2) // 5) - drop]
        if out is None:
            frames = 5 + 17 * total
            # frames-first, so the final reshape is a view and the clip is not copied once more at the size of a whole video
            out = frame_buffer.empty((frames, *piece.shape[2:]), piece.dtype, piece.device)
        part = piece.reshape(-1, *piece.shape[2:])
        out[position:position + part.shape[0]] = part
        position += part.shape[0]
        progress.update(1)
        del part, piece
    return out[:position].unsqueeze(0)


class PROMPTFORGE_MiniMaxDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "0 = auto: 256 px tiles, the size H3's decoder was built for. Larger tiles or the whole frame decode faster but look fried and banded. Overlap is a quarter of the tile."}),
                "frames_per_pass": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 17, "tooltip": "0 = auto: the whole clip in one pass, which is the fastest. A number decodes it in passes of about that many frames with causal context: less VRAM, but each pass decodes its context again."}),
                "tiles_per_call": ("INT", {"default": 0, "min": 0, "max": TILES_PER_CALL_MAX, "tooltip": "0 = auto: from free memory. Neighbouring tiles decode in one batched call: more is faster and uses more VRAM."}),
                "output_device": (["auto", "gpu", "cpu"], {"default": "auto", "tooltip": "Where the finished frames are kept. auto: gpu while the whole clip fits there four times over, else cpu. gpu skips the copies to RAM but needs VRAM for the whole clip."}),
                "retry_on_oom": ("BOOLEAN", {"default": True, "tooltip": "On out-of-memory, decode again in shorter passes and with fewer tiles instead of failing."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "INFO")
    FUNCTION = "decode"
    CATEGORY = "🔮 PromptForge/Video"
    DESCRIPTION = "MiniMax H3 video decode with 256px tiles that steps down to shorter passes on out-of-memory instead of failing, and remembers what worked."

    def decode(self, samples, vae, tile_size, frames_per_pass, tiles_per_call, output_device, retry_on_oom):
        inner = _inner(vae)
        video = _video(samples)
        _check_packing(inner)
        key = tuple(video.shape)

        vae.throw_exception_if_invalid()
        with model_management.cuda_device_context(vae.device):
            model_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype), force_full_load=vae.disable_offload)
        model_management.soft_empty_cache()  # the plan below must see the memory that is really left, not reserved blocks
        dtype, final = _output(vae, video)
        dest = _destination(output_device, final)
        total = (video.shape[2] - 2) // 5
        tile = max(256, tile_size // 16 * 16) if tile_size else 256
        # one pass writes the frames straight into the output, so shorter passes only save VRAM; start with the
        # whole clip and only the pass size this latent size already had to drop to wins over that
        groups = min(total, _worked.get(key, total))
        if frames_per_pass:
            groups = min(total, max(1, round(frames_per_pass / 17)))

        saved = (inner.tiling, inner.tile_size, inner.tile_overlap_min, inner.tiles_per_call)
        failures = 0
        try:
            while True:
                try:
                    images = _run(vae, video, dest, dtype, tile, groups, tiles_per_call)
                except model_management.OOM_EXCEPTION:
                    gc.collect()
                    model_management.soft_empty_cache()
                    if not retry_on_oom or groups == 1 and tile <= 256:
                        raise
                    failures += 1
                    if groups > AUTO_GROUPS:
                        groups = AUTO_GROUPS
                    else:
                        groups = (groups + 1) // 2
                        if groups == 1:
                            tile = 256
                    if dest.type == "cuda":
                        dest = model_management.intermediate_device()
                    continue
                break
        finally:
            inner.tiling, inner.tile_size, inner.tile_overlap_min, inner.tiles_per_call = saved

        if not frames_per_pass and groups < _worked.get(key, total):
            _worked[key] = groups
        repaired = False
        for start in range(0, images.shape[1], 16):
            part = images[:, start:start + 16]
            if not torch.isfinite(part).all():
                torch.nan_to_num_(part, nan=0.0, posinf=1.0, neginf=0.0)
                repaired = True

        how = f"{tile}px tiles" + (f", {17 * groups} frames per pass" if groups < total else "")
        info = (f"{images.shape[1]} frames {images.shape[3]}x{images.shape[2]}, {how}, {tiles_per_call or 'auto'} tiles per call, kept in {str(dest).upper()}"
                + (f" (after {failures} out-of-memory retries)" if failures else "") + (", non-finite pixels were repaired" if repaired else ""))
        return (images.reshape(-1, *images.shape[-3:]), info)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_MiniMaxDecode": PROMPTFORGE_MiniMaxDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_MiniMaxDecode": "PromptForge MiniMax H3 VAE Decode"}
