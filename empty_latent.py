"""PromptForge Empty Latent: one empty-latent node for every model family, image or video.

The latent layout (channels, spatial and temporal compression) comes from ComfyUI's own latent format for
the family, or from the connected model when there is one. Size can be typed, picked from a preset, or
derived from an aspect ratio and a megapixel budget, and is snapped to what the family needs. Video frame
counts are snapped to the family's grid (4n+1, 8n+1 ...).
"""
import math
import random

import torch

import comfy.latent_formats
import comfy.model_management as model_management
import nodes
from comfy_extras import nodes_minimax_h3

from . import families

# landscape ratios, lowest to highest; portrait is the same ratio turned around
RATIOS = {name: tuple(int(part) for part in name.split(":")) for name in (
    "1:1", "6:5", "5:4", "4:3", "7:5", "3:2", "16:10", "5:3", "16:9", "2:1", "21:9", "12:5", "5:2", "3:1", "32:9")}
PRESETS = (
    "1024x1024", "768x768", "512x512", "1152x896", "1216x832", "1344x768", "1536x640",
    "848x480", "1024x576", "1280x720", "1920x1080", "2560x1440", "768x512", "1344x768",
)
# family -> comfy.latent_formats class; minimax_h3 builds its audio+video latent through ComfyUI's own node
LATENT_FORMATS = {
    "sd15": "SD15", "sdxl": "SDXL", "sd3": "SD3", "flux": "Flux", "flux2": "Flux2", "chroma": "Flux",
    "qwen_image": "Wan21", "anima": "Wan21", "krea2": "Wan21", "moss_image": "Flux2", "zimage": "Flux",
    "lumina2": "Flux", "hidream": "Flux", "wan": "Wan21", "ltxv": "LTXV", "hunyuan_video": "HunyuanVideo",
    "cosmos": "Cosmos1CV8x8x8", "mochi": "Mochi",
}
# families whose size must be a multiple of something other than 16
SIZE_MULTIPLE = {"sd15": 64, "sdxl": 64, "ltxv": 32, "minimax_h3": 32}
# frames must be step * n + 1
FRAME_STEP = {"wan": 4, "hunyuan_video": 4, "ltxv": 8, "cosmos": 8, "mochi": 6}


def _snap(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_size(res_mode, width, height, preset, ratio, megapixels, orientation, multiple):
    if res_mode == "preset":
        width, height = (int(part) for part in preset.split("x"))
    elif res_mode == "aspect + megapixels":
        wide, tall = RATIOS[ratio]
        height = math.sqrt(megapixels * 1_000_000 * tall / wide)
        width = height * wide / tall
    if orientation == "landscape" and height > width or orientation == "portrait" and width > height:
        width, height = height, width
    return _snap(width, multiple), _snap(height, multiple)


def snap_frames(family_key, frames):
    """Nearest valid frame count at or above `frames` for the family."""
    if family_key == "minimax_h3":
        return nodes_minimax_h3.temporal_shape(frames)[0]
    step = FRAME_STEP.get(family_key, 1)
    return max(1, math.ceil((frames - 1) / step) * step + 1)


def frame_info(width, height, kind, count):
    """What the prompt writer needs to plan a composition that fits."""
    from math import gcd
    divisor = gcd(int(width), int(height))
    orientation = "portrait" if height > width else "landscape" if width > height else "square"
    ratio = f"{width // divisor}:{height // divisor}"
    return f"{width}x{height} px, {ratio} {orientation}, {kind}" + (f", {count} frames" if kind == "video" else "")


class PROMPTFORGE_EmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "family": (["auto"] + [family.label for family in families.FAMILIES if family.key in LATENT_FORMATS or family.key == "minimax_h3"], {"default": "auto", "tooltip": "auto uses the connected model or pipe."}),
                "kind": (["image", "video"], {"default": "image"}),
                "res_mode": (["aspect + megapixels", "preset", "custom"], {"default": "aspect + megapixels"}),
                "aspect_ratio": (list(RATIOS), {"default": "1:1", "tooltip": "Landscape ratios, lowest to highest; orientation turns them around."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 64.0, "step": 0.05}),
                "preset": (list(dict.fromkeys(PRESETS)), {"default": "1024x1024"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "orientation": (["landscape", "portrait"], {"default": "landscape", "tooltip": "Turns width and height around when needed, in every size mode."}),
                "randomize_ratio": ("BOOLEAN", {"default": False, "tooltip": "Picks the aspect ratio from ratio_pool with the seed. Uses the megapixels size mode."}),
                "ratio_pool": ("STRING", {"default": "", "tooltip": "Ratios to pick from, comma separated, for example 1:1, 3:2, 16:9. Empty = all of: " + ", ".join(RATIOS)}),
                "randomize_orientation": ("BOOLEAN", {"default": False, "tooltip": "Picks landscape or portrait with the seed."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1, "tooltip": "Drives the random ratio and orientation; change it between runs."}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 4096, "tooltip": "Video only. Snapped up to the family's frame grid. Ignored when seconds is above 0."}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip": "Video only. When above 0, frames = seconds * fps."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "When connected, its own latent format is used."}),
                "pipe": ("PROMPTFORGE_PIPE", {"tooltip": "From a PromptForge Loader: sets the family."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT", "FRAMES", "INFO", "FRAME_INFO")
    FUNCTION = "make"
    CATEGORY = "🔮 PromptForge/Sampling"
    DESCRIPTION = "Empty latent for any model family: custom size, preset or aspect ratio + megapixels, image or video."

    def make(self, family, kind, res_mode, aspect_ratio, megapixels, preset, width, height, orientation, randomize_ratio, ratio_pool, randomize_orientation, seed, frames, seconds, fps, batch_size, model=None, pipe=None):
        chosen = next((f for f in families.FAMILIES if f.label == family), None)
        if chosen is None and pipe and pipe.get("family"):
            chosen = families.BY_KEY.get(pipe["family"])
        key = chosen.key if chosen else ""
        latent_format = model.get_model_object("latent_format") if model is not None else None
        if latent_format is None and key in LATENT_FORMATS:
            latent_format = getattr(comfy.latent_formats, LATENT_FORMATS[key])()
        if key != "minimax_h3" and latent_format is None:
            raise ValueError("[PromptForge] Empty Latent needs a family, a PromptForge Loader pipe or a model to know the latent layout.")

        ratio = latent_format.spacial_downscale_ratio if latent_format is not None else 16
        multiple = SIZE_MULTIPLE.get(key, max(16, ratio * 2 if ratio == 8 else ratio))
        rng = random.Random(seed)
        if randomize_ratio:
            pool = [name.strip() for name in ratio_pool.split(",") if name.strip()] or list(RATIOS)
            unknown = [name for name in pool if name not in RATIOS]
            if unknown:
                raise ValueError(f"[PromptForge] unknown ratio {unknown[0]} in ratio_pool; use any of {', '.join(RATIOS)}.")
            aspect_ratio, res_mode = rng.choice(pool), "aspect + megapixels"
        if randomize_orientation:
            orientation = rng.choice(("landscape", "portrait"))
        width, height = resolve_size(res_mode, width, height, preset, aspect_ratio, megapixels, orientation, multiple)

        if kind == "image":
            count = 1
        else:
            count = snap_frames(key, round(seconds * fps) if seconds > 0 else frames)

        if key == "minimax_h3":
            latent = nodes.NODE_CLASS_MAPPINGS["EmptyMiniMaxH3LatentAV"].execute(width, height, count).args[0]
            if batch_size > 1:
                raise ValueError("[PromptForge] MiniMax H3 latents are made one at a time (batch_size 1).")
            return (latent, width, height, count, f"{width}x{height}, {count} frames (MiniMax H3 grid)", frame_info(width, height, kind, count))

        shape = [batch_size, latent_format.latent_channels, height // ratio, width // ratio]
        if kind == "video":
            shape.insert(2, (count - 1) // latent_format.temporal_downscale_ratio + 1)
        samples = torch.zeros(shape, device=model_management.intermediate_device(), dtype=model_management.intermediate_dtype())
        latent = {"samples": samples, "downscale_ratio_spacial": ratio}
        label = chosen.label if chosen else type(latent_format).__name__
        return (latent, width, height, count, f"{label}: {width}x{height}" + (f" ({aspect_ratio} {orientation})" if randomize_ratio or randomize_orientation else "") + (f", {count} frames" if kind == "video" else "") + f" -> latent {list(samples.shape)}",
                frame_info(width, height, kind, count))


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_EmptyLatent": PROMPTFORGE_EmptyLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_EmptyLatent": "PromptForge Empty Latent"}
