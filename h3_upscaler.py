"""Upscaling of MiniMax H3 video latents with ComfyUI's own MiniMax H3 latent upscaler (models in models/latent_upscale_models).

ComfyUI loads and offloads the model and runs it in slabs of frames, so it needs far less memory than a plain implementation."""
import os

import folder_paths
from comfy_extras import nodes_hunyuan, nodes_minimax_h3

FOLDER = "latent_upscale_models"
DOWNSAMPLE = 16
MODES = ("scale by multiplier", "megapixels", "target dimensions")


def model_names():
    names = [name for name in folder_paths.get_filename_list(FOLDER) if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")]
    h3 = [name for name in names if "minimax" in name.lower() or "h3" in name.lower()]
    return h3 or names or ["(put an H3 latent upscaler in models/latent_upscale_models)"]


def default_model():
    names = model_names()
    return next((name for name in names if "minimax" in name.lower() and "fp16" in name.lower()), next((name for name in names if "minimax" in name.lower()), names[0]))


def target_shape(height, width, mode, scale, megapixels, target_width, target_height, align):
    """(h, w) of the upscaled latent for a latent of h x w, with the pixel size rounded to `align` and to the VAE grid."""
    if mode == "scale by multiplier":
        pixel_w, pixel_h = width * DOWNSAMPLE * scale, height * DOWNSAMPLE * scale
    elif mode == "megapixels":
        ratio = width / height
        pixel_h = (megapixels * 1024 * 1024 / ratio) ** 0.5
        pixel_w = pixel_h * ratio
    else:
        pixel_w, pixel_h = float(target_width), float(target_height)
    align = max(1, align)
    snap = lambda value: max(1, round(round(value / align) * align / DOWNSAMPLE))
    return snap(pixel_h), snap(pixel_w)


def upscale(video, name, mode, scale=2.0, megapixels=1.0, target_width=1280, target_height=704, align=32):
    """Upscales an H3 video latent [B, 24, T, H, W]. Raises when the target is smaller than the source."""
    height, width = video.shape[-2:]
    out_h, out_w = target_shape(height, width, mode, scale, megapixels, target_width, target_height, align)
    if out_h < height or out_w < width:
        raise ValueError("[PromptForge] the latent upscaler only makes latents larger; raise the upscale size.")
    model = nodes_hunyuan.LatentUpscaleModelLoader.execute(name).args[0]
    return nodes_minimax_h3.MiniMaxH3LatentUpscaleWithModel.execute(model, {"samples": video}, out_w * DOWNSAMPLE, out_h * DOWNSAMPLE).args[0]["samples"]
