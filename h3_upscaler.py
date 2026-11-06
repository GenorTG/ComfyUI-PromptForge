"""MiniMax H3 3D latent upscaler: a learned upscaler for H3 video latents (models in models/latent_upscale_models).

The network, model detection and temporal chunking are from "Minimax H3 Latent Upscaler (3D)" by LBH-123-AI (MIT, see
LICENSE-MiniMaxH3-Latent-Upscaler), included here so the Endless Video node needs no other pack. The attention block
and einops are left out: the shipped models do not use the attention block (the original forces it off too).
"""
import gc
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as model_management
import folder_paths

FOLDER = "latent_upscale_models"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

DOWNSAMPLE = 16
MODES = ("scale by multiplier", "megapixels", "target dimensions")
PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]
CHUNK = 32


def model_names():
    names = [name for name in folder_paths.get_filename_list(FOLDER) if os.path.splitext(name)[1].lower() in (".pth", ".safetensors")]
    h3 = [name for name in names if "minimax" in name.lower() or "h3" in name.lower()]
    return h3 or names or ["(put an H3 latent upscaler in models/latent_upscale_models)"]


def default_model():
    names = model_names()
    return next((name for name in names if "minimax" in name.lower() and "fp16" in name.lower()), next((name for name in names if "minimax" in name.lower()), names[0]))


def _norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, channels, embed, dropout=0):
        super().__init__()
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed, 2 * channels))
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout), nn.Conv3d(channels, channels, 3, padding=1))
        self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = torch.chunk(self.emb_layers(emb).type(h.dtype)[..., None, None, None], 2, dim=1)
        return self.skip(x) + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel=5):
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        def blocks(count):
            layers = nn.ModuleList()
            for index in range(count):
                layers.append(ResBlock(channels, 64, 0.1))
                if temporal_every > 0 and index % temporal_every == 0:
                    layers.append(TemporalConv(channels, temporal_kernel))
            return layers

        self.in_blocks, self.out_blocks = blocks(in_blocks), blocks(out_blocks)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale, size):
        """x [B, C, T, H, W] normalized latents -> [B, C, T, size[1], size[2]]; long clips run in overlapping temporal chunks."""
        frames = x.shape[2]
        if frames <= CHUNK:
            return self._segment(x, scale, size)
        overlap = next((block.dwconv.weight.shape[2] for block in self.in_blocks if isinstance(block, TemporalConv)), 0)
        padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        out = torch.zeros(x.shape[0], x.shape[1], frames, size[1], size[2], device=x.device, dtype=x.dtype)
        total = torch.zeros(1, 1, frames, 1, 1, device=x.device, dtype=x.dtype)
        for start in range(0, frames, CHUNK):
            end = min(frames, start + CHUNK)
            first, last = max(0, start - overlap), min(frames, end + overlap)
            lo, hi = max(0, first - overlap), min(frames + 2 * overlap, last + overlap)
            result = self._segment(padded[:, :, lo:hi].contiguous(), scale, (hi - lo, size[1], size[2]))
            offset = first + overlap - lo
            valid = result[:, :, offset:offset + last - first]
            weight = torch.ones(last - first, device=x.device, dtype=x.dtype)
            if start > first:
                weight[:start - first] = torch.arange(1, start - first + 1, device=x.device, dtype=x.dtype) / (start - first + 1)
            if last > end:
                weight[-(last - end):] = torch.arange(last - end, 0, -1, device=x.device, dtype=x.dtype) / (last - end + 1)
            out[:, :, first:last] += valid * weight.view(1, 1, -1, 1, 1)
            total[:, :, first:last] += weight.view(1, 1, -1, 1, 1)
        return out / total.clamp(min=1e-8)

    def _segment(self, x, scale, size):
        emb = self.embed(torch.tensor([[scale - 1]], dtype=x.dtype, device=x.device))
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlock) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlock) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def _load(name, device, dtype):
    path = folder_paths.get_full_path_or_raise(FOLDER, name)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(key.startswith("upscaler.") for key in state):
        state = {key[len("upscaler."):]: value for key, value in state.items() if key.startswith("upscaler.")}
    state = {key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value for key, value in state.items()}

    ids = lambda prefix: {int(m.group(1)) for key in state if (m := re.match(rf"{prefix}\.(\d+)\.in_layers\.", key))}
    kernel = next((value.shape[2] for key, value in state.items() if key.endswith("dwconv.weight")), 0)
    model = LatentResizer(in_channels=state["conv_in.weight"].shape[1], channels=state["conv_in.weight"].shape[0], in_blocks=len(ids("in_blocks")) or 12,
                          out_blocks=len(ids("out_blocks")) or 12, temporal_every=2 if kernel else 0, temporal_kernel=kernel or 5)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


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


def upscale(video, name, mode, scale=2.0, megapixels=1.0, target_width=1280, target_height=704, align=32, precision="fp16"):
    """Upscales an H3 video latent [B, 24, T, H, W]; returns it on the CPU in its own dtype. Raises when the target is smaller than the source."""
    height, width = video.shape[-2:]
    out_h, out_w = target_shape(height, width, mode, scale, megapixels, target_width, target_height, align)
    if out_h < height or out_w < width:
        raise ValueError("[PromptForge] the latent upscaler only makes latents larger; raise the upscale size.")
    if (out_h, out_w) == (height, width):
        return video
    device, dtype = model_management.get_torch_device(), PRECISIONS[precision]
    factor = (out_w / width + out_h / height) / 2
    model = _load(name, device, dtype)
    try:
        mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        source = (video.to(device=device, dtype=dtype) - mean) / std
        result = model(source, factor, (video.shape[2], out_h, out_w)) * std + mean
        return result.to(device="cpu", dtype=video.dtype)
    finally:
        del model
        gc.collect()
        model_management.soft_empty_cache()
