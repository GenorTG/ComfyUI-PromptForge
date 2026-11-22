"""Tiny decoders for live previews: the small TAESD-style decoders in models/vae_approx (taeh3 for MiniMax H3).

These files are flat decoders whose keys are the positions of their modules (`N.conv.0.weight` is a block, `N.weight`
a convolution, a missing number is a module without weights), so the network is rebuilt from the file itself with
ComfyUI's own TAESD blocks. Each latent frame is decoded on its own, which is what a preview needs.
"""
import logging

import torch
import torch.nn as nn

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy.taesd.taesd import Block, Clamp, conv

LATENT2RGB = "latent2rgb (instant, no file)"


def decoder_names():
    return [LATENT2RGB] + folder_paths.get_filename_list("vae_approx")


def default_decoder():
    return next((name for name in folder_paths.get_filename_list("vae_approx") if name.lower().startswith("taeh3")), LATENT2RGB)


def _build(state):
    groups = {}
    for key, value in state.items():
        head, _, rest = key.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat tiny decoder (key {key})")
        groups.setdefault(int(head), {})[rest] = value
    modules = []
    for index in range(max(groups) + 1):
        entry = groups.get(index)
        if entry is None:
            modules.append(Clamp() if index == 0 else nn.ReLU() if index == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            weight = entry["conv.0.weight"]
            modules.append(Block(weight.shape[1], weight.shape[0], use_midblock_gn="pool.0.weight" in entry))
        elif "weight" in entry:
            weight = entry["weight"]
            modules.append(conv(weight.shape[1], weight.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized module {index} ({sorted(entry)})")
    return nn.Sequential(*modules)


class TinyDecoder:
    def __init__(self, state, device, dtype):
        self.model = _build(state)
        self.model.load_state_dict(state)
        self.model = self.model.eval().to(device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.channels = self.model[1].weight.shape[1]

    def frames(self, video, indices):
        """The latent frames `indices` of a video latent [1, C, T, h, w] as RGB in 0..1, [n, h', w', 3] on the CPU."""
        out = []
        for index in indices:
            frame = self.model(video[0, :, index].unsqueeze(0).to(device=self.device, dtype=self.dtype))[0]
            out.append(frame.clamp(0, 1).float().cpu().movedim(0, -1))
        return torch.stack(out)


def load(name):
    """The tiny decoder `name` from models/vae_approx, or None (with a log line) when it is not a flat decoder or does not fit."""
    path = folder_paths.get_full_path("vae_approx", name) if name != LATENT2RGB else None
    if path is None:
        return None
    try:
        device = model_management.get_torch_device()
        return TinyDecoder(comfy.utils.load_torch_file(path, safe_load=True), device, model_management.vae_dtype(device, [torch.float16, torch.bfloat16]))
    except (ValueError, KeyError, RuntimeError) as error:
        logging.warning(f"[PromptForge] preview decoder {name} cannot be used ({error}); using latent2rgb.")
        return None
