"""Model families: one table that the loader and the LoRA loader share.

A family answers "what is this model": its name, how ComfyUI should load its text encoders, and which
VAEs it uses. Files are matched to a family from several signals, best first:

* ComfyUI's own detection on the tensor shapes in the safetensors header (no weights are read),
* the folder the file is in and the words in its name,
* for LoRAs, the LoRA Manager sidecar (`<name>.metadata.json`) and the training metadata in the header.
"""
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.model_detection
import comfy.sd
import folder_paths


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    kind: str  # image, video or audio
    aliases: tuple  # names as they appear in folders, file names and metadata
    configs: tuple  # prefixes of comfy.supported_models class names
    clip_type: str  # comfy.sd.CLIPType member name, "" when the checkpoint always carries its encoder
    clips: tuple  # roles of the text encoder files, in the order ComfyUI expects them
    vaes: tuple = ("image",)  # kinds of VAE the family can use


def _family(key, label, kind, aliases, configs, clip_type, clips, vaes=("image",)):
    return Family(key, label, kind, tuple(aliases), tuple(configs), clip_type, tuple(clips), tuple(vaes))


FAMILIES = (
    _family("sd15", "SD 1.5", "image", ("sd15", "sd1", "sd1.5", "stablediffusionv1", "stablediffusion15"), ("SD15", "SD20", "SD21"), "STABLE_DIFFUSION", ("clip_l",)),
    _family("sdxl", "SDXL / Illustrious / Pony", "image", ("sdxl", "illustrious", "illustriousxl", "pony", "ponyxl", "noobai", "sdxl10"), ("SDXL", "SSD1B", "Segmind_Vega", "KOALA"), "STABLE_DIFFUSION", ("clip_l", "clip_g")),
    _family("sd3", "SD 3 / 3.5", "image", ("sd3", "sd35", "stablediffusion3"), ("SD3",), "SD3", ("clip_l", "clip_g", "t5xxl")),
    _family("flux", "Flux.1", "image", ("flux", "flux1", "fluxd", "fluxs", "fluxdev", "fluxschnell", "flux1d", "flux1s"), ("Flux",), "FLUX", ("clip_l", "t5xxl")),
    _family("flux2", "Flux.2 / Klein", "image", ("flux2", "flux2klein", "fluxklein", "klein"), ("Flux2",), "FLUX2", ("text_encoder",)),
    _family("flux3", "Flux.3", "image", ("flux3",), ("Flux3",), "FLUX3", ("text_encoder",)),
    _family("chroma", "Chroma", "image", ("chroma", "chromaradiance"), ("Chroma",), "CHROMA", ("t5xxl",)),
    _family("qwen_image", "Qwen Image", "image", ("qwenimage", "qwenimageedit", "qwen"), ("QwenImage",), "QWEN_IMAGE", ("text_encoder",)),
    _family("anima", "Anima", "image", ("anima", "animapreview"), ("Anima",), "STABLE_DIFFUSION", ("text_encoder",)),
    _family("krea2", "Krea 2", "image", ("krea2", "krea2turbo"), ("Krea2",), "KREA2", ("text_encoder",)),
    _family("moss_image", "MOSS Image", "image", ("mossimage", "moss"), ("MOSSImage",), "MOSS_IMAGE", ("text_encoder",)),
    _family("zimage", "Z-Image", "image", ("zimage", "zimageturbo"), ("ZImage",), "LUMINA2", ("text_encoder",)),
    _family("lumina2", "Lumina 2", "image", ("lumina2", "lumina"), ("Lumina2",), "LUMINA2", ("text_encoder",)),
    _family("hidream", "HiDream", "image", ("hidream",), ("HiDream",), "HIDREAM", ("clip_l", "clip_g", "t5xxl", "llama")),
    _family("hunyuan_image", "Hunyuan Image", "image", ("hunyuanimage",), ("HunyuanImage",), "HUNYUAN_IMAGE", ("qwen_vl", "byt5")),
    _family("omnigen2", "OmniGen 2", "image", ("omnigen2", "omnigen"), ("Omnigen2",), "OMNIGEN2", ("text_encoder",)),
    _family("pixart", "PixArt", "image", ("pixart", "pixartalpha", "pixartsigma"), ("PixArt",), "PIXART", ("t5xxl",)),
    _family("auraflow", "AuraFlow", "image", ("auraflow",), ("AuraFlow",), "STABLE_DIFFUSION", ("text_encoder",)),
    _family("minimax_h3", "MiniMax H3", "video", ("minimaxh3", "minimax", "minimaxh3ref2va", "h3"), ("MiniMaxH3",), "MINIMAX", ("text_encoder",), ("video", "audio", "image")),
    _family("wan", "Wan 2.x", "video", ("wan", "wan21", "wan22", "wanvideo"), ("WAN",), "WAN", ("umt5",), ("video",)),
    _family("ltxv", "LTX Video", "video", ("ltxv", "ltx", "ltxav", "ltxv23", "ltx2"), ("LTXV", "LTXAV"), "LTXV", ("text_encoder",), ("video", "audio")),
    _family("hunyuan_video", "Hunyuan Video", "video", ("hunyuanvideo", "hunyuanvideo15"), ("HunyuanVideo",), "HUNYUAN_VIDEO", ("clip_l", "llama"), ("video",)),
    _family("cosmos", "Cosmos", "video", ("cosmos",), ("Cosmos",), "COSMOS", ("t5xxl",), ("video",)),
    _family("mochi", "Mochi", "video", ("mochi", "genmo"), ("GenmoMochi",), "MOCHI", ("t5xxl",), ("video",)),
    _family("cogvideox", "CogVideoX", "video", ("cogvideox",), ("CogVideoX",), "COGVIDEOX", ("t5xxl",), ("video",)),
    _family("kandinsky5", "Kandinsky 5", "video", ("kandinsky5", "kandinsky"), ("Kandinsky5",), "KANDINSKY5", ("qwen_vl", "clip_l"), ("video",)),
    _family("ace", "ACE-Step (audio)", "audio", ("acestep", "aceaudio", "ace"), ("ACEStep",), "ACE", ("text_encoder",), ("audio",)),
    _family("stable_audio", "Stable Audio", "audio", ("stableaudio",), ("StableAudio",), "STABLE_AUDIO", ("t5",), ("audio",)),
)
BY_KEY = {family.key: family for family in FAMILIES}

# aliases that are too generic to trust on their own in a file name; they still count in a folder name or metadata
_WEAK = {"qwen", "ace", "h3", "klein", "moss", "flux", "wan", "ltx", "sd1"}


def _tokens(text):
    """Words of a name with CamelCase and digit boundaries split: 'ZImageTurbo_v2' -> z, image, turbo, v, 2."""
    return [token.lower() for token in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|\d+|[A-Z]+", text)]


def family_from_text(text, trust_weak=True):
    """The family a folder, file name or metadata string names, or None. The longest matching alias wins."""
    words = _tokens(text or "")
    best, best_len = None, 0
    for start in range(len(words)):
        joined = ""
        for end in range(start, min(start + 4, len(words))):
            joined += words[end]
            for family in FAMILIES:
                if joined in family.aliases and len(joined) > best_len and (trust_weak or joined not in _WEAK):
                    best, best_len = family, len(joined)
    return best


def family_from_config(name):
    """Family for a comfy.supported_models class name such as 'Krea2' or 'WAN22_T2V'."""
    best, best_len = None, 0
    for family in FAMILIES:
        for prefix in family.configs:
            if name.startswith(prefix) and len(prefix) > best_len:
                best, best_len = family, len(prefix)
    return best


def _header(path):
    """(tensor name -> (shape, dtype), metadata) from a safetensors header, without reading weights."""
    with safe_open(path, "pt") as handle:
        slices = {key: handle.get_slice(key) for key in handle.keys()}
        shapes = {key: (tuple(piece.get_shape()), piece.get_dtype()) for key, piece in slices.items()}
        return shapes, handle.metadata() or {}


_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64, "I8": torch.int8, "U8": torch.uint8,
           "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def family_from_weights(path):
    """Ask ComfyUI's own model detection about a diffusion model or checkpoint file; None when it does not know it."""
    if not str(path).endswith(".safetensors"):
        return None
    shapes, metadata = _header(path)
    state = {key: torch.empty(shape, dtype=_DTYPES.get(dtype, torch.float16), device="meta") for key, (shape, dtype) in shapes.items()}
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
    config = comfy.model_detection.model_config_from_unet(state, prefix, metadata=metadata)
    return family_from_config(type(config).__name__) if config is not None else None


def family_of_model(relative_path, folder, full_path):
    """Family of a checkpoint / diffusion model: the weights first, then the folder and file name."""
    try:
        family = family_from_weights(full_path)
    except (OSError, ValueError, KeyError, RuntimeError):
        family = None
    if family is None:
        parts = Path(relative_path).parts
        for text in (*reversed(parts[:-1]), parts[-1]):
            family = family_from_text(text, trust_weak=text is not parts[-1])
            if family is not None:
                break
    return family


# LoRA metadata keys that name the base model
_LORA_META_KEYS = ("ss_base_model_version", "modelspec.architecture", "base_model", "ss_sd_model_name")


def family_of_lora(relative_path, full_path):
    """Family of a LoRA file, or None. Sidecar and training metadata beat key shapes, which beat the name."""
    sidecar = Path(os.path.splitext(full_path)[0] + ".metadata.json")
    if sidecar.is_file():
        try:
            family = family_from_text(str(json.loads(sidecar.read_text(encoding="utf-8")).get("base_model") or ""))
        except (OSError, ValueError):
            family = None
        if family is not None:
            return family
    keys = ()
    if full_path.endswith(".safetensors"):
        try:
            with open(full_path, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(min(size, 64 * 1024 ** 2)))
            metadata = header.pop("__metadata__", {}) or {}
            keys = tuple(header)
            for name in _LORA_META_KEYS:
                family = family_from_text(str(metadata.get(name, "")))
                if family is not None:
                    return family
        except (OSError, ValueError, struct.error):
            pass
    if any(key.startswith("lora_te2_") for key in keys):
        return BY_KEY["sdxl"]
    parts = Path(relative_path).parts
    for text in (*reversed(parts[:-1]), parts[-1]):
        family = family_from_text(text, trust_weak=text is not parts[-1])
        if family is not None:
            return family
    return None


def clip_type(family):
    """comfy.sd.CLIPType for a family (STABLE_DIFFUSION when ComfyUI does not know the name)."""
    return getattr(comfy.sd.CLIPType, family.clip_type, comfy.sd.CLIPType.STABLE_DIFFUSION) if family and family.clip_type else comfy.sd.CLIPType.STABLE_DIFFUSION
