"""PromptForge Loader: one node for checkpoint / diffusion model / GGUF + text encoders + VAEs.

It calls ComfyUI's own loaders (GGUF files go through gguf_native, the GGUF loading code of ComfyUI-GGUF), so new
model types and loader fixes in ComfyUI apply here automatically. Its own jobs are choosing the text encoder type
from the model family, picking baked-in components, and choosing weight settings for this machine.

File choices carry a source tag: "[ckpt] x", "[unet] x", "[gguf] x" for models and "[te] x", "[gguf] x" for
text encoders. web/promptforge_loader.js narrows the lists by family; the node accepts any tagged name.
"""
import os
import re

import torch
from aiohttp import web
from server import PromptServer

import comfy.sd
import comfy.utils
import folder_paths
import nodes

from . import autotune, families, gguf_native

AUTO = "auto"
BAKED = "(baked-in)"
NONE = "(none)"
CLIP_SLOTS = 4
TAG = re.compile(r"^\[(ckpt|unet|gguf|te)\] (.+)$")
# (tag, folder key) per kind of file; the gguf folders are made by gguf_native
MODEL_SOURCES = (("ckpt", "checkpoints"), ("unet", "diffusion_models"), ("gguf", "unet_gguf"))
CLIP_SOURCES = (("te", "text_encoders"), ("gguf", "clip_gguf"))
WEIGHT_DTYPES = (AUTO, "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "bf16", "fp16", "fp32")
# weight_dtype -> comfy model_options
DTYPE_OPTIONS = {
    "bf16": {"dtype": torch.bfloat16},
    "fp16": {"dtype": torch.float16},
    "fp32": {"dtype": torch.float32},
    "fp8_e4m3fn": {"dtype": torch.float8_e4m3fn},
    "fp8_e4m3fn_fast": {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True},
    "fp8_e5m2": {"dtype": torch.float8_e5m2},
}


def _tagged(sources):
    names = []
    for tag, folder in sources:
        if folder in folder_paths.folder_names_and_paths:
            names.extend(f"[{tag}] {name}" for name in folder_paths.get_filename_list(folder))
    return names


def _resolve(choice, sources):
    """(tag, folder, name, full path) for a tagged choice."""
    match = TAG.match(choice)
    folders = dict(sources)
    if not match or match.group(1) not in folders:
        raise ValueError(f"[PromptForge] '{choice}' is not a file choice of this loader.")
    tag, name = match.groups()
    folder = folders[tag]
    return tag, folder, name, folder_paths.get_full_path_or_raise(folder, name)


def _vae_names():
    return nodes.VAELoader.vae_list(nodes.VAELoader)


_model_families = {}  # full path -> ((mtime, size), family key)


def _model_family_key(name, folder):
    path = folder_paths.get_full_path(folder, name)
    stat = os.stat(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _model_families.get(path)
    if cached is None or cached[0] != signature:
        family = families.family_of_model(name, folder, path)
        cached = _model_families[path] = (signature, family.key if family else "")
    return cached[1]


def catalog():
    """Everything the UI needs: files with their family, and what each family expects."""
    models = []
    for tag, folder in MODEL_SOURCES:
        if folder not in folder_paths.folder_names_and_paths:
            continue
        for name in folder_paths.get_filename_list(folder):
            models.append({"value": f"[{tag}] {name}", "family": _model_family_key(name, folder)})
    return {
        "models": models,
        "families": [{"key": f.key, "label": f.label, "clips": list(f.clips), "vaes": list(f.vaes), "kind": f.kind} for f in families.FAMILIES],
    }


@PromptServer.instance.routes.get("/promptforge/catalog")
async def _catalog_route(request):
    return web.json_response(catalog())


class PROMPTFORGE_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _tagged(MODEL_SOURCES) or ["(no models found)"]
        clips = [BAKED, NONE] + _tagged(CLIP_SOURCES)
        vaes = [BAKED, NONE] + _vae_names()
        return {
            "required": {
                "model": (models, {"tooltip": "[ckpt] checkpoint, [unet] diffusion model, [gguf] GGUF model."}),
                "family": ([AUTO] + [family.label for family in families.FAMILIES], {"default": AUTO, "tooltip": "auto detects it from the file. Sets the text encoder type and the model list filter."}),
                "weight_dtype": (list(WEIGHT_DTYPES), {"default": AUTO, "tooltip": "auto picks for this machine using optimize_for."}),
                "optimize_for": (["balanced", "quality", "speed"], {"default": "balanced", "tooltip": "Used by weight_dtype auto. quality keeps stored precision; balanced uses fp8 storage only when the model does not fit in VRAM; speed also uses fp8 compute when a quick benchmark shows it is faster on this GPU."}),
                "clip_1": (clips, {"default": BAKED, "tooltip": "Text encoder. (baked-in) uses the one inside the checkpoint."}),
                "clip_2": (clips, {"default": NONE}),
                "clip_3": (clips, {"default": NONE}),
                "clip_4": (clips, {"default": NONE}),
                "clip_type": ([AUTO] + [member.name.lower() for member in comfy.sd.CLIPType], {"default": AUTO, "tooltip": "auto uses the family's type."}),
                "clip_device": (["default", "cpu"], {"default": "default"}),
                "vae": (vaes, {"default": BAKED, "tooltip": "Main VAE (image or video)."}),
                "audio_vae": (vaes, {"default": NONE}),
                "extra_vae": (vaes, {"default": NONE, "tooltip": "Optional second VAE, for example an upscaling one to switch to later."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "VAE", "STRING", "PROMPTFORGE_PIPE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "AUDIO_VAE", "EXTRA_VAE", "INFO", "PIPE")
    FUNCTION = "load"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Loads a model, its text encoders and VAEs in one node, choosing the text encoder type and weight settings automatically."

    def load(self, model, family, weight_dtype, optimize_for, clip_1, clip_2, clip_3, clip_4, clip_type, clip_device, vae, audio_vae, extra_vae):
        tag, folder, name, path = _resolve(model, MODEL_SOURCES)
        chosen = next((f for f in families.FAMILIES if f.label == family), None)
        if chosen is None:
            chosen = families.family_of_model(name, folder, path)
        info = [f"model: {name} ({chosen.label if chosen else 'unknown family'})"]

        model_options = {}
        if tag == "gguf":
            if weight_dtype not in (AUTO, "default"):
                info.append("weight_dtype is ignored for GGUF models")
        else:
            dtype, reason = self._weight_dtype(weight_dtype, optimize_for, path)
            model_options.update(DTYPE_OPTIONS.get(dtype, {}))
            info.append(f"weights: {dtype} ({reason})")

        clip_names = [choice for choice in (clip_1, clip_2, clip_3, clip_4) if choice not in (BAKED, NONE)]
        want_baked_clip = clip_1 == BAKED and not clip_names
        want_baked_vae = vae == BAKED

        loaded_clip = loaded_vae = None
        if tag == "ckpt":
            unet, loaded_clip, loaded_vae, _ = comfy.sd.load_checkpoint_guess_config(
                path, output_vae=want_baked_vae, output_clip=want_baked_clip,
                embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options)
        elif tag == "unet":
            unet = comfy.sd.load_diffusion_model(path, model_options=model_options)
        else:
            unet = self._load_gguf(name)

        if clip_names:
            loaded_clip = self._load_clip(clip_names, clip_type, chosen, clip_device)
            info.append("clip: " + " + ".join(TAG.match(choice).group(2) for choice in clip_names))
        elif loaded_clip is not None:
            info.append("clip: baked-in")
        elif chosen is not None and chosen.clips and clip_1 != NONE:
            info.append(f"clip: none loaded; {chosen.label} needs {' + '.join(chosen.clips)}")

        vaes = []
        for slot, choice in (("vae", vae), ("audio_vae", audio_vae), ("extra_vae", extra_vae)):
            if slot == "vae" and choice == BAKED:
                vaes.append(loaded_vae)
            elif choice in (BAKED, NONE):
                vaes.append(None)
            else:
                vaes.append(nodes.VAELoader().load_vae(choice)[0])
        info.extend(f"{label}: {'loaded' if loaded is not None else 'none'}" for label, loaded in zip(("vae", "audio_vae", "extra_vae"), vaes))
        pipe = {"family": chosen.key if chosen else "", "model": name, "folder": folder}
        return (unet, loaded_clip, *vaes, "\n".join(info), pipe)

    @staticmethod
    def _weight_dtype(weight_dtype, optimize_for, path):
        if weight_dtype != AUTO:
            return weight_dtype, "chosen"
        header, _ = families.read_header(path) if path.endswith(".safetensors") else ({}, {})
        stored = {dtype for _, dtype in header.values()}
        return autotune.choose_weight_dtype(stored, os.path.getsize(path), optimize_for)

    def _load_gguf(self, name):
        return gguf_native.load_unet(folder_paths.get_full_path_or_raise("unet_gguf", name))

    def _load_clip(self, choices, clip_type, chosen, device):
        resolved = [_resolve(choice, CLIP_SOURCES) for choice in choices]
        paths = [entry[3] for entry in resolved]
        if clip_type == AUTO:
            type_ = families.clip_type(chosen)
        else:
            type_ = getattr(comfy.sd.CLIPType, clip_type.upper())
        if any(entry[0] == "gguf" for entry in resolved):
            return gguf_native.load_clip(paths, type_)
        options = {}
        if device == "cpu":
            options["load_device"] = options["offload_device"] = torch.device("cpu")
        return comfy.sd.load_clip(ckpt_paths=paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=type_, model_options=options)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_Loader": PROMPTFORGE_Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_Loader": "PromptForge Loader"}
