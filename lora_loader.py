"""PromptForge LoRA loader: a searchable list of every LoRA you have, with per-LoRA strengths.

The list is drawn by web/promptforge_lora.js; this file holds the state format, the library index
(served at /promptforge/loras) and the node. The state is one JSON string:

    {"filter": "all", "items": [{"name": "sub/a.safetensors", "on": true, "model": 1.0, "clip": null,
                                 "random": false, "min": 0.6, "max": 1.0, "group": ""}]}

* `clip` null means "same as model".
* `random` draws the strength between `min` and `max`, seeded by the node seed.
* Items sharing a non-empty `group` are alternatives: one of the enabled ones is picked per run.
"""
import json
import os
import random

from aiohttp import web
from server import PromptServer

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

from . import families

_index = {}  # LoRA name -> ((mtime, size), entry)


def _trigger_words(sidecar):
    words = sidecar.get("trainedWords") or (sidecar.get("civitai") or {}).get("trainedWords") or []
    return [word.strip() for word in words if isinstance(word, str) and word.strip()]


def _describe(name):
    path = folder_paths.get_full_path("loras", name)
    if path is None:
        return None
    stat = os.stat(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _index.get(name)
    if cached and cached[0] == signature:
        return cached[1]
    family = families.family_of_lora(name, path)
    sidecar = {}
    sidecar_path = path.rsplit(".", 1)[0] + ".metadata.json"
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, ValueError):
        pass
    entry = {"name": name, "family": family.key if family else "", "words": _trigger_words(sidecar), "size": stat.st_size}
    _index[name] = (signature, entry)
    return entry


def library():
    """Every LoRA with its family and trigger words, plus the families present (for the filter)."""
    entries = [entry for entry in map(_describe, folder_paths.get_filename_list("loras")) if entry]
    present = sorted({entry["family"] for entry in entries if entry["family"]})
    return {
        "loras": entries,
        "families": [{"key": key, "label": families.BY_KEY[key].label} for key in present],
    }


@PromptServer.instance.routes.get("/promptforge/loras")
async def _library_route(request):
    return web.json_response(library())


def _model_family(model):
    config = getattr(getattr(model, "model", None), "model_config", None)
    return families.family_from_config(type(config).__name__) if config is not None else None


def choose(items, seed):
    """The (item, model strength, clip strength) list that applies for this seed."""
    rng = random.Random(seed)
    groups = {}
    for item in items:
        if item.get("on"):
            groups.setdefault((item.get("group") or "").strip(), []).append(item)
    chosen = []
    picked = {group: rng.choice(members) for group, members in groups.items() if group}
    for item in items:
        if not item.get("on"):
            continue
        group = (item.get("group") or "").strip()
        if group and picked[group] is not item:
            continue
        if item.get("random"):
            low, high = sorted((float(item.get("min", 0.0)), float(item.get("max", 1.0))))
            strength = round(rng.uniform(low, high), 2)
        else:
            strength = float(item.get("model", 1.0))
        clip = item.get("clip")
        chosen.append((item, strength, strength if clip is None or item.get("random") else float(clip)))
    return chosen


class PROMPTFORGE_LoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras": ("STRING", {"default": "{}", "tooltip": "Managed by the list below; edit the list, not this text."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1, "tooltip": "Seeds random strengths and random group picks."}),
                "skip_mismatched": ("BOOLEAN", {"default": True, "tooltip": "Skip LoRAs trained for a different model family than the connected model."}),
            },
            "optional": {"clip": ("CLIP",)},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "TRIGGER_WORDS", "REPORT")
    FUNCTION = "load"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Searchable LoRA list with per-LoRA strengths, random strengths and random groups."

    def load(self, model, loras, seed, skip_mismatched, clip=None):
        try:
            items = json.loads(loras or "{}").get("items", [])
        except ValueError as exc:
            raise ValueError(f"[PromptForge] the LoRA list is not valid JSON: {exc}")
        target = _model_family(model) if skip_mismatched else None

        report, words = [], []
        patched_model, patched_clip = model, clip
        key_map = None
        for item, strength_model, strength_clip in choose(items, seed):
            name = item["name"]
            entry = _describe(name)
            if entry is None:
                raise ValueError(f"[PromptForge] LoRA '{name}' was not found in the loras folders.")
            if target is not None and entry["family"] and entry["family"] != target.key:
                report.append(f"skipped {name}: made for {entry['family']}, model is {target.key}")
                continue
            if strength_model == 0 and strength_clip == 0:
                continue
            lora, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True, return_metadata=True)
            if key_map is None:  # the key maps depend only on the base model and clip, so build them once for all LoRAs
                key_map = comfy.lora.model_lora_keys_unet(model.model, {})
                if clip is not None:
                    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
            patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
            if strength_model:
                if patched_model is model:
                    patched_model = model.clone()
                patched_model.add_patches(patches, strength_model)
                if metadata:
                    patched_model.set_attachments("lora_metadata", metadata)
            if clip is not None and strength_clip:
                if patched_clip is clip:
                    patched_clip = clip.clone()
                patched_clip.add_patches(patches, strength_clip)
                if metadata:
                    patched_clip.patcher.set_attachments("lora_metadata", metadata)
            report.append(f"{name}: model {strength_model:g}, clip {strength_clip:g}")
            words.extend(word for word in entry["words"] if word not in words)
        return (patched_model, patched_clip, ", ".join(words), "\n".join(report))


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_LoraLoader": PROMPTFORGE_LoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_LoraLoader": "PromptForge LoRA Loader"}
