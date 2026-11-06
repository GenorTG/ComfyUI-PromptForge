"""PromptForge LoRA loader: a searchable list of every LoRA you have, with per-LoRA strengths.

The list is drawn by web/promptforge_lora.js; this file holds the state format, the library index
(served at /promptforge/loras) and the node. The state is one JSON string:

    {"filter": "all", "items": [{"name": "sub/a.safetensors", "on": true, "model": 1.0, "clip": null,
                                 "random": false, "min": 0.6, "max": 1.0, "group": "", "part": "all"}]}

* `clip` null means "same as model".
* `random` draws the strength between `min` and `max`, seeded by the node seed.
* Items sharing a non-empty `group` are alternatives: one of the enabled ones is picked per run.
* `part` limits a LoRA to the video layers or the audio layers of an audio+video model (MiniMax H3, LTX-2).
* With two or more LoRAs on, the report says how much they overlap and how hard they push together; `balance`
  lowers all strengths by the same factor when together they push harder than the strongest one alone.
"""
import json
import os
import random
import re

import torch
from aiohttp import web
from safetensors import safe_open
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


PREVIEW_SUFFIXES = (".preview.jpeg", ".preview.jpg", ".preview.png", ".preview.webp", ".jpeg", ".jpg", ".png", ".webp")


def _preview_path(name):
    stem = os.path.splitext(folder_paths.get_full_path_or_raise("loras", name))[0]
    return next((stem + suffix for suffix in PREVIEW_SUFFIXES if os.path.isfile(stem + suffix)), None)


def _plain(html):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", re.sub(r"<(?:br|/p|/li)\s*/?>", "\n", html or ""))).strip()


TRAINING_FIELDS = (("ss_network_module", "Network"), ("ss_network_dim", "Rank"), ("ss_network_alpha", "Alpha"), ("ss_base_model_version", "Trained on"), ("modelspec.architecture", "Architecture"),
                   ("ss_resolution", "Resolution"), ("ss_num_epochs", "Epochs"), ("ss_max_train_steps", "Steps"), ("ss_learning_rate", "Learning rate"), ("ss_optimizer", "Optimizer"),
                   ("ss_num_train_images", "Training images"), ("modelspec.title", "Title"), ("ss_training_comment", "Comment"))


def _training_info(path):
    """What the file itself says about its training: [(label, value)] and the most common training tags."""
    try:
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
    except (OSError, ValueError):
        return [], []
    fields = [(label, str(metadata[key])[:200]) for key, label in TRAINING_FIELDS if metadata.get(key) not in (None, "", "None")]
    counts = {}
    try:
        for dataset in json.loads(metadata.get("ss_tag_frequency") or "{}").values():
            for tag, count in dataset.items():
                counts[tag.strip()] = counts.get(tag.strip(), 0) + int(count)
    except (ValueError, AttributeError, TypeError):
        pass
    return fields, [tag for tag, _ in sorted(counts.items(), key=lambda entry: -entry[1])[:24] if tag]


@PromptServer.instance.routes.get("/promptforge/lora_info")
async def _info_route(request):
    name = request.query.get("name", "")
    path = folder_paths.get_full_path("loras", name)
    if path is None:
        return web.json_response({"error": "not found"}, status=404)
    try:
        with open(os.path.splitext(path)[0] + ".metadata.json", encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, ValueError):
        sidecar = {}
    civitai = sidecar.get("civitai") or {}
    model_id = civitai.get("modelId")
    model = civitai.get("model") or {}
    training, training_tags = _training_info(path)
    images = []
    for image in civitai.get("images") or []:
        meta = image.get("meta") or {}
        if image.get("url"):
            images.append({"url": image["url"], "video": image.get("type") == "video", "seed": meta.get("seed"), "steps": meta.get("steps"), "cfg": meta.get("cfgScale"),
                           "sampler": meta.get("sampler"), "model": meta.get("Model"), "positive": meta.get("prompt"), "negative": meta.get("negativePrompt")})
    return web.json_response({
        "name": name,
        "title": sidecar.get("model_name") or os.path.basename(name),
        "version": civitai.get("name") or "",
        "type": model.get("type") or "",
        "base_model": sidecar.get("base_model", ""),
        "sha256": sidecar.get("sha256", ""),
        "size": sidecar.get("size") or os.path.getsize(path),
        "creator": (civitai.get("creator") or {}).get("username", ""),
        "downloads": (civitai.get("stats") or {}).get("downloadCount"),
        "likes": (civitai.get("stats") or {}).get("thumbsUpCount"),
        "words": _trigger_words(sidecar),
        "training": training,
        "training_tags": training_tags,
        "tips": sidecar.get("usage_tips") or "",
        "notes": sidecar.get("notes") or "",
        "tags": sidecar.get("tags") or [],
        "description": _plain(sidecar.get("modelDescription") or ""),
        "url": f"https://civitai.com/models/{model_id}" + (f"?modelVersionId={civitai['id']}" if civitai.get("id") else "") if model_id else "",
        "preview": _preview_path(name) is not None,
        "images": images,
    })


@PromptServer.instance.routes.get("/promptforge/lora_preview")
async def _preview_route(request):
    name = request.query.get("name", "")
    if folder_paths.get_full_path("loras", name) is None or (path := _preview_path(name)) is None:
        return web.Response(status=404)
    return web.FileResponse(path)


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


AUDIO_KEY = "audio"


def in_part(key, part):
    """Whether a patched weight belongs to the chosen part of an audio+video model."""
    return part == "all" or (AUDIO_KEY in str(key).lower()) == (part == "audio")


def _factors(adapter):
    """(up, down, scale) of a plain LoRA patch, or None for other kinds of patches."""
    if getattr(adapter, "name", "") != "lora":
        return None
    up, down, alpha, mid, dora, _ = adapter.weights
    if mid is not None or dora is not None:
        return None
    down = down.flatten(1).float()
    return up.float(), down, (alpha if alpha is not None else down.shape[0]) / down.shape[0]


def stack_stats(stack, limit=160):
    """Energy of every LoRA's change and the overlap of each pair, from the low-rank factors (no full weight matrices).

    stack: [(name, strength, patches)]. Returns (norms2, dots) where dots[(i, j)] is the inner product of the two changes
    over the layers both touch, and norms2[i] the squared size of change i over its own layers; each is measured on at
    most `limit` layers per LoRA.
    """
    factors = []
    for _, _, patches in stack:
        layers = {key: f for key, adapter in patches.items() if (f := _factors(adapter)) is not None}
        keys = sorted(layers, key=str)
        step = max(1, len(keys) // limit)
        factors.append({key: layers[key] for key in keys[::step]})
    norms2 = [0.0] * len(stack)
    dots = {}
    for i, mine in enumerate(factors):
        for key, (up, down, scale) in mine.items():
            norms2[i] += float(scale ** 2 * ((up.T @ up) * (down @ down.T)).sum())
        for j in range(i + 1, len(stack)):
            total, shared = 0.0, 0
            for key, (up, down, scale) in mine.items():
                other = factors[j].get(key)
                if other is None or other[0].shape[0] != up.shape[0] or other[1].shape[1] != down.shape[1]:
                    continue
                total += float(scale * other[2] * ((up.T @ other[0]) * (down @ other[1].T)).sum())
                shared += 1
            dots[(i, j)] = (total, shared)
    return norms2, dots


def balance_factor(strengths, norms2, dots, floor):
    """(factor, combined/strongest energy ratio): the uniform scale that brings the combined push down to that of the strongest LoRA alone."""
    single = max(s * s * n for s, n in zip(strengths, norms2))
    combined = sum(s * s * n for s, n in zip(strengths, norms2)) + 2 * sum(strengths[i] * strengths[j] * dot for (i, j), (dot, _) in dots.items())
    if single <= 0 or combined <= 0:
        return 1.0, 1.0
    return min(1.0, max(floor, (single / combined) ** 0.5)), combined / single


def overlap_words(cosine):
    return "opposing" if cosine < -0.15 else "aligned" if cosine > 0.4 else "partly aligned" if cosine > 0.15 else "independent"


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
            "optional": {
                "clip": ("CLIP",),
                "pipe": ("PROMPTFORGE_PIPE", {"tooltip": "From a PromptForge Loader: the list switches to that model's family."}),
                "balance": ("BOOLEAN", {"default": False, "tooltip": "When the LoRAs together push harder than the strongest one alone, lower all strengths by the same factor (the report shows the numbers). Image models can go down to 0.85 of the set strengths; video and audio models are left alone unless balance_floor is set."}),
                "balance_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Lowest factor balance may apply. 0 = by model kind: 0.85 for image models, 1.0 (no change) for video and audio."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "TRIGGER_WORDS", "REPORT")
    FUNCTION = "load"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Searchable LoRA list with per-LoRA strengths, random strengths and random groups."

    def load(self, model, loras, seed, skip_mismatched, clip=None, pipe=None, balance=False, balance_floor=0.0):
        try:
            items = json.loads(loras or "{}").get("items", [])
        except ValueError as exc:
            raise ValueError(f"[PromptForge] the LoRA list is not valid JSON: {exc}")
        target = _model_family(model) if skip_mismatched else None

        report, words = [], []
        key_map = None
        stack = []  # (name, model strength, clip strength, patches, metadata)
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
            part = item.get("part") or "all"
            if part != "all":
                patches = {key: patch for key, patch in patches.items() if in_part(key, part)}
                if not patches:
                    report.append(f"skipped {name}: it has no {part} layers")
                    continue
            stack.append((name, strength_model, strength_clip, patches, metadata, part))
            words.extend(word for word in entry["words"] if word not in words)

        factor = 1.0
        if stack:
            norms2, dots = stack_stats([(name, strength, patches) for name, strength, _, patches, _, _ in stack])
            if balance and len(stack) > 1:
                floor = balance_floor or (0.85 if target is None or target.kind == "image" else 1.0)
                factor, ratio = balance_factor([strength for _, strength, *_ in stack], norms2, dots, floor)
            for (name, strength_model, strength_clip, patches, _, part), norm2 in zip(stack, norms2):
                rank = next((a.weights[1].shape[0] for a in patches.values() if getattr(a, "name", "") == "lora"), 0)
                report.append(f"{name}: model {strength_model * factor:g}, clip {strength_clip * factor:g}" + (f" ({part} layers only)" if part != "all" else "")
                              + f", {len(patches)} layers" + (f", rank {rank}" if rank else "") + (f", strength-weighted size {strength_model * norm2 ** 0.5:.3g}" if norm2 else ""))
            for (i, j), (dot, shared) in dots.items():
                size = (norms2[i] * norms2[j]) ** 0.5
                if shared and size:
                    cosine = dot / size
                    report.append(f"overlap {stack[i][0]} / {stack[j][0]}: {shared} shared layers, similarity {cosine:+.2f} ({overlap_words(cosine)})")
            if len(stack) > 1:
                single = max(s * s * n for (_, s, *_), n in zip(stack, norms2))
                combined = sum(s * s * n for (_, s, *_), n in zip(stack, norms2)) + 2 * sum(stack[i][1] * stack[j][1] * dot for (i, j), (dot, _) in dots.items())
                if single > 0:
                    report.append(f"together they push {max(combined, 0) / single:.2f}x as hard as the strongest alone"
                                  + (f"; balance lowered every strength to {factor:.2f}x" if factor < 1 else "; turn on balance to even this out" if combined / single > 1.5 else ""))

        patched_model, patched_clip = model, clip
        for name, strength_model, strength_clip, patches, metadata, _ in stack:
            if strength_model:
                if patched_model is model:
                    patched_model = model.clone()
                patched_model.add_patches(patches, strength_model * factor)
                if metadata:
                    patched_model.set_attachments("lora_metadata", metadata)
            if clip is not None and strength_clip:
                if patched_clip is clip:
                    patched_clip = clip.clone()
                patched_clip.add_patches(patches, strength_clip * factor)
                if metadata:
                    patched_clip.patcher.set_attachments("lora_metadata", metadata)
        return (patched_model, patched_clip, ", ".join(words), "\n".join(report))


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_LoraLoader": PROMPTFORGE_LoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_LoraLoader": "PromptForge LoRA Loader"}
