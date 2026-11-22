"""Sampler settings recommended for a model.

Layers, later ones winning: generic defaults, the family, hints in the file name (turbo / distilled ...), and what
the model's own sidecar says. The sidecar (`<model>.metadata.json`, written by LoRA Manager / CivitAI helpers)
lists the sampler, scheduler, steps and CFG of the author's example images; the most common value of each is used.
Only those four fields are read from it, never the prompts.

Family values come from community workflow documentation (ComfyUI docs for LTXVScheduler, the MiniMax H3 Turbo
and Krea 2 Turbo node pages), not from the model makers, so treat them as starting points.
"""
import json
import re
from collections import Counter
from pathlib import Path

import comfy.samplers

GENERIC = {"sampler_name": "euler", "scheduler": "simple", "steps": 20, "cfg": 4.0}
# keyed by family; a name hint adds the distilled values on top
FAMILY = {
    "krea2": {"sampler_name": "res_multistep", "scheduler": "simple"},
    "minimax_h3": {"sampler_name": "euler", "scheduler": "simple"},
    "ltxv": {"scheduler": "ltxv"},
    "moss_image": {"sampler_name": "er_sde", "scheduler": "simple", "steps": 20, "cfg": 3.0},
}
DISTILLED_HINT = re.compile(r"turbo|distill|lightning|hyper|lcm|schnell|dmd|\b[2-9]step|\b[1-9]\d?step", re.I)
DISTILLED = {"steps": 8, "cfg": 1.0}
# names used by A1111-style metadata -> ComfyUI names
SAMPLER_ALIASES = {
    "euler a": "euler_ancestral", "euler ancestral": "euler_ancestral", "er sde": "er_sde", "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde", "dpm++ sde": "dpmpp_sde", "dpm++ 3m sde": "dpmpp_3m_sde", "dpm++ 2s a": "dpmpp_2s_ancestral",
    "unipc": "uni_pc", "heun": "heun", "lms": "lms", "ddim": "ddim", "lcm": "lcm",
}


def _sampler(name):
    key = str(name).strip().lower()
    key = SAMPLER_ALIASES.get(key, key.replace(" ", "_"))
    return key if key in comfy.samplers.KSampler.SAMPLERS else None


def _scheduler(name):
    key = str(name).strip().lower().replace(" ", "_")
    return key if key in comfy.samplers.KSampler.SCHEDULERS else None


def from_sidecar(model_path):
    """Most common sampler, scheduler, steps and cfg among the author's example images."""
    sidecar = Path(str(Path(model_path).with_suffix("")) + ".metadata.json")
    try:
        images = (json.loads(sidecar.read_text(encoding="utf-8")).get("civitai") or {}).get("images") or []
    except (OSError, ValueError, AttributeError):
        return {}
    votes = {"sampler_name": Counter(), "scheduler": Counter(), "steps": Counter(), "cfg": Counter()}
    for image in images:
        meta = image.get("meta") or {} if isinstance(image, dict) else {}
        for field, key, convert in (("sampler_name", "sampler", _sampler), ("scheduler", "scheduler", _scheduler),
                                    ("steps", "steps", lambda v: int(v) or None), ("cfg", "cfgScale", lambda v: float(v) or None)):
            if key in meta:
                try:
                    value = convert(meta[key])
                except (TypeError, ValueError):
                    continue
                if value is not None:
                    votes[field][value] += 1
    return {field: counter.most_common(1)[0][0] for field, counter in votes.items() if counter}


def recommend(family_key="", model_name="", model_path=None):
    """(settings, where each came from)."""
    settings, source = dict(GENERIC), {key: "default" for key in GENERIC}

    def apply(values, origin):
        for key, value in values.items():
            settings[key], source[key] = value, origin

    apply(FAMILY.get(family_key, {}), f"{family_key} family")
    if DISTILLED_HINT.search(model_name or ""):
        apply(DISTILLED, "name looks distilled")
    if model_path:
        apply(from_sidecar(model_path), "model sidecar")
    return settings, source
