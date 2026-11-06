"""PromptForge Sampler: KSampler / KSampler (Advanced) / custom sigmas in one node.

It runs ComfyUI's own `comfy.sample.sample`, so every sampler, scheduler, model patch and preview works as in
the stock nodes. On top of that:

* `scheduler = auto` picks the family's schedule (LTX Video's token-count dependent one, otherwise simple),
* sigmas can come from a SIGMAS input, a typed list, or the LTXV schedule with its parameters,
* `runs` samples several seeds in one go and returns them as one batch,
* settings can be saved as named presets, and recommended settings come from the model's sidecar and family
  (see sampler_defaults.py).
"""
import json
from pathlib import Path

import torch
from aiohttp import web
from server import PromptServer

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
from comfy_extras import nodes_easycache, nodes_lt

from . import families, sampler_defaults

AUTO = "auto"
LTXV = "ltxv"
PRESET_FIELDS = ("steps", "cfg", "sampler_name", "scheduler", "denoise", "runs", "step_cache", "cache_threshold")
# step caching skips whole model calls, which only pays off (and only stays clean) with enough steps
AUTO_CACHE_MIN_STEPS = 15
CACHES = {"easycache": nodes_easycache.EasyCacheNode, "lazycache": nodes_easycache.LazyCacheNode}


def _preset_file():
    return Path(folder_paths.get_user_directory()) / "promptforge" / "sampler_presets.json"


def load_presets():
    try:
        return json.loads(_preset_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@PromptServer.instance.routes.get("/promptforge/sampler_presets")
async def _presets_route(request):
    return web.json_response(load_presets())


@PromptServer.instance.routes.post("/promptforge/sampler_presets")
async def _save_preset_route(request):
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "a preset needs a name"}, status=400)
    presets = load_presets()
    if body.get("delete"):
        presets.pop(name, None)
    else:
        presets[name] = {field: body["settings"][field] for field in PRESET_FIELDS if field in body.get("settings", {})}
    _preset_file().parent.mkdir(parents=True, exist_ok=True)
    _preset_file().write_text(json.dumps(presets, indent=1), encoding="utf-8")
    return web.json_response(presets)


@PromptServer.instance.routes.get("/promptforge/recommend")
async def _recommend_route(request):
    family, name, folder = (request.query.get(key, "") for key in ("family", "model", "folder"))
    path = folder_paths.get_full_path(folder, name) if folder and name else None
    settings, source = sampler_defaults.recommend(family, name, path)
    return web.json_response({"settings": settings, "source": source})


def parse_sigmas(text):
    """'1, 0.8, 0.5' -> descending sigma tensor ending in 0."""
    values = [float(part) for part in text.replace("\n", ",").split(",") if part.strip()]
    if len(values) < 2 or any(a < b for a, b in zip(values, values[1:])):
        raise ValueError("[PromptForge] custom_sigmas must be a descending list of at least two numbers, for example 1, 0.8, 0.5, 0.")
    if values[-1] > 0:
        values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def trim_for_denoise(sigmas, steps):
    """Keep the last `steps` intervals, which is what denoise < 1 means for a schedule computed at steps / denoise."""
    return sigmas[-(steps + 1):]


class PROMPTFORGE_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 64 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": ([AUTO, LTXV] + comfy.samplers.KSampler.SCHEDULERS, {"default": AUTO, "tooltip": "auto: the family's schedule (ltxv for LTX Video, otherwise simple). ltxv: LTX Video's token-count dependent schedule."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "runs": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Samples this many seeds (seed, seed+1 ...) and returns them as one batch."}),
            },
            "optional": {
                "pipe": ("PROMPTFORGE_PIPE", {"tooltip": "From a PromptForge Loader: sets the family and fills recommended settings when connected."}),
                "sigmas": ("SIGMAS", {"tooltip": "Overrides the scheduler."}),
                "custom_sigmas": ("STRING", {"default": "", "tooltip": "Comma separated descending sigmas, for example 1, 0.8, 0.5, 0. Overrides the scheduler."}),
                "step_cache": (["off", "auto", "easycache", "lazycache"], {"default": "off", "tooltip": "Skips model calls on steps where the output barely changes. easycache is general and also handles audio+video models; lazycache is cheaper but rougher. auto = easycache when steps >= 15, else off. Useful for many-step runs; distilled 4-8 step models should stay off."}),
                "cache_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 3.0, "step": 0.01, "tooltip": "Higher skips more steps (faster, softer). 0.1-0.2 is conservative."}),
                "cache_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Fraction of the run before which nothing is skipped."}),
                "cache_end": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "advanced": True}),
                "add_noise": ("BOOLEAN", {"default": True, "advanced": True}),
                "ltxv_max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "ltxv_base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
                "ltxv_terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01, "advanced": True}),
                "preview": (["default", "taesd", "latent2rgb", "none"], {"default": "default", "tooltip": "Live preview while sampling. default follows ComfyUI's --preview-method; taesd uses the model's tiny decoder when it is in models/vae_approx (taeh3 for MiniMax H3) and latent2rgb otherwise."}),
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING")
    RETURN_NAMES = ("LATENT", "SIGMAS", "INFO")
    FUNCTION = "sample"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "KSampler with family-aware schedules, custom sigmas, multiple seeds per run and saved presets."

    @staticmethod
    def _family(model, pipe):
        if pipe and pipe.get("family"):
            return pipe["family"]
        config = getattr(getattr(model, "model", None), "model_config", None)
        family = families.family_from_config(type(config).__name__) if config is not None else None
        return family.key if family else ""

    def _sigmas(self, model, sampler_name, scheduler, steps, denoise, latent_image, sigmas, custom_sigmas, shifts):
        """The sigmas to sample with: an input, a typed list, the LTXV schedule, or the named scheduler's."""
        if sigmas is not None:
            return sigmas
        if custom_sigmas.strip():
            return parse_sigmas(custom_sigmas)
        if scheduler == LTXV:
            total = max(steps, int(steps / denoise)) if denoise > 0 else steps
            schedule = nodes_lt.LTXVScheduler.execute(total, shifts[0], shifts[1], True, shifts[2], {"samples": latent_image}).args[0]
            return trim_for_denoise(schedule, steps)
        return comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options).sigmas

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, runs,
               pipe=None, sigmas=None, custom_sigmas="", step_cache="off", cache_threshold=0.2, cache_start=0.15, cache_end=0.95,
               start_at_step=0, end_at_step=10000, add_noise=True, ltxv_max_shift=2.05, ltxv_base_shift=0.95, ltxv_terminal=0.1, preview="default"):
        family = self._family(model, pipe)
        if step_cache == "auto":
            step_cache = "easycache" if steps >= AUTO_CACHE_MIN_STEPS else "off"
        if step_cache != "off":
            model = CACHES[step_cache].execute(model, cache_threshold, cache_start, cache_end, False).args[0]
        if scheduler == AUTO:
            scheduler = LTXV if family == "ltxv" else "simple"

        samples = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"], latent_image.get("downscale_ratio_spacial"), latent_image.get("downscale_ratio_temporal"))
        used = self._sigmas(model, sampler_name, scheduler, steps, denoise, samples, sigmas, custom_sigmas, (ltxv_max_shift, ltxv_base_shift, ltxv_terminal))
        if runs > 1 and not isinstance(samples, torch.Tensor):
            raise ValueError("[PromptForge] runs > 1 needs a plain latent; this model uses a combined audio+video latent.")

        device = comfy.model_management.get_torch_device()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latent_preview.set_preview_method(None if preview == "default" else preview)
        try:
            try:
                callback = latent_preview.prepare_callback(model, steps)
            except AttributeError:
                # the tiny decoder file is not a loadable VAE; the colour preview needs no file
                print("[PromptForge] the tiny preview decoder could not be loaded; using latent2rgb previews.")
                latent_preview.set_preview_method("latent2rgb")
                callback = latent_preview.prepare_callback(model, steps)
            progress = comfy.utils.PROGRESS_BAR_ENABLED
            results = []
            for index in range(runs):
                current = seed + index
                noise = comfy.sample.prepare_empty_noise(samples) if not add_noise else comfy.sample.prepare_noise(samples, current, latent_image.get("batch_index"))
                results.append(comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                    denoise=denoise, disable_noise=not add_noise, start_step=start_at_step or None,
                    last_step=end_at_step if end_at_step < steps else None, force_full_denoise=end_at_step >= steps,
                    noise_mask=latent_image.get("noise_mask"), sigmas=used, callback=callback, disable_pbar=not progress, seed=current))
        finally:
            latent_preview.set_preview_method()
        merged = results[0] if runs == 1 else torch.cat(results)

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = merged
        schedule = "custom sigmas" if sigmas is not None or custom_sigmas.strip() else scheduler
        info = f"{sampler_name} / {schedule}, {steps} steps, cfg {cfg:g}, denoise {denoise:g}, seed {seed}" + (f" x{runs}" if runs > 1 else "") + (f" ({family})" if family else "") + (f", {step_cache} {cache_threshold:g}" if step_cache != "off" else "")
        if device.type == "cuda":
            info += f", peak VRAM {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.1f} GB"
        return (out, used, info)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_Sampler": PROMPTFORGE_Sampler}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_Sampler": "PromptForge Sampler"}
