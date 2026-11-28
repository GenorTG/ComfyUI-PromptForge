"""V2 image stages. Live artifacts stay in the versioned pipe, never in settings."""
import math

import torch

import comfy.model_management
import comfy.samplers
import nodes
from comfy_api.latest import io
from comfy import supported_models
from comfy_extras.nodes_easycache import EasyCacheNode

from .. import unified_loader as loader
from .contracts import schema as nodes_schema, PIPE, new_pipe, ensure_pipe, evolve, require, take
from .krea import encode_grounded, patch_model
from .shared import expand_tokens, retry_unloaded
from .upscale import upscale_latent

Pipe = io.Custom(PIPE)
FAMILIES = {"krea2": "Krea 2", "moss_image": "MOSS Image", "anima": "Anima"}


def image_pipe(pipe):
    """Image stages take the image families; decode/save also accept the neutral family."""
    if pipe["family"] not in (*FAMILIES, "image"):
        raise ValueError("Image stages require krea2, moss_image or anima.")
    return pipe


def schema(key, title, inputs, outputs=None, **kwargs):
    return nodes_schema("IMAGE", key, title, inputs, outputs, **kwargs)


class ImageModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("MODEL", "Model", [
            io.Combo.Input("family", options=list(FAMILIES)),
            io.Combo.Input("model", options=loader._tagged(loader.MODEL_SOURCES) or ["(no models found)"]),
            io.Combo.Input("clip", options=[loader.BAKED] + loader._tagged(loader.CLIP_SOURCES)),
            io.Combo.Input("vae", options=[loader.BAKED] + loader._vae_names()),
            io.Combo.Input("weight_dtype", options=list(loader.WEIGHT_DTYPES), default="auto"),
            io.Combo.Input("optimize_for", options=["balanced", "quality", "speed"]),
            io.Combo.Input("clip_device", options=["default", "cpu"]),
        ], [Pipe.Output("pipe"), io.Model.Output("model"), io.Clip.Output("clip"), io.Vae.Output("vae"), io.String.Output("info")])

    @classmethod
    def execute(cls, family, model, clip, vae, weight_dtype="auto", optimize_for="balanced", clip_device="default"):
        if family not in FAMILIES:
            raise ValueError("Unsupported image family: " + family)
        # VAELoader handles built-in approximate VAEs as well as ordinary files
        if vae != loader.BAKED and vae not in loader._vae_names():
            raise ValueError("Select a registered VAE file.")
        result = loader.PROMPTFORGE_Loader().load(
            model, FAMILIES[family], weight_dtype, optimize_for, clip,
            loader.NONE, loader.NONE, loader.NONE,
            {"krea2": "krea2", "moss_image": "moss_image", "anima": "stable_diffusion"}[family], clip_device,
            vae, loader.NONE, loader.NONE)
        model_obj, clip_obj, vae_obj = result[:3]
        if clip_obj is None or vae_obj is None:
            raise ValueError("Image Model needs an explicit encoder and VAE for diffusion-only files.")
        expected = {"krea2": supported_models.Krea2, "moss_image": supported_models.MOSSImage, "anima": supported_models.Anima}
        actual = model_obj.model.model_config
        if not isinstance(actual, expected[family]):
            raise ValueError(f"Selected {family}, but the loaded model is {type(actual).__name__}.")
        pipe = evolve(new_pipe(family), "models", {"model": model, "clip": clip, "vae": vae, "weight_dtype": weight_dtype,
                                                       "optimize_for": optimize_for, "clip_device": clip_device},
                      {"model": model_obj, "clip": clip_obj, "vae": vae_obj})
        return io.NodeOutput(pipe, model_obj, clip_obj, vae_obj, result[5])


# landscape ratios, widest to square; orientation turns them around
ASPECT_NAMES = ("32:9", "3:1", "5:2", "12:5", "21:9", "2:1", "16:9", "5:3", "16:10", "3:2", "7:5", "4:3", "5:4", "6:5", "1:1")
ASPECTS = {name: int(name.split(":")[0]) / int(name.split(":")[1]) for name in ASPECT_NAMES}

# named sizes: (width, height, aspect, orientation). Presets only use the aspect keys above so switching to
# megapixels with the same choice lands on the same frame.
PRESETS = {
    "krea square 1024x1024": (1024, 1024, "1:1", "landscape"),
    "krea landscape 4:3 1152x864": (1152, 864, "4:3", "landscape"),
    "krea portrait 4:3 864x1152": (864, 1152, "4:3", "portrait"),
    "krea landscape 3:2 1200x800": (1200, 800, "3:2", "landscape"),
    "krea portrait 3:2 800x1200": (800, 1200, "3:2", "portrait"),
    "krea landscape 16:9 1344x768": (1344, 768, "16:9", "landscape"),
    "krea portrait 16:9 768x1344": (768, 1344, "16:9", "portrait"),
    "krea landscape 21:9 1456x624": (1456, 624, "21:9", "landscape"),
    "anima square 1024x1024": (1024, 1024, "1:1", "landscape"),
    "anima portrait 4:3 864x1152": (864, 1152, "4:3", "portrait"),
    "anima landscape 4:3 1152x864": (1152, 864, "4:3", "landscape"),
    "anima portrait 3:2 832x1248": (832, 1248, "3:2", "portrait"),
    "anima landscape 3:2 1248x832": (1248, 832, "3:2", "landscape"),
    "anima portrait 16:9 768x1344": (768, 1344, "16:9", "portrait"),
    "anima landscape 16:9 1344x768": (1344, 768, "16:9", "landscape"),
    "moss cover 1024x1536": (1024, 1536, "3:2", "portrait"),
    "moss cover 4:3 960x1280": (960, 1280, "4:3", "portrait"),
    "moss square 1024x1024": (1024, 1024, "1:1", "landscape"),
    "moss spread 1536x1024": (1536, 1024, "3:2", "landscape"),
    "moss spread 4:3 1280x960": (1280, 960, "4:3", "landscape"),
    "moss cinema 16:9 1344x768": (1344, 768, "16:9", "landscape"),
    "moss banner 21:9 1680x720": (1680, 720, "21:9", "landscape"),
}
GENERIC = {
    "generic square 512x512": (512, 512, "1:1", "landscape"),
    "generic square 768x768": (768, 768, "1:1", "landscape"),
    "generic square 1024x1024": (1024, 1024, "1:1", "landscape"),
    "generic hd 1280x720": (1280, 720, "16:9", "landscape"),
    "generic portrait hd 720x1280": (720, 1280, "16:9", "portrait"),
    "generic large 1536x864": (1536, 864, "16:9", "landscape"),
    "generic portrait large 864x1536": (864, 1536, "16:9", "portrait"),
}
SIZING = ("megapixels", "presets", "generic", "custom")


def snap16(value):
    return max(16, round(value / 16) * 16)


class ImageFormat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("FORMAT", "Format", [Pipe.Input("pipe", optional=True),
            io.Model.Input("model", optional=True),
            io.Combo.Input("sizing", options=list(SIZING), default="megapixels",
                           tooltip="How the size is chosen: a family or generic resolution list, aspect ratio + megapixels, or exact width and height."),
            io.Combo.Input("preset", options=[*PRESETS, *GENERIC], default="krea square 1024x1024",
                           tooltip="Named size for the presets and generic modes. Family presets follow each model's own training sizes."),
            io.Combo.Input("aspect", options=[*ASPECTS, "custom"], default="1:1"),
            io.Float.Input("megapixels", default=1.0, min=0.01, max=64.0, step=0.1),
            io.Combo.Input("orientation", options=["landscape", "portrait"]),
            io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
            io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
            io.Int.Input("batch_size", default=1, min=1, max=4096)],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, sizing="megapixels", preset="krea square 1024x1024", aspect="1:1", megapixels=1.0,
                orientation="landscape", width=1024, height=1024, batch_size=1, model=None):
        pipe = image_pipe(take(ensure_pipe(pipe, model=model), model=model))
        entry = (GENERIC if sizing == "generic" else PRESETS).get(preset)
        if sizing == "custom":
            width, height = snap16(width), snap16(height)
            orientation = "portrait" if height > width else "landscape"
            megapixels = round(width * height / 1_000_000, 3)
        elif entry is not None and sizing in ("presets", "generic"):
            width, height, aspect, orientation = entry
            megapixels = round(width * height / 1_000_000, 3)
        elif aspect != "custom":
            ratio = ASPECTS[aspect]
            if orientation == "portrait":
                ratio = 1 / ratio
            width = max(16, round(math.sqrt(megapixels * 1_000_000 * ratio) / 16) * 16)
            height = max(16, round(math.sqrt(megapixels * 1_000_000 / ratio) / 16) * 16)
        settings = {"aspect": aspect, "megapixels": megapixels, "orientation": orientation,
                    "width": width, "height": height, "batch_size": batch_size}
        result = evolve(pipe, section="format", settings=settings,
                        invalidate=("positive", "negative", "latent", "images", "krea_source_latents", "krea_references", "krea_text_references", "krea_masks"))
        model_obj = result["artifacts"].get("model")
        latent = None
        if model_obj is not None:
            lf = model_obj.model.latent_format
            ratio = lf.spacial_downscale_ratio
            shape = [batch_size, lf.latent_channels, height // ratio, width // ratio]
            latent = {"samples": torch.zeros(shape, device=comfy.model_management.intermediate_device(),
                                             dtype=comfy.model_management.intermediate_dtype()),
                      "downscale_ratio_spacial": ratio}
            result = evolve(result, artifacts={"latent": latent})
        return io.NodeOutput(result, latent)


class ImageSamplingSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SAMPLING_SETTINGS", "Sampling Settings", [Pipe.Input("pipe", optional=True),
            io.Model.Input("model", optional=True), io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            io.Int.Input("steps", default=10, min=1, max=10000),
            io.Float.Input("cfg", default=1.0, min=0.0, max=100.0),
            io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
            io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
            io.Float.Input("denoise", default=1.0, min=0.0, max=1.0),
            io.Float.Input("refine_denoise", default=0.25, min=0.0, max=1.0,
                           tooltip="Denoise of the second pass after Latent Upscale. 0.20 to 0.25 keeps the first pass; around 0.5 recovers a plain bicubic."),
            io.Int.Input("refine_steps", default=4, min=1, max=1000,
                         tooltip="Steps of the second pass, which runs only when Latent Upscale is enabled."),
            io.Boolean.Input("use_nag", default=False),
            io.Float.Input("nag_phi", default=4.0, min=0.0, max=20.0),
            io.Float.Input("nag_tau", default=2.5, min=0.01, max=20.0),
            io.Float.Input("nag_alpha", default=0.25, min=0.0, max=1.0),
            io.Boolean.Input("easycache", default=False, optional=True,
                             tooltip="Experimental: ComfyUI's EasyCache reuses earlier results to skip steps in the first pass. Faster on some models, and the picture changes."),
            io.Float.Input("easycache_threshold", default=0.2, min=0.0, max=3.0, step=0.01, optional=True,
                           tooltip="How much change EasyCache lets a skipped step hide. Higher skips more and drifts further from the normal picture.")])

    @classmethod
    def execute(cls, pipe=None, seed=0, steps=10, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0,
                refine_denoise=0.25, refine_steps=4, use_nag=False, nag_phi=4.0, nag_tau=2.5, nag_alpha=0.25, model=None,
                easycache=False, easycache_threshold=0.2):
        pipe = image_pipe(ensure_pipe(pipe, model=model))
        if use_nag and (pipe["family"] != "krea2" or cfg != 1.0):
            raise ValueError("Krea NAG requires family krea2 and CFG 1.")
        settings = {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler,
                        "denoise": denoise, "refine_denoise": refine_denoise, "refine_steps": refine_steps,
                        "use_nag": use_nag, "nag_phi": nag_phi, "nag_tau": nag_tau, "nag_alpha": nag_alpha,
                        "easycache": easycache, "easycache_threshold": easycache_threshold}
        return io.NodeOutput(evolve(pipe, section="sampling", settings=settings, invalidate=("images",)))


def refine(pipe, latent):
    """Second pass: upscale the sampled latent, then denoise it at the refine strength."""
    settings = pipe["settings"].get("upscale", {})
    sampling = pipe["settings"].get("sampling")
    if not sampling:
        raise ValueError("The second pass needs the Sampling Settings stage before Latent Upscale.")
    model, positive, negative = require(pipe, "model", "positive", "negative")
    latent = upscale_latent(pipe, latent, settings.get("method", "bicubic"), settings.get("scale", 1.41))
    if pipe["family"] == "krea2" and (pipe["artifacts"].get("krea_source_latents") or sampling.get("use_nag")):
        model = patch_model(model, pipe["artifacts"]["negative"], pipe["artifacts"].get("krea_source_latents", []),
                            {**pipe["settings"].get("krea_edit", {}), "masks": pipe["artifacts"].get("krea_masks")}, sampling)
    upscaled = latent
    latent = retry_unloaded("refining", lambda: nodes.common_ksampler(model, (sampling.get("seed", 0) + 1) % 2**64, sampling.get("refine_steps", 4),
                                   sampling.get("cfg", 1.0), sampling.get("sampler_name", "euler"),
                                   sampling.get("scheduler", "simple"), positive, negative, upscaled,
                                   denoise=sampling.get("refine_denoise", 0.25))[0])
    fmt = pipe["settings"].get("format")
    if fmt:
        ratio = latent.get("downscale_ratio_spacial") or 8
        height, width = latent["samples"].shape[-2] * ratio, latent["samples"].shape[-1] * ratio
        pipe = evolve(pipe, "format", dict(fmt, width=width, height=height, megapixels=round(width * height / 1_000_000, 3)))
    return evolve(pipe, artifacts={"latent": latent}, invalidate=("images",))


class ImageEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("ENCODE", "Encode", [Pipe.Input("pipe", optional=True),
            io.Model.Input("model", optional=True), io.Combo.Input("task", options=["t2i", "edit"]),
            io.String.Input("text", optional=True, force_input=True, tooltip="Prompt text when no prompt stage is chained."),
            io.Int.Input("clip_skip", default=-1, min=-24, max=-1, step=1,
                         tooltip="Encoder layer stop: -1 uses the last layer, -2 skips one, -3 two. Older anime checkpoints are trained with -2."),
            io.Clip.Input("clip", optional=True), io.Vae.Input("vae", optional=True),
            io.Latent.Input("latent", optional=True), io.Boolean.Input("pipe_format", default=True, tooltip="On: take the size from the Format node when one is chained."),
            io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
            io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
            io.Int.Input("batch_size", default=1, min=1, max=4096)],
            [Pipe.Output("pipe"), io.Conditioning.Output("positive"), io.Conditioning.Output("negative"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, task="t2i", clip=None, vae=None, latent=None, model=None, text=None,
                pipe_format=True, width=1024, height=1024, batch_size=1, clip_skip=-1):
        pipe = image_pipe(ensure_pipe(pipe, model=model))
        pipe = take(pipe, model=model, clip=clip, vae=vae, latent=latent)
        if text and not pipe["artifacts"].get("text"):
            pipe = evolve(pipe, artifacts={"text": text})
            if not pipe["settings"].get("prompt"):
                pipe = evolve(pipe, section="prompt", settings={"source": "raw", "text": text, "negative": ""})
        require(pipe, "model", "clip", "vae")
        require(pipe, "text")
        prompt = pipe["settings"].get("prompt", {})
        writer = pipe["artifacts"].get("writer", {})
        if prompt.get("source") != "raw" and writer.get("status") != "valid":
            raise ValueError("Run the prompt writer successfully before encoding enhanced text.")
        positive, negative = pipe["artifacts"]["text"], prompt.get("negative", "")
        artifacts = pipe["artifacts"]
        clip = artifacts["clip"]
        if clip_skip != -1:
            clip = nodes.CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
        refs = artifacts.get("krea_references", [])
        if task == "edit" and pipe["family"] == "krea2":
            if not refs:
                raise ValueError("Run Krea Edit Prepare before Krea edit encoding.")
            pos, neg = (encode_grounded(clip, text, artifacts["krea_text_references"], pipe["settings"]["krea_edit"]["grounding_px"]) for text in (positive, negative))
        elif task == "edit":
            raise ValueError("Edit encoding is supported for Krea 2 only.")
        else:
            if refs:
                raise ValueError("A prepared Krea edit pipe requires task=edit; branch before Edit Prepare for T2I.")
            pos, neg = (clip.encode_from_tokens_scheduled(clip.tokenize(text)) for text in (positive, negative))
        if latent is None:
            fmt = pipe["settings"].get("format") if pipe_format else None
            fmt = fmt or {"width": width, "height": height, "batch_size": batch_size}
            lf = artifacts["model"].model.latent_format
            ratio = lf.spacial_downscale_ratio
            shape = [fmt["batch_size"], lf.latent_channels, fmt["height"] // ratio, fmt["width"] // ratio]
            latent = {"samples": torch.zeros(shape, device=comfy.model_management.intermediate_device(),
                                             dtype=comfy.model_management.intermediate_dtype()), "downscale_ratio_spacial": ratio}
        out = evolve(pipe, artifacts={"positive": pos, "negative": neg, "latent": latent}, invalidate=("images",))
        return io.NodeOutput(out, pos, neg, latent)


class ImageSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SAMPLE", "Sample", [Pipe.Input("pipe", optional=True),
            io.Conditioning.Input("positive", optional=True), io.Conditioning.Input("negative", optional=True),
            io.Latent.Input("latent", optional=True), io.Model.Input("model", optional=True),
            io.Boolean.Input("pipe_settings", default=True, tooltip="On: use the Sampling Settings node when one is chained. Off: use the widgets here."),
            io.Int.Input("seed", optional=True, force_input=True,
                         tooltip="Connected seed for this call. Unconnected takes the seed from Sampling Settings."),
            io.Int.Input("steps", default=10, min=1, max=10000),
            io.Float.Input("cfg", default=1.0, min=0.0, max=100.0),
            io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
            io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
            io.Float.Input("denoise", default=1.0, min=0.0, max=1.0)],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, positive=None, negative=None, latent=None, model=None, pipe_settings=True,
                seed=None, steps=10, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0):
        pipe = image_pipe(ensure_pipe(pipe, model=model))
        pipe = take(pipe, model=model, positive=positive, negative=negative, latent=latent)
        if not pipe_settings or not pipe["settings"].get("sampling"):
            pipe = evolve(pipe, section="sampling", settings={
                "seed": seed if seed is not None else pipe["settings"].get("sampling", {}).get("seed", 0),
                "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler,
                "denoise": denoise, "refine_denoise": 0.25, "refine_steps": 4,
                "use_nag": False, "nag_phi": 4.0, "nag_tau": 2.5, "nag_alpha": 0.25})
        require(pipe, "model", "positive", "negative", "latent")
        a = pipe["artifacts"]
        s = pipe["settings"]["sampling"]
        model = a["model"]
        # before the Krea patch, so the cache wraps it; a model that arrives with its own EasyCache keeps that one
        if s.get("easycache") and "easycache" not in model.model_options["transformer_options"]:
            model = EasyCacheNode.execute(model, s.get("easycache_threshold", 0.2), 0.15, 0.95, False).args[0]
        if pipe["family"] == "krea2" and (a.get("krea_source_latents") or s.get("use_nag")):
            model = patch_model(model, a["negative"], a.get("krea_source_latents", []),
                                {**pipe["settings"].get("krea_edit", {}), "masks": a.get("krea_masks")}, s)
        latent = retry_unloaded("sampling", lambda: nodes.common_ksampler(model, s["seed"], s["steps"], s["cfg"], s["sampler_name"],
                                      s["scheduler"], a["positive"], a["negative"], a["latent"], denoise=s["denoise"])[0])
        return io.NodeOutput(evolve(pipe, artifacts={"latent": latent}, invalidate=("images",)), latent)


class ImageDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("DECODE", "Decode", [Pipe.Input("pipe", optional=True), io.Vae.Input("vae", optional=True),
            io.Latent.Input("latent", optional=True)], [Pipe.Output("pipe"), io.Image.Output("images")])

    @classmethod
    def execute(cls, pipe=None, vae=None, latent=None):
        pipe = take(image_pipe(ensure_pipe(pipe)), vae=vae, latent=latent)
        require(pipe, "vae", "latent")
        images = nodes.VAEDecode().decode(pipe["artifacts"]["vae"], pipe["artifacts"]["latent"])[0]
        return io.NodeOutput(evolve(pipe, artifacts={"images": images}), images)


class ImageSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SAVE", "Save", [Pipe.Input("pipe", optional=True), io.Image.Input("images", optional=True),
                      io.String.Input("filename_prefix", default="PromptForgeV2",
                                      tooltip="Prefix for the saved file. Tokens expand per run: %date:yyyy-MM-dd%, %date:yyyy-MM-dd_HH-mm-ss%, %date:yyyy%, %date:MM%, %date:dd%, plus %family%, %model%, %mode%, %source%, %category%, %action% and %seed% from the run.")],
                      outputs=[Pipe.Output("pipe"), io.Image.Output("images")],
                      output=True, hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo])

    @classmethod
    def execute(cls, pipe=None, images=None, filename_prefix="PromptForgeV2"):
        pipe = take(image_pipe(ensure_pipe(pipe)), images=images)
        require(pipe, "images")
        hidden = getattr(cls, "hidden", None)
        saved = nodes.SaveImage().save_images(pipe["artifacts"]["images"], expand_tokens(filename_prefix, pipe),
                                             prompt=getattr(hidden, "prompt", None),
                                             extra_pnginfo=getattr(hidden, "extra_pnginfo", None))
        return io.NodeOutput(pipe, pipe["artifacts"]["images"], ui=saved["ui"])


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_V2_IMAGE_" + key: node for key, node in (
    ("MODEL", ImageModel), ("FORMAT", ImageFormat), ("SAMPLING_SETTINGS", ImageSamplingSettings),
    ("ENCODE", ImageEncode), ("SAMPLE", ImageSample), ("DECODE", ImageDecode), ("SAVE", ImageSave))}
NODE_DISPLAY_NAME_MAPPINGS = {key: "PromptForge V2 · Image · " + key.removeprefix("PROMPTFORGE_V2_IMAGE_").replace("_", " ").title()
                              for key in NODE_CLASS_MAPPINGS}
