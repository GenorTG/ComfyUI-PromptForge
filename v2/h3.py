"""Explicit MiniMax H3 audiovisual stages using local ComfyUI core APIs.

Reference banks contain ordered entries {name, kind, data, audio}; selection
uses comma-separated names and preserves that order. Video references are 24fps.
"""
import math
import re
import os

import torch
import folder_paths
import nodes
import node_helpers
import comfy.audio
import comfy.utils
import comfy.samplers
import comfy.model_management
from comfy_api.latest import io, ui, InputImpl
from comfy_extras import nodes_minimax_h3 as core
from comfy_extras import nodes_hunyuan, nodes_custom_sampler, nodes_audio
from .. import frame_buffer, minimax_decode, video_save, unified_loader as loader
from .contracts import schema as nodes_schema, PIPE, new_pipe, ensure_pipe, evolve, require, take
from .image import ASPECT_NAMES
from .shared import expand_tokens, free_vram, retry_unloaded
# the sequence nodes use these same encode/sample/upscale/refine/decode/join services
from .h3_sequence import H3Parts, H3Continuity, H3Sequence

Pipe = io.Custom(PIPE)
DIRTY = ("positive", "negative", "latent", "images", "audio", "video", "first_pass_audio")
OUTPUTS = ("images", "audio", "video")
MODES = ["t2v", "i2v", "l2v", "fl2v", "ref2v", "continue"]
DECODE_VAE = "decode vae"
DECODE_CHOICES = [DECODE_VAE, "video vae"]
DECODE_TOOLTIP = ("decode vae: the Video Model's decode_vae_name (the video VAE when it says same); a wired vae input is not used then. "
                  "video vae: the main video VAE, or the one wired into vae.")


def schema(key, title, inputs, outputs=None, output=False):
    return nodes_schema("H3", key, title, inputs, outputs, output)


def checked(pipe):
    if pipe["family"] != "minimax_h3":
        raise ValueError("H3 stages require a minimax_h3 family pipe.")
    return pipe


def grown(kind, name):
    return io.Autogrow.Input(name, optional=True, template=io.Autogrow.TemplatePrefix(
        input=kind.Input("item"), prefix="item_", min=0))


class H3Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("MODEL", "Video Model", [
            io.Combo.Input("model_name", options=loader._tagged(loader.MODEL_SOURCES) or ["(no models found)"]),
            io.Combo.Input("clip_name", options=loader._tagged(loader.CLIP_SOURCES) or ["(no models found)"]),
            io.Combo.Input("vae_name", options=loader._vae_names() or ["(no models found)"]),
            io.Combo.Input("audio_vae_name", options=loader._vae_names() or ["(no models found)"]),
            io.Combo.Input("decode_vae_name", options=["same"] + loader._vae_names(), default="same"),
            io.Combo.Input("weight_dtype", options=list(loader.WEIGHT_DTYPES), default="default"),
            io.Combo.Input("optimize_for", options=["balanced", "quality", "speed"], default="balanced"),
            Pipe.Input("pipe", optional=True),
            # last, after the widgets older workflows saved; the node's own menu filters model_name with it
            io.Combo.Input("family", options=["minimax_h3", "all"], default="minimax_h3", optional=True,
                           tooltip="Which files model_name lists: MiniMax H3 models only, or every model file.")],
            [Pipe.Output("pipe"), io.Model.Output("model"), io.Clip.Output("clip"), io.Vae.Output("vae"),
             io.Vae.Output("audio_vae"), io.Vae.Output("decode_vae", tooltip="The decode VAE, or the video VAE when decode_vae_name is same.")])

    @classmethod
    def execute(cls, model_name, clip_name, vae_name, audio_vae_name, decode_vae_name="same", weight_dtype="default", optimize_for="balanced", pipe=None,
                family="minimax_h3"):
        pipe = checked(pipe) if pipe is not None else new_pipe("minimax_h3")
        model, clip, vae, audio_vae, decoder, _, _ = loader.PROMPTFORGE_Loader().load(
            model_name, "auto", weight_dtype, optimize_for, clip_name, loader.NONE, loader.NONE, loader.NONE,
            "minimax", "default", vae_name, audio_vae_name,
            loader.NONE if decode_vae_name == "same" else decode_vae_name)
        decoder = decoder or vae
        result = evolve(pipe, section="models", settings={"model": model_name, "clip": clip_name, "vae": vae_name,
            "audio_vae": audio_vae_name, "decode_vae": decode_vae_name, "weight_dtype": weight_dtype, "optimize_for": optimize_for},
            artifacts={"model": model, "clip": clip, "vae": vae, "audio_vae": audio_vae, "decode_vae": decoder}, invalidate=DIRTY)
        return io.NodeOutput(result, model, clip, vae, audio_vae, decoder)


class H3Format(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("FORMAT", "Format", [Pipe.Input("pipe", optional=True), io.Int.Input("width", default=1344, min=32, max=16384, step=32),
            io.Int.Input("height", default=768, min=32, max=16384, step=32),
            io.Float.Input("seconds", default=5.0, min=0.01, max=3600),
            io.Combo.Input("aspect", options=[*ASPECT_NAMES, "custom"], default="custom", tooltip="Width to height; orientation turns it upright. custom uses width and height."),
            io.Float.Input("megapixels", default=1.0, min=0.01, max=64),
            io.Combo.Input("orientation", options=["landscape", "portrait"], default="landscape")],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def validate_inputs(cls, aspect="custom", orientation="landscape"):
        # saved API graphs may still hold an upright ratio such as 9:16, or the old "as selected"
        return True

    @classmethod
    def execute(cls, pipe=None, width=1344, height=768, seconds=5.0, aspect="custom", megapixels=1.0, orientation="landscape"):
        pipe = checked(ensure_pipe(pipe, family="minimax_h3"))
        if aspect != "custom":
            aw, ah = (float(value) for value in aspect.split(":"))
            upright = orientation == "portrait" or orientation != "landscape" and aw < ah
            ratio = max(aw, ah) / min(aw, ah)
            height = math.sqrt(megapixels * 1024 * 1024 / ratio)
            width = height * ratio
            if upright:
                width, height = height, width
        orientation = "portrait" if height > width else "landscape" if width > height else "square"
        frames = core.align_frame_count(max(5, math.ceil(seconds * 24)))
        fmt = {"width": max(32, round(width / 32) * 32), "height": max(32, round(height / 32) * 32),
                   "fps": 24, "seconds": frames / 24, "frames": frames, "requested_seconds": seconds, "aspect": aspect, "megapixels": megapixels, "orientation": orientation}
        result = evolve(pipe, section="format", settings=fmt, invalidate=DIRTY)
        model = result["artifacts"].get("model")
        latent = None
        if model is not None:
            lf = model.model.latent_format
            ratio = lf.spacial_downscale_ratio
            # H3 packs 17 frames into 5 latents plus a 5 frame offset
            latent_frames = 5 * ((fmt["frames"] - 5) // 17) + 2 if fmt["frames"] >= 22 else 2
            shape = [1, lf.latent_channels, latent_frames, fmt["height"] // ratio, fmt["width"] // ratio]
            latent = {"samples": torch.zeros(shape, device=comfy.model_management.intermediate_device(),
                                             dtype=comfy.model_management.intermediate_dtype()),
                      "downscale_ratio_spacial": ratio}
            result = evolve(result, artifacts={"latent": latent})
        return io.NodeOutput(result, latent)


class H3References(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("REFERENCES", "References", [Pipe.Input("pipe", optional=True), io.Combo.Input("mode", options=MODES, default="t2v"),
            io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
            io.Combo.Input("frame_fit", options=["match first frame", "crop to workflow aspect"], default="match first frame",
                           tooltip="Match first frame sizes the video from the frame, so nothing is stretched or cut. Crop keeps the format size."),
            io.Image.Input("first_frame", optional=True), io.Image.Input("last_frame", optional=True),
            io.Image.Input("reference_video", optional=True), io.Audio.Input("reference_audio", optional=True),
            grown(io.Image, "reference_images")])

    @classmethod
    def execute(cls, pipe=None, mode="t2v", ref_image_size="match", frame_fit="match first frame",
                first_frame=None, last_frame=None, reference_video=None, reference_audio=None, reference_images=None):
        checked(pipe)
        if mode not in MODES:
            raise ValueError("Unknown H3 reference mode.")
        artifacts = {name: value for name, value in {"first_frame": first_frame, "last_frame": last_frame,
            "reference_video": reference_video, "reference_audio": reference_audio}.items() if value is not None}
        if reference_images:
            artifacts["reference_images"] = [value for value in reference_images.values() if value is not None]
        a = {**pipe["artifacts"], **artifacts}
        if mode in ("i2v", "fl2v") and a.get("first_frame") is None and a.get("source_image") is not None:
            artifacts["first_frame"] = a["source_image"][:1]
        if mode in ("l2v", "fl2v") and a.get("last_frame") is None and a.get("source_image") is not None:
            artifacts["last_frame"] = a["source_image"][-1:]
        if mode == "continue":
            for target, source in (("reference_video", "continue_video"), ("reference_audio", "continue_audio")):
                if target not in artifacts and a.get(source) is not None:
                    artifacts[target] = a[source]
        frame = first_frame if first_frame is not None else last_frame
        for candidate in ("first_frame", "last_frame", "images"):
            if frame is None and a.get(candidate) is not None:
                frame = a[candidate]
        if frame_fit == "match first frame" and frame is not None and mode in ("i2v", "fl2v", "l2v"):
            pipe = _frame_format(pipe, frame)
        refs = dict(pipe["settings"].get("references", {}), mode=mode, ref_image_size=ref_image_size, frame_fit=frame_fit)
        return io.NodeOutput(evolve(pipe, section="references", settings=refs, artifacts=artifacts, invalidate=DIRTY))


def _frame_format(pipe, frame):
    """Size the clip from the frame: same aspect, same megapixels, no stretch or crop."""
    fmt = dict(pipe["settings"]["format"])
    megapixels = fmt.get("megapixels") or fmt["width"] * fmt["height"] / 1e6
    height, width = frame.shape[1:3]
    scale = (megapixels * 1e6 / (width * height)) ** 0.5
    # 32 is the smallest safe step: 16 for the VAE and 2 for the patch
    width, height = max(64, round(width * scale / 32) * 32), max(64, round(height * scale / 32) * 32)
    pipe = evolve(pipe, "format", dict(fmt, width=width, height=height, aspect=f"{width}:{height}",
                                       orientation="portrait" if height > width else "landscape",
                                       megapixels=round(width * height / 1e6, 2)))
    return pipe


def fit_frame(image, width, height):
    """Centre-crop to the target aspect, then scale: never stretch."""
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)


def video_at_24(frames, fps):
    if fps == 24:
        return frames
    if fps <= 0:
        raise ValueError("Reference video frame rate must be positive.")
    count = max(1, round(frames.shape[0] * 24 / fps))
    indices = torch.arange(count, device=frames.device).mul(fps / 24).floor().long().clamp(max=frames.shape[0] - 1)
    return torch.index_select(frames, 0, indices, out=frame_buffer.empty((count, *frames.shape[1:]), frames.dtype, frames.device))


_SECTIONS = {"t2v": ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
             "i2v": ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
             "l2v": ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
             "fl2v": ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
             "ref2v": ("subject_definitions", "summary", "retention_analysis", "detailed_description",
                       "overall_soundscape", "non_diegetic_music"),
             "continue": ("subject_definitions", "summary", "retention_analysis", "detailed_description",
                          "overall_soundscape", "non_diegetic_music")}


def check_syntax(text, mode):
    """H3 reads fixed sections with numbered shots; refuse text the model cannot parse."""
    sections = _SECTIONS.get(mode, _SECTIONS["t2v"])
    positions = []
    for name in sections:
        match = re.search(rf"^{name}:", text, re.M)
        if match is None:
            raise ValueError(f"H3 {mode} prompt is missing the '{name}:' section heading.")
        positions.append(match.start())
    if positions != sorted(positions):
        raise ValueError("H3 prompt sections are out of order: " + ", ".join(sections))
    shots = [int(number) for number in re.findall(r"\[Shot (\d+)\]", text)]
    if not shots or shots != list(range(1, len(shots) + 1)):
        raise ValueError("H3 prompt needs [Shot 1] first, then consecutive [Shot N] labels.")


def encode(pipe):
    checked(pipe)
    prompt = pipe["settings"].get("prompt", {})
    writer = pipe["artifacts"].get("writer", {})
    if prompt.get("source") != "raw" and writer.get("status") != "valid":
        raise ValueError("Run the prompt writer successfully before encoding enhanced text.")
    clip, vae, text = require(pipe, "clip", "vae", "text")
    fmt = pipe["settings"]["format"]
    refs = pipe["settings"].get("references", {})
    mode = refs.get("mode", "t2v")
    from .prompting import _h3_repair_audio, _h3_repair_sections
    text = _h3_repair_audio(_h3_repair_sections(text, pipe, mode), pipe, mode)
    check_syntax(text, mode)
    a = pipe["artifacts"]
    args = {"clip": clip, "vae": vae, "prompt": text, "width": fmt["width"], "height": fmt["height"], "length": fmt["frames"]}
    if mode == "ref2v":
        print(f"[PromptForge] references: {len(a.get('reference_images') or [])} pictures, "
              f"{len(a.get('reference_videos') or [])} videos, {len(a.get('reference_audios') or [])} voices", flush=True)
        videos = a.get("reference_videos") or ([video_at_24(a["reference_video"], a.get("reference_video_fps", 24))] if a.get("reference_video") is not None else [])
        audios = a.get("reference_audios") or ([a["reference_audio"]] if a.get("reference_audio") is not None else [])
        positive, latent = core.MiniMaxH3ReferenceToVideo.execute(**args, audio_vae=a.get("audio_vae"),
            ref_image_size=refs.get("ref_image_size", "match"),
            ref_images={f"ref_image_{i}": v for i, v in enumerate(a.get("reference_images", []), 1)},
            ref_videos={f"ref_video_{i}": v for i, v in enumerate(videos, 1)},
            ref_video_audios={f"ref_video_audio_{i}": v for i, v in enumerate(a.get("reference_video_audios") or [a.get("reference_video_audio")], 1)},
            ref_audios={f"ref_audio_{i}": v for i, v in enumerate(audios, 1)}).args
    else:
        if mode in ("i2v", "fl2v"):
            args["first_frame"] = fit_frame(require(pipe, "first_frame"), fmt["width"], fmt["height"])
        if mode in ("l2v", "fl2v"):
            args["last_frame"] = fit_frame(require(pipe, "last_frame"), fmt["width"], fmt["height"])
        positive, latent = core.MiniMaxH3ImageToVideo.execute(**args).args
        if mode == "continue":
            source = video_at_24(require(pipe, "reference_video"), a.get("reference_video_fps", a.get("continue_fps", 24)))
            count = min(pipe["settings"].get("continuity", {}).get("carried_frames", 22), source.shape[0], fmt["frames"])
            count = max(1, count)
            count = 1 if count < 5 else count - ((count - 5) % 17)
            guide_audio = a.get("reference_audio") or a.get("reference_video_audio")
            if guide_audio is not None:
                rate = guide_audio["sample_rate"]
                guide_audio = dict(guide_audio, waveform=guide_audio["waveform"][..., -round(count / 24 * rate):].clone())
            positive = core.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, audio_vae=a.get("audio_vae"),
                image=source[-count:], audio=guide_audio).args[0]
            pipe = evolve(pipe, artifacts={"continuation_overlap": count})
    negative_text = pipe["settings"].get("prompt", {}).get("negative", "")
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
    return evolve(pipe, artifacts={"positive": positive, "negative": negative, "latent": latent}, invalidate=OUTPUTS)


class H3Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("ENCODE", "Encode", [Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
            io.String.Input("text", optional=True, force_input=True, tooltip="Prompt text when no prompt stage is chained."),
            io.Int.Input("clip_skip", default=-1, min=-24, max=-1, step=1,
                         tooltip="Encoder layer stop: -1 uses the last layer, -2 skips one, -3 two. Older anime checkpoints are trained with -2."),
            io.Clip.Input("clip", optional=True), io.Vae.Input("vae", optional=True), io.Latent.Input("latent", optional=True)], [Pipe.Output("pipe"),
            io.Conditioning.Output("positive"), io.Conditioning.Output("negative"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, clip=None, vae=None, latent=None, model=None, text=None, clip_skip=-1):
        pipe = take(checked(ensure_pipe(pipe, model=model)), model=model, clip=clip, vae=vae, latent=latent)
        if clip_skip != -1:
            pipe = evolve(pipe, artifacts={"clip": nodes.CLIPSetLastLayer().set_last_layer(pipe["artifacts"]["clip"], clip_skip)[0]})
        if text and not pipe["artifacts"].get("text"):
            pipe = evolve(pipe, artifacts={"text": text})
            if not pipe["settings"].get("prompt"):
                pipe = evolve(pipe, section="prompt", settings={"source": "raw", "text": text, "negative": ""})
        result = retry_unloaded("encoding", lambda: encode(pipe))
        return io.NodeOutput(result, *require(result, "positive", "negative", "latent"))


def sampling_inputs():
    return [Pipe.Input("pipe", optional=True), io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
        io.Int.Input("steps", default=8, min=1, max=1000), io.Float.Input("cfg", default=1.0, min=0, max=100),
        io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
        io.Combo.Input("scheduler", options=list(dict.fromkeys([*comfy.samplers.KSampler.SCHEDULERS, "beta57"])), default="beta57"),
        io.Float.Input("shift_video", default=0.0, min=0.0, max=100),
        io.Float.Input("shift_audio", default=0.0, min=0.0, max=100)]


def refresh_keyframes(pipe, positive, latent):
    if not positive or not positive[0][1].get("minimax_keyframes"):
        return positive
    a = pipe["artifacts"]
    refs = pipe["settings"].get("references", {})
    mode = refs.get("mode", "t2v")
    positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": []})
    video = latent["samples"].unbind()[0]
    width, height = video.shape[-1] * 16, video.shape[-2] * 16
    for name, index in (("first_frame", 0), ("last_frame", -1)):
        if name == "first_frame" and mode not in ("i2v", "fl2v") or name == "last_frame" and mode not in ("l2v", "fl2v"):
            continue
        image = require(pipe, name)[:1]
        if name == "first_frame":
            image = core._resize(image, width, height, "disabled")
        positive = core.MiniMaxH3AddGuide.execute(positive, latent, index, vae=require(pipe, "vae"), image=image).args[0]
    if mode == "continue":
        source = video_at_24(require(pipe, "reference_video"), a.get("reference_video_fps", a.get("continue_fps", 24)))
        count = a["continuation_overlap"]
        audio = a.get("reference_audio") or a.get("reference_video_audio")
        if audio is not None:
            audio = dict(audio, waveform=audio["waveform"][..., -round(count / 24 * audio["sample_rate"]):].clone())
        positive = core.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=require(pipe, "vae"),
            audio_vae=a.get("audio_vae"), image=source[-count:], audio=audio).args[0]
    return positive


def sample(pipe, seed=0, steps=30, cfg=1.0, sampler_name="euler", scheduler="simple", shift_video=12.0, shift_audio=3.0, start_sigma=None):
    checked(pipe)
    model, positive, negative, latent = require(pipe, "model", "positive", "negative", "latent")
    # a leftover encoder can starve the sampler; the model comes back from the patcher
    if comfy.model_management.get_free_memory() < 1 << 30:
        free_vram()
    if start_sigma is not None:
        positive = refresh_keyframes(pipe, positive, latent)
    if shift_video or shift_audio:
        original = model.get_model_object("model_sampling")
        video_shift = shift_video or original.shift
        audio_shift = shift_audio or original.audio_shift or original.shift
        model = core.MiniMaxH3SigmaShift.execute(model, video_shift, audio_shift).args[0]
    if start_sigma is None and scheduler != "beta57":
        output = retry_unloaded("sampling", lambda: nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent)[0])
    else:
        if scheduler == "beta57":
            sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=0.5, beta=0.7)
            if start_sigma is not None:
                sigmas = sigmas * (start_sigma / float(sigmas[0]))
        elif scheduler == "linear from start":
            # 1.0486 bows the line like the reference 3-step sigmas (0.9035, 0.6316, 0.3158); it is capped so any step count keeps decreasing
            bow = min(1.0486, steps / max(steps - 1, 1) * 0.999)
            sigmas = torch.tensor([start_sigma] + [start_sigma * bow * (steps - i) / steps for i in range(1, steps)] + [0.0])
        else:
            sigmas = nodes_custom_sampler.BasicScheduler.execute(model, scheduler, steps, 1.0).args[0]
            sigmas = sigmas * (start_sigma / float(sigmas[0]))
        sampler = comfy.samplers.sampler_object(sampler_name)
        output = retry_unloaded("sampling", lambda: nodes_custom_sampler.SamplerCustom.execute(model, True, seed, cfg, positive, negative, sampler, sigmas, latent).args[0])
    settings = dict(pipe["settings"].get("sampling", {}))
    settings["refine" if start_sigma is not None else "first"] = dict(seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, shift_video=shift_video, shift_audio=shift_audio, **({"start_sigma": start_sigma} if start_sigma is not None else {}))
    return evolve(pipe, section="sampling", settings=settings, artifacts={"latent": output, "positive": positive},
                  invalidate=OUTPUTS if start_sigma is not None else (*OUTPUTS, "first_pass_audio"))


class H3SamplingSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = sampling_inputs()
        inputs += [io.Int.Input("refine_steps", default=3, min=1, max=1000,
                tooltip="The refine_* inputs and start_sigma drive the second pass, which runs only when Upscale Settings is enabled."),
            io.Float.Input("refine_cfg", default=1.0, min=0, max=100),
            io.Combo.Input("refine_sampler", options=comfy.samplers.KSampler.SAMPLERS, default="er_sde"),
            io.Combo.Input("refine_scheduler", options=list(dict.fromkeys(["linear from start", *comfy.samplers.KSampler.SCHEDULERS, "beta57"])), default="linear from start"),
            io.Float.Input("start_sigma", default=0.9035, min=0.001, max=1.0)]
        return schema("SAMPLING_SETTINGS", "Sampling Settings", inputs)

    @classmethod
    def execute(cls, pipe=None, seed=0, steps=8, cfg=1.0, sampler_name="euler", scheduler="beta57",
                shift_video=0.0, shift_audio=0.0, refine_steps=3, refine_cfg=1.0, refine_sampler="er_sde",
                refine_scheduler="linear from start", start_sigma=0.9035):
        pipe = checked(ensure_pipe(pipe, family="minimax_h3"))
        first = {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler,
                     "shift_video": shift_video, "shift_audio": shift_audio}
        second = dict(first, seed=(seed + 7) % 2**64, steps=refine_steps, cfg=refine_cfg,
                      sampler_name=refine_sampler, scheduler=refine_scheduler, start_sigma=start_sigma)
        return io.NodeOutput(evolve(pipe, section="sampling", settings={"first": first, "refine": second},
            invalidate=OUTPUTS))


class H3Sample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SAMPLE", "Sample First Pass", [Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
            io.Conditioning.Input("positive", optional=True), io.Conditioning.Input("negative", optional=True),
            io.Latent.Input("latent", optional=True),
            io.Boolean.Input("pipe_settings", default=True, tooltip="On: use the Sampling Settings node when one is chained. Off: use the widgets here."),
            io.Int.Input("seed", optional=True, force_input=True,
                         tooltip="Connected seed for this call. Unconnected takes the seed from Sampling Settings."),
            io.Int.Input("steps", default=8, min=1, max=10000),
            io.Float.Input("cfg", default=1.0, min=0.0, max=100.0),
            io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
            io.Combo.Input("scheduler", options=list(dict.fromkeys([*comfy.samplers.KSampler.SCHEDULERS, "beta57"])), default="beta57"),
            io.Float.Input("shift_video", default=0.0, min=0.0, max=100),
            io.Float.Input("shift_audio", default=0.0, min=0.0, max=100)],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, model=None, positive=None, negative=None, latent=None, pipe_settings=True,
                seed=None, steps=8, cfg=1.0, sampler_name="euler", scheduler="beta57", shift_video=0.0, shift_audio=0.0):
        pipe = take(checked(ensure_pipe(pipe, model=model)), model=model, positive=positive, negative=negative, latent=latent)
        if not pipe_settings or not pipe["settings"].get("sampling"):
            seed = seed if seed is not None else pipe["settings"].get("sampling", {}).get("first", {}).get("seed", 0)
            pipe = H3SamplingSettings.execute(pipe, seed, steps, cfg, sampler_name, scheduler, shift_video, shift_audio).args[0]
        result = sample(pipe, **pipe["settings"]["sampling"]["first"])
        return io.NodeOutput(result, require(result, "latent"))


class H3Refine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("REFINE", "Refine", [Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
            io.Latent.Input("latent", optional=True), io.Boolean.Input("enabled", default=True)],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, model=None, latent=None, enabled=True):
        pipe = take(checked(ensure_pipe(pipe, model=model)), model=model, latent=latent)
        settings = pipe["settings"]["sampling"]
        second_pass = pipe["settings"].get("upscale", {}).get("enabled", False)
        result = sample(pipe, **settings["refine"]) if enabled and second_pass else pipe
        return io.NodeOutput(result, require(result, "latent"))


class H3UpscaleSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("UPSCALE_SETTINGS", "Upscale Settings", [Pipe.Input("pipe", optional=True),
            io.Combo.Input("upscale_model", options=folder_paths.get_filename_list("latent_upscale_models") or ["(no models found)"]),
            io.Boolean.Input("enabled", default=False, tooltip="On: upscale the first pass, then refine it with the refine settings from Sampling Settings."),
            io.Combo.Input("mode", options=["megapixels", "percent simplified", "percent per side"], default="megapixels",
                           tooltip="Megapixels sets the second pass area (0.5 to 1.0). Percent simplified scales the pixel area, so 200 doubles it. Percent per side scales width and height, so 200 makes four times the pixels. The aspect never changes."),
            io.Float.Input("percent", default=200.0, min=100.0, max=800.0, step=5.0),
            io.Float.Input("megapixels", default=1.0, min=0.05, max=64.0, step=0.05)])

    @classmethod
    def execute(cls, pipe=None, upscale_model="", enabled=False, mode="percent", percent=200.0, megapixels=1.0):
        pipe = checked(ensure_pipe(pipe, family="minimax_h3"))
        return io.NodeOutput(evolve(pipe, section="upscale", settings={"model": upscale_model, "enabled": enabled,
            "mode": mode, "percent": percent, "megapixels": megapixels}, invalidate=OUTPUTS))


class H3LatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("LATENT_UPSCALE", "Latent Upscale", [Pipe.Input("pipe", optional=True), io.Latent.Input("latent", optional=True),
            io.Boolean.Input("enabled", default=True)],
            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, latent=None, enabled=True):
        pipe = take(checked(ensure_pipe(pipe, family="minimax_h3")), latent=latent)
        settings = pipe["settings"].get("upscale", {})
        if not enabled or not settings.get("enabled", False):
            return io.NodeOutput(pipe, require(pipe, "latent"))
        source = require(pipe, "latent")
        # size the second pass from the latent that is actually here, never from the format
        video = source["samples"].tensors[0] if source["samples"].is_nested else source["samples"]
        height, width = video.shape[-2] * 16, video.shape[-1] * 16
        percent = settings.get("percent", 200.0) / 100.0
        if settings.get("mode") == "percent per side":
            scale = percent
        elif settings.get("mode") == "percent simplified":
            scale = percent ** 0.5
        else:
            scale = (settings["megapixels"] * 1e6 / (width * height)) ** 0.5
        width = max(64, round(width * scale / 32) * 32)
        height = max(64, round(height * scale / 32) * 32)
        # the upscaler loads its own checkpoint; give it every byte the sampler held
        free_vram()
        model = nodes_hunyuan.LatentUpscaleModelLoader.execute(settings["model"]).args[0]
        latent = retry_unloaded("upscaling", lambda: core.MiniMaxH3LatentUpscaleWithModel.execute(model, source, width, height).args[0])
        free_vram()
        video = latent["samples"].tensors[0] if latent["samples"].is_nested else latent["samples"]
        height, width = video.shape[-2] * 16, video.shape[-1] * 16
        pipe = evolve(pipe, "format", dict(pipe["settings"]["format"], width=width, height=height,
            aspect=f"{width}:{height}", orientation="portrait" if height > width else "landscape",
            megapixels=round(width * height / 1e6, 2)))
        result = evolve(pipe, artifacts={"latent": latent, "first_pass_audio": source["samples"].unbind()[1]}, invalidate=OUTPUTS)
        return io.NodeOutput(result, latent)


def decode(pipe, tile_size=256, frames_per_pass=0, output_device="auto", audio_normalize=False, audio_gain_db=0.0, audio_from="second pass", decode_with=DECODE_VAE, keep_models=False):
    checked(pipe)
    latent, vae, audio_vae = require(pipe, "latent", "vae", "audio_vae")
    # H3 Model's decode_vae is the video VAE itself when decode_vae_name is "same"
    decoder = pipe["artifacts"].get("decode_vae", vae) if decode_with == DECODE_VAE else vae
    if not keep_models:
        free_vram()
    # 256 px tiles are what the upscale VAE decodes cleanly; under pressure only the
    # frame batch shrinks, never the tile, so the seams stay consistent
    free = comfy.model_management.get_free_memory()
    if free < 2 << 30 and not keep_models:  # kept models are moved aside by the decoder's own load; an OOM still frees everything below
        frames_per_pass = frames_per_pass or 8
    attempts = [(256, frames_per_pass)]
    attempts += [(256, frames_per_pass or 8), (256, frames_per_pass or 4)]
    images = None
    for index, (tiles, per_pass) in enumerate(attempts):
        try:
            images = minimax_decode.PROMPTFORGE_MiniMaxDecode().decode(latent, decoder, tiles, per_pass, 0, output_device, True)[0]
            tile_size, frames_per_pass = tiles, per_pass
            break
        except torch.cuda.OutOfMemoryError:
            if index == len(attempts) - 1:
                raise
            free_vram()
    audio_latent = pipe["artifacts"].get("first_pass_audio") if audio_from == "first pass" else None
    audio = nodes_audio.vae_decode_audio(audio_vae, {"samples": latent["samples"].unbind()[1] if audio_latent is None else audio_latent})
    frames = pipe["settings"]["format"]["frames"]
    images = images[:frames]
    length = round(images.shape[0] / 24 * audio["sample_rate"])
    waveform = audio["waveform"][..., :length]
    if waveform.shape[-1] < length:
        waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
    if audio_normalize:
        peak = waveform.abs().amax()
        if peak > 1e-4:
            waveform = waveform * (0.708 / peak)
    if audio_gain_db:
        waveform = waveform * (10 ** (audio_gain_db / 20))
    audio = dict(audio, waveform=waveform.clamp(-1.0, 1.0))
    pipe = evolve(pipe, section="timing", settings={"fps": 24})
    return evolve(pipe, section="decode", settings={"tile_size": tile_size, "frames_per_pass": frames_per_pass,
                  "output_device": output_device, "audio_normalize": audio_normalize, "audio_gain_db": audio_gain_db, "audio_from": audio_from,
                  "decode_with": decode_with},
                  artifacts={"images": images, "audio": audio}, invalidate=("video",))


class H3Decode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("DECODE", "Decode", [Pipe.Input("pipe", optional=True), io.Vae.Input("vae", optional=True),
            io.Latent.Input("latent", optional=True), io.Int.Input("tile_size", default=256, min=0, max=4096, step=16),
            io.Int.Input("frames_per_pass", default=0, min=0, max=4096),
            io.Combo.Input("output_device", options=["auto", "gpu", "cpu"], default="auto"),
            io.Boolean.Input("audio_normalize", default=False, tooltip="Peak-normalise the decoded audio to -3 dBFS."),
            io.Float.Input("audio_gain_db", default=0.0, min=-30.0, max=30.0, step=0.5),
            io.Combo.Input("audio_from", options=["second pass", "first pass"], default="second pass", optional=True,
                           tooltip="Which pass the soundtrack comes from. First pass keeps the audio the first sampling made, before the refine pass reworked it; only differs when a second pass runs."),
            io.Combo.Input("decode_with", options=DECODE_CHOICES, default=DECODE_VAE, optional=True, tooltip=DECODE_TOOLTIP)],
            [Pipe.Output("pipe"), io.Image.Output("images"), io.Audio.Output("audio")])

    @classmethod
    def execute(cls, pipe=None, vae=None, latent=None, tile_size=256, frames_per_pass=0, output_device="auto",
                audio_normalize=False, audio_gain_db=0.0, audio_from="second pass", decode_with=DECODE_VAE):
        result = decode(take(checked(ensure_pipe(pipe, family="minimax_h3")), vae=vae, latent=latent),
                        tile_size, frames_per_pass, output_device, audio_normalize, audio_gain_db, audio_from, decode_with)
        return io.NodeOutput(result, *require(result, "images", "audio"))


def match_levels(tail, right, fade):
    """Move the new part's per-channel level and contrast onto the previous tail, fading the correction out over `fade` frames."""
    count = min(tail.shape[0], right.shape[0])
    mean_t, mean_r = tail[-count:].mean((0, 1, 2)), right[:count].mean((0, 1, 2))
    std_t, std_r = tail[-count:].std((0, 1, 2)), right[:count].std((0, 1, 2))
    gain = torch.where(std_r > 1e-3, (std_t / std_r.clamp(min=1e-3)).clamp(.8, 1.25), torch.ones_like(std_r))
    corrected = (right * gain + (mean_t - mean_r * gain)).clamp(0, 1)
    weight = torch.cat((torch.ones(count), torch.linspace(1, 0, fade + 2)[1:-1])).to(right)[:right.shape[0]]
    weight = torch.nn.functional.pad(weight, (0, right.shape[0] - weight.shape[0])).view(-1, 1, 1, 1)
    return right + (corrected - right) * weight


def deflicker(images, count):
    """A continued clip pumps in brightness for a few frames after its guide frames (measured: swings of +-8 levels within 6 frames).
    Each of the first `count` frames moves to the clip's own smooth brightness: a 15-frame median, then a 9-frame mean, so slow changes stay."""
    count = min(count, images.shape[0])
    if count < 9:
        return images
    weights = torch.tensor([.299, .587, .114], device=images.device, dtype=images.dtype)
    luma = (images[:count, ..., :3] * weights).sum(-1).mean((1, 2))
    padded = torch.nn.functional.pad(luma[None, None], (7, 7), mode="replicate")[0, 0]
    median = padded.unfold(0, 15, 1).median(-1).values
    smooth = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(median[None, None], (4, 4), mode="replicate"), 9, 1)[0, 0]
    shift = (smooth - luma).clamp(-.1, .1)
    shift[-8:] *= torch.linspace(1, 0, 8, device=shift.device, dtype=shift.dtype)
    corrected = images.clone()
    corrected[:count] = (images[:count] + shift.view(-1, 1, 1, 1)).clamp(0, 1)
    return corrected


def repair_stalls(images, count):
    """The frames a continued clip pins to the source arrive almost without motion (measured: 0.4x the usual step at frames 17 and 22), then the
    motion jumps: two stutters about 0.2 s apart. A frame that arrives with under half the surrounding motion becomes the midpoint of its neighbours."""
    count = min(count, images.shape[0])
    if count < 14:
        return images
    gray = images[:count, ..., :3].mean(-1)
    step = (gray[1:] - gray[:-1]).abs().mean((1, 2))
    local = torch.nn.functional.pad(step[None, None], (6, 6), mode="replicate")[0, 0].unfold(0, 13, 1).median(-1).values
    repaired = images.clone()
    for index in (step < .5 * local).nonzero().flatten().tolist():
        held = index + 1
        if 0 < held < count - 1:
            repaired[held] = (repaired[index] + repaired[held + 1]) / 2
    return repaired


def join(previous, current, overlap_frames=0, transition="crossfade", level=True):
    checked(previous)
    checked(current)
    left, la = require(previous, "images", "audio")
    right, ra = require(current, "images", "audio")
    if previous["settings"]["format"]["fps"] != current["settings"]["format"]["fps"]:
        raise ValueError("Join requires matching frame rates; use Frame Interpolate first.")
    if left.shape[1:] != right.shape[1:]:
        raise ValueError("Join requires matching decoded dimensions.")
    n = min(max(0, overlap_frames), left.shape[0], right.shape[0])
    kept = left.shape[0] - n
    images = frame_buffer.empty((kept + right.shape[0], *left.shape[1:]), left.dtype, left.device)
    # only the start of the new part is ever corrected or blended, so only that is held a second time
    edge = n or min(4, left.shape[0], right.shape[0])
    fade = max(2 * edge, 24)
    changed = min(right.shape[0], max(n + 36, edge + fade) if level else n)
    head = right[:changed].to(left, copy=True)
    if level:
        head = match_levels(left[-edge:], repair_stalls(deflicker(head, n + 36), n + 36), fade)
    images[:kept] = left[:kept]
    if n and transition == "crossfade":
        weight = torch.linspace(0, 1, n, device=left.device, dtype=left.dtype).view(-1, 1, 1, 1)
        images[kept:left.shape[0]] = left[kept:] * (1 - weight) + head[:n] * weight
    else:
        images[kept:left.shape[0]] = left[kept:]
    images[left.shape[0]:kept + changed] = head[n:]
    images[kept + changed:] = right[changed:]
    rate = la["sample_rate"]
    lw = la["waveform"]
    rw = ra["waveform"]
    if ra["sample_rate"] != rate:
        rw = comfy.audio.resample(rw, ra["sample_rate"], rate)
    rw = rw.to(lw)
    if lw.shape[1] != rw.shape[1]:
        channels = max(lw.shape[1], rw.shape[1])
        lw, rw = (wave.expand(-1, channels, -1) if wave.shape[1] == 1 else wave for wave in (lw, rw))
    if rw.shape[:2] != lw.shape[:2]:
        raise ValueError(f"Join requires matching audio batch and channel counts, got {tuple(lw.shape[:2])} and {tuple(rw.shape[:2])}.")
    fps = current["settings"]["format"]["fps"]
    # Align both audio tracks to their video clocks before splicing.
    def fit(wave, length):
        return torch.nn.functional.pad(wave[..., :length], (0, max(0, length - wave.shape[-1])))
    lw = fit(lw, round(left.shape[0] / fps * rate))
    rw = fit(rw, round(right.shape[0] / fps * rate))
    ns = min(round(n / fps * rate), lw.shape[-1], rw.shape[-1])
    if ns and transition == "crossfade":
        weight = torch.linspace(0, 1, ns, device=lw.device, dtype=lw.dtype)
        wave = torch.cat((lw[..., :-ns], lw[..., -ns:] * (1 - weight) + rw[..., :ns] * weight, rw[..., ns:]), -1)
    else:
        wave = torch.cat((lw, rw[..., ns:]), -1)
    wave = fit(wave, round(images.shape[0] / fps * rate))
    fmt = dict(current["settings"]["format"], frames=images.shape[0], seconds=images.shape[0] / fps,
               width=images.shape[2], height=images.shape[1])
    return evolve(current, section="format", settings=fmt, artifacts={"images": images, "audio": {"waveform": wave, "sample_rate": rate}},
                  invalidate=("video",))


class H3SourceClip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SOURCE_CLIP", "Source Clip", [Pipe.Input("pipe", optional=True)],
            [Pipe.Output("pipe"), io.Image.Output("images"), io.Audio.Output("audio")])

    @classmethod
    def execute(cls, pipe=None):
        pipe = checked(ensure_pipe(pipe, family="minimax_h3"))
        a = pipe["artifacts"]
        frames = require(pipe, "continue_video")
        frames = video_at_24(frames, a.get("reference_video_fps", a.get("continue_fps", 24)))
        audio = a.get("continue_audio")
        if audio is None:
            audio = {"waveform": torch.zeros(1, 2, round(frames.shape[0] / 24 * 48000)), "sample_rate": 48000}
        fmt = dict(pipe["settings"].get("format", {}), width=frames.shape[2], height=frames.shape[1],
                   fps=24, seconds=frames.shape[0] / 24, frames=frames.shape[0])
        result = evolve(pipe, section="format", settings=fmt, artifacts={"images": frames, "audio": audio}, invalidate=("video",))
        result = evolve(result, section="timing", settings={"fps": 24})
        return io.NodeOutput(result, frames, audio)


class H3Join(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("JOIN", "Join", [Pipe.Input("pipe", optional=True), Pipe.Input("previous", optional=True),
            io.Image.Input("images", optional=True), io.Audio.Input("audio", optional=True),
            io.Boolean.Input("use_source", default=False),
            io.Int.Input("overlap_frames", default=0, min=0, max=36000),
            io.Combo.Input("transition", options=["cut", "crossfade"], default="crossfade"),
            io.Boolean.Input("match_levels", default=True, tooltip="Match the new part's brightness, colour and contrast to the end of the previous part over the overlap, fading out afterwards.")],
            [Pipe.Output("pipe"), io.Image.Output("images"), io.Audio.Output("audio")])

    @classmethod
    def execute(cls, pipe=None, previous=None, images=None, audio=None, use_source=False, overlap_frames=0, transition="crossfade", match_levels=True):
        pipe = take(checked(ensure_pipe(pipe, family="minimax_h3")), images=images, audio=audio)
        if use_source:
            previous = H3SourceClip.execute(pipe).args[0]
        if previous is None:
            raise ValueError("Join needs a previous clip pipe or use_source enabled.")
        result = join(previous, pipe, overlap_frames, transition, match_levels)
        return io.NodeOutput(result, *require(result, "images", "audio"))


class H3PreviewSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("PREVIEW_SAVE", "Preview / Save", [Pipe.Input("pipe", optional=True), io.Image.Input("images", optional=True),
            io.Audio.Input("audio", optional=True), io.Boolean.Input("save", default=False),
            io.String.Input("filename_prefix", default="video/PromptForge_H3",
                            tooltip="Prefix for the saved file. Tokens expand per run: %date:yyyy-MM-dd%, %date:yyyy-MM-dd_HH-mm-ss%, %date:yyyy%, %date:MM%, %date:dd%, plus %family%, %model%, %mode%, %source%, %category%, %action% and %seed% from the run."),
            io.Combo.Input("encoder_preset", options=["high quality", "balanced", "small file", "lossless-ish", "fast (GPU)", "compatible"], default="high quality",
                           tooltip="Writes a coherent encoder set into the widgets below when picked. The widgets stay editable and are what actually gets used."),
            io.Combo.Input("codec", options=list(video_save.CODECS), default="h265 (cpu x265)"),
            io.Combo.Input("speed", options=list(video_save.SPEEDS), default="slow"),
            io.Combo.Input("rate_control", options=["quality", "bitrate"], default="quality"),
            io.Int.Input("quality", default=17, min=0, max=63),
            io.Int.Input("bitrate_mbit", default=40, min=1, max=999),
            io.Combo.Input("bit_depth", options=["8", "10"], default="10"),
            io.Int.Input("audio_bitrate_kbps", default=320, min=32, max=512),
            io.Float.Input("film_grain", default=0.0, min=0.0, max=0.1),
            # keep pingpong/save_metadata after film_grain: the frontend serializes them there,
            # so a saved workflow lines up with the declaration order
            io.Boolean.Input("pingpong", default=False),
            io.Boolean.Input("save_metadata", default=True),
            io.Boolean.Input("show_preview", default=True),
            io.Boolean.Input("loop_preview", default=True),
            io.Boolean.Input("autoplay_preview", default=True)],
            [Pipe.Output("pipe"), io.Video.Output("video")], output=True)

    @classmethod
    def execute(cls, pipe=None, images=None, audio=None, save=False, filename_prefix="video/PromptForge_H3",
                encoder_preset="high quality", codec="h265 (cpu x265)", speed="slow", rate_control="quality",
                quality=17, bitrate_mbit=40, bit_depth="10", pingpong=False, save_metadata=True,
                audio_bitrate_kbps=320, film_grain=0.0, show_preview=True, loop_preview=True, autoplay_preview=True):
        pipe = take(checked(ensure_pipe(pipe, family="minimax_h3")), images=images, audio=audio)
        if codec not in video_save.CODECS or speed not in video_save.SPEEDS or rate_control not in ("quality", "bitrate") or bit_depth not in ("8", "10"):
            raise ValueError("Unsupported video encoder settings.")
        bit_depth = video_save.bit_depth_for(codec, bit_depth)
        images, audio = require(pipe, "images", "audio")
        fps = pipe["settings"].get("timing", {}).get("fps", pipe["settings"]["format"]["fps"])
        root = folder_paths.get_output_directory() if save else folder_paths.get_temp_directory()
        filename_prefix = expand_tokens(filename_prefix, pipe)
        full, _, _, _, _ = folder_paths.get_save_image_path(filename_prefix, root, images.shape[2], images.shape[1])
        if os.path.commonpath((os.path.realpath(full), os.path.realpath(root))) != os.path.realpath(root):
            raise ValueError("Video filename prefix must stay within the output directory.")
        saved = video_save.PROMPTFORGE_SaveVideo().save(images, fps, filename_prefix, codec, speed,
            rate_control, quality, bitrate_mbit, bit_depth, save, pingpong, save_metadata, audio=audio,
            audio_bitrate_kbps=audio_bitrate_kbps, show_preview=show_preview, loop_preview=loop_preview,
            autoplay_preview=autoplay_preview, film_grain=film_grain)
        item = saved["ui"]["images"][0]
        path = os.path.join(root, item["subfolder"], item["filename"])
        video = InputImpl.VideoFromFile(path)
        result = evolve(pipe, section="output", settings={"save": save, "codec": codec, "speed": speed,
            "rate_control": rate_control, "quality": quality, "bitrate_mbit": bitrate_mbit, "bit_depth": bit_depth,
            "pingpong": pingpong, "save_metadata": save_metadata, "audio_bitrate_kbps": audio_bitrate_kbps,
            "film_grain": film_grain, "show_preview": show_preview, "loop_preview": loop_preview,
            "autoplay_preview": autoplay_preview, "filename_prefix": filename_prefix,
            "encoder_preset": encoder_preset}, artifacts={"video": video})
        # sent even with the preview off: the node hides the player, so the switch works without another run
        preview = ui.PreviewVideo([ui.SavedResult(item["filename"], item["subfolder"],
            io.FolderType.output if save else io.FolderType.temp)])
        return io.NodeOutput(result, video, ui=preview)


class H3FirstPassPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("FIRST_PASS_PREVIEW", "First Pass Preview", [Pipe.Input("pipe", optional=True),
            io.Boolean.Input("enabled", default=True), io.Boolean.Input("save", default=True, tooltip="Write the first pass to the output folder instead of only previewing it."),
            io.String.Input("filename_prefix", default="video/PromptForge_H3_first_pass"),
            io.Combo.Input("decode_with", options=DECODE_CHOICES, default=DECODE_VAE, optional=True, tooltip=DECODE_TOOLTIP)], output=True)

    @classmethod
    def execute(cls, pipe=None, enabled=False, save=True, filename_prefix="video/PromptForge_H3_first_pass", decode_with=DECODE_VAE):
        pipe = checked(ensure_pipe(pipe, family="minimax_h3"))
        if not enabled:
            return io.NodeOutput(pipe)
        first = pipe["artifacts"].get("sequence_first_pass")
        if first is not None:
            preview = evolve(pipe, artifacts=first["artifacts"])
        elif pipe["artifacts"].get("latent") is None:
            return io.NodeOutput(pipe)
        else:
            preview = decode(pipe, **{**pipe["settings"].get("decode", {}), "decode_with": decode_with},
                             keep_models=pipe["settings"].get("speedup", {}).get("keep_models_for_first_pass", False))
        # a look at the first pass, so it encodes fast: about 40x less encoder time than the final's x265 slow, and 8-bit h264 plays in every browser
        saved = H3PreviewSave.execute(preview, save=save, filename_prefix=filename_prefix, codec="h264 (cpu x264)", speed="fast",
                                      quality=18, bit_depth="8", audio_bitrate_kbps=192)
        return io.NodeOutput(pipe, ui=saved.ui)


NODES = [H3Model, H3Format, H3References, H3SamplingSettings,
         H3Encode, H3Sample, H3UpscaleSettings, H3LatentUpscale, H3Refine, H3Decode, H3SourceClip, H3Join, H3PreviewSave, H3FirstPassPreview,
         H3Parts, H3Continuity, H3Sequence]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
