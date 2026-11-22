"""Pipe-aware reference loading and time-preserving video postprocessing."""
from pathlib import Path

import gc

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np

import comfy.model_management as mm
import comfy.utils
import folder_paths
import node_helpers
from aiohttp import web
from server import PromptServer
from comfy_api.latest import io, InputImpl
from comfy_extras import nodes_audio

from ..frame_interpolation import PROMPTFORGE_FrameInterpolate, _checkpoints, BLEND
from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require
from . import rtx

Pipe = io.Custom(PIPE)
CATEGORY = "PromptForge V2/References"
INVALIDATE = ("text", "writer", "positive", "negative", "latent", "images", "audio")


@PromptServer.instance.routes.get("/promptforge/v2/media")
async def media_files(request):
    root = Path(folder_paths.get_input_directory())
    extensions = {"image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"},
                  "video": {".mp4", ".mkv", ".webm", ".mov", ".avi"},
                  "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".webm"}}
    kind = request.query.get("kind", "image")
    if kind not in extensions:
        raise web.HTTPBadRequest(text="Unknown media kind")
    files = [path.relative_to(root).as_posix() for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in extensions[kind] and root.resolve() in path.resolve().parents]
    return web.json_response({"files": sorted(files)})


def input_path(filename):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Select an existing file inside ComfyUI's input directory.")
    return path


def changed(filename, enabled=True, **kwargs):
    if not enabled or not filename:
        return "disabled"
    path = input_path(filename)
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


class LoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("LOAD", "IMAGE", "Load Reference Image", inputs=[Pipe.Input("pipe", optional=True), io.String.Input("filename", default=""),
                                 io.Combo.Input("role", options=["reference", "first_frame", "last_frame"]),
                                 io.Int.Input("index", default=1, min=1, max=1024), io.Boolean.Input("enabled", default=True),
                                 io.String.Input("description", default="", multiline=True)],
                         outputs=[Pipe.Output(), io.Image.Output(), io.Mask.Output()])

    @classmethod
    def fingerprint_inputs(cls, filename, enabled=True, **kwargs):
        return changed(filename, enabled)

    @classmethod
    def execute(cls, pipe=None, filename=None, role="reference", index=1, enabled=True, description=""):
        pipe = ensure_pipe(pipe)
        if not enabled:
            return io.NodeOutput(pipe, None, None)
        if not filename:
            raise ValueError("Choose an image first.")
        with node_helpers.pillow(Image.open, str(input_path(filename))) as opened:
            frame = node_helpers.pillow(ImageOps.exif_transpose, opened)
            image = torch.from_numpy(np.array(frame.convert("RGB"), dtype=np.float32) / 255.0)[None]
            mask = 1.0 - torch.from_numpy(np.array(frame.getchannel("A"), dtype=np.float32) / 255.0)[None] if "A" in frame.getbands() else torch.zeros(image.shape[:3])
        artifacts = {"source_image": image, "source_mask": mask}
        refs = dict(pipe["settings"].get("media", {}))
        refs[f"{role}:{index}"] = {"filename": filename, "description": description}
        if role == "reference":
            bank = dict(pipe["artifacts"].get("reference_image_bank", {}))
            bank[index] = image
            artifacts.update(reference_image_bank=bank, reference_images=[bank[k] for k in sorted(bank)])
        else:
            artifacts[role] = image
        return io.NodeOutput(evolve(pipe, "media", refs, artifacts, INVALIDATE), image, mask)


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("LOAD", "VIDEO", "Load Reference Video", inputs=[Pipe.Input("pipe", optional=True), io.String.Input("filename", default=""),
                                 io.Combo.Input("role", options=["reference", "continue"]),
                                 io.Float.Input("start_seconds", default=0.0, min=0.0, max=86400),
                                 io.Float.Input("duration", default=0.0, min=0.0, max=86400, tooltip="0 loads the remaining clip."),
                                 io.Boolean.Input("enabled", default=True), io.String.Input("description", default="", multiline=True)],
                         outputs=[Pipe.Output(), io.Image.Output("frames"), io.Audio.Output(), io.Float.Output("fps")])

    @classmethod
    def fingerprint_inputs(cls, filename, enabled=True, **kwargs):
        return changed(filename, enabled)

    @classmethod
    def execute(cls, pipe=None, filename=None, role="reference", start_seconds=0.0, duration=0.0, enabled=True, description=""):
        pipe = ensure_pipe(pipe)
        if not enabled:
            return io.NodeOutput(pipe, None, None, 0.0)
        if not filename:
            raise ValueError("Choose a video first.")
        components = InputImpl.VideoFromFile(str(input_path(filename))).get_components()
        frames, fps, audio = components.images, float(components.frame_rate), components.audio
        start = round(start_seconds * fps)
        end = min(len(frames), start + round(duration * fps)) if duration > 0 else len(frames)
        if start >= end:
            raise ValueError("The selected video interval contains no frames.")
        frames = frames[start:end].clone()
        if audio is not None:
            rate = audio["sample_rate"]
            audio = {**audio, "waveform": audio["waveform"][..., round(start / fps * rate):round(end / fps * rate)].clone()}
        settings = dict(pipe["settings"].get("media", {}))
        settings["video"] = {"filename": filename, "role": role, "start_seconds": start_seconds, "duration": duration, "description": description, "fps": fps}
        artifacts = {"reference_video": frames, "reference_video_fps": fps, "reference_video_audio": audio}
        if role == "continue":
            artifacts.update(continue_video=frames, continue_audio=audio, continue_fps=fps)
        return io.NodeOutput(evolve(pipe, "media", settings, artifacts, INVALIDATE), frames, audio, fps)


class LoadAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("LOAD", "AUDIO", "Load Reference Audio", inputs=[Pipe.Input("pipe", optional=True), io.String.Input("filename", default=""), io.Boolean.Input("enabled", default=True),
                                 io.String.Input("description", default="", multiline=True)], outputs=[Pipe.Output(), io.Audio.Output()])

    @classmethod
    def fingerprint_inputs(cls, filename, enabled=True, **kwargs):
        return changed(filename, enabled)

    @classmethod
    def execute(cls, pipe=None, filename=None, enabled=True, description=""):
        pipe = ensure_pipe(pipe)
        if not enabled:
            return io.NodeOutput(pipe, None)
        if not filename:
            raise ValueError("Choose an audio file first.")
        waveform, rate = nodes_audio.load(str(input_path(filename)))
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": rate}
        settings = dict(pipe["settings"].get("media", {}))
        settings["audio"] = {"filename": filename, "description": description}
        return io.NodeOutput(evolve(pipe, "media", settings, {"reference_audio": audio}, INVALIDATE), audio)


class VideoCleanup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("VIDEO", "CLEANUP", "Video Cleanup", description="Optional same-size NVIDIA RTX passes for the finished clip: deblur restores skin and contact detail, denoise removes decode grain. Run it after interpolation.",
                         inputs=[Pipe.Input("pipe", optional=True), io.Image.Input("images", optional=True),
                                 io.Boolean.Input("enabled", default=False),
                                 io.Boolean.Input("denoise", default=False, tooltip="Remove decode grain and compression noise."),
                                 io.Combo.Input("denoise_quality", options=rtx.QUALITIES, default="Ultra"),
                                 io.Boolean.Input("deblur", default=False, tooltip="Restore skin, fabric and contact detail."),
                                 io.Combo.Input("deblur_quality", options=rtx.QUALITIES, default="Ultra")],
                         outputs=[Pipe.Output(), io.Image.Output("frames")])

    @classmethod
    def execute(cls, pipe=None, images=None, enabled=False, denoise=False, denoise_quality="Ultra",
                deblur=False, deblur_quality="Ultra"):
        pipe = ensure_pipe(pipe)
        if images is not None:
            pipe = evolve(pipe, artifacts={"images": images})
        frames = require(pipe, "images")
        if not enabled or not (denoise or deblur):
            return io.NodeOutput(pipe, frames)
        cleaned = rtx.clean(frames, denoise, denoise_quality, deblur, deblur_quality)
        return io.NodeOutput(evolve(pipe, artifacts={"images": cleaned}), cleaned)


class FrameInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("FRAME", "INTERPOLATE", "Frame Interpolate", inputs=[Pipe.Input("pipe", optional=True), io.Boolean.Input("enabled", default=False),
                                 io.Int.Input("multiplier", default=2, min=2, max=8),
                                 io.Combo.Input("model_name", options=list(_checkpoints()) + [BLEND]),
                                 io.Int.Input("batch_size", default=0, min=0, max=240),
                                 io.Int.Input("overlap", default=2, min=1, max=12),
                                 io.Float.Input("scale", default=1.0, min=0.25, max=1.0, step=0.05)],
                         outputs=[Pipe.Output(), io.Image.Output("frames"), io.Float.Output("fps")])

    @classmethod
    def execute(cls, pipe=None, enabled=False, multiplier=2, model_name=BLEND, batch_size=0, overlap=2, scale=1.0):
        pipe = ensure_pipe(pipe)
        frames = require(pipe, "images")
        fmt = pipe["settings"].get("format", {})
        timing = pipe["settings"].get("timing", {})
        fps = float(timing.get("fps", fmt.get("fps", 24)))
        if not enabled:
            return io.NodeOutput(pipe, frames, fps)
        count = len(frames)
        if count < 2:
            return io.NodeOutput(pipe, frames, fps)
        interpolator = PROMPTFORGE_FrameInterpolate()
        try:
            result = interpolator.interpolate(frames, multiplier, model_name, batch_size, overlap, scale)[0]
        except ValueError as exc:
            if "GB of RAM" not in str(exc):
                raise
            # the estimate runs before any batching; give it the RAM the models and reachable input still hold
            mm.free_registrations(1 << 62)
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            result = interpolator.interpolate(frames, multiplier, model_name, batch_size, overlap, scale)[0]
        # The core interpolator produces (N-1)*M+1 frames. Extend the final hold
        # to N*M so frame-based playback duration and audio length stay unchanged.
        target = count * multiplier
        if len(result) < target:
            result = torch.cat((result, result[-1:].expand(target - len(result), -1, -1, -1)), dim=0)
        fps *= multiplier
        out = evolve(pipe, "timing", {"fps": fps, "native_fps": timing.get("native_fps", fmt.get("fps", 24)),
                                     "frames": len(result), "duration": len(result) / fps}, {"images": result})
        return io.NodeOutput(out, result, fps)


class ImageScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("IMAGE", "SCALE", "Image Scale", inputs=[io.Image.Input("image"),
                                 Pipe.Input("pipe", optional=True), io.Boolean.Input("use_format", default=False),
                                 io.Int.Input("width", default=1024, min=16, max=16384, step=16),
                                 io.Int.Input("height", default=1024, min=16, max=16384, step=16),
                                 io.Combo.Input("crop", options=["disabled", "center"])], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, image, width, height, crop="disabled", pipe=None, use_format=False):
        if use_format:
            pipe = ensure_pipe(pipe)
            fmt = (pipe or {}).get("settings", {}).get("format")
            if fmt is None:
                raise ValueError("Image Scale needs a pipe with a format when use_format is on.")
            width, height = fmt["width"], fmt["height"]
        return io.NodeOutput(comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bicubic", crop).movedim(1, -1))


class MaskPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("MASK", "PREPARE", "Mask Prepare", inputs=[io.Mask.Input("mask"), io.Boolean.Input("invert", default=False),
                                 io.Int.Input("grow", default=0, min=-128, max=128),
                                 io.Int.Input("feather", default=0, min=0, max=128)], outputs=[io.Mask.Output()])

    @classmethod
    def execute(cls, mask, invert=False, grow=0, feather=0):
        value = (1 - mask if invert else mask).unsqueeze(1)
        if grow:
            radius = abs(grow)
            value = F.max_pool2d(value if grow > 0 else -value, 2 * radius + 1, stride=1, padding=radius)
            if grow < 0:
                value = -value
        if feather:
            value = F.avg_pool2d(F.pad(value, (feather,) * 4, mode="replicate"), 2 * feather + 1, stride=1)
        return io.NodeOutput(value.squeeze(1).clamp(0, 1))


class ImageComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("IMAGE", "COMPOSITE", "Image Composite", inputs=[io.Image.Input("destination"), io.Image.Input("source"), io.Mask.Input("mask")], outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, destination, source, mask):
        h, w = destination.shape[1:3]
        source = comfy.utils.common_upscale(source.movedim(-1, 1), w, h, "bicubic", "disabled").movedim(1, -1)
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1).clamp(0, 1)
        return io.NodeOutput(torch.lerp(destination, source, mask))


class ImageResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("IMAGE", "RESULT", "Image Result", description="Set the final image after compositing or other standard IMAGE processing.",
                         inputs=[Pipe.Input("pipe", optional=True), io.Image.Input("images")],
                         outputs=[Pipe.Output(), io.Image.Output("images")])

    @classmethod
    def execute(cls, pipe=None, images=None):
        pipe = ensure_pipe(pipe)
        return io.NodeOutput(evolve(pipe, artifacts={"images": images}), images)


NODES = [LoadImage, LoadVideo, LoadAudio, VideoCleanup, FrameInterpolate, ImageScale, MaskPrepare, ImageComposite, ImageResult]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
