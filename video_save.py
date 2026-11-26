"""Fast video + audio saving.

Frames are converted to BT.709 YUV 4:2:0 on the GPU, in chunks, on a worker thread while the encoder runs (PyAV,
which ComfyUI ships), and the audio is muxed in the same pass, so the video is written once. The CPU float -> uint8
conversion and the swscale pass of the VHS combine node are what this avoids.
"""
import gc
import json
import os
import queue
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.utils
import folder_paths

# name -> (encoder, speed presets fastest..slowest, quality option, 10-bit pixel format)
CODECS = {
    "h264 (nvenc)": ("h264_nvenc", ("p1", "p3", "p5", "p7"), "cq", "p010le"),
    "h265 (nvenc)": ("hevc_nvenc", ("p1", "p3", "p5", "p7"), "cq", "p010le"),
    "av1 (nvenc, RTX 40+)": ("av1_nvenc", ("p1", "p3", "p5", "p7"), "cq", "p010le"),
    "h264 (cpu x264)": ("libx264", ("ultrafast", "veryfast", "medium", "slow"), "crf", "yuv420p10le"),
    "h265 (cpu x265)": ("libx265", ("ultrafast", "veryfast", "medium", "slow"), "crf", "yuv420p10le"),
    "av1 (cpu svt-av1)": ("libsvtav1", ("12", "10", "8", "5"), "crf", "yuv420p10le"),
}
SPEEDS = ("fastest", "fast", "balanced", "slow")
# BT.709 RGB -> Y, Cb, Cr rows
BT709 = ((0.2126, 0.7152, 0.0722), (-0.1146, -0.3854, 0.5), (0.5, -0.4542, -0.0458))
CHUNK_BYTES = 256 * 1024 ** 2
# h264 10-bit is refused by most GPUs and players, so that pair is never offered
EIGHT_BIT_ONLY = "h264 (nvenc)"
AUDIO_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}
BT709_TAG, LIMITED_RANGE = 1, 1  # ffmpeg's numbers for BT.709 matrix / primaries / transfer and for tv range


def bit_depth_for(codec, bit_depth):
    """The bit depth the codec can be saved with; an unsupported choice falls back to 8-bit instead of failing the run."""
    if codec == EIGHT_BIT_ONLY and bit_depth == "10":
        print(f"[PromptForge] Save Video: {codec} has no 10-bit, saving 8-bit.")
        return "8"
    return bit_depth


def _to_yuv(frames, bit_depth, device, grain=0.0):
    """[B,H,W,C] float images -> [B, planar yuv420 samples] uint8 (8-bit) or int16 (10-bit), BT.709 limited range."""
    x = frames.to(device=device, dtype=torch.float32)[..., :3].clamp(0, 1)
    if grain:
        x = (x + torch.randn_like(x) * grain).clamp(0, 1)
    if x.shape[1] % 2:
        x = torch.cat([x, x[:, -1:]], 1)
    if x.shape[2] % 2:
        x = torch.cat([x, x[:, :, -1:]], 2)
    matrix = torch.tensor(BT709, device=device)
    k = 1 if bit_depth == 8 else 4
    dtype = torch.uint8 if bit_depth == 8 else torch.int16
    luma = (x @ matrix[0] * 219 * k + 16 * k + 0.5).clamp(16 * k, 235 * k).to(dtype)
    chroma = F.avg_pool2d(torch.stack([x @ matrix[1], x @ matrix[2]], 1), 2) * 224 * k + 128 * k + 0.5
    chroma = chroma.clamp(16 * k, 240 * k).to(dtype)
    return torch.cat([luma.flatten(1), chroma[:, 0].flatten(1), chroma[:, 1].flatten(1)], 1).cpu()


def _fill(frame, data):
    """Copy one planar yuv420 sample row into the frame's planes, which may be padded to a wider line size."""
    offset = 0
    for plane in frame.planes:
        samples = np.frombuffer(plane, dtype=data.dtype).reshape(plane.height, -1)
        size = plane.height * plane.width
        samples[:, :plane.width] = data[offset:offset + size].reshape(plane.height, plane.width)
        offset += size


def _audio_stream(container, audio, seconds, bitrate_kbps):
    """The AAC stream and its frames, cut to the video's length."""
    waveform = audio["waveform"]
    waveform = waveform[0] if waveform.dim() == 3 else waveform
    rate = int(audio["sample_rate"])
    waveform = waveform[:, :round(seconds * rate)].float().cpu().contiguous()
    layout = AUDIO_LAYOUTS.get(waveform.shape[0], f"{waveform.shape[0]}c")
    stream = container.add_stream("aac", rate=rate, layout=layout)
    stream.bit_rate = bitrate_kbps * 1000

    def frames():
        for start in range(0, waveform.shape[1], rate):
            frame = av.AudioFrame.from_ndarray(waveform[:, start:start + rate].numpy().copy(order="C"), format="fltp", layout=layout)
            frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
            yield frame
    return stream, frames()


class PROMPTFORGE_SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames. An empty input (for example the first pass when it is not returned) saves nothing."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "PromptForge/video", "tooltip": "Saved under the output folder (or temp when save_output is off)."}),
                "codec": (list(CODECS), {"default": "h264 (nvenc)", "tooltip": "nvenc encodes on the GPU (fastest). The cpu encoders are slower but work everywhere. av1 nvenc needs an RTX 40 series or newer."}),
                "speed": (list(SPEEDS), {"default": "slow", "tooltip": "Encoder preset: fastest trades file size for time."}),
                "rate_control": (["quality", "bitrate"], {"default": "quality", "tooltip": "quality uses crf/cq, bitrate uses bitrate_mbit."}),
                "quality": ("INT", {"default": 19, "min": 0, "max": 63, "tooltip": "crf / cq: lower is better quality and a bigger file. Used when rate_control is quality."}),
                "bitrate_mbit": ("INT", {"default": 15, "min": 1, "max": 999, "tooltip": "Used when rate_control is bitrate."}),
                "bit_depth": (["8", "10"], {"default": "8", "tooltip": "10-bit is not offered for h264 nvenc and is not supported by every player."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "Off writes to the temp folder (preview only)."}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed the workflow in the file."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_bitrate_kbps": ("INT", {"default": 192, "min": 32, "max": 512}),
                "show_preview": ("BOOLEAN", {"default": True, "tooltip": "Show the video inside the node after it is saved."}),
                "loop_preview": ("BOOLEAN", {"default": True, "tooltip": "Loop the preview. Takes effect straight away, no rerun needed."}),
                "autoplay_preview": ("BOOLEAN", {"default": True, "tooltip": "Start the preview playing when a video finishes. Off leaves it paused until you press play."}),
                "film_grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Fine grain added while encoding; hides the plastic look of AI video. 0 = off, 0.005 to 0.01 is subtle."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "🔮 PromptForge/Video"
    DESCRIPTION = "VHS-style video save with GPU colour conversion and a single encoding pass for video and audio."

    def save(self, images, frame_rate, filename_prefix, codec, speed, rate_control, quality, bitrate_mbit, bit_depth,
             save_output, pingpong, save_metadata, audio=None, audio_bitrate_kbps=192, show_preview=True, loop_preview=True, autoplay_preview=True, film_grain=0.0, prompt=None, extra_pnginfo=None):
        if images is None:
            print("[PromptForge] Save Video: nothing to save, its input is empty.")
            return {"ui": {}}
        count, height, width = (int(size) for size in images.shape[:3])
        if speed != "slow":
            print(f"[PromptForge] quality: the {speed} encoder preset bands gradients more than slow.")
        bits = int(bit_depth_for(codec, bit_depth))
        encoder, presets, quality_flag, pix_fmt_10 = CODECS[codec]
        order = list(range(count)) + (list(range(count - 2, 0, -1)) if pingpong else [])
        out_width, out_height = width + width % 2, height + height % 2

        root = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, root, width, height)
        os.makedirs(folder, exist_ok=True)
        filename = f"{name}_{counter:05}.mp4"
        target = Path(folder) / filename

        pix_fmt = "yuv420p" if bits == 8 else pix_fmt_10
        options = {"preset": presets[SPEEDS.index(speed)]}
        if rate_control == "quality":
            options[quality_flag] = str(quality)
            if encoder.endswith("_nvenc"):
                options["rc"] = "vbr"
        if encoder == "libx265":
            options["x265-params"] = "log-level=error"
        rate = Fraction(frame_rate).limit_denominator(100000)
        try:
            with av.open(str(target), "w", format="mp4", options={"movflags": "faststart"}) as container:
                if save_metadata:
                    data = dict(extra_pnginfo or {})
                    if prompt is not None:
                        data["prompt"] = prompt
                    container.metadata["comment"] = json.dumps(data)  # where ComfyUI looks, so the video can be dropped back in
                stream = container.add_stream(encoder, rate=rate, options=options)
                stream.width, stream.height, stream.pix_fmt = out_width, out_height, pix_fmt
                context = stream.codec_context
                context.bit_rate = bitrate_mbit * 1_000_000 if rate_control == "bitrate" else 0
                context.color_range, context.colorspace = LIMITED_RANGE, BT709_TAG
                context.color_primaries, context.color_trc = BT709_TAG, BT709_TAG
                context.thread_type = "AUTO"
                sound = _audio_stream(container, audio, len(order) / frame_rate, audio_bitrate_kbps) if audio is not None else None
                self._encode(container, stream, sound, images, order, bits, film_grain)
        except av.FFmpegError as exc:
            target.unlink(missing_ok=True)
            raise ValueError(f"[PromptForge] video encoding failed ({encoder}): {exc}") from exc
        except BaseException:
            target.unlink(missing_ok=True)
            raise

        preview = {"filename": filename, "subfolder": subfolder, "type": "output" if save_output else "temp"}
        info = f"{out_width}x{out_height} - {frame_rate:g} fps - {len(order)} frames ({len(order) / frame_rate:.1f} s) - {codec}, {bits}-bit - {'with sound' if audio is not None else 'no sound'} - {target.stat().st_size / 1024 ** 2:.1f} MB"
        return {"ui": {"images": [preview], "animated": (True,), "promptforge_info": [info]}}

    def _encode(self, container, stream, sound, images, order, bits, grain):
        total = len(order)
        raw_format = "yuv420p" if bits == 8 else "yuv420p10le"
        device = model_management.get_torch_device()
        height, width = int(images.shape[1]), int(images.shape[2])
        chunk = max(1, min(32, CHUNK_BYTES // (height * width * 12)))
        ready = queue.Queue(maxsize=3)
        stop = threading.Event()

        def produce():
            nonlocal chunk, device
            try:
                position = 0
                while position < total and not stop.is_set():
                    try:
                        batch = images[order[position:position + chunk]]
                        data = _to_yuv(batch, bits, device, grain)
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if not (isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()):
                            raise
                        gc.collect()
                        model_management.soft_empty_cache()
                        if chunk > 1:
                            chunk //= 2
                        elif device.type != "cpu":
                            device = torch.device("cpu")
                        else:
                            raise
                        continue
                    position += len(batch)
                    while not stop.is_set():
                        try:
                            ready.put(data, timeout=0.5)
                            break
                        except queue.Full:
                            pass
                ready.put(None)
            except BaseException as exc:
                ready.put(exc)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        progress = comfy.utils.ProgressBar(total)
        audio_stream, audio_frames = sound or (None, iter(()))
        pending = next(audio_frames, None)
        position = 0
        try:
            while True:
                model_management.throw_exception_if_processing_interrupted()
                try:
                    item = ready.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                for data in item.numpy().view("uint8" if bits == 8 else "uint16"):
                    frame = av.VideoFrame(stream.width, stream.height, raw_format)
                    _fill(frame, data)
                    frame.pts, frame.time_base = position, 1 / stream.codec_context.framerate
                    frame.colorspace, frame.color_range = BT709_TAG, LIMITED_RANGE
                    container.mux(stream.encode(frame))
                    position += 1
                    # sound is written alongside the picture so the file stays interleaved
                    while pending is not None and pending.pts * pending.time_base <= position / stream.codec_context.framerate:
                        container.mux(audio_stream.encode(pending))
                        pending = next(audio_frames, None)
                progress.update(len(item))
            container.mux(stream.encode())
            if audio_stream is not None:
                while pending is not None:
                    container.mux(audio_stream.encode(pending))
                    pending = next(audio_frames, None)
                container.mux(audio_stream.encode())
        finally:
            stop.set()
            worker.join()


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_SaveVideo": PROMPTFORGE_SaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_SaveVideo": "PromptForge Save Video (fast)"}
