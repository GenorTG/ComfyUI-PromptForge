"""Fast video + audio saving.

Frames are converted to BT.709 YUV 4:2:0 on the GPU, in chunks, on a worker thread while ffmpeg encodes, and the
audio goes into the same ffmpeg run, so the video is written once. The CPU float -> uint8 conversion and
ffmpeg's swscale pass of the VHS combine node are what this avoids.
"""
import gc
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.utils
import folder_paths

# name -> (encoder, speed presets fastest..slowest, quality flag, 10-bit pixel format)
CODECS = {
    "h264 (nvenc)": ("h264_nvenc", ("p1", "p3", "p5", "p7"), "-cq", "p010le"),
    "h265 (nvenc)": ("hevc_nvenc", ("p1", "p3", "p5", "p7"), "-cq", "p010le"),
    "av1 (nvenc, RTX 40+)": ("av1_nvenc", ("p1", "p3", "p5", "p7"), "-cq", "p010le"),
    "h264 (cpu x264)": ("libx264", ("ultrafast", "veryfast", "medium", "slow"), "-crf", "yuv420p10le"),
    "h265 (cpu x265)": ("libx265", ("ultrafast", "veryfast", "medium", "slow"), "-crf", "yuv420p10le"),
    "av1 (cpu svt-av1)": ("libsvtav1", ("12", "10", "8", "5"), "-crf", "yuv420p10le"),
}
SPEEDS = ("fastest", "fast", "balanced", "slow")
# BT.709 RGB -> Y, Cb, Cr rows
BT709 = ((0.2126, 0.7152, 0.0722), (-0.1146, -0.3854, 0.5), (0.5, -0.4542, -0.0458))
CHUNK_BYTES = 256 * 1024 ** 2


def _ffmpeg():
    path = os.environ.get("VHS_FORCE_FFMPEG_PATH") or shutil.which("ffmpeg")
    if path is None:
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise ValueError("[PromptForge] ffmpeg was not found. Install ffmpeg or put it on PATH.")
    return path


def _to_yuv(frames, bit_depth, device):
    """[B,H,W,C] float images -> [B, planar yuv420 samples] uint8 (8-bit) or int16 (10-bit), BT.709 limited range."""
    x = frames.to(device=device, dtype=torch.float32)[..., :3].clamp(0, 1)
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


def _metadata_file(directory, prompt, extra_pnginfo):
    """ffmetadata file holding the workflow, so the video can be dropped back into ComfyUI like VHS videos."""
    data = dict(extra_pnginfo or {})
    if prompt is not None:
        data["prompt"] = prompt
    value = json.dumps(data)
    for char in ("\\", "=", ";", "#", "\n"):
        value = value.replace(char, "\\" + char)
    path = Path(directory) / "metadata.txt"
    path.write_text(";FFMETADATA1\ncomment=" + value + "\n", encoding="utf-8")
    return path


def _write_audio(audio, directory):
    """Raw float32 PCM for ffmpeg; returns (path, sample rate, channels)."""
    waveform = audio["waveform"]
    waveform = waveform[0] if waveform.dim() == 3 else waveform
    path = Path(directory) / "audio.f32"
    path.write_bytes(waveform.float().cpu().t().contiguous().numpy().tobytes())
    return path, int(audio["sample_rate"]), int(waveform.shape[0])


class PROMPTFORGE_SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "PromptForge/video", "tooltip": "Saved under the output folder (or temp when save_output is off)."}),
                "codec": (list(CODECS), {"default": "h264 (nvenc)", "tooltip": "nvenc encodes on the GPU (fastest). The cpu encoders are slower but work everywhere. av1 nvenc needs an RTX 40 series or newer."}),
                "speed": (list(SPEEDS), {"default": "balanced", "tooltip": "Encoder preset: fastest trades file size for time."}),
                "rate_control": (["quality", "bitrate"], {"default": "quality", "tooltip": "quality uses crf/cq, bitrate uses bitrate_mbit."}),
                "quality": ("INT", {"default": 19, "min": 0, "max": 63, "tooltip": "crf / cq: lower is better quality and a bigger file. Used when rate_control is quality."}),
                "bitrate_mbit": ("INT", {"default": 15, "min": 1, "max": 999, "tooltip": "Used when rate_control is bitrate."}),
                "bit_depth": (["8", "10"], {"default": "8", "tooltip": "10-bit needs h265/av1 on nvenc and is not supported by every player."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "Off writes to the temp folder (preview only)."}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed the workflow in the file."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_bitrate_kbps": ("INT", {"default": 192, "min": 32, "max": 512}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "PromptForge"
    DESCRIPTION = "VHS-style video save with GPU colour conversion and a single ffmpeg pass for video and audio."

    def save(self, images, frame_rate, filename_prefix, codec, speed, rate_control, quality, bitrate_mbit, bit_depth,
             save_output, pingpong, save_metadata, audio=None, audio_bitrate_kbps=192, prompt=None, extra_pnginfo=None):
        count, height, width = (int(size) for size in images.shape[:3])
        bits = int(bit_depth)
        encoder, presets, quality_flag, pix_fmt_10 = CODECS[codec]
        if bits == 10 and encoder == "h264_nvenc":
            raise ValueError("[PromptForge] h264 nvenc cannot encode 10-bit; use h265/av1 nvenc or a cpu codec.")
        order = list(range(count)) + (list(range(count - 2, 0, -1)) if pingpong else [])
        out_width, out_height = width + width % 2, height + height % 2

        root = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, root, width, height)
        os.makedirs(folder, exist_ok=True)
        filename = f"{name}_{counter:05}.mp4"
        target = Path(folder) / filename

        args = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "yuv420p" if bits == 8 else "yuv420p10le", "-s", f"{out_width}x{out_height}",
                "-color_range", "tv", "-colorspace", "bt709", "-framerate", str(frame_rate), "-i", "-"]
        scratch = tempfile.mkdtemp(prefix="promptforge_video_")
        try:
            inputs = 1
            if audio is not None:
                audio_path, sample_rate, channels = _write_audio(audio, scratch)
                args += ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", str(audio_path)]
                inputs += 1
            if save_metadata:
                args += ["-i", str(_metadata_file(scratch, prompt, extra_pnginfo))]
                inputs += 1
            args += ["-map", "0:v"]
            if audio is not None:
                args += ["-map", "1:a", "-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k", "-t", f"{len(order) / frame_rate:.6f}"]
            if save_metadata:
                args += ["-map_metadata", str(inputs - 1)]
            args += ["-c:v", encoder, "-pix_fmt", "yuv420p" if bits == 8 else pix_fmt_10,
                     "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
            preset = presets[SPEEDS.index(speed)]
            args += ["-preset", preset]
            if rate_control == "bitrate":
                args += ["-b:v", f"{bitrate_mbit}M"]
            elif encoder.endswith("_nvenc"):
                args += ["-rc", "vbr", quality_flag, str(quality), "-b:v", "0"]
            else:
                args += [quality_flag, str(quality)]
            if encoder == "libx265":
                args += ["-x265-params", "log-level=error"]
            args += ["-movflags", "+faststart", str(target)]

            self._encode(args, images, order, bits, len(order))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        preview = {"filename": filename, "subfolder": subfolder, "type": "output" if save_output else "temp"}
        return {"ui": {"images": [preview], "animated": (True,)}}

    def _encode(self, args, images, order, bits, total):
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
                        data = _to_yuv(batch, bits, device)
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

        errors = tempfile.TemporaryFile()
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=errors)
        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        progress = comfy.utils.ProgressBar(total)
        failed = False
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
                try:
                    process.stdin.write(memoryview(item.numpy()))
                except (BrokenPipeError, OSError):
                    failed = True
                    break
                progress.update(len(item))
        except BaseException:
            process.kill()
            raise
        finally:
            stop.set()
            worker.join()
            try:
                process.stdin.close()
            except OSError:
                pass
            code = process.wait()
            errors.seek(0)
            message = errors.read().decode(errors="replace").strip()
            errors.close()
        if failed or code != 0:
            raise ValueError(f"[PromptForge] ffmpeg failed: {message[-800:] or f'exit code {code}'}")


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_SaveVideo": PROMPTFORGE_SaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_SaveVideo": "PromptForge Save Video (fast)"}
