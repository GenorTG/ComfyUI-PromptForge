"""PromptForge Preview Override: live previews of the sampling, drawn and shown by PromptForge itself.

A model patch around ComfyUI's outer sampling call. While a sampler runs it takes the latents of every step, draws a few
(or all) of their frames with a tiny decoder from models/vae_approx (or the model's own latent-to-RGB factors), and sends
them to the node as a short animation, so a video model's progress can be watched as it happens. It works for any sampler
and for audio+video models (the video stream is unpacked from the packed latent). The Endless Video node uses the same
patch for its own stages; the `PromptForge Preview Override` node puts it in front of any other sampler.

Designed after the preview override of KJNodes (behaviour: step callback wrapper, tiny decoder choice, animated preview,
optional suppression of ComfyUI's own preview); the code here is new.
"""
import base64
import io
import logging
import queue
import threading

import torch
import torch.nn.functional as F
from PIL import Image
from server import PromptServer

import comfy.patcher_extension
import comfy.utils
import comfy.model_prefetch

from . import preview_decoder

try:
    import av
except ImportError:
    av = None

KEY = "promptforge_preview"


class _Encoder:
    """Encodes and sends previews on a worker thread; when it is busy new previews are dropped so the sampler never waits."""

    def __init__(self):
        self.jobs = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, job):
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            pass

    def _run(self):
        while (job := self.jobs.get()) is not None:
            try:
                job()
            except Exception as error:
                logging.warning(f"[PromptForge] preview could not be sent: {error}")

    def stop(self):
        self.jobs.put(None)
        self.thread.join(timeout=5)


def _mp4(frames, fps):
    """H.264 mp4 (fragmented, so it plays from a data URL) of uint8 frames [n, H, W, 3], or None without PyAV."""
    if av is None:
        return None
    height, width = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}) as container:
        stream = container.add_stream("libx264", rate=max(1, round(fps)))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "26"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame[:height, :width], format="rgb24")))
        container.mux(stream.encode())
    return buffer.getvalue()


def _encode(frames, fps, quality):
    """(mime, bytes) of the preview: an mp4 for several frames (animated WebP without PyAV), a JPEG for one."""
    if len(frames) > 1:
        try:
            data = _mp4(frames, fps)
            if data:
                return "video/mp4", data
        except Exception as error:  # PyAV raises its own error types
            logging.warning(f"[PromptForge] mp4 preview failed ({error}); using animated WebP.")
        buffer = io.BytesIO()
        pictures = [Image.fromarray(frame) for frame in frames]
        pictures[0].save(buffer, "WEBP", save_all=True, append_images=pictures[1:], duration=int(1000 / max(1, fps)), loop=0, quality=quality)
        return "image/webp", buffer.getvalue()
    buffer = io.BytesIO()
    Image.fromarray(frames[0]).save(buffer, "JPEG", quality=quality)
    return "image/jpeg", buffer.getvalue()


class PreviewOverrideWrapper:
    def __init__(self, node_id, label="", max_resolution=768, quality=80, suppress_default=True, frames=8, fps=12, tiny_vae=""):
        self.node_id, self.label = str(node_id) if node_id is not None else None, label
        self.max_resolution, self.quality, self.suppress_default = int(max_resolution), int(quality), suppress_default
        self.frames, self.fps, self.tiny_vae = int(frames), int(fps), tiny_vae

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        latent_format = executor.class_obj.model_patcher.model.latent_format
        factors = torch.tensor(latent_format.latent_rgb_factors, dtype=torch.float32) if getattr(latent_format, "latent_rgb_factors", None) is not None else None
        bias = torch.tensor(latent_format.latent_rgb_factors_bias, dtype=torch.float32) if getattr(latent_format, "latent_rgb_factors_bias", None) is not None else None
        decoder = preview_decoder.load(self.tiny_vae) if self.tiny_vae and self.tiny_vae != preview_decoder.LATENT2RGB else None
        progress = comfy.utils.ProgressBar(max(1, len(sigmas) - 1))
        encoder = _Encoder()

        def draw(step, x0, total):
            nonlocal decoder
            if getattr(x0, "is_nested", False):
                video = x0.tensors[0]
            elif x0.ndim == 3 and latent_shapes:
                video = comfy.utils.unpack_latents(x0, latent_shapes)[0]  # audio+video models sample one packed tensor
            else:
                video = x0
            if video.ndim != 5:
                return
            count = video.shape[2]
            indices = list(range(count)) if self.frames >= count else sorted({round(i * (count - 1) / max(1, self.frames - 1)) for i in range(self.frames)})
            rgb = None
            if decoder is not None and video.shape[1] == decoder.channels:
                try:
                    with comfy.model_prefetch.pause_malloc_graph():
                        rgb = decoder.frames(video, indices)
                except RuntimeError as error:
                    logging.warning(f"[PromptForge] the preview decoder failed ({error}); using latent2rgb.")
                    decoder = None
            if rgb is None:
                if factors is None:
                    return
                rgb = torch.einsum("cthw,cr->thwr", video[0][:, indices].float().cpu(), factors)
                rgb = ((rgb + bias) if bias is not None else rgb)
                rgb = ((rgb + 1) / 2).clamp(0, 1)
            scale = self.max_resolution / max(rgb.shape[1], rgb.shape[2])
            if scale != 1:
                rgb = F.interpolate(rgb.permute(0, 3, 1, 2), scale_factor=scale, mode="area" if scale < 1 else "bilinear").permute(0, 2, 3, 1)
            frames = (rgb.clamp(0, 1) * 255).byte().numpy()
            payload = {"node": self.node_id, "label": self.label, "step": step + 1, "total": total, "fps": self.fps, "w": frames.shape[2], "h": frames.shape[1]}

            def send():
                mime, data = _encode(frames, self.fps, self.quality)
                PromptServer.instance.send_sync("promptforge.preview", {**payload, "mime": mime, "image": base64.b64encode(data).decode()}, PromptServer.instance.client_id)
            encoder.submit(send)

        def new_callback(step, x0, x, total):
            if self.node_id is not None:
                try:
                    draw(step, x0, total)
                except (RuntimeError, ValueError, AttributeError) as error:
                    logging.warning(f"[PromptForge] preview failed: {error}")
            if self.suppress_default:
                progress.update_absolute(step + 1, total)
            elif callback is not None:
                callback(step, x0, x, total)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            encoder.stop()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # the decodes grow torch's pool; hand it back before the next stage


def apply(model, node_id, **options):
    """A copy of `model` that previews its sampling in the node `node_id`."""
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, PreviewOverrideWrapper(node_id, **options))
    return patched


class PROMPTFORGE_PreviewOverride:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "max_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16, "tooltip": "Longest side of the preview frames in pixels."}),
                "jpeg_quality": ("INT", {"default": 80, "min": 10, "max": 100, "tooltip": "Quality of still previews and animated WebP previews."}),
                "suppress_default_preview": ("BOOLEAN", {"default": True, "tooltip": "Switch ComfyUI's own preview off while this one runs (the progress bar stays)."}),
                "preview_frames": ("INT", {"default": 8, "min": 1, "max": 99999, "tooltip": "How many latent frames the preview shows; they play as a loop. 99999 shows every frame."}),
                "preview_fps": ("INT", {"default": 12, "min": 1, "max": 60, "tooltip": "Playback speed of the preview loop."}),
                "tiny_vae": (preview_decoder.decoder_names(), {"default": preview_decoder.default_decoder(), "tooltip": "What draws the preview: latent2rgb is instant and needs no file; a tiny decoder from models/vae_approx (taeh3 for MiniMax H3) gives real colours and detail."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Live preview of any sampler, shown in this node: a short animation of the latent frames, drawn by a tiny decoder or latent2rgb."

    def patch(self, model, max_resolution, jpeg_quality, suppress_default_preview, preview_frames, preview_fps, tiny_vae, unique_id=None):
        return (apply(model, unique_id, label="", max_resolution=max_resolution, quality=jpeg_quality, suppress_default=suppress_default_preview, frames=preview_frames, fps=preview_fps, tiny_vae=tiny_vae),)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_PreviewOverride": PROMPTFORGE_PreviewOverride}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_PreviewOverride": "PromptForge Preview Override"}
