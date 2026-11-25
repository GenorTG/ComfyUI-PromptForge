"""Optional image second stage: learned SesquiLSR latent upscale, or a plain bicubic resize."""
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require, take

Pipe = io.Custom(PIPE)
METHODS = ["SesquiLSR Wan 2.1", "SesquiLSR Flux 2", "bicubic"]
# format: (weight prefix, external latent channels, model channels)
FORMATS = {"Wan 2.1": ("upscaler_Wan21", 16, 16), "Flux 2": ("upscaler_Flux2", 128, 32)}
FAMILY_FORMAT = {"anima": "Wan 2.1", "krea2": "Wan 2.1", "moss_image": "Flux 2"}


def find_model(format):
    prefix = FORMATS[format][0]
    return next((name for name in folder_paths.get_filename_list("latent_upscale_models")
                 if name.rsplit("/", 1)[-1].startswith(prefix)), None)


def target_size(samples, scale):
    height, width = samples.shape[-2:]
    return max(1, round(height * scale)), max(1, round(width * scale))


def sesqui_upscale(samples, format, name, height, width):
    try:
        import sesqui_lsr as sesqui
    except ImportError as exc:
        raise ValueError("SesquiLSR is not installed; clone it into custom_nodes or pick bicubic.") from exc
    # Wan latents are already raw VAE latents; Flux2 needs its batch-norm adaptor
    adaptor = sesqui.make_flux2() if format == "Flux 2" else sesqui.make_identity(16)
    model = sesqui.LatentUpscaler(in_channels=FORMATS[format][2])
    model.load_state_dict(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("latent_upscale_models", name), safe_load=True))
    model.eval()
    device = comfy.model_management.get_torch_device()
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=device, offload_device=torch.device("cpu"))
    try:
        comfy.model_management.load_models_gpu([patcher], memory_required=max(4 * samples.shape[-2] * samples.shape[-1],
                                                             height * width) * 128 * 4 * 16, force_full_load=True)
        latent = adaptor.to_vae_latent(samples.to(device=device, dtype=torch.float32))
        upscaled = model(latent, adaptor.vae_target_size((height, width)))
        return adaptor.from_vae_latent(upscaled.float()).to(device=samples.device, dtype=samples.dtype)
    finally:
        patcher.detach()


def upscale_latent(pipe, latent, method, scale):
    samples = latent["samples"]
    if samples.ndim not in (4, 5):
        raise ValueError(f"Image latent upscaling expects a 4D or 5D latent, got {tuple(samples.shape)}.")
    height, width = target_size(samples, scale)
    format = None
    if method != "bicubic":
        format = "Flux 2" if method.endswith("Flux 2") else "Wan 2.1"
        prefix, channels, _ = FORMATS[format]
        expected = FAMILY_FORMAT.get(pipe["family"])
        if samples.shape[1] != channels or (expected is not None and expected != format):
            used = expected or f"{samples.shape[1]}-channel"
            raise ValueError(f"{method} needs {channels}-channel {format} latents; the {pipe['family']} family uses {used}.")
        name = find_model(format)
        if name is None:
            print(f"[PromptForge] no {prefix}* model in latent_upscale_models; using bicubic", flush=True)
            format = None
    result = dict(latent)
    if format is None:
        result["samples"] = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
    elif samples.ndim == 5:
        # Anima keeps a length-1 temporal axis; the learned upscaler works per frame
        batch, channels, frames, old_height, old_width = samples.shape
        frames_4d = samples.movedim(2, 0).reshape(batch * frames, channels, old_height, old_width)
        upscaled = sesqui_upscale(frames_4d, format, name, height, width)
        result["samples"] = upscaled.reshape(frames, batch, channels, height, width).movedim(0, 2)
    else:
        result["samples"] = sesqui_upscale(samples, format, name, height, width)
    if "noise_mask" in latent:
        mask = latent["noise_mask"]
        if mask.shape[-2:] != result["samples"].shape[-2:]:
            result["noise_mask"] = torch.nn.functional.interpolate(mask.reshape(-1, 1, *mask.shape[-2:]).float(),
                                                                  size=result["samples"].shape[-2:], mode="nearest").reshape(*mask.shape[:-2], *result["samples"].shape[-2:]).to(mask.dtype)
    return result


class ImageLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("IMAGE", "LATENT_UPSCALE", "Latent Upscale",
                            [Pipe.Input("pipe", optional=True), io.Latent.Input("latent", optional=True),
                             io.Combo.Input("method", options=METHODS, default="SesquiLSR Wan 2.1",
                                            tooltip="SesquiLSR Wan 2.1 for Anima/Krea 2, SesquiLSR Flux 2 for MOSS. bicubic always works. Without the SesquiLSR weights the method falls back to bicubic."),
                             io.Float.Input("scale", default=1.41, min=1.0, max=2.0, step=0.01,
                                            tooltip="Per-side scale. 1.41 doubles the megapixels."),
                             io.Boolean.Input("enabled", default=True)],
                            [Pipe.Output("pipe"), io.Latent.Output("latent")])

    @classmethod
    def execute(cls, pipe=None, latent=None, method="SesquiLSR Wan 2.1", scale=1.41, enabled=True):
        from .image import image_pipe, refine
        pipe = take(image_pipe(ensure_pipe(pipe)), latent=latent)
        pipe = evolve(pipe, "upscale", {"method": method, "scale": scale, "enabled": enabled}, invalidate=("images",))
        source = require(pipe, "latent")
        if not enabled:
            return io.NodeOutput(pipe, source)
        result = refine(pipe, source)
        return io.NodeOutput(result, require(result, "latent"))


NODES = [ImageLatentUpscale]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
