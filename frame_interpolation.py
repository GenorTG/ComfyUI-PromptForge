"""PromptForge Frame Interpolate: RIFE / FILM frame interpolation in one node that backs off instead of crashing.

It uses ComfyUI's own interpolation models and node (no other custom node pack is needed). The checkpoint list holds
the models in `models/frame_interpolation` (RIFE 4.25+ and FILM; older RIFE .pth files such as rife49 are not supported by ComfyUI's loader). When the whole clip does not fit
in memory, the clip is interpolated in shorter overlapping runs (down to a few frames), and what is still too much is
cross-faded instead of failing.
"""
import gc

import psutil
import torch

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths
from comfy_extras.nodes_frame_interpolation import FrameInterpolate, FrameInterpolationModelLoader

BLEND = "blend (no model, instant)"
NO_CKPT = "(no frame interpolation model found)"
MIN_RUN = 4
RUN = 24


def _checkpoints():
    return {name: folder_paths.get_full_path("frame_interpolation", name) for name in folder_paths.get_filename_list("frame_interpolation")}


def _load(path):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    model = FrameInterpolationModelLoader._detect_and_load(state)
    dtype = torch.float16 if model_management.should_use_fp16(model_management.get_torch_device()) else torch.float32
    model.eval().to(dtype)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device(),
                                                fast_disk=comfy.storage.state_dict_fast_disk(state))


def _blend(frames, multiplier):
    out = [frames[:1]]
    for index in range(frames.shape[0] - 1):
        steps = torch.arange(1, multiplier + 1, dtype=frames.dtype).view(-1, 1, 1, 1) / multiplier
        out.append(torch.lerp(frames[index:index + 1], frames[index + 1:index + 2], steps))
    return torch.cat(out)


class PROMPTFORGE_FrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        names = list(_checkpoints()) or [NO_CKPT]
        preferred = next((name for name in names if "rife" in name.lower()), names[0])
        return {
            "required": {
                "frames": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8, "tooltip": "2 doubles the frame count, 3 triples it, and so on."}),
                "ckpt_name": (names + [BLEND], {"default": preferred, "tooltip": "RIFE or FILM model. blend is a plain cross-fade: instant, no model, ghosting on motion."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "🔮 PromptForge/Video"
    DESCRIPTION = "RIFE / FILM frame interpolation with ComfyUI's own models that interpolates in shorter runs, or cross-fades, instead of running out of memory."

    def interpolate(self, frames, multiplier, ckpt_name):
        count = int(frames.shape[0])
        if count < 2:
            return (frames,)
        frames = frames[..., :3]
        total = (count - 1) * multiplier + 1
        available = psutil.virtual_memory().available
        values = total * frames.shape[1] * frames.shape[2] * 3
        dtype = torch.float32 if values * 4 < 0.1 * available else torch.float16
        if values * 2 > 0.9 * available:
            raise ValueError(f"[PromptForge] the interpolated clip needs {values * 2 / 1024 ** 3:.1f} GB of RAM even in half precision and only {available / 1024 ** 3:.1f} GB is free. Use a lower multiplier or fewer frames.")
        out = torch.empty((total, *frames.shape[1:]), dtype=dtype)
        if ckpt_name == BLEND:
            out.copy_(_blend(frames, multiplier))
            return (out,)
        paths = _checkpoints()
        if ckpt_name not in paths:
            raise ValueError("[PromptForge] no frame interpolation model was found. Put a RIFE 4.25+ or FILM model in models/frame_interpolation (for example rife_v4.26.safetensors), or choose blend.")
        patcher = _load(paths[ckpt_name])

        start, span, filled = 0, min(count, RUN), 0
        pbar = comfy.utils.ProgressBar(count - 1)
        while start < count - 1:
            model_management.throw_exception_if_processing_interrupted()
            end = min(count, start + span)
            run = frames[start:end]
            try:
                result = FrameInterpolate.execute(patcher, run, multiplier).args[0]
            except model_management.OOM_EXCEPTION:
                gc.collect()
                model_management.soft_empty_cache()
                if span > MIN_RUN:
                    span = max(MIN_RUN, span // 2)
                    print(f"[PromptForge] frame interpolation ran out of memory; continuing in runs of {span} frames.")
                    continue
                print("[PromptForge] frame interpolation ran out of memory even in short runs; cross-fading the rest.")
                result = _blend(run, multiplier)
            result = result if filled == 0 else result[1:]
            out[filled:filled + len(result)] = result
            filled += len(result)
            pbar.update(end - 1 - start)
            start = end - 1
        del patcher
        gc.collect()
        model_management.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": PROMPTFORGE_FrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": "PromptForge Frame Interpolate (low memory)"}
