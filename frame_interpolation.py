"""Low-memory RIFE frame interpolation that backs off by itself instead of crashing ComfyUI.

Reuses the IFNet architecture and checkpoints of the comfyui-frame-interpolation pack. Frames stay on the
CPU, one pair at a time goes to the GPU, and on an out-of-memory error the node walks down a ladder:
free ComfyUI's loaded models, then estimate flow at 1/2 and 1/4 resolution, then run on the CPU.
"""
import gc
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.utils
import folder_paths

ARCH_VERSIONS = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
}
NO_CKPT = "(no RIFE checkpoint found)"
# flow-estimation scale per rung; the final image is always warped at full resolution
GPU_SCALES = (1.0, 0.5, 0.25)
CPU_SCALE = 0.5

_arch_module = None


def _vfi_pack_dir():
    for path in Path(__file__).resolve().parent.parent.iterdir():
        if "frame-interpolation" in path.name.lower() and (path / "vfi_models" / "rife" / "rife_arch.py").is_file():
            return path
    return None


def _checkpoint_dirs():
    dirs = [Path(folder_paths.models_dir) / "frame_interpolation"]
    pack = _vfi_pack_dir()
    if pack is not None:
        dirs.append(pack / "ckpts" / "rife")
    return dirs


def _checkpoints():
    found = {}
    for directory in _checkpoint_dirs():
        for name in ARCH_VERSIONS:
            if (directory / name).is_file():
                found.setdefault(name, directory / name)
    return found


def _rife_arch():
    global _arch_module
    if _arch_module is None:
        pack = _vfi_pack_dir()
        if pack is None:
            raise ValueError("[PromptForge] Frame Interpolate needs the comfyui-frame-interpolation pack (it provides the RIFE network). Install it from the ComfyUI Manager.")
        spec = importlib.util.spec_from_file_location("promptforge_rife_arch", pack / "vfi_models" / "rife" / "rife_arch.py")
        _arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_arch_module)
    return _arch_module


def _is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _scale_list(arch_ver, scale):
    base = (16, 8, 4, 2, 1) if arch_ver == "4.26" else (8, 4, 2, 1)
    return [step / scale for step in base]


class PROMPTFORGE_FrameInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        names = list(_checkpoints()) or [NO_CKPT]
        return {
            "required": {
                "frames": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8, "tooltip": "2 doubles the frame count, 3 triples it, and so on."}),
                "ckpt_name": (names, {"default": "rife49.pth" if "rife49.pth" in names else names[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "PromptForge"
    DESCRIPTION = "RIFE frame interpolation that picks its own memory settings: slower but does not run out of VRAM or RAM on large frames."

    def interpolate(self, frames, multiplier, ckpt_name):
        path = _checkpoints().get(ckpt_name)
        if path is None:
            raise ValueError(f"[PromptForge] RIFE checkpoint '{ckpt_name}' not found. Put rife49.pth in models/frame_interpolation or the frame-interpolation pack's ckpts/rife folder.")
        count = int(frames.shape[0])
        if count < 2:
            return (frames,)
        frames = frames[..., :3]

        arch = _rife_arch()
        arch_ver = ARCH_VERSIONS[ckpt_name]
        network = arch.IFNet(arch_ver=arch_ver)
        network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        network.eval()

        height, width = int(frames.shape[1]), int(frames.shape[2])
        output = torch.empty((count - 1) * multiplier + 1, height, width, 3, dtype=torch.float32)
        state = {"rung": 0, "freed": False, "cpu": False}
        gpu = model_management.get_torch_device()
        if gpu.type == "cpu":
            state["rung"] = len(GPU_SCALES)

        def place():
            """Put the network where the current rung runs, and keep the module-level warp grid on the same device."""
            on_cpu = state["rung"] >= len(GPU_SCALES)
            device = torch.device("cpu") if on_cpu else gpu
            dtype = torch.float32 if on_cpu else torch.float16
            arch.device = device
            arch.backwarp_tenGrid.clear()
            network.to(device=device, dtype=dtype)
            return device, dtype, CPU_SCALE if on_cpu else GPU_SCALES[state["rung"]]

        def back_off(exc):
            gc.collect()
            model_management.soft_empty_cache()
            if not state["freed"] and state["rung"] < len(GPU_SCALES):
                state["freed"] = True
                print("[PromptForge] RIFE ran out of VRAM; unloading ComfyUI's models and retrying.")
                model_management.free_memory(10 ** 30, gpu)
            elif state["rung"] < len(GPU_SCALES):
                state["rung"] += 1
                where = f"flow at 1/{round(1 / GPU_SCALES[state['rung']])} resolution" if state["rung"] < len(GPU_SCALES) else "the CPU (slow)"
                print(f"[PromptForge] RIFE still out of memory; continuing with {where}.")
            else:
                raise exc

        device, dtype, scale = place()
        progress = comfy.utils.ProgressBar(count - 1)
        print(f"[PromptForge] RIFE {ckpt_name}: {count} frames of {width}x{height} -> {output.shape[0]}.")
        try:
            with torch.inference_mode():
                for index in range(count - 1):
                    model_management.throw_exception_if_processing_interrupted()
                    output[index * multiplier] = frames[index].float().cpu()
                    for step in range(1, multiplier):
                        while True:
                            try:
                                # the flow blocks need sizes divisible by 128 / scale
                                multiple = max(128, int(128 / scale))
                                padding = (0, -width % multiple, 0, -height % multiple)
                                first = F.pad(frames[index:index + 1].permute(0, 3, 1, 2).to(device, dtype), padding, mode="replicate")
                                second = F.pad(frames[index + 1:index + 2].permute(0, 3, 1, 2).to(device, dtype), padding, mode="replicate")
                                timestep = torch.full((1, 1, 1, 1), step / multiplier, device=device, dtype=dtype)
                                middle = network(first, second, timestep, _scale_list(arch_ver, scale), training=False, fastmode=True, ensemble=False)
                                output[index * multiplier + step] = middle[:, :, :height, :width].clamp(0, 1).permute(0, 2, 3, 1)[0].float().cpu()
                                break
                            except (torch.OutOfMemoryError, RuntimeError) as exc:
                                if not _is_oom(exc):
                                    raise
                                first = second = timestep = None
                                back_off(exc)
                                device, dtype, scale = place()
                    progress.update(1)
            output[-1] = frames[-1].float().cpu()
        finally:
            network.to("cpu")
            arch.backwarp_tenGrid.clear()
            gc.collect()
            model_management.soft_empty_cache()
        return (output,)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": PROMPTFORGE_FrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": "PromptForge Frame Interpolate (low memory)"}
