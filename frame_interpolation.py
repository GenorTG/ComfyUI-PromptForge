"""PromptForge Frame Interpolate: RIFE / FILM frame interpolation in one node that backs off instead of crashing.

It uses ComfyUI's own interpolation models and node (no other custom node pack is needed). The checkpoint list holds
the models in `models/frame_interpolation` (RIFE 4.25+ and FILM; older RIFE .pth files such as rife49 are not supported by ComfyUI's loader). The clip is
interpolated in short runs that share a frame (the models only look at neighbouring frames, so the result is the same), and
a clip that does not fit in RAM is kept on disk. On out of GPU memory the models are unloaded and the runs shortened; a
pair of frames that still does not fit is cross-faded instead of failing.

`auto` streams one pair of frames at a time through the model on the GPU, with no batch size to pick. Measured on 16 GB
cards it is about twice as fast as the runs, and larger batches of pairs were slower while taking more memory.
"""
import gc

import torch
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths
from comfy.ldm.common_dit import pad_to_patch_size
from comfy_extras.nodes_frame_interpolation import FrameInterpolate, FrameInterpolationModelLoader

from . import frame_buffer

BLEND = "blend (no model, instant)"
NO_CKPT = "(no frame interpolation model found)"
RUN_BYTES = 1024 ** 3  # ComfyUI's node builds the frames of a run in full precision in RAM
AUTO_TOOLTIP = ("Fastest mode, nothing to tune: one frame pair at a time on the GPU, which needs the least memory. batch_size, "
                "overlap and clear_cache are not used. Not used with blend.")


def _checkpoints():
    return {name: folder_paths.get_full_path("frame_interpolation", name) for name in folder_paths.get_filename_list("frame_interpolation")}


def _load(path):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    model = FrameInterpolationModelLoader._detect_and_load(state)
    dtype = torch.float16 if model_management.should_use_fp16(model_management.get_torch_device()) else torch.float32
    model.eval().to(dtype)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device(),
                                                fast_disk=comfy.storage.state_dict_fast_disk(state))


def _resize(frames, size):
    return F.interpolate(frames.movedim(-1, 1).float(), size=size, mode="bilinear", align_corners=False).movedim(1, -1)


def _blend(frames, multiplier):
    out = [frames[:1]]
    for index in range(frames.shape[0] - 1):
        steps = torch.arange(1, multiplier + 1, dtype=frames.dtype).view(-1, 1, 1, 1) / multiplier
        out.append(torch.lerp(frames[index:index + 1], frames[index + 1:index + 2], steps))
    return torch.cat(out)


def _stream(patcher, frames, multiplier, out, small):
    """Fills `out` with the interpolated clip pair by pair. Each frame is uploaded and encoded once, and only finished
    frames come back, already in the output's dtype."""
    device, dtype, net = patcher.load_device, patcher.model_dtype(), patcher.model
    count, height, width = frames.shape[:3]
    align = getattr(net, "pad_align", 1)
    multi = getattr(net, "forward_multi_timestep", None)
    times = [step / multiplier for step in range(1, multiplier)]

    def prepare(index):
        frame = frames[index:index + 1].to(device).movedim(-1, 1).to(dtype)
        if small:
            frame = F.interpolate(frame.float(), size=small, mode="bilinear", align_corners=False).to(dtype)
        if align > 1:
            frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
        return frame, net.extract_features(frame)

    def between(first, second):
        (img0, features0), (img1, features1) = first, second
        cache = {"img0": features0, "img1": features1}
        if multi is not None and len(times) > 1:
            mids = multi(img0, img1, times, cache=cache)
        else:
            mids = torch.cat([net(img0, img1, timestep=img0.new_full((1, 1, *img0.shape[2:]), t), cache=cache) for t in times])
        mids = mids[:, :, :small[0], :small[1]] if small else mids[:, :, :height, :width]
        if small:
            mids = F.interpolate(mids.float(), size=(height, width), mode="bilinear", align_corners=False)
        return mids.clamp(0, 1).to(out.dtype).movedim(1, -1).cpu()

    previous, unloaded, blended = None, False, 0
    pbar = comfy.utils.ProgressBar(count - 1)
    out[0] = frames[0]
    index = 0
    while index < count - 1:
        model_management.throw_exception_if_processing_interrupted()
        mids = None
        try:
            model_management.load_models_gpu([patcher], memory_required=net.memory_used_forward(frames.shape, dtype))
            previous = previous or prepare(index)
            current = prepare(index + 1)
            mids = between(previous, current)
        except Exception as exc:
            model_management.raise_non_oom(exc)
        if mids is None:
            previous = current = None
            gc.collect()
            model_management.soft_empty_cache()
            if not unloaded:
                print("[PromptForge] frame interpolation ran out of memory; unloading the other models.")
                model_management.unload_all_models()
                unloaded = True
                continue
            blended += 1
            mids = _blend(frames[index:index + 2], multiplier)[1:-1]
        out[index * multiplier + 1:(index + 1) * multiplier] = mids
        out[(index + 1) * multiplier] = frames[index + 1]
        previous = current
        index += 1
        pbar.update_absolute(index)
    if blended:
        print(f"[PromptForge] frame interpolation did not fit for {blended} frame pairs; those were cross-faded.")


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
            },
            "optional": {
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 240, "tooltip": "Most frames per model run. 0 = automatic from the frame size. The result is the same either way; runs are shortened further on OOM."}),
                "overlap": ("INT", {"default": 2, "min": 1, "max": 12, "tooltip": "Extra frames shared by neighbouring runs and cross-faded."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Interpolate at a smaller size, then scale back. 0.5 is much faster and uses far less memory; fine detail softens."}),
                "clear_cache": ("BOOLEAN", {"default": True, "tooltip": "Empty the cache after each run when the clip is split. Off is slightly faster, on is safer on low VRAM."}),
                "auto": ("BOOLEAN", {"default": False, "tooltip": AUTO_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "🔮 PromptForge/Video"
    DESCRIPTION = "RIFE / FILM frame interpolation with ComfyUI's own models, in memory-sized runs; long clips go to disk instead of running out of memory."

    def interpolate(self, frames, multiplier, ckpt_name, batch_size=0, overlap=2, scale=1.0, clear_cache=True, auto=False, hold=0):
        count = int(frames.shape[0])
        if count < 2:
            return (frames,)
        frames = frames[..., :3]
        height, width = frames.shape[1:3]
        small = (max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2)) if scale < 1.0 else None
        patcher = None
        if ckpt_name != BLEND:
            paths = _checkpoints()
            if ckpt_name not in paths:
                raise ValueError("[PromptForge] no frame interpolation model was found. Put a RIFE 4.25+ or FILM model in models/frame_interpolation (for example rife_v4.26.safetensors), or choose blend.")
            patcher = _load(paths[ckpt_name])
        total = (count - 1) * multiplier + 1
        shape = (total + hold, height, width, 3)
        out = frame_buffer.empty(shape, frame_buffer.video_dtype(shape[0] * height * width * 3))

        if auto and patcher is not None:
            _stream(patcher, frames, multiplier, out, small)
            return self._finish(out, total, hold)

        longest = 1 + max(1, RUN_BYTES // (multiplier * height * width * 12))
        span = max(2, min(count, longest, batch_size or count))
        start, filled, written_end, unloaded = 0, 0, 0, False
        pbar = comfy.utils.ProgressBar(count - 1)
        while start < count - 1:
            model_management.throw_exception_if_processing_interrupted()
            end = min(count, start + span)
            run = frames[start:end]
            if small:
                run = _resize(run, small)
            result = None
            if patcher is None:
                result = _blend(run, multiplier)
            else:
                try:
                    result = FrameInterpolate.execute(patcher, run, multiplier).args[0]
                except Exception as exc:
                    model_management.raise_non_oom(exc)
            if result is None:
                gc.collect()
                model_management.soft_empty_cache()
                if not unloaded:
                    print("[PromptForge] frame interpolation ran out of memory; unloading the other models.")
                    model_management.unload_all_models()
                    unloaded = True
                    continue
                if span > 2:
                    span = max(2, span // 2)
                    print(f"[PromptForge] frame interpolation ran out of memory; continuing in runs of {span} frames.")
                    continue
                print(f"[PromptForge] frame interpolation ran out of memory even for two frames; cross-fading frames {start + 1} to {end}.")
                result = _blend(run, multiplier)
            if small:
                result = _resize(result, (height, width))
            shared = (written_end - 1 - start) * multiplier + 1 if filled else 0
            if shared:
                head = result[:shared].to(out.dtype)
                weights = torch.arange(1, shared + 1, dtype=out.dtype).view(-1, 1, 1, 1) / shared
                before = out[filled - shared:filled]
                out[filled - shared:filled] = before * (1 - weights) + head * weights
            tail = result[shared:]
            out[filled:filled + len(tail)] = tail
            filled += len(tail)
            written_end = end
            pbar.update_absolute(end - 1)
            start = end - 1 if end == count else max(start + 1, end - 1 - min(overlap, (span - 1) // 4))
            del run, result, tail
            if clear_cache:
                model_management.soft_empty_cache()
        del patcher
        return self._finish(out, total, hold)

    def _finish(self, out, total, hold):
        if hold:
            out[total:] = out[total - 1]
        gc.collect()
        model_management.soft_empty_cache()
        return (out,)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": PROMPTFORGE_FrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_FrameInterpolate": "PromptForge Frame Interpolate (low memory)"}
