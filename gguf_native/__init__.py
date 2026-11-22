"""GGUF diffusion models and text encoders for the PromptForge Loader, without the ComfyUI-GGUF pack.

dequant.py, ops.py, loader.py and tools/convert.py are ComfyUI-GGUF by City96 (Apache-2.0, see LICENSE-ComfyUI-GGUF),
included unchanged. This file is the part of that pack's nodes the Loader needs: the `unet_gguf` / `clip_gguf` file
lists, the model patcher that keeps quantized weights packed, and the two load functions.
"""
import collections
import inspect
import logging

import torch

import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths

from .dequant import is_quantized, is_torch_compatible
from .loader import gguf_clip_loader, gguf_sd_loader
from .ops import GGMLOps, move_patch_to_device


def _register(key, targets):
    """File list `key` for .gguf files in the folders of the first of `targets` that exists (skipped when ComfyUI-GGUF already made it)."""
    if key in folder_paths.folder_names_and_paths:
        return
    target = next((name for name in targets if name in folder_paths.folder_names_and_paths), targets[0])
    paths, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (paths, {".gguf"})


_register("unet_gguf", ["diffusion_models", "unet"])
_register("clip_gguf", ["text_encoders", "clip"])

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n


def load_unet(path, dequant_dtype=None, patch_dtype=None, patch_on_device=False):
    """A GGUF diffusion model as a MODEL (the quantized weights stay packed and are dequantized per layer)."""
    ops = GGMLOps()
    for attribute, value in (("dequant_dtype", dequant_dtype), ("patch_dtype", patch_dtype)):
        setattr(ops.Linear, attribute, value if value in (None, "target") else getattr(torch, value))
    state, extra = gguf_sd_loader(path)
    kwargs = {"metadata": extra.get("metadata", {})} if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters else {}
    model = comfy.sd.load_diffusion_model_state_dict(state, model_options={"custom_operations": ops}, **kwargs)
    if model is None:
        raise RuntimeError(f"[PromptForge] could not detect the model type of {path}")
    model = GGUFModelPatcher.clone(model)
    model.patch_on_device = patch_on_device
    return model


def load_clip(paths, clip_type):
    """GGUF (or mixed GGUF and regular) text encoder files as a CLIP."""
    data = []
    for path in paths:
        if path.endswith(".gguf"):
            data.append(gguf_clip_loader(path))
        else:
            state = comfy.utils.load_torch_file(path, safe_load=True)
            if "scaled_fp8" in state:
                raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported; use the regular loader or switch files ({path})")
            data.append(state)
    clip = comfy.sd.load_text_encoder_state_dicts(clip_type=clip_type, state_dicts=data, model_options={"custom_operations": GGMLOps, "initial_device": comfy.model_management.text_encoder_offload_device()},
                                                  embedding_directory=folder_paths.get_folder_paths("embeddings"))
    clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    return clip
