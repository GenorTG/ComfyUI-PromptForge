"""PromptForge Quantize Model: convert a diffusion model to NVFP4 or INT8 (ConvRot) on this machine.

The result is a normal ComfyUI quantized checkpoint (`<layer>.weight`, its scales and a `<layer>.comfy_quant`
config per layer), saved next to the source file so the loaders list it. Which layers are quantized follows a
structural rule instead of a list per model: only 2D weights inside the repeated transformer blocks, minus
names that are known to be sensitive (embeddings, norms, modulation, input/output projections), plus a few
extra names for families that need them. Sources are read tensor by tensor, so the full model never sits in RAM.
"""
import json
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from comfy_kitchen.tensor import TensorCoreNVFP4Layout, TensorWiseINT8Layout

from . import families

CONVROT_GROUPSIZE = 256
# a 2D weight is a candidate when its name is inside a numbered block of the transformer
BLOCK = re.compile(r"(?:^|\.)(?:[a-z_]*blocks|layers)\.(\d+)\.")
# kept in the original precision for every family
SENSITIVE = ("norm", "embed", "modulation", "adaln", "final_layer", "final_proj", "proj_out", "head.", "time_in", "vector_in",
             "guidance_in", "img_in", "txt_in", "pos_emb", "rope", "patch_proj", "patchify", "router")
# extra names per family (substrings), from what is known to hurt quality when quantized
EXTRA_KEEP = {
    "krea2": ("first", "last", "tmlp", "tproj", "txtfusion", "txtmlp"),
    "minimax_h3": ("adaln_proj", "adaln_t_table", "condition_proj", "token_refiner"),
    "anima": ("llm_adapter",),
    "qwen_image": ("time_text_embed", "img_mod.1"),
    "wan": ("text_embedding", "time_embedding", "time_projection"),
    "ltxv": ("connector", "vocoder", "projection", "table"),
    "flux": ("txt_attn", "class_embedding"),
    "chroma": ("txt_attn",),
    "zimage": ("cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder"),
}
FORMATS = {
    "nvfp4": "NVFP4: about 3.5x smaller, native on RTX 50 (Blackwell)",
    "int8_convrot": "INT8 ConvRot: about 2x smaller, native on RTX 20 and newer",
}


def _convrot_groupsize(k):
    """Largest ConvRot group size that divides the layer's input width."""
    return next((size for size in (CONVROT_GROUPSIZE, 128, 64, 32, 16) if size <= k and k % size == 0), None)


def _quantize(weight, fmt):
    """(tensors keyed by name suffix, comfy_quant config) for one weight, or None when the layer does not fit."""
    if fmt == "nvfp4":
        if weight.shape[0] % 16 or weight.shape[1] % 16:
            return None
        layout, (data, params), config = TensorCoreNVFP4Layout, TensorCoreNVFP4Layout.quantize(weight), {"format": "nvfp4"}
    else:
        size = _convrot_groupsize(weight.shape[1])
        if size is None:
            return None
        layout = TensorWiseINT8Layout
        data, params = layout.quantize(weight, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=size, stochastic_rounding=0)
        config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": size}
    return layout, data, params, config


def _dequantized_error(layout, data, params, weight):
    restored = layout.dequantize(data, params).float()
    return ((restored - weight.float()).norm() / weight.float().norm()).item()


def plan(names_and_shapes, family_key, edge_blocks):
    """Names of the weights to quantize."""
    keep = SENSITIVE + EXTRA_KEEP.get(family_key, ())
    candidates, last_index = [], {}
    for name, shape in names_and_shapes:
        match = BLOCK.search(name)
        if len(shape) != 2 or not name.endswith(".weight") or match is None or any(word in name for word in keep):
            continue
        group = name[:match.start(1)]
        last_index[group] = max(last_index.get(group, 0), int(match.group(1)))
        candidates.append((name, group, int(match.group(1))))
    return [name for name, group, index in candidates if edge_blocks <= index <= last_index[group] - edge_blocks]


class PROMPTFORGE_QuantizeModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "A bf16 / fp16 diffusion model (already quantized files are not converted again)."}),
                "format": (list(FORMATS), {"default": "nvfp4", "tooltip": " | ".join(FORMATS.values())}),
                "family": (["auto"] + [family.label for family in families.FAMILIES], {"default": "auto", "tooltip": "Adds the family's extra sensitive layers to the keep list."}),
                "keep_edge_blocks": ("INT", {"default": 0, "min": 0, "max": 16, "tooltip": "Keep this many blocks at the start and at the end in the original precision. Raises quality and size."}),
                "extra_keep": ("STRING", {"default": "", "tooltip": "Comma separated name fragments of layers to keep in the original precision."}),
                "output_name": ("STRING", {"default": "", "tooltip": "Empty = source name + format. Saved next to the source file."}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("INFO",)
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Converts a diffusion model to NVFP4 or INT8 ConvRot and saves it next to the source. Refresh the model lists afterwards (r) to see the new file."

    def convert(self, model, format, family, keep_edge_blocks, extra_keep, output_name, overwrite):
        source = Path(folder_paths.get_full_path_or_raise("diffusion_models", model))
        if source.suffix != ".safetensors":
            raise ValueError("[PromptForge] only .safetensors models can be converted.")
        stem = output_name.strip().removesuffix(".safetensors") or f"{source.stem}_{format}"
        target = source.with_name(f"{Path(stem).name}.safetensors")
        if target.exists() and not overwrite:
            raise ValueError(f"[PromptForge] {target.name} already exists; turn on overwrite or choose another output_name.")

        chosen = next((f for f in families.FAMILIES if f.label == family), None) or families.family_of_model(model, "diffusion_models", str(source))
        family_key = chosen.key if chosen else ""
        device = model_management.get_torch_device()
        extra = tuple(part.strip() for part in extra_keep.split(",") if part.strip())

        started = time.time()
        with safe_open(str(source), "pt") as handle:
            metadata = handle.metadata() or {}
            if "_quantization_metadata" in metadata or any(key.endswith(".comfy_quant") for key in handle.keys()):
                raise ValueError("[PromptForge] this model is already quantized; convert the original bf16 / fp16 file.")
            shapes = [(key, tuple(handle.get_slice(key).get_shape())) for key in handle.keys()]
            names = {name for name in plan(shapes, family_key, keep_edge_blocks) if not any(word in name for word in extra)}
            if not names:
                raise ValueError("[PromptForge] no transformer block weights to quantize were found; pick the family or lower keep_edge_blocks.")

            output, errors, skipped = {}, {}, 0
            progress = comfy.utils.ProgressBar(len(shapes))
            for index, (name, _) in enumerate(shapes):
                model_management.throw_exception_if_processing_interrupted()
                tensor = handle.get_tensor(name)
                weight = tensor.to(device=device, dtype=torch.bfloat16) if name in names else None
                converted = _quantize(weight, format) if weight is not None else None
                if name in names and converted is None:
                    skipped += 1
                if converted is None:
                    output[name] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
                else:
                    layout, data, params, config = converted
                    layer = name.removesuffix(".weight")
                    for suffix, piece in layout.state_dict_tensors(data, params).items():
                        output[f"{layer}.weight{suffix}"] = piece.cpu()
                    output[f"{layer}.comfy_quant"] = torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)
                    errors[name] = _dequantized_error(layout, data, params, weight)
                progress.update_absolute(index + 1)

        header = {key: value for key, value in metadata.items() if key not in ("workflow", "prompt")}
        header.update({"quantized_by": "PromptForge", "quant_format": format})
        save_file(output, str(target), metadata=header)

        worst = sorted(errors, key=errors.get, reverse=True)[:3]
        size_before, size_after = source.stat().st_size / 1024 ** 3, target.stat().st_size / 1024 ** 3
        info = [
            f"{target.name}: {size_before:.1f} GB -> {size_after:.1f} GB in {time.time() - started:.0f}s ({chosen.label if chosen else 'unknown family'}, {format})",
            f"quantized {len(errors)} layers" + (f", {skipped} did not fit the format and stayed as they were" if skipped else ""),
            f"relative weight error: mean {sum(errors.values()) / len(errors):.3f}, worst {errors[worst[0]]:.3f} ({worst[0]})",
        ]
        return {"ui": {"text": info}, "result": ("\n".join(info),)}


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_QuantizeModel": PROMPTFORGE_QuantizeModel}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_QuantizeModel": "PromptForge Quantize Model (NVFP4 / INT8)"}
