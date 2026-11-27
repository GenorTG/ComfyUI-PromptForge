"""Automatic reference masking using ComfyUI's native SAM3 implementation."""
import hashlib
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock

import comfy.sd
import comfy.utils
import folder_paths
import torch.nn.functional as F
from comfy_api.latest import io
from comfy_extras.nodes_sam3 import SAM3_Detect

from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve

Pipe = io.Custom(PIPE)
_SAM3_ATTENTION_LOCK = RLock()
# SAM3 upsamples every candidate mask to the picture size: a 16 MP photo needs 6 GB for that step
SAM3_MAX_SIDE = 1536
# the last masks Subject Mask made: a new seed upstream reruns the node on the same picture, and SAM3 takes seconds to load and run
_MASKS = OrderedDict()
KEPT_MASKS = 8


@lru_cache(maxsize=1024)
def _holds_sam3(path, modified, size):
    header = comfy.utils.safetensors_header(path, max_size=64 * 1024 * 1024)
    try:
        keys = json.loads(header) if header else {}
    except ValueError:
        return False
    # the two weights ComfyUI's model detection takes as SAM3, under any prefix
    return all(any(key.endswith(wanted) for key in keys) for wanted in
               ("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight", "detector.transformer.decoder.query_embed.weight"))


def _is_sam3(path):
    """Whether a file can be offered as SAM3. Only safetensors can be looked into without loading them; others go by name."""
    path = Path(path)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return "sam3" in path.name.lower()
    stat = path.stat()
    return _holds_sam3(str(path), stat.st_mtime_ns, stat.st_size)


def sam3_checkpoints():
    regular = [name for name in folder_paths.get_filename_list("checkpoints")
               if (path := folder_paths.get_full_path("checkpoints", name)) and _is_sam3(path)]
    root = Path(folder_paths.models_dir) / "sam3"
    local = [f"sam3::{path.relative_to(root).as_posix()}" for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".pth") and _is_sam3(path)]
    return ["(select SAM3 checkpoint)"] + regular + sorted(local)


class SubjectMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("SUBJECT", "MASK", "Subject Mask", description="Native SAM3 text segmentation. Select a SAM3 checkpoint containing its text encoder; no external node pack is needed.",
                         inputs=[Pipe.Input("pipe", optional=True), io.Image.Input("image"),
                                 io.Combo.Input("checkpoint", options=sam3_checkpoints()),
                                 io.String.Input("subject", default="face, hair, neck"),
                                 io.Float.Input("threshold", default=0.5, min=0, max=1, step=0.01),
                                 io.Int.Input("refine_iterations", default=2, min=0, max=5)],
                         outputs=[Pipe.Output(), io.Mask.Output("mask")])

    @classmethod
    def execute(cls, pipe=None, image=None, checkpoint=None, subject="face, hair, neck", threshold=0.5, refine_iterations=2):
        pipe = ensure_pipe(pipe)
        if not checkpoint:
            raise ValueError("Select a SAM3 checkpoint first.")
        if image is None:
            raise ValueError("Connect an image to mask.")
        mask = remembered_mask(checkpoint, image, subject, threshold, refine_iterations)
        out = evolve(pipe, "mask", {"checkpoint": checkpoint, "subject": subject, "threshold": threshold,
                                       "refine_iterations": refine_iterations}, {"source_mask": mask})
        return io.NodeOutput(out, mask)


def sam3_path(checkpoint):
    if not checkpoint.startswith("sam3::"):
        return Path(folder_paths.get_full_path_or_raise("checkpoints", checkpoint))
    root = (Path(folder_paths.models_dir) / "sam3").resolve()
    path = (root / checkpoint.removeprefix("sam3::")).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Select an existing SAM3 checkpoint inside models/sam3.")
    return path


def remembered_mask(checkpoint, image, subject, threshold, refine_iterations):
    """segment(), skipped together with loading SAM3 when the same file, picture and settings were just masked."""
    path = sam3_path(checkpoint)
    pixels = hashlib.sha1(image.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()
    key = (str(path), path.stat().st_mtime_ns, tuple(image.shape), pixels, subject, threshold, refine_iterations)
    mask = _MASKS.get(key)
    if mask is None:
        mask = segment(*load_sam3(checkpoint), image, subject, threshold, refine_iterations)
        _MASKS[key] = mask
        while len(_MASKS) > KEPT_MASKS:
            _MASKS.popitem(last=False)
    _MASKS.move_to_end(key)
    return mask.clone()


def load_sam3(checkpoint):
    """(model, clip) of a SAM3 checkpoint picked from the list."""
    path = sam3_path(checkpoint)
    model, clip, *_ = comfy.sd.load_checkpoint_guess_config(str(path), output_vae=False, output_clip=True,
                                                           embedding_directory=folder_paths.get_folder_paths("embeddings"))
    if model.model.model_config.unet_config.get("image_model") not in ("SAM3", "SAM31") or clip is None:
        raise ValueError("Subject Mask requires a native SAM3/SAM3.1 checkpoint with its text encoder.")
    return model, clip


def segment(model, clip, image, subject, threshold=0.5, refine_iterations=2):
    """The mask of every `subject` SAM3 finds in the image, at the image's size."""
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(subject))
    # SAM3's masked cross-attention is unsupported by some xformers builds,
    # including the current Blackwell kernels. Use ComfyUI's PyTorch path.
    import comfy.ldm.sam3.detector as detector
    from comfy.ldm.modules.attention import attention_pytorch
    height, width = image.shape[1:3]
    scale = min(1.0, SAM3_MAX_SIDE / max(height, width))
    small = image if scale == 1.0 else comfy.utils.common_upscale(image.movedim(-1, 1), round(width * scale), round(height * scale), "area", "disabled").movedim(1, -1)
    with _SAM3_ATTENTION_LOCK:
        previous = detector.optimized_attention
        detector.optimized_attention = attention_pytorch
        try:
            mask = SAM3_Detect.execute(model, small, conditioning=conditioning, threshold=threshold,
                                      refine_iterations=refine_iterations, individual_masks=False).args[0]
        finally:
            detector.optimized_attention = previous
    if scale != 1.0:
        mask = F.interpolate(mask.reshape(-1, 1, *mask.shape[-2:]).float(), size=(height, width), mode="bilinear")[:, 0]
    return mask


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_V2_SUBJECT_MASK": SubjectMask}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_V2_SUBJECT_MASK": "PromptForge V2 · Subject Mask"}
