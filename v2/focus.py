"""Optional per-picture focus: zoom a reference picture onto a SAM3 subject, optionally painting out everything else."""
import asyncio
import io as bytes_io
from collections import OrderedDict

import comfy.utils
import numpy as np
import torch
from aiohttp import web
from PIL import Image
from server import PromptServer

from . import masking

# the settings a reference carries; the same names are widgets on Load Reference Image and fields of a Reference Manager row
DEFAULTS = {"focus": "", "focus_crop": True, "focus_background": "keep", "focus_padding": 0.15, "focus_threshold": 0.5,
            "focus_grow": 0, "focus_feather": 2}
# keep only crops the box around the subject; a colour also paints out everything outside the mask
BACKGROUNDS = {"keep": None, "grey": 0.5, "white": 1.0, "black": 0.0}
_CACHE = OrderedDict()  # the last few focused pictures, so a preview or a rerun does not mask them again
_CACHE_SIZE = 8


def checkpoints():
    return ["auto", *masking.sam3_checkpoints()[1:]]


def _checkpoint(name):
    if name != "auto":
        return name
    found = next((entry for entry in checkpoints()[1:] if "sam3" in entry.lower()), None)
    if found is None:
        raise ValueError("Reference focus needs a SAM3 checkpoint with its text encoder: put one in models/sam3.")
    return found


def compose(image, mask, background="keep", crop=True, padding=0.15):
    """Cut to the mask's box widened by `padding` of its size; a background colour first paints everything outside the mask."""
    rows, cols = (mask[0] > .5).any(1).nonzero().flatten(), (mask[0] > .5).any(0).nonzero().flatten()
    level = BACKGROUNDS[background]
    if level is not None:
        alpha = mask[..., None].to(image)
        image = image * alpha + level * (1 - alpha)
    if crop and len(rows):
        height, width = int(rows[-1] - rows[0]) + 1, int(cols[-1] - cols[0]) + 1
        top, bottom = max(0, int(rows[0] - padding * height)), min(mask.shape[1], int(rows[-1] + 1 + padding * height))
        left, right = max(0, int(cols[0] - padding * width)), min(mask.shape[2], int(cols[-1] + 1 + padding * width))
        image, mask = image[:, top:bottom, left:right], mask[:, top:bottom, left:right]
    return image.clone(), mask.clone()


def apply(path, image, settings, checkpoint="auto", loaded=None):
    """(image, mask) of the focused subject. `loaded` keeps SAM3 open across the pictures of one call."""
    from .media import MaskPrepare
    options = {**DEFAULTS, **settings}
    subject = str(options["focus"]).strip()
    if options["focus_background"] not in BACKGROUNDS:
        raise ValueError(f"Unknown focus background '{options['focus_background']}'.")
    checkpoint = _checkpoint(checkpoint)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, checkpoint, tuple(sorted(options.items())))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    loaded = {} if loaded is None else loaded
    if checkpoint not in loaded:
        loaded[checkpoint] = masking.load_sam3(checkpoint)
    mask = masking.segment(*loaded[checkpoint], image, subject, float(options["focus_threshold"]), 2).float().cpu()
    mask = mask.reshape(-1, *mask.shape[-2:])[:1]
    if not (mask > .5).any():
        raise ValueError(f"SAM3 found no '{subject}' in {path.name}: change the focus words or lower focus_threshold.")
    mask = MaskPrepare.execute(mask, False, int(options["focus_grow"]), int(options["focus_feather"])).args[0]
    result = compose(image, mask, options["focus_background"], bool(options["focus_crop"]), float(options["focus_padding"]))
    _CACHE[key] = result
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def _preview_png(filename, settings, checkpoint):
    from .media import LoadImage
    # no prompt is executing: the server's progress hook has no prompt or node to report to and raises
    hook = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.set_progress_bar_global_hook(None)
    try:
        with torch.inference_mode():
            image = LoadImage.execute(filename=filename, sam3_checkpoint=checkpoint, **settings).args[1]
    finally:
        comfy.utils.set_progress_bar_global_hook(hook)
    picture = Image.fromarray((image[0].clamp(0, 1).numpy() * 255).astype(np.uint8))
    picture.thumbnail((900, 900))
    buffer = bytes_io.BytesIO()
    picture.save(buffer, "PNG")
    return buffer.getvalue()


@PromptServer.instance.routes.post("/promptforge/v2/focus-preview")
async def focus_preview(request):
    body = await request.json()
    if PromptServer.instance.prompt_queue.get_current_queue()[0]:
        return web.Response(status=409, text="A generation is running; preview the focus when it has finished.")
    settings = {name: body["settings"][name] for name in DEFAULTS if name in body.get("settings", {})}
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, _preview_png, str(body.get("file", "")), settings, str(body.get("checkpoint", "auto")))
    except ValueError as exc:
        return web.Response(status=400, text=str(exc))
    return web.Response(body=data, content_type="image/png")
