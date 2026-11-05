"""PromptForge Load Image: pick any image from the input folder (subfolders included) in a thumbnail browser,
and keep a text description per image that the node outputs next to the picture.

The descriptions live in the user directory (`promptforge/image_descriptions.json`, keyed by the path inside
the input folder), so switching to another character also switches to its description. A description typed into
the node wins over the stored one, which makes the node usable with one-off text as well.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

import folder_paths
import node_helpers

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}


def _store_dir():
    return Path(folder_paths.get_user_directory()) / "promptforge"


def _descriptions():
    try:
        return json.loads((_store_dir() / "image_descriptions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _resolve(name):
    """Absolute path of `name` inside the input folder; raises when it points elsewhere or is not an image."""
    root = Path(os.path.abspath(folder_paths.get_input_directory()))
    path = Path(os.path.abspath(root / name))
    if root not in path.parents or path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"[PromptForge] {name} is not an image inside the input folder.")
    return path


def _scan():
    root = folder_paths.get_input_directory()
    found = []
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for file in files:
            if os.path.splitext(file)[1].lower() in EXTENSIONS:
                full = os.path.join(folder, file)
                found.append({"name": os.path.relpath(full, root).replace(os.sep, "/"), "mtime": os.path.getmtime(full)})
    return found


@PromptServer.instance.routes.get("/promptforge/images")
async def _images_route(request):
    described = _descriptions()
    images = await asyncio.get_running_loop().run_in_executor(None, _scan)
    for image in images:
        image["described"] = bool(described.get(image["name"], "").strip())
        image["description"] = described.get(image["name"], "")[:300]
    return web.json_response({"images": images})


def _thumbnail(path, size):
    stat = path.stat()
    cached = _store_dir() / "thumbs" / (hashlib.sha1(f"{path}|{stat.st_mtime_ns}|{size}".encode()).hexdigest() + ".jpg")
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        with node_helpers.pillow(Image.open, str(path)) as image:
            image = node_helpers.pillow(ImageOps.exif_transpose, image)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                backdrop = Image.new("RGBA", image.size, (40, 40, 40, 255))
                image = Image.alpha_composite(backdrop, image)
            image = image.convert("RGB")
            image.thumbnail((size, size))
            image.save(cached, "JPEG", quality=82)
    return cached


@PromptServer.instance.routes.get("/promptforge/image_thumb")
async def _thumb_route(request):
    try:
        path = _resolve(request.query.get("name", ""))
        size = min(1024, max(32, int(request.query.get("size", 200))))
        cached = await asyncio.get_running_loop().run_in_executor(None, _thumbnail, path, size)
    except (ValueError, OSError):
        return web.Response(status=404)
    return web.FileResponse(cached, headers={"Cache-Control": "max-age=86400"})


@PromptServer.instance.routes.get("/promptforge/image_description")
async def _description_route(request):
    return web.json_response({"description": _descriptions().get(request.query.get("name", ""), "")})


@PromptServer.instance.routes.post("/promptforge/image_description")
async def _save_description_route(request):
    body = await request.json()
    try:
        name = _resolve(str(body.get("name", ""))).relative_to(os.path.abspath(folder_paths.get_input_directory())).as_posix()
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    store = _descriptions()
    text = str(body.get("description", "")).strip()
    if text:
        store[name] = text
    else:
        store.pop(name, None)
    _store_dir().mkdir(parents=True, exist_ok=True)
    (_store_dir() / "image_descriptions.json").write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")
    return web.json_response({"saved": bool(text)})


class PROMPTFORGE_LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": "", "tooltip": "Path inside the input folder. Use Choose image to pick it from thumbnails."}),
                "description": ("STRING", {"default": "", "multiline": True, "placeholder": "Description of this image / character. Empty = the stored description.", "tooltip": "Output as DESCRIPTION. Choose image fills in the stored description; Save description stores this text for the image."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "DESCRIPTION", "NAME")
    FUNCTION = "load"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Load an image from a thumbnail browser of the input folder and output a per-image description (stored, or typed in the node) as text."

    @classmethod
    def IS_CHANGED(cls, image, description):
        try:
            path = _resolve(image)
            return f"{path.stat().st_mtime_ns}|{description}|{_descriptions().get(image, '')}"
        except (ValueError, OSError):
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image.strip():
            return "choose an image first"
        try:
            return True if _resolve(image).is_file() else f"{image} does not exist"
        except ValueError as error:
            return str(error)

    def load(self, image, description):
        path = _resolve(image)
        with node_helpers.pillow(Image.open, str(path)) as opened:
            frame = node_helpers.pillow(ImageOps.exif_transpose, opened)
            if frame.mode == "I":
                frame = frame.point(lambda value: value * (1 / 255))
            rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
            alpha = np.asarray(frame.getchannel("A"), dtype=np.float32) / 255.0 if "A" in frame.getbands() else None
        mask = 1.0 - torch.from_numpy(alpha) if alpha is not None else torch.zeros(rgb.shape[:2], dtype=torch.float32)
        text = description.strip() or _descriptions().get(image, "").strip()
        return (torch.from_numpy(rgb)[None], mask[None], text, path.stem)


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_LoadImage": PROMPTFORGE_LoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_LoadImage": "PromptForge Load Image"}
