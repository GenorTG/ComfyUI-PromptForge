"""PromptForge image library: tags, themes and NSFW flags for the images in the input folder.

A local vision model (a GGUF with its mmproj) looks at each image once and writes keywords, a one line caption,
how many people are in it and whether it is NSFW. The results live in the user directory
(`promptforge/image_tags.json`) and are used by the Load Image picker for search and blurring. A second job moves
images into theme folders (female, male, group, nsfw ...) and can be undone.

Everything here runs locally. Jobs are started from the picker or the sidebar tab, one at a time, in a thread.
"""
import asyncio
import base64
import gc
import importlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path

from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

import folder_paths
import node_helpers

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
THEMES = ("female", "male", "group", "people", "nsfw", "animals", "scenery", "objects", "other")
SUBJECT_THEMES = {"animal": "animals", "landscape": "scenery", "building": "scenery", "object": "objects", "vehicle": "objects", "food": "objects", "text": "other"}
MAX_SIDE = 768
FORCE_CPU = bool(os.getenv("PROMPTFORGE_TAGGER_CPU"))

PROMPT = """You catalogue images for a personal library. Look at the image and answer with ONE JSON object and nothing else:
{"people": <number of people visible, 3 means three or more>, "gender": "female" | "male" | "mixed" | "none", "nsfw": true | false, "subject": "person" | "animal" | "landscape" | "building" | "object" | "vehicle" | "food" | "text" | "other", "style": "photo" | "illustration" | "anime" | "3d" | "screenshot" | "other", "tags": [10 to 15 short lowercase keywords: hair colour, clothing, setting, pose, mood, notable objects], "caption": "one plain sentence describing the image"}
"nsfw" is true when nudity, sexual content or explicit imagery is visible, in any style including drawings. "gender" is about the main person. Output the JSON only."""


# ---------------------------------------------------------------- stores

def store_dir():
    return Path(folder_paths.get_user_directory()) / "promptforge"


def _read(name):
    try:
        return json.loads((store_dir() / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(name, data):
    store_dir().mkdir(parents=True, exist_ok=True)
    (store_dir() / name).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


descriptions = lambda: _read("image_descriptions.json")
save_descriptions = lambda data: _write("image_descriptions.json", data)
tags = lambda: _read("image_tags.json")
save_tags = lambda data: _write("image_tags.json", data)
_store_lock = threading.Lock()


def input_root():
    return Path(os.path.abspath(folder_paths.get_input_directory()))


def resolve(name):
    """Absolute path of `name` inside the input folder; raises when it points elsewhere or is not an image."""
    root = input_root()
    path = Path(os.path.abspath(root / name))
    if root not in path.parents or path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"[PromptForge] {name} is not an image inside the input folder.")
    return path


def scan():
    root = str(input_root())
    found = []
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for file in files:
            if os.path.splitext(file)[1].lower() in EXTENSIONS:
                full = os.path.join(folder, file)
                found.append({"name": os.path.relpath(full, root).replace(os.sep, "/"), "mtime": os.path.getmtime(full)})
    return found


def theme_of(result):
    if result.get("nsfw"):
        return "nsfw"
    people = int(result.get("people") or 0)
    if people >= 2:
        return "group"
    if people == 1:
        return {"female": "female", "male": "male"}.get(result.get("gender"), "people")
    return SUBJECT_THEMES.get(result.get("subject"), "other")


# ---------------------------------------------------------------- vision model

def _package():
    return importlib.import_module(__package__)


VISION_FAMILIES = ("vl", "qwen3.5", "qwen3.6", "qwen35", "gemma-3", "gemma3", "gemma-4", "gemma4", "llava", "minicpm")


def _is_vision(path, mmproj):
    """An mmproj named after the model always counts; a generic one only for model families that have vision."""
    stem = re.split(r"[-_.](?:q\d|f16|bf16|f32|iq\d)", path.stem.lower())[0]
    if len(stem) >= 8 and mmproj.name.lower().startswith(stem):
        return True
    return any(family in path.name.lower() for family in VISION_FAMILIES)


def vision_models():
    """GGUF models on this machine that have a matching mmproj next to them: [{"id", "label"}]."""
    package = _package()
    models = []
    for entry in package.LOCAL_TEXT_MODELS:
        path = Path(entry["path"])
        mmproj = package._local_mmproj(path, None) if entry["kind"] == "gguf" and path.is_file() else None
        if mmproj is not None and _is_vision(path, mmproj):
            models.append({"id": str(path), "label": path.name})
    return models


class Tagger:
    def __init__(self, model_path):
        package = _package()
        if model_path not in {m["id"] for m in vision_models()}:
            raise ValueError("[PromptForge] pick one of the listed vision models.")
        from llama_cpp import Llama, llama_chat_format
        path = Path(model_path)
        mmproj = package._local_mmproj(path, None)
        handler = getattr(llama_chat_format, package._gguf_vision_handler(path.name), None)
        if handler is None:
            raise ValueError("[PromptForge] this llama-cpp-python build cannot run this vision model.")
        package._free_vram()
        layers = 0 if FORCE_CPU else -1
        if layers and package._gguf_memory_estimate(path, 4096, layers, mmproj) > _free_bytes():
            layers = package._gguf_layers_that_fit(path, 4096, mmproj)
        handler_instance = handler(clip_model_path=str(mmproj), verbose=False)
        for gpu_layers, kv_on_gpu in ((layers, layers != 0), (0, False)):
            try:
                self.llm = Llama(model_path=str(path), chat_handler=handler_instance, n_ctx=4096, n_gpu_layers=gpu_layers, offload_kqv=kv_on_gpu, verbose=False)
                break
            except (ValueError, RuntimeError):
                if gpu_layers == 0:
                    raise
                print("[PromptForge] the vision model does not fit the free VRAM; running it on the CPU.")
        self.split_thinking = package._split_thinking

    def ask(self, path):
        with node_helpers.pillow(Image.open, str(path)) as opened:
            image = node_helpers.pillow(ImageOps.exif_transpose, opened).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88)
        url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        reply = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}],
            max_tokens=500, temperature=0.0)
        return self.split_thinking(reply["choices"][0]["message"]["content"] or "")[0]

    def close(self):
        self.llm.close()
        del self.llm
        gc.collect()
        _package()._free_vram()


def _free_bytes():
    import torch
    try:
        return torch.cuda.mem_get_info()[0]
    except (RuntimeError, AssertionError):
        return 0


def parse(text):
    """The catalogue entry from the model's answer, or None when it is not usable."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or "nsfw" not in data:
        return None
    words = data.get("tags") if isinstance(data.get("tags"), list) else str(data.get("tags", "")).split(",")
    keywords = list(dict.fromkeys(w.strip().lower()[:40] for w in map(str, words) if w.strip()))[:20]
    nsfw = data["nsfw"] if isinstance(data["nsfw"], bool) else str(data["nsfw"]).lower() == "true"
    try:
        people = max(0, int(data.get("people") or 0))
    except (TypeError, ValueError):
        people = 0
    entry = {"people": people, "gender": str(data.get("gender", "none")).lower(), "nsfw": nsfw, "subject": str(data.get("subject", "other")).lower(),
             "style": str(data.get("style", "")).lower(), "tags": keywords, "caption": str(data.get("caption", "")).strip()[:400]}
    entry["theme"] = theme_of(entry)
    return entry


# ---------------------------------------------------------------- jobs

_job = {"running": False, "kind": "", "total": 0, "done": 0, "current": "", "failed": [], "message": "", "cancel": False}
_job_lock = threading.Lock()


def _set(**values):
    with _job_lock:
        _job.update(values)


def _tag_worker(model, names):
    tagger = None
    try:
        _set(message="Loading the vision model...")
        tagger = Tagger(model)
        store = tags()
        for index, name in enumerate(names):
            if _job["cancel"]:
                break
            _set(done=index, current=name, message="Tagging")
            try:
                path = resolve(name)
                text = tagger.ask(path)
                entry = parse(text) or parse(tagger.ask(path))
                if entry is None:
                    raise ValueError("no usable answer")
            except Exception as error:
                with _job_lock:
                    _job["failed"].append(f"{name}: {str(error)[:80]}")
                continue
            entry.update({"mtime": path.stat().st_mtime, "model": Path(model).name, "tagged_at": int(time.time())})
            with _store_lock:
                store = tags()
                store[name] = entry
                save_tags(store)
        _set(done=len(names), message="Cancelled" if _job["cancel"] else "Done")
    except Exception as error:
        _set(message=f"Failed: {error}")
    finally:
        if tagger is not None:
            tagger.close()
        _set(running=False, current="")


def _start(kind, target, *args):
    with _job_lock:
        if _job["running"]:
            return False
        _job.update(running=True, kind=kind, total=len(args[1]), done=0, current="", failed=[], message="Starting", cancel=False)
    threading.Thread(target=target, args=args, daemon=True).start()
    return True


def _targets(names, retag):
    known = tags()
    root = input_root()
    chosen = names if isinstance(names, list) else [image["name"] for image in scan()]
    result = []
    for name in chosen:
        try:
            path = resolve(name)
            current = known.get(name)
            if retag or current is None or abs(current.get("mtime", 0) - path.stat().st_mtime) > 1:
                result.append(path.relative_to(root).as_posix())
        except (ValueError, OSError):
            continue
    return result


def _plan(names, include_subfolders):
    """[(name, theme)] of tagged images that are not already in their theme folder."""
    known = tags()
    chosen = names if isinstance(names, list) else [image["name"] for image in scan()]
    plan = []
    for name in chosen:
        entry = known.get(name)
        if not entry or not (include_subfolders or "/" not in name):
            continue
        if name.split("/")[0] != entry["theme"]:
            plan.append((name, entry["theme"]))
    return plan


def _unique(target):
    if not target.exists():
        return target
    for number in range(2, 10000):
        candidate = target.with_name(f"{target.stem}_{number}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("[PromptForge] too many files with the same name.")


def _organize(plan):
    root = input_root()
    moves = []
    with _store_lock:
        store, notes = tags(), descriptions()
        for name, theme in plan:
            source = resolve(name)
            if not source.is_file():
                continue
            target = _unique(Path(os.path.abspath(root / theme / source.name)))
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            new = target.relative_to(root).as_posix()
            for table in (store, notes):
                if name in table:
                    table[new] = table.pop(name)
            moves.append([name, new])
        save_tags(store)
        save_descriptions(notes)
        _write("organize_log.json", {"moves": moves, "time": int(time.time())})
    return moves


def _undo():
    log = _read("organize_log.json").get("moves", [])
    root = input_root()
    restored = 0
    with _store_lock:
        store, notes = tags(), descriptions()
        for old, new in reversed(log):
            source, target = Path(os.path.abspath(root / new)), Path(os.path.abspath(root / old))
            if not source.is_file() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            for table in (store, notes):
                if new in table:
                    table[old] = table.pop(new)
            restored += 1
            try:
                source.parent.rmdir()
            except OSError:
                pass
        save_tags(store)
        save_descriptions(notes)
        _write("organize_log.json", {"moves": [], "time": int(time.time())})
    return restored


# ---------------------------------------------------------------- routes

routes = PromptServer.instance.routes


@routes.get("/promptforge/library/models")
async def _models_route(request):
    return web.json_response({"models": vision_models()})


@routes.get("/promptforge/library/status")
async def _status_route(request):
    with _job_lock:
        return web.json_response({key: value for key, value in _job.items() if key != "cancel"})


@routes.get("/promptforge/library/info")
async def _info_route(request):
    return web.json_response(tags().get(request.query.get("name", ""), {}))


@routes.post("/promptforge/library/tag")
async def _tag_route(request):
    body = await request.json()
    names = _targets(body.get("names") if isinstance(body.get("names"), list) else "all", bool(body.get("retag")))
    if not names:
        return web.json_response({"started": False, "message": "Nothing to tag."})
    try:
        started = _start("tag", _tag_worker, str(body.get("model", "")), names)
    except ValueError as error:
        return web.json_response({"started": False, "message": str(error)}, status=400)
    return web.json_response({"started": started, "count": len(names), "message": "" if started else "A job is already running."})


@routes.post("/promptforge/library/cancel")
async def _cancel_route(request):
    _set(cancel=True)
    return web.json_response({})


@routes.post("/promptforge/library/organize")
async def _organize_route(request):
    body = await request.json()
    with _job_lock:
        if _job["running"]:
            return web.json_response({"error": "A job is running."}, status=409)
    plan = _plan(body.get("names") if isinstance(body.get("names"), list) else "all", bool(body.get("include_subfolders")))
    if body.get("dry_run", True):
        counts = {}
        for _, theme in plan:
            counts[theme] = counts.get(theme, 0) + 1
        return web.json_response({"counts": counts, "total": len(plan), "sample": [[a, b] for a, b in plan[:6]]})
    moves = await asyncio.get_running_loop().run_in_executor(None, _organize, plan)
    return web.json_response({"moved": len(moves)})


@routes.post("/promptforge/library/undo")
async def _undo_route(request):
    return web.json_response({"restored": await asyncio.get_running_loop().run_in_executor(None, _undo)})


@routes.post("/promptforge/library/edit")
async def _edit_route(request):
    body = await request.json()
    name = str(body.get("name", ""))
    try:
        resolve(name)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    with _store_lock:
        store = tags()
        entry = store.setdefault(name, {"people": 0, "gender": "none", "subject": "other", "style": "", "tags": [], "caption": "", "model": "manual"})
        if "nsfw" in body:
            entry["nsfw"] = bool(body["nsfw"])
        entry.setdefault("nsfw", False)
        if "tags" in body:
            entry["tags"] = list(dict.fromkeys(w.strip().lower() for w in str(body["tags"]).split(",") if w.strip()))[:30]
        if "caption" in body:
            entry["caption"] = str(body["caption"])[:400]
        if body.get("theme") in THEMES:
            entry["theme"] = body["theme"]
        else:
            entry["theme"] = theme_of(entry)
        entry["mtime"] = resolve(name).stat().st_mtime
        save_tags(store)
    return web.json_response(entry)
