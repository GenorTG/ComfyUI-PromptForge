"""Sidebar browser: the output and input folders, per-file notes, and dry workflow templates.

Files in either folder can be deleted; files in the input folder can also be renamed, tagged and described;
the vision analysis reuses the pack's local vision GGUF loader, so it runs on the machine and
never blocks the request that starts it. Notes live in `promptforge/browser_meta.json` in the
user directory, keyed by root and relative path.

Every route takes `root=output|input` plus a path relative to that root and refuses anything
that resolves outside it. Templates are read from `example_workflows/manual`, one folder per
model family.
"""
import asyncio
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av

from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

import folder_paths
import node_helpers

from .. import image_library

ROOT_NAMES = {"output": folder_paths.get_output_directory, "input": folder_paths.get_input_directory}
MANUAL = Path(__file__).resolve().parents[1] / "example_workflows" / "manual"
KINDS = {"image": image_library.EXTENSIONS,
         "video": {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"},
         "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}}
LIMIT = 2000

routes = PromptServer.instance.routes
_store_lock = threading.Lock()
# thumbnails and video conversions get their own few threads: on the shared default pool a folder of hundreds
# of pictures starved the folder listing and the file the user had just clicked
_thumb_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="pf-thumb")
_convert_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pf-convert")


def _root(name):
    if name not in ROOT_NAMES:
        raise ValueError("root must be output or input")
    return Path(ROOT_NAMES[name]()).resolve()


def _resolve(name, rel):
    root = _root(name)
    path = (root / (rel or "")).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path leaves its root folder.")
    return root, path


def _store():
    return Path(folder_paths.get_user_directory()) / "promptforge" / "browser_meta.json"


def _read():
    try:
        return json.loads(_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(store):
    _store().parent.mkdir(parents=True, exist_ok=True)
    _store().write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")


def _legacy_note(legacy):
    """The image library's tags, caption and NSFW flag in the shape the browser stores."""
    if not legacy:
        return {}
    categories = [legacy[key] for key in ("subject", "style", "theme") if legacy.get(key) not in (None, "", "other")]
    return {"tags": [*legacy.get("tags", []), *categories], "description": legacy.get("caption", ""), "nsfw": bool(legacy.get("nsfw"))}


def _kind(path):
    suffix = path.suffix.lower()
    for kind, extensions in KINDS.items():
        if suffix in extensions:
            return kind
    return "other"


def _entry(root, path):
    stat = path.stat()
    return {"name": path.name, "path": path.relative_to(root).as_posix(), "kind": _kind(path),
            "size": stat.st_size, "mtime": stat.st_mtime}


@routes.get("/promptforge/browser/list")
async def _list(request):
    name = request.query.get("root", "output")
    rel = request.query.get("path", "")
    query = request.query.get("q", "").strip().lower()
    kind = request.query.get("kind", "all")
    try:
        payload = await asyncio.get_running_loop().run_in_executor(None, _listing, name, rel, query, kind)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if payload is None:
        return web.json_response({"error": "Folder not found."}, status=404)
    return web.json_response(payload)


def _listing(name, rel, query, kind):
    root, base = _resolve(name, rel)
    if not base.is_dir():
        return None
    notes = _read()
    legacy = image_library.tags() if name == "input" else {}
    dirs, files, total = [], [], 0
    for folder, children, names in os.walk(base, followlinks=False):
        children[:] = sorted(child for child in children if not child.startswith("."))
        current = Path(folder)
        if current != base:
            dirs.append({"name": current.name, "path": current.relative_to(root).as_posix(), "count": 0})
        for file_name in names:
            if file_name.startswith("."):
                continue
            path = current / file_name
            entry = _entry(root, path)
            note = notes.get(f"{name}:{entry['path']}") or _legacy_note(legacy.get(entry["path"]))
            if query and query not in entry["path"].lower() and query not in " ".join(note.get("tags", [])).lower() and query not in str(note.get("description", "")).lower():
                continue
            if kind != "all" and entry["kind"] != kind:
                continue
            entry["meta"] = note
            files.append(entry)
            if query:
                # keep the folder visible when searching below it
                folder_rel = current.relative_to(root).as_posix()
                for folder_entry in dirs:
                    if folder_entry["path"] == folder_rel:
                        folder_entry["count"] += 1
                        break
            total += 1
    files.sort(key=lambda item: item["mtime"], reverse=True)
    crumbs, walk = [], ""
    for part in base.relative_to(root).parts:
        walk = f"{walk}/{part}".lstrip("/")
        crumbs.append({"name": part, "path": walk})
    dirs.sort(key=lambda item: item["name"].lower())
    return {"root": name, "path": base.relative_to(root).as_posix() if base != root else "", "crumbs": crumbs,
            "dirs": dirs, "files": files[:LIMIT], "total": total, "truncated": total > LIMIT}


@routes.get("/promptforge/browser/file")
async def _file(request):
    try:
        _, path = _resolve(request.query.get("root", "output"), request.query.get("path", ""))
    except ValueError as error:
        return web.Response(status=400, text=str(error))
    if not path.is_file():
        return web.Response(status=404, text="File not found.")
    headers = {"Cache-Control": "private, max-age=300"}
    return web.FileResponse(path, headers=headers)


def _thumbnail(path, size):
    stat = path.stat()
    digest = hashlib.sha1(f"{path}|{stat.st_mtime_ns}|{stat.st_size}|{size}".encode()).hexdigest()
    cached = image_library.store_dir() / "browser_thumbs" / f"{digest}.jpg"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        with node_helpers.pillow(Image.open, str(path)) as image:
            image.draft("RGB", (size, size))
            image = node_helpers.pillow(ImageOps.exif_transpose, image)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", image.size, (24, 24, 24, 255)), image)
            image = image.convert("RGB")
            image.thumbnail((size, size))
            image.save(cached, "JPEG", quality=82)
    return cached


@routes.get("/promptforge/browser/thumb")
async def _thumb(request):
    try:
        _, path = _resolve(request.query.get("root", "output"), request.query.get("path", ""))
        size = min(512, max(48, int(request.query.get("size", 192))))
    except ValueError:
        return web.Response(status=400, text="Use root=output|input, a path and a size.")
    if not path.is_file() or path.suffix.lower() not in KINDS["image"]:
        return web.Response(status=404, text="No thumbnail for that file.")
    try:
        cached = await asyncio.get_running_loop().run_in_executor(_thumb_pool, _thumbnail, path, size)
    except OSError:
        return web.Response(status=422, text="The image could not be decoded.")
    return web.FileResponse(cached, headers={"Cache-Control": "private, max-age=86400"})


def _playable(path):
    """An 8-bit H.264 copy of a video for browsers that cannot decode its codec (10-bit HEVC is what H3 saves)."""
    stat = path.stat()
    digest = hashlib.sha1(f"{path}|{stat.st_mtime_ns}|{stat.st_size}|play".encode()).hexdigest()
    cached = image_library.store_dir() / "browser_thumbs" / f"{digest}.mp4"
    if cached.exists():
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(".part.mp4")
    with av.open(str(path)) as source, av.open(str(partial), "w", format="mp4") as target:
        video = source.streams.video[0]
        rate = video.average_rate or 24
        out_video = target.add_stream("libx264", rate=rate, options={"crf": "23", "preset": "veryfast", "movflags": "+faststart"})
        out_video.width, out_video.height, out_video.pix_fmt = video.width - video.width % 2, video.height - video.height % 2, "yuv420p"
        audio = source.streams.audio[0] if source.streams.audio else None
        out_audio = target.add_stream("aac", rate=audio.rate) if audio else None
        for packet in source.demux(video, *([audio] if audio else [])):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    frame = frame.reformat(width=out_video.width, height=out_video.height, format="yuv420p")
                    for encoded in out_video.encode(frame):
                        target.mux(encoded)
                else:
                    frame.pts = None
                    for encoded in out_audio.encode(frame):
                        target.mux(encoded)
        for stream in (out_video, out_audio):
            if stream is not None:
                for encoded in stream.encode(None):
                    target.mux(encoded)
    partial.replace(cached)
    return cached


@routes.get("/promptforge/browser/playable")
async def _play(request):
    try:
        _, path = _resolve(request.query.get("root", "output"), request.query.get("path", ""))
    except ValueError as error:
        return web.Response(status=400, text=str(error))
    if not path.is_file() or path.suffix.lower() not in KINDS["video"]:
        return web.Response(status=404, text="No video at that path.")
    try:
        cached = await asyncio.get_running_loop().run_in_executor(_convert_pool, _playable, path)
    except (av.error.FFmpegError, OSError, IndexError) as error:
        return web.Response(status=422, text=f"The video could not be converted: {error}")
    return web.FileResponse(cached, headers={"Cache-Control": "private, max-age=3600"})


@routes.post("/promptforge/browser/rename")
async def _rename(request):
    body = await request.json()
    name = str(body.get("root", ""))
    try:
        root, path = _resolve(name, str(body.get("path", "")))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if not path.is_file():
        return web.json_response({"error": "File not found."}, status=404)
    new_name = Path(str(body.get("name", "")).strip()).name
    if not new_name or new_name.startswith("."):
        return web.json_response({"error": "Enter a file name."}, status=400)
    if not Path(new_name).suffix:
        new_name += path.suffix
    target = path.with_name(new_name)
    if target != path and target.exists():
        return web.json_response({"error": "A file with that name already exists."}, status=409)
    old_rel = path.relative_to(root).as_posix()
    path.rename(target)
    new_rel = target.relative_to(root).as_posix()
    with _store_lock:
        store = _read()
        key, new_key = f"{name}:{old_rel}", f"{name}:{new_rel}"
        if key in store:
            store[new_key] = store.pop(key)
            _write(store)
    return web.json_response({"path": new_rel, "name": new_name})


@routes.post("/promptforge/browser/delete")
async def _delete(request):
    body = await request.json()
    name = str(body.get("root", ""))
    try:
        root, path = _resolve(name, str(body.get("path", "")))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if not path.is_file() or _kind(path) == "other":
        return web.json_response({"error": "Only picture, video and audio files can be deleted."}, status=404)
    rel = path.relative_to(root).as_posix()
    path.unlink()
    with _store_lock:
        store = _read()
        if store.pop(f"{name}:{rel}", None) is not None:
            _write(store)
    if name == "input":
        for load, save in ((image_library.tags, image_library.save_tags), (image_library.descriptions, image_library.save_descriptions)):
            data = load()
            if data.pop(rel, None) is not None:
                save(data)
    return web.json_response({"deleted": rel})


@routes.post("/promptforge/browser/meta")
async def _meta(request):
    body = await request.json()
    name = str(body.get("root", ""))
    try:
        root, path = _resolve(name, str(body.get("path", "")))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if not path.is_file():
        return web.json_response({"error": "File not found."}, status=404)
    key = f"{name}:{path.relative_to(root).as_posix()}"
    with _store_lock:
        store = _read()
        entry = store.setdefault(key, {"tags": [], "description": "", "nsfw": False})
        if "tags" in body:
            entry["tags"] = list(dict.fromkeys(word.strip().lower() for word in str(body["tags"]).split(",") if word.strip()))[:40]
        if "description" in body:
            entry["description"] = str(body["description"]).strip()[:2000]
        if "nsfw" in body:
            entry["nsfw"] = bool(body["nsfw"])
        entry["updated"] = int(time.time())
        _write(store)
    return web.json_response(entry)


_job = {"running": False, "root": "", "path": "", "error": "", "entry": None, "saved": False}
_job_lock = threading.Lock()


def _analyse(path, model):
    tagger = None
    try:
        tagger = image_library.Tagger(model)
        text = tagger.ask(path)
        entry = image_library.parse(text) or image_library.parse(tagger.ask(path))
        if entry is None:
            raise ValueError("The vision model returned no usable description.")
        with _job_lock:
            _job["entry"] = {"description": entry["caption"], "tags": entry["tags"], "nsfw": entry["nsfw"], "model": Path(model).name}
    except Exception as error:
        with _job_lock:
            _job["error"] = str(error)
    finally:
        try:
            if tagger is not None:
                tagger.close()
        finally:
            with _job_lock:
                _job["running"] = False


@routes.post("/promptforge/browser/analyse")
async def _analyse_start(request):
    body = await request.json()
    name = str(body.get("root", ""))
    try:
        root, path = _resolve(name, str(body.get("path", "")))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if path.suffix.lower() not in KINDS["image"]:
        return web.json_response({"error": "Analysis works on image files only."}, status=400)
    if not path.is_file():
        return web.json_response({"error": "File not found."}, status=404)
    models = image_library.vision_models()
    if not models:
        return web.json_response({"error": "No local vision model with an mmproj was found in the model folders."}, status=400)
    with _job_lock:
        if _job["running"]:
            return web.json_response({"started": False, "error": "Another analysis is running."}, status=409)
        _job.update(running=True, root=name, path=path.relative_to(root).as_posix(), error="", entry=None, saved=False)
    threading.Thread(target=_analyse, args=(path, models[0]["id"]), daemon=True).start()
    return web.json_response({"started": True})


@routes.get("/promptforge/browser/analyse")
async def _analyse_status(request):
    with _job_lock:
        job = dict(_job)
        save = bool(job["entry"]) and not job["running"] and not job["saved"]
        if save:
            _job["saved"] = True
    if save:
        key = f"{job['root']}:{job['path']}"
        with _store_lock:
            store = _read()
            entry = store.setdefault(key, {"tags": [], "description": "", "nsfw": False})
            entry.update(job["entry"], updated=int(time.time()))
            _write(store)
    return web.json_response(job)


@routes.get("/promptforge/browser/templates")
async def _templates(request):
    groups = {}
    if MANUAL.is_dir():
        for family in sorted(MANUAL.iterdir(), key=lambda item: item.name.lower()):
            if not family.is_dir():
                continue
            items = []
            for path in sorted(family.rglob("*.json")):
                items.append({"name": path.relative_to(MANUAL).as_posix(), "title": path.stem.replace("_", " ")})
            if items:
                groups[family.name] = items
    return web.json_response({"groups": [{"family": family, "templates": items} for family, items in groups.items()]})


@routes.get("/promptforge/browser/template")
async def _template(request):
    name = request.query.get("name", "")
    root = MANUAL.resolve()
    path = (root / name).resolve()
    if root not in path.parents or path.suffix.lower() != ".json" or not path.is_file():
        return web.Response(status=404, text="Template not found.")
    return web.json_response(json.loads(path.read_text(encoding="utf-8")))


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
