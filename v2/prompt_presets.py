"""Saved presets of the V2 Prompt node, one pool per prompt source, in user/promptforge/prompt_presets.json.

The first read takes over the older single ideas list (ideas.json, still used by the legacy H3 panel) and sorts it by source.
"""
import importlib
import json
import os
import threading

from aiohttp import web
from server import PromptServer

POOLS = ("raw", "idea", "advanced", "action")
_library = importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.image_library")
_lock = threading.Lock()


def _path(name="prompt_presets.json"):
    return _library.store_dir() / name


def read():
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        saved = None
    if saved is not None:
        return {pool: dict(saved.get(pool) or {}) for pool in POOLS}
    pools = {pool: {} for pool in POOLS}
    try:
        old = json.loads(_path("ideas.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    for name, data in old.items():
        if isinstance(data, dict):
            source = data.get("source")
            pools[source if source in POOLS else "idea"][name] = data
    return pools


def change(pool, name, data=None):
    """Saves a preset, or deletes it when data is None; returns the pool."""
    if pool not in POOLS:
        raise ValueError(f"Unknown preset pool: {pool}")
    with _lock:
        pools = read()
        if data is None:
            pools[pool].pop(name, None)
        else:
            pools[pool][name] = data
        _path().parent.mkdir(parents=True, exist_ok=True)
        temporary = _path("prompt_presets.json.tmp")
        temporary.write_text(json.dumps(pools, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, _path())
    return pools[pool]


def _error(message):
    return web.json_response({"error": message}, status=400)


@PromptServer.instance.routes.get("/promptforge/v2/prompt-presets")
async def _list(request):
    pool = request.query.get("pool", "")
    return web.json_response(read()[pool]) if pool in POOLS else _error("unknown pool")


@PromptServer.instance.routes.post("/promptforge/v2/prompt-presets")
async def _save(request):
    body = await request.json()
    pool, name, data = body.get("pool"), str(body.get("name", "")).strip(), body.get("data")
    if pool not in POOLS or not name or not isinstance(data, dict):
        return _error("a preset needs a pool, a name and data")
    return web.json_response(change(pool, name, data))


@PromptServer.instance.routes.delete("/promptforge/v2/prompt-presets")
async def _delete(request):
    pool = request.query.get("pool", "")
    return web.json_response(change(pool, request.query.get("name", ""))) if pool in POOLS else _error("unknown pool")
