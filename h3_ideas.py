"""Saved ideas of the H3 Control Panel: the texts you wrote before the prompt writer turns them into prompts.

They are kept in user/promptforge/ideas.json, so they survive workflows and browsers, and are listed by the panel."""
import json
import threading

from aiohttp import web
from server import PromptServer

from . import h3_actions, image_library

_lock = threading.Lock()


def _path():
    return image_library.store_dir() / "ideas.json"


def _read():
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(ideas):
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(ideas, indent=1, ensure_ascii=False), encoding="utf-8")


@PromptServer.instance.routes.get("/promptforge/ideas")
async def _list(request):
    return web.json_response(_read())


@PromptServer.instance.routes.post("/promptforge/ideas")
async def _save(request):
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "an idea needs a name"}, status=400)
    with _lock:
        ideas = _read()
        ideas[name] = body.get("data") or {}
        _write(ideas)
    return web.json_response(ideas)


@PromptServer.instance.routes.delete("/promptforge/ideas")
async def _delete(request):
    with _lock:
        ideas = _read()
        ideas.pop(request.query.get("name", ""), None)
        _write(ideas)
    return web.json_response(ideas)


def _library_path():
    return image_library.store_dir() / "prompt_library.json"


def _library_read():
    try:
        return json.loads(_library_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@PromptServer.instance.routes.get("/promptforge/prompts")
async def _prompts(request):
    return web.json_response(_library_read())


@PromptServer.instance.routes.post("/promptforge/prompts")
async def _prompts_save(request):
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "a prompt needs a name"}, status=400)
    data = _library_read()
    data[name] = str(body.get("text", ""))
    _library_path().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return web.json_response(data)


@PromptServer.instance.routes.delete("/promptforge/prompts")
async def _prompts_delete(request):
    data = _library_read()
    data.pop(request.query.get("name", ""), None)
    _library_path().write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return web.json_response(data)


@PromptServer.instance.routes.get("/promptforge/actions")
async def _actions(request):
    return web.json_response({label: {"roles": h3_actions.role_info(label), "aspect": h3_actions.aspect(label),
                                      "category": h3_actions.action_category(label), "cameras": h3_actions.cameras(label), "penetration": h3_actions.penetration_modes(label)}
                              for label in h3_actions.ACTIONS})
