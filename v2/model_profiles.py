"""Sampler settings per checkpoint: shipped profiles from the creators' recommendations, plus your own saved ones.

presets/model_profiles.json is shipped. presets/model_profiles_user.json (gitignored) holds what you save from the
model node and wins over the shipped entry. Only sampler fields are stored, so applying a profile never touches
prompts, LoRAs or anything else in the workflow.
"""
import json
import re
from pathlib import Path

from aiohttp import web
from server import PromptServer

DIR = Path(__file__).resolve().parent.parent / "presets"
SHIPPED = DIR / "model_profiles.json"
USER = DIR / "model_profiles_user.json"
FIELDS = ("steps", "cfg", "sampler_name", "scheduler")
H3_FIELDS = ("shift_video", "shift_audio", "refine_steps", "refine_cfg", "refine_sampler", "refine_scheduler", "start_sigma")
QUANT = re.compile(r"\b(int8|convrot|row|wise|nvfp4|fp8|fp16|bf16|fp32)\b")


def key(model):
    """One name per checkpoint whatever its folder, loader tag or quantisation (int8 / convrot / fp8 ...)."""
    name = re.sub(r"^\[[^\]]+\]\s*", "", str(model)).replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = re.sub(r"\.(safetensors|gguf|ckpt|pth)$", "", name)
    return " ".join(QUANT.sub(" ", re.sub(r"[^a-z0-9.]+", " ", name)).split())


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def profile(model):
    name = key(model)
    user = _read(USER).get(name)
    if user:
        return {**user, "user": True}
    shipped = _read(SHIPPED).get(name)
    return {**shipped, "user": False} if shipped else None


@PromptServer.instance.routes.get("/promptforge/v2/model-profile")
async def get_profile(request):
    return web.json_response({"key": key(request.query.get("model", "")), "profile": profile(request.query.get("model", ""))})


@PromptServer.instance.routes.post("/promptforge/v2/model-profile")
async def save_profile(request):
    body = await request.json()
    settings = {name: body["settings"][name] for name in FIELDS + H3_FIELDS if name in body.get("settings", {})}
    if not key(body.get("model", "")) or not settings:
        return web.json_response({"error": "A model and at least one sampler setting are required."}, status=400)
    stored = _read(USER)
    stored[key(body["model"])] = {"label": "saved by you", "settings": settings}
    USER.write_text(json.dumps(stored, indent=1, ensure_ascii=False), encoding="utf-8")
    return web.json_response({"profile": profile(body["model"])})


@PromptServer.instance.routes.delete("/promptforge/v2/model-profile")
async def delete_profile(request):
    stored = _read(USER)
    stored.pop(key(request.query.get("model", "")), None)
    USER.write_text(json.dumps(stored, indent=1, ensure_ascii=False), encoding="utf-8")
    return web.json_response({"profile": profile(request.query.get("model", ""))})
