import json
import re
import time
from pathlib import Path

import folder_paths
from aiohttp import web
from comfy_api.latest import io
from server import PromptServer

from .. import lora_loader
from . import action_store
from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, metadata, require, take

Pipe = io.Custom(PIPE)
CATEGORY = "PromptForge V2/Shared"
DATE_TOKEN = re.compile(r"%date:([^%]+)%")
DATE_FORMATS = {"yyyy": "%Y", "MM": "%m", "dd": "%d", "HH": "%H", "mm": "%M", "ss": "%S"}


def expand_date_tokens(prefix, moment=None):
    """Expand %date:yyyy-MM-dd_HH-mm-ss% from one timestamp; core handles the plain %year% style tokens."""
    if "%date:" not in prefix:
        return prefix
    stamp = moment or time.localtime()

    def fill(match):
        pattern = match.group(1)
        for token, directive in DATE_FORMATS.items():
            pattern = pattern.replace(token, directive)
        return time.strftime(pattern, stamp)

    return DATE_TOKEN.sub(fill, prefix)


FILENAME_TOKENS = ("%family%", "%model%", "%mode%", "%source%", "%category%", "%action%", "%seed%")


def expand_tokens(prefix, pipe):
    """Date tokens plus the run's own facts, so a saved file says which model, mode, action and seed made it."""
    prefix = expand_date_tokens(prefix)
    if not any(token in prefix for token in FILENAME_TOKENS):
        return prefix
    settings = pipe["settings"]
    prompt = settings.get("prompt", {})
    family, _, label = (prompt.get("action", "") if prompt.get("source") == "action" else "").partition(": ")
    known = bool(label) and action_store.has(family, label)
    sampling = settings.get("sampling", {})
    h3 = pipe["family"] == "minimax_h3"
    artifacts = pipe["artifacts"]
    values = {
        "%family%": pipe["family"],
        "%model%": Path(re.sub(r"^\[[^\]]+\]\s*", "", str(settings.get("models", {}).get("model", "")))).stem,
        "%mode%": settings.get("references", {}).get("mode", "t2v") if h3 else ("edit" if artifacts.get("krea_references") or artifacts.get("reference_images") else "t2i"),
        "%source%": prompt.get("source", ""),
        "%category%": action_store.get(family, label)["category"] if known else prompt.get("source", ""),
        "%action%": label if known else prompt.get("source", ""),
        "%seed%": str(sampling.get("seed", sampling.get("first", {}).get("seed", ""))),
    }
    for token, value in values.items():
        prefix = prefix.replace(token, re.sub(r"[^A-Za-z0-9 ._()-]+", "_", str(value)).strip(" ._") or "none")
    return prefix


@PromptServer.instance.routes.get("/promptforge/v2/loras")
async def lora_library(request):
    if request.query.get("refresh") == "1":
        lora_loader._index.clear()
    return web.json_response(lora_loader.library())


class WorkflowPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("PRESETS", "", "Workflow Presets", description="Capture and restore upstream pipe-node settings before queueing. Execution passes the pipe unchanged.",
                         inputs=[Pipe.Input("pipe", optional=True), io.String.Input("preset_name", default=""),
                                 io.String.Input("presets", default='{"schema_version":2,"presets":{}}', multiline=True),
                                 io.Boolean.Input("include_text", default=False)], outputs=[Pipe.Output()])

    @classmethod
    def execute(cls, pipe=None, preset_name="", presets='{"schema_version":2,"presets":{}}', include_text=False):
        pipe = ensure_pipe(pipe)
        return io.NodeOutput(pipe)


class LoraStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("LORA", "STACK", "LoRA Stack", inputs=[Pipe.Input("pipe", optional=True),
                                 io.Model.Input("model", optional=True), io.Clip.Input("clip", optional=True),
                                 io.String.Input("loras", default='{"filter":"all","items":[]}', multiline=True),
                                 io.Combo.Input("layer_controls", options=["auto", "standard", "minimax audiovisual"], default="auto"),
                                 io.Int.Input("seed", default=0, min=0, max=2**32-1),
                                 io.Boolean.Input("skip_mismatched", default=True), io.Boolean.Input("balance", default=False),
                                 io.Float.Input("balance_floor", default=0.0, min=0.0, max=1.0, step=0.05)],
                         outputs=[Pipe.Output(), io.Model.Output("model"), io.Clip.Output("clip"),
                                  io.String.Output(display_name="triggers"), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, pipe=None, loras='{"filter":"all","items":[]}', layer_controls="auto", seed=0, skip_mismatched=True,
                balance=False, balance_floor=0.0, model=None, clip=None):
        pipe = take(ensure_pipe(pipe, model=model), model=model, clip=clip)
        model = require(pipe, "model")
        state = json.loads(loras)
        if not isinstance(state, dict) or not isinstance(state.get("items", []), list):
            raise ValueError("LoRA stack must contain an items list.")
        items = []
        for item in state["items"]:
            item = dict(item)
            item["on"] = item.pop("enabled", item.get("on", True))
            item["model"] = item.pop("model_strength", item.get("model", 1.0))
            item["clip"] = item.pop("clip_strength", item.get("clip"))
            items.append(item)
        state = {**state, "items": items}
        resolved_controls = ("minimax audiovisual" if pipe.get("family") == "minimax_h3" else "standard") if layer_controls == "auto" else layer_controls
        if resolved_controls == "standard":
            state = {**state, "items": [{**item, "part": "all"} for item in state["items"]]}
        model, clip, triggers, report = lora_loader.PROMPTFORGE_LoraLoader().load(
            model, json.dumps(state), seed, skip_mismatched, clip=pipe["artifacts"].get("clip"),
            balance=balance, balance_floor=balance_floor)
        updated = evolve(pipe, "loras", {"loras": state, "seed": seed, "layer_controls": layer_controls,
                                        "resolved_layer_controls": resolved_controls,
                                        "skip_mismatched": skip_mismatched, "balance": balance, "balance_floor": balance_floor},
                         {"model": model, "clip": clip, "lora_triggers": triggers, "lora_report": report},
                         invalidate=("positive", "negative", "latent", "images", "audio"))
        return io.NodeOutput(updated, model, clip, triggers, report)


class SingleLora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("SINGLE", "LORA", "Single LoRA", inputs=[Pipe.Input("pipe", optional=True),
                                 io.Model.Input("model", optional=True), io.Clip.Input("clip", optional=True),
                                 io.Combo.Input("lora_name", options=["(none)"] + folder_paths.get_filename_list("loras")),
                                 io.Boolean.Input("enabled", default=True), io.Float.Input("model_strength", default=1.0, min=-20, max=20, step=0.05),
                                 io.Float.Input("clip_strength", default=1.0, min=-20, max=20, step=0.05),
                                 io.Boolean.Input("mlp_only", default=False, optional=True, tooltip="Keep only the feed-forward layers of the LoRA. For a LoRA trained on a base model whose attention would damage a fine-tune."),
                                 io.Int.Input("max_rank", default=0, min=0, max=512, optional=True, tooltip="Cut the LoRA to this many strongest directions per layer. 0 keeps its own rank.")],
                         outputs=[Pipe.Output(), io.Model.Output("model"), io.Clip.Output("clip")])

    @classmethod
    def execute(cls, pipe=None, lora_name="(none)", enabled=True, model_strength=1.0, clip_strength=1.0, mlp_only=False, max_rank=0, model=None, clip=None):
        pipe = take(ensure_pipe(pipe, model=model), model=model, clip=clip)
        items = list(pipe["settings"].get("single_loras", {}).get("items", []))
        items.append({"name": lora_name, "enabled": enabled, "model": model_strength, "clip": clip_strength})
        if not enabled or lora_name == "(none)":
            result = evolve(pipe, "single_loras", {"items": items})
            return io.NodeOutput(result, result["artifacts"].get("model"), result["artifacts"].get("clip"))
        folder_paths.get_full_path_or_raise("loras", lora_name)
        patch = {**({"layers": "mlp"} if mlp_only else {}), **({"rank": max_rank} if max_rank else {})}
        state = json.dumps({"items": [{"name": lora_name, "on": True, "model": model_strength, "clip": clip_strength, **patch}]})
        result = LoraStack.execute(pipe, state, skip_mismatched=False).args[0]
        artifacts = {key: result["artifacts"][key] for key in ("model", "clip")}
        artifacts["lora_report"] = "\n".join(filter(None, (pipe["artifacts"].get("lora_report"), result["artifacts"].get("lora_report"))))
        artifacts["lora_triggers"] = ", ".join(filter(None, (pipe["artifacts"].get("lora_triggers"), result["artifacts"].get("lora_triggers"))))
        result = evolve(pipe, "single_loras", {"items": items}, artifacts,
                        invalidate=("positive", "negative", "latent", "images", "audio"))
        return io.NodeOutput(result, result["artifacts"].get("model"), result["artifacts"].get("clip"))


class ShowText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("SHOW", "TEXT", "Show Text", output=True,
                            inputs=[Pipe.Input("pipe", optional=True), io.String.Input("text", optional=True, force_input=True)],
                            outputs=[Pipe.Output()])

    @classmethod
    def execute(cls, pipe=None, text=None):
        pipe = ensure_pipe(pipe)
        value = text or pipe["artifacts"].get("text") or ""
        return io.NodeOutput(pipe, ui={"text": [value]})


class RunInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("RUN", "INSPECT", "Run Inspect", output=True, inputs=[Pipe.Input("pipe", optional=True)], outputs=[Pipe.Output(), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, pipe=None):
        pipe = ensure_pipe(pipe)
        pipe = ensure_pipe(pipe)
        report = json.dumps(metadata(pipe), ensure_ascii=False, indent=2)
        return io.NodeOutput(pipe, report, ui={"text": [report]})


class PromptLibrary(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("PROMPT", "LIBRARY", "Prompt Library", inputs=[Pipe.Input("pipe", optional=True), io.String.Input("entry_name", default=""),
                                 io.String.Input("text", default="", multiline=True),
                                 io.String.Input("entries", default="{}", multiline=True)],
                         outputs=[Pipe.Output(), io.String.Output("text")])

    @classmethod
    def execute(cls, pipe=None, entry_name="", text="", entries="{}"):
        pipe = ensure_pipe(pipe)
        return io.NodeOutput(pipe, text)


class SaveRunMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("SAVE", "METADATA", "Save Run Metadata", output=True, inputs=[Pipe.Input("pipe", optional=True), io.String.Input("filename_prefix", default="PromptForgeV2/run"),
                                                    io.Boolean.Input("include_text", default=False)],
                         outputs=[Pipe.Output(), io.String.Output("path")])

    @classmethod
    def execute(cls, pipe=None, filename_prefix="PromptForgeV2/run", include_text=False):
        pipe = ensure_pipe(pipe)
        data = metadata(pipe)
        if not include_text:
            excluded = {"text", "positive", "negative", "description", "notes", "subject", "subjects", "general_prompt"}
            def redact(value):
                if isinstance(value, dict):
                    return {key: redact(item) for key, item in value.items() if key not in excluded}
                if isinstance(value, list):
                    return [redact(item) for item in value]
                return value
            data = redact(data)
        root = Path(folder_paths.get_output_directory()).resolve()
        parent, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(root))
        target = (Path(parent) / f"{name}_{counter:05}_.json").resolve()
        if root not in target.parents:
            raise ValueError("Metadata filename must stay within the output directory.")
        with target.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return io.NodeOutput(pipe, str(target), ui={"text": [str(target)]})


NODES = [ShowText, WorkflowPresets, LoraStack, SingleLora, RunInspect, PromptLibrary, SaveRunMetadata]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
