"""V2 execution state. Configuration is copied; model/tensor artifacts are shared."""
from copy import deepcopy

from comfy_api.latest import io

PIPE = "PROMPTFORGE_V2_PIPE"
VERSION = 2

FAMILIES = {"IMAGE": "Image", "H3": "MiniMax H3", "KREA": "Krea 2", "ACE": "ACE-Step",
            "PROMPT": "Prompts", "SHARED": "Shared", "MEDIA": "Media", "PERFORMANCE": "Performance", "MASK": "Mask"}


def schema(family, key, title, inputs, outputs=None, output=False, **extra):
    """One node schema shape: keyed id, consistent display name and category."""
    label = FAMILIES.get(family, "Shared")
    suffix = f"_{key}" if key else ""
    return io.Schema(node_id=f"PROMPTFORGE_V2_{family}{suffix}", display_name=f"PromptForge V2 · {label} · {title}",
                     category=f"PromptForge V2/{label}", inputs=inputs,
                     outputs=outputs or [io.Custom(PIPE).Output()], is_output_node=output, **extra)


def new_pipe(family):
    return {"schema_version": VERSION, "family": family, "settings": {}, "artifacts": {}, "history": []}


def validate(pipe):
    if not isinstance(pipe, dict) or pipe.get("schema_version") != VERSION:
        raise ValueError("Connect a PromptForge V2 pipe; legacy pipes use a different contract.")
    for key, kind in (("family", str), ("settings", dict), ("artifacts", dict), ("history", list)):
        if not isinstance(pipe.get(key), kind):
            raise ValueError(f"Invalid V2 pipe field: {key}.")
    return pipe


def evolve(pipe, section=None, settings=None, artifacts=None, invalidate=()):
    validate(pipe)
    result = {**pipe, "settings": deepcopy(pipe["settings"]), "artifacts": dict(pipe["artifacts"]), "history": list(pipe["history"])}
    if settings is not None:
        if not section:
            raise ValueError("A settings update needs its owning section.")
        result["settings"][section] = deepcopy(settings)
    for name in invalidate:
        result["artifacts"].pop(name, None)
    if artifacts:
        result["artifacts"].update(artifacts)
    if section:
        result["history"].append(section)
    return result


_FAMILY_BY_MODEL = {"Krea2": "krea2", "MOSSImage": "moss_image", "Anima": "anima", "MiniMaxH3VideoVAE": "minimax_h3"}


def family_of_model(model):
    config = getattr(getattr(model, "model", None), "model_config", None)
    family = _FAMILY_BY_MODEL.get(type(config).__name__)
    if family is None:
        raise ValueError(f"Unsupported model family: {type(config).__name__}")
    return family


def ensure_pipe(pipe, family=None, model=None):
    """Standalone nodes start their own pipe; a connected pipe always wins.

    Nodes that do not care about the family pass none and get a neutral one; nodes that do
    pass the family or the model.
    """
    if pipe is not None:
        return validate(pipe)
    family = family or (family_of_model(model) if model is not None else "image")
    return new_pipe(family)


def section(pipe, name, values):
    """Store widget-provided settings, skipping the ones left at None."""
    return {key: value for key, value in values.items() if value is not None} if name else values


def take(pipe, **artifacts):
    """Merge real socket artifacts into the pipe; unconnected sockets arrive as None."""
    provided = {name: value for name, value in artifacts.items() if value is not None}
    return evolve(pipe, artifacts=provided) if provided else pipe


def require(pipe, *artifact_names):
    validate(pipe)
    missing = [name for name in artifact_names if pipe["artifacts"].get(name) is None]
    if missing:
        raise ValueError("Missing V2 artifacts: " + ", ".join(missing) + ". Connect the preceding preparation stage.")
    values = tuple(pipe["artifacts"][name] for name in artifact_names)
    return values[0] if len(values) == 1 else values


def metadata(pipe):
    """Only configuration belongs in JSON; artifacts can contain live models."""
    validate(pipe)
    return {"schema_version": VERSION, "family": pipe["family"], "settings": deepcopy(pipe["settings"]), "history": list(pipe["history"])}
