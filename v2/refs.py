"""One reference manager: ordered picture, video and voice lanes with names and
descriptions, filled from the node's own panel and handed to the writer/encoder.
"""
import json

from comfy_api.latest import io

from .contracts import schema, PIPE, ensure_pipe, evolve
from . import media

Pipe = io.Custom(PIPE)

KINDS = ("image", "video", "audio")
LANES = {"image": "Pictures", "video": "Videos", "audio": "Voices"}


def parse(items):
    try:
        data = json.loads(items or "[]")
    except ValueError as exc:
        raise ValueError("Reference list is not valid JSON.") from exc
    if not isinstance(data, list):
        raise ValueError("Reference list must be a list.")
    return [item for item in data if isinstance(item, dict)]


def _load(pipe, item):
    kind = item.get("kind")
    file = item.get("file") or ""
    if kind not in KINDS or not file:
        raise ValueError("Every reference needs a kind and a file.")
    if kind == "image":
        result = media.LoadImage.execute(pipe=pipe, filename=file, role="reference", enabled=True).args
        return result[0], {"kind": "image", "data": result[1]}, None
    if kind == "video":
        result = media.LoadVideo.execute(pipe=pipe, filename=file, role="reference", enabled=True).args
        return result[0], {"kind": "video", "data": result[1], "audio": result[2], "fps": result[3]}, None
    result = media.LoadAudio.execute(pipe=pipe, filename=file, enabled=True).args
    return result[0], {"kind": "audio", "data": result[1]}, None


class ReferenceManager(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("REFERENCES", "MANAGER", "Reference Manager",
                      description="Ordered picture, video and voice references, each with a name and an optional description. Row order sets the <Picture N>/<Video N>/<Audio N> numbering.",
                      inputs=[Pipe.Input("pipe", optional=True),
                              io.String.Input("items", default="[]", multiline=True, dynamic_prompts=True)],
                      outputs=[Pipe.Output()])

    @classmethod
    def fingerprint_inputs(cls, items="[]", **kwargs):
        stamps = []
        for item in parse(items):
            try:
                stamps.append(media.changed(item.get("file", "")))
            except ValueError:
                stamps.append("missing")
        return "|".join(stamps)
    @classmethod
    def execute(cls, pipe=None, items="[]"):
        pipe = ensure_pipe(pipe)
        entries, descriptions, numbers, audio_refs = [], [], {"image": 0, "video": 0, "audio": 0}, []
        for item in parse(items):
            if item.get("enabled") is False:
                continue
            pipe, entry, _ = _load(pipe, item)
            numbers[entry["kind"]] += 1
            name = item.get("name") or f"{entry['kind']} {numbers[entry['kind']]}"
            entry.update(name=name, description=item.get("description", ""))
            entries.append(entry)
            descriptions.append({"name": name, "kind": entry["kind"], "description": entry.get("description", ""),
                                 "index": numbers[entry["kind"]]})
            if entry["kind"] == "audio":
                audio_refs.append(entry["data"])
        images = [entry["data"] for entry in entries if entry["kind"] == "image"]
        videos = [entry for entry in entries if entry["kind"] == "video"]
        refs = dict(pipe["settings"].get("references", {}), selected_names=[entry["name"] for entry in entries],
                    descriptions=descriptions)
        artifacts = {"reference_images": images,
                     "reference_videos": [entry["data"] for entry in videos],
                     "reference_video_audios": [entry.get("audio") for entry in videos],
                     "reference_video_fps": [entry.get("fps", 24) for entry in videos],
                     "reference_audios": audio_refs}
        artifacts["reference_video"] = artifacts["reference_videos"][0] if videos else None
        artifacts["reference_video_audio"] = artifacts["reference_video_audios"][0] if videos else None
        artifacts["reference_video_fps"] = artifacts["reference_video_fps"][0] if videos else None
        artifacts["reference_audio"] = audio_refs[0] if audio_refs else None
        return io.NodeOutput(evolve(pipe, section="references", settings=refs, artifacts=artifacts))


NODES = [ReferenceManager]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
