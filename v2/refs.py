"""One reference manager: ordered picture, video and voice lanes with names and
descriptions, filled from the node's own panel and handed to the writer/encoder.
"""
import json

from comfy_api.latest import io

from .contracts import schema, PIPE, ensure_pipe, evolve
from . import media
from .h3 import DIRTY, grown, video_at_24

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


def _sockets(group):
    """Autogrow values in socket order."""
    return [group[key] for key in sorted(group, key=lambda key: int(key.rsplit("_", 1)[1]))] if group else []


class ReferenceManager(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("REFERENCES", "MANAGER", "Reference Manager",
                      description="Ordered picture, video and voice references, each with a name and an optional description. Row order sets the <Picture N>/<Video N>/<Audio N> numbering; media from sockets follows the rows.",
                      inputs=[Pipe.Input("pipe", optional=True),
                              io.String.Input("items", default="[]", multiline=True, dynamic_prompts=True),
                              io.String.Input("names", default="", tooltip="Comma-separated reference names to use this run; empty uses every enabled reference."),
                              io.Float.Input("extra_video_fps", default=24.0, min=0.01, max=1000, tooltip="Frame rate of the videos connected to extra_videos."),
                              grown(io.Image, "extra_images"), grown(io.Image, "extra_videos"), grown(io.Audio, "extra_audios")],
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
    def execute(cls, pipe=None, items="[]", names="", extra_video_fps=24.0, extra_images=None, extra_videos=None, extra_audios=None):
        pipe = ensure_pipe(pipe)
        entries = []
        for item in parse(items):
            if item.get("enabled") is False:
                continue
            pipe, entry, _ = _load(pipe, item)
            entries.append({**entry, "name": item.get("name") or "", "description": item.get("description", "")})
        for batch in _sockets(extra_images):
            entries += [{"kind": "image", "data": batch[index:index + 1], "name": "", "description": ""} for index in range(batch.shape[0])]
        entries += [{"kind": "video", "data": frames, "audio": None, "fps": extra_video_fps, "name": "", "description": ""} for frames in _sockets(extra_videos)]
        entries += [{"kind": "audio", "data": audio, "name": "", "description": ""} for audio in _sockets(extra_audios)]
        numbers = dict.fromkeys(KINDS, 0)
        for entry in entries:
            numbers[entry["kind"]] += 1
            entry["name"] = entry["name"] or f"{entry['kind']} {numbers[entry['kind']]}"
        wanted = [name.strip() for name in names.split(",") if name.strip()]
        if wanted:
            by_name = {entry["name"]: entry for entry in entries}
            missing = [name for name in wanted if name not in by_name]
            if missing:
                raise ValueError("Unknown references: " + ", ".join(missing))
            entries = [by_name[name] for name in wanted]
        numbers = dict.fromkeys(KINDS, 0)
        descriptions = []
        for entry in entries:
            numbers[entry["kind"]] += 1
            descriptions.append({"name": entry["name"], "kind": entry["kind"], "description": entry["description"], "index": numbers[entry["kind"]]})
        images = [entry["data"] for entry in entries if entry["kind"] == "image"]
        videos = [entry for entry in entries if entry["kind"] == "video"]
        audios = [entry["data"] for entry in entries if entry["kind"] == "audio"]
        clips = [video_at_24(entry["data"], entry.get("fps", 24)) for entry in videos]
        refs = dict(pipe["settings"].get("references", {}), selected_names=[entry["name"] for entry in entries], descriptions=descriptions)
        return io.NodeOutput(evolve(pipe, section="references", settings=refs, invalidate=DIRTY + ("reference_video", "reference_audio"), artifacts={
            "reference_images": images, "reference_videos": clips, "reference_video_audios": [entry.get("audio") for entry in videos],
            "reference_audios": audios, "reference_video": clips[0] if clips else None,
            "reference_video_audio": videos[0].get("audio") if videos else None,
            "reference_audio": audios[0] if audios else None, "reference_video_fps": 24}))


NODES = [ReferenceManager]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
