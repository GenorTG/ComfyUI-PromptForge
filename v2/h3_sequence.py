"""Inspected multipart plans; execution delegates to the explicit H3 stages."""
import json
import math

from .prompting import H3PromptWriter

import comfy.model_management
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as core
from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require

Pipe = io.Custom(PIPE)
Writer = io.Custom("PROMPTFORGE_V2_WRITER_MODEL")


def schema(key, title, inputs, outputs=None):
    return nodes_schema("H3", key, title, inputs, outputs)


class H3Parts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("PARTS", "Parts", [Pipe.Input("pipe", optional=True),
            io.String.Input("parts_json", default='[{"text":"", "seconds":5}]', multiline=True),
            io.Combo.Input("seed_policy", options=["increment", "fixed", "explicit"], default="increment")],
            [Pipe.Output("pipe"), io.String.Output("plan")])

    @classmethod
    def execute(cls, pipe=None, parts_json='[{"text":"", "seconds":5}]', seed_policy="increment"):
        pipe = ensure_pipe(pipe, family="minimax_h3")
        from .h3 import checked
        checked(pipe)
        parts = json.loads(parts_json)
        if not isinstance(parts, list) or not parts:
            raise ValueError("Parts needs a nonempty JSON list of {text, seconds, seed?} objects.")
        plan = []
        prompt = pipe["settings"].get("prompt", {})
        fallback_text = (prompt.get("text") or "").strip()
        for index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise ValueError(f"Part {index + 1} must be an object.")
            text = (part.get("text") or "").strip() or fallback_text
            if not text:
                raise ValueError(f"Part {index + 1} needs explicit, prompt text or an idea/action brief.")
            part = {**part, "text": text}
            seconds = float(part.get("seconds", pipe["settings"].get("format", {}).get("seconds", 5)))
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Part {index + 1} needs a positive finite duration.")
            source = part.get("source", prompt.get("source", "raw"))
            if source not in ("raw", "idea", "action"):
                raise ValueError(f"Part {index + 1} source must be raw, idea or action.")
            item = {"text": text, "seconds": seconds, "source": source}
            if "seed" in part:
                item["seed"] = int(part["seed"]) % 2**64
            if seed_policy == "explicit" and "seed" not in item:
                raise ValueError(f"Part {index + 1} needs an explicit seed.")
            plan.append(item)
        result = evolve(pipe, section="parts", settings={"plan": plan, "seed_policy": seed_policy},
                        invalidate=("sequence_parts", "sequence_next", "sequence_report"))
        return io.NodeOutput(result, json.dumps(plan, ensure_ascii=False, indent=2))


class H3Continuity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("CONTINUITY", "Continuity", [Pipe.Input("pipe", optional=True),
            io.Combo.Input("mode", options=["independent", "last_frame", "continue"], default="continue"),
            io.Int.Input("carried_frames", default=22, min=1, max=3600),
            io.Combo.Input("transition", options=["cut", "crossfade"], default="cut"),
            io.Int.Input("overlap_frames", default=0, min=0, max=3600)])

    @classmethod
    def execute(cls, pipe=None, mode="continue", carried_frames=22, transition="cut", overlap_frames=0):
        pipe = ensure_pipe(pipe, family="minimax_h3")
        from .h3 import checked
        checked(pipe)
        return io.NodeOutput(evolve(pipe, section="continuity", settings={"mode": mode,
            "carried_frames": carried_frames, "transition": transition, "overlap_frames": overlap_frames}))


class H3Sequence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("SEQUENCE", "Sequence", [Pipe.Input("pipe", optional=True),
            io.Int.Input("max_parts", default=0, min=0, max=10000, tooltip="0 runs all remaining parts; a limit produces a resumable pipe."),
            Pipe.Input("resume", optional=True), Writer.Input("writer_model", optional=True, lazy=True)], [Pipe.Output("pipe"), io.Image.Output("images"),
            io.Audio.Output("audio"), io.String.Output("report")])

    @classmethod
    def check_lazy_status(cls, pipe, writer_model=None, resume=None, **kwargs):
        start = len(resume["artifacts"].get("sequence_parts", [])) if resume is not None else 0
        parts = pipe["settings"]["parts"]["plan"][start:]
        if kwargs.get("max_parts", 0):
            parts = parts[:kwargs["max_parts"]]
        return ["writer_model"] if writer_model is None and any(p.get("source", "raw") != "raw" for p in parts) else []

    @classmethod
    def execute(cls, pipe=None, max_parts=0, resume=None, writer_model=None):
        pipe = ensure_pipe(pipe, family="minimax_h3")
        if not pipe["settings"].get("parts"):
            raise ValueError("Connect Parts before Sequence.")
        from . import h3
        h3.checked(pipe)
        plan = pipe["settings"]["parts"]["plan"]
        policy = pipe["settings"]["parts"]["seed_policy"]
        sampling = pipe["settings"]["sampling"]
        continuity = pipe["settings"].get("continuity", {})
        fmt = pipe["settings"]["format"]
        if resume is not None:
            h3.checked(resume)
            if resume["artifacts"].get("sequence_format") != fmt:
                raise ValueError("Resume pipe has different source format settings.")
            if any(resume["settings"].get(key) != pipe["settings"].get(key) for key in ("parts", "models", "sampling", "continuity", "upscale")):
                raise ValueError("Resume pipe has different plan, models, sampling or continuity settings.")
        clips = list(resume["artifacts"].get("sequence_parts", [])) if resume is not None else []
        assembled = resume if clips else None
        report = list(resume["artifacts"].get("sequence_report", [])) if clips else []
        start = len(clips)
        end = min(len(plan), start + max_parts) if max_parts else len(plan)
        for index in range(start, end):
            comfy.model_management.throw_exception_if_processing_interrupted()
            part = plan[index]
            frames = core.align_frame_count(max(5, math.ceil(part["seconds"] * 24)))
            working = evolve(pipe, section="format", settings=dict(fmt, frames=frames, seconds=frames / 24),
                             artifacts={"text": part["text"]}, invalidate=h3.DIRTY)
            previous = clips[-1] if clips else None
            if previous is not None and continuity.get("mode", "continue") != "independent":
                refs = dict(working["settings"].get("references", {}))
                if continuity.get("mode", "continue") == "last_frame":
                    refs["mode"] = "i2v"
                    artifacts = {"first_frame": require(previous, "images")[-1:].clone()}
                else:
                    refs["mode"] = "continue"
                    artifacts = {"reference_video": require(previous, "images"), "reference_audio": require(previous, "audio"), "reference_video_fps": 24, "continue_fps": 24}
                working = evolve(working, section="references", settings=refs, artifacts=artifacts)
            seed = part.get("seed", (sampling["first"]["seed"] + (index if policy == "increment" else 0)) % 2**64)
            prompt = dict(pipe["settings"].get("prompt", {}), text=part["text"], source=part.get("source", "raw"))
            working = evolve(working, section="prompt", settings=prompt, invalidate=("writer",))
            if prompt["source"] == "raw":
                from .prompting import _h3_sections
                working = evolve(working, artifacts={"text": _h3_sections(
                    working, part["text"], prompt.get("soundscape", ""), prompt.get("music", "N/A")),
                    "writer": {"status": "raw", "attempts": [], "validation": []}})
            else:
                # Writer vision uses the last frame; the full AV tail remains on the generation pipe.
                writer_pipe = working
                if working["settings"].get("references", {}).get("mode") == "continue":
                    source = require(working, "reference_video")
                    writer_pipe = evolve(working, section="references",
                        settings=dict(working["settings"]["references"], mode="i2v"),
                        artifacts={"first_frame": source[-1:].clone(), "reference_images": []},
                        invalidate=("last_frame", "reference_video", "reference_audio"))
                descriptions = [str(item.get("description", "")) for item in working["settings"].get("media", {}).values() if isinstance(item, dict) and item.get("description")]
                descriptions += [item["description"] for item in working["settings"].get("references", {}).get("descriptions", []) if item.get("description")]
                context = "\nReference descriptions:\n" + "\n".join(descriptions) if descriptions else ""
                if previous is not None:
                    context += "\nPrevious part, for story continuity:\n" + require(previous, "text")
                writer_pipe = evolve(writer_pipe, section="prompt", settings=dict(prompt, text=prompt["text"] + context))
                writer_pipe = H3PromptWriter.execute(writer_pipe, writer_model=writer_model, seed=seed % 2**32).args[0]
                working = evolve(working, artifacts={"text": require(writer_pipe, "text"),
                    "writer": writer_pipe["artifacts"]["writer"]})
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
            report.append(f"Part {index + 1}: writer {prompt['source']}")
            working = h3.encode(working)
            # the text encoder is the biggest resident model; drop it before sampling/decode
            working = evolve(working, invalidate=("clip",))
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(force=True)
            working = h3.sample(working, **dict(sampling["first"], seed=seed))
            upscale = pipe["settings"].get("upscale")
            if upscale and upscale.get("enabled", True):
                working = h3.H3LatentUpscale.execute(working).args[0]
            if sampling.get("refine_enabled", False):
                working = h3.sample(working, **dict(sampling["refine"], seed=(seed + 7) % 2**64))
            working = h3.decode(working, **pipe["settings"].get("decode", {}))
            overlap = working["artifacts"].get("continuation_overlap", continuity.get("overlap_frames", 0))
            assembled = h3.join(assembled, working, overlap, continuity.get("transition", "cut")) if assembled is not None else working
            # Only decoded clips are retained for resume and continuity, not per-part model/latent copies.
            # a retained clip keeps no model patchers: their refcount would pin staged weights in VRAM
            working = evolve(working, invalidate=("model", "clip", "vae", "audio_vae", "decode_vae",
                                                  "latent", "positive", "negative"))
            # decoded clips live on the CPU: two of them on the card OOM a 16 GB GPU
            clips.append({**working, "artifacts": {
                "images": working["artifacts"]["images"].to("cpu"), "audio": working["artifacts"]["audio"],
                "text": working["artifacts"]["text"], "writer": working["artifacts"]["writer"]}})
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            report.append(f"Part {index + 1}/{len(plan)}: seed {seed}, {frames} frames, decoded")
        if assembled is None:
            raise ValueError("Sequence has no completed parts.")
        assembled = evolve(assembled, section="sampling", settings=sampling)
        result = evolve(assembled, artifacts={"sequence_parts": clips, "sequence_next": len(clips), "sequence_report": report, "sequence_format": dict(fmt)})
        return io.NodeOutput(result, *require(result, "images", "audio"), "\n".join(report))
