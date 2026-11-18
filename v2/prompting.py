"""V2 intent, local writer adapters and inspectable positive prompt output."""
import importlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import gc

import folder_paths
from aiohttp import web
from server import PromptServer

import comfy.model_management
import numpy as np
import torch
from PIL import Image, ImageDraw

from comfy_api.latest import io, ui

from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require


Pipe = io.Custom(PIPE)
Writer = io.Custom("PROMPTFORGE_V2_WRITER_MODEL")
INVALIDATE = ("text", "writer", "positive", "negative", "latent", "samples", "images", "video", "audio")


def _services():
    # The package owns the existing writer services; importing it here avoids its node import cycle.
    return importlib.import_module(__package__.rsplit(".", 1)[0])


def _library(name):
    return importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.{name}")


_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]*\'(?!\w)')
_QUOTE_CONTEXT = re.compile(
    r'(?:says?|speaks?|whispers?|shouts?|replies|asks?|reads?|reading|written|'
    r'dialogue|lyrics?|sign|caption|title|named|called|text|sings?)\b[^.!?\n]{0,70}$', re.I)
_NEGATIVE = re.compile(
    r"\b(?:no|not|never|without|avoid(?:ing)?|exclude|excluding|omit|remove|"
    r"do\s+not|don['’]t|mustn['’]t|shouldn['’]t|can['’]t|cannot|won['’]t)\b|"
    r"\b(?:free\s+of|negative\s+prompt|negative\s+constraints?)\b|\b\w+[- ]free\b", re.I)
_REASONING = re.compile(
    r"</?(?:think|analysis|reasoning)>|<\|(?:analysis|assistant|im_start)\|>|"
    r"(?:^|\n)\s*(?:analysis|reasoning|final answer|final prompt|here(?:'s| is) (?:the|your) prompt)\s*:|"
    r"\b(?:I (?:need to|should|will|must)|let me (?:think|analy[sz]e)|"
    r"the user (?:wants|asked|requests)|we need to|I cannot (?:help|assist))\b", re.I)


def validation_issues(text, family=None):
    """Lint descriptive text, preserving quoted in-scene language and H3 sentinels."""
    if not isinstance(text, str) or not text.strip():
        return ["empty positive prompt"]
    masked = text
    for match in reversed(list(_QUOTED.finditer(text))):
        context = text[max(0, match.start() - 100):match.start()]
        if _QUOTE_CONTEXT.search(context):
            masked = masked[:match.start()] + ' ' * len(match.group()) + masked[match.end():]
    masked = re.sub(r'(?im)^\s*(?:non_diegetic_music|overall_soundscape)\s*:\s*(?:N/A|none)\s*$', '', masked)
    issues = []
    if _REASONING.search(masked):
        issues.append("reasoning or instruction leakage")
    negative_sentences = [s for s in re.split(r"(?<=[.!?])\s+", masked) if _NEGATIVE.search(s)]
    words = len(re.findall(r"\w+", masked))
    allowance = max(6, words // 50) if family == "minimax_h3" else max(1, words // 100)
    if len(negative_sentences) > allowance:
        issues.append(f"negative wording in positive description ({len(negative_sentences)} sentences)")
    words_lower = re.findall(r'\w+', masked.lower())
    if re.search(r'(\S)\1{15,}', masked) or (len(words_lower) > 30 and Counter(words_lower).most_common(1)[0][1] / len(words_lower) > .45):
        issues.append("repetitive output")
    lines = [line.strip() for line in masked.splitlines() if line.strip()]
    if lines and Counter(lines).most_common(1)[0][1] >= 4:
        issues.append("repeated lines")
    if re.search(r"\b(?:as an AI|I'm sorry|I am sorry|I refuse|unable to (?:help|assist))\b", masked, re.I):
        issues.append("refusal instead of prompt")
    return issues


def _strip_context_echo(text):
    """Writers sometimes repeat the request's JSON context; it is not part of the prompt."""
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text
    try:
        data, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return text
    if not isinstance(data, dict) or "family" not in data:
        return text
    return stripped[end:].lstrip("\n ") or text


ANIMA_PREFIX = "masterpiece, best quality, score_7, nsfw, anime screencap, 2d, flat colors"


def _prompt(pipe):
    return pipe["settings"].get("prompt", {})


def _store_prompt(pipe, source, text, negative, **extra):
    if source not in ("raw", "idea", "action"):
        raise ValueError(f"Unknown prompt source: {source}")
    settings = {"source": source, "text": text, "negative": negative, **extra}
    artifacts = {}
    if source == "raw":
        artifacts = {"text": text, "writer": {"status": "raw", "attempts": [], "validation": validation_issues(text, pipe["family"])}}
    return evolve(pipe, section="prompt", settings=settings, artifacts=artifacts, invalidate=INVALIDATE)


def _action_options():
    images = _library("image_actions")
    return [f"image: {key}" for key in images.ACTIONS] + [f"h3: {key}" for key in _library("h3_actions").ACTIONS]


def _action_genders(label, count, h3):
    if h3:
        return [role["gender"] for role in _library("h3_actions").role_info(label)]
    # image briefs always call subject 1 the woman and subject 2 the man
    return ["female", "male"] if count == 2 else ["female"]


def _action_catalog():
    """The picker's data: label, one-line description and metadata, split by generation."""
    catalog = {"image": {}, "minimax": {}}
    for family, library, h3 in (("image", "image_actions", False), ("minimax", "h3_actions", True)):
        for label, entry in _library(library).ACTIONS.items():
            count, arrangement = (len(entry[0]), entry[1]) if h3 else (entry[0], entry[2])
            description = re.split(r"(?<=[.!?])\s", " ".join(arrangement.split()))[0]
            catalog[family][label] = {
                "category": label.split(" - ")[0],
                "description": description if len(description) <= 180 else description[:177].rsplit(" ", 1)[0] + "…",
                "gender": _action_genders(label, count, h3),
                "aspect": _library(library).aspect(label) if h3 else ""}
    return catalog


@PromptServer.instance.routes.get("/promptforge/v2/actions")
async def _actions(request):
    return web.json_response(_action_catalog())


def _ref_subjects(pipe):
    """Bind every picture to a subject: a cock picture belongs to the man above it, never a third body."""
    described = pipe["settings"].get("references", {}).get("descriptions") or []
    pictures = [entry for entry in described if entry.get("kind") == "image"]
    if not pictures:
        return "subject_definitions:\n<Subject 1> is the referenced person."
    lines, index = [], 0
    for entry in pictures:
        detail = (entry.get("description") or "").strip()
        number = entry.get("index")
        if re.search(r"\b(penis|cock|shaft|dick)\b", detail, re.I):
            if lines:
                lines[-1] += (f" The shape, size and appearance of his erect penis come from <Picture {number}> "
                              "and it stays attached at his pelvis, unchanged.")
            else:
                lines.append(f"<Picture {number}> is his cock: anatomy reference, not a third body.")
            continue
        index += 1
        name = entry.get("name") or f"reference {number}"
        line = f"<Subject {index}> is the person in <Picture {number}> ({name})"
        line += f": {detail}. " if detail else ". "
        line += f"Use <Picture {number}> only for face, hair and body."
        lines.append(line)
    return "subject_definitions:\n" + "\n".join(lines)


def _ref_retention(pipe):
    described = pipe["settings"].get("references", {}).get("descriptions") or []
    lines, index = [], 0
    for entry in described:
        if entry.get("kind") != "image":
            continue
        detail = entry.get("description") or ""
        if re.search(r"\b(penis|cock|shaft|dick)\b", detail, re.I):
            lines.append(f"<Picture {entry.get('index')}> (penis appearance): fully_preserved - shape and attachment retained.")
            continue
        index += 1
        lines.append(f"<Subject {index}>: fully_preserved - identity, hair and body kept.")
    return "\n".join(lines) or "<Subject 1>: fully_preserved."


def _h3_sections(pipe, description, soundscape, music):
    """Wrap a raw H3 prompt in the section headings the mode expects, so the syntax is always valid."""
    mode = pipe["settings"].get("references", {}).get("mode", "t2v")
    mode_key = {"t2v": "t2va", "i2v": "i2va", "l2v": "l2va", "fl2v": "fl2va",
                "ref2v": "ref2va", "continue": "ref2va"}.get(str(mode).lower().split(" - ")[0], "t2va")
    description = (description or "").strip() or "[Shot 1] the scene plays out."
    if "[Shot 1]" not in description:
        description = "[Shot 1] " + description
    block = ""
    if mode_key == "ref2va":
        block += _ref_subjects(pipe) + "\n\n"
        block += "summary:\nThe clip shows the requested action.\n\n"
        block += "retention_analysis:\n" + _ref_retention(pipe) + "\n\n"
        block += "detailed_description:\n" + description + "\n\n"
    else:
        block += "integrated_multimodal_description:\n" + description + "\n\n"
    block += "overall_soundscape:\n" + ((soundscape or "").strip() or "soft room tone, breathing and quiet moans") + "\n\n"
    block += "non_diegetic_music:\n" + ((music or "").strip() or "N/A")
    return block


def _action_text(action, subjects, notes, camera):
    kind, label = action.split(": ", 1)
    if kind == "h3":
        library = _library("h3_actions")
        return library.brief(label, library.ANY_CAMERA if camera == "auto" else camera, subjects, notes)
    if kind != "image":
        raise ValueError(f"Unknown action library: {kind}")
    return _library("image_actions").brief(label, subjects[:2], camera, "auto", "auto", "as in the image", "auto", notes, "", "text description", "")


class WriterModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        service = _services()
        return nodes_schema(
            "WRITER_MODEL", "", "Writer Model",
            inputs=[io.Combo.Input("model", options=service.MODEL_OPTIONS, default=service._DEFAULT_MODEL),
                    io.Combo.Input("quantization", options=service.QUANT_OPTIONS, default="auto"),
                    io.Int.Input("context_size", default=0, min=0, max=262144),
                    io.String.Input("local_model_path", default=""), io.String.Input("mmproj_path", default=""),
                    io.Clip.Input("clip", optional=True)],
            outputs=[Writer.Output("writer_model"), io.String.Output("report")])

    @classmethod
    def execute(cls, model, quantization="auto", context_size=0, local_model_path="", mmproj_path="", clip=None):
        config, report = _services().PROMPTFORGE_Model().build(
            model, quantization, context_size, local_model_path, mmproj_path, clip=clip)
        return io.NodeOutput(config, report)


class ImagePrompt(io.ComfyNode):
    H3 = False
    GENERATION = "image"

    @classmethod
    def define_schema(cls):
        return nodes_schema("H3" if cls.H3 else "IMAGE", "PROMPT", "H3 Prompt" if cls.H3 else "Image Prompt", inputs=[Pipe.Input("pipe", optional=True), io.Combo.Input("source", options=["raw", "idea", "action"], default="raw"),
                    io.String.Input("text", default="", multiline=True), io.String.Input("text_in", optional=True, force_input=True),
                    io.String.Input("negative", default="", multiline=True),
                    io.Combo.Input("action", options=_action_options()),
                    io.String.Input("subject_1", default="", multiline=True),
                    io.String.Input("subject_2", default="", multiline=True),
                    io.String.Input("subject_3", default="", multiline=True),
                    io.Combo.Input("camera", options=["auto"] + sorted(_services()._CAMERAS(cls.H3)), default="auto"),
                    io.String.Input("notes", default="", multiline=True),
                    io.String.Input("soundscape", default="", multiline=True, tooltip="Minimax generation only, raw mode: overall_soundscape line (room tone, dialogue, effects)."),
                    io.String.Input("music", default="N/A", tooltip="Minimax generation only, raw mode: non_diegetic_music. N/A when there is no score."),
                    # last, so workflows saved before it existed keep their value positions; "" is the old empty slot
                    io.Combo.Input("generation", options=["image", "minimax", ""], default=cls.GENERATION)],
            outputs=[Pipe.Output("pipe"), io.String.Output("text")])

    @classmethod
    def execute(cls, pipe=None, source="raw", text="", negative="", action="", subject_1="", subject_2="", subject_3="",
                camera="auto", notes="", text_in=None, soundscape="", music="N/A", generation=None):
        pipe = ensure_pipe(pipe, family="minimax_h3" if cls.H3 else "image")
        audio = (generation or cls.GENERATION) == "minimax"
        if text_in:
            text = text_in
        if cls.H3 and pipe["family"] != "minimax_h3":
            raise ValueError("H3 Prompt requires a minimax_h3 pipe.")
        if source == "action":
            text = _action_text(action, (subject_1, subject_2, subject_3), notes, camera) + (f"\nAdditional intent: {text}" if text else "")
            if audio and soundscape.strip():
                text += f"\nSound direction: {soundscape.strip()}"
            if audio and music.strip() and music.strip() != "N/A":
                text += f"\nMusic direction: {music.strip()}"
        elif cls.H3 and source == "raw":
            text = _h3_sections(pipe, text, soundscape if audio else "", music if audio else "")
        result = _store_prompt(pipe, source, text, negative, action=action if source == "action" else "",
                               **({"soundscape": soundscape, "music": music} if audio else {}))
        return io.NodeOutput(result, text)


class H3Prompt(ImagePrompt):
    H3 = True
    GENERATION = "minimax"


H3_PRESETS = {"t2va": "MiniMax H3 T2VA", "i2va": "MiniMax H3 I2VA",
              "l2va": "MiniMax H3 L2VA", "fl2va": "MiniMax H3 FL2VA", "ref2va": "MiniMax H3 Ref2VA"}


def _preset_family(preset):
    """Family a named template belongs to; the shared style presets belong to none."""
    if preset in H3_PRESETS.values():
        return "minimax_h3"
    for family, names in _services().FAMILY_PRESETS.items():
        if preset in names:
            return family
    return None


def _contact_sheet(images, mapping):
    columns = math.ceil(math.sqrt(len(images)))
    cell = 384
    sheet = Image.new("RGB", (columns * cell, math.ceil(len(images) / columns) * (cell + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, tensor in enumerate(images):
        pixels = tensor.detach().float().cpu().clamp(0, 1).numpy()
        picture = Image.fromarray((pixels[..., :3] * 255).astype(np.uint8)).convert("RGB")
        picture.thumbnail((cell, cell))
        x, y = (index % columns) * cell, (index // columns) * (cell + 24)
        sheet.paste(picture, (x + (cell - picture.width) // 2, y + 24))
        draw.text((x + 4, y + 4), mapping[index].split(":", 1)[0], fill="black")
        mapping[index] += f"; contact sheet row {index // columns + 1}, column {index % columns + 1}"
    return torch.from_numpy(np.asarray(sheet).copy()).float().div(255).unsqueeze(0)


def _writer_context(pipe, h3):
    artifacts = pipe["artifacts"]
    references = pipe["settings"].get("references", {})
    fmt = pipe["settings"].get("format", {})
    mode = references.get("mode", "t2v" if h3 else "text_to_image")
    mode_key = str(mode).lower().split(" - ")[0]
    mode_key = {"t2v": "t2va", "i2v": "i2va", "l2v": "l2va", "fl2v": "fl2va",
                "ref2v": "ref2va", "continue": "ref2va", "continuation": "ref2va"}.get(mode_key, mode_key)
    if h3 and mode_key not in H3_PRESETS:
        raise ValueError(f"Unsupported H3 reference mode: {mode}")
    if h3:
        if mode_key == "ref2va":
            inputs = [(f"reference_images[{i}]", image) for i, image in enumerate(artifacts.get("reference_images", []))]
        else:
            roles = {"i2va": ("first_frame",), "l2va": ("last_frame",), "fl2va": ("first_frame", "last_frame")}.get(mode_key, ())
            inputs = [(key, artifacts[key]) for key in roles if artifacts.get(key) is not None]
    else:
        key = "krea_references" if artifacts.get("krea_references") else "reference_images"
        inputs = [(f"{key}[{i}]", image) for i, image in enumerate(artifacts.get(key, []))]
        if not inputs and artifacts.get("source_image") is not None:
            inputs = [("source_image", artifacts["source_image"])]
    images, mapping = [], []
    for name, tensor in inputs:
        frames = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        if h3:
            frames = frames[:1]
        for index, frame in enumerate(frames):
            images.append(frame)
            mapping.append(f"Picture {len(images)}: {name}, image {index + 1}")
    videos = list(artifacts.get("reference_videos") or [])
    video = artifacts.get("reference_video")
    if not videos and video is not None:
        videos = [video]
    if video is None and videos:
        video = videos[0]
    picture_count = len(images)
    # Extra clips are represented by sampled frames, retaining their own video labels.
    for index, frames in enumerate(videos[1:], 2):
        for frame_index in sorted({round(i * (len(frames) - 1) / 3) for i in range(4)}):
            images.append(frames[frame_index])
            mapping.append(f"Video {index}: frame {frame_index}, visual context only")
    contact_sheet = len(images) > 3
    writer_images = [_contact_sheet(images, mapping)] if contact_sheet else [image.unsqueeze(0) for image in images]
    audios = list(artifacts.get("reference_audios") or [])
    audio = artifacts.get("reference_audio")
    if not audios and audio is not None:
        audios = [audio]
    audio_context = []
    for index, value in enumerate(audios, 1):
        waveform = value.get("waveform") if isinstance(value, dict) else None
        sample_rate = value.get("sample_rate") if isinstance(value, dict) else None
        audio_context.append({"label": f"Audio {index}", "sample_rate": sample_rate,
                              "duration": waveform.shape[-1] / sample_rate if waveform is not None and sample_rate else None,
                              "semantics": "unavailable; use supplied user descriptions"})
    width, height = fmt.get("width"), fmt.get("height")
    orientation = fmt.get("orientation") or ("landscape" if width > height else "portrait" if height > width else "square") if width and height else fmt.get("orientation")
    context = {"family": pipe["family"], "mode": mode, "writer_mode": mode_key,
               "orientation": orientation, "width": width, "height": height,
               "duration": fmt.get("duration", fmt.get("seconds")), "fps": fmt.get("fps"),
               "references": mapping + [f"Video {i + 1}: reference video" for i in range(len(videos))],
               "picture_count": picture_count, "video_count": len(videos), "contact_sheet": contact_sheet,
               "audio": audio_context, "descriptions": references.get("descriptions", []),
               "audio_semantics": "unavailable" if audios else "no audio reference",
               "continuity": pipe["settings"].get("continuity", {}),
               "media": pipe["settings"].get("media", {})}
    return writer_images, video, mode_key, context


def _write(encoder, service, config, system, text, images, video, seed, max_tokens, temperature, thinking=False,
           top_p=.9, top_k=40, repetition_penalty=1.05, presence_penalty=.4):
    source = config.get("source", "safetensors")
    model_name = config.get("gguf_model", "auto") if source == "GGUF" else config.get("safetensors_model", "auto")
    # a resident video model starves the writer: the 9 GB encoder then streams from RAM at single-digit tok/s
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    result = encoder._generate(
        full_prompt=system + "\n\n" + text, system_prompt=system, user_prompt=text,
        mode="inputs" if images or video is not None else "text_only", engine="gguf" if source == "GGUF" else "text_encoder",
        model_name=model_name, quantization=config.get("quantization", "auto"),
        context_size=config.get("context_size", 16384), gpu_layers=config.get("gpu_layers", -1),
        batch_size=config.get("batch_size", 2048), ubatch_size=config.get("ubatch_size", 512),
        kv_cache_type=config.get("kv_cache_type", "f16"), offload_kqv=config.get("offload_kqv", True),
        swa_full=config.get("swa_full", False), flash_attn=config.get("flash_attn", "auto"),
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=1,
        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
        image=images[0] if images else None, image2=images[1] if len(images) > 1 else None,
        image3=images[2] if len(images) > 2 else None, video=video, frame_count=16, seed=seed,
        clip=config.get("clip"), local_model_path=config.get("local_model_path", ""),
        mmproj_path=config.get("mmproj_path", ""), disable_thinking=not thinking, auto_retry=False)
    answer, thinking = service._split_thinking(result)
    return service._clean_prompt_output(answer), thinking


def _h3_headings(text, mode):
    """One heading per line, colon attached, content underneath: repair what the writer returns."""
    for section in _services()._MINIMAX_SECTIONS[mode]:
        text = re.sub(rf"(?<=[^\n])({section}:)", r"\n\n\1", text)          # heading written mid-line
        text = re.sub(rf"(?m)^({section}):?[ \t]*:?[ \t]*(?=\S)", rf"\1:\n", text)  # content glued to it
        text = re.sub(rf"(?m)^({section}):?[ \t]*:?[ \t]*$", r"\1:", text)     # bare heading, double colon
    text = re.sub(r"(?m)^:[ \t]*$\n?", "", text)
    return re.sub(r"(?<!\[)\bShot (\d+)\b(?!\])", r"[Shot \1]", text)


def _h3_repair_audio(text, pipe, mode):
    """Append the sound sections when the writer spends its budget on the description."""
    sections = _services()._MINIMAX_SECTIONS[mode]
    prompt = _prompt(pipe)
    if "overall_soundscape" in sections and not re.search(r"(?m)^overall_soundscape:", text):
        text += "\n\noverall_soundscape:\n" + (prompt.get("soundscape") or "").strip().replace("\n", " ")
        if text.rstrip().endswith("overall_soundscape:"):
            text += " soft room tone, breathing and quiet moans"
    if "non_diegetic_music" in sections and not re.search(r"(?m)^non_diegetic_music:", text):
        text += "\n\nnon_diegetic_music:\n" + (prompt.get("music") or "").strip() or "N/A"
    return text


def _h3_issues(text, mode, context):
    sections = _services()._MINIMAX_SECTIONS[mode]
    positions = []
    for section in sections:
        match = re.search(rf"^{section}:\s*", text, re.M | re.I)
        if match is None:
            return [f"missing section {section}"]
        positions.append(match.start())
    if positions != sorted(positions):
        return ["sections out of order"]
    shots = [int(number) for number in re.findall(r"\[Shot (\d+)\]", text)]
    issues = [] if shots == list(range(1, len(shots) + 1)) and shots else ["shot labels must start at [Shot 1] and increase consecutively"]
    for label, count in (("Picture", context["picture_count"]), ("Video", context["video_count"]), ("Audio", len(context["audio"]))):
        if any(int(number) < 1 or int(number) > count for number in re.findall(rf"<{label}\s+(\d+)>", text)):
            issues.append(f"unknown {label} reference label")
    return issues


def _instructions(pipe, mode, context, max_words, style_prefix="", sections=None):
    family = pipe["family"]
    instruction = (
        "Write the final generation prompt from the user's intent. Every sentence must carry a positional or visible "
        "fact: who or what is where, in what direction, at what height. Describe visible desired outcomes affirmatively. "
        "Convert exclusions into positive visible facts. Keep the brief's camera, angle, framing and stacking exactly; "
        "never swap them for a closer crop. In a couple act the person on top sits between the camera and the person "
        "below. Keep every position, contact and camera fact from the brief; drop ornamental words. Do not invent limbs. ")
    if family == "minimax_h3":
        sections = sections or _services()._MINIMAX_SECTIONS[mode]
        labels = [f"<{name} 1>" + (f" to <{name} {count}>" if count > 1 else "")
                  for name, count in (("Picture", context["picture_count"]), ("Video", context["video_count"]),
                                      ("Audio", len(context["audio"]))) if count]
        guide = {"integrated_multimodal_description": "the shot description, 950-1250 words",
                 "detailed_description": "the shot description, 950-1250 words",
                 "subject_definitions": "one line per subject, 40-80 words in total",
                 "summary": "two or three sentences",
                 "retention_analysis": "one line per subject",
                 "overall_soundscape": "90-150 words of sound only",
                 "non_diegetic_music": "one line, or N/A"}
        instruction += (
            f"Use at most {max_words} words in total. Exact layout, with nothing before the first heading and nothing "
            "after the last one: "
            + "; ".join(f"{name}: {guide[name]}" for name in sections)
            + ". One heading per line, trailing colon, content directly under it, no other headings. "
            "Open the visual description with [Shot 1]; later shots need a cut that changes subject, space or viewpoint, "
            "never a slight reframe. "
            "One action carries the clip. Add a second one only when a hand or mouth is free and the clip is long. "
            "Start from anatomy, then mood: who owns the shaft, which hole it enters, what wraps it, what drives it. "
            "Name the opening once and never switch it. Short shallow strokes with the shaft held inside beat full "
            "withdrawals: keep the base attached to the pelvis, only the visible part changes. "
            "Frame from hip height, a little above, so the join sits in the lower middle and both faces stay inside: no thigh, "
            "belly or hand covers the entry, the labia part around the shaft and stay visible while it moves. "
            "Register contact in three visible steps - the path, the first touch, the kept pressure - and never teleport a hand. "
            "The reaction lands one beat after its cause; show it on brows, eyes and jaw. "
            "One camera idea per shot, canonical wording, positive: static shot, push in, pan left, tilt down. "
            "Say <Subject 2>'s penis, never make a <Picture> an actor. "
            + ("A <Picture> of a cock is that man's anatomy, not a third body. In subject_definitions the man's line also "
               "carries: the shape, size and appearance of his erect penis come from <Picture N> and it stays attached at his "
               "pelvis, unchanged; retention_analysis keeps a matching fully_preserved line for that picture. "
               if mode == "ref2v" else "")
            + "Every sentence names the labelled actor who moves. "
            + ("Start exactly on the supplied first frame, same bodies, light and framing, and move only the named parts. "
               if mode in ("i2v", "fl2v") else "")
            + ("End exactly on the supplied last frame: same pose, same framing, same light. "
               if mode in ("l2v", "fl2v") else "")
            + (("Reference labels, and nothing else: " + ", ".join(labels) + ". ") if labels
               else "No reference is attached, so no <Picture N> or <Video N> label appears. ")
            + ("subject_definitions names each subject and its reference. " if "subject_definitions" in sections else "")
            + ("retention_analysis states which appearance features are kept. " if "retention_analysis" in sections else "")
            + "overall_soundscape: room tone, body impact, wet penetration, hands and bedding, breath and voice, speech as "
            "<d>[English] spoken words</d>. non_diegetic_music: N/A unless asked. "
            "Plain present tense, one visible fact per clause, no rating words, no (weight:1.2) syntax, no quoted speech: "
            "dialogue lives in <d>[English] words</d> and the speaker carries (S1) or (S2) at the first vocal event. "
            "Never copy the brief or the notes into the output, never explain a choice, never summarise at the end. ")
    elif family == "anima":
        instruction += (
            "Output ONE line of lowercase comma-separated booru tags and nothing else: no sentences, no full stops. "
            f"Always start with: {style_prefix or ANIMA_PREFIX}. "
            "Then, in this tag order: shot distance and angle, who is where relative to the other, count tag "
            "(1girl for a solo woman, 2girls, 1girl 1boy), act, anatomy, pose, hands, place, light. "
            "Use Gelbooru spelling and the shortest clear tag; weight a stressed feature as (tag:1.2) at most. "
            "Every item stays a tag, at most four words, never a clause: write `penis in pussy`, not `the shaft is buried "
            "deep inside her`. For penetration always include `sex`, the position tag and `penis in pussy` (or `anal`), "
            "then `nude`, then the bodies. One body fact per tag so limbs stay countable. "
            f"At most {min(max_words, 70)} tags, so the framing tags are not drowned. ")
    elif pipe["artifacts"].get("krea_references"):
        identity = "identical face, hair and body of the photo"
        extra = ""
        if context.get("picture_count", 0) > 1:
            extra = ("References after the first show the same person's outfit and jewellery: name each piece they show "
                     "in the second sentence and keep it. ")
        instruction += (
            "This is a Krea identity edit. Output one paragraph, no labels, no 'Notes:'.\n"
            "Sentence 1 is identity only: \"The same person as the photo, the " + identity + ", in <place>.\" Name no age, "
            "nationality or build there unless the idea gives one. "
            + extra +
            "Then the change: \"The <garment> is removed\" and the bare skin, breasts and vulva that appear; do not also "
            "say she wears it. Any garment a reference wears goes unless the idea keeps it; jewellery, necklaces, glasses and "
            "shoes stay unless the idea removes them. If the idea "
            "says naked, nude or topless, every garment goes and nothing from the reference outfit stays on. Then the new "
            "pose with concrete body positions, then the camera. Close with one clause on the identity and skin that stay. "
            "Fold the extra notes into the flow instead of quoting them. "
            f"Do not exceed {min(max_words, 140)} words. ")
    else:
        instruction += (
            "Write one flowing paragraph without numbered labels. Cover these in order, one short sentence each:\n"
            "1. Shot: begin with the distance (wide or full body shot when the whole figure fits head to toe), then angle, "
            "height and which side, for example wide three-quarter view from the foot of the bed, just above mattress height.\n"
            "2. Staging: what supports each body, who is above or below, hips aligned, legs on both sides of the partner.\n"
            "3. Contact: which part enters what, how much shaft or fingers stay visible, where each hand and arm is.\n"
            "4. Bodies: count them (one head and one torso for a solo, two heads and four arms for a couple); which face the camera sees.\n"
            "5. Place and light in one clause.\n"
            f"Do not exceed {max_words} words. Banned filler: intimate, connection, composition centres, concrete detail, "
            "frozen frame, atmosphere. ")
    return instruction


class ImagePromptWriter(io.ComfyNode):
    H3 = False

    @classmethod
    def define_schema(cls):
        return nodes_schema("H3" if cls.H3 else "IMAGE", "PROMPT_WRITER", "H3 Prompt Writer" if cls.H3 else "Image Prompt Writer",
            inputs=[Pipe.Input("pipe", optional=True), Writer.Input("writer_model", optional=True, lazy=True),
                    io.Combo.Input("writer_preset", options=["auto", *service.PROMPTFORGE_PRESETS], default="auto",
                           tooltip="Template for this writer call. Auto follows the pipe family and its references; a pick is used for this call only and leaves the pipe settings alone."),
                    io.Int.Input("seed", default=1, min=0, max=2**32-1),
                    io.Int.Input("max_tokens", default=4096, min=128, max=32768),
                                        io.String.Input("text", optional=True, force_input=True, tooltip="Prompt text from a Prompt or Prompt Library node when no pipe is used."),
                    io.String.Input("style_prefix", default="", tooltip="Tags or words put in front of every written prompt. Empty uses the family default (Anima gets its anime quality tags)."),
                    io.Float.Input("temperature", default=.55, min=0, max=2),
                    io.Float.Input("top_p", default=.9, min=0, max=1, step=.01),
                    io.Int.Input("top_k", default=40, min=0, max=200),
                    io.Float.Input("repetition_penalty", default=1.05, min=1, max=2, step=.01),
                    io.Float.Input("presence_penalty", default=.4, min=0, max=2, step=.05),
                    io.Combo.Input("thinking", options=["off", "on"], default="off",
                           tooltip="On lets the enhancer think before writing. Slower, sometimes better on dense scenes."),
                    io.Combo.Input("checks", options=["strict", "off"], default="strict",
                           tooltip="Strict runs the prompt lint and the H3 section syntax. Off keeps whatever came back."),
                    io.Combo.Input("retries", options=["off", "1", "2", "3"], default="1",
                           tooltip="Extra tries when the written prompt fails the checks. The first try that passes is kept, no further copy is written.")],
            outputs=[Pipe.Output("pipe"), io.String.Output("text")])

    @classmethod
    def check_lazy_status(cls, pipe, writer_model=None, **kwargs):
        return ["writer_model"] if _prompt(pipe).get("source") != "raw" and writer_model is None else []

    @classmethod
    def execute(cls, pipe=None, writer_model=None, writer_preset="auto", seed=1, max_tokens=4096, temperature=.55, top_p=.9, top_k=40, repetition_penalty=1.05, presence_penalty=.4, retries="1", checks="strict", thinking="off",
                style_prefix="", text=None):
        pipe = ensure_pipe(pipe, family="minimax_h3" if cls.H3 else "image")
        if text and not _prompt(pipe).get("text"):
            pipe = _store_prompt(pipe, "raw", text, "")
        prompt = _prompt(pipe)
        if prompt.get("source") == "raw":
            text = prompt["text"]
            result = evolve(pipe, artifacts={"text": text, "writer": {
                "status": "raw", "attempts": [], "validation": validation_issues(text, pipe["family"])}},
                invalidate=("positive", "negative"))
            return io.NodeOutput(result, text)
        if prompt.get("source") not in ("idea", "action"):
            raise ValueError("Connect a Prompt node and select raw, idea or action before writing.")
        if writer_model is None:
            raise ValueError("Connect Writer Model to write an idea or action, or select raw.")
        if cls.H3 and pipe["family"] != "minimax_h3":
            raise ValueError("H3 Prompt Writer requires a minimax_h3 pipe.")
        attempts = (int(retries) if retries not in (None, "", "off") else 0) + 1
        images, video, mode, context = _writer_context(pipe, cls.H3)
        service = _services()
        override = bool(writer_preset) and writer_preset != "auto"
        preset = writer_preset if override else (H3_PRESETS[mode] if cls.H3 else
                  service._family_preset(pipe["family"], bool(images), video is not None))
        if preset not in service.PROMPTFORGE_PRESETS:
            raise ValueError(f"Writer template is unavailable: {preset}")
        # a picked template from another family still writes; the report says why it may read wrong
        preset_family = _preset_family(preset) if override else None
        mismatch = (f"{preset} is a {preset_family} template, the pipe is {pipe['family']}"
                    if preset_family and preset_family != pipe["family"] else "")
        manual_template = service.PROMPTFORGE_PRESETS[preset]["text"] if override else ""
        encoder = service.PROMPTFORGE_PromptEnhancer()
        budget = {"krea2": 260, "moss_image": 320, "anima": 120, "minimax_h3": 2600}.get(pipe["family"], 300)
        template = _instructions(pipe, mode, context, budget, style_prefix)
        if manual_template:
            template = manual_template + "\n\n" + template
        # H3 writes the picture and the sound in separate calls: one long output tends to
        # run past every section and lose the audio, so each call gets its own stop.
        if cls.H3:
            audio_sections = [name for name in _services()._MINIMAX_SECTIONS[mode]
                              if name in ("overall_soundscape", "non_diegetic_music")]
            template = _instructions(pipe, mode, context, budget, style_prefix,
                                     [name for name in _services()._MINIMAX_SECTIONS[mode] if name not in audio_sections])
            sound_template = _instructions(pipe, mode, context, 220, style_prefix, audio_sections)
        reference_rules = (
            "Reference mapping and target format: " + json.dumps(context) +
            "\nContact sheet cells are separate reference pictures, not the target composition. "
            "Use the original Picture numbers from the mapping in the final prompt. "
            "Audio semantics are unavailable to this writer; use only user-supplied audio descriptions. "
            "For continuation, continue the supplied video using the continuity settings. " +
            ("Preserve the required H3 sections and reference labels." if cls.H3 else ""))
        user_text = prompt["text"]
        if prompt.get("source") == "action" and cls.H3:
            # action briefs carry NOT / never lines as constraints, and the writer must still emit the audio sections
            user_text = ("Turn this brief into the shot description, following the layout and the word count. Keep every "
                         "position, contact and camera fact as it is written and do not lengthen it. A NOT, never or no line is "
                         "a constraint: fold it into the state described, but keep the word when it prevents a real error "
                         "(which hole, a second tip, an extra limb, the body drifting).\n\n" + user_text)
        elif prompt.get("source") == "action":
            user_text = ("Condense this brief into affirmative scene description. Keep every position, contact and camera "
                         "fact; drop ornamental words. Its NOT, never, without and no lines are constraints: describe the "
                         "state they ask for and leave those words out. Do not invent limbs or second parts.\n\n" + user_text)
        records = []
        feedback = ""
        for index in range(attempts):
            attempt_seed = (int(seed) + index) % (2**32 - 1) or 1
            system = template + "\n\n" + reference_rules + (
                "\nDescribe the desired result affirmatively in every descriptive field. "
                "Return only the target model prompt. Writer constraints describe desired outcomes: " +
                json.dumps({"excluded_from_scene": prompt.get("negative", "")}) + "\n" + feedback)
            text, thinking = _write(encoder, service, writer_model, system, user_text, images, video,
                                    attempt_seed, min(max_tokens, 4096), temperature, thinking == "on",
                                    top_p, top_k, repetition_penalty, presence_penalty)
            text = _strip_context_echo(text)
            audio_text = ""
            if cls.H3:
                text = _h3_headings(text, mode)
                sound_system = sound_template + "\n\n" + reference_rules + "\nReturn only the sound sections."
                sound_user = ("Name the sound for this clip. Return only overall_soundscape and non_diegetic_music.\n\nClip:\n"
                              + text + "\n\nBrief for reference:\n" + user_text)
                audio_text, sound_thinking = _write(encoder, service, writer_model, sound_system, sound_user,
                                                    images, video, (attempt_seed + 7919) % (2**32 - 1), 1024, temperature)
                audio_text = _h3_headings(_strip_context_echo(audio_text), mode)
                if "overall_soundscape" in audio_text:
                    text = text.rstrip() + "\n\n" + audio_text.strip()
                text = _h3_repair_audio(text, pipe, mode)
            request = json.dumps({"system": system, "user": user_text, "context": context}, ensure_ascii=False)
            issues = [] if checks == "off" else validation_issues(text, pipe["family"])
            if cls.H3 and checks != "off":
                issues.extend(_h3_issues(text, mode, context))
            records.append({"seed": attempt_seed, "text": text, "validation": issues, "request": request, "thinking": thinking})
            if not issues:
                result = evolve(pipe, artifacts={"text": text, "writer": {
                    "status": "valid", "validation": [], "attempts": records, "max_tokens": max_tokens,
                    "context": context, "preset": preset, **({"preset_mismatch": mismatch} if mismatch else {})}},
                    invalidate=("positive", "negative"))
                return io.NodeOutput(result, text)
            feedback = ("Rewrite affirmatively: drop these words and replace each with what is visible in the scene: "
                        "no, not, never, without, avoid. Failures: " + "; ".join(issues))
        dump = Path(folder_paths.get_temp_directory()) / "PromptForge_writer_failed.json"
        dump.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        reasons = "; ".join(f"attempt {index + 1}: {', '.join(record['validation'])}" for index, record in enumerate(records))
        raise ValueError(f"Prompt writer exhausted {attempts} validation attempts, encoding blocked. {reasons}. "
                         f"The attempts are in {dump}.")


class H3PromptWriter(ImagePromptWriter):
    H3 = True


class PromptInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("PROMPT", "INSPECT", "Prompt Inspect",
                            inputs=[Pipe.Input("pipe", optional=True), io.String.Input("text", optional=True, force_input=True)],
                            outputs=[Pipe.Output("pipe"), io.String.Output("report")], output=True)

    @classmethod
    def execute(cls, pipe=None, text=None):
        pipe = ensure_pipe(pipe, family="image")
        if text:
            pipe = evolve(pipe, artifacts={"text": text})
        require(pipe, "text")
        text = pipe["artifacts"]["text"]
        writer = pipe["artifacts"].get("writer", {})
        counts = {}
        clip = pipe["artifacts"].get("clip")
        if clip is not None:
            tokens = clip.tokenize(text)
            if isinstance(tokens, dict):
                for name, chunks in tokens.items():
                    if isinstance(chunks, list) and all(isinstance(chunk, list) for chunk in chunks):
                        counts[name] = sum(len(chunk) for chunk in chunks)
        report = json.dumps({"family": pipe["family"], "prompt": _prompt(pipe), "text": text,
                             "validation": validation_issues(text, pipe["family"]), "writer": writer,
                             "token_counts": counts, "token_count_status": "generation text tokenizer, including padding; multimodal conditioning may add tokens" if counts else "tokenizer count unavailable",
                             "sections": dict(re.findall(r'(?m)^([a-z_]+):\s*(.*(?:\n(?![a-z_]+:).*)*)', text))},
                            ensure_ascii=False, indent=2)
        return io.NodeOutput(pipe, report, ui=ui.PreviewText(report))


NODES = [WriterModel, ImagePrompt, ImagePromptWriter, H3Prompt, H3PromptWriter, PromptInspect]
NODE_CLASS_MAPPINGS = {
    "PROMPTFORGE_V2_WRITER_MODEL": WriterModel,
    "PROMPTFORGE_V2_IMAGE_PROMPT": ImagePrompt,
    "PROMPTFORGE_V2_IMAGE_PROMPT_WRITER": ImagePromptWriter,
    "PROMPTFORGE_V2_H3_PROMPT": H3Prompt,
    "PROMPTFORGE_V2_H3_PROMPT_WRITER": H3PromptWriter,
    "PROMPTFORGE_V2_PROMPT_INSPECT": PromptInspect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    key: "PromptForge V2 · " + name for key, name in zip(NODE_CLASS_MAPPINGS,
        ("Writer Model", "Image Prompt", "Image Prompt Writer", "H3 Prompt", "H3 Prompt Writer", "Prompt Inspect"), strict=True)
}
