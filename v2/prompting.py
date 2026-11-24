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


_store = _library("v2.action_store")
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\'[^\'\n]*\'(?!\w)')
_QUOTE_CONTEXT = re.compile(
    r'(?:says?|speaks?|whispers?|shouts?|replies|asks?|reads?|reading|written|'
    r'dialogue|lyrics?|sign|caption|title|named|called|text|sings?)\b[^.!?\n]{0,70}$', re.I)
_NEGATIVE = re.compile(
    r"\b(?:no|not|never|without|avoid(?:ing)?|exclude|excluding|omit|remove|"
    r"do\s+not|don['’]t|mustn['’]t|shouldn['’]t|can['’]t|cannot|won['’]t)\b|"
    r"\b(?:free\s+of|negative\s+prompt|negative\s+constraints?)\b|\b(?!(?:hands?|arms?|legs?|fingers?)\b)\w+[- ]free\b", re.I)
_REASONING = re.compile(
    r"</?(?:think|analysis|reasoning)>|<\|(?:analysis|assistant|im_start)\|>|"
    r"(?:^|\n)\s*(?:analysis|reasoning|final answer|final prompt|here(?:'s| is) (?:the|your) prompt)\s*:|"
    r"\b(?:I (?:need to|should|will|must)|let me (?:think|analy[sz]e)|"
    r"the user (?:wants|asked|requests)|we need to|I cannot (?:help|assist))\b", re.I)


_SHOT_TAGS = ("close-up", "portrait", "upper body", "cowboy shot", "full body", "wide shot")
_HEAD_COUNT = re.compile(r"\b(\d+)(?:girls?|boys?|others?)\b")


def _shot_conflict(text):
    """Several shot sizes in one tag list: the model draws a mix of them."""
    tags = [tag.strip().lower() for tag in text.split(",")]
    shots = [shot for shot in _SHOT_TAGS if shot in tags]
    return f"conflicting shot sizes ({', '.join(shots)}): keep only the one the brief names" if len(shots) > 1 else ""


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
    return stripped[end:].lstrip("\n ")


SHOTS = ["auto", "close up", "upper body", "cowboy shot", "full body", "wide shot"]
ANGLES = ["auto", "front view", "side view", "three-quarter view", "from above", "from below", "from behind", "pov"]


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
    return [f"image: {key}" for key in _store.labels("image")] + [f"h3: {key}" for key in _store.labels("h3")]


def _action_genders(label, count, h3):
    if h3:
        return [role["gender"] for role in _library("h3_actions").role_info(label)]
    # image briefs always call subject 1 the woman and subject 2 the man
    return ["female", "male"] if count == 2 else ["female"]


def _action_catalog():
    """The picker's data: label, one-line description and metadata, split by generation."""
    catalog = {"image": {}, "minimax": {}}
    for family, h3 in (("image", False), ("h3", True)):
        for label, entry in _store.merged(family).items():
            arrangement = entry["brief"]
            description = re.split(r"(?<=[.!?])\s", " ".join(arrangement.split()))[0]
            catalog["minimax" if h3 else "image"][label] = {
                "category": entry["category"],
                "description": description if len(description) <= 180 else description[:177].rsplit(" ", 1)[0] + "…",
                "brief": entry["brief"],
                "gender": _action_genders(label, entry.get("people") or len(entry.get("roles", ())), h3),
                "aspect": entry.get("aspect", ""),
                "nsfw": _store.nsfw(family, label)}
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


_SUBJECT_LINE = re.compile(r"^Subject (\d) is (the [a-z0-9 ]+?) \((.+?)\):\s*(.*)$")


def _action_text(action, subjects, notes, camera):
    kind, label = action.split(": ", 1)
    if kind == "h3":
        library = _library("h3_actions")
        text = library.brief(label, library.ANY_CAMERA if camera == "auto" else camera, subjects, notes)
        names = library.subject_names(label)
        # these lines restate the writer instruction; drop them so the brief is content only
        drop = ("This action is exactly", "The act is already", "The join is the", "Recommended aspect",
                "Motion beats", "Write ", "Its NOT", "Anatomy anchors", "His hips drive", "Action preset:")
        kept, seen = [], set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(drop):
                continue
            subject = _SUBJECT_LINE.match(line)
            if subject:
                name, role, who = subject.group(2), subject.group(3), subject.group(4)
                if role == "solo" and not who:
                    continue
                line = name.capitalize() + ("" if role == "solo" else f" is {role}") + (f": {who}" if who else "") + "."
            if line.startswith("Camera:"):
                line = "Camera: " + line[len("Camera:"):].strip().rstrip(".") + "."
            for number, name in enumerate(names, 1):
                line = line.replace(f"Subject {number}", name)
            line = line.replace("Subject 3", "the other person")
            if line.startswith("Action: "):
                line = line[len("Action: "):]
            payload = line.split(": ", 1)[-1]
            if len(payload) > 40 and payload in seen:
                continue
            seen.add(payload)
            kept.append(line[:1].upper() + line[1:])
        return " ".join(kept)
    if kind != "image":
        raise ValueError(f"Unknown action library: {kind}")
    lines, seen = [], set()
    for line in _library("image_actions").brief(label, subjects[:2], camera, "auto", "auto", "as in the image", "auto", notes, "", "text description", "").splitlines():
        # the first states a pose the action may contradict (it keys on the label), the second invites invented hand and limb detail
        if line.startswith(("Replace the source pose", "Both bodies are horizontal", "The pose is exactly this arrangement")):
            continue
        # the camera stage line talks about the join of the pelvises (wrong for oral, hands and every SFW pair) and the contact line makes the writer
        # elaborate anatomy; the plain Camera line already carries the framing. The limb counts get quoted into the prompt and invite extra limbs
        if line.startswith(("Camera stage:", "Show the point of contact")):
            continue
        line = line.replace("Two complete, separate bodies: two heads, two torsos, four arms and four legs.", "Write two separate bodies.")
        line = _positive(line)
        payload = line.split(": ", 1)[-1]
        if not line or len(payload) > 40 and payload in seen:
            continue
        seen.add(payload)
        lines.append(line)
    return "\n".join(lines)


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
                    io.Combo.Input("shot", options=SHOTS, default="auto", tooltip="Distance. auto keeps the brief's framing."),
                    io.Combo.Input("angle", options=ANGLES, default="auto", tooltip="Viewpoint. auto keeps the brief's angle."),
                    io.String.Input("notes", default="", multiline=True),
                    io.String.Input("soundscape", default="", multiline=True, tooltip="Minimax generation only, raw mode: overall_soundscape line (room tone, dialogue, effects)."),
                    io.String.Input("music", default="N/A", tooltip="Minimax generation only, raw mode: non_diegetic_music. N/A when there is no score."),
                    # last, so workflows saved before it existed keep their value positions; "" is the old empty slot
                    io.Combo.Input("generation", options=["image", "minimax", ""], default=cls.GENERATION)],
            outputs=[Pipe.Output("pipe"), io.String.Output("text")])

    @classmethod
    def execute(cls, pipe=None, source="raw", text="", negative="", action="", subject_1="", subject_2="", subject_3="",
                camera="auto", notes="", text_in=None, soundscape="", music="N/A", generation=None,
                shot="auto", angle="auto"):
        pipe = ensure_pipe(pipe, family="minimax_h3" if cls.H3 else "image")
        audio = (generation or cls.GENERATION) == "minimax"
        if text_in:
            text = text_in
        if cls.H3 and pipe["family"] != "minimax_h3":
            raise ValueError("H3 Prompt requires a minimax_h3 pipe.")
        if source == "action":
            frame = " and ".join(part for part in (shot if shot != "auto" else "", angle if angle != "auto" else "") if part)
            text = _action_text(action, (subject_1, subject_2, subject_3), notes, camera)
            if cls.H3 and frame:
                text += f"\nFraming: {frame}. The framing note wins over the camera line." + (f"\nAdditional intent: {text}" if text else "")
            if not cls.H3 and frame:
                text += f"\nFraming: {frame}."
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
    # a resident video model starves the writer: the 9 GB encoder then streams from RAM at single-digit tok/s;
    # once the writer is loaded for this run, the next calls reuse it
    if not encoder.resident:
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


# female anatomy written onto the scrotum ("a slit down the middle of his sac"); the verbs of contact keep "his balls slap against her vulva" legal
_SCROTUM_AS_VULVA = re.compile(r"(?i)\b(?:scrotum|balls|testicles|sac)\b(?:(?!\b(?:against|onto|on|between|slaps?|presses|press|swings?|hits?|touch\w*|brush\w*|rest\w*|rubs?|grind\w*|bounc\w*|her)\b)[^.,;]){0,40}?\b(?:labia|vulva|pussy|slit|cleft|folds|vagina\w*)\b")
# the template says to keep a given age "exactly as written"; small writers write that sentence into the scene
_H3_AGE_ECHO = re.compile(r"(?i),?\s*\b(?:aged?|age)\s+(?:exactly\s+)?as\s+(?:specified|written|given|stated)\s+(?:in|by)\s+the\s+(?:request|user|brief|prompt)\b,?")


# a penetrative stroke that leaves the body gets drawn as the shaft falling out and landing back in
_LEAVES_THE_BODY = re.compile(r"(?i)\b(?:pull(?:s|ed|ing)?|slid(?:es|ing)?|slip(?:s|ped|ping)?|pop(?:s|ped|ping)?|fall(?:s|ing)?|withdraw(?:s|n|ing)?)\s+(?:all the way |completely |fully |right |back )?out\b"
                              r"|\bleaves? (?:her|the) (?:vagina|body|opening|entrance|anus)\b|\b(?:completely|fully) withdraw\w*")


# "(appears in )" / "(the first frame of )": the template's placeholders written out with nothing in them
_H3_EMPTY_REF = re.compile(r"\s*\((?:appears in|the first frame of|from)?\s*\)")


_FACT_STOP = set("""that this with from into onto over under while being then than their there these those would could should about after before also just very some such only same exact
exactly form provided image picture photo now but and the her his she for are was were has have had who whom which what when where each every other another like make made show shows
showing scene style realistic more most much many them they your you its not never without within between through while during still again here keep keeps kept look looks looking""".split())
_FACT_SYNONYMS = [{"naked", "nude", "bare", "unclothed", "undressed", "topless", "bottomless", "unclad"}, {"blushing", "blush", "flushed", "flush", "reddened", "crimson", "embarrassed", "embarrassment", "embarassment"},
                  {"fucked", "fucking", "sex", "penetrating", "penetrated", "thrusting", "inside", "behind"}, {"students", "student", "classmates", "pupils", "classroom", "class"},
                  {"teacher", "professor", "lecturer", "instructor"}, {"breasts", "breast", "chest", "boobs"}, {"ass", "butt", "buttocks", "rear", "bottom"},
                  {"panting", "gasping", "breathless", "breathing"}, {"toes", "toe", "feet", "foot"}, {"staring", "stare", "watching", "looking", "gazing"}]


def _dropped_facts(idea, text):
    """Content words of the idea that the finished prompt never mentions, synonyms and word stems allowed."""
    have = {word[:5] for word in re.findall(r"[a-z]+", text.lower())}
    have_words = set(re.findall(r"[a-z]+", text.lower()))
    missing = []
    for word in dict.fromkeys(re.findall(r"[a-z]{4,}", idea.lower())):
        if word in _FACT_STOP or word[:5] in have:
            continue
        group = next((g for g in _FACT_SYNONYMS if word in g), None)
        if group and (group & have_words or {w[:5] for w in group} & have):
            continue
        missing.append(word)
    return missing


def _h3_headings(text, mode):
    """One heading per line, colon attached, content underneath: repair what the writer returns."""
    for section in _h3_section_names(mode):
        text = re.sub(rf"(?<=[^\n])({section}:)", r"\n\n\1", text)          # heading written mid-line
        text = re.sub(rf"(?m)^({section}):?[ \t]*:?[ \t]*(?=\S)", r"\1:\n", text)  # content glued to it
        text = re.sub(rf"(?m)^({section}):?[ \t]*:?[ \t]*$", r"\1:", text)     # bare heading, double colon
    text = re.sub(r"(?m)^:[ \t]*$\n?", "", text)
    return re.sub(r"(?<!\[)\bShot (\d+)\b(?!\])", r"[Shot \1]", text)


def _h3_section_names(mode):
    table = _services()._MINIMAX_SECTIONS
    if mode in table:
        return table[mode]
    short = {"t2v": "t2va", "i2v": "i2va", "l2v": "l2va", "fl2v": "fl2va", "ref2v": "ref2va", "continue": "ref2va"}
    return table[short.get(mode, "t2va")]


_SHOT_LABEL = re.compile(r"(?i)\[\s*shot\s*#?\s*\d+[^\]\n]{0,40}\]")


def _number_shots(body):
    """Shot labels belong to the description alone: drop repeats, number the rest from 1, and add [Shot 1] when the writer wrote none."""
    seen, count = set(), [0]

    def renumber(match):
        if match.group(0) in seen:
            return ""
        seen.add(match.group(0))
        count[0] += 1
        return f"[Shot {count[0]}]"

    body = re.sub(r"[ \t]{2,}", " ", _SHOT_LABEL.sub(renumber, body))
    return body if count[0] else "[Shot 1] " + body


def _h3_repair_sections(text, pipe, mode):
    """Rebuild missing sections in order and keep the shot labels in the description only."""
    text = re.sub(r"[ \t]{2,}", " ", text)

    sections = _h3_section_names(mode)
    blocks = {}
    headings = list(re.finditer(r"(?m)^([a-z_]+):[ \t]*$", text))
    for position, match in enumerate(headings):
        name = match.group(1)
        # a t2v style body written while the mode wants ref2v keeps its text, only the heading changes
        if name not in sections and name == "integrated_multimodal_description" and "detailed_description" in sections:
            name = "detailed_description"
        # the model likes to write a section twice: the first one wins, a body ends at the next heading
        if name in sections and name not in blocks:
            blocks[name] = text[match.end():headings[position + 1].start() if position + 1 < len(headings) else len(text)].strip()
    for name, body in blocks.items():
        if name in ("integrated_multimodal_description", "detailed_description"):
            blocks[name] = _number_shots(body) if body else body
        else:
            blocks[name] = re.sub(r"[ \t]{2,}", " ", _SHOT_LABEL.sub("", body)).strip()
    fallback = (pipe["settings"].get("prompt", {}).get("text") or "").strip() if blocks else text.strip()
    rebuilt = []
    for name in sections:
        if name in blocks and blocks[name]:
            body = blocks[name]
        elif name in ("overall_soundscape", "non_diegetic_music"):
            continue  # the audio repair adds those
        elif name in ("detailed_description", "integrated_multimodal_description"):
            body = _number_shots(fallback or "the requested action plays out.")
        elif name == "summary":
            body = "The clip shows the requested action."
        elif name == "subject_definitions":
            body = _ref_subjects(pipe)
        elif name == "retention_analysis":
            body = _ref_retention(pipe)
        else:
            continue
        rebuilt.append(f"{name}:\n{body}")
    return "\n\n".join(rebuilt)


def _h3_repair_audio(text, pipe, mode):
    """Append the sound sections when the writer spends its budget on the description."""
    sections = _h3_section_names(mode)
    prompt = _prompt(pipe)
    if "overall_soundscape" in sections and not re.search(r"(?m)^overall_soundscape:", text):
        text += "\n\noverall_soundscape:\n" + (prompt.get("soundscape") or "").strip().replace("\n", " ")
        if text.rstrip().endswith("overall_soundscape:"):
            text += " soft room tone, breathing and quiet moans"
    if "non_diegetic_music" in sections and not re.search(r"(?m)^non_diegetic_music:", text):
        text += "\n\nnon_diegetic_music:\n" + ((prompt.get("music") or "").strip() or "N/A")
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
    body = re.search(r"(?ms)^(?:integrated_multimodal_description|detailed_description):\s*(.*?)(?=^[a-z_]+:\s*$|\Z)", text)
    words = len(body.group(1).split()) if body else 0
    if words > 1500:
        issues.append(f"description too long ({words} words): the brief asks for 650-900")
    return issues


_REFERENCE_LABEL = {"image": "Picture", "video": "Video", "audio": "Audio"}


_EXPLICIT = re.compile(r"\b(?:nsfw|nude|naked|topless|bottomless|sex|sexual|explicit|erotic|porn\w*|fuck\w*|cock|dick|penis|pussy|vagina|vulva|clit\w*|cum|cumming|blowjob|handjob|anal|nipples?|breasts?|boobs?|tits|orgasm\w*|masturbat\w*|genitals?)\b", re.I)


# the template sections that only matter when the request is sexual; a small writer copies their example lists into any prompt
_EXPLICIT_SECTIONS = re.compile(r"(?ms)^# (?:NSFW|NSFW standard|NSFW audio \(mandatory\)|Genital anatomy and penetration|Genitals)\n.*?(?=^# |\Z)")
# booru tags that state an absence in the picture; every other "no X" tag is a copied instruction
_ABSENCE_TAGS = {"no humans", "no panties", "no bra", "no pants", "no shirt", "no shoes", "no socks", "no legwear"}


def _clean_tags(text):
    tags = [tag.strip() for tag in text.split(",")]
    tags = [tag for tag in tags if tag and (tag.lower() in _ABSENCE_TAGS or not re.match(r"(?i)(?:no|not|without|never)\b", tag))]
    # "solo" next to a second person makes the model draw one of them
    if sum(int(count) for tag in tags for count in _HEAD_COUNT.findall(tag.lower())) > 1:
        tags = [tag for tag in tags if tag.lower() != "solo"]
    return ", ".join(tags)


_NEGATION_START = re.compile(r"(?i)^\W*(?:no|not|never|without|avoid)\b")


# a negated clause after a comma or dash ("..., no props"), or a trailing "without X" / "but not X"; a plain "is not moving" stays
_NEGATION_CLAUSE = re.compile(r"(?i)(?:\s*[,;\u2014\u2013]\s*(?:(?:but|and|with|while)\s+)?(?:no|not|without|never)\b|\s+(?:but\s+not|without)\b)[^,;.!?\u2014\u2013]*")
_AGE_ECHO = re.compile(r"(?i),?\s*\bage (?:un)?(?:specified|stated|given)\b,?")


def _drop_negations(text):
    """A prose prompt states what is there: negations the writer wrote are the brief's constraints echoed back, and an image model draws the noun."""
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if not _NEGATION_START.match(sentence)]
    return " ".join(_AGE_ECHO.sub("", _NEGATION_CLAUSE.sub("", sentence)) for sentence in sentences)


_GENITALS = {name: re.compile(rf"(?i)\b(?:{words})\b") for name, words in (
    ("female", "vulva|labia|clitoris|clitoral|pussy|vagina|vaginal"), ("male", "penis|cock|scrotum|testicles|balls|glans|foreskin"), ("anal", "anus|anal|asshole"))}
_CAMERA_SENTENCE = re.compile(r"(?i)^\W*(?:the )?camera\b")


def _grounded(text, brief, tags=False):
    """An action brief is the whole scene. A genital it never names, or a camera it never gives, is the writer's
    invention from the template lists, and the image model draws it (a second set of genitals, a stranger's shoulder)."""
    absent = [pattern for pattern in _GENITALS.values() if not pattern.search(brief)]
    camera = bool(re.search(r"(?m)(?:^|[.!?]\s+)(?:The )?Camera\b", brief))
    def unsupported(part):
        return any(pattern.search(part) for pattern in absent)

    if tags:
        kept = [tag.strip() for tag in text.split(",") if not unsupported(tag)]
        return ", ".join(kept) if kept else text
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not camera and _CAMERA_SENTENCE.match(sentence):
            continue
        # only the clause that names the invented part goes: the act often shares its sentence
        body, end = re.match(r"(.*?)([.!?]*)$", sentence, re.S).groups()
        clauses = [clause for clause in re.split(r"\s*[,;]\s+", body) if not unsupported(clause)]
        if clauses:
            joined = re.sub(r"(?i)^(?:and|but|while|with)\s+", "", ", ".join(clauses))
            sentences.append(joined[0].upper() + joined[1:] + end)
    return " ".join(sentences) if sentences else text


def _grounded_h3(text, brief):
    """The same guard for the H3 description body; the brief's own 'never a vulva' constraints do not count as naming one."""
    named = re.sub(r"(?i)\b(?:never|no) (?:(?:gets|grows|has|shows) )?(?:a |an |any )?(?:penis|vulva|vagina|pussy|cock)\b", "", brief)
    parts = re.split(r"(?m)^([a-z_]+:)[ \t]*$", text)
    for index in range(2, len(parts), 2):
        if parts[index - 1] in ("integrated_multimodal_description:", "detailed_description:"):
            parts[index] = "\n" + _grounded(parts[index].strip(), named) + "\n\n"
    return "".join(parts)


def _positive(line):
    return " ".join(sentence for sentence in re.split(r"(?<=[.!?])\s+", line) if not _NEGATIVE.search(sentence))


def _is_explicit(prompt):
    """Whether the request asks for nudity or sex: an action carries its own flag, an idea is read for the words."""
    if prompt.get("source") == "action":
        family, label = prompt["action"].split(": ", 1)
        return _store.nsfw(family, label)
    return bool(_EXPLICIT.search(prompt.get("text", "")))


_SIZE_SECTION = re.compile(r"(?ms)^# Body size and scale\n.*?(?=^# |\Z)")
_SIZE_WORDS = re.compile(r"(?i)\b(?:flat|small|narrow|skinny|slim|thick|curvy|petite|tiny|huge|plump|muscular|bigger|smaller|taller|shorter|larger|big (?:breasts|tits|ass)|large (?:breasts|tits))\b")
_MALE_ANATOMY = re.compile(r"(?i)\b(?:penis|glans|foreskin|shaft|erect\w*|balls|scrotum)\b")
_MALE_WORDS = re.compile(r"(?i)\b(?:man|men|male|boy|guy|he|his|him|husband|boyfriend|penis|cock|dick)\b")
_CAPABLE = re.compile(r"(?:The preset is NSFW capable[^.\n]*\.|You are NSFW generation capable\.)\s*")
_RATING = 'Rating: "safe" for a clothed everyday scene, "nsfw" for nudity or sex, "explicit" for hardcore.'


def _male_present(prompt):
    """Whether a man is in the scene: an action knows its cast, an idea is read for the words."""
    if prompt.get("source") == "action":
        family, label = prompt["action"].split(": ", 1)
        entry = _store.get(family, label)
        return any(role["gender"] == "male" for role in entry["roles"]) if "roles" in entry else entry.get("people", 1) > 1
    return bool(_MALE_WORDS.search(prompt.get("text", "")))


_CROWD = re.compile(r"(?i)\b(?:students|classmates|crowd|audience|onlookers|spectators|bystanders|everyone|other people|people watching|watching|staring)\b")


_CAMERA_BULLETS = (("- POV means the camera IS", re.compile(r"(?i)\bpov\b|point of view|first person")),
                   ("- Over-the-shoulder keeps the near shoulder", re.compile(r"(?i)over[- ]the[- ]shoulder|from behind|top[- ]down|low angle|high angle|dutch|close[- ]up")))


def _drop_unused_camera_rules(text, request):
    """The camera section describes every geometry; an idea that picked none spends its words on all of them."""
    for start, wanted in _CAMERA_BULLETS:
        if not wanted.search(request):
            text = re.sub(rf"(?m)^{re.escape(start)}.*\n", "", text)
    return text


def _fit_to_idea(text, idea):
    """Output limits are written for a short idea: a long one with many facts loses them to the word range, the two-body rule or the prop limit."""
    words = len(idea.split())
    need = min(450, round(words * 2.4)) if words > 40 else 0
    text = re.sub(r"(# Output\n[^\n]*?)(\d+)-(\d+) words", lambda m: f"{m.group(1)}usually {m.group(2)}-{max(int(m.group(3)), need)} words, more when the request has more facts", text, count=1)
    text = text.replace("One or two sentences,", "One paragraph,")
    text = text.replace("and the two or three props in the frame", "and every prop and person the request names")
    if _CROWD.search(idea):
        text = text.replace("Keep two complete bodies: two heads, four arms, four legs.",
                            "Keep every person's body complete and separate (count each head, arms and legs); the main pair's torsos meet only at the named contact. People the request places in the background stay in the picture as small whole figures.")
    return text


def _system_text(service, preset, context, style_prefix="", sections=None, negative="", explicit=True, male=True, sized=True, brief=False, idea="", request=None):
    """The writer's system prompt: the family template, then only the facts of this request."""
    seconds = float(context.get("duration") or 5)
    text = service.PROMPTFORGE_PRESETS[preset]["text"].replace("{video_seconds}", f"{seconds:.2f}")
    # the Order list asks for "two or three props": a small writer invents them (an apple, a pillow, a newspaper)
    text = _fit_to_idea(text, idea)
    if request is not None:
        text = _drop_unused_camera_rules(text, request)
    text = text.replace("and only the two or three props actually in frame", "and only the props the request names").replace("5. Place and the props in the frame.", "5. Place and only the props the request names.")
    if not explicit:
        text = _EXPLICIT_SECTIONS.sub("", text).replace("Start with: nsfw, anime style, 2d, flat colors.", "Start with: anime style, 2d, flat colors.").replace(_RATING, 'Rating: "safe".')
        # the Order list offers "open, pulled aside or off" for every garment; a small writer takes the offer
        text = _CAPABLE.sub("", text.replace("whether it is worn, open, pulled aside or off", "and how it is worn"))
    else:
        text = text.replace(_RATING, 'Rating: "nsfw" for nudity, "explicit" for sex acts.')
        if not male and context["family"] != "minimax_h3":
            kept = [line for line in text.split("\n") if not _MALE_ANATOMY.search(line)]
            if len(kept) < len(text.split("\n")):
                text = "\n".join(kept) + "\n\nEveryone in this scene is a woman: name only the anatomy the camera shows, by its exact name."
    if brief and explicit and context["family"] == "anima":
        # an action brief already names the act; the template's act lists get copied into the tags whole
        text = re.sub(r"(?m)^- (?:Acts|Choreography):.*\n", "", text)
        text = re.sub(r"(?ms)^# Genitals\n.*?(?=^# |\Z)", "# Genitals\n- The brief names the act, positions, hands, gaze and framing. Tag those facts with the usual booru tag for each and add no other act, position, motion, sound or anatomy tag.\n\n", text)
    if not sized:
        text = _SIZE_SECTION.sub("", text)
    parts = [text]
    # no model here truncates its text (the encoders and the image models take any length), so length is a guide: the facts decide it
    cap = service._output_cap(preset, seconds)
    if cap and not sections:
        parts.append(f"# Output length\nA typical prompt for this model runs about {cap} {'tags' if 'tags' in preset.lower() else 'words'}. "
                     "That is a guide, not a limit: write as much as the request needs, and never drop, merge or soften a fact to stay short.")
    if context.get("width") and context.get("height"):
        parts.append(f"# Target frame\nThe picture is {context['orientation']}: compose the scene to fit it. Never write the frame size.")
    if context["picture_count"] or context["video_count"] or context["audio"]:
        lines = [f"<{_REFERENCE_LABEL[item['kind']]} {item['index']}> {item['name']}" + (f": {item['description']}" if item["description"] else "")
                 for item in context["descriptions"]] or list(context["references"])
        if context["contact_sheet"]:
            lines.append("The attached picture is a contact sheet: its cells are separate references, not the target composition.")
        if context["audio"]:
            lines.append("Audio references cannot be heard by you; use only what the user wrote about them.")
        parts.append("# References\n" + "\n".join(lines))
    if sections:
        parts.append("# This call\nWrite only these sections, each with its heading: " + ", ".join(sections)
                     + ". The other sections are written by a separate call; do not write them.")
    if style_prefix:
        parts.append(f"Always start with: {style_prefix}.")
    if negative:
        parts.append("# Negative constraints\nThe scene must not contain: " + negative + ". Write the scene so they are absent, "
                     "describing the desired state instead; never name them inside the prompt.")
    if not explicit:
        parts.append("If a person appears, they are dressed: name every garment they wear, with colour and material.")
    parts.append("Keep what the request states exactly: when it places the arms, hands, legs or camera, repeat that placement and give them no other position. "
                 "Add nothing that covers a face or body (helmet, mask, hood, shadow) unless the request names it.")
    if idea.strip():
        parts.append("# Priority\nEvery fact the request states appears in the prompt: each person and how many, the place, body state, expression, posture and each detail it names. "
                     "Nudity is stated in plain words: when the request says naked, nude, topless or bare, the prompt says so and names the bare skin; clothing is never invented, and a photo's clothes are replaced. "
                     "The rules above for size, camera and anatomy shape how those facts are written; they never replace one. When words run short, shorten the rule-driven detail, never the request.")
    if explicit and context["family"] != "minimax_h3":
        parts.append("Describe only the anatomy the pose and camera show: a part the pose hides or turns away from the lens is left out, and nothing is shown twice.")
    parts.append("The lists and examples in this template show the format: write only what this request needs, never a whole list.")
    parts.append(service.OUTPUT_GUARD + " Do not output chain-of-thought or reasoning; start with the final prompt.")
    return "\n\n".join(parts)


class ImagePromptWriter(io.ComfyNode):
    H3 = False

    @classmethod
    def define_schema(cls):
        return nodes_schema("H3" if cls.H3 else "IMAGE", "PROMPT_WRITER", "H3 Prompt Writer" if cls.H3 else "Image Prompt Writer",
            inputs=[Pipe.Input("pipe", optional=True), Writer.Input("writer_model", optional=True, lazy=True),
                    io.Combo.Input("writer_preset", options=["auto", *_services().PROMPTFORGE_PRESETS], default="auto",
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
        encoder = service.PROMPTFORGE_PromptEnhancer()
        # H3 writes the picture and the sound in separate calls: one long output tends to
        # run past every section and lose the audio, so each call gets its own stop.
        sections = _services()._MINIMAX_SECTIONS[mode] if cls.H3 else None
        sound_sections = [name for name in sections or () if name in ("overall_soundscape", "non_diegetic_music")]
        picture_sections = [name for name in sections or () if name not in sound_sections]
        negative = (prompt.get("negative") or "").strip()
        explicit, male = _is_explicit(prompt), _male_present(prompt)
        sized = bool(_SIZE_WORDS.search(prompt.get("text", "")))
        system_base = _system_text(service, preset, context, style_prefix, picture_sections if cls.H3 else None, negative, explicit, male, sized,
                                   brief=prompt.get("source") == "action", idea=prompt.get("text", "") if prompt.get("source") == "idea" else "",
                                   request=" ".join(str(prompt.get(key, "")) for key in ("text", "camera", "shot", "angle")) if prompt.get("source") == "idea" and not cls.H3 else None)
        if not cls.H3 and len(images) == 2 and pipe["artifacts"].get("krea_references"):
            # two photos are two people: the template line that opens every description names the photo, so no line can be copied for both
            system_base = system_base.replace('"The same person as the photo: <build> build', '"The person in the first photo: <man or woman>, <build> build')
        sound_base = _system_text(service, preset, context, "", sound_sections, negative, explicit, male, sized) if sound_sections else ""
        user_text = prompt["text"]
        if prompt.get("source") == "action" and cls.H3:
            user_text = (user_text + "\n\nWrite this as 650-900 words of continuous shot prose only: expand every "
                         "position, contact, angle and reaction into visible detail and start with `[Shot 1]`.")
        elif prompt.get("source") == "action" and pipe["family"] == "anima":
            user_text = ("Turn this brief into booru tags. Keep every position, contact and camera fact. Lines with NOT, never, "
                         "without or no are constraints: tag the state they ask for and leave those words out. "
                         "Camera facts become framing and angle tags (full body, from side); never a camera tag, the model draws the object."
                         + ("" if explicit else " Tag every garment each person wears.") + "\n\n" + user_text)
        elif prompt.get("source") == "action":
            user_text = ("Condense this brief into affirmative scene description. Keep every position, contact and camera "
                         "fact. Lines with NOT, never, without or no are constraints: describe the state they ask for and "
                         "leave those words out. Write a camera only when the brief gives one." + ("" if explicit else " Name what each person wears, with colour and material.") + "\n\n" + user_text)
        elif prompt.get("source") == "idea":
            if not cls.H3:
                what = "tags" if pipe["family"] == "anima" else "prompt"
                user_text = (f"Write the {what} for this request. Every fact it states belongs in the {what}, in plain words, and nothing it does not state is added. "
                             "Where it says to keep something unchanged, keep it.\n\nRequest:\n" + user_text)
            if not explicit:
                user_text += "\n\nName what each person wears, with colour and material." if pipe["family"] != "anima" else "\n\nTag every garment each person wears."
        encoder.resident = {}
        try:
            records = []
            feedback = ""
            for index in range(attempts):
                attempt_seed = (int(seed) + index) % (2**32 - 1) or 1
                system = system_base + ("\n\n" + feedback if feedback else "")
                text, thought = _write(encoder, service, writer_model, system, user_text, images, video,
                                       attempt_seed, min(max_tokens, 4096), temperature, thinking == "on",
                                       top_p, top_k, repetition_penalty, presence_penalty)
                text = _strip_context_echo(text)
                if pipe["family"] == "anima":
                    text = _clean_tags(text)
                elif not cls.H3:
                    text = _drop_negations(text)
                if prompt.get("source") == "action" and not cls.H3:
                    text = _grounded(text, user_text, tags=pipe["family"] == "anima")
                audio_text = ""
                if cls.H3 and text.strip():
                    text = _H3_EMPTY_REF.sub("", _H3_AGE_ECHO.sub("", _h3_headings(text, mode)))
                    if prompt.get("source") == "action":
                        text = _grounded_h3(text, user_text)
                    sound_user = ("Name the sound for this clip.\n\nClip:\n" + text + "\n\nBrief for reference:\n" + user_text)
                    audio_text, _ = _write(encoder, service, writer_model, sound_base, sound_user,
                                           images, video, (attempt_seed + 7919) % (2**32 - 1), 1024, temperature)
                    audio_text = _h3_headings(_strip_context_echo(audio_text), mode)
                    if "overall_soundscape" in audio_text:
                        # the dedicated sound call wins over the audio sections the picture call wrote on the side
                        text = re.split(r"(?m)^(?:overall_soundscape|non_diegetic_music):", text)[0].rstrip() + "\n\n" + audio_text.strip()
                    text = _h3_repair_audio(_h3_repair_sections(text, pipe, mode), pipe, mode)
                request = json.dumps({"system": system, "user": user_text, "context": context}, ensure_ascii=False)
                issues = [] if checks == "off" else validation_issues(text, pipe["family"])
                if cls.H3 and checks != "off":
                    issues.extend(_h3_issues(text, mode, context))
                    if _SCROTUM_AS_VULVA.search(text):
                        issues.append("female anatomy written onto the man's scrotum: write it as a loose rounded pouch hanging below the shaft")
                    if prompt.get("source") == "action" and _LEAVES_THE_BODY.search(text) and _library("h3_actions").penetration_modes(str(prompt.get("action", "")).split(": ", 1)[-1]):
                        issues.append("the shaft leaves the body during the stroke: keep it inside through the whole stroke")
                records.append({"seed": attempt_seed, "text": text, "validation": issues, "request": request, "thinking": thought})
                if prompt.get("source") == "idea" and checks != "off":
                    # measured, never enforced: the report lists what the prompt left out so a weak template shows up instead of costing retries
                    dropped = _dropped_facts(prompt.get("text", ""), text)
                    if len(dropped) >= 3:
                        records[-1]["dropped_facts"] = dropped
                conflict = _shot_conflict(text) if pipe["family"] == "anima" and checks != "off" else ""
                # negative wording alone is a blemish, not a failure: after the last retry the text is used and the warning stays in the report
                last = index == attempts - 1
                blemish = all(issue.startswith(("negative wording", "description too long", "the shaft leaves", "female anatomy written")) for issue in issues)
                if (not issues or last and blemish) and (not conflict or last):
                    result = evolve(pipe, artifacts={"text": text, "writer": {
                        "status": "valid", "validation": issues, "attempts": records, "max_tokens": max_tokens,
                        "context": context, "preset": preset, **({"preset_mismatch": mismatch} if mismatch else {})}},
                        invalidate=("positive", "negative"))
                    return io.NodeOutput(result, text)
                feedback = ("Rewrite affirmatively: drop these words and replace each with what is visible in the scene: "
                            "no, not, never, without, avoid. Failures: " + "; ".join(issues + [conflict] if conflict else issues))
            dump = Path(folder_paths.get_temp_directory()) / "PromptForge_writer_failed.json"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            reasons = "; ".join(f"attempt {index + 1}: {', '.join(record['validation'])}" for index, record in enumerate(records))
            raise ValueError(f"Prompt writer exhausted {attempts} validation attempts, encoding blocked. {reasons}. "
                             f"The attempts are in {dump}.")
        finally:
            encoder.release()


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
