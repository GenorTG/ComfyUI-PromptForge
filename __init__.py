"""ComfyUI PromptForge - prompt enhancement nodes.

  - PromptForge Model: picks a connected CLIP, a safetensors model or a GGUF
    and plans quant / context / GPU layers for the current GPU
  - PromptForge Enhancer: writes the prompt with a preset system prompt,
    optional images / video frames, and output cleanup
  - PromptForge Config Finder: reports the planned settings for local models
"""

import base64
import io
import json
import gc
import importlib.util
import inspect
import os
import re
import struct
import time
from collections import Counter
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from .frame_interpolation import NODE_CLASS_MAPPINGS as VFI_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VFI_DISPLAY_NAMES
from .video_save import NODE_CLASS_MAPPINGS as SAVE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY_NAMES
from .lora_loader import NODE_CLASS_MAPPINGS as LORA_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LORA_DISPLAY_NAMES
from .unified_loader import NODE_CLASS_MAPPINGS as LOADER_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY_NAMES
from .empty_latent import NODE_CLASS_MAPPINGS as LATENT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LATENT_DISPLAY_NAMES
from .sampler import NODE_CLASS_MAPPINGS as SAMPLER_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_DISPLAY_NAMES
from .quantize_model import NODE_CLASS_MAPPINGS as QUANT_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as QUANT_DISPLAY_NAMES
from .minimax_decode import NODE_CLASS_MAPPINGS as DECODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DECODE_DISPLAY_NAMES
from . import image_library
from .h3_endless import NODE_CLASS_MAPPINGS as ENDLESS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ENDLESS_DISPLAY_NAMES
from .preview_override import NODE_CLASS_MAPPINGS as PREVIEW_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_DISPLAY_NAMES
from .load_image import NODE_CLASS_MAPPINGS as IMAGE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_NAMES
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from transformers.utils.chat_template_utils import render_jinja_template

try:
    from transformers import AutoModelForImageTextToText as AutoVision
except Exception:  # fallback for older transformers naming
    from transformers import AutoModelForVision2Seq as AutoVision

try:
    import folder_paths
except Exception:  # ComfyUI injects this module; keep a safe fallback
    folder_paths = None

PACK_DIR = Path(__file__).resolve().parent
PRESETS_PATH = PACK_DIR / "presets" / "promptforge_presets.json"
MODELS_PATH = PACK_DIR / "catalog" / "models.json"

with open(PRESETS_PATH, "r", encoding="utf-8") as _f:
    PROMPTFORGE_PRESETS = json.load(_f).get("presets") or {}

with open(MODELS_PATH, "r", encoding="utf-8") as _f:
    PROMPTFORGE_MODELS = json.load(_f) or {}

OUTPUT_GUARD = (
    "Output rules that override all style preferences: return only the final usable prompt text. "
    "Do not include analysis, planning, bullet points, headings, markdown, JSON, explanations, alternatives, "
    "self-corrections, notes, apologies, or messages to the user. Do not write phrases such as Final Plan, Final Check, "
    "Wait, Okay, First, Next, Then, I will, or I need. "
    "Never write in first person and never comment on your own work: do not say you followed the instructions, "
    "did a good job, respected the rules, avoided something, or met the request. Never add a closing note, "
    "self-assessment, or summary about what you wrote. Start directly with the required prompt text. "
    "Never repeat, quote, paraphrase, or answer any instruction, rule, constraint, policy, or note from this system prompt; "
    "the output must contain only the prompt text for the target model."
)

_META_PATTERNS = (
    r"^#{1,6}\s",
    r"^(note|remember|important|rule|constraint|instruction|instructions|output|response)\b\s*[:\-]",
    r"^(here is|here's|this is)\b[^.!?]{0,40}\bprompt\b",
    r"^(okay|ok|sure|certainly|alright|absolutely|great|perfect)\s*[,!:.]\s*",
    r"\b(explicitly stated|as requested|as instructed|anything other than)\b",
    r"\b(the user's (request|text|wording|prompt)|system prompt|this prompt|the generated prompt|output contract|content policy|negative constraints?)\b",
    r"\b(output only|only output|in the output)\b",
    r"\b(the prompt|the response|the answer)\s+(should|must|is|will|needs to)\b",
    r"\b(do not|don't|never|avoid|must not|should not)\b[^.!?]{0,80}\b(add|include|mention|name|output|write|describe|use|state|say|deviate|copy|repeat|paraphrase)\b",
    r"\b(I will|I'll|we need to|we should|the goal is|my task|the task is)\b",
    r"\b(here is|here's|below is)\s+(the|your|an?)\s+(enhanced|final|generated|resulting)?\s*(prompt|output|response)\b",
    r"\b(i'?ve|i have|we'?ve|we have)\b[^.!?]{0,90}\b(done|followed|listened|obeyed|applied|respected|honou?red|ensured|made sure|crafted|created|generated|written|wrote|included|avoided|kept|used|focused|paid attention)\b",
    r"\b(i|we)\b[^.!?]{0,90}\b(followed|listened|obeyed|applied|respected|honou?red|ensured|made sure|crafted|generated|avoided|paid attention)\b[^.!?]{0,60}\b(instruction|instructions|rules?|constraints?|guidelines?|request|prompt)\b",
    r"\b(i|we)\s+(made sure|ensured)\b",
    r"\b(done a good job|did a good job|good job|well done|task complete|successfully completed)\b",
    r"\b(the|this|your)\s+(enhanced|final|generated|resulting)?\s*(prompt|output|response)\s+(now\s+)?(is|includes|contains|follows|respects|meets|has been|was|will)\b",
    r"\b(the user|user'?s)\s+(asked|requested|wants|wanted|instructions?)\b",
    r"\b(i hope|hope (this|that) helps|let me know|feel free|if you (want|need|would like))\b",
    r"\b(note|disclaimer|warning)\s*[:\-]",
    r"\b(i|we)\s+(will|would|can|am going to|are going to)\s+(now\s+)?(write|create|generate|provide|describe)\b",
)
_META_RE = re.compile("|".join(f"(?:{p})" for p in _META_PATTERNS), re.I)

_CHAT_MARKER_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|begin_of_text|start_header_id|end_header_id|eot_id|end|system|user|assistant|model|think|/think)\|>\s*(?:system|user|assistant|model)?\s*>?",
    re.I,
)
_CHAT_TAG_RE = re.compile(
    r"</?(?:start_of_turn|end_of_turn|bos|eos|pad|unk|s)>|\[/?INST\]|<<?/?SYS>>|<\|?/?channel\|?>",
    re.I,
)
_ROLE_LINE_RE = re.compile(r"^(system|user|assistant|model)\s*:\s*", re.I)


def _extract_last_turn(text):
    """Keep only the final assistant/model turn when template markers are present."""
    patterns = (
        (r"<\|im_start\|>\s*(system|user|assistant|model)\s*>?", r"<\|im_end\|>", ("assistant", "model")),
        (r"<start_of_turn>\s*(system|user|model|assistant)\s*>?", r"<end_of_turn>", ("model", "assistant")),
        (r"<\|start_header_id\|>\s*(system|user|assistant)\s*<\|end_header_id\|>[ \t]*", r"<\|eot_id\|>", ("assistant",)),
    )
    for start_pattern, end_pattern, roles in patterns:
        matches = list(re.finditer(start_pattern, text, re.I))
        if not matches:
            continue
        for match in reversed(matches):
            if match.group(1).lower() in roles:
                tail = text[match.end():]
                end = re.search(end_pattern, tail)
                return tail[:end.start()] if end else tail
        return ""
    return text


def _strip_chat_tokens(text):
    """Remove ChatML/Gemma/Llama role markers and keep the assistant answer."""
    text = _extract_last_turn(text or "")
    text = _CHAT_MARKER_RE.sub("", text)
    text = _CHAT_TAG_RE.sub("", text)
    lines = text.splitlines()
    role_positions = []
    assistant_positions = []
    for index, line in enumerate(lines):
        role = line.strip().lower().rstrip(":")
        if role in ("system", "user", "assistant", "model"):
            role_positions.append(index)
            if role in ("assistant", "model"):
                assistant_positions.append(index)
    if role_positions:
        if not assistant_positions:
            return ""
        lines = lines[assistant_positions[-1] + 1:]
    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if stripped.lower().rstrip(":") in ("system", "user", "assistant", "model"):
            continue
        kept_lines.append(_ROLE_LINE_RE.sub("", stripped))
    cleaned = "\n".join(kept_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

_NEGATIVE_PROMPT_TERMS = (
    "blur", "blurry", "watermark", "logo", "extra fingers", "extra limbs",
    "bad anatomy", "low quality", "worst quality", "deformed", "sweat",
    "wetness", "moisture", "droplets",
)
_NEGATIVE_PROMPT_RE = re.compile(
    r"\b(no|without|avoid|free of|don't|do not|must not|should not)\b[^.!?]{0,50}\b("
    + "|".join(re.escape(term) for term in _NEGATIVE_PROMPT_TERMS)
    + r")\b",
    re.I,
)


def _clean_prompt_output(text):
    """Remove leaked instructions, reasoning and self-commentary while keeping the prompt."""
    text = _strip_chat_tokens(text)
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if re.match(r"^negative prompt\s*[:\-]", stripped, re.I):
            continue
        stripped = re.sub(r"^(?:enhanced prompt|positive prompt|prompt)\s*[:\-]\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"^(?:okay|ok|sure|certainly|alright|absolutely|great|perfect)\s*[,!:.]\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"^(?:here is|here's|below is)\s+(?:the|your|an?)\s+(?:enhanced|final|generated|resulting)?\s*(?:prompt|output|response)\s*[:\-]?\s*", "", stripped, flags=re.I)
        if not stripped:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        kept = [sentence.strip() for sentence in sentences if sentence.strip() and not _META_RE.search(sentence)]
        if kept:
            kept_lines.append(" ".join(kept))
    cleaned = "\n".join(kept_lines)
    cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip() or re.search(r"[A-Za-z0-9]", line))
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned.strip('"').strip("`").strip()

THINK_TAGS = (
    ("\u27e8think\u27e9", "\u27e8/think\u27e9"),
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
)

REFUSAL_MARKERS = (
    "i cannot fulfill", "i can't fulfill", "i cannot comply", "i can't comply",
    "i cannot assist", "i can't assist", "i cannot help", "i can't help",
    "i cannot create", "i can't create", "i cannot generate", "i can't generate",
    "cannot provide", "can't provide", "unable to comply", "unable to help",
    "i must decline", "i have to decline", "i won't", "i will not",
    "as an ai", "against my guidelines", "violates my", "not able to help",
    "not appropriate", "inappropriate content", "sorry, but i can",
    "i'm sorry, but", "i am sorry, but", "i apologize, but",
)


def _is_repetitive_text(text):
    """True for degenerate loops; structured prompts that reuse a few phrases are fine."""
    if not text or len(text) < 60:
        return False
    normalized = re.sub(r"\s+", " ", text.lower()).strip()[:8000]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if len(s.strip()) >= 30]
    if sentences:
        counts = Counter(sentences)
        # a long prompt may reuse a sentence or two; a loop is mostly copies
        if max(counts.values()) >= 5 or sum(len(s) * (c - 1) for s, c in counts.items()) > 0.3 * len(normalized):
            return True
    words = re.findall(r"\w+", normalized)
    for size, min_count, share in ((8, 3, 0.5), (5, 4, 0.75)):
        if len(words) < size * 4:
            continue
        grams = [" ".join(words[i:i + size]) for i in range(len(words) - size + 1)]
        counts = Counter(grams)
        covered = set()
        for i, gram in enumerate(grams):
            if counts[gram] >= min_count:
                covered.update(range(i, i + size))
        if len(covered) > share * len(words):
            return True
    lines = [line.strip().lower() for line in text.splitlines() if len(line.strip()) >= 30]
    if lines:
        counts = Counter(lines)
        if max(counts.values()) >= 5 or sum(len(l) * (c - 1) for l, c in counts.items()) > 0.3 * len(text):
            return True
    return False


def _dedupe_repeats(text):
    """Drop repeated sentences and lines, keeping the first occurrence."""
    seen = set()
    lines = []
    for line in text.splitlines():
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            key = re.sub(r"\s+", " ", sentence.lower()).strip()
            if len(key) >= 30:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(sentence)
        lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_VIDEO_FIELD_RE = re.compile(
    r"^(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music|soundscape|audio(?:_design)?|sound_effects?|music)\s*:",
    re.I,
)
_VIDEO_SENTENCE_RE = re.compile(
    r"\b(?:non-diegetic music|soundscape|sound effects?|audio cue|voice-?over|background music|diegetic sound)\b",
    re.I,
)
_SHOT_LINE_RE = re.compile(r"^\s*(?:\[\s*)?shot\s+\d+", re.I)


def _strip_video_meta(text):
    """Remove video/audio fields from image-only presets."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _VIDEO_FIELD_RE.match(stripped) or _SHOT_LINE_RE.match(stripped):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        kept = [sentence for sentence in sentences if not _VIDEO_SENTENCE_RE.search(sentence)]
        if kept:
            lines.append(" ".join(kept))
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _next_seed(seed):
    return (int(seed) + 1) % (2 ** 32 - 1) or 1


_TOP_P_WARNED = False


def _clip_top_p_supported():
    try:
        major, minor = (int(part) for part in str(torch.__version__).split("+")[0].split(".")[:2])
        return (major, minor) < (2, 13)
    except Exception:
        return True


_REASONING_OPENER_RE = re.compile(
    r"^\W*(?:okay|ok|alright|let me|let's|first,|the user (?:wants|asked|is asking|provided)|i need to|we need to|i should|"
    r"breaking (?:this|it) down|to (?:start|begin) with|so,? the)\b",
    re.I,
)


def _bad_output_reason(text, min_chars=20):
    """Return a short reason when output is empty, short, degenerate, leaked reasoning or a refusal."""
    answer, _ = _split_thinking(text or "")
    if _REASONING_OPENER_RE.match(answer):
        return "reasoning leak"
    clean = _clean_prompt_output(answer)
    if len(clean) < min_chars:
        return "too short"
    if _is_repetitive_text(clean):
        return "repetitive"
    if _NEGATIVE_PROMPT_RE.search(clean):
        return "negative-prompt wording"
    if re.search(r"(.)\1{15,}", clean):
        return "runaway repeat"
    if len(clean) > 200 and len(set(clean)) < 8:
        return "low variety"
    words = re.findall(r"\w+", clean.lower())
    if len(words) > 30:
        if Counter(words).most_common(1)[0][1] / len(words) > 0.45:
            return "repeated words"
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if len(lines) >= 4:
        if Counter(lines).most_common(1)[0][1] >= 4:
            return "repeated lines"
    lowered = clean.lower()
    if len(clean) < 600 and any(marker in lowered for marker in REFUSAL_MARKERS):
        return "refusal"
    if any(lowered.startswith(marker) for marker in REFUSAL_MARKERS):
        return "refusal"
    return ""


def _generate_checked(attempt, seed, min_chars, auto_retry, tries=3):
    """Run attempt(seed) until the output passes the checks, up to `tries` seeds; keep the best on failure."""
    texts = []
    for index in range(tries if auto_retry else 1):
        text = attempt(seed)
        reason = _bad_output_reason(_split_thinking(text)[0], min_chars)
        if not reason:
            return text
        texts.append(text)
        seed = _next_seed(seed)
        if index + 1 < tries and auto_retry:
            print(f"[PromptForge] output rejected ({reason}); retrying with a new seed ({index + 2}/{tries}).")
    if reason == "repetitive":
        return _dedupe_repeats(max(texts, key=len))
    return texts[-1]


# MiniMax H3 modes: how many images the mode can use, whether a reference video is allowed, and each image's role.
# Labels follow the H3 text encoder: <Picture i> / <Video k> are 1-based per type in connection order, gaps closed up.
_MINIMAX_MODES = {
    "fl2va": (2, False, ("the exact first frame", "the exact last frame")),
    "i2va": (1, False, ("the exact first frame",)),
    "l2va": (1, False, ("the exact last frame",)),
    "ref2va": (None, True, ()),
    "t2va": (0, False, ()),
}
_MINIMAX_MODE_RE = re.compile(r"minimax h3 (" + "|".join(_MINIMAX_MODES) + r")\b", re.I)


# languages a video family speaks reliably (third-party model cards; other languages work partially)
DIALOGUE_LANGUAGES = ("Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish")
DIALOGUE_AUTO = "auto (from the instructions)"
DIALOGUE_OTHER = "other (type below)"
# these are counted in characters; every other language in words
CHARACTER_LANGUAGES = {"Chinese", "Japanese", "Korean"}


def _dialogue_language(language, other):
    """The language name to write in, or '' when it is left to the instructions."""
    if language == DIALOGUE_OTHER:
        return (other or "").strip()
    return "" if language == DIALOGUE_AUTO else language


def _dialogue_block(instructions, language, other, seconds):
    """System prompt section asking for dialogue in the video prompt, sized to what fits in `seconds`; '' when off."""
    instructions = (instructions or "").strip()
    chosen = _dialogue_language(language, other)
    if not instructions and not chosen:
        return ""
    unit, rate = ("characters", 4.0) if chosen in CHARACTER_LANGUAGES else ("words", 2.3)
    budget = max(3, round(max(seconds - 1.0, 1.0) * rate))
    name = chosen or "LANGUAGE"
    where = f"in {chosen}" if chosen else "in the language the dialogue instructions imply (English when they do not say)"
    topic = f"Dialogue instructions from the user: {instructions}" if instructions else "The user gave no dialogue instructions: invent short natural dialogue that fits the scene."
    return f"""# Dialogue (required in this video)
{topic}
Write every spoken line {where}. Only the spoken words are in that language; the rest of the prompt stays in English. The words are what a native speaker would really say, never a translation.

Exact shape of every spoken line, copy it:
the young woman with a quiet, breathy voice (S1) says: <d>[{name}] the spoken words in {name}</d>

Rules, all mandatory:
1. The words are wrapped in <d> and </d>. The closing </d> is never left out.
2. Directly after <d> comes the language tag in square brackets, [{name}], then a space, then the words. The tag always holds the English name of the language, spelled the same way every time, even when the words are not English.
3. Between the tags there is only speech in {name}: no quotation marks, no stage directions, no translation, no English words, no emoji.
4. Name the speaker before every line with the ID: first appearance with a short voice description ("the tired baker with a low, gravelly voice (S1) says:"), then "(S1) says:" or "(S2) answers:". People who do not speak get no ID.
5. Write each line inside the shot where it is spoken, at its timestamp.
6. Timing: the video lasts {seconds:g} seconds and people speak about {rate:g} {unit} per second. Leave a second before the first line and a short pause between speakers, so all speakers together say at most {budget} {unit}. Every line ends before the video does. Short natural exchanges beat speeches; say less when the topic needs more than fits.

Wrong, never write: She says "<words>"   |   <d><words></d> (no language tag)   |   <d>[{name}] <words> (no closing tag)   |   <d>[{name}] <words in English></d> (words in the wrong language)   |   [{name}] <d><words></d> (tag outside)
Before you answer, check every <d>: it has [{name}] right after it, the words are {name}, and </d> closes it."""


DIALOGUE_TAG = re.compile(r"<d>\s*\[([^\]]*)\]\s*(.+?)\s*</d>", re.S)


def _dialogue_problem(text, language):
    """What is wrong with the dialogue tags in `text`, or '' when they are right. `language` '' skips the language check."""
    tags = DIALOGUE_TAG.findall(text)
    if not tags:
        return "no <d>[Language] words</d> dialogue line was written" if "<d>" not in text and "</d>" not in text else "the <d> tags are incomplete: every line needs <d>[Language] words</d>"
    if text.count("<d>") != text.count("</d>") or text.count("<d>") != len(tags):
        return "some <d> tags are missing their [Language] tag or their closing </d>"
    if language:
        wrong = next((tag for tag, _ in tags if tag.strip().lower() != language.lower()), None)
        if wrong is not None:
            return f"a language tag says [{wrong}] but it must be [{language}]"
    return ""


FAMILY_PRESETS = {
    "krea2": "Krea 2 T2I",
    "moss_image": "MOSS Image T2I",
    "qwen_image": "Qwen Image T2I",
    "anima": "Anima Tags",
    "ace": "ACE-Step 1.5",
}


def _family_preset(family, has_images, has_video):
    """Preset for a model family when the preset is on auto; MiniMax H3 follows the connected media."""
    if family == "minimax_h3":
        return "MiniMax H3 Ref2VA" if has_images or has_video else "MiniMax H3 T2VA"
    return FAMILY_PRESETS.get(family, DEFAULT_PRESET)


def _minimax_mode(preset_name):
    match = _MINIMAX_MODE_RE.search(preset_name or "")
    return match.group(1).lower() if match else None


def _minimax_alignment_line(mode_key, picture_count, seconds):
    """The frame-alignment line the frame modes open with, limited to the pictures that are attached ('' = omit it)."""
    if not picture_count:
        return ""
    head = "How the reference pictures align with the target video — "
    if mode_key == "i2va":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if mode_key == "l2va":
        return head + f"<Picture 1> (from [Shot N]) aligns with the {seconds}-second mark of the target video."
    if mode_key == "fl2va":
        first = "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video"
        if picture_count == 1:
            return head + first + "."
        return head + first + f"; Picture 2 (from Shot N) aligns with the {seconds}-second mark of the target video."
    return ""


_MINIMAX_REF_RE = re.compile(r"\b(Picture|Video|Audio)\s+(\d+)\b")


def _drop_absent_references(text, picture_count, has_video, user_text):
    """Remove sentences citing a picture, video or audio label that is not attached (audio the user's own text names is allowed)."""
    def absent(match):
        kind, number = match.group(1), int(match.group(2))
        if kind == "Picture":
            return number > picture_count
        if kind == "Video":
            return number > (1 if has_video else 0)
        return not re.search(rf"\bAudio\s+{number}\b", user_text)

    removed = 0
    lines = []
    for line in text.splitlines():
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if any(absent(m) for m in _MINIMAX_REF_RE.finditer(sentence)):
                removed += 1
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    if removed:
        print(f"[PromptForge] removed {removed} sentence(s) that cited media which is not connected.")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _minimax_inputs_block(mode_key, picture_count, video_frames, dropped, seconds):
    """Authoritative list of what exists for this MiniMax mode; everything else must never be mentioned."""
    max_images, allows_video, roles = _MINIMAX_MODES[mode_key]
    lines = ["# MiniMax inputs (authoritative)", f"Mode: {mode_key.upper()}."]
    present = []
    for index in range(picture_count):
        role = f" ({roles[index]})" if index < len(roles) else ""
        present.append(f"<Picture {index + 1}>{role}")
    if video_frames:
        present.append(f"<Video 1> (one reference video, {video_frames} frames sampled in time order)")
    lines.append("Attached: " + (", ".join(present) if present else "nothing") + ".")
    absent = []
    if max_images is None:
        absent.append(f"<Picture {picture_count + 1}> and any higher picture number")
    else:
        absent.extend(f"<Picture {index + 1}> ({roles[index]})" for index in range(picture_count, max_images))
        if max_images == 0 or picture_count >= max_images:
            absent.append(f"<Picture {max(picture_count, max_images) + 1}> and any higher picture number")
    if not video_frames:
        absent.append("<Video 1> and any other video")
    absent.append("any <Audio N> that the user's text does not itself describe (this node attaches no audio)")
    lines.append("Not attached, so never mention, cite or invent: " + "; ".join(absent) + ".")
    lines.append(
        "Where the instructions above refer to something that is not attached, ignore that part and work only from the user's text. "
        "Use each attached label exactly as written and never renumber it."
    )
    if mode_key in ("fl2va", "i2va", "l2va"):
        alignment = _minimax_alignment_line(mode_key, picture_count, seconds)
        if alignment:
            lines.append(f"Alignment line: write exactly this, replacing N with the number of the last shot, and name no other picture in it: {alignment}")
        else:
            lines.append("Alignment line: nothing is attached, so omit the alignment line entirely and start with the first field.")
    if dropped:
        lines.append("Connected but unused by this mode (do not mention): " + ", ".join(dropped) + ".")
    return "\n".join(lines)


def _tokens_text(clip, tokens):
    batch = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    return clip.decode([t[0] for row in batch for t in row if isinstance(t[0], int)], skip_special_tokens=False)


def _tokens_have_images(tokens):
    if isinstance(tokens, dict):
        if tokens.get("type") == "image":
            return True
        return any(_tokens_have_images(value) for value in tokens.values())
    if isinstance(tokens, (list, tuple)):
        return any(_tokens_have_images(value) for value in tokens)
    return False


def _split_thinking(text):
    """Remove reasoning blocks from model output and return (answer, thinking).

    The opening think tag can come from the prompt template rather than the
    generated text, so a stray closing tag also splits reasoning from answer.
    """
    answer = text or ""
    thoughts = []
    channel_thought = re.findall(
        r"(?is)<\|channel>\s*thought\b(.*?)(?=<\|channel>\s*final\b|<channel\|>|$)", answer
    )
    thoughts.extend(part.strip() for part in channel_thought if part.strip())
    answer = re.sub(
        r"(?is)<\|channel>\s*thought\b.*?(?=<\|channel>\s*final\b|<channel\|>|$)", "", answer
    )
    channel_final = re.search(r"(?is)<\|channel>\s*final\b(.*?)(?:<channel\|>|$)", answer)
    if channel_final:
        answer = channel_final.group(1)
    answer = answer.replace("<|channel>", "").replace("<channel|>", "")
    for open_tag, close_tag in THINK_TAGS:
        pattern = r"(?is)" + re.escape(open_tag) + r".*?" + re.escape(close_tag)
        for match in re.finditer(pattern, answer):
            inner = match.group(0).replace(open_tag, "").replace(close_tag, "").strip()
            if inner:
                thoughts.append(inner)
        answer = re.sub(pattern, "", answer)

    lowered = answer.lower()
    split_idx = -1
    split_len = 0
    for _, close_tag in THINK_TAGS:
        idx = lowered.rfind(close_tag.lower())
        if idx > split_idx:
            split_idx, split_len = idx, len(close_tag)
    if split_idx != -1:
        head = answer[:split_idx].strip()
        if head:
            thoughts.append(head)
        answer = answer[split_idx + split_len:]
    else:
        open_idx = -1
        open_len = 0
        for open_tag, _ in THINK_TAGS:
            idx = lowered.rfind(open_tag.lower())
            if idx > open_idx:
                open_idx, open_len = idx, len(open_tag)
        if open_idx != -1:
            tail = answer[open_idx + open_len:].strip()
            if tail:
                thoughts.append(tail)
            answer = answer[:open_idx]
    return answer.strip(), "\n\n".join(thoughts).strip()


DEFAULT_PRESET = "Enhance"

HERETIC_MODELS = PROMPTFORGE_MODELS.get("heretic_hf_models") or []
GGUF_MODELS = PROMPTFORGE_MODELS.get("gguf_models") or []
LOCAL_SCAN_DIRS = PROMPTFORGE_MODELS.get("local_scan_dirs") or []

def _safetensors_is_decoder(path):
    """True when a single safetensors file holds a decoder LLM (has token embeddings)."""
    try:
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    except (OSError, ValueError, struct.error):
        return False
    return any(key.endswith("embed_tokens.weight") for key in header)


def _scan_roots():
    """Directories to scan: every registered text_encoders folder plus the catalog subdirs of models_dir."""
    roots = []
    if folder_paths is not None:
        for folder in folder_paths.get_folder_paths("text_encoders"):
            roots.append(Path(folder))
        models_dir = Path(folder_paths.models_dir)
        roots.extend(models_dir / sub for sub in LOCAL_SCAN_DIRS)
    seen = set()
    unique = []
    for root in roots:
        if root.is_dir() and str(root) not in seen:
            seen.add(str(root))
            unique.append(root)
    return unique


def _scan_local_text_models():
    """Scan ComfyUI text-model directories for local LLM / text encoder models.

    Multi-part checkpoints collapse to their containing HF directory, mmproj
    vision sidecars and non-generative encoders (CLIP, T5, projections) are
    skipped. The picker option is the path relative to models_dir so a
    workflow saved on one machine resolves on another with the same layout.
    """
    found = []
    seen_paths = set()
    models_dir = Path(folder_paths.models_dir) if folder_paths is not None else None
    for root in _scan_roots():
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in (".safetensors", ".gguf", ".bin"):
                continue
            if p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower():
                continue
            if p.suffix.lower() in (".safetensors", ".bin") and (p.parent / "config.json").is_file():
                model_path = p.parent
            elif p.suffix.lower() == ".safetensors" and not _safetensors_is_decoder(p):
                continue
            else:
                model_path = p
            key = str(model_path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            option = key
            if models_dir is not None and model_path.is_relative_to(models_dir):
                option = str(model_path.relative_to(models_dir))
            found.append({"path": key, "option": option, "name": model_path.name, "kind": "gguf" if p.suffix.lower() == ".gguf" else "safetensors"})
    return found


LOCAL_TEXT_MODELS = _scan_local_text_models()
_SCANNED_GGUF_NAMES = {Path(m.get("path") or "").name for m in LOCAL_TEXT_MODELS if str(m.get("path") or "").lower().endswith(".gguf")}

def _catalog_entry(item, kind):
    return {
        "option": item.get("label") or item.get("name") or item.get("key"),
        "kind": kind,
        "key": item.get("key"),
        "params_b": item.get("params_b"),
        "avail_match": item.get("avail_match") or [],
        "repo_id": item.get("repo_id"),
        "base_dir": item.get("base_dir"),
        "quant_files": item.get("quant_files") or {},
        "local_path": item.get("local_path") or "",
    }


HERETIC_MODEL_ENTRIES = (
    [_catalog_entry(item, "hf") for item in HERETIC_MODELS]
    + [_catalog_entry(item, "gguf") for item in GGUF_MODELS]
)


def _cache_has(repo_id, filename):
    if not repo_id or not filename:
        return False
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    hub = Path(cache) if cache else Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snapshot / filename).is_file() for snapshot in snapshots.iterdir())


def _gguf_local_quants(entry):
    quant_files = entry.get("quant_files") or {}
    if not quant_files:
        return []
    base_dir = Path(entry.get("base_dir") or "LLM/GGUF")
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", "") or ""
        if models_dir:
            base_dir = Path(models_dir) / base_dir
    scanned = _SCANNED_GGUF_NAMES
    repo_id = entry.get("repo_id")
    return [
        quant for quant, filename in quant_files.items()
        if (base_dir / filename).is_file() or filename in scanned or _cache_has(repo_id, filename)
    ]


def _entry_local_path(entry):
    direct = entry.get("local_path") or ""
    if direct and Path(direct).exists():
        return direct
    matches = entry.get("avail_match") or []
    if isinstance(matches, str):
        matches = [matches]
    matches = [m.lower() for m in matches if m]
    if not matches:
        return ""
    for m in LOCAL_TEXT_MODELS:
        name = (m.get("name") or "").lower()
        if not all(token in name for token in matches):
            continue
        path = m.get("path") or ""
        if entry.get("kind") == "hf" and path.lower().endswith(".gguf"):
            continue
        if entry.get("kind") == "gguf" and not path.lower().endswith(".gguf"):
            continue
        return path
    return ""


def _entry_label(entry):
    """(picker label, is_local): [LOCAL ...] for models on disk or in the HF cache, [DOWNLOAD] for the rest."""
    label = entry["option"]
    if entry.get("kind") == "gguf":
        quants = _gguf_local_quants(entry)
        if quants:
            ordered = [q for q in ("q4", "q5", "q6", "q8") if q in quants]
            return f"[LOCAL {'/'.join(q.upper() for q in ordered)}] {label}", True
        return f"[DOWNLOAD] {label}", False
    if _entry_local_path(entry) or _cache_has(entry.get("repo_id"), "config.json"):
        return f"[LOCAL] {label}", True
    return f"[DOWNLOAD] {label}", False


def _source_options(kind):
    """Picker labels for one source, local models first: catalog entries on disk, scanned files, then downloads."""
    labelled = [_entry_label(e) for e in HERETIC_MODEL_ENTRIES if e["kind"] == kind]
    local = [label for label, is_local in labelled if is_local]
    local += [f"[LOCAL] {e['option']}" for e in LOCAL_TEXT_MODELS if e["kind"] == ("gguf" if kind == "gguf" else "safetensors")]
    remote = [label for label, is_local in labelled if not is_local]
    return local, remote


LOCAL_SAFETENSORS_OPTIONS, REMOTE_SAFETENSORS_OPTIONS = _source_options("hf")
LOCAL_GGUF_OPTIONS, REMOTE_GGUF_OPTIONS = _source_options("gguf")
SAFETENSORS_MODEL_OPTIONS = (LOCAL_SAFETENSORS_OPTIONS + REMOTE_SAFETENSORS_OPTIONS) or ["none"]
GGUF_MODEL_OPTIONS = (LOCAL_GGUF_OPTIONS + REMOTE_GGUF_OPTIONS) or ["none"]

PRESET_OPTIONS = ["auto"] + sorted(
    k for k, v in PROMPTFORGE_PRESETS.items()
    if isinstance(v, dict) and (v.get("text") or v.get("system_prompt"))
)

MODE_OPTIONS = ["text_only", "inputs"]
QUANT_OPTIONS = ["auto", "fp16", "q8", "q6", "q5", "q4"]
GGUF_QUANT_ORDER = ("q8", "q6", "q5", "q4")


def _gguf_quant_order(quantization, quant_files):
    requested = {
        "q8": "q8", "8bit": "q8", "q6": "q6", "6bit": "q6",
        "q5": "q5", "5bit": "q5", "q4": "q4", "4bit": "q4",
        "fp16": "q8", "auto": "q8",
    }.get(quantization or "auto", "q8")
    order = [requested] + [key for key in GGUF_QUANT_ORDER if key != requested]
    return [key for key in order if key in quant_files]


_STATUS_PREFIX_RE = re.compile(r"^\[(?:AVAILABLE|LOCAL|DOWNLOAD)[^\]]*\]\s*", re.I)


def _bare_option(option):
    return _STATUS_PREFIX_RE.sub("", str(option or ""))


def _resolve_model_entry(option):
    """Map a friendly model option to a runtime descriptor; the '[LOCAL ...] ' prefix is stripped first."""
    if not option or option == "auto":
        ent = HERETIC_MODELS[0] if HERETIC_MODELS else {}
        first = HERETIC_MODEL_ENTRIES[0] if HERETIC_MODEL_ENTRIES else {}
        return {"kind": "hf", "key": (ent or {}).get("key") or "heretic_qwen35_9b", "params_b": (ent or {}).get("params_b") or 9, "repo_id": (ent or {}).get("repo_id"), "local_path": _entry_local_path(first) if first else ""}
    label = _bare_option(option)
    for e in HERETIC_MODEL_ENTRIES:
        if e["option"] == label:
            if e["kind"] == "hf":
                ent = next((m for m in HERETIC_MODELS if m.get("key") == e["key"]), {})
                return {"kind": "hf", "key": e["key"], "params_b": e["params_b"], "repo_id": ent.get("repo_id"), "local_path": _entry_local_path(e)}
            ent = next((g for g in GGUF_MODELS if g.get("key") == e["key"]), {})
            return {"kind": "gguf", "key": e["key"], "params_b": e["params_b"], "entry": ent}
    return None


def _local_model_option(option):
    """Return the absolute path when the picker option is a scanned local model (relative label or old absolute path)."""
    opt = _bare_option(option)
    if not opt or opt == "auto":
        return ""
    for e in LOCAL_TEXT_MODELS:
        if opt in (e["option"], e["path"]):
            return e["path"]
    return ""


def _hf_dtype_kwargs(dtype):
    """transformers 4.56+ renamed `torch_dtype` to `dtype` on from_pretrained."""
    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        if (major, minor) >= (4, 56):
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def _hf_compute_dtype():
    """bf16 where the GPU supports it; Gemma/Qwen logits overflow in fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _hf_max_memory(reserve=3 * 1024 ** 3):
    """GPU budget for device_map='auto', leaving headroom for activations and KV cache."""
    try:
        free_bytes = torch.cuda.mem_get_info()[0]
    except Exception:
        return None
    budget = max(int(free_bytes) - reserve, 768 * 1024 ** 2)
    return {torch.cuda.current_device(): budget, "cpu": psutil.virtual_memory().available}


def _vision_size(width, height, max_pixels=1024 * 1024):
    """Target (w, h) for references above ~1 megapixel, else None. Vision encoders cost about 1k tokens per megapixel, and every token also slows each generated token."""
    if width * height <= max_pixels:
        return None
    scale = (max_pixels / (width * height)) ** 0.5
    return max(32, round(width * scale)), max(32, round(height * scale))


def _free_vram(keep=None, needed=10 ** 30):
    """Unload ComfyUI's models (least recently used first) until `needed` bytes are free."""
    try:
        import comfy.model_management as model_management
        keep_loaded = []
        if keep is not None:
            keep_loaded = [m for m in model_management.current_loaded_models if m.model is keep]
        model_management.free_memory(needed, model_management.get_torch_device(), keep_loaded=keep_loaded)
        if needed >= 10 ** 30:
            model_management.free_memory(needed, torch.device("cpu"))
        model_management.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


_GGUF_VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_skip(f, vtype):
    if vtype == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == 9:
        elem = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if elem == 8:
            for _ in range(count):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        else:
            f.seek(_GGUF_VALUE_SIZE[elem] * count, 1)
    else:
        f.seek(_GGUF_VALUE_SIZE[vtype], 1)


def _gguf_read(f, vtype):
    if vtype == 8:
        return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
    if vtype == 9:
        elem = struct.unpack("<I", f.read(4))[0]
        return [_gguf_read(f, elem) for _ in range(struct.unpack("<Q", f.read(8))[0])]
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _gguf_metadata(path):
    """Read GGUF metadata without mapping the tensor data."""
    meta = {}
    needed = None
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return meta
        f.read(4)
        f.read(8)
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            if needed is not None and key not in needed:
                _gguf_skip(f, vtype)
                continue
            value = _gguf_read(f, vtype)
            meta[key] = value
            if key == "general.architecture":
                needed = {f"{value}.{k}" for k in (
                    "block_count", "attention.head_count", "attention.head_count_kv",
                    "embedding_length", "attention.key_length", "attention.value_length",
                    "attention.sliding_window",
                )}
                needed.add("tokenizer.chat_template")
            if needed is not None and needed <= meta.keys():
                break
    return meta


def _gguf_kv_profile(path):
    """Return (layers, kv bytes per token, swa window or None) from GGUF metadata."""
    try:
        meta = _gguf_metadata(path)
        prefix = f"{meta['general.architecture']}."
        swa = meta.get(f"{prefix}attention.sliding_window")
        n_layers = int(meta[f"{prefix}block_count"])
        n_heads = int(meta[f"{prefix}attention.head_count"])
        kv_heads = meta[f"{prefix}attention.head_count_kv"]
        embd = int(meta[f"{prefix}embedding_length"])
        key_len = meta.get(f"{prefix}attention.key_length", embd // n_heads)
        value_len = meta.get(f"{prefix}attention.value_length", embd // n_heads)
        def per_layer(value):
            if isinstance(value, list):
                return [int(v) for v in value] if len(value) == n_layers else None
            return [int(value)] * n_layers
        kv_heads_l = per_layer(kv_heads)
        key_l = per_layer(key_len)
        value_l = per_layer(value_len)
        if kv_heads_l is None or key_l is None or value_l is None:
            return None
        kv_bytes = sum(kv_heads_l[i] * (key_l[i] + value_l[i]) * 2 for i in range(n_layers))
        return n_layers, kv_bytes, int(swa) if swa is not None else None
    except Exception:
        return None


def _gguf_memory_estimate(path, n_ctx, layers, mmproj_path=None):
    """Approximate VRAM needed for the GPU-resident part of a GGUF model."""
    weights = path.stat().st_size
    extra = mmproj_path.stat().st_size + 1024 ** 3 if mmproj_path is not None else 0  # + image encoder compute buffers
    profile = _gguf_kv_profile(path)
    if profile is None:
        return weights + extra + 768 * 1024 ** 2
    n_layers, kv_per_token, swa = profile
    if swa is not None:
        kv_bytes = kv_per_token * (min(n_ctx, swa) * 0.95 + n_ctx * 0.05)
    else:
        kv_bytes = kv_per_token * n_ctx
    if layers == 0:
        return extra + 768 * 1024 ** 2
    fraction = 1.0 if layers < 0 else min(1.0, max(0.0, layers / n_layers))
    return (weights + kv_bytes) * fraction + extra + 768 * 1024 ** 2


def _gguf_layers_that_fit(path, n_ctx, mmproj_path=None):
    """GPU layer count for the VRAM that is free right now (other processes included)."""
    try:
        free_bytes = torch.cuda.mem_get_info()[0]
    except Exception:
        return 0
    profile = _gguf_kv_profile(path)
    if profile is None:
        return 0
    n_layers, kv_per_token, swa = profile
    kv = kv_per_token * (min(n_ctx, swa) * 0.95 + n_ctx * 0.05) if swa is not None else kv_per_token * n_ctx
    extra = mmproj_path.stat().st_size if mmproj_path is not None else 0
    cost = (path.stat().st_size + kv) / n_layers
    return max(0, min(n_layers, int((free_bytes - extra - 768 * 1024 ** 2) / cost)))


def _gpu_budget_bytes():
    try:
        total = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return 0
    return max(int(total) - 2 * 1024 ** 3, 1024 ** 3)


def _auto_batch(budget):
    if budget >= 12 * 1024 ** 3:
        return 2048, 512
    if budget >= 8 * 1024 ** 3:
        return 1024, 256
    return 512, 128


def _auto_gguf_fit(path, mmproj_path, budget):
    profile = _gguf_kv_profile(path)
    if profile is None:
        return None
    n_layers, kv_per_token, swa = profile
    weights = path.stat().st_size
    extra = mmproj_path.stat().st_size if mmproj_path is not None else 0
    scales = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.28}
    for ctx in (16384, 12288, 8192, 6144, 4096):
        for kind in ("f16", "q8_0", "q4_0"):
            kv = kv_per_token * (min(ctx, swa) * 0.95 + ctx * 0.05) if swa is not None else kv_per_token * ctx
            if weights + kv * scales[kind] + extra <= budget:
                return {"context_size": ctx, "gpu_layers": -1, "kv_cache_type": kind, "offload_kqv": True}
    ctx, kind = 4096, "q8_0"
    kv = kv_per_token * (min(ctx, swa) * 0.95 + ctx * 0.05) if swa is not None else kv_per_token * ctx
    cost = (weights + kv * scales[kind] + extra) / n_layers
    layers = max(0, int((budget - extra - 512 * 1024 ** 2) / cost)) if cost > 0 else 0
    return {"context_size": ctx, "gpu_layers": layers, "kv_cache_type": kind, "offload_kqv": True}


def _auto_gguf_settings(entry, quantization, mmproj_path=None):
    budget = _gpu_budget_bytes()
    if budget <= 0:
        return {}
    candidates = []
    if quantization in ("auto", "fp16", ""):
        for q in ("q8", "q6", "q5", "q4"):
            path = _find_gguf(entry, q, download=False, exact=True)
            if path is not None:
                candidates.append((q, path))
    else:
        path = _find_gguf(entry, quantization, download=False, exact=True)
        if path is not None:
            candidates.append((quantization, path))
        else:
            for q in ("q8", "q6", "q5", "q4"):
                local = _find_gguf(entry, q, download=False, exact=True)
                if local is not None:
                    candidates.append((q, local))
            if candidates:
                print(f"[PromptForge] {quantization} is not local; using local {candidates[0][0]}.")
    chosen = None
    for q, path in candidates:
        fit = _auto_gguf_fit(path, mmproj_path, budget)
        if fit is not None:
            chosen = (q, fit)
            break
    if chosen is None:
        return {"quantization": "q4" if quantization in ("auto", "fp16", "") else quantization}
    q, fit = chosen
    batch, ubatch = _auto_batch(budget)
    settings = {
        "quantization": q, "batch_size": batch, "ubatch_size": ubatch,
        "swa_full": False, "flash_attn": "auto",
    }
    settings.update(fit)
    return settings


def _auto_gguf_path_settings(path, quantization, mmproj=None):
    budget = _gpu_budget_bytes()
    batch, ubatch = _auto_batch(budget)
    file_quant = re.search(r"[-_.](q[2-8])_", path.name.lower())
    settings = {
        "quantization": file_quant.group(1) if file_quant else quantization,
        "batch_size": batch, "ubatch_size": ubatch, "swa_full": False, "flash_attn": "auto",
    }
    settings.update(_auto_gguf_fit(path, mmproj, budget) or {})
    return settings


def _dir_params_b(path):
    """Parameter count in billions from the safetensors bytes of an HF model directory (bf16 weights)."""
    path = Path(str(path or ""))
    if not path.is_dir():
        return None
    total = sum(f.stat().st_size for f in path.glob("*.safetensors") if "mtp" not in f.name.lower())
    return round(total / 2 / 1e9, 1) if total else None


def _safetensors_params_b(resolved, src=""):
    return (resolved or {}).get("params_b") or _dir_params_b((resolved or {}).get("local_path") or src)


def _auto_safetensors_settings(resolved, src=""):
    budget = _gpu_budget_bytes()
    params_b = _safetensors_params_b(resolved, src)
    if not params_b or budget <= 0:
        return {}
    if params_b * 2.0 * 1024 ** 3 <= budget * 0.85:
        return {"quantization": "auto"}
    if params_b * 1.35 * 1024 ** 3 + 2 * 1024 ** 3 <= budget * 0.95:
        return {"quantization": "q8"}
    return {"quantization": "q4"}


def _gguf_vision_handler(name):
    low = (name or "").lower()
    if "gemma-4" in low or "gemma4" in low:
        return "Gemma4ChatHandler"
    if "gemma-3" in low or "gemma3" in low:
        return "Gemma3ChatHandler"
    if "qwen2.5" in low or "qwen25" in low:
        return "Qwen25VLChatHandler"
    if "qwen3.5" in low or "qwen3.6" in low or "qwen35" in low:
        return "Qwen35ChatHandler"
    return "Qwen3VLChatHandler"


def _gguf_chat_format(name):
    low = (name or "").lower()
    if "gemma" in low:
        return "gemma"
    if "llama" in low:
        return "llama-3"
    if "ministral" in low or "mistral" in low:
        return "mistral-instruct"
    if "qwen" in low or "heretic" in low or "phi" in low:
        return "chatml"
    return None


def _gguf_avail_tokens(entry):
    key = (entry or {}).get("key")
    for item in HERETIC_MODEL_ENTRIES:
        if item.get("key") == key:
            return [t.lower() for t in (item.get("avail_match") or [])]
    return []


def _scanned_ggufs():
    return [m["path"] for m in LOCAL_TEXT_MODELS if (m.get("path") or "").lower().endswith(".gguf")]


def _find_gguf(entry, quantization, download=True, exact=False):
    """Find or download a GGUF for a catalog entry, honouring the requested quant."""
    qf = (entry or {}).get("quant_files") or {}
    if not qf:
        return None
    order = _gguf_quant_order(quantization, qf) or list(qf)
    if exact:
        order = order[:1]

    base_dir = Path(entry.get("base_dir") or "LLM/GGUF")
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", "") or ""
        if models_dir:
            base_dir = Path(models_dir) / base_dir
    scanned = _scanned_ggufs()
    repo_id = (entry or {}).get("repo_id")

    def local(name):
        candidate = base_dir / name
        if candidate.is_file():
            return candidate
        for path in scanned:
            if Path(path).name == name:
                return Path(path)
        return None

    def download_q(name):
        if not repo_id:
            return None
        from huggingface_hub import hf_hub_download
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=name, local_files_only=True))
        except Exception:
            pass
        if not download:
            return None
        try:
            print("[PromptForge] downloading GGUF model (first use).")
            return Path(hf_hub_download(repo_id=repo_id, filename=name))
        except Exception:
            return None

    found = local(qf[order[0]])
    if found is not None:
        return found
    found = download_q(qf[order[0]])
    if found is not None:
        return found

    for key in order[1:]:
        found = local(qf[key])
        if found is not None:
            return found
        found = download_q(qf[key])
        if found is not None:
            return found

    if exact:
        return None

    tokens = _gguf_avail_tokens(entry)
    if tokens:
        matches = [p for p in scanned if all(t in Path(p).name.lower() for t in tokens)]
        if matches:
            return Path(matches[0])
    return None


def _find_mmproj(gguf_path, entry):
    names = []
    if (entry or {}).get("mmproj_file"):
        names.append(entry["mmproj_file"])
    dirs = [gguf_path.parent]
    for path in _scanned_ggufs():
        parent = Path(path).parent
        if parent not in dirs:
            dirs.append(parent)
    for directory in dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        for candidate in sorted(directory.glob("*mmproj*.gguf")):
            return candidate
    return None


def _local_mmproj(gguf_path, entry):
    name = (entry or {}).get("mmproj_file")
    if name:
        candidate = gguf_path.parent / name
        if candidate.is_file():
            return candidate
    matches = sorted(gguf_path.parent.glob("*mmproj*.gguf"))
    return matches[0] if matches else None


def _resolve_mmproj(gguf_path, entry, override=""):
    """Pick the mmproj for a vision GGUF: override, matching catalog file, HF download, then any local."""
    if override:
        path = Path(override)
        if not path.is_file():
            raise ValueError(f"[PromptForge] mmproj_path does not exist: {path}")
        return path
    name = (entry or {}).get("mmproj_file")
    if name:
        candidate = gguf_path.parent / name
        if candidate.is_file():
            return candidate
    repo_id = (entry or {}).get("repo_id")
    if repo_id and name:
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=repo_id, filename=name))
        except Exception as exc:
            raise ValueError(f"[PromptForge] Could not download mmproj '{name}' from {repo_id}: {exc}") from exc
    return _find_mmproj(gguf_path, entry)


def _gguf_target(option, quantization):
    """(catalog entry, gguf path) for a picker option: scanned local file (relative label or absolute path) or catalog model."""
    text = str(option or "")
    local = _local_model_option(text) or (text if text.lower().endswith(".gguf") and Path(text).is_file() else "")
    if local:
        return None, Path(local)
    resolved = _resolve_model_entry(text)
    entry = resolved.get("entry") if resolved else None
    path = _find_gguf(entry, quantization, download=False) if entry else None
    return entry, path


def _option_for_key(key):
    for entry in HERETIC_MODEL_ENTRIES:
        if entry.get("key") == key:
            return entry.get("option")
    return None


LOCAL_HEADER = "────── local models (on disk, ready) ──────"
REMOTE_HEADER = "────── not downloaded (fetched on first use) ──────"
MODEL_OPTIONS = (
    ["connected clip", LOCAL_HEADER]
    + [f"[gguf] {option}" for option in LOCAL_GGUF_OPTIONS]
    + [f"[safetensors] {option}" for option in LOCAL_SAFETENSORS_OPTIONS]
    + [REMOTE_HEADER]
    + [f"[gguf] {option}" for option in REMOTE_GGUF_OPTIONS]
    + [f"[safetensors] {option}" for option in REMOTE_SAFETENSORS_OPTIONS]
)
print(
    f"[PromptForge] {len(LOCAL_GGUF_OPTIONS)} local GGUF and {len(LOCAL_SAFETENSORS_OPTIONS)} local safetensors models ready; "
    f"{len(REMOTE_GGUF_OPTIONS) + len(REMOTE_SAFETENSORS_OPTIONS)} catalog models would download on first use."
)


def _check_not_header(option):
    if option in (LOCAL_HEADER, REMOTE_HEADER):
        raise ValueError("[PromptForge] that picker entry is a section header; choose a model below it.")


def _default_model_option():
    """A local GGUF is the fastest path, so prefer one when llama.cpp is installed; else the first safetensors entry."""
    if importlib.util.find_spec("llama_cpp") is not None:
        for option in MODEL_OPTIONS:
            if option.startswith("[gguf] [LOCAL"):
                return option
    return next((option for option in MODEL_OPTIONS if option.startswith("[safetensors] ")), MODEL_OPTIONS[0])


_DEFAULT_MODEL = _default_model_option()


def _parse_model_option(option):
    text = str(option or "")
    if text == "connected clip":
        return "connected clip", "auto", "auto"
    if text.startswith("[safetensors] "):
        return "safetensors", text[len("[safetensors] "):], "auto"
    if text.startswith("[gguf] "):
        return "GGUF", "auto", text[len("[gguf] "):]
    return "safetensors", text, "auto"


_KV_TYPE_VALUES = {"f16": 1, "q8_0": 8, "q5_1": 7, "q5_0": 6, "q4_1": 3, "q4_0": 2}


def _kv_type_value(name):
    return _KV_TYPE_VALUES.get(str(name).lower(), 1)


def _display_model(name, source):
    """Re-check local availability at run time so the report never shows a stale widget label."""
    bare = _bare_option(name)
    if _local_model_option(bare) or Path(bare).exists():
        return f"[LOCAL] {bare}"
    resolved = _resolve_model_entry(bare)
    if source == "GGUF":
        entry = resolved.get("entry") if resolved else None
        quants = _gguf_local_quants(entry) if entry else []
        if quants:
            ordered = [q for q in ("q4", "q5", "q6", "q8") if q in quants]
            return f"[LOCAL {'/'.join(q.upper() for q in ordered)}] {bare}"
        return f"[DOWNLOAD] {bare}"
    local = (resolved or {}).get("local_path") or ""
    repo = (resolved or {}).get("repo_id") or ""
    if local or (repo and _cache_has(repo, "config.json")):
        return f"[LOCAL] {bare}"
    return f"[DOWNLOAD] {bare}"


def _settings_report(settings):
    source = settings.get("source", "safetensors")
    if source == "GGUF":
        return (
            f"GGUF {_display_model(settings.get('gguf_model'), 'GGUF')} | quant {settings.get('quantization')} | "
            f"ctx {settings.get('context_size')} | gpu_layers {settings.get('gpu_layers')} | "
            f"kv {settings.get('kv_cache_type')} | batch {settings.get('batch_size')}/{settings.get('ubatch_size')}"
        )
    if source == "safetensors":
        return f"safetensors {_display_model(settings.get('safetensors_model'), 'safetensors')} | quant {settings.get('quantization')}"
    return "connected clip | no model loaded"


def _recommend_settings(data):
    _check_not_header(data.get("model", ""))
    source, safetensors_model, gguf_model = _parse_model_option(data.get("model", ""))
    quantization = data.get("quantization", "auto")
    settings = {
        "source": source, "safetensors_model": safetensors_model, "gguf_model": gguf_model,
        "quantization": quantization,
    }
    if source == "GGUF":
        entry, path = _gguf_target(gguf_model, quantization)
        mmproj = _local_mmproj(path, entry) if path is not None else None
        if entry is None and path is not None:
            settings.update(_auto_gguf_path_settings(path, quantization, mmproj))
        else:
            settings.update(_auto_gguf_settings(entry, quantization, mmproj))
    elif source == "safetensors":
        settings.update(_auto_safetensors_settings(_resolve_model_entry(safetensors_model), _local_model_option(safetensors_model)))
    for key, value in (
        ("context_size", 16384), ("gpu_layers", -1), ("batch_size", 2048), ("ubatch_size", 512),
        ("kv_cache_type", "f16"), ("offload_kqv", True), ("swa_full", False), ("flash_attn", "auto"),
    ):
        settings.setdefault(key, value)
    return settings


def _test_settings(data):
    _check_not_header(data.get("model", ""))
    source, safetensors_model, gguf_model = _parse_model_option(data.get("model", ""))
    quantization = data.get("quantization", "auto")
    if source == "connected clip":
        return {"ok": True, "message": "connected clip is verified when the Model node runs.", "settings": {}}
    if source == "GGUF":
        return _test_gguf(gguf_model, quantization)
    return _test_safetensors(safetensors_model, quantization)


def _test_gguf(option, quantization):
    entry, path = _gguf_target(option, quantization)
    if path is None:
        return {"ok": False, "message": "model is not downloaded yet; run it once, then test."}
    mmproj = _local_mmproj(path, entry)
    plan = _auto_gguf_path_settings(path, quantization, mmproj) if entry is None else _auto_gguf_settings(entry, quantization, mmproj)
    ctx = int(plan.get("context_size", 16384))
    kv = plan.get("kv_cache_type", "f16")
    layers = int(plan.get("gpu_layers", -1))
    batch = int(plan.get("batch_size", 2048))
    ubatch = int(plan.get("ubatch_size", 512))
    candidates = [(ctx, kv)]
    for fallback_ctx in (12288, 8192, 6144, 4096):
        if fallback_ctx < ctx:
            candidates.append((fallback_ctx, kv))
    for fallback_kv in ("q8_0", "q4_0"):
        if fallback_kv != kv:
            candidates.append((ctx, fallback_kv))
    handler = None
    if mmproj is not None:
        try:
            from llama_cpp import llama_chat_format
            handler_name = _gguf_vision_handler(f"{path.name} {(entry or {}).get('name', '')}")
            handler_cls = getattr(llama_chat_format, handler_name, None)
            if handler_cls is not None:
                handler = handler_cls(clip_model_path=str(mmproj), verbose=False)
        except Exception:
            handler = None
    from llama_cpp import Llama
    _free_vram()
    last_error = ""
    for test_ctx, test_kv in candidates[:4]:
        kwargs = {
            "model_path": str(path),
            "n_ctx": test_ctx,
            "n_batch": min(batch, test_ctx),
            "n_ubatch": min(ubatch, min(batch, test_ctx)),
            "n_gpu_layers": layers,
            "offload_kqv": True,
            "swa_full": False,
            "type_k": _kv_type_value(test_kv),
            "type_v": _kv_type_value(test_kv),
            "use_mmap": True,
            "verbose": False,
        }
        if handler is not None:
            kwargs["chat_handler"] = handler
        try:
            llm = Llama(**kwargs)
            llm.close()
            return {
                "ok": True,
                "message": f"loaded {path.name} at {test_ctx} tokens, kv {test_kv}, gpu_layers {layers}.",
                "settings": {
                    "quantization": plan.get("quantization", quantization),
                    "context_size": test_ctx, "kv_cache_type": test_kv, "gpu_layers": layers,
                    "batch_size": batch, "ubatch_size": ubatch, "offload_kqv": True,
                    "swa_full": False, "flash_attn": "auto",
                },
            }
        except Exception as exc:
            last_error = str(exc)
            gc.collect()
    return {"ok": False, "message": last_error or "context creation failed"}


def _load_hf_model(src, load_kwargs):
    """Load with the vision class when the config has one (text-only prompts work there too), else as a causal LM."""
    try:
        return AutoVision.from_pretrained(src, **load_kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
    return AutoModelForCausalLM.from_pretrained(src, **load_kwargs)


def _test_safetensors(option, quantization):
    resolved = _resolve_model_entry(option)
    src = (resolved or {}).get("local_path") or (resolved or {}).get("repo_id") or _local_model_option(option)
    path = Path(src) if src else None
    if path is None or not path.exists():
        return {"ok": False, "message": "model is not downloaded yet; run it once, then test."}
    if not path.is_dir():
        return {"ok": False, "message": "single-file encoders are verified on first generation."}
    plan = _auto_safetensors_settings(resolved, src)
    quant = plan.get("quantization") or ("q4" if quantization in ("auto", "fp16", "") else quantization)
    dtype = _hf_compute_dtype()
    quant_cfg = None
    if quant in ("q8", "q4"):
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=(quant == "q8"),
            load_in_4bit=(quant == "q4"),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    kwargs = {"quantization_config": quant_cfg, "device_map": "auto", **_hf_dtype_kwargs(dtype)}
    max_memory = _hf_max_memory()
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    _free_vram()
    try:
        model = _load_hf_model(str(path), kwargs)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {"ok": True, "message": f"loaded {path.name} in {quant}.", "settings": {"quantization": quant}}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def _scan_configs(data):
    scope = data.get("scope", "all local")
    configs = {}
    lines = []
    if scope == "all local":
        for entry in GGUF_MODELS:
            path = _find_gguf(entry, "auto", download=False)
            if path is None:
                continue
            key = entry.get("key") or path.stem
            mmproj = _local_mmproj(path, entry)
            cfg = _auto_gguf_settings(entry, "auto", mmproj)
            cfg.update({"source": "GGUF", "gguf_model": _option_for_key(key) or key, "safetensors_model": "auto"})
            configs[key] = cfg
            lines.append(f"{key}: {_settings_report(cfg)}")
        for option in LOCAL_SAFETENSORS_OPTIONS:
            local = _local_model_option(option)
            if local and not Path(local).is_dir():
                continue
            cfg = _auto_safetensors_settings(_resolve_model_entry(option), local)
            cfg.update({"source": "safetensors", "safetensors_model": option, "gguf_model": "auto"})
            key = _bare_option(option)
            configs[key] = cfg
            lines.append(f"{key}: {_settings_report(cfg)}")
        lines.append(f"not downloaded: {len(REMOTE_GGUF_OPTIONS)} GGUF, {len(REMOTE_SAFETENSORS_OPTIONS)} safetensors catalog models (they fetch on first use).")
    else:
        cfg = _recommend_settings(data)
        configs[cfg["source"]] = cfg
        lines.append(_settings_report(cfg))
    if not lines:
        lines.append("No local models found. Download a model first or use 'selected'.")
    return {"report": "\n".join(lines), "configs": configs}


class PROMPTFORGE_Model:
    CATEGORY = "🔮 PromptForge"
    RETURN_TYPES = ("PROMPTFORGE_MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "REPORT")
    FUNCTION = "build"
    DESCRIPTION = "PromptForge model and runtime settings. One picker for clip / safetensors / GGUF; everything else is configured for the current GPU automatically."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {"default": _DEFAULT_MODEL, "tooltip": "What to run. The list is split into local models (on disk or in the HF cache, ready to use; [LOCAL Q4/Q8] names the GGUF quants present) and [DOWNLOAD] catalog models that fetch on first use."}),
                "quantization": (QUANT_OPTIONS, {"default": "auto", "tooltip": "Quantization. auto picks the best local option for this GPU."}),
                "context_size": ("INT", {"default": 0, "min": 0, "max": 262144, "step": 512, "tooltip": "GGUF context length. 0 = auto (largest that fits, 16k first)."}),
                "local_model_path": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional local safetensors dir/file or GGUF file; overrides the picker."}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "Optional mmproj (.gguf) for GGUF image/video mode."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Connected ComfyUI text encoder. Required when model is 'connected clip'."}),
            },
        }

    def build(self, model, quantization, context_size, local_model_path, mmproj_path, clip=None, **kwargs):
        _check_not_header(model)
        source, safetensors_model, gguf_model = _parse_model_option(model)
        local = (local_model_path or "").strip()
        if local:
            if not Path(local).exists():
                raise ValueError(f"[PromptForge] local_model_path does not exist on this machine: {local}")
            source = "GGUF" if local.lower().endswith(".gguf") else "safetensors"
            if source == "GGUF":
                gguf_model = local
            else:
                safetensors_model = local
        settings = {
            "source": source,
            "safetensors_model": safetensors_model,
            "gguf_model": gguf_model,
            "quantization": quantization,
            "local_model_path": local_model_path,
            "mmproj_path": mmproj_path,
            "clip": clip,
        }
        if source == "GGUF":
            entry, path = _gguf_target(gguf_model, quantization)
            mmproj = _local_mmproj(path, entry) if path is not None else None
            if entry is None and path is not None:
                settings.update(_auto_gguf_path_settings(path, quantization, mmproj))
            else:
                settings.update(_auto_gguf_settings(entry, quantization, mmproj))
        elif source == "safetensors":
            settings.update(_auto_safetensors_settings(_resolve_model_entry(safetensors_model), _local_model_option(safetensors_model) or local))
        for key, value in (
            ("context_size", 16384), ("gpu_layers", -1), ("batch_size", 2048), ("ubatch_size", 512),
            ("kv_cache_type", "f16"), ("offload_kqv", True), ("swa_full", False), ("flash_attn", "auto"),
        ):
            settings.setdefault(key, value)
        if int(context_size) > 0:
            settings["context_size"] = int(context_size)
        if source == "GGUF":
            entry, _ = _gguf_target(gguf_model, settings.get("quantization", quantization))
            chosen = settings.get("quantization", quantization)
            if entry is not None and _find_gguf(entry, chosen, download=False, exact=True) is None:
                print(f"[PromptForge] {gguf_model} ({chosen}) is not local; it will download on first use.")
        elif source == "safetensors":
            resolved = _resolve_model_entry(safetensors_model)
            repo = (resolved or {}).get("repo_id") or ""
            local = (resolved or {}).get("local_path") or ""
            if repo and not local and not _cache_has(repo, "config.json"):
                print(f"[PromptForge] {safetensors_model} is not local; it will download on first use.")
        if source == "connected clip":
            if clip is None:
                raise ValueError("[PromptForge] model is 'connected clip' but no CLIP input is connected.")
            try:
                clip.tokenize("test")
            except Exception as exc:
                raise ValueError(f"[PromptForge] connected CLIP failed to tokenize: {exc}") from exc
        report = _settings_report(settings)
        print(f"[PromptForge] model: {report}")
        return (settings, report)


class PROMPTFORGE_ConfigFinder:
    CATEGORY = "🔮 PromptForge"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("REPORT",)
    FUNCTION = "scan"
    OUTPUT_NODE = True
    DESCRIPTION = "Scan local models and compute the best PromptForge settings for this GPU. Runs on its own; no generation."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (["all local", "selected"], {"default": "all local", "tooltip": "Scan every locally available model, or only the selected one."}),
                "model": (MODEL_OPTIONS, {"default": _DEFAULT_MODEL, "tooltip": "Model to plan for when scope is 'selected'."}),
                "quantization": (QUANT_OPTIONS, {"default": "auto", "tooltip": "Quantization to plan for; auto picks the best local quant."}),
            },
        }

    def scan(self, scope, model, quantization, **kwargs):
        result = _scan_configs({"scope": scope, "model": model, "quantization": quantization})
        for line in result["report"].splitlines():
            print(f"[PromptForge] {line}")
        return {"ui": {"text": [result["report"]]}, "result": (result["report"],)}


try:
    import asyncio
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/promptforge/test")
    async def _promptforge_test(request):
        data = await request.json()
        return web.json_response(await asyncio.to_thread(_test_settings, data))

    @PromptServer.instance.routes.post("/promptforge/scan")
    async def _promptforge_scan(request):
        data = await request.json()
        return web.json_response(await asyncio.to_thread(_scan_configs, data))
except Exception:
    pass


class PROMPTFORGE_PromptEnhancer:
    CATEGORY = "🔮 PromptForge"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ENHANCED_PROMPT", "THINKING")
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PROMPTFORGE_MODEL", {"tooltip": "MODEL output from a PromptForge Model node (source, model, quant and runtime settings)."}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Raw user request / media-derived text to enhance. Can be loose natural language, random idea, detailed description, keyword pile, anything."}),
                "mode": (MODE_OPTIONS, {"default": "text_only", "tooltip": "text_only ignores all media; inputs uses every connected image (image / image2 / image3) and the sampled video frames."}),
                "enhancement_preset": (PRESET_OPTIONS, {"default": "auto", "tooltip": "Preset system prompt: auto, or one of the PromptForge presets (Krea 2 T2I, MOSS Image T2I, Qwen Image T2I, generic styles...)."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2 ** 32 - 1, "tooltip": "Sampling seed."}),
                "max_tokens": ("INT", {"default": 2048, "min": 32, "max": 65536, "tooltip": "Max new tokens for the enhanced prompt. Keep it below the model context for GGUF."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Sampling temperature."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling cutoff."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Top-K. 0 = model default."}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Beam search width. 1 = fast sampling; >1 trades speed for stability."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "Repetition penalty."}),
                "system_prompt_override": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom system prompt override. Leave empty to use the selected preset."}),
            },
            "optional": {
                "video_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "tooltip": "Target video duration in seconds."}),
                "image": ("IMAGE", {"tooltip": "First reference image."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image / edit source."}),
                "image3": ("IMAGE", {"tooltip": "Third reference image / edit source."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": "Frames sampled from video input."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "What the generated prompt/scene must not contain."}),
                "pipe": ("PROMPTFORGE_PIPE", {"tooltip": "From a PromptForge Loader. With enhancement_preset on auto, the preset for that model family is used."}),
                "dialogue": ("STRING", {"default": "", "multiline": True, "tooltip": "Video presets only (MiniMax H3). What the dialogue should be about and how it should sound, in your own words. The enhancer writes the lines, speaker IDs and tags, sized to video_seconds. Empty (with language on auto) = no dialogue instructions."}),
                "dialogue_language": ([DIALOGUE_AUTO, *DIALOGUE_LANGUAGES, DIALOGUE_OTHER], {"default": DIALOGUE_AUTO, "tooltip": "Language of the spoken lines. The list is what MiniMax H3 speaks reliably."}),
                "dialogue_language_other": ("STRING", {"default": "", "tooltip": "Used when dialogue_language is 'other', for example Polish. Languages outside the list are only partially supported by the video model."}),
                "auto_retry": ("BOOLEAN", {"default": True, "tooltip": "Regenerate with a new seed (up to 3 tries) when the output looks empty, repetitive, leaked reasoning or a refusal. Turn off to always keep the first output."}),
                "continuation": ("STRING", {"default": "", "multiline": True, "tooltip": "Video presets only (MiniMax H3). A continuation brief added to the system prompt; the Endless Video node fills this in."}),
                "context_video": ("IMAGE", {"tooltip": "Video presets only (MiniMax H3). Frames the generator will start from. The model looks at them to continue the action, but they are never cited as <Video 1>."}),
            },
        }

    # ------------------------------------------------------------------ #
    # Node entry point
    # ------------------------------------------------------------------ #
    def enhance(
        self,
        model,
        prompt_text,
        mode,
        enhancement_preset,
        seed,
        max_tokens,
        temperature,
        top_p,
        top_k,
        num_beams,
        repetition_penalty,
        system_prompt_override,
        video_seconds=5,
        image=None,
        image2=None,
        image3=None,
        video=None,
        frame_count=16,
        negative_prompt="",
        auto_retry=True,
        pipe=None,
        dialogue="",
        dialogue_language=DIALOGUE_AUTO,
        dialogue_language_other="",
        continuation="",
        context_video=None,
        **kwargs,
    ):
        if enhancement_preset == "auto" and pipe:
            enhancement_preset = _family_preset(pipe.get("family"), any(value is not None for value in (image, image2, image3)), video is not None)
        print("[PromptForge] request received.")
        cfg = model if isinstance(model, dict) else {}
        source = cfg.get("source", "safetensors")
        safetensors_model = cfg.get("safetensors_model", "auto")
        gguf_model = cfg.get("gguf_model", "auto")
        quantization = cfg.get("quantization", "auto")
        context_size = cfg.get("context_size", 16384)
        gpu_layers = cfg.get("gpu_layers", -1)
        batch_size = cfg.get("batch_size", 2048)
        ubatch_size = cfg.get("ubatch_size", 512)
        kv_cache_type = cfg.get("kv_cache_type", "f16")
        offload_kqv = cfg.get("offload_kqv", True)
        swa_full = cfg.get("swa_full", False)
        flash_attn = cfg.get("flash_attn", "auto")
        local_model_path = cfg.get("local_model_path", "")
        mmproj_path = cfg.get("mmproj_path", "")
        clip = cfg.get("clip")
        override_text = (system_prompt_override or "").strip()
        sys_text = override_text
        if not sys_text:
            key = enhancement_preset if enhancement_preset and enhancement_preset in PROMPTFORGE_PRESETS else DEFAULT_PRESET
            if enhancement_preset and enhancement_preset not in PRESET_OPTIONS:
                print(f"[PromptForge] preset '{enhancement_preset}' not found; using {DEFAULT_PRESET}.")
            entry = PROMPTFORGE_PRESETS.get(key) or {}
            sys_text = entry.get("text") or entry.get("system_prompt") or (PROMPTFORGE_PRESETS.get(DEFAULT_PRESET) or {}).get("text") or ""

        user_text = (prompt_text or "").strip()
        duration_text = f"{float(video_seconds):.2f}"
        duration_line = ""
        if sys_text and "{video_seconds}" in sys_text:
            sys_text = sys_text.replace("{video_seconds}", duration_text)
        elif "minimax" in (enhancement_preset or "").lower():
            duration_line = f"Target video duration: {duration_text} seconds."
        guard = OUTPUT_GUARD
        guard += " Do not output chain-of-thought or reasoning; start with the final prompt."
        minimax_key = None if override_text else _minimax_mode(enhancement_preset)
        minimax_block = ""
        if minimax_key:
            image, image2, image3, video, minimax_block = self._minimax_inputs(minimax_key, mode, image, image2, image3, video, frame_count, duration_text)
        media_line = self._media_inventory(mode, image, image2, image3, video, frame_count)
        labelled_video = video is not None
        if minimax_key and mode != "text_only" and self._has_media(context_video):
            video = context_video
        dialogue_block = _dialogue_block(dialogue, dialogue_language, dialogue_language_other, float(video_seconds)) if minimax_key else ""
        negative_text = (negative_prompt or "").strip()
        negative_line = ""
        if negative_text:
            negative_line = (
                "# Negative constraints\n"
                "The generated prompt and the scene it describes must NOT contain any of the following: "
                + negative_text +
                ". Write the scene so they are absent, converting each exclusion into a positive description "
                "of the desired state; never name them inside the prompt itself."
            )
        continuation_block = (continuation or "").strip() if minimax_key else ""
        system_parts = [p for p in (sys_text, media_line, minimax_block, continuation_block, dialogue_block, negative_line, guard) if p and p.strip()]
        user_parts = [p for p in (user_text, duration_line) if p and p.strip()]
        system_prompt = "\n\n".join(system_parts)
        user_prompt = "\n\n".join(user_parts)
        full_prompt = "\n\n".join(p for p in (system_prompt, user_prompt) if p)

        eff_local = (local_model_path or "").strip()
        if source == "connected clip":
            if clip is None:
                raise ValueError("[PromptForge] Source is 'connected clip' but no CLIP input is connected.")
            engine = "text_encoder"
            model_name = "auto"
            clip_for_generate = clip
        elif source == "GGUF":
            engine = "gguf"
            model_name = gguf_model
            clip_for_generate = None
        else:
            engine = "text_encoder"
            model_name = safetensors_model
            clip_for_generate = None

        if eff_local:
            is_gguf = eff_local.lower().endswith(".gguf")
            if source == "GGUF" and not is_gguf:
                raise ValueError("[PromptForge] Source is GGUF but local_model_path is not a .gguf file.")
            if source != "GGUF" and is_gguf:
                raise ValueError("[PromptForge] Source is safetensors but local_model_path points to a .gguf file; select the GGUF source.")
        elif source == "safetensors" and model_name == "none":
            raise ValueError("[PromptForge] No safetensors model available; add one to the model dirs or set local_model_path.")
        elif source == "GGUF" and model_name == "none":
            raise ValueError("[PromptForge] No GGUF model available; add a .gguf to the model dirs or set local_model_path.")

        short_form = any(word in (enhancement_preset or "").lower() for word in ("tags", "refine", "technical"))
        min_chars = 40 if short_form else (400 if max_tokens >= 192 else 120)
        def run(attempt_seed, attempt_system):
            return self._generate(
                full_prompt=full_prompt,
                system_prompt=attempt_system,
                user_prompt=user_prompt,
                mode=mode,
                engine=engine,
                model_name=model_name,
                quantization=quantization,
                context_size=context_size,
                gpu_layers=gpu_layers,
                batch_size=batch_size,
                ubatch_size=ubatch_size,
                kv_cache_type=kv_cache_type,
                offload_kqv=offload_kqv,
                swa_full=swa_full,
                flash_attn=flash_attn,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                image=image,
                image2=image2,
                image3=image3,
                video=video,
                frame_count=frame_count,
                seed=attempt_seed,
                clip=clip_for_generate,
                local_model_path=local_model_path,
                mmproj_path=mmproj_path,
                disable_thinking=True,
                auto_retry=auto_retry,
                min_chars=min_chars,
            )
        enhanced = run(seed, system_prompt)
        if dialogue_block and auto_retry:
            language = _dialogue_language(dialogue_language, dialogue_language_other)
            for attempt in range(1, 3):
                problem = _dialogue_problem(_split_thinking(enhanced)[0], language)
                if not problem:
                    break
                print(f"[PromptForge] dialogue rejected ({problem}); retrying ({attempt}/2).")
                seed = _next_seed(seed)
                enhanced = run(seed, system_prompt + f"\n\n# Correction\nYour previous answer broke the dialogue format: {problem}. Write the whole prompt again and follow the Dialogue section exactly.")
        answer, thinking = _split_thinking(enhanced)
        answer = _clean_prompt_output(answer)
        if minimax_key:
            answer = _drop_absent_references(answer, sum(value is not None for value in (image, image2, image3)), labelled_video, user_text)
        preset_lower = (enhancement_preset or "").lower()
        if not any(marker in preset_lower for marker in ("minimax", "wan", "ltx", "ace-step")):
            answer = _strip_video_meta(answer)
        if not answer.strip():
            detail = "only reasoning/thinking" if thinking.strip() else "no usable text"
            raise ValueError(
                f"[PromptForge] the model returned {detail} and no final prompt. "
                "Use a non-thinking model (Qwen3-VL for vision), raise max_tokens, or change the preset."
            )
        return (answer, thinking)

    # ------------------------------------------------------------------ #
    # Model lifecycle: unload after the prompt has been generated
    # ------------------------------------------------------------------ #
    def _release(self, llm=None):
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Generation dispatch
    # ------------------------------------------------------------------ #
    def _generate(self, full_prompt, system_prompt, user_prompt, mode, engine, model_name, quantization, context_size, gpu_layers, batch_size, ubatch_size, kv_cache_type, offload_kqv, swa_full, flash_attn, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, local_model_path, mmproj_path, disable_thinking, auto_retry, min_chars=20):
        try:
            if engine == "gguf":
                return self._generate_gguf(
                    full_prompt=full_prompt, system_prompt=system_prompt, user_prompt=user_prompt,
                    mode=mode, model_name=model_name, quantization=quantization,
                    context_size=context_size, gpu_layers=gpu_layers, batch_size=batch_size, ubatch_size=ubatch_size,
                    kv_cache_type=kv_cache_type, offload_kqv=offload_kqv, swa_full=swa_full,
                    flash_attn=flash_attn,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                    num_beams=num_beams, repetition_penalty=repetition_penalty, seed=seed,
                    image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
                    local_model_path=local_model_path, mmproj_path=mmproj_path,
                    disable_thinking=disable_thinking, auto_retry=auto_retry, min_chars=min_chars,
                )
            return self._generate_text_encoder(
                full_prompt=full_prompt, system_prompt=system_prompt, user_prompt=user_prompt,
                mode=mode, model_name=model_name, quantization=quantization,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                num_beams=num_beams, repetition_penalty=repetition_penalty,
                image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
                seed=seed, clip=clip, local_model_path=local_model_path,
                disable_thinking=disable_thinking, auto_retry=auto_retry, min_chars=min_chars,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # GGUF / llama.cpp engine
    # ------------------------------------------------------------------ #
    def _generate_gguf(self, full_prompt, system_prompt, user_prompt, mode, model_name, quantization, context_size, gpu_layers, batch_size, ubatch_size, kv_cache_type, offload_kqv, swa_full, flash_attn, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, seed, image, image2, image3, video, frame_count, local_model_path, mmproj_path, disable_thinking, auto_retry, min_chars=20):
        entry = None
        gguf_path = None
        eff_local = (local_model_path or "").strip() or _local_model_option(model_name)
        if not eff_local and str(model_name).lower().endswith(".gguf") and Path(str(model_name)).is_file():
            eff_local = str(model_name)
        if eff_local and str(eff_local).lower().endswith(".gguf"):
            gguf_path = Path(str(eff_local))
        if gguf_path is None:
            resolved = _resolve_model_entry(model_name) if (model_name and model_name != "auto") else _resolve_model_entry("auto")
            if resolved is None:
                raise ValueError(f"[PromptForge] GGUF model '{model_name}' is not in the catalog and was not found in the model folders on this machine.")
            entry = resolved.get("entry") if resolved.get("kind") == "gguf" else None
            if entry is None:
                raise ValueError(f"[PromptForge] '{model_name}' is a safetensors catalog entry; pick a [gguf] option for the GGUF engine.")
            if entry is not None:
                exact = _find_gguf(entry, quantization, download=False, exact=True)
                if exact is not None:
                    gguf_path = exact
                else:
                    local_any = _find_gguf(entry, "auto", download=False)
                    if local_any is not None:
                        print(f"[PromptForge] {quantization} is not local; using {local_any.name}.")
                        gguf_path = local_any
                    else:
                        gguf_path = _find_gguf(entry, quantization)
        if gguf_path is None:
            raise ValueError("[PromptForge] No GGUF model selected / available. Put the .gguf in a scanned models dir (text_encoders / LLM / gguf) or set local_model_path.")

        images = self._mode_images(mode, image, image2, image3, video, frame_count) if mode != "text_only" else []
        mmproj = _resolve_mmproj(gguf_path, entry, mmproj_path) if images else None
        if images and mmproj is None:
            raise ValueError("[PromptForge] Vision GGUF needs an mmproj file next to the model (e.g. mmproj-F16.gguf). Put one in the same folder, set mmproj_path, or switch mode to text_only.")

        from llama_cpp import Llama

        n_ctx = max(512, int(context_size))
        estimated = len(full_prompt) // 3 + len(images) * 1024 + int(max_tokens) + 512
        if estimated > n_ctx:
            print(f"[PromptForge] context_size={n_ctx} may truncate this request (about {estimated} tokens needed).")

        kv_type = _kv_type_value(kv_cache_type)
        flash_types = {"auto": -1, "on": 1, "off": 0}
        flash_type = flash_types.get(flash_attn, flash_types["auto"])
        n_batch = min(max(64, int(batch_size)), n_ctx)
        n_ubatch = max(32, min(int(ubatch_size), n_batch))

        llama_kwargs = {
            "model_path": str(gguf_path),
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_gpu_layers": int(gpu_layers),
            "offload_kqv": bool(offload_kqv),
            "swa_full": bool(swa_full),
            "type_k": kv_type,
            "type_v": kv_type,
            "use_mmap": True,
            "verbose": False,
        }
        supports_flash = "flash_attn_type" in inspect.signature(Llama.__init__).parameters
        if supports_flash:
            llama_kwargs["flash_attn_type"] = flash_type
        if images:
            from llama_cpp import llama_chat_format
            model_name_lower = f"{gguf_path.name} {(entry or {}).get('name', '')}".lower()
            handler_name = _gguf_vision_handler(model_name_lower)
            handler_cls = getattr(llama_chat_format, handler_name, None)
            if handler_cls is None:
                raise ValueError(f"[PromptForge] this llama-cpp-python build has no {handler_name}; upgrade llama-cpp-python for this vision architecture.")
            llama_kwargs["chat_handler"] = handler_cls(clip_model_path=str(mmproj), verbose=False)
        else:
            metadata = _gguf_metadata(gguf_path)
            if "tokenizer.chat_template" not in metadata:
                chat_format = _gguf_chat_format(f"{gguf_path.name} {(entry or {}).get('name', '')}")
                if chat_format:
                    llama_kwargs["chat_format"] = chat_format
        if llama_kwargs["n_gpu_layers"] != 0:
            try:
                free_bytes = torch.cuda.mem_get_info()[0]
            except Exception:
                free_bytes = 0
            required = _gguf_memory_estimate(gguf_path, n_ctx, llama_kwargs["n_gpu_layers"], mmproj)
            if free_bytes < required:
                _free_vram(needed=int(required))

        fallbacks = []
        if kv_type != _kv_type_value("f16"):
            fallbacks.append(("kv_cache_type", {"type_k": _kv_type_value("f16"), "type_v": _kv_type_value("f16")}))
        if supports_flash and flash_type == flash_types["on"]:
            fallbacks.append(("flash_attn", {"flash_attn_type": flash_types["auto"]}))
        if bool(swa_full):
            fallbacks.append(("swa_full", {"swa_full": False}))

        llm = None
        last_error = None
        retried_layers = False
        index = 0
        while index <= len(fallbacks):
            try:
                llm = Llama(**llama_kwargs)
                break
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                if "Failed to load model" in message and llama_kwargs["n_gpu_layers"] != 0 and not retried_layers:
                    retried_layers = True
                    llama_kwargs["n_gpu_layers"] = _gguf_layers_that_fit(gguf_path, n_ctx, mmproj)
                    print(f"[PromptForge] not enough free VRAM for the whole GGUF; retrying with {llama_kwargs['n_gpu_layers']} GPU layers.")
                    gc.collect()
                    continue
                if "Failed to create llama context" not in message:
                    raise
                last_error = message
                if index >= len(fallbacks):
                    break
                label, overrides = fallbacks[index]
                index += 1
                print(f"[PromptForge] llama.cpp rejected {label}; retrying with the safer setting.")
                llama_kwargs.update(overrides)
                gc.collect()
        if llm is None:
            raise ValueError(
                f"[PromptForge] llama.cpp could not create a {n_ctx}-token context. "
                "Lower context_size, gpu_layers or batch_size, set kv_cache_type to q8_0/q4_0, or turn offload_kqv off. "
                + (last_error or "")
            )
        print(f"[PromptForge] GGUF ready: ctx {n_ctx}, gpu_layers {llama_kwargs['n_gpu_layers']}, kv {kv_cache_type}.")

        model_label_lower = f"{gguf_path.name} {(entry or {}).get('name', '')}".lower()
        gemma4 = "gemma-4" in model_label_lower or "gemma4" in model_label_lower
        chat_template = llm.metadata.get("tokenizer.chat_template") if not images else None
        user_text_value = user_prompt or full_prompt
        if images and gemma4:
            # the vision chat handler renders its own template: fold the system turn into the user turn and ask for no thought channel
            if (system_prompt or "").strip():
                user_text_value = f"{system_prompt.strip()}\n\n{user_text_value}".strip()
            user_text_value += "\n\nIMPORTANT: Return the final prompt directly. Do not use a thought channel. Do not write reasoning."
        if images:
            content = [{"type": "text", "text": user_text_value}]
            content.extend({"type": "image_url", "image_url": {"url": self._pil_data_url(p)}} for p in images)
        else:
            content = user_text_value
        messages = []
        if (system_prompt or "").strip() and not (images and gemma4):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        completion_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            "seed": seed,
        }
        if top_k:
            completion_kwargs["top_k"] = top_k
        # older llama-cpp-python builds lack penalty_last_n / reasoning_budget
        supported = inspect.signature(llm.create_completion if chat_template else llm.create_chat_completion).parameters
        for key, value in (("frequency_penalty", 0.1), ("penalty_last_n", 256)):
            if key in supported:
                completion_kwargs[key] = value
        if chat_template:
            # The model's own template with thinking disabled (empty think block / primed thought channel) is the
            # reliable non-thinking mode; llama.cpp adds BOS itself, so the template gets an empty bos_token.
            prompt_text = render_jinja_template(
                [messages], chat_template=chat_template, add_generation_prompt=True, bos_token="", eos_token="",
                enable_thinking=not disable_thinking,
            )[0][0]
            completion_kwargs["prompt"] = prompt_text
            completion_kwargs["stop"] = ["<|im_end|>", "<turn|>", "<end_of_turn>", "<|eot_id|>", "<|end|>"]
        else:
            completion_kwargs["messages"] = messages
            if disable_thinking and "reasoning_budget" in supported:
                completion_kwargs["reasoning_budget"] = 0

        def attempt(seed_value):
            kwargs = dict(completion_kwargs)
            kwargs["seed"] = seed_value
            started = time.time()
            if chat_template:
                out = llm.create_completion(**kwargs)
                text = out.get("choices", [{}])[0].get("text", "").strip()
            else:
                out = llm.create_chat_completion(**kwargs)
                text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            tokens = (out.get("usage") or {}).get("completion_tokens")
            if tokens:
                print(f"[PromptForge] GGUF generated {tokens} tokens in {time.time() - started:.1f}s.")
            return text

        try:
            enhanced = _generate_checked(attempt, seed, min_chars, auto_retry)
        except ValueError as exc:
            if images and "mmproj" in str(exc).lower():
                expected = (entry or {}).get("mmproj_file") or "mmproj-*.gguf"
                raise ValueError(f"[PromptForge] Vision GGUF could not load mmproj '{mmproj}'. It must match the text model {gguf_path.name}. Download {expected} next to the model or set mmproj_path.") from exc
            raise
        finally:
            self._release(llm=llm)
        return enhanced

    # ------------------------------------------------------------------ #
    # Text encoder engines (native ComfyUI CLIP route / bitsandbytes route)
    # ------------------------------------------------------------------ #
    def _generate_text_encoder(self, full_prompt, system_prompt, user_prompt, mode, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, local_model_path, disable_thinking, auto_retry, min_chars=20):
        if clip is not None:
            enhanced = self._generate_with_clip(
                system_prompt=system_prompt, user_prompt=user_prompt or full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
                clip=clip, disable_thinking=disable_thinking, auto_retry=auto_retry, min_chars=min_chars,
            )
            return enhanced

        eff_local = (local_model_path or "").strip() or _local_model_option(model_name)
        if not eff_local and Path(str(model_name)).exists():
            eff_local = str(model_name)
        src = eff_local or ""
        resolved = _resolve_model_entry(model_name) if (model_name and model_name != "auto") else _resolve_model_entry("auto")
        if not src and resolved is not None and resolved.get("kind") == "hf":
            src = resolved.get("local_path") or resolved.get("repo_id") or ""
        if not src:
            raise ValueError(f"[PromptForge] safetensors model '{model_name}' is not in the catalog and was not found in the model folders on this machine.")
        params_b = _safetensors_params_b(resolved, src)

        quant = {"8bit": "q8", "4bit": "q4"}.get(quantization or "auto", quantization or "auto")
        if quant in ("q5", "q6"):
            print(f"[PromptForge] {quant} is a GGUF quant level; loading safetensors in q8 instead.")
            quant = "q8"
        if quant == "q8" and params_b and torch.cuda.is_available():
            try:
                vram = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                vram = 0
            if vram and float(params_b) * 1.15 * 1024 ** 3 > vram * 0.85:
                print("[PromptForge] q8 does not fit this GPU; using q4 instead.")
                quant = "q4"
        src_path = Path(src)
        if not src_path.exists():
            from huggingface_hub import snapshot_download
            src = snapshot_download(repo_id=src, local_dir=None)
            src_path = Path(src)

        if quant in ("q8", "q4") and not src_path.is_dir():
            print(f"[PromptForge] {quant} quantization needs an HF model directory; loading this single file in fp16 instead.")
            quant = "auto"

        if src_path.is_dir():
            if torch.cuda.is_available():
                _free_vram()
            hf_dtype = torch.float16 if quant == "fp16" else _hf_compute_dtype()
            if quant in ("auto", "q8") and params_b:
                try:
                    free_bytes = torch.cuda.mem_get_info()[0]
                except Exception:
                    free_bytes = 0
                per_param = 2.0 if quant == "auto" else 1.35
                if free_bytes and float(params_b) * per_param * 1024 ** 3 + 1.5 * 1024 ** 3 > free_bytes:
                    print(f"[PromptForge] {'bf16' if quant == 'auto' else quant} does not fit the free VRAM; using q4 instead.")
                    quant = "q4"
            quant_cfg = None
            if quant in ("q8", "q4"):
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=(quant == "q8"),
                    load_in_4bit=(quant == "q4"),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=hf_dtype,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            device_map = "auto"
            if quant in ("q8", "q4") and torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
            load_kwargs = {
                "quantization_config": quant_cfg,
                "device_map": device_map,
                "attn_implementation": "sdpa",
                **_hf_dtype_kwargs(hf_dtype),
            }
            if torch.cuda.is_available() and device_map == "auto":
                max_memory = _hf_max_memory()
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
            model = processor = tokenizer = None

            def _load():
                try:
                    return _load_hf_model(src, load_kwargs)
                except (TypeError, ValueError) as exc:
                    message = str(exc).lower()
                    if "attn_implementation" in message or "sdpa" in message:
                        load_kwargs.pop("attn_implementation", None)
                        return _load_hf_model(src, load_kwargs)
                    raise

            try:
                model = _load()
                device_map = getattr(model, "hf_device_map", None) or {}
                if any(str(device) in ("cpu", "disk") for device in device_map.values()):
                    print("[PromptForge] model is partly on CPU/disk; generation will be slower (use q4 or a smaller model).")
                processor = AutoProcessor.from_pretrained(src)
                tokenizer = AutoTokenizer.from_pretrained(src)
                conversation = []
                if (system_prompt or "").strip():
                    conversation.append({"role": "system", "content": system_prompt})
                refs = self._mode_images(mode, image, image2, image3, video, frame_count)
                if refs and getattr(processor, "image_processor", None) is None:
                    raise ValueError("[PromptForge] The selected safetensors model has no image processor; image/video mode needs a vision-capable model.")
                user_content = user_prompt or full_prompt
                if refs:
                    user_content = [{"type": "image", "image": r} for r in refs if r is not None]
                    user_content.append({"type": "text", "text": user_prompt or full_prompt})
                conversation.append({"role": "user", "content": user_content})
                chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=not disable_thinking)
                processed = processor(text=chat, images=[r for r in refs if r is not None] or None, return_tensors="pt")
                try:
                    model_device = model.get_input_embeddings().weight.device
                except Exception:
                    model_device = next(
                        (param.device for param in model.parameters() if param.device.type != "meta"),
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    )
                if model_device.type == "meta":
                    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                allowed_inputs = set(inspect.signature(model.forward).parameters)
                prepare = getattr(model, "prepare_inputs_for_generation", None)
                if prepare is not None:
                    allowed_inputs |= set(inspect.signature(prepare).parameters)
                allowed_inputs.discard("kwargs")
                allowed_inputs.discard("model_kwargs")
                model_inputs = {
                    k: (v.to(model_device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in processed.items()
                    if k in allowed_inputs
                }
                stop_tokens = [tokenizer.eos_token_id]
                if getattr(tokenizer, "eot_id", None) is not None:
                    stop_tokens.append(tokenizer.eot_id)
                kwargs = {
                    "max_new_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "num_beams": num_beams,
                    "eos_token_id": stop_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if max_tokens >= 8:
                    kwargs["no_repeat_ngram_size"] = 4
                if num_beams == 1:
                    kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
                    if params_b is not None and params_b >= 3.5:
                        kwargs["top_k"] = top_k or 20

                input_len = model_inputs["input_ids"].shape[-1]

                class _FiniteLogits(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

                class _LoopStop(StoppingCriteria):
                    def __init__(self):
                        self.step = 0

                    def __call__(self, input_ids, scores, **kw):
                        self.step += 1
                        if self.step % 32:
                            return False
                        text = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)
                        return _is_repetitive_text(text)

                kwargs["stopping_criteria"] = StoppingCriteriaList([_LoopStop()])
                kwargs["logits_processor"] = LogitsProcessorList([_FiniteLogits()])

                def attempt(seed_value):
                    torch.manual_seed(seed_value)
                    started = time.time()
                    outputs = model.generate(**model_inputs, **kwargs)
                    tokens = outputs.shape[-1] - input_len
                    if tokens:
                        print(f"[PromptForge] safetensors generated {tokens} tokens in {time.time() - started:.1f}s.")
                    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

                return _generate_checked(attempt, seed, min_chars, auto_retry)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise ValueError(
                        "[PromptForge] not enough free VRAM for this safetensors quantization. "
                        "Use q4, a smaller model, or the GGUF version."
                    ) from exc
                raise
            finally:
                del model, processor, tokenizer

        import comfy.sd
        try:
            clip_obj = comfy.sd.load_clip(ckpt_paths=[src])
        except Exception as exc:
            raise ValueError(f"[PromptForge] ComfyUI could not load this text encoder: {exc}") from exc
        enhanced = self._generate_with_clip(
            system_prompt=system_prompt, user_prompt=user_prompt or full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
            clip=clip_obj, disable_thinking=disable_thinking, auto_retry=auto_retry, min_chars=min_chars,
        )
        return enhanced

    # ------------------------------------------------------------------ #
    # Native ComfyUI CLIP generation (connected clip or locally loaded clip)
    # ------------------------------------------------------------------ #
    def _generate_with_clip(self, system_prompt, user_prompt, mode, max_tokens, temperature, top_p, top_k, repetition_penalty, seed, image, image2, image3, video, frame_count, clip, disable_thinking, auto_retry, min_chars=20):
        global _TOP_P_WARNED
        images = self._clip_images(mode, image, image2, image3, video, frame_count)
        # Gemma/LTX tokenizers take one [B, H, W, C] batch as `image`; Qwen tokenizers also accept an `images` list.
        tokenize_kwargs = {"min_length": 1, "thinking": not disable_thinking, "system_prompt": system_prompt or ""}
        if images:
            if all(t.shape == images[0].shape for t in images):
                tokenize_kwargs["image"] = torch.cat(images, dim=0)
            else:
                tokenize_kwargs["images"] = images
        tokens = clip.tokenize(user_prompt, **tokenize_kwargs)
        if system_prompt and system_prompt[:40] not in _tokens_text(clip, tokens):
            # this encoder's tokenizer has no chat template (Anima / ACE Qwen encoders): wrap the turns in ChatML ourselves
            chat = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if disable_thinking:
                chat += "<think>\n\n</think>\n\n"
            tokens = clip.tokenize(chat, **tokenize_kwargs)
        if images and not _tokens_have_images(tokens):
            raise ValueError("[PromptForge] The selected text encoder has no vision support; the provided image/video would be ignored. Connect a vision-capable encoder or use text_only.")

        effective_top_p = top_p
        if top_p < 1.0 and not _clip_top_p_supported():
            effective_top_p = 1.0
            if not _TOP_P_WARNED:
                print("[PromptForge] ComfyUI's CLIP sampler top_p is incompatible with this torch build; using top_p=1.0 (temperature, top_k, repetition and seed still apply).")
                _TOP_P_WARNED = True

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 2 * 1024 ** 3:
            _free_vram(getattr(clip, "patcher", None))

        max_length = min(max_tokens, 2048)
        if max_tokens > max_length:
            print("[PromptForge] the connected CLIP sampler is limited to 2048 tokens; using 2048.")

        def run(top_p_value, seed_value):
            return clip.generate(
                tokens,
                do_sample=True,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p_value,
                min_p=0.0,
                repetition_penalty=repetition_penalty,
                seed=seed_value,
                presence_penalty=0.0,
            )

        def attempt(seed_value):
            try:
                return clip.decode(run(effective_top_p, seed_value), skip_special_tokens=True)
            except AttributeError as exc:
                raise ValueError("[PromptForge] this text encoder cannot generate text (CLIP/T5-style encoder). Use a Qwen / Gemma LLM encoder, or a safetensors/GGUF model.") from exc
            except RuntimeError as exc:
                if effective_top_p < 1.0 and "single memory location" in str(exc):
                    print("[PromptForge] ComfyUI's CLIP sampler hit a torch overlap error on top_p; retrying with top_p=1.0 (temperature, top_k, repetition and seed still apply).")
                    return clip.decode(run(1.0, seed_value), skip_special_tokens=True)
                raise

        return _generate_checked(attempt, seed, min_chars, auto_retry)

    # ------------------------------------------------------------------ #
    # Input adapters
    # ------------------------------------------------------------------ #
    def _clip_images(self, mode, image, image2, image3, video, frame_count):
        """Build the [1, H, W, C] tensor list ComfyUI multimodal tokenizers expect."""
        if mode == "text_only":
            return []
        images = []
        for tensor in (image, image2, image3):
            if torch.is_tensor(tensor) and tensor.ndim == 4:
                images.append(tensor[:1])
            elif torch.is_tensor(tensor) and tensor.ndim == 3:
                images.append(tensor.unsqueeze(0))
        if torch.is_tensor(video) and video.ndim == 4 and video.shape[0] > 0:
            steps = min(max(1, frame_count), video.shape[0])
            idx = torch.linspace(0, video.shape[0] - 1, steps=steps).round().long()
            images.extend(video[i:i + 1] for i in idx)
        fitted = []
        for t in images:
            t = t.detach().float().cpu().clamp(0.0, 1.0)
            size = _vision_size(t.shape[2], t.shape[1])
            if size is not None:
                t = F.interpolate(t.permute(0, 3, 1, 2), size=(size[1], size[0]), mode="area").permute(0, 2, 3, 1)
            fitted.append(t)
        return fitted

    def _media_inventory(self, mode, image, image2, image3, video, frame_count):
        """Describe only the media that actually reached the node."""
        if mode == "text_only":
            return (
                "# Provided inputs (authoritative)\n"
                "No media is used in text_only mode. Write a text-only prompt. "
                "Do not mention images, references, video, or frames."
            )
        images = sum(1 for value in (image, image2, image3) if self._has_media(value))
        frames = 0
        if self._has_media(video):
            if torch.is_tensor(video) and video.ndim == 4:
                frames = min(max(1, int(frame_count)), int(video.shape[0]))
            elif isinstance(video, (list, tuple)):
                frames = min(max(1, int(frame_count)), len(video))
            else:
                frames = 1
        media = []
        if images == 1:
            media.append("1 image")
        elif images > 1:
            media.append(f"{images} images")
        if frames:
            media.append(f"1 video ({frames} sampled frame{'s' if frames != 1 else ''})")
        if not media:
            return (
                "# Provided inputs (authoritative)\n"
                "No image or video is connected. Write a text-only prompt. "
                "Do not mention images, references, video, or frames."
            )
        return (
            "# Provided inputs (authoritative)\n"
            f"The only media connected to this node: {' and '.join(media)}. "
            "Nothing else is connected: no additional images, references, video, or frames. "
            "Write the prompt for exactly these inputs. Never mention, imply, reference, or invent "
            "any other image, reference, video, frame, or media."
        )

    def _minimax_inputs(self, mode_key, mode, image, image2, image3, video, frame_count, seconds):
        """Trim the connected media to what the MiniMax mode can use and describe it with the H3 labels."""
        max_images, allows_video, _ = _MINIMAX_MODES[mode_key]
        pictures = [value for value in (image, image2, image3) if mode != "text_only" and self._has_media(value)]
        has_video = mode != "text_only" and self._has_media(video)
        dropped = []
        if max_images is not None and len(pictures) > max_images:
            dropped.extend(f"image {index + 1}" for index in range(max_images, len(pictures)))
            pictures = pictures[:max_images]
        if has_video and not allows_video:
            dropped.append("video")
            has_video = False
        if dropped:
            print(f"[PromptForge] MiniMax {mode_key.upper()} cannot use: {', '.join(dropped)}; ignoring.")
        pictures += [None] * (3 - len(pictures))
        video = video if has_video else None
        frames = 0
        if has_video:
            frames = min(max(1, int(frame_count)), int(video.shape[0])) if torch.is_tensor(video) and video.ndim == 4 else 1
        block = _minimax_inputs_block(mode_key, sum(value is not None for value in pictures), frames, dropped, seconds)
        return pictures[0], pictures[1], pictures[2], video, block

    def _has_media(self, value):
        if value is None:
            return False
        if torch.is_tensor(value):
            return value.numel() > 0
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        return True

    def _mode_images(self, mode, image, image2, image3, video, frame_count):
        if mode == "text_only":
            return []
        images = [self._pil_from_input(image), self._pil_from_input(image2), self._pil_from_input(image3)]
        images = [p for p in images if p is not None]
        if video is not None:
            images.extend(self._frames_from_input(video, frame_count))
        return [self._fit_for_vision(p) for p in images]

    def _fit_for_vision(self, im):
        size = _vision_size(im.width, im.height)
        if size is None:
            return im
        from PIL import Image
        return im.resize(size, Image.LANCZOS)

    def _pil_from_input(self, img):
        from PIL import Image
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        if torch.is_tensor(img):
            img = img.detach().float().cpu()
            if img.ndim == 4:
                img = img[0]
            arr = img.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        else:
            import numpy as np
            arr = np.asarray(img)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        im = Image.fromarray(arr)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im

    def _pil_data_url(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _frames_from_input(self, video, frame_count):
        from PIL import Image
        frames = []
        if video is None:
            return frames
        if isinstance(video, list) and all(isinstance(f, Image.Image) for f in video):
            frames = list(video)
        elif torch.is_tensor(video) and video.ndim == 4:
            steps = min(max(1, frame_count), video.shape[0])
            idx = torch.linspace(0, video.shape[0] - 1, steps=steps).round().long()
            frames = [self._pil_from_input(video[i]) for i in idx]
        return [f for f in frames if f is not None][: max(1, frame_count)]


NODE_CLASS_MAPPINGS = {
    "PROMPTFORGE_Model": PROMPTFORGE_Model,
    "PROMPTFORGE_PromptEnhancer": PROMPTFORGE_PromptEnhancer,
    "PROMPTFORGE_ConfigFinder": PROMPTFORGE_ConfigFinder,
    **VFI_CLASS_MAPPINGS,
    **SAVE_CLASS_MAPPINGS,
    **LORA_CLASS_MAPPINGS,
    **LOADER_CLASS_MAPPINGS,
    **LATENT_CLASS_MAPPINGS,
    **SAMPLER_CLASS_MAPPINGS,
    **QUANT_CLASS_MAPPINGS,
    **DECODE_CLASS_MAPPINGS,
    **IMAGE_CLASS_MAPPINGS,
    **ENDLESS_CLASS_MAPPINGS,
    **PREVIEW_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPTFORGE_Model": "PromptForge Model",
    "PROMPTFORGE_PromptEnhancer": "PromptForge Enhancer",
    "PROMPTFORGE_ConfigFinder": "PromptForge Config Finder",
    **VFI_DISPLAY_NAMES,
    **SAVE_DISPLAY_NAMES,
    **LORA_DISPLAY_NAMES,
    **LOADER_DISPLAY_NAMES,
    **LATENT_DISPLAY_NAMES,
    **SAMPLER_DISPLAY_NAMES,
    **QUANT_DISPLAY_NAMES,
    **DECODE_DISPLAY_NAMES,
    **IMAGE_DISPLAY_NAMES,
    **ENDLESS_DISPLAY_NAMES,
    **PREVIEW_DISPLAY_NAMES,
}

WEB_DIRECTORY = "./web"
