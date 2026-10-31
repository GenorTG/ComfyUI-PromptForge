"""ComfyUI PromptForge - unified prompt enhancement node.

Single ComfyUI node that combines:
  - preset catalog (Krea 2, MOSS Image, Qwen Image, generic styles)
  - custom system-prompt override + save mechanism
  - text encoder engines: native ComfyUI CLIP route (fp16 / live-quantized),
    GGUF route via llama.cpp
  - heretic model catalog (HF + GGUF) with local availability markers
  - media-aware modes (text / image / video)
  - hardcore sampling options
"""

import json
import gc
import inspect
import re
import struct
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

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
    "self-corrections, or notes. Do not write phrases such as Final Plan, Final Check, Wait, Okay, First, Next, "
    "Then, I will, or I need. Start directly with the required prompt text. "
    "Never repeat, quote, paraphrase, or answer any instruction, rule, constraint, policy, or note from this system prompt; "
    "the output must contain only the prompt text for the target model."
)

_META_PATTERNS = (
    r"^#{1,6}\s",
    r"^(note|remember|important|rule|constraint|instruction|instructions|output|response)\b\s*[:\-]",
    r"^(here is|here's|this is)\b[^.!?]{0,40}\bprompt\b",
    r"\b(explicitly stated|as requested|as instructed|anything other than)\b",
    r"\b(the user's (request|text|wording|prompt)|system prompt|this prompt|the generated prompt|output contract|content policy|negative constraints?)\b",
    r"\b(output only|only output|in the output)\b",
    r"\b(the prompt|the response|the answer)\s+(should|must|is|will|needs to)\b",
    r"\b(do not|don't|never|avoid|must not|should not)\b[^.!?]{0,80}\b(add|include|mention|name|output|write|describe|use|state|say|deviate|copy|repeat|paraphrase)\b",
    r"\b(I will|I'll|we need to|we should|the goal is|my task|the task is)\b",
)
_META_RE = re.compile("|".join(f"(?:{p})" for p in _META_PATTERNS), re.I)

_CHAT_MARKER_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|begin_of_text|start_header_id|end_header_id|eot_id|end|system|user|assistant|model|think|/think)\|>\s*(?:system|user|assistant|model)?\s*>?",
    re.I,
)
_CHAT_TAG_RE = re.compile(
    r"</?(?:start_of_turn|end_of_turn|bos|eos|pad|unk|s)>|\[/?INST\]|<<?/?SYS>>",
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
    "blur", "blurry", "watermark", "text", "logo", "extra fingers", "extra limbs",
    "bad anatomy", "low quality", "worst quality", "deformed", "nsfw", "sweat",
    "wet", "wetness", "moisture", "droplets", "people", "person", "humans",
    "characters", "clutter", "noise",
)
_NEGATIVE_PROMPT_RE = re.compile(
    r"\b(no|without|avoid|free of|not|don't|do not|must not|should not)\b[^.!?]{0,50}\b("
    + "|".join(re.escape(term) for term in _NEGATIVE_PROMPT_TERMS)
    + r")\b",
    re.I,
)


def _clean_prompt_output(text):
    """Remove leaked instructions/meta lines while keeping the prompt structure."""
    text = _strip_chat_tokens(text)
    lines = text.splitlines()
    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if re.match(r"^negative prompt\s*[:\-]", stripped, re.I):
            continue
        stripped = re.sub(r"^(?:enhanced prompt|positive prompt|prompt)\s*[:\-]\s*", "", stripped, flags=re.I)
        if not stripped:
            continue
        if _META_RE.search(stripped):
            parts = re.split(r"(?<=[.!?])\s+", stripped)
            kept = [part for part in parts if not _META_RE.search(part)]
            if kept:
                kept_lines.append(" ".join(kept).strip())
            continue
        kept_lines.append(stripped)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned.strip('"').strip("`").strip()

THINK_TAGS = (
    ("\u27e8think\u27e9", "\u27e8/think\u27e9"),
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
)

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


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
    if not text or len(text) < 60:
        return False
    tail = text[-1500:]
    for size in (60, 40, 24):
        block = tail[-size:]
        if len(block) >= size and tail.count(block) >= 3:
            return True
    sentences = [s.strip() for s in re.split(r"[.!?]+", tail) if len(s.strip()) > 40]
    if sentences:
        from collections import Counter
        if Counter(sentences).most_common(1)[0][1] >= 3:
            return True
    words = tail.split()
    if len(words) >= 40:
        from collections import Counter
        n = 6
        grams = [" ".join(words[i:i + n]) for i in range(0, len(words) - n, 3)]
        if grams and Counter(grams).most_common(1)[0][1] >= 4:
            return True
    return False


def _next_seed(seed):
    return (int(seed) + 1) % (2 ** 32 - 1) or 1


_TOP_P_WARNED = False


def _clip_top_p_supported():
    try:
        major, minor = (int(part) for part in str(torch.__version__).split("+")[0].split(".")[:2])
        return (major, minor) < (2, 13)
    except Exception:
        return True


def _is_bad_output(text):
    """Detect empty, degenerate or refusal output. Content is never logged."""
    clean = _clean_prompt_output(text)
    if len(clean) < 20:
        return True
    if _is_repetitive_text(clean):
        return True
    if _NEGATIVE_PROMPT_RE.search(clean):
        return True
    if re.search(r"(.)\1{15,}", clean):
        return True
    if len(clean) > 200 and len(set(clean)) < 8:
        return True
    words = re.findall(r"\w+", clean.lower())
    if len(words) > 30:
        from collections import Counter
        if Counter(words).most_common(1)[0][1] / len(words) > 0.45:
            return True
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if len(lines) >= 4:
        from collections import Counter
        if Counter(lines).most_common(1)[0][1] >= 4:
            return True
    lowered = clean.lower()
    if len(clean) < 600 and any(marker in lowered for marker in REFUSAL_MARKERS):
        return True
    if any(lowered.startswith(marker) for marker in REFUSAL_MARKERS):
        return True
    return False


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


def _generation_defaults_from_files(src):
    """Read sampling defaults that actually exist in a model's generation_config.json."""
    if not src:
        return {}
    path = Path(src)
    if path.is_dir():
        config_path = path / "generation_config.json"
    elif path.suffix.lower() in (".safetensors", ".bin", ".gguf"):
        config_path = path.parent / "generation_config.json"
    else:
        return {}
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {key: config[key] for key in SAMPLING_KEYS if isinstance(config.get(key), (int, float)) and not isinstance(config.get(key), bool)}


def _generation_defaults_from_gguf(llm):
    """Read llama.cpp metadata sampling defaults when the GGUF carries them."""
    metadata = getattr(llm, "metadata", None) or {}
    defaults = {}
    for meta_key, key in (
        ("general.sampling.temperature", "temperature"),
        ("general.sampling.top_p", "top_p"),
        ("general.sampling.top_k", "top_k"),
        ("general.sampling.repetition_penalty", "repetition_penalty"),
        ("general.sampling.repeat_penalty", "repetition_penalty"),
    ):
        value = metadata.get(meta_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            defaults[key] = value
    return defaults


def _apply_sampling_defaults(defaults, temperature, top_p, top_k, repetition_penalty):
    return (
        defaults.get("temperature", temperature),
        defaults.get("top_p", top_p),
        defaults.get("top_k", top_k),
        defaults.get("repetition_penalty", repetition_penalty),
    )

DEFAULT_PRESET = "Enhance"

HERETIC_MODELS = PROMPTFORGE_MODELS.get("heretic_hf_models") or []
GGUF_MODELS = PROMPTFORGE_MODELS.get("gguf_models") or []
LOCAL_SCAN_DIRS = PROMPTFORGE_MODELS.get("local_scan_dirs") or []

def _scan_local_text_models():
    """Scan ComfyUI text-model directories for local LLM / text encoder models.

    Multi-part checkpoints collapse to their containing HF directory and mmproj
    vision sidecars are skipped, so the picker shows one entry per model.
    """
    roots = []
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", "") or ""
        if models_dir:
            roots.append(Path(models_dir))
    roots.append(Path("/home/genortg/ComfyUI/Comfy/models"))
    roots.append(Path("/home/genortg/1TB-SAMSUNG/Comfy/ComfyUI/models"))

    found = []
    seen_paths = set()
    for root in roots:
        for sub in LOCAL_SCAN_DIRS:
            c = root / sub
            if not c.is_dir():
                continue
            for p in sorted(c.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in (".safetensors", ".gguf", ".bin"):
                    continue
                name = p.name.lower()
                if p.suffix.lower() == ".gguf" and "mmproj" in name:
                    continue
                if p.suffix.lower() in (".safetensors", ".bin") and (p.parent / "config.json").is_file():
                    model_path = p.parent
                else:
                    model_path = p
                key = str(model_path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                found.append({"path": key, "name": model_path.name, "kind": "gguf" if p.suffix.lower() == ".gguf" else "safetensors"})
    return found


LOCAL_TEXT_MODELS = _scan_local_text_models()

def _catalog_entry(item, kind):
    return {
        "option": item.get("label") or item.get("name") or item.get("key"),
        "kind": kind,
        "key": item.get("key"),
        "params_b": item.get("params_b"),
        "avail_match": item.get("avail_match") or [],
    }


HERETIC_MODEL_ENTRIES = (
    [_catalog_entry(item, "hf") for item in HERETIC_MODELS]
    + [_catalog_entry(item, "gguf") for item in GGUF_MODELS]
)


def _entry_local_path(entry):
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


def _entry_local(entry):
    return bool(_entry_local_path(entry))


def _entry_label(entry):
    label = entry["option"]
    if _entry_local(entry):
        label = "[AVAILABLE] " + label
    return label


SOURCE_OPTIONS = ["safetensors", "GGUF", "connected clip"]

SAFETENSORS_MODEL_OPTIONS = (
    [_entry_label(e) for e in HERETIC_MODEL_ENTRIES if e["kind"] == "hf"]
    + [e["path"] for e in LOCAL_TEXT_MODELS if (e.get("path") or "") and not e["path"].lower().endswith(".gguf")]
) or ["none"]

GGUF_MODEL_OPTIONS = (
    [_entry_label(e) for e in HERETIC_MODEL_ENTRIES if e["kind"] == "gguf"]
    + [e["path"] for e in LOCAL_TEXT_MODELS if (e.get("path") or "").lower().endswith(".gguf")]
) or ["none"]


def _preferred_option(options):
    return options[0] if options else "none"

PRESET_OPTIONS = ["auto"] + sorted(
    k for k, v in PROMPTFORGE_PRESETS.items()
    if isinstance(v, dict) and (v.get("text") or v.get("system_prompt"))
)

MODE_OPTIONS = ["text_only", "inputs"]
QUANT_OPTIONS = ["auto", "fp16", "q8", "q6", "q5", "q4"]
KV_CACHE_OPTIONS = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
FLASH_ATTN_OPTIONS = ["auto", "on", "off"]
GGUF_QUANT_ORDER = ("q8", "q6", "q5", "q4")


def _gguf_quant_order(quantization, quant_files):
    requested = {
        "q8": "q8", "8bit": "q8", "q6": "q6", "6bit": "q6",
        "q5": "q5", "5bit": "q5", "q4": "q4", "4bit": "q4",
        "fp16": "q8", "auto": "q8",
    }.get(quantization or "auto", "q8")
    order = [requested] + [key for key in GGUF_QUANT_ORDER if key != requested]
    return [key for key in order if key in quant_files]


def _resolve_model_entry(option):
    """Map a friendly model option to a runtime descriptor.

    '[AVAILABLE] ' prefixes mark locally downloaded models and are stripped here.
    """
    if not option or option == "auto":
        ent = HERETIC_MODELS[0] if HERETIC_MODELS else {}
        first = HERETIC_MODEL_ENTRIES[0] if HERETIC_MODEL_ENTRIES else {}
        return {"kind": "hf", "key": (ent or {}).get("key") or "heretic_qwen35_9b", "params_b": (ent or {}).get("params_b") or 9, "repo_id": (ent or {}).get("repo_id"), "local_path": _entry_local_path(first) if first else ""}
    label = str(option)
    if label.startswith("[AVAILABLE] "):
        label = label[len("[AVAILABLE] "):]
    for e in HERETIC_MODEL_ENTRIES:
        if e["option"] == label:
            if e["kind"] == "hf":
                ent = next((m for m in HERETIC_MODELS if m.get("key") == e["key"]), {})
                return {"kind": "hf", "key": e["key"], "params_b": e["params_b"], "repo_id": ent.get("repo_id"), "local_path": _entry_local_path(e)}
            ent = next((g for g in GGUF_MODELS if g.get("key") == e["key"]), {})
            return {"kind": "gguf", "key": e["key"], "params_b": e["params_b"], "entry": ent}
    return None


def _local_model_option(option):
    """Return the path when the picker option is a scanned local model file."""
    if not option:
        return ""
    opt = str(option)
    if opt == "auto":
        return ""
    return opt if any(opt == e.get("path") for e in LOCAL_TEXT_MODELS) else ""


def _model_kind(model_name, local_model_path):
    """Return 'gguf' or 'text_encoder' for the effective model."""
    src = (local_model_path or "").strip() or _local_model_option(model_name)
    if src:
        return "gguf" if str(src).lower().endswith(".gguf") else "text_encoder"
    entry = _resolve_model_entry(model_name if model_name and model_name != "auto" else "auto")
    if not entry:
        return "text_encoder"
    return entry.get("kind") or "text_encoder"


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
    import psutil
    try:
        free_bytes = torch.cuda.mem_get_info()[0]
    except Exception:
        return None
    budget = max(int(free_bytes) - reserve, 768 * 1024 ** 2)
    return {torch.cuda.current_device(): budget, "cpu": psutil.virtual_memory().available}


def _free_vram(keep=None):
    """Unload ComfyUI's models so a separately loaded transformers model has room."""
    try:
        import comfy.model_management as model_management
        keep_loaded = []
        if keep is not None:
            keep_loaded = [m for m in model_management.current_loaded_models if m.model is keep]
        model_management.free_memory(10 ** 30, model_management.get_torch_device(), keep_loaded=keep_loaded)
        model_management.free_memory(10 ** 30, torch.device("cpu"))
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
    extra = mmproj_path.stat().st_size if mmproj_path is not None else 0
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
            path = _find_gguf(entry, q, download=False)
            if path is not None:
                candidates.append((q, path))
    else:
        path = _find_gguf(entry, quantization, download=False)
        if path is not None:
            candidates.append((quantization, path))
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
        "swa_full": False, "flash_attn": "auto", "auto_fit": False,
    }
    settings.update(fit)
    return settings


def _auto_safetensors_settings(resolved):
    budget = _gpu_budget_bytes()
    params_b = (resolved or {}).get("params_b")
    if not params_b or budget <= 0:
        return {}
    if params_b * 2.0 * 1024 ** 3 <= budget * 0.85:
        return {"quantization": "auto"}
    if params_b * 1.1 * 1024 ** 3 <= budget * 0.85:
        return {"quantization": "q8"}
    return {"quantization": "q4"}


def _gguf_layer_budget(path, n_ctx, mmproj_path=None):
    """Layers that fit on the GPU alongside their KV cache; the rest go to CPU."""
    profile = _gguf_kv_profile(path)
    if profile is None:
        return None
    n_layers, kv_per_token, swa = profile
    if swa is not None:
        kv_bytes = kv_per_token * (min(n_ctx, swa) * 0.95 + n_ctx * 0.05)
    else:
        kv_bytes = kv_per_token * n_ctx
    weights = path.stat().st_size
    extra = mmproj_path.stat().st_size if mmproj_path is not None else 0
    try:
        free = torch.cuda.mem_get_info()[0]
    except Exception:
        return None
    budget = free - 1536 * 1024 ** 2 - extra
    if budget <= 0:
        return 0, n_layers
    per_layer = (weights + kv_bytes) / n_layers
    return max(0, min(n_layers, int(budget / per_layer))), n_layers


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


def _find_gguf(entry, quantization, download=True):
    """Find or download a GGUF for a catalog entry, honouring the requested quant."""
    qf = (entry or {}).get("quant_files") or {}
    if not qf:
        return None
    order = _gguf_quant_order(quantization, qf) or list(qf)

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


def _user_dir():
    try:
        if folder_paths is not None:
            return Path(getattr(folder_paths, "user_directory", "") or "/home/genortg/ComfyUI/ComfyUI/user")
    except Exception:
        pass
    return Path("/home/genortg/ComfyUI/ComfyUI/user")


def _profiles_file():
    return _user_dir() / "promptforge_profiles.json"


def _load_profiles():
    try:
        data = json.loads(_profiles_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_profile(name, settings):
    profiles = _load_profiles()
    profiles[name] = settings
    path = _profiles_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")


def _option_for_key(key):
    for entry in HERETIC_MODEL_ENTRIES:
        if entry.get("key") == key:
            return entry.get("option")
    return None


def _settings_report(settings):
    source = settings.get("source", "safetensors")
    if source == "GGUF":
        return (
            f"GGUF {settings.get('gguf_model')} | quant {settings.get('quantization')} | "
            f"ctx {settings.get('context_size')} | gpu_layers {settings.get('gpu_layers')} | "
            f"kv {settings.get('kv_cache_type')} | batch {settings.get('batch_size')}/{settings.get('ubatch_size')}"
        )
    if source == "safetensors":
        return f"safetensors {settings.get('safetensors_model')} | quant {settings.get('quantization')}"
    return "connected clip | no model loaded"


def _recommend_settings(data):
    source = data.get("source", "safetensors")
    quantization = data.get("quantization", "auto")
    settings = {"source": source, "quantization": quantization}
    if source == "GGUF":
        resolved = _resolve_model_entry(data.get("gguf_model", "auto"))
        entry = resolved.get("entry") if resolved else None
        path = _find_gguf(entry, quantization, download=False) if entry else None
        mmproj = _local_mmproj(path, entry) if path is not None else None
        settings.update(_auto_gguf_settings(entry, quantization, mmproj))
    elif source == "safetensors":
        settings.update(_auto_safetensors_settings(_resolve_model_entry(data.get("safetensors_model", "auto"))))
    for key, value in (
        ("context_size", 16384), ("gpu_layers", -1), ("batch_size", 2048), ("ubatch_size", 512),
        ("kv_cache_type", "f16"), ("offload_kqv", True), ("swa_full", False),
        ("flash_attn", "auto"), ("auto_fit", False),
    ):
        settings.setdefault(key, value)
    return settings


def _scan_configs(data):
    scope = data.get("scope", "all local")
    quantization = data.get("quantization", "auto")
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
        for option in SAFETENSORS_MODEL_OPTIONS:
            if not option.startswith("[AVAILABLE] "):
                continue
            cfg = _auto_safetensors_settings(_resolve_model_entry(option))
            cfg.update({"source": "safetensors", "safetensors_model": option, "gguf_model": "auto"})
            key = option[len("[AVAILABLE] "):]
            configs[key] = cfg
            lines.append(f"{key}: {_settings_report(cfg)}")
    else:
        cfg = _recommend_settings(data)
        cfg["gguf_model"] = data.get("gguf_model", "auto")
        cfg["safetensors_model"] = data.get("safetensors_model", "auto")
        configs[cfg["source"]] = cfg
        lines.append(_settings_report(cfg))
    if data.get("write_config", True):
        path = _user_dir() / "promptforge_autoconfig.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8")
    prefix = (data.get("profile_prefix") or "").strip()
    if prefix:
        for name, cfg in configs.items():
            _save_profile(prefix + name, cfg)
    if not lines:
        lines.append("No local models found. Download a model first or use 'selected'.")
    return {"report": "\n".join(lines), "configs": configs}


class PROMPTFORGE_Model:
    CATEGORY = "🔮 PromptForge"
    RETURN_TYPES = ("PROMPTFORGE_MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "REPORT")
    FUNCTION = "build"
    DESCRIPTION = "PromptForge model and runtime settings. Auto-configures context, KV cache, batch and offload for this GPU."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (SOURCE_OPTIONS, {"default": "safetensors", "tooltip": "Model backend: safetensors (transformers), GGUF (llama.cpp) or connected clip (no model load)."}),
                "safetensors_model": (SAFETENSORS_MODEL_OPTIONS, {"default": _preferred_option(SAFETENSORS_MODEL_OPTIONS), "tooltip": "Safetensors / HF model used when source is safetensors."}),
                "gguf_model": (GGUF_MODEL_OPTIONS, {"default": _preferred_option(GGUF_MODEL_OPTIONS), "tooltip": "GGUF model used when source is GGUF."}),
                "quantization": (QUANT_OPTIONS, {"default": "auto", "tooltip": "Quantization. auto picks the best local option for this GPU."}),
                "auto_config": ("BOOLEAN", {"default": True, "tooltip": "Compute context, KV cache, batch and offload from the GPU and the selected model."}),
                "profile_name": ("STRING", {"default": "", "tooltip": "Name used by load_profile / save_profile."}),
                "load_profile": ("BOOLEAN", {"default": False, "tooltip": "Load profile_name and use it instead of auto/manual settings."}),
                "save_profile": ("BOOLEAN", {"default": False, "tooltip": "Save the effective settings under profile_name."}),
                "custom_repo_id": ("STRING", {"default": "", "tooltip": "Optional custom HuggingFace repo id for the safetensors source."}),
                "local_model_path": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional local safetensors dir/file or GGUF file; overrides the model picker."}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "Optional mmproj (.gguf) for GGUF image/video mode."}),
                "context_size": ("INT", {"default": 16384, "min": 512, "max": 262144, "step": 512, "tooltip": "GGUF context length when auto_config is off."}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "tooltip": "GGUF layers on the GPU when auto_config is off: -1 = all, 0 = CPU."}),
                "batch_size": ("INT", {"default": 2048, "min": 64, "max": 32768, "step": 64, "tooltip": "GGUF n_batch when auto_config is off."}),
                "ubatch_size": ("INT", {"default": 512, "min": 32, "max": 16384, "step": 32, "tooltip": "GGUF n_ubatch when auto_config is off."}),
                "kv_cache_type": (KV_CACHE_OPTIONS, {"default": "f16", "tooltip": "GGUF KV cache precision when auto_config is off."}),
                "offload_kqv": ("BOOLEAN", {"default": True, "tooltip": "Keep the GGUF KV cache on the GPU when auto_config is off."}),
                "swa_full": ("BOOLEAN", {"default": False, "tooltip": "Full-size sliding-window cache when auto_config is off."}),
                "flash_attn": (FLASH_ATTN_OPTIONS, {"default": "auto", "tooltip": "llama.cpp flash attention when auto_config is off."}),
                "auto_fit": ("BOOLEAN", {"default": False, "tooltip": "Compute the layer split at generation time when auto_config is off."}),
            },
        }

    def build(self, source, safetensors_model, gguf_model, quantization, auto_config, profile_name,
              load_profile, save_profile, custom_repo_id, local_model_path, mmproj_path,
              context_size, gpu_layers, batch_size, ubatch_size, kv_cache_type, offload_kqv,
              swa_full, flash_attn, auto_fit):
        name = (profile_name or "").strip()
        settings = {
            "source": source,
            "safetensors_model": safetensors_model,
            "gguf_model": gguf_model,
            "quantization": quantization,
            "custom_repo_id": custom_repo_id,
            "local_model_path": local_model_path,
            "mmproj_path": mmproj_path,
            "context_size": int(context_size),
            "gpu_layers": int(gpu_layers),
            "batch_size": int(batch_size),
            "ubatch_size": int(ubatch_size),
            "kv_cache_type": kv_cache_type,
            "offload_kqv": bool(offload_kqv),
            "swa_full": bool(swa_full),
            "flash_attn": flash_attn,
            "auto_fit": bool(auto_fit),
            "profile_name": name,
        }
        loaded = False
        if load_profile and name:
            saved = _load_profiles().get(name)
            if saved:
                settings.update(saved)
                loaded = True
                print(f"[PromptForge] profile '{name}' loaded.")
            else:
                print(f"[PromptForge] profile '{name}' not found; using auto/manual settings.")
        if not loaded and auto_config:
            if source == "GGUF":
                resolved = _resolve_model_entry(gguf_model)
                entry = resolved.get("entry") if resolved else None
                path = _find_gguf(entry, quantization, download=False) if entry else None
                mmproj = _local_mmproj(path, entry) if path is not None else None
                settings.update(_auto_gguf_settings(entry, quantization, mmproj))
            elif source == "safetensors":
                settings.update(_auto_safetensors_settings(_resolve_model_entry(safetensors_model)))
            print("[PromptForge] model auto-config applied.")
        if save_profile:
            if not name:
                print("[PromptForge] set profile_name before saving a profile.")
            else:
                _save_profile(name, settings)
                print(f"[PromptForge] profile '{name}' saved.")
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
                "source": (SOURCE_OPTIONS, {"default": "GGUF", "tooltip": "Model backend for the selected scope."}),
                "safetensors_model": (SAFETENSORS_MODEL_OPTIONS, {"default": _preferred_option(SAFETENSORS_MODEL_OPTIONS), "tooltip": "Safetensors model for the selected scope."}),
                "gguf_model": (GGUF_MODEL_OPTIONS, {"default": _preferred_option(GGUF_MODEL_OPTIONS), "tooltip": "GGUF model for the selected scope."}),
                "quantization": (QUANT_OPTIONS, {"default": "auto", "tooltip": "Quantization to plan for; auto picks the best local quant."}),
                "write_config": ("BOOLEAN", {"default": True, "tooltip": "Write the recommendations to user/promptforge_autoconfig.json."}),
                "profile_prefix": ("STRING", {"default": "", "tooltip": "When set, also save every recommendation as a profile named <prefix><model>."}),
            },
        }

    def scan(self, scope, source, safetensors_model, gguf_model, quantization, write_config, profile_prefix):
        result = _scan_configs({
            "scope": scope, "source": source, "safetensors_model": safetensors_model,
            "gguf_model": gguf_model, "quantization": quantization,
            "write_config": write_config, "profile_prefix": profile_prefix,
        })
        for line in result["report"].splitlines():
            print(f"[PromptForge] {line}")
        return {"ui": {"text": [result["report"]]}, "result": (result["report"],)}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/promptforge/recommend")
    async def _promptforge_recommend(request):
        return web.json_response(_recommend_settings(await request.json()))

    @PromptServer.instance.routes.post("/promptforge/scan")
    async def _promptforge_scan(request):
        return web.json_response(_scan_configs(await request.json()))
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
                "max_tokens": ("INT", {"default": 8192, "min": 32, "max": 65536, "tooltip": "Max new tokens for the enhanced prompt. Keep it below the model context for GGUF."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Sampling temperature."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling cutoff."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Top-K. 0 = model default."}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Beam search width. 1 = fast sampling; >1 trades speed for stability."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "Repetition penalty."}),
                "use_default_sampling": ("BOOLEAN", {"default": False, "tooltip": "Use sampling defaults from generation_config.json or GGUF metadata when the model provides them."}),
                "disable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Disable model reasoning/thinking before the answer where supported. Thinking is always stripped from ENHANCED_PROMPT and exposed on THINKING."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep ComfyUI's models resident after generation."}),
                "use_system_prompt_override": ("BOOLEAN", {"default": True, "tooltip": "Use the system_prompt_override text when it is non-empty."}),
                "system_prompt_override": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom system prompt override. Leave empty to use the selected preset."}),
                "save_system_prompt": ("BOOLEAN", {"default": False, "tooltip": "Save the system_prompt_override into the user's PromptForge custom store."}),
            },
            "optional": {
                "video_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "tooltip": "Target video duration in seconds."}),
                "image": ("IMAGE", {"tooltip": "First reference image."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image / edit source."}),
                "image3": ("IMAGE", {"tooltip": "Third reference image / edit source."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": "Frames sampled from video input."}),
                "clip": ("CLIP", {"tooltip": "Connected ComfyUI text encoder. Used when the MODEL source is connected clip."}),
                "auto_retry": ("BOOLEAN", {"default": True, "tooltip": "Retry once with the next seed on empty, repetitive or refusal output."}),
                "use_negative_prompt": ("BOOLEAN", {"default": True, "tooltip": "Apply the negative prompt text to the generation request."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "What the generated prompt/scene must not contain."}),
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
        disable_thinking,
        use_default_sampling,
        keep_model_loaded,
        use_system_prompt_override,
        system_prompt_override,
        save_system_prompt,
        video_seconds=5,
        image=None,
        image2=None,
        image3=None,
        video=None,
        frame_count=16,
        clip=None,
        auto_retry=True,
        use_negative_prompt=True,
        negative_prompt="",
    ):
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
        auto_fit = cfg.get("auto_fit", False)
        custom_repo_id = cfg.get("custom_repo_id", "")
        local_model_path = cfg.get("local_model_path", "")
        mmproj_path = cfg.get("mmproj_path", "")
        override_text = (system_prompt_override or "").strip()
        if save_system_prompt and override_text:
            self._save_custom_system_prompt(override_text)

        sys_text = override_text if (use_system_prompt_override and override_text) else ""
        if not sys_text:
            key = enhancement_preset if enhancement_preset and enhancement_preset in PROMPTFORGE_PRESETS else DEFAULT_PRESET
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
        if disable_thinking:
            guard += " Do not output chain-of-thought or reasoning; start with the final prompt."
        media_line = self._media_inventory(mode, image, image2, image3, video, frame_count)
        negative_text = (negative_prompt or "").strip()
        negative_line = ""
        if use_negative_prompt and negative_text:
            negative_line = (
                "# Negative constraints\n"
                "The generated prompt and the scene it describes must NOT contain any of the following: "
                + negative_text +
                ". Write the scene so they are absent, converting each exclusion into a positive description "
                "of the desired state; never name them inside the prompt itself."
            )
        parts = [p for p in (user_text, duration_line, sys_text, media_line, negative_line, guard) if p and p.strip()]
        full_prompt = "\n\n".join(parts)

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

        enhanced = self._generate(
            full_prompt=full_prompt,
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
            auto_fit=auto_fit,
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
            seed=seed,
            clip=clip_for_generate,
            custom_repo_id=custom_repo_id,
            local_model_path=local_model_path,
            mmproj_path=mmproj_path,
            disable_thinking=disable_thinking,
            use_default_sampling=use_default_sampling,
            auto_retry=auto_retry,
            keep_model_loaded=keep_model_loaded,
        )
        answer, thinking = _split_thinking(enhanced)
        answer = _clean_prompt_output(answer)
        return (answer, thinking)

    # ------------------------------------------------------------------ #
    # Custom system prompt store
    # ------------------------------------------------------------------ #
    def _save_custom_system_prompt(self, text):
        import json as _json
        p = _user_dir() / "promptforge_custom_system_prompts.json"
        data = {}
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        store = data.get("custom_system_prompts") or {}
        store["Custom Override"] = {"text": text}
        data["custom_system_prompts"] = store
        try:
            p.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print("[PromptForge] failed to save custom system prompt.")

    def _load_custom_system_prompts(self):
        p = _user_dir() / "promptforge_custom_system_prompts.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            store = data.get("custom_system_prompts") or {}
            return {k: (v or {}).get("text") or "" for k, v in store.items() if isinstance(v, dict)}
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Model lifecycle: unload after the prompt has been generated
    # ------------------------------------------------------------------ #
    def _release(self, keep_model_loaded, llm=None):
        if keep_model_loaded:
            return
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
            return
        import comfy.model_management as model_management
        model_management.free_memory(10 ** 30, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model_management.free_memory(10 ** 30, torch.device("cpu"))
        model_management.cleanup_models_gc()
        model_management.soft_empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ #
    # Generation dispatch
    # ------------------------------------------------------------------ #
    def _generate(self, full_prompt, mode, engine, model_name, quantization, context_size, gpu_layers, batch_size, ubatch_size, kv_cache_type, offload_kqv, swa_full, flash_attn, auto_fit, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, custom_repo_id, local_model_path, mmproj_path, disable_thinking, use_default_sampling, auto_retry, keep_model_loaded):
        if engine == "gguf":
            return self._generate_gguf(
                full_prompt=full_prompt, mode=mode, model_name=model_name, quantization=quantization,
                context_size=context_size, gpu_layers=gpu_layers, batch_size=batch_size, ubatch_size=ubatch_size,
                kv_cache_type=kv_cache_type, offload_kqv=offload_kqv, swa_full=swa_full,
                flash_attn=flash_attn, auto_fit=auto_fit,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                num_beams=num_beams, repetition_penalty=repetition_penalty, seed=seed,
                image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
                custom_repo_id=custom_repo_id, local_model_path=local_model_path, mmproj_path=mmproj_path,
                disable_thinking=disable_thinking, use_default_sampling=use_default_sampling,
                auto_retry=auto_retry, keep_model_loaded=keep_model_loaded,
            )
        return self._generate_text_encoder(
            full_prompt=full_prompt, mode=mode, model_name=model_name, quantization=quantization,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            num_beams=num_beams, repetition_penalty=repetition_penalty,
            image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
            seed=seed, clip=clip, custom_repo_id=custom_repo_id, local_model_path=local_model_path,
            disable_thinking=disable_thinking, use_default_sampling=use_default_sampling,
            auto_retry=auto_retry, keep_model_loaded=keep_model_loaded,
        )

    # ------------------------------------------------------------------ #
    # GGUF / llama.cpp engine
    # ------------------------------------------------------------------ #
    def _generate_gguf(self, full_prompt, mode, model_name, quantization, context_size, gpu_layers, batch_size, ubatch_size, kv_cache_type, offload_kqv, swa_full, flash_attn, auto_fit, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, seed, image, image2, image3, video, frame_count, custom_repo_id, local_model_path, mmproj_path, disable_thinking, use_default_sampling, auto_retry, keep_model_loaded):
        entry = None
        gguf_path = None
        eff_local = (local_model_path or "").strip() or _local_model_option(model_name)
        if eff_local and str(eff_local).lower().endswith(".gguf"):
            gguf_path = Path(str(eff_local))
        if gguf_path is None:
            resolved = _resolve_model_entry(model_name) if (model_name and model_name != "auto") else _resolve_model_entry("auto")
            if resolved is not None and resolved.get("kind") == "gguf":
                entry = resolved.get("entry")
            if entry is None and resolved is not None:
                entry = next((g for g in GGUF_MODELS if g.get("key") == resolved.get("key")), None)
            if entry is None and GGUF_MODELS:
                entry = GGUF_MODELS[0]
            if entry is not None:
                gguf_path = _find_gguf(entry, quantization)
        if gguf_path is None:
            raise ValueError("[PromptForge] No GGUF model selected / available. Put the .gguf in a scanned models dir (text_encoders / LLM / gguf) or set local_model_path.")

        images = self._mode_images(mode, image, image2, image3, video, frame_count) if mode != "text_only" else []
        mmproj = _resolve_mmproj(gguf_path, entry, mmproj_path) if images else None
        if images and mmproj is None:
            raise ValueError("[PromptForge] Vision GGUF needs an mmproj file next to the model (e.g. mmproj-F16.gguf). Put one in the same folder, set mmproj_path, or switch mode to text_only.")

        from llama_cpp import Llama
        from llama_cpp._ggml import GGMLType
        from llama_cpp.llama_cpp import llama_flash_attn_type

        n_ctx = max(512, int(context_size))
        estimated = len(full_prompt) // 3 + len(images) * 1024 + int(max_tokens) + 512
        if estimated > n_ctx:
            print(f"[PromptForge] context_size={n_ctx} may truncate this request (about {estimated} tokens needed).")

        kv_type = int(getattr(GGMLType, f"GGML_TYPE_{str(kv_cache_type).upper()}", GGMLType.GGML_TYPE_F16))
        flash_types = {
            "auto": int(llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO),
            "on": int(llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED),
            "off": int(llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED),
        }
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
            "flash_attn_type": flash_type,
            "use_mmap": True,
            "verbose": False,
        }
        if images:
            from llama_cpp import llama_chat_format
            model_name_lower = f"{gguf_path.name} {(entry or {}).get('name', '')}".lower()
            handler_name = _gguf_vision_handler(model_name_lower)
            handler_cls = getattr(llama_chat_format, handler_name, None)
            if handler_cls is None:
                raise ValueError(f"[PromptForge] this llama-cpp-python build has no {handler_name}; upgrade llama-cpp-python for this vision architecture.")
            llama_kwargs["chat_handler"] = handler_cls(clip_model_path=str(mmproj), verbose=False)
        else:
            chat_format = _gguf_chat_format(f"{gguf_path.name} {(entry or {}).get('name', '')}")
            if chat_format:
                llama_kwargs["chat_format"] = chat_format
        if auto_fit and llama_kwargs["n_gpu_layers"] == -1:
            _free_vram()
            budget = _gguf_layer_budget(gguf_path, n_ctx, mmproj)
            if budget is not None:
                fitted, total = budget
                if fitted < total:
                    print(f"[PromptForge] auto_fit: {total - fitted}/{total} layers on CPU for a {n_ctx}-token context.")
                llama_kwargs["n_gpu_layers"] = fitted
        elif llama_kwargs["n_gpu_layers"] != 0:
            try:
                free_bytes = torch.cuda.mem_get_info()[0]
            except Exception:
                free_bytes = 0
            if free_bytes < _gguf_memory_estimate(gguf_path, n_ctx, llama_kwargs["n_gpu_layers"], mmproj):
                _free_vram()

        fallbacks = []
        if kv_type != int(GGMLType.GGML_TYPE_F16):
            fallbacks.append(("kv_cache_type", {"type_k": int(GGMLType.GGML_TYPE_F16), "type_v": int(GGMLType.GGML_TYPE_F16)}))
        if flash_type == flash_types["on"]:
            fallbacks.append(("flash_attn", {"flash_attn_type": flash_types["auto"]}))
        if bool(swa_full):
            fallbacks.append(("swa_full", {"swa_full": False}))

        llm = None
        last_error = None
        for index in range(len(fallbacks) + 1):
            try:
                llm = Llama(**llama_kwargs)
                break
            except RuntimeError as exc:
                message = str(exc)
                if "Failed to create llama context" not in message:
                    raise
                last_error = message
                if index >= len(fallbacks):
                    break
                label, overrides = fallbacks[index]
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

        if use_default_sampling:
            temperature, top_p, top_k, repetition_penalty = _apply_sampling_defaults(
                _generation_defaults_from_gguf(llm), temperature, top_p, top_k, repetition_penalty
            )
        content = [{"type": "text", "text": full_prompt}]
        content.extend({"type": "image_url", "image_url": {"url": self._pil_data_url(p)}} for p in images)
        messages = [{"role": "user", "content": content}]
        completion_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            "seed": seed,
        }
        if top_k:
            completion_kwargs["top_k"] = top_k
        if disable_thinking and "reasoning_budget" in inspect.signature(llm.create_chat_completion).parameters:
            completion_kwargs["reasoning_budget"] = 0

        def attempt(seed_value):
            kwargs = dict(completion_kwargs)
            kwargs["seed"] = seed_value
            started = time.time()
            out = llm.create_chat_completion(**kwargs)
            text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            tokens = (out.get("usage") or {}).get("completion_tokens")
            if tokens:
                print(f"[PromptForge] GGUF generated {tokens} tokens in {time.time() - started:.1f}s.")
            return text

        try:
            enhanced = attempt(seed)
            if auto_retry and _is_bad_output(_split_thinking(enhanced)[0]):
                print("[PromptForge] output rejected (empty/repetitive/refusal); retrying with a new seed.")
                enhanced = attempt(_next_seed(seed))
        except ValueError as exc:
            if images and "mmproj" in str(exc).lower():
                expected = (entry or {}).get("mmproj_file") or "mmproj-*.gguf"
                raise ValueError(f"[PromptForge] Vision GGUF could not load mmproj '{mmproj}'. It must match the text model {gguf_path.name}. Download {expected} next to the model or set mmproj_path.") from exc
            raise
        finally:
            self._release(keep_model_loaded=keep_model_loaded, llm=llm)
        return enhanced

    # ------------------------------------------------------------------ #
    # Text encoder engines (native ComfyUI CLIP route / bitsandbytes route)
    # ------------------------------------------------------------------ #
    def _generate_text_encoder(self, full_prompt, mode, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, custom_repo_id, local_model_path, disable_thinking, use_default_sampling, auto_retry, keep_model_loaded):
        if clip is not None:
            enhanced = self._generate_with_clip(
                full_prompt=full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
                clip=clip, disable_thinking=disable_thinking, auto_retry=auto_retry,
            )
            self._release(keep_model_loaded=keep_model_loaded)
            return enhanced

        eff_local = (local_model_path or "").strip() or _local_model_option(model_name)
        src = eff_local or ""
        params_b = None
        resolved = _resolve_model_entry(model_name) if (model_name and model_name != "auto") else _resolve_model_entry("auto")
        if not src and resolved is not None:
            if resolved.get("kind") == "hf":
                src = resolved.get("local_path") or resolved.get("repo_id") or ""
                params_b = resolved.get("params_b")
            elif resolved.get("kind") == "gguf":
                first = HERETIC_MODELS[0] if HERETIC_MODELS else {}
                src = (first or {}).get("repo_id") or ""
                params_b = (first or {}).get("params_b") or 9
        if not src and custom_repo_id:
            src = custom_repo_id
        if not src:
            raise ValueError("[PromptForge] No text encoder selected.")

        quant = {"8bit": "q8", "4bit": "q4"}.get(quantization or "auto", quantization or "auto")
        if quant in ("q5", "q6"):
            print(f"[PromptForge] {quant} is a GGUF quant level; loading safetensors in q8 instead.")
            quant = "q8"
        src_path = Path(src)
        if not src_path.exists():
            from huggingface_hub import snapshot_download
            src = snapshot_download(repo_id=src, local_dir=None)
            src_path = Path(src)

        if use_default_sampling:
            temperature, top_p, top_k, repetition_penalty = _apply_sampling_defaults(
                _generation_defaults_from_files(src), temperature, top_p, top_k, repetition_penalty
            )

        if quant in ("q8", "q4") and not src_path.is_dir():
            print(f"[PromptForge] {quant} quantization needs an HF model directory; loading this single file in fp16 instead.")
            quant = "auto"

        if src_path.is_dir():
            quant_cfg = None
            hf_dtype = torch.float16 if quant == "fp16" else _hf_compute_dtype()
            if quant in ("q8", "q4"):
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=(quant == "q8"),
                    load_in_4bit=(quant == "q4"),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=hf_dtype,
                )
            load_kwargs = {
                "quantization_config": quant_cfg,
                "device_map": "auto",
                **_hf_dtype_kwargs(hf_dtype),
            }
            if torch.cuda.is_available():
                _free_vram()
                max_memory = _hf_max_memory()
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
            model = processor = tokenizer = None
            try:
                if mode != "text_only":
                    model = AutoVision.from_pretrained(src, **load_kwargs)
                else:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs)
                processor = AutoProcessor.from_pretrained(src)
                tokenizer = AutoTokenizer.from_pretrained(src)
                conversation = [{"role": "user", "content": []}]
                refs = self._mode_images(mode, image, image2, image3, video, frame_count)
                if refs and getattr(processor, "image_processor", None) is None:
                    raise ValueError("[PromptForge] The selected safetensors model has no image processor; image/video mode needs a vision-capable model.")
                if refs:
                    conversation[0]["content"].extend([{"type": "image", "image": r} for r in refs if r is not None])
                conversation[0]["content"].append({"type": "text", "text": full_prompt})
                chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=not disable_thinking)
                processed = processor(text=chat, images=[r for r in refs if r is not None] or None, return_tensors="pt")
                model_device = next(model.parameters()).device
                model_inputs = {k: (v.to(model_device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in processed.items()}
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
                if num_beams == 1:
                    kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
                    if params_b is not None and params_b >= 3.5:
                        kwargs["top_k"] = top_k or 20

                from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
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
                    outputs = model.generate(**model_inputs, **kwargs)
                    return tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

                enhanced = attempt(seed)
                if auto_retry and _is_bad_output(_split_thinking(enhanced)[0]):
                    print("[PromptForge] output rejected (empty/repetitive/refusal); retrying with a new seed.")
                    enhanced = attempt(_next_seed(seed))
                return enhanced
            finally:
                del model, processor, tokenizer
                self._release(keep_model_loaded=keep_model_loaded)

        import comfy.sd
        clip_obj = comfy.sd.load_clip(ckpt_paths=[src], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={"dtype": torch.float16})
        enhanced = self._generate_with_clip(
            full_prompt=full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
            clip=clip_obj, disable_thinking=disable_thinking, auto_retry=auto_retry,
        )
        self._release(keep_model_loaded=keep_model_loaded)
        return enhanced

    # ------------------------------------------------------------------ #
    # Native ComfyUI CLIP generation (connected clip or locally loaded clip)
    # ------------------------------------------------------------------ #
    def _generate_with_clip(self, full_prompt, mode, max_tokens, temperature, top_p, top_k, repetition_penalty, seed, image, image2, image3, video, frame_count, clip, disable_thinking, auto_retry):
        global _TOP_P_WARNED
        images = self._clip_images(mode, image, image2, image3, video, frame_count)
        tokenize_kwargs = {"images": images} if images else {}
        if disable_thinking:
            tokenize_kwargs["thinking"] = False
        tokens = clip.tokenize(full_prompt, **tokenize_kwargs)
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
                return run(effective_top_p, seed_value)
            except RuntimeError as exc:
                if effective_top_p < 1.0 and "single memory location" in str(exc):
                    print("[PromptForge] ComfyUI's CLIP sampler hit a torch overlap error on top_p; retrying with top_p=1.0 (temperature, top_k, repetition and seed still apply).")
                    return run(1.0, seed_value)
                raise

        seed_value = seed
        for attempt_index in range(2):
            generated = attempt(seed_value)
            decoded = clip.decode(generated, skip_special_tokens=True)
            if auto_retry and attempt_index == 0 and _is_bad_output(_split_thinking(decoded)[0]):
                print("[PromptForge] output rejected (empty/repetitive/refusal); retrying with a new seed.")
                seed_value = _next_seed(seed_value)
                continue
            return decoded

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
        return [t.detach().float().cpu().clamp(0.0, 1.0) for t in images]

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
        return images

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
        import base64
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPTFORGE_Model": "PromptForge Model",
    "PROMPTFORGE_PromptEnhancer": "PromptForge Enhancer",
    "PROMPTFORGE_ConfigFinder": "PromptForge Config Finder",
}

WEB_DIRECTORY = "./web"
