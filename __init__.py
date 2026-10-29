"""ComfyUI PromptForge - unified prompt enhancement node.

Single ComfyUI node that combines:
  - preset catalog (Krea 2, MOSS Image, Qwen Image, generic styles)
  - custom system-prompt override + save mechanism
  - text encoder engines: native ComfyUI CLIP route (fp16 / live-quantized),
    GGUF route via llama.cpp
  - heretic model catalog (HF + GGUF) with local availability markers
  - media-aware modes (text / image / image2 / image3 / video)
  - hardcore sampling options
"""

import json
import gc
from pathlib import Path

import torch
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
    "Then, I will, or I need. Start directly with the required prompt text."
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

HERETIC_MODEL_ENTRIES = [
    {"option": "Qwen3.5-9B Defiant Fable Heretic (HF) [9B]", "kind": "hf", "key": "heretic_qwen35_9b", "params_b": 9, "avail_match": ["Defiant"]},
    {"option": "Qwen3.5-4B Deckard Heretic Thinking (HF) [4B, text-only]", "kind": "hf", "key": "heretic_qwen35_4b_thinking", "params_b": 4, "avail_match": ["Deckard"]},
    {"option": "Qwen3.6-27B Fable Fusion Heretic (HF) [27B]", "kind": "hf", "key": "heretic_qwen36_27b", "params_b": 27, "avail_match": ["27B", "Fable"]},
    {"option": "Qwen3.5-9B Defiant Fable GGUF (llama.cpp) [9B]", "kind": "gguf", "key": "gguf_qwen35_9b_fable", "params_b": 9, "avail_match": ["Defiant"]},
    {"option": "Qwen3.6-27B Fable Fusion GGUF (llama.cpp) [27B]", "kind": "gguf", "key": "gguf_qwen36_27b_fusion", "params_b": 27, "avail_match": ["27B", "Fable"]},
    {"option": "Qwen3.5-4B Claude OS Heretic GGUF (llama.cpp) [4B]", "kind": "gguf", "key": "gguf_qwen35_4b_claude", "params_b": 4, "avail_match": ["Claude"]},
]


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


MODEL_OPTIONS = (
    [_entry_label(e) for e in HERETIC_MODEL_ENTRIES]
    + [e["path"] for e in LOCAL_TEXT_MODELS if e.get("path")]
    + ["auto"]
)

PRESET_OPTIONS = ["auto"] + sorted(
    k for k, v in PROMPTFORGE_PRESETS.items()
    if isinstance(v, dict) and (v.get("text") or v.get("system_prompt"))
)

ENGINE_OPTIONS = ["auto", "text_encoder", "gguf"]
MODE_OPTIONS = ["text_only", "image", "image2", "image3", "video"]
QUANT_OPTIONS = ["auto", "fp16", "q8", "q4"]


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


def _effective_engine(engine, model_name, local_model_path):
    if engine and engine != "auto":
        return engine
    return _model_kind(model_name, local_model_path)


def _user_dir():
    try:
        if folder_paths is not None:
            return Path(getattr(folder_paths, "user_directory", "") or "/home/genortg/ComfyUI/ComfyUI/user")
    except Exception:
        pass
    return Path("/home/genortg/ComfyUI/ComfyUI/user")


class PROMPTFORGE_PromptEnhancer:
    CATEGORY = "🔮 PromptForge"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_PROMPT",)
    FUNCTION = "enhance"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Raw user request / media-derived text to enhance. Can be loose natural language, random idea, detailed description, keyword pile, anything."}),
                "mode": (MODE_OPTIONS, {"default": "text_only", "tooltip": "What the prompt is for: text_only (T2I / text-to-video), image (image-to-image / reference), image2 (second reference / edit), image3 (third reference), video (video frame analysis)."}),
                "engine": (ENGINE_OPTIONS, {"default": "auto", "tooltip": "Model engine: auto (detect from the selected model), text_encoder (native ComfyUI CLIP / safetensors), gguf (llama.cpp)."}),
                "model_name": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0], "tooltip": "Model: researched heretic HF / GGUF models first (marked [AVAILABLE] when downloaded), then every model detected in the ComfyUI model dirs, then auto. custom_repo_id / local_model_path allow models outside this list."}),
                "quantization": (QUANT_OPTIONS, {"default": "auto", "tooltip": "Text encoder precision for safetensors models: auto (fp16), fp16 (native fp16), q8 / q4 (live bitsandbytes quantization before generation). Ignored for GGUF models."}),
                "enhancement_preset": (PRESET_OPTIONS, {"default": "auto", "tooltip": "Preset system prompt: auto, or one of the PromptForge presets (Krea 2 T2I, MOSS Image T2I, Qwen Image T2I, generic styles...)."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2 ** 32 - 1, "tooltip": "Sampling seed for the Qwen-style generation. 1 = deterministic-ish / keep last logic."}),
                "max_tokens": ("INT", {"default": 8192, "min": 32, "max": 16384, "tooltip": "Max new tokens for the enhanced prompt."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "tooltip": "Sampling temperature."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": "Nucleus sampling cutoff."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Top-K (Qwen3.5 heretic recommended: 20). 0 = model default."}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Beam search width. 1 = fast sampling; >1 trades speed for stability."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "tooltip": "Repetition penalty."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Unload the text encoder / LLM after the prompt has been generated. Turn on to keep it resident."}),
                "system_prompt_override": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom system prompt override. Leave empty to use the selected preset. Toggle save_system_prompt to write it to your saved custom templates."}),
                "save_system_prompt": ("BOOLEAN", {"default": False, "tooltip": "When true, the system_prompt_override is saved into the user's PromptForge custom system-prompt store."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First reference image (or first video-frame context)."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image / edit source."}),
                "image3": ("IMAGE", {"tooltip": "Third reference image / edit source."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": "Frames sampled from video input."}),
                "clip": ("CLIP", {"tooltip": "Native ComfyUI text encoder (CLIP) to use for generation. When connected, the node generates from this encoder instead of loading its own model."}),
                "custom_repo_id": ("STRING", {"default": "", "tooltip": "Optional custom HuggingFace repo id for text_encoder engine (e.g. 'foo/bar'). Overrides model_name when set."}),
                "local_model_path": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional local model path (safetensors dir/file or GGUF file) - overrides model selection. Use for full safetensors models or custom local GGUF files."}),
            },
        }

    # ------------------------------------------------------------------ #
    # Node entry point
    # ------------------------------------------------------------------ #
    def enhance(
        self,
        prompt_text,
        mode,
        engine,
        model_name,
        quantization,
        enhancement_preset,
        seed,
        max_tokens,
        temperature,
        top_p,
        top_k,
        num_beams,
        repetition_penalty,
        keep_model_loaded,
        system_prompt_override,
        save_system_prompt,
        image=None,
        image2=None,
        image3=None,
        video=None,
        frame_count=16,
        clip=None,
        custom_repo_id="",
        local_model_path="",
    ):
        if save_system_prompt:
            custom = (system_prompt_override or "").strip()
            if custom:
                self._save_custom_system_prompt(custom)

        sys_text = (system_prompt_override or "").strip()
        if not sys_text:
            key = enhancement_preset if enhancement_preset and enhancement_preset in PROMPTFORGE_PRESETS else DEFAULT_PRESET
            entry = PROMPTFORGE_PRESETS.get(key) or {}
            sys_text = entry.get("text") or entry.get("system_prompt") or (PROMPTFORGE_PRESETS.get(DEFAULT_PRESET) or {}).get("text") or ""

        user_text = (prompt_text or "").strip()
        parts = [p for p in (user_text, sys_text, OUTPUT_GUARD) if p and p.strip()]
        full_prompt = "\n\n".join(parts)

        enhanced = self._generate(
            full_prompt=full_prompt,
            mode=mode,
            engine=_effective_engine(engine, model_name, local_model_path),
            model_name=model_name,
            quantization=quantization,
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
            clip=clip,
            custom_repo_id=custom_repo_id,
            local_model_path=local_model_path,
            keep_model_loaded=keep_model_loaded,
        )
        return (enhanced,)

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
            print(f"[PromptForge] failed to save custom system prompt: {e}")

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
    def _generate(self, full_prompt, mode, engine, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, custom_repo_id, local_model_path, keep_model_loaded):
        if engine == "gguf":
            return self._generate_gguf(
                full_prompt=full_prompt, model_name=model_name, quantization=quantization,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                num_beams=num_beams, repetition_penalty=repetition_penalty,
                custom_repo_id=custom_repo_id, local_model_path=local_model_path,
                keep_model_loaded=keep_model_loaded,
            )
        return self._generate_text_encoder(
            full_prompt=full_prompt, mode=mode, model_name=model_name, quantization=quantization,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            num_beams=num_beams, repetition_penalty=repetition_penalty,
            image=image, image2=image2, image3=image3, video=video, frame_count=frame_count,
            seed=seed, clip=clip, custom_repo_id=custom_repo_id, local_model_path=local_model_path,
            keep_model_loaded=keep_model_loaded,
        )

    # ------------------------------------------------------------------ #
    # GGUF / llama.cpp engine
    # ------------------------------------------------------------------ #
    def _generate_gguf(self, full_prompt, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, custom_repo_id, local_model_path, keep_model_loaded):
        gguf_path = None
        entry = None
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
                base_dir = Path(entry.get("base_dir") or "LLM/GGUF")
                try:
                    if folder_paths is not None:
                        _md = getattr(folder_paths, "models_dir", "") or ""
                        if str(_md) and str(_md) != ".":
                            base_dir = Path(_md) / base_dir
                except Exception:
                    pass
                qf = entry.get("quant_files") or {}
                if qf:
                    quant_key = quantization if quantization in qf else ("8bit" if quantization in (None, "auto", "fp16") else "4bit")
                    gguf_path = base_dir / qf.get(quant_key, next(iter(qf.values())))
        if gguf_path is None:
            raise ValueError("[PromptForge] No GGUF model selected / available.")

        from llama_cpp import Llama
        defaults = (entry or {}).get("defaults") or {}
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=int(defaults.get("context_length") or 65536),
            n_gpu_layers=int(defaults.get("gpu_layers") if defaults.get("gpu_layers") is not None else -1),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        enhanced = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        self._release(keep_model_loaded=keep_model_loaded, llm=llm)
        return enhanced

    # ------------------------------------------------------------------ #
    # Text encoder engines (native ComfyUI CLIP route / bitsandbytes route)
    # ------------------------------------------------------------------ #
    def _generate_text_encoder(self, full_prompt, mode, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, image3, video, frame_count, seed, clip, custom_repo_id, local_model_path, keep_model_loaded):
        if clip is not None:
            enhanced = self._generate_with_clip(
                full_prompt=full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count, clip=clip,
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

        quant = quantization or "auto"
        src_path = Path(src)
        if not src_path.exists():
            from huggingface_hub import snapshot_download
            src = snapshot_download(repo_id=src, local_dir=None)
            src_path = Path(src)

        if src_path.is_dir() or quant in ("q8", "q4"):
            if quant in ("q8", "q4") and not src_path.is_dir():
                raise ValueError("[PromptForge] q8/q4 quantization needs an HF model directory; use auto/fp16 for single-file ComfyUI text encoders.")
            quant_cfg = BitsAndBytesConfig(load_in_8bit=(quant == "q8"), load_in_4bit=(quant == "q4")) if quant in ("q8", "q4") else None
            if mode in ("image", "image2", "image3", "video"):
                model = AutoVision.from_pretrained(src, quantization_config=quant_cfg, torch_dtype=torch.float16, device_map="auto")
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(src, quantization_config=quant_cfg, torch_dtype=torch.float16, device_map="auto")
            processor = AutoProcessor.from_pretrained(src)
            tokenizer = AutoTokenizer.from_pretrained(src)
            conversation = [{"role": "user", "content": []}]
            refs = self._mode_images(mode, image, image2, image3, video, frame_count)
            if refs:
                conversation[0]["content"].extend([{"type": "image", "image": r} for r in refs if r is not None])
            conversation[0]["content"].append({"type": "text", "text": full_prompt})
            chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
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
            outputs = model.generate(**model_inputs, **kwargs)
            input_len = model_inputs["input_ids"].shape[-1]
            enhanced = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
            del model, processor, tokenizer, outputs
            self._release(keep_model_loaded=keep_model_loaded)
            return enhanced

        import comfy.sd
        clip_obj = comfy.sd.load_clip(ckpt_paths=[src], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={"dtype": torch.float16})
        enhanced = self._generate_with_clip(
            full_prompt=full_prompt, mode=mode, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            seed=seed, image=image, image2=image2, image3=image3, video=video, frame_count=frame_count, clip=clip_obj,
        )
        self._release(keep_model_loaded=keep_model_loaded)
        return enhanced

    # ------------------------------------------------------------------ #
    # Native ComfyUI CLIP generation (connected clip or locally loaded clip)
    # ------------------------------------------------------------------ #
    def _generate_with_clip(self, full_prompt, mode, max_tokens, temperature, top_p, top_k, repetition_penalty, seed, image, image2, image3, video, frame_count, clip):
        images = self._mode_images(mode, image, image2, image3, video, frame_count)
        tokens = clip.tokenize(full_prompt, images=images) if images else clip.tokenize(full_prompt)
        generated = clip.generate(
            tokens,
            do_sample=True,
            max_length=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=0.0,
            repetition_penalty=repetition_penalty,
            seed=seed,
            presence_penalty=0.0,
        )
        return clip.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Input adapters
    # ------------------------------------------------------------------ #
    def _mode_images(self, mode, image, image2, image3, video, frame_count):
        if mode == "video":
            return self._frames_from_input(video, frame_count)
        if mode == "image":
            return [self._pil_from_input(image)]
        if mode == "image2":
            return [self._pil_from_input(image2)]
        if mode == "image3":
            return [self._pil_from_input(image3)]
        return []

    def _pil_from_input(self, img):
        from PIL import Image
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        try:
            import numpy as np
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.ndim == 3 and arr.shape[0] > 4:
                arr = arr[::4, ::4, :]
            im = Image.fromarray(arr)
            if im.mode != "RGB":
                im = im.convert("RGB")
            return im
        except Exception:
            return None

    def _frames_from_input(self, video, frame_count):
        from PIL import Image
        frames = []
        if video is None:
            return frames
        if isinstance(video, list) and all(isinstance(f, Image.Image) for f in video):
            frames = list(video)
        elif hasattr(video, "mean"):
            try:
                shape = list(video.shape)
                n = shape[0] if len(shape) > 0 else len(video)
                idx = [int(i * (len(video) - 1) / max(1, frame_count - 1)) for i in range(frame_count)] if hasattr(video, "__len__") else list(range(min(frame_count, 4)))
                for i in idx:
                    frames.append(self._pil_from_input(video[i]))
            except Exception:
                frames = []
        return frames[: max(1, frame_count)]


NODE_CLASS_MAPPINGS = {
    "PROMPTFORGE_PromptEnhancer": PROMPTFORGE_PromptEnhancer,
}
