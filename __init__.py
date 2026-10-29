"""ComfyUI PromptForge - unified, fully-loaded prompt enhancement node.

Single ComfyUI node that combines:
  - a large preset catalog (Krea 2, MOSS Image, Qwen Image, generic styles)
  - a custom system-prompt override field + save mechanism
  - full model-loading options: HF text-encoder engines (safetensors / full precision),
    GGUF engines (llama.cpp), quantization (auto/fp16/8bit/4bit), local model scanning
  - heretic VL model catalog (HF, free, VL-capable, text-only capable for T2I)
  - hardcore sampling options (seed, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty)
  - recommended option sets (auto / fast_4b / balanced_9b / max_27b)
"""

import json
import gc
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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

TEXT_MODEL_OPTIONS = ["auto"] + [m.get("key") for m in HERETIC_MODELS if m.get("key")] + [
    g.get("key") for g in GGUF_MODELS if g.get("key")
]

PRESET_OPTIONS = ["auto"] + sorted(
    k for k, v in PROMPTFORGE_PRESETS.items()
    if isinstance(v, dict) and (v.get("text") or v.get("system_prompt"))
)

RECOMMENDED_SETS = ["auto", "fast_4b", "balanced_9b", "max_27b"]

RECOMMENDED_SET_MAP = {
    "fast_4b": {
        "engine": "text_encoder",
        "model_name": "heretic_qwen35_4b_thinking",
        "quantization": "4bit",
        "enhancement_preset": "Krea 2 T2I",
    },
    "balanced_9b": {
        "engine": "text_encoder",
        "model_name": "heretic_qwen35_9b",
        "quantization": "8bit",
        "enhancement_preset": "Qwen Image T2I",
    },
    "max_27b": {
        "engine": "text_encoder",
        "model_name": "heretic_qwen36_27b",
        "quantization": "8bit",
        "enhancement_preset": "MOSS Image T2I",
    },
}


def _scan_local_text_models():
    """Scan ComfyUI model directories for local text/LLM/GGUF models."""
    found = []
    base = None
    if folder_paths is not None:
        base = Path(getattr(folder_paths, "models_dir", "") or "")
    candidates = []
    if base is not None and str(base) and str(base) != ".":
        for d in LOCAL_SCAN_DIRS:
            candidates.append(base / d)
    candidates.append(Path("/models"))
    candidates.append(Path("/home/genortg/ComfyUI/Comfy/models"))
    candidates.append(Path("/home/genortg/1TB-SAMSUNG/Comfy/Comfy/models"))
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if not c.is_dir():
            continue
        for p in sorted(c.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".gguf", ".bin"):
                found.append(
                    {
                        "path": str(p),
                        "name": p.name,
                        "kind": p.suffix.lower().lstrip("."),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 1) if p.stat().st_size < 10**10 else -1,
                    }
                )
    return found


LOCAL_TEXT_MODELS = _scan_local_text_models()


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
                "mode": (["text_only", "image", "image2", "video"], {"default": "text_only", "tooltip": "What the prompt is for: text_only (T2I / text-to-video), image (image-to-image / reference), image2 (second reference / edit), video (video frame analysis)."}),
                "engine": (["auto", "text_encoder", "gguf"], {"default": "auto", "tooltip": "Model engine: auto (pick recommended), text_encoder (native ComfyUI HF safetensors route), gguf (llama.cpp GGUF route)."}),
                "model_name": (TEXT_MODEL_OPTIONS, {"default": "auto", "tooltip": "Model: auto-detect, heretic HF VL models, or GGUF heretic models. Optional custom_repo_id allows any HF repo id."}),
                "quantization": (["auto", "fp16", "8bit", "4bit"], {"default": "auto", "tooltip": "Quantization for text_encoder models: auto, full fp16, 8-bit, 4-bit. GGUF: fp16->Q8_0, 8bit->Q8_0, 4bit->Q4_K_M."}),
                "enhancement_preset": (PRESET_OPTIONS, {"default": "auto", "tooltip": "Preset system prompt: auto, or one of the PromptForge presets (Krea 2 T2I, MOSS Image T2I, Qwen Image T2I, generic styles...)."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2 ** 32 - 1, "tooltip": "Sampling seed for the Qwen-style generation. 1 = deterministic-ish / keep last logic."}),
                "max_tokens": ("INT", {"default": 8192, "min": 32, "max": 16384, "tooltip": "Max new tokens for the enhanced prompt."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "tooltip": "Sampling temperature."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": "Nucleus sampling cutoff."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Top-K (Qwen3.5 heretic recommended: 20). 0 = model default."}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Beam search width. 1 = fast sampling; >1 trades speed for stability."}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "tooltip": "Repetition penalty."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the model loaded after the run."}),
                "device": (["auto", "cpu", "mps", "cuda"], {"default": "auto", "tooltip": "Compute device."}),
                "system_prompt_override": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom system prompt override. Leave empty to use the selected preset. Toggle save_system_prompt to write it to your saved custom templates."}),
                "save_system_prompt": ("BOOLEAN", {"default": False, "tooltip": "When true, the system_prompt_override is saved into the user's PromptForge custom system-prompt store."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Reference image for image / image2 / video modes. For video modes this is frames; use frame_count."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image / edit source."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": "Frames sampled from video input."}),
                "custom_repo_id": ("STRING", {"default": "", "tooltip": "Optional custom HuggingFace repo id for text_encoder engine (e.g. 'foo/bar'). Overrides model_name when set."}),
                "recommended_set": (RECOMMENDED_SETS, {"default": "auto", "tooltip": "Recommended preset option set: auto, fast_4b, balanced_9b, max_27b. Fills engine/model/quant/preset automatically."}),
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
        device,
        system_prompt_override,
        save_system_prompt,
        image=None,
        image2=None,
        video=None,
        frame_count=16,
        custom_repo_id="",
        recommended_set="auto",
    ):
        if recommended_set and recommended_set != "auto":
            override_map = RECOMMENDED_SET_MAP.get(recommended_set)
            if override_map:
                engine = override_map.get("engine") or engine
                model_name = override_map.get("model_name") or model_name
                quantization = override_map.get("quantization") or quantization
                enhancement_preset = override_map.get("enhancement_preset") or enhancement_preset

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
            engine=engine,
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
            video=video,
            frame_count=frame_count,
            device=device,
            custom_repo_id=custom_repo_id,
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
    # Generation dispatch
    # ------------------------------------------------------------------ #
    def _generate(self, full_prompt, mode, engine, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, video, frame_count, device, custom_repo_id):
        if engine == "gguf":
            return self._generate_gguf(
                full_prompt=full_prompt, model_name=model_name, quantization=quantization,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                num_beams=num_beams, repetition_penalty=repetition_penalty,
                image=image, image2=image2, video=video, frame_count=frame_count,
                device=device, custom_repo_id=custom_repo_id,
            )
        return self._generate_hf(
            full_prompt=full_prompt, mode=mode, model_name=model_name, quantization=quantization,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            num_beams=num_beams, repetition_penalty=repetition_penalty,
            image=image, image2=image2, video=video, frame_count=frame_count,
            device=device, custom_repo_id=custom_repo_id,
        )

    # ------------------------------------------------------------------ #
    # HF / safetensors text-encoder engine
    # ------------------------------------------------------------------ #
    def _generate_hf(self, full_prompt, mode, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, video, frame_count, device, custom_repo_id):
        repo_id = None
        params_b = None
        if model_name and model_name != "auto":
            for m in HERETIC_MODELS:
                if m.get("key") == model_name:
                    repo_id = m.get("repo_id")
                    params_b = m.get("params_b")
                    break
            if not repo_id:
                s = str(model_name)
                if s.startswith("http://") or s.startswith("https://") or s.startswith("hf.co/"):
                    repo_id = s
            if not repo_id and custom_repo_id:
                repo_id = custom_repo_id
        if not repo_id:
            for m in HERETIC_MODELS:
                repo_id = m.get("repo_id")
                params_b = m.get("params_b")
                break

        needs_image = mode in ("image", "image2", "video")
        dtype = torch.float16 if (quantization in (None, "auto", "fp16")) and torch.cuda.is_available() else torch.float32
        quant_cfg = None
        if quantization == "8bit" or quantization == "4bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=(quantization == "8bit"), load_in_4bit=(quantization == "4bit"))

        if needs_image:
            model = AutoVision.from_pretrained(repo_id, quantization_config=quant_cfg, torch_dtype=dtype, device_map="auto")
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(repo_id, quantization_config=quant_cfg, torch_dtype=dtype, device_map="auto")

        processor = AutoProcessor.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)

        conversation = [{"role": "user", "content": []}]
        images = []
        if mode in ("image", "video"):
            pil = self._pil_from_input(image)
            if pil is not None:
                conversation[0]["content"].append({"type": "image", "image": pil})
                images.append(pil)
        elif mode == "image2":
            pil = self._pil_from_input(image2)
            if pil is not None:
                conversation[0]["content"].append({"type": "image", "image": pil})
                images.append(pil)

        num_images = len(images)
        if video is not None and mode == "video":
            frames = self._frames_from_input(video, frame_count)
            for fr in frames:
                conversation[0]["content"].append({"type": "image", "image": fr})
                images.append(fr)

        conversation[0]["content"].append({"type": "text", "text": full_prompt})
        chat = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        processed = processor(text=chat, images=images or None, return_tensors="pt")
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
        text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------ #
    # GGUF / llama.cpp engine
    # ------------------------------------------------------------------ #
    def _generate_gguf(self, full_prompt, model_name, quantization, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, image, image2, video, frame_count, device, custom_repo_id):
        entry = None
        for g in GGUF_MODELS:
            if g.get("key") == model_name:
                entry = g
                break
        if entry is None:
            entry = GGUF_MODELS[0] if GGUF_MODELS else None
        if entry is None:
            raise ValueError("[PromptForge] No GGUF model selected / available.")

        base_dir = Path(entry.get("base_dir") or "LLM/GGUF")
        try:
            import folder_paths as _fp
            _md = getattr(_fp, "models_dir", "") or ""
            base_dir = (Path(_md) if str(_md) and str(_md) != "." else Path("/home/genortg/ComfyUI/Comfy/models")) / base_dir
        except Exception:
            pass
        qf = entry.get("quant_files") or {}
        quant_key = quantization if quantization in qf else ("8bit" if quantization in (None, "auto", "fp16") else "4bit")
        gguf_file = qf.get(quant_key) or next(iter(qf.values()))
        gguf_path = base_dir / gguf_file

        from llama_cpp import Llama
        defaults = entry.get("defaults") or {}
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
        return out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------ #
    # Input adapters
    # ------------------------------------------------------------------ #
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
            # tensor-like: sample along first dim
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
