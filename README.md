# ComfyUI PromptForge

Unified, fully-loaded ComfyUI prompt enhancement node.

## Node

**`🔮 PromptForge / PROMPTFORGE_PromptEnhancer`**

One singular node that does the whole prompt-enhancement job:

- **Request fidelity**: every preset treats the user's text as the authority on subject, acts, intensity and tone. Hardcore, explicit or intense requests are written directly and specifically, without sanitizing, softening, euphemizing, fading to black, or adding warnings or moral notes.
- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only` (ignore media) or `inputs` (use every connected image from `image` / `image2` / `image3` plus frames sampled from `video` via `frame_count`). The prompt is told exactly which inputs arrived: bypassed or missing sockets are never mentioned, and the model is told not to invent extra images, references, video or frames.
- **Video duration**: `video_seconds` sets the exact clip length. MiniMax H3 presets use it for shot budget, timestamps, and the frame-alignment line (`{video_seconds}`).
- **Source**: explicit choice — `safetensors` (HF dir / single-file text encoder, loaded by the node), `GGUF` (llama.cpp), or `connected clip` (uses the CLIP input; no model loaded). All fields stay visible; only the ones for the selected source are used.
- **Models**: separate pickers per source — `safetensors_model` lists HF / safetensors models only, `gguf_model` lists GGUF models only (both marked `[AVAILABLE]` when downloaded). The catalog covers Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Qwen2.5, Qwen3-VL, Qwen2.5-VL, Gemma 3/4, Llama 3, Mistral/Ministral and Phi-4 heretic/abliterated models; GGUF entries offer q4/q5/q6/q8 and download automatically from Hugging Face on first use. `custom_repo_id` / `local_model_path` / `mmproj_path` cover anything outside the lists.
- **Vision safety**: image/video mode never silently falls back to text-only. A connected text encoder without vision support, a text-only safetensors model, or a GGUF without a matching mmproj raises a clear error. Vision GGUFs load the family-specific llama.cpp chat handler (Gemma 3/4, Qwen 2.5/3/3.5) and prefer the mmproj stored next to the model.
- **Quantization**: `auto`, `fp16`, `q8`, `q4` (live bitsandbytes quantization before generation; q4 uses NF4 with double quantization; `auto`/`q8`/`q4` use bf16 on supported GPUs, which avoids the fp16 logit overflow that breaks Gemma sampling; ignored for GGUF).
- **GGUF controls**: `context_size`, `gpu_layers`, `batch_size`, `ubatch_size`, `kv_cache_type` (`f16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`), `offload_kqv`, `swa_full`, `flash_attn` and `auto_fit`. Values are used as set; an incompatible KV/flash/SWA choice is retried with the safe value and a status line instead of failing.
- **Quant fallback**: an unsupported quant is not an error — `q5`/`q6` on safetensors load as `q8`, `q8`/`q4` on a single-file encoder load in fp16, and a missing GGUF quant falls back through the available files, each with a warning.
- **Preset**: `auto` or any of the bundled presets (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, generic styles…).
- **Custom system prompt**: override field + `use_system_prompt_override` on/off toggle (text stays populated between runs) + save toggle → persisted to user dir.
- **Sampling**: seed, max_tokens (up to 65536, not capped by the node), temperature, top_p, top_k, num_beams, repetition_penalty, keep_model_loaded. Floats use 0.01 steps for fine control. If ComfyUI's CLIP sampler throws a torch overlap error on `top_p`, the node retries that generation once with `top_p=1.0` (other sampler settings still apply).
- **Model sampling defaults**: `use_default_sampling` reads temperature / top_p / top_k / repetition_penalty from `generation_config.json` (HF dirs and single-file encoders) or GGUF metadata when the model provides them, falling back to the node values otherwise.
- **Auto-retry**: `auto_retry` (default on) rejects empty, degenerate/repetitive, or refusal output and retries once with the next seed. Only a status line is printed; no prompt content.
- **Repetition guard**: CLIP generation is capped at 2048 tokens and checked after each run; HF generation uses a stopping criterion during sampling. Detected repetition retries once with a new seed.
- **Negative prompt**: `use_negative_prompt` + `negative_prompt` (both at the end) feed your exclusions into the generation request so the model writes around them instead of naming them. Works for keywords and natural language (for example “blurry, extra fingers, sweat, water droplets”).
- **No embellishment**: every preset forbids adding unrequested moisture (sweat, wetness, droplets, sheen) and forbids inventing subjects, actions or scene changes beyond the user's text.
- **No negatives in the positive prompt**: none of these families benefit from negative words in the positive prompt — Krea 2, Qwen Image and MiniMax H3 run at distilled CFG where negatives are weak or ignored, and MOSS Image / Anima have a separate negative prompt. Presets require positive-only wording; the node retries if a negative-prompt phrase appears.
- **Chat templates**: GGUF uses family-correct formats (Gemma, Llama-3, ChatML/Qwen, Mistral) and output cleanup strips ChatML/Gemma/Llama role markers (including malformed markers and system/user turns) across all engines.
- **Thinking**: `disable_thinking` (default on) turns reasoning off where the model template supports it. Reasoning is always stripped from `ENHANCED_PROMPT` and returned on the second output, `THINKING`, for debugging.

## Settings node

**`🔮 PromptForge / PROMPTFORGE_Settings`**

Outputs a `SETTINGS` bundle for the enhancer. Connect it to the enhancer's optional
`settings` input; the enhancer's `use_settings` toggle (default on) decides whether
the bundle overrides its own fields.

- **Model selection**: `source` (`safetensors` / `GGUF` / `connected clip`),
  `safetensors_model`, `gguf_model`, `quantization`.
- **Auto profile** (default on): reads the GPU and picks the best local GGUF quant
  that fits, a 16k-first context, KV precision (`f16` → `q8_0` → `q4_0`), layer
  split, batch sizes, SWA and flash-attention settings. For safetensors it picks
  bf16/fp16, q8 or q4 from the model size and VRAM.
- **Manual**: turn `auto_profile` off and set `context_size`, `gpu_layers`,
  `batch_size`, `ubatch_size`, `kv_cache_type`, `offload_kqv`, `swa_full`,
  `flash_attn`, `auto_fit` and the sampling values yourself.
- **Profiles**: `profile_name` + `load_profile` / `save_profile` store the effective
  bundle in `user/promptforge_profiles.json`. Loading a profile overrides both auto
  and the visible fields.

## Outputs

- `ENHANCED_PROMPT` — the final prompt text only.
- `THINKING` — reasoning that was removed from the output, empty when the model produced none.

## Workflow values and schema stability

ComfyUI execution prompts are keyed by **input name**, so queued/API workflows are safe. Saved UI workflows historically stored widget values as a positional array (`widgets_values`); changing input order could shift values between fields. This pack keeps input order append-only. For full protection, enable **Settings → Workflow → Named values restore** (`Comfy.Workflow.NamedValuesRestore`), which makes ComfyUI restore values by name (`widgets_values_named`) instead of position.

## Privacy

The node never prints or stores prompts, responses, enhanced prompts, thinking, or token text. Only short status lines are printed (model load, token counts, auto settings, profile save/load); llama.cpp verbose output is disabled. The optional `save_system_prompt` toggle is the only persistence path for prompt text and stores just the system-prompt you explicitly choose to save.

## Memory
After the enhanced prompt has been generated the text encoder / LLM is unloaded
(`comfy.model_management` free/cleanup helpers) so it does not hog VRAM/RAM for
other tasks. Before a safetensors model is loaded those helpers also unload
ComfyUI's currently loaded models, and `device_map` gets an explicit VRAM budget
that keeps headroom for activations while offloading overflow to CPU instead of
failing with an OOM. GGUF models unload ComfyUI's models (GPU and CPU) only
when the model plus its KV cache does not fit in free VRAM. `context_size`,
`gpu_layers`, `batch_size`, `ubatch_size`, `kv_cache_type`, `offload_kqv`,
`swa_full` and `flash_attn` are used exactly as set; `auto_fit` computes a
GPU/CPU layer split from free VRAM when enabled. Set `keep_model_loaded` to true
to keep ComfyUI's models
resident. A native ComfyUI `CLIP` socket (`clip`) can be connected to
generate from an already-loaded encoder instead of loading its own model.

## Presets

`presets/promptforge_presets.json` — the crafted prompt templates (from the QwenVL-Mod custom presets plus the generic style set), including the refactored **Krea 2** prompts and the new **MOSS Image** prompts.

## Models catalog

`catalog/models.json` — heretic VL-capable HF models (4B / 9B / 27B, free, image-capable or text-only) and GGUF variants, plus local scan dirs for text models.

## Install

Drop this folder into a ComfyUI `custom_nodes` directory (already placed at `ComfyUI-PromptForge/`). Requires: torch, transformers, bitsandbytes, accelerate, llama-cpp-python (for GGUF engine), huggingface-hub.
