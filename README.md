# ComfyUI PromptForge

Prompt enhancement split into three focused nodes: model/config, prompt generation, and a standalone config finder.

## Nodes

### PromptForge Model (`PROMPTFORGE_Model`)

Owns model choice and runtime configuration; outputs a `MODEL` bundle and a `REPORT` string.

- **Source**: `safetensors`, `GGUF`, or `connected clip`.
- **Models**: separate `safetensors_model` / `gguf_model` pickers. The catalog covers Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Qwen2.5, Qwen3-VL, Qwen2.5-VL, Gemma 3/4, Llama 3, Mistral/Ministral and Phi-4 heretic/abliterated models; GGUF entries offer q4/q5/q6/q8 and download automatically on first use. `custom_repo_id`, `local_model_path` and `mmproj_path` cover anything outside the lists.
- **Auto config** (default on): reads total VRAM and the selected model, then picks the best **local** GGUF quant that fits, the largest context (16k first), KV precision (`f16` → `q8_0` → `q4_0`), GPU layers, batch sizes, windowed SWA and flash attention. For safetensors it picks bf16/fp16, q8 or q4 from the model size and VRAM.
- **Manual**: turn `auto_config` off and set `context_size`, `gpu_layers`, `batch_size`, `ubatch_size`, `kv_cache_type`, `offload_kqv`, `swa_full`, `flash_attn` and `auto_fit` yourself.
- **Profiles**: `profile_name` + `load_profile` / `save_profile` store the effective bundle in `user/promptforge_profiles.json`.
- **Quant fallback**: unsupported combinations warn and use a working option — q5/q6 on safetensors load as q8, q8/q4 on a single-file encoder load in fp16, a missing GGUF quant falls through the available files, and an incompatible KV/flash/SWA choice is retried with the safe value.

### PromptForge Enhancer (`PROMPTFORGE_PromptEnhancer`)

Takes the `MODEL` bundle and writes the prompt. Only generation inputs live here.

- **Request fidelity**: every preset treats the user's text as the authority on subject, acts, intensity and tone. Hardcore, explicit or intense requests are written directly and specifically, without sanitizing, softening, euphemizing, fading to black, or adding warnings or moral notes.
- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only` (ignore media) or `inputs` (use every connected image from `image` / `image2` / `image3` plus frames sampled from `video` via `frame_count`). The prompt is told exactly which inputs arrived: bypassed or missing sockets are never mentioned, and the model is told not to invent extra images, references, video or frames.
- **Preset**: `auto` or any bundled preset (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, MiniMax H3, Anima, ACE-Step, generic styles…).
- **Video duration**: `video_seconds` sets the exact clip length; MiniMax H3 presets use it for shot budget, timestamps and the frame alignment line.
- **Sampling**: seed, max_tokens (up to 65536, not capped by the node), temperature, top_p, top_k, num_beams, repetition_penalty, keep_model_loaded. `use_default_sampling` reads values from `generation_config.json` / GGUF metadata when available.
- **Auto-retry**: rejects empty, degenerate/repetitive or refusal output and retries once with the next seed.
- **Repetition guard**: CLIP generation is capped at 2048 tokens and checked after each run; HF generation uses a stopping criterion during sampling.
- **Negative prompt**: `use_negative_prompt` + `negative_prompt` feed exclusions into the generation request so the model writes around them instead of naming them.
- **No embellishment**: every preset forbids unrequested moisture (sweat, wetness, droplets, sheen) and forbids inventing subjects, actions or scene changes.
- **No negatives in the positive prompt**: Krea 2, Qwen Image and MiniMax H3 run at distilled CFG where negatives are weak or ignored; MOSS Image / Anima have a separate negative prompt. Presets require positive-only wording; the node retries if a negative-prompt phrase appears.
- **Chat templates**: GGUF uses family-correct formats (Gemma, Llama-3, ChatML/Qwen, Mistral) and output cleanup strips role markers across all engines.
- **Thinking**: `disable_thinking` (default on) turns reasoning off where the template supports it; reasoning is always stripped from `ENHANCED_PROMPT` and returned on `THINKING`.
- **Vision safety**: image/video mode never silently falls back to text-only. Vision GGUFs load the family-specific llama.cpp chat handler (Gemma 3/4, Qwen 2.5/3/3.5) and prefer the mmproj stored next to the model.
- **Custom system prompt**: override field + `use_system_prompt_override` toggle + optional save to the user store.

**Outputs**: `ENHANCED_PROMPT` (final prompt only) and `THINKING` (removed reasoning, empty when none).

### PromptForge Config Finder (`PROMPTFORGE_ConfigFinder`)

Standalone output node; it never generates a prompt.

- Scans every locally available GGUF/safetensors model (or one selected model), computes the recommended settings for this GPU, prints a compact report, and writes `user/promptforge_autoconfig.json`.
- `profile_prefix` also saves every recommendation as a profile named `<prefix><model>`.
- `OUTPUT_NODE`-only: queue it on its own (or use its button) to inspect configs without running the image workflow.

## One-click config

The front-end extension adds a button to both nodes:

- **Find best config** on `PromptForge Model` calls `/promptforge/recommend`, fills `context_size`, `gpu_layers`, `batch_size`, `ubatch_size`, `kv_cache_type`, `offload_kqv`, `swa_full`, `flash_attn` and `quantization`, and turns `auto_config` off so you see exactly what will run.
- **Scan local models** on `PromptForge Config Finder` calls `/promptforge/scan` and shows the report without queueing the rest of the workflow.

## Profiles

`profile_name` + `load_profile` / `save_profile` persist a bundle in `user/promptforge_profiles.json`. A loaded profile overrides auto and the visible fields; save stores the effective settings. The Config Finder can mass-create profiles with `profile_prefix`.

## Workflow values and schema stability

ComfyUI execution prompts are keyed by **input name**, so queued/API workflows are safe. This refactor splits the old all-in-one node into `PromptForge Model` + `PromptForge Enhancer`; saved PromptForge nodes need to be rebuilt once — connect a `PromptForge Model` node to the enhancer's `model` input, and replace the old `PROMPTFORGE_Settings` node. Enable **Settings → Workflow → Named values restore** (`Comfy.Workflow.NamedValuesRestore`) so widget values are restored by name while you migrate.

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
