# ComfyUI PromptForge

Unified, fully-loaded ComfyUI prompt enhancement node.

## Node

**`🔮 PromptForge / PROMPTFORGE_PromptEnhancer`**

One singular node that does the whole prompt-enhancement job:

- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only`, `image` (uses every connected reference from `image` / `image2` / `image3`), `video` (frames sampled from `video` via `frame_count`).
- **Video duration**: `video_seconds` sets the exact clip length. MiniMax H3 presets use it for shot budget, timestamps, and the frame-alignment line (`{video_seconds}`).
- **Source**: explicit choice — `safetensors` (HF dir / single-file text encoder, loaded by the node), `GGUF` (llama.cpp), or `connected clip` (uses the CLIP input; no model loaded). The UI hides model fields that do not apply to the selected source.
- **Models**: separate pickers per source — `safetensors_model` lists HF / safetensors models only, `gguf_model` lists GGUF models only (both marked `[AVAILABLE]` when downloaded). `custom_repo_id` / `local_model_path` / `mmproj_path` cover anything outside the lists.
- **Vision safety**: image/video mode never silently falls back to text-only. A connected text encoder without vision support, a text-only safetensors model, or a GGUF without a matching mmproj raises a clear error.
- **Quantization**: `auto`, `fp16`, `q8`, `q4` (live bitsandbytes quantization before generation; ignored for GGUF).
- **Preset**: `auto` or any of the bundled presets (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, generic styles…).
- **Custom system prompt**: override field + `use_system_prompt_override` on/off toggle (text stays populated between runs) + save toggle → persisted to user dir.
- **Sampling**: seed, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, keep_model_loaded. Floats use 0.01 steps for fine control. If ComfyUI's CLIP sampler throws a torch overlap error on `top_p`, the node retries that generation once with `top_p=1.0` (other sampler settings still apply).
- **Model sampling defaults**: `use_default_sampling` reads temperature / top_p / top_k / repetition_penalty from `generation_config.json` (HF dirs and single-file encoders) or GGUF metadata when the model provides them, falling back to the node values otherwise.
- **Thinking**: `disable_thinking` (default on) turns reasoning off where the model template supports it. Reasoning is always stripped from `ENHANCED_PROMPT` and returned on the second output, `THINKING`, for debugging.

## Outputs

- `ENHANCED_PROMPT` — the final prompt text only.
- `THINKING` — reasoning that was removed from the output, empty when the model produced none.

## Privacy

The node never prints or stores prompts, responses, enhanced prompts, thinking, or token text. The only runtime console line is `[PromptForge] request received.`; llama.cpp verbose output is disabled. The optional `save_system_prompt` toggle is the only persistence path and stores just the system-prompt text you explicitly choose to save.

## Memory
After the enhanced prompt has been generated the text encoder / LLM is unloaded
(`comfy.model_management` free/cleanup helpers) so it does not hog VRAM/RAM for
other tasks. Set `keep_model_loaded` to true to keep it resident. A native ComfyUI
`CLIP` socket (`clip`) can be connected to generate from an already-loaded encoder
instead of loading its own model.

## Presets

`presets/promptforge_presets.json` — the crafted prompt templates (from the QwenVL-Mod custom presets plus the generic style set), including the refactored **Krea 2** prompts and the new **MOSS Image** prompts.

## Models catalog

`catalog/models.json` — heretic VL-capable HF models (4B / 9B / 27B, free, image-capable or text-only) and GGUF variants, plus local scan dirs for text models.

## Install

Drop this folder into a ComfyUI `custom_nodes` directory (already placed at `ComfyUI-PromptForge/`). Requires: torch, transformers, bitsandbytes, accelerate, llama-cpp-python (for GGUF engine), huggingface-hub.
