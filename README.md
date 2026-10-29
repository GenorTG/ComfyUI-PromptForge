# ComfyUI PromptForge

Unified, fully-loaded ComfyUI prompt enhancement node.

## Node

**`🔮 PromptForge / PROMPTFORGE_PromptEnhancer`**

One singular node that does the whole prompt-enhancement job:

- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only`, `image`, `image2`, `image3`, `video`.
- **Engines**: `auto` (detect from the selected model), `text_encoder` (native ComfyUI CLIP / safetensors route), `gguf` (llama.cpp GGUF route).
- **Models**: one picker — researched heretic HF VL models (4B / 9B / 27B) and GGUF variants first (marked `[AVAILABLE]` when downloaded), then every model detected in the ComfyUI model dirs, then `auto`; `custom_repo_id` / `local_model_path` allow anything else.
- **Quantization**: `auto`, `fp16`, `q8`, `q4` (live bitsandbytes quantization before generation; ignored for GGUF).
- **Preset**: `auto` or any of the bundled presets (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, generic styles…).
- **Custom system prompt**: override field + save toggle → persisted to user dir.
- **Sampling**: seed, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, keep_model_loaded.

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
