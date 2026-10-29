# ComfyUI PromptForge

Unified, fully-loaded ComfyUI prompt enhancement node.

## Node

**`🔮 PromptForge / PROMPTFORGE_PromptEnhancer`**

One singular node that does the whole prompt-enhancement job:

- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only`, `image`, `image2`, `video`.
- **Engines**: `text_encoder` (native ComfyUI HF safetensors route via transformers, full precision / 8-bit / 4-bit via bitsandbytes) and `gguf` (llama.cpp GGUF route).
- **Models**: `auto` (recommended heretic HF VL model), heretic HF catalog, heretic GGUF catalog, or `custom_repo_id` for any HF repo.
- **Quantization**: `auto`, `fp16`, `8bit`, `4bit` (GGUF maps fp16/8bit → Q8_0, 4bit → Q4_K_M).
- **Preset**: `auto` or any of the bundled presets (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, generic styles…).
- **Custom system prompt**: override field + save toggle → persisted to user dir.
- **Sampling**: seed, max_tokens, temperature, top_p, top_k, num_beams, repetition_penalty, keep_model_loaded, device.
- **Recommended sets**: `auto`, `fast_4b`, `balanced_9b`, `max_27b`.

## Presets

`presets/promptforge_presets.json` — the crafted prompt templates (from the QwenVL-Mod custom presets plus the generic style set), including the refactored **Krea 2** prompts and the new **MOSS Image** prompts.

## Models catalog

`catalog/models.json` — heretic VL-capable HF models (4B / 9B / 27B, free, image-capable or text-only) and GGUF variants, plus local scan dirs for text models.

## Install

Drop this folder into a ComfyUI `custom_nodes` directory (already placed at `ComfyUI-PromptForge/`). Requires: torch, transformers, bitsandbytes, accelerate, llama-cpp-python (for GGUF engine), huggingface-hub.
