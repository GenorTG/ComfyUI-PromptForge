# ComfyUI PromptForge

Prompt enhancement split into three focused nodes: model/config, prompt generation, and a standalone config finder.

## Nodes

### PromptForge Model (`PROMPTFORGE_Model`)

Owns model choice and runtime configuration; outputs a `MODEL` bundle and a `REPORT` string.

- **Model picker**: one list split into two sections — **local models (on disk, ready)** first, then **not downloaded (fetched on first use)**. Every entry is tagged: `[LOCAL Q4/Q8]` names the GGUF quant files present, `[LOCAL]` marks a cached or scanned model, `[DOWNLOAD]` marks a catalog model that is not on this machine yet (the Model node prints that warning before loading). The console prints a one-line count of local vs. downloadable models at startup, and the Config Finder report ends with the not-downloaded count. If a requested quant is not local but another quant is, the node uses the local one and says so instead of downloading. The catalog covers Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Qwen2.5, Qwen3-VL, Qwen2.5-VL, Gemma 3/4, Llama 3, Mistral/Ministral and Phi-4 heretic/abliterated models. Local models found in the registered `text_encoders` folders (including `extra_model_paths.yaml`) and the `LLM`, `llm`, `gguf` … subfolders of `models/` are listed by their path relative to `models/`, so a workflow saved on one machine resolves on another with the same layout. Single-file encoders without token embeddings (CLIP-L/G, T5, LTX projections) are hidden because they cannot generate text. `local_model_path` and `mmproj_path` cover anything outside the lists. The `clip` input lives here and is passed to the enhancer in the model bundle. The default is the first local GGUF when llama-cpp-python is installed.
- **Automatic runtime**: the node reads total VRAM and the selected model, then picks the best **local** GGUF quant that fits, the largest context (16k first), KV precision, GPU layers, batch sizes, windowed SWA and flash attention. For safetensors it picks bf16/fp16, q8 or q4; local HF directories get their size from the safetensors files. At run time the free VRAM is checked again: bf16/q8 drops to q4 when it does not fit, and a GGUF that fails to load retries with as many GPU layers as actually fit (the rest run on CPU). The only remaining override is `context_size` (0 = auto).
- **Quant fallback**: unsupported combinations warn and use a working option — q5/q6 on safetensors load as q8, q8/q4 on a single-file encoder load in fp16, q8 switches to q4 when free VRAM is insufficient, CPU-offloaded layers are allowed (slower) instead of raising, a missing GGUF quant falls through the available files, and an incompatible KV/flash/SWA choice is retried with the safe value. The report re-checks local availability at run time, so stale `[AVAILABLE]` widget labels still show `[LOCAL]`.

### PromptForge Enhancer (`PROMPTFORGE_PromptEnhancer`)

Takes the `MODEL` bundle and writes the prompt. Only generation inputs live here.

- **Request fidelity**: every preset treats the user's text as the authority on subject, acts, intensity and tone. Hardcore, explicit or intense requests are written directly and specifically, without sanitizing, softening, euphemizing, fading to black, or adding warnings or moral notes.
- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model.
- **Modes**: `text_only` (ignore media) or `inputs` (use every connected image from `image` / `image2` / `image3` plus frames sampled from `video` via `frame_count`). The prompt is told exactly which inputs arrived: bypassed or missing sockets are never mentioned, and the model is told not to invent extra images, references, video or frames.
- **Preset**: `auto` or any bundled preset (Krea 2 T2I, Krea 2 Identity Edit, MOSS Image T2I/I2I, Qwen Image T2I, MiniMax H3, Anima, ACE-Step, generic styles…).
- **MiniMax inputs**: the MiniMax H3 presets (T2VA, I2VA, L2VA, FL2VA, Ref2VA, plus the NSFW variants) get an authoritative list of what is connected. Labels follow the H3 encoder: `<Picture 1>`, `<Picture 2>`, `<Picture 3>` for the connected images in order with gaps closed up (only `image2` connected makes it `<Picture 1>`), and `<Video 1>` for the video. Each mode uses only what it can: T2VA none, I2VA/L2VA one image as first/last frame, FL2VA two images as first and last frame, Ref2VA all images plus the video; anything else that is connected is ignored and the console says so. The frame-alignment line is written by the node for exactly the attached pictures (omitted when none), the model is told which labels do not exist, and any sentence that still cites an unattached picture, video or audio label is removed. The node attaches no audio, so `<Audio N>` only appears when your own text describes it.
- **Video duration**: `video_seconds` sets the exact clip length; MiniMax H3 presets use it for shot budget, timestamps and the frame alignment line.
- **Sampling**: seed, max_tokens (default 2048, up to 65536, not otherwise capped), temperature, top_p, top_k, num_beams, repetition_penalty.
- **Auto-retry** (`auto_retry` toggle on the Prompt Enhancer, on by default; off keeps the first output): rejects empty, too-short, degenerate/repetitive, leaked-reasoning ("Okay, the user wants…") or refusal output and retries up to two more times with new seeds; if every try is a loop, the longest one is returned with repeated sentences removed. Status lines name the reason (too short, repetitive, reasoning leak, refusal, negative wording). Outputs under 400 chars (40 for tag/refine/technical presets, 120 when max_tokens is below 192) count as too short.
- **Repetition guard**: repeated sentences and repeated 5–10 word phrases are rejected across the whole output. HF generation adds `no_repeat_ngram_size=4`; GGUF uses a mild frequency penalty and a longer repeat window. CLIP generation is capped at 2048 tokens and checked afterwards.
- **Negative prompt**: a non-empty `negative_prompt` feeds exclusions into the generation request so the model writes around them instead of naming them.
- **No embellishment**: every preset forbids unrequested moisture (sweat, wetness, droplets, sheen) and forbids inventing subjects, actions or scene changes.
- **No negatives in the positive prompt**: Krea 2, Qwen Image and MiniMax H3 run at distilled CFG where negatives are weak or ignored; MOSS Image / Anima have a separate negative prompt. Presets require positive-only wording; the node retries if classic negative-prompt vocabulary appears ("no blurry", "without watermark", "avoid extra fingers"). Ordinary wording such as "no people" is accepted.
- **Chat templates**: GGUF text generation renders the model's own `tokenizer.chat_template` with `enable_thinking=false` (Qwen3/3.5 empty think block, Gemma 4 primed thought channel) and runs a plain completion, so thinking models answer directly. Models without a template fall back to a family format (Gemma, Llama-3, ChatML/Qwen, Mistral) with the llama.cpp reasoning budget set to zero. Prompts are sent as a real system turn plus a user turn. The connected-CLIP route passes the preset as the encoder's `system_prompt` (same contract as ComfyUI's Generate Text node) and wraps the turns in ChatML itself for encoders whose tokenizer has no chat template. Output cleanup strips role markers across all engines.
- **Output purity**: the system prompt forbids first-person commentary and self-assessment, and the post-clean removes instruction references, reasoning, closings, “Here is the prompt” lead-ins, and sign-offs such as “I followed the instructions / I did a good job”. Only the prompt text leaves the node.
- **Thinking**: reasoning is disabled where the template supports it and always stripped from `ENHANCED_PROMPT`; it is returned on `THINKING`.
- **Vision speed**: reference images and video frames are scaled to about 1 megapixel before the vision model sees them (about 1k tokens each instead of thousands for a 4K image), and ComfyUI models are only unloaded as far as the GGUF plus image encoder actually needs.
- **Vision safety**: image/video mode never silently falls back to text-only. Vision GGUFs load the family-specific llama.cpp chat handler (Gemma 3/4, Qwen 2.5/3/3.5) and prefer the mmproj stored next to the model. Gemma 4 12B Heretic can emit only a `thought` channel for long multimodal prompts; the node detects that and reports it instead of returning an empty prompt — use Qwen3-VL for vision with that model.
- **Custom system prompt**: a non-empty override field replaces the selected preset.

**Outputs**: `ENHANCED_PROMPT` (final prompt only) and `THINKING` (removed reasoning, empty when none).

### PromptForge Config Finder (`PROMPTFORGE_ConfigFinder`)

Standalone output node; it never generates a prompt.

- Scans every locally available GGUF/safetensors model (or one selected model) and prints the recommended settings for this GPU.
- `OUTPUT_NODE`-only: queue it on its own (or use its button) to inspect configs without running the image workflow.

### PromptForge Frame Interpolate (`PROMPTFORGE_FrameInterpolate`)

RIFE frame interpolation for large frames (for example the 2x-upscaled MiniMax output) that backs off by itself instead of crashing ComfyUI. It reuses the network of the `comfyui-frame-interpolation` pack, which must be installed.

- **`ckpt_name`**: every RIFE checkpoint found in `models/frame_interpolation`, `models/rife` or the pack's `ckpts/rife` (4.7/4.9, 4.17, 4.26; `.pth` or `.safetensors`), plus `blend`, a plain cross-fade that needs no model and is instant (ghosts on motion).
- **`mode`**: `balanced` (full-resolution flow), `fast` (flow at half resolution), `best` (ensemble, 4.7/4.9/4.17 only). **`output_precision`**: float16 halves the RAM of the result.
- **Memory**: frames stay on the CPU and the output is one preallocated tensor. Before running, the node checks free RAM; if float32 does not fit it switches to float16, and if that does not fit either it raises a clear error instead of letting the OS kill ComfyUI.
- **VRAM**: one frame pair at a time in fp16. On an out-of-memory error it unloads ComfyUI's models and retries, then estimates flow at 1/2 and 1/4 resolution (the result is still warped at full resolution), and finally runs on the CPU.
- About 25–30 frame pairs per second at 1920x1088 on an RTX 5080 (rife49).

## One-click config

The front-end extension adds a button to both nodes:

- **Test load & apply** on `PromptForge Model` calls `/promptforge/test`: it actually loads the selected model (llama.cpp context, or the transformers model) with the planned settings, falls back through smaller contexts/KV types on failure, then fills `context_size` and `quantization`. A failed test reports the real error instead of guessing.
- **Scan local models** on `PromptForge Config Finder` calls `/promptforge/scan` and shows the report without queueing the rest of the workflow.

## Workflow values and schema stability

ComfyUI execution prompts are keyed by **input name**, so queued/API workflows are safe. This refactor splits the old all-in-one node into `PromptForge Model` + `PromptForge Enhancer`; saved PromptForge nodes need to be rebuilt once — connect a `PromptForge Model` node to the enhancer's `model` input, and replace the old `PROMPTFORGE_Settings` node. Enable **Settings → Workflow → Named values restore** (`Comfy.Workflow.NamedValuesRestore`) so widget values are restored by name while you migrate.

## Privacy

The node never prints or stores prompts, responses, enhanced prompts, thinking, or token text. Only short status lines are printed (model load, token counts); llama.cpp verbose output is disabled.

## Memory
After generation the llama.cpp context or transformers model is closed and its
memory released. Before a safetensors model is loaded ComfyUI's currently loaded
models are unloaded and `device_map` gets a VRAM budget with headroom for
activations, offloading overflow to CPU instead of failing with an OOM. GGUF
unloads ComfyUI's models only when the model plus KV cache does not fit. The
`clip` input on the Model node uses already-loaded encoders instead of loading
its own model.

## Presets

`presets/promptforge_presets.json` — the crafted prompt templates (from the QwenVL-Mod custom presets plus the generic style set), including the refactored **Krea 2** prompts and the new **MOSS Image** prompts.

Every preset has an `NSFW › …` variant with explicit adult writing instructions for nudity, anatomy, positions and choreography; no safety boilerplate is added. Image presets also carry an image-only guard, and the node strips any leaked audio/soundscape/music/shot fields from image outputs.

## Models catalog

`catalog/models.json` — heretic VL-capable HF models (4B / 9B / 27B, free, image-capable or text-only) and GGUF variants, plus local scan dirs for text models.

## Install

Drop this folder into a ComfyUI `custom_nodes` directory. Install the requirements into the Python environment that runs ComfyUI (not the system Python): torch, transformers, bitsandbytes, accelerate, llama-cpp-python with CUDA (for the GGUF engine), huggingface-hub, psutil. Text-only models that have no `AutoModelForCausalLM` mapping (Qwen3-VL, Qwen2.5-VL) load through the vision class, so text mode works for every catalog family.

Qwen3.5/Qwen3-Next hybrid attention uses reference kernels unless matching optional CUDA extensions are available. The node does not install packages at runtime; on this torch/CUDA build the available `flash-linear-attention` wheels are not recognized by transformers 5.16 and `causal-conv1d` has no matching wheel, so those transformers warnings are expected. For speed on a 16 GB GPU the Model node now selects **q4** for 9B safetensors models, keeps quantized models fully on the GPU, and the GGUF version is faster still.
