# ComfyUI PromptForge

A full model workflow suite, not just a prompt enhancer: load the model, encode the prompt and references,
generate, upscale and save. It covers **Krea 2**, **MOSS Image**, **Qwen Image**, **Anima**, **MiniMax H3** and
other families, plus shared loader, sampler, latent, action, prompt and save tooling.

- **Loading** — one loader for checkpoints, diffusion models and GGUF, with quant, context and GPU-layer planning; LoRA loader, and image / video / audio loaders with a picker and library.
- **Encoding** — family-aware text encoding, edit encoders for Krea and MOSS, VAE encode, and reference conditioning.
- **Prompting** — the Prompt Enhancer with per-family presets, the Action libraries, and the H3 Idea / Parts nodes.
- **Generating** — family-aware sampler with saved presets, empty latent with aspect and size presets, preview override, frame interpolation.
- **Saving** — image and video savers with codec control, plus batch and upscale helpers.

The sections below cover the three main families and their workflows, then the shared node reference.

## Krea 2

Required extras: `comfyui-krea2edit`, `krea2-nag` (only for the transfer workflow) and the Krea 2 models you
own. The LoRA node starts with the identity LoRA selected in the edit workflow.

### Workflows

- `KREA2_T2I_v2.json` — text to image at 1.5 MP. Write in `prompt_text`, pick an action or switch `use_action`
  off, then queue.
- `KREA2_EDIT_v8 AAAA.json` — identity edit from `LoadImage`; the person comes from the picture and is cloned
  into the scene you write. The sampler and the patched target use an empty latent; only `source_latent` and
  `source_image` get the encoded picture.
- `KREA TRANSFER.json` — face transfer between a source and a target picture.

### Face mask and reference position

Edit ships a face/hair/neck mask into the patch `ref_boost_mask` with `ref_boost` 3, so identity is pulled from the face instead of the whole picture, and `ref_position = beside` lets the pose move away from the source layout. Both servers run ComfyUI with `--use-pytorch-cross-attention`, needed for masked reference attention on RTX 50.

### Settings

Use the `Krea` presets in the sampler list: FinePorn 8-10 steps, cfg 1, `euler`/`simple`; turbo checkpoints
8 steps. Krea edit keeps identity best at 8-10 steps because the identity LoRA does the work, not more steps.

### Negative prompts at cfg 1

The NAG switch is `use_nag` on the model patch: `PromptForge Krea NAG` (text-to-image) and
`PromptForge Krea Edit Model Patch` (edit and transfer). Off passes the model through; on applies normalized
attention guidance from the negative prompt at cfg 1, without changing cfg. The encoders always encode the
negative, which costs nothing when NAG is off.

## MOSS Image

Required extras: ComfyUI with MOSS support, `ComfyUI-MOSS-Toolkit`, `moss_image_edit_lora_rank_64_bf16.safetensors`
for edit, and the MOSS models. Edit always keeps one reference image, `Moss-Image-Edit` and `denoise 1`: the image is
VAE-encoded into reference latents at t = 0, not used as an img2img latent, so the sampler redraws the same frame.

### Workflows

- `MOSS_T2I_v2.json` — text to image at 1.5 MP.
- `MOSS_EDIT_v4.json` — reference edit; pick the picture in `LoadImage`, write the change.

### Settings and DMD

Use the `MOSS` presets, matched to the model. Editing runs at 1 MP with the target size taken from the
scaled source image (Empty Latent `res_mode = custom`, WIDTH/HEIGHT from Image Scale), 30 steps, cfg 4,
`euler`/`normal`, `image_strength` 1.0. Below cfg 4 the negative branch stops holding and accessories appear.
Use the `MOSS Image Edit` prompt preset (short "Change only X to Y. Keep the character, face, hair, pose,
background and composition unchanged.").

MOSS alone holds the face only on small attribute edits; a new pose or scene redraws it. The edit workflow
ships a bypassed optional face transplant (`Forbidden Vision` Face Edit Prep / Face Edit Merge) for when the
face must match: Ctrl+B on **Face Transplant** turns it on. It scores 0.91-0.95 by face embedding, but reads
as a pasted face; left off, the scene looks cleaner and the face drifts. Crop the source at the chin
(`ImageCrop`) when it is on, so the collar and background do not come along. The DMD LoRAs (`lodestones-8-steps`, `sft-4-steps`) run at 8 steps, cfg 1,
`er_sde`/`simple`; on PrimeLust use 12 steps and cfg 1.5. A denoise below 0.8 tiles the edit result.

## MiniMax H3

Required extras: the H3 model, its VAE and an audio VAE. The `PromptForge_H3_AllInOne_Modes_v6.json` workflow
covers all modes; switch them in the Modes node. Everything changed often (mode, action, idea, writer, format, sampling, LoRA, generate) sits in one row next to each other. The v5 file is left untouched.

### Action node and roles

85 acts, each with an explicit who-is-where description, the orifice for sex acts (vagina or anus, never swapped), and a 9:16 or 16:9 frame. The camera list is filtered per act and puts POV of one named person first; a POV line is enforced as that person's eyes for the whole clip, their face never shown, the other person always in frame. Every role has
its own picture picker (Woman, Man, Partner): choose the reference slot, the first frame, none and only a
description, or `disabled`. Subject 3 defaults to `disabled`; pictures left unassigned become anatomy/detail
references, not extra people. See the [H3 Action reference](#promptforge-h3-action-adult-presets).

### Writer and API

The writer settings (template, style, sampling, retries) are described under the
[Prompt Enhancer](#promptforge-enhancer-promptforge_promptenhancer). For headless use,
`PromptForge_H3_AllInOne_Modes_v5.api.json` is already in `/prompt` format; the normal workflow stores the
image picker as an extra leading widget, which order-based converters must skip.

H3 prompts always open on `[Shot 1]` with no timestamp; later shots are `[Shot 2] At MM:SS.mmm, the camera cuts to...` with strictly increasing times. The node adds a missing `[Shot 1]` check, strips a timestamp from the first shot, and keeps the mode's fixed frame-alignment line as the first line.

### Body and age sliders

The base model treats body size loosely; a prompt alone cannot hold a flat chest or a narrow frame. These
community sliders work on top of H3 and stack, so test one at a time at a low strength:

- `HMBreastsV2.safetensors` — trigger `HMBreasts`, then plain size/shape words (`tiny`, `small`, `natural`, `perky`).
- `ButtSlider_V1.safetensors` — signed; negative is smaller, positive larger.
- `MM-H3 - Weight Slider.safetensors` — signed overall figure weight, roughly -5 to +5.
- `Minimax_H3-Age_Slider` — age; use the `Ref2VA - Age Slider v2 - EXPERIMENTAL` file with reference modes.

The first three are trained on the base model, not on reference mode, so reference conditioning can fight
them. Keep the strength modest and keep the body words in the prompt exact.

## Shared node reference

### PromptForge Model (`PROMPTFORGE_Model`)

Owns model choice and runtime configuration; outputs a `MODEL` bundle and a `REPORT` string.

- **Model picker**: one list split into two sections — **local models (on disk, ready)** first, then **not downloaded (fetched on first use)**. Every entry is tagged: `[LOCAL Q4/Q8]` names the GGUF quant files present, `[VISION]`/`[TEXT]` says whether a matching mmproj exists (`[VISION dl]` = catalog mmproj not downloaded), `[LOCAL]` marks a cached or scanned model, `[DOWNLOAD]` marks a catalog model that is not on this machine yet (the Model node prints that warning before loading). The console prints a one-line count of local vs. downloadable models at startup, and the Config Finder report ends with the not-downloaded count. If a requested quant is not local but another quant is, the node uses the local one and says so instead of downloading. The catalog covers Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Qwen2.5, Qwen3-VL, Qwen2.5-VL, Gemma 3/4, Llama 3, Mistral/Ministral and Phi-4 heretic/abliterated models. Local models found in the registered `text_encoders` folders (including `extra_model_paths.yaml`) and the `LLM`, `llm`, `gguf` … subfolders of `models/` are listed by their path relative to `models/`, so a workflow saved on one machine resolves on another with the same layout. Single-file encoders without token embeddings (CLIP-L/G, T5, LTX projections) are hidden because they cannot generate text. GGUF generation streams tokens into ComfyUI’s progress bar. `local_model_path` and `mmproj_path` cover anything outside the lists. The `clip` input lives here and is passed to the enhancer in the model bundle. The default is the first local GGUF when llama-cpp-python is installed.
- **Automatic runtime**: the node reads total VRAM and the selected model, then picks the best **local** GGUF quant that fits, the largest context (16k first), KV precision, GPU layers, batch sizes, windowed SWA and flash attention. For safetensors it picks bf16/fp16, q8 or q4; local HF directories get their size from the safetensors files. At run time the free VRAM is checked again: bf16/q8 drops to q4 when it does not fit, and a GGUF that fails to load retries with as many GPU layers as actually fit (the rest run on CPU). The only remaining override is `context_size` (0 = auto).
- **Quant fallback**: unsupported combinations warn and use a working option — q5/q6 on safetensors load as q8, q8/q4 on a single-file encoder load in fp16, q8 switches to q4 when free VRAM is insufficient, CPU-offloaded layers are allowed (slower) instead of raising, a missing GGUF quant falls through the available files, and an incompatible KV/flash/SWA choice is retried with the safe value. The report re-checks local availability at run time, so stale `[AVAILABLE]` widget labels still show `[LOCAL]`.


### PromptForge Action (`PROMPTFORGE_ImageAction`)

A pose library for any image model: 90 named actions (portraits, SFW solo and couple poses, nude and
explicit solo sets, sex positions, oral, hands and explicit detail), 29 places, 27 cameras (framing,
angles, POV of either person, over the shoulder, handheld, locked off), 12 light
setups, 15 outfits and 9 styles. Aspect ratio stays with the Empty Latent / resolution node. Category filters the action list; every action states the exact body
arrangement, hands, legs, face and camera so there is no ambiguity. `source` switches between keeping
the person in the connected image (image-to-image) and inventing them from `subject_1` / `subject_2`
text (text-to-image). Notes and overrides are appended last. Every picker (category, action, place, camera, lighting, outfit, style) has a `random` option that rolls a concrete value on each run and shows the roll on the node. The enhancer has `use_action`; turn it off to ignore the action for a fully custom prompt without disconnecting the node. Output connects to the enhancer's `action` input; while `use_action` is on the brief replaces the enhancer prompt, and the `instructions` field carries custom additions.

### PromptForge Enhancer (`PROMPTFORGE_PromptEnhancer`)

Takes the `MODEL` bundle and writes the prompt. Only generation inputs live here.

- **Request fidelity**: every preset treats the user's text as the authority on subject, acts, intensity and tone. Hardcore, explicit or intense requests are written directly and specifically, without sanitizing, softening, euphemizing or fading to black.
- **Prompt input**: free-form text box — loose natural language, random ideas, keyword piles, detailed descriptions; the model always expands to a proper prompt for the target mode/model. With `use_action` on, this text is ignored and the Action brief drives the prompt; custom additions go in the Action node's `instructions` field.
- **Modes**: `text_only` (ignore media) or `inputs` (use every connected image from `image` / `image2` / `image3` plus frames sampled from `video` via `frame_count`). The prompt is told exactly which inputs arrived: bypassed or missing sockets are never mentioned, and the model is told not to invent extra images, references, video or frames.
- **Preset**: `auto` or any bundled preset (Krea 2 T2I / Edit, MOSS Image T2I / Edit, Qwen Image T2I / Edit, MiniMax H3, Anima, ACE-Step, generic styles…). With a Loader PIPE connected and the preset on `auto`, the family's preset is used, and its edit preset when an image is connected.
- **MiniMax inputs**: the MiniMax H3 presets (T2VA, I2VA, L2VA, FL2VA, Ref2VA) get an authoritative list of what is connected. Labels follow the H3 encoder: `<Picture 1>`, `<Picture 2>`, `<Picture 3>` for the connected images in order with gaps closed up (only `image2` connected makes it `<Picture 1>`), and `<Video 1>` for the video. Each mode uses only what it can: T2VA none, I2VA/L2VA one image as first/last frame, FL2VA two images as first and last frame, Ref2VA all images plus the video; anything else that is connected is ignored and the console says so. The frame-alignment line is written by the node for exactly the attached pictures (omitted when none), the model is told which labels do not exist, and any sentence that still cites an unattached picture, video or audio label is removed. The node attaches no audio, so `<Audio N>` only appears when your own text describes it.
- **Video duration**: `video_seconds` sets the exact clip length; MiniMax H3 presets use it for shot budget, timestamps and the frame alignment line.
- **Headless / API**: use `PromptForge_H3_AllInOne_Modes_v5.api.json`. It is already in `/prompt` format; the regular workflow file stores the image picker as an extra widget, so homemade order-based converters read the filename as the description and send no picture.
- **NSFW audio**: the H3 presets make `overall_soundscape` name room tone, skin impact, wet penetration sounds and breathy voices in that order, and put contact sounds into the action beats. Dialogue is breathy and pauses for breath.
- **Output length**: each family gets an automatic output cap — Krea 220 words, Krea edit 220, MOSS 320, MOSS edit 80, Qwen 380, Qwen edit 160, Anima 150 tags, H3 750-900 depending on clip length. The cap is written into the request and sets `max_tokens` to that family's ceiling, so the writer wraps up instead of running to another model's length. `length_words` overrides the cap; `0` keeps the automatic value.
- **Age**: the user's age descriptor or number is used exactly as written and never overwritten. Without one, no number is required and no age number is inferred; 18 is not the default.
- **Sampling**: seed, max_tokens (a requested value is raised to a safe budget: 1600 for H3 actions, 1200 for H3 ideas, 900 for Anima tags, 700 for image prompts, 220 for refine/technical) , temperature 0.7, top_p 0.8, top_k 20, num_beams, repetition_penalty 1.0, presence_penalty 1.0 (Qwen serving uses 1.5; 0.5-1.0 is safer on long structured prompts, 0 turns it off).
- **Auto-retry** (`auto_retry` toggle on the Prompt Enhancer, on by default; off keeps the first output): rejects empty, too-short, degenerate/repetitive, leaked-reasoning ("Okay, the user wants…"), refusal or broken-contract output and retries up to two more times with new seeds; if every try is a loop, the longest one is returned with repeated sentences removed. Status lines name the reason (too short, repetitive, reasoning leak, refusal, negative wording). Outputs under 400 chars (40 for refine/technical presets, 400 for the Anima tag preset, 2 characters per word of the applied cap) count as too short.
- **Repetition guard**: the sampler is only stopped on an unmistakable 12-gram loop; repeated sentences and repeated 5–10 word phrases are rejected afterwards. Qwen3-VL runs at its tuned 0.7 / 0.8 / 20 / 1.0 with presence penalty 1.5; a hard `no_repeat_ngram_size` is not forced, because it distorts long structured prompts.
- **Negative prompt**: a non-empty `negative_prompt` feeds exclusions into the generation request so the model writes around them instead of naming them.
- **Text beats pictures**: the user's prompt and action block win on age, body size, proportions, clothing state, pose, act and camera. Images supply identity and detail; a caption's guesses never override a direct user word.
- **Size fidelity**: size words are exact (`flat`, `small`, `narrow`, `skinny`, `slim`, `full`, `thick`, `curvy`), never promoted to a safer synonym or hedged with “average”. A requested size difference is carried by relative height, width, hands and how the larger body covers or lifts the smaller one.
- **Camera fidelity**: the chosen camera is written as its exact geometry. POV is the viewer's eyes, tilted down their own body: their chest, stomach and hips stay in the lower third, their face stays out, and the framing stays medium-wide rather than a macro crop.
- **No embellishment**: every preset forbids unrequested moisture (sweat, wetness, droplets, sheen) and forbids inventing subjects, actions or scene changes.
- **No negatives in the positive prompt**: Krea 2, Qwen Image and MiniMax H3 run at distilled CFG where negatives are weak or ignored; MOSS Image / Anima have a separate negative prompt. Presets require positive-only wording; the node retries if classic negative-prompt vocabulary appears ("no blurry", "without watermark", "avoid extra fingers"). Ordinary wording such as "no people" is accepted.
- **Chat templates**: GGUF text generation renders the model's own `tokenizer.chat_template` with `enable_thinking=false` (Qwen3/3.5 empty think block, Gemma 4 primed thought channel) and runs a plain completion, so thinking models answer directly. Models without a template fall back to a family format (Gemma, Llama-3, ChatML/Qwen, Mistral) with the llama.cpp reasoning budget set to zero. Prompts are sent as a real system turn plus a user turn. The connected-CLIP route passes the preset as the encoder's `system_prompt` (same contract as ComfyUI's Generate Text node) and wraps the turns in ChatML itself for encoders whose tokenizer has no chat template. Output cleanup strips role markers across all engines.
- **Output purity**: the system prompt forbids first-person commentary and self-assessment, and the post-clean removes instruction references, reasoning, closings, “Here is the prompt” lead-ins, and sign-offs such as “I followed the instructions / I did a good job”. Only the prompt text leaves the node.
- **Thinking**: reasoning is disabled where the template supports it and always stripped from `ENHANCED_PROMPT`; it is returned on `THINKING`.
- **Vision speed**: reference images and video frames are scaled to about 1 megapixel before the vision model sees them (about 1k tokens each instead of thousands for a 4K image), and ComfyUI models are only unloaded as far as the GGUF plus image encoder actually needs.
- **Vision safety**: image/video mode never silently falls back to text-only. Vision GGUFs load the family-specific llama.cpp chat handler (Gemma 3/4, Qwen 2.5/3/3.5) and prefer the mmproj stored next to the model. Gemma 4 12B Heretic can emit only a `thought` channel for long multimodal prompts; the node detects that and reports it instead of returning an empty prompt — use Qwen3-VL for vision with that model.
- **Custom system prompt**: a non-empty override field replaces the selected preset.

**Outputs**: `ENHANCED_PROMPT` (final prompt only), `THINKING` (removed reasoning, empty when none) and `REQUEST` (the exact system + user text sent to the writer, for debugging).

### PromptForge Config Finder (`PROMPTFORGE_ConfigFinder`)

Standalone output node; it never generates a prompt.

- Scans every locally available GGUF/safetensors model (or one selected model) and prints the recommended settings for this GPU.
- `OUTPUT_NODE`-only: queue it on its own (or use its button) to inspect configs without running the image workflow.

### PromptForge Frame Interpolate (`PROMPTFORGE_FrameInterpolate`)

RIFE / FILM frame interpolation for large frames (for example the 2x-decoded MiniMax output) that backs off by itself instead of crashing ComfyUI. It uses ComfyUI's own interpolation models, so no other node pack is needed.

- **`ckpt_name`**: every model in `models/frame_interpolation` (RIFE 4.25+, FILM), plus `blend`, a plain cross-fade that needs no model and is instant (ghosts on motion). Older RIFE `.pth` files (rife47 / 49 / 417) are not supported by ComfyUI's loader.
- **`batch_size`, `overlap`, `scale`, `clear_cache`**: the whole clip runs in one piece by default (`batch_size 0`), which gives the cleanest motion. If memory is tight, `batch_size` splits it, `overlap` shares context frames and cross-fades the seam, `scale` interpolates smaller and scales back, and `clear_cache` releases memory between runs. OOM still backs off automatically (down to 4 frames), then cross-fades.

### PromptForge Save Video (`PROMPTFORGE_SaveVideo`)

A faster stand-in for VHS Video Combine. Frames are converted to BT.709 YUV on the GPU while ffmpeg encodes, and video, audio and the embedded workflow go into one ffmpeg run. About 2x faster than the VHS path on 1920x1088 clips (240 frames in ~1.7 s on an RTX 5080 against ~3.2 s).

- **Codecs**: h264 / h265 / av1 on nvenc (GPU) or x264 / x265 / SVT-AV1 (CPU), `speed` preset, `quality` (crf/cq) or `bitrate`, 8 or 10-bit.
- **Options**: `film_grain` (fine grain added while encoding, 0.005-0.01 is subtle), `pingpong`, `save_output` (off writes to temp), `save_metadata` (workflow in the file), optional `audio` with `audio_bitrate_kbps`.
- Odd sizes are padded by one pixel to an even size. Needs `ffmpeg` on PATH (or `VHS_FORCE_FFMPEG_PATH`). nvenc rejects very small frames (under about 150 px); use a CPU codec for those.
- Keep the interpolation node's `output_precision` on float32 when feeding this node or VHS.

### PromptForge Loader (`PROMPTFORGE_Loader`)

One node for the model, its text encoders and its VAEs. It calls ComfyUI's own loaders (and ComfyUI-GGUF's, when installed), so new model types work as soon as ComfyUI supports them.

- **Model**: one list with `[ckpt]` checkpoints, `[unet]` diffusion models and `[gguf]` models. `family` is auto-detected from the file (ComfyUI's own detection on the file header, then the folder and file name) or can be set to narrow the list.
- **Text encoders**: `clip_1`..`clip_4` take safetensors (`[te]`) or GGUF files. The node shows as many slots as the family uses and names their roles. The encoder type is chosen from the family (`clip_type` overrides). `(baked-in)` uses the encoder inside a checkpoint.
- **VAEs**: `vae` (image or video, `(baked-in)` for checkpoints), `audio_vae`, and an optional `extra_vae` (for example an upscaling VAE to switch to later).
- **Weights**: `weight_dtype` auto picks per machine from `optimize_for` (default `quality`). `quality` keeps the stored precision. `balanced` uses fp8 storage only when the model does not fit in VRAM. `speed` also uses fp8 compute when a 0.3 s matmul benchmark (run once per GPU and torch version, kept in `autotune.json`) shows it is at least 1.2x faster than bf16. Already-quantized files are never changed.
- **Outputs**: MODEL, CLIP, VAE, AUDIO_VAE, EXTRA_VAE and an INFO text saying what was loaded and why.

### PromptForge Empty Latent (`PROMPTFORGE_EmptyLatent`)

Empty latent for any family, image or video. Size is a custom width/height, a preset, or an aspect ratio plus megapixels (15 ratios, lowest to highest), snapped to the family's multiple (16, 32 for LTX/MiniMax, 64 for SD/SDXL). `orientation` (landscape by default) flips landscape/portrait. `randomize_ratio` picks a ratio from `ratio_pool` (empty = all) and `randomize_orientation` picks landscape or portrait, both driven by `seed`, so repeated runs vary. Video frames come from `frames` or `seconds` x `fps` and are snapped to the family's grid (Wan/Hunyuan 4n+1, LTX/Cosmos 8n+1, MiniMax H3 17k+5). The latent layout comes from the connected `model`, the `pipe` or the chosen family, using ComfyUI's own latent formats (MiniMax H3 goes through ComfyUI's own empty AV latent node). Outputs LATENT, WIDTH, HEIGHT, FRAMES and INFO.

### PromptForge Sampler (`PROMPTFORGE_Sampler`)

KSampler, KSampler (Advanced) and custom sigmas in one node, built on `comfy.sample.sample`, so all samplers, schedulers and model patches behave as in the stock nodes.

- **Schedules**: `scheduler = auto` uses LTX Video's token-count dependent schedule (max_shift, base_shift, terminal exposed) for LTX models and `simple` otherwise. Sigmas can also come from a SIGMAS input or a typed list (`custom_sigmas`), and `denoise` works with all of them. The sigmas used are an output.
- **runs**: samples seed, seed+1 ... in one node and returns one batch.
- **Advanced**: `start_at_step`, `end_at_step`, `add_noise`.
- **Recommended settings**: press `Use recommended settings` with a Loader's PIPE connected to fill sampler, scheduler, steps and cfg (nothing changes when you only connect the pipe). They come from the model's sidecar (the most common sampler, scheduler, steps and cfg of the author's example images; prompts are never read), then name hints (turbo / distilled -> 8 steps, cfg 1), then family defaults from community documentation (Krea 2: res_multistep/simple; MiniMax H3: euler/simple; LTX: the LTXV schedule). These are starting points, not official values.
- **Step cache**: `step_cache` (off / auto / easycache / lazycache) wraps ComfyUI's own EasyCache and LazyCache, which skip model calls on steps where the output barely changes and work across architectures (including audio+video models). `auto` turns on easycache from 15 steps up; distilled 4-8 step runs should stay off. `cache_threshold` (higher = faster, softer), `cache_start` and `cache_end` set when it may skip. Measured on Anima, 30 steps, RTX 5080: easycache 0.1 skipped 15/30 steps (1.8x, similarity to uncached 0.991), 0.2 skipped 17 (2.05x, 0.978), 0.4 skipped 19 (2.4x, 0.971); lazycache 0.2 gave 2.05x at 0.988. Results vary by model.
- **Presets**: save, apply and delete named presets (steps, cfg, sampler, scheduler, denoise, runs, step cache). They are stored in `models/PromptForge/sampler_presets.json`, so a models sync carries them to other machines; an old copy in `user/promptforge/` is read once as a fallback. Connect the Loader's PIPE to the sampler and the preset list shows only presets for that model family and model, plus generic presets; `show all presets` reveals the rest. A saved preset records the connected family and model, so it keeps matching that model and does not clutter other models.

Tested starting points at about 1.5 MP (identity tests used a source picture through the edit graphs; every row was rendered and checked):

| Model | T2I | Edit |
|---|---|---|
| Krea 2 FinePorn V6.1 | 10 steps, cfg 1, `euler`/`simple` (or `er_sde`/`beta`) | 10 steps, cfg 1, `euler`/`simple`, denoise 1 |
| Krea 2 Turbo Anime / ReaKrea2 Turbo | 8 steps, cfg 1, `res_multistep`/`simple` or `euler`/`simple` | 8 steps, cfg 1, same |
| MOSS FinalCut int8 / FP8 | 25 steps, cfg 3, `er_sde`/`sgm_uniform` | 30 steps, cfg 4, `euler`/`normal`, denoise 1 |
| MOSS Jib Mix | 20 steps, cfg 1, `euler`/`simple` | 30 steps, cfg 4, `euler`/`normal`, denoise 1 |
| MOSS Dirty | 25 steps, cfg 2, `er_sde`/`simple` | 30 steps, cfg 4, `euler`/`normal`, denoise 1 |
| MOSS PrimeLust fp16 / nvfp4 | 30 steps, cfg 5, `er_sde`/`normal` | 30 steps, cfg 4, `euler`/`normal`, denoise 1 |
| with a DMD LoRA (lodestones 8-step, sft 4-step) | 8 steps, cfg 1, `er_sde`/`simple`, LoRA strength 1 | 12 steps, cfg 1.5, `euler`/`normal`, denoise 1 |

Without a DMD LoRA, 30 steps at cfg 4 hold identity best; a DMD LoRA trades some of it for speed. Keep the target size equal to the scaled source: a different aspect ratio makes the model reframe the person. Krea edit holds identity strongly at 8-10 steps, cfg 1. Presets with these names are already in the sampler list.

### PromptForge Quantize Text Encoder (`PROMPTFORGE_QuantizeTextEncoder`)
Turns a Qwen3.5 model folder (shards plus, if present, a separate `model-mtp*.safetensors`) into one quantized ComfyUI text encoder in `models/text_encoders`, in under a minute. Format `mixed` (default) is NVFP4 for the MLP and INT8 ConvRot for the attention layers, which ComfyUI runs reliably for long prompts; an all-NVFP4 file fails on long prompts. The vision tower, embeddings, output layer and the multi-token-prediction (MTP) head stay in bf16. Pick the file as the prompt writer: ComfyUI then writes with speculative decoding from the MTP head. A 9B model: 18 GB -> 9.6 GB, a prompt written in 8 to 10 s, against 18 s from the folder with transformers. Run the node once per model and refresh the lists (r).

### PromptForge Quantize Model (`PROMPTFORGE_QuantizeModel`)

Converts a bf16 / fp16 diffusion model to **NVFP4** (about 3.5x smaller, native on RTX 50) or **INT8 ConvRot** (about 2x smaller, native on RTX 20 and newer, so it helps the 3080 laptop) and saves it next to the source in the same models folder. Refresh the model lists afterwards.

- **What gets quantized**: only 2D weights inside the numbered transformer blocks, minus embeddings, norms, modulation and input/output projections, plus extra sensitive names for the families that need them (Krea 2, MiniMax H3, Anima, Qwen Image, Wan, LTX, Flux, Chroma, Z-Image). `keep_edge_blocks` keeps the first and last N blocks unquantized, `extra_keep` adds your own name fragments.
- **Memory**: the source is read tensor by tensor, so the full model is never in RAM; the converted model is built in RAM. An Anima model (3.9 GB) converts in a few seconds. Measured on Anima, 30 steps, RTX 5080: INT8 ConvRot 1.63x faster (similarity to bf16 output 0.992), NVFP4 1.37x (0.929). NVFP4 loses more on small models, so compare before replacing a bf16 file.
- **Report**: file sizes and the relative weight error after dequantizing (mean and the worst layer). For reference, NVFP4 is about 0.095 and INT8 ConvRot about 0.009 on Anima.
- Already quantized files are refused; start from the original bf16 / fp16 file. Output is the ComfyUI format (`.comfy_quant` per layer), so any ComfyUI loader reads it.

### PromptForge MiniMax H3 VAE Decode (`PROMPTFORGE_MiniMaxDecode`)
Decodes MiniMax H3 video latents with H3's own decoder. It always tiles at 256 px (whole-frame and larger tiles look fried and banded with H3) and, on an out-of-memory error, steps down to 85 / 34 / 17 frames per pass, without the generic tiler that ignores H3's blend rules. The step that worked is remembered per latent size. Frames are written into one preallocated buffer, non-finite pixels are repaired and reported, and `tile_size` / `frames_per_pass` can be forced. Works with the 2x packed-projection decoders too.

### PromptForge Load Audio / Load Video (`PROMPTFORGE_LoadAudio`, `PROMPTFORGE_LoadVideo`)
A sound, or a video (as 24 fps frames with its sound), from the input folder with a description. Empty, or `enabled` off, = no output, so the nodes can stay in a workflow unused; Load Image has the same `enabled` switch.

### PromptForge Load Image (`PROMPTFORGE_LoadImage`)
Loads an image from the input folder through a thumbnail browser (search, folder filter, upload). Each image can have a stored description (`user/promptforge/image_descriptions.json`): *Save description* stores the text typed in the node (picking another image never changes that box; the stored text shows as its placeholder), and the node outputs it as `DESCRIPTION` next to `IMAGE`, `MASK` and `NAME`. Text typed in the node is used as is; an empty box falls back to the stored description. Handy for switching a character and its description together in reference-to-video workflows.

### PromptForge Text Encode (`PROMPTFORGE_TextEncode`)
The positive prompt and the negative prompt in one node: `text` becomes POSITIVE and `negative` becomes NEGATIVE on the same CLIP, so a graph needs one encode instead of two. The negative is always encoded; the model patch's `use_nag` decides whether it is used.

### PromptForge Edit LoRA (`PROMPTFORGE_EditLora`)

The edit LoRA of a family, fixed by name: `Krea 2/tool/krea2_identity_edit_v2` or
`MOSS Image/tool/moss_image_edit_lora_rank_64_bf16`. Pick the family, adjust `strength`, or switch `enabled` off.
Nothing else can be cleared, so the identity/edit path cannot be broken by an empty LoRA list.

### PromptForge Krea NAG / Krea Edit Model Patch (`PROMPTFORGE_KreaNAG`, `PROMPTFORGE_KreaEditModelPatch`)

Model patches for Krea 2. `PromptForge Krea NAG` is the text-to-image NAG path; `PromptForge Krea Edit Model Patch`
adds the source-reference path and can also apply NAG for the edit and transfer workflows. Both have `use_nag`
(off for the NAG node, on for the edit patch) and work at cfg 1 without changing cfg.

### PromptForge MOSS Edit Encode / Krea Edit Encode (`PROMPTFORGE_MossEditEncode`, `PROMPTFORGE_KreaEditEncode`)
Edit conditioning for both CFG passes in one node. The MOSS node VAE-encodes the reference images into reference latents and attaches them to the positive and to the negative, each with its own image strength (the strength scales the text-encoder vision tokens, so it only applies to the ai-toolkit formats or when no VAE is connected). The Krea node grounds the positive and the negative on the source image through the `Krea2EditGroundedEncode` path. The prompt is the positive instruction, the negative is what to avoid. The negative is always encoded; the model patch's `use_nag` decides whether NAG uses it at cfg 1.

#### Image library (picker and sidebar)
The Choose image picker, and an **Images** tab in the left sidebar, share one library of the input folder. **Tools** runs a local vision model (a GGUF with its mmproj, found in your text_encoders / LLM folders) that gives every image keywords, a caption, a people count and an NSFW flag; results are stored in `user/promptforge/image_tags.json`.
- **Search** looks through names, keywords, captions and descriptions; filters for folder and theme (female, male, group, people, nsfw, animals, scenery, objects, other).
- **NSFW** images are blurred everywhere (picker, node preview, editor) until *Show NSFW* (or the eye button on the node) is on.
- **Tag untagged / Tag selected / Re-tag all** run as a background job with progress and cancel. ComfyUI's loaded models are unloaded while it runs.
- **Organize into theme folders** shows what will move and asks first; tags and descriptions follow the files. *Undo last organize* moves everything back. Workflows that point at an old path need the image picked again.
- The **i** button on a tile edits keywords, theme, NSFW flag and description by hand.

### Endless Video (category `🔮 PromptForge/Endless Video`)
Makes a long MiniMax H3 video part by part. One node does the generating (**PromptForge H3 Endless Video**); small nodes plug into it, and every one of them is optional except the Story. The generator saves nothing: it outputs `frames` (IMAGE), `audio`, `prompts` and `report`, so you save them with PromptForge Save Video (or anything else) afterwards. A ready all-in-one workflow (`PromptForge_H3_AllInOne_Modes_v5.json`) is provided separately, in your workflow folder: text to video, image to video, characters / references, voice, extending an existing video, first-pass preview and endless multi-part video in one graph; you switch modes by bypassing the optional nodes (its READ ME note lists which). (The old `H3 Endless` and `H3 Options` nodes are gone; workflows built with them need rewiring.)

### PromptForge H3 Action (adult presets)

| Node | What it holds |
|---|---|
| **H3 Modes** | What writes the prompt (`prompt_source`: my idea or action preset; the other node is ignored) and what kind of video: **T2V** (text only, empty latent), **I2V** (first frame encoded and anchored at frame 0), **L2V** (last frame anchored at the end), **FL2V** (both), **REF2V** (pictures, videos and sounds encoded as references). Also one clip or several parts, extend an existing video, second stage on/off, and the prompt source: your idea, the Action node, or the Parts node. Its overview lists the files chosen in the Load nodes with a tick for each; rows the mode does not use are dimmed. Each mode also picks its prompt template (T2VA, I2VA, L2VA, FL2VA, Ref2VA). |
| **H3 Idea (my text)** | Your own text with the saved-ideas bar (Save / Save as / Delete, arrows to flip through; stored in `user/promptforge/ideas.json`), negative prompt and dialogue with its language. |
| **H3 Action (adult presets)** | A solo act or a position (57, each with an explicit who-is-where definition and a recommended 9:16 or 16:9 frame), a camera angle (13) and who is who: every role has its own picture picker (Woman, Man, Partner) for a reference picture by its input slot, the first frame, none and only a description (`look` text), or `disabled`. Subject 3 defaults to `disabled`; a picture left unattached to any role is passed to the writer as an anatomy/detail reference, not as a third person. The roles of the chosen action are shown under it (for cowgirl: subject 1 on top, subject 2 on the back). Category, act, camera and penetration all have `random`, rolled per run and shown on the node. Notes carry setting and mood. The prompt writer writes the whole video prompt template as one continuous shot with the act already in progress, for one clip or spread over the parts: lead-in, main action, peak. Works with every mode; where a picture is not available the person is described in words. |
| **H3 Prompt Writer Settings** | How the writer writes, for the Idea and the Action node alike: use the writer or not, template (one NSFW-capable template per mode; a template you pick wins over the style), style, temperature, top_p, top_k, repetition penalty, max tokens, retries, approval. |
| **H3 Parts** (optional) | A long-video node: the same action category/act, penetration, camera and subject pickers as Action, plus one custom note line per part (speed up, pause and talk, pull out, come). Used when Modes says several parts and the prompt source is Parts; the action brief is written per part with that part's note. A box can start with `(10s)` for its own length or `raw:` for a finished prompt. Without it the generator makes one clip. |
| **H3 References** | The first and last frame (I2V, L2V, FL2V), up to 9 reference pictures, 3 reference videos (with soundtracks) and 3 reference sounds, each with a description (wire a Load Image / Load Audio DESCRIPTION, or type it). The prompt writer is told what `<Picture N>`, `<Video N>` and `<Audio N>` are, so the Story can name them; the descriptions stay put while you rewrite the Story. |
| **H3 Video Format** | Aspect ratio, orientation, megapixels, seconds (any length; the length of each part with Parts). |
| **H3 Continuity** | How parts join (carry the last frames forward / give the ending as a reference video / independent parts), story style, carried frames, join smoothing, join check and retries. |
| **H3 Sampling** | Sampler, scheduler, steps, cfg of the first pass and the video / audio sigma shifts (0 / 0 by default; no separate shift node needed). Defaults follow the 10Eros Max two-pass workflow: euler + beta57, 8 steps, cfg 1. |
| **H3 Second Stage (upscale)** | Connect it to turn on two-stage sampling: latent upscale model (ComfyUI's own MiniMax H3 latent upscaler, `models/latent_upscale_models`), size by multiplier / megapixels / dimensions, steps (3), start sigma (0.9035), sampler (er_sde). Use the fp32 upscaler; the fp16 one washes out contrast and skin texture. 3 steps give 0.9035, 0.6316, 0.3158, 0. The text and the references are encoded once and used again in the second stage; only the first / last frame and the carried ending are encoded again at the new size. The optional `model` input samples the second stage with another model, for example without the LoRAs. The generator's `first_pass_frames` / `first_pass_audio` outputs give the first pass (before the upscale) for saving the small version separately; it is decoded only when they are connected. |
| **H3 Preview** | Live preview mode, frames (up to 99999 = all), fps, size, tiny decoder from `models/vae_approx`. |
| **H3 Performance** | Attention backend, feed-forward chunking and video sparse attention (VSA, 10% kept: faster, and what FastH3-based checkpoints such as 10Eros Max betas are trained for; without it they can flicker. Turn it off for other H3 checkpoints). |
| **H3 Continue Existing Video** | Frames and sound of a video you already have: the first part continues from its ending and the result is the old video followed by the new parts. |

The generator itself takes the model, CLIP, VAE, audio VAE, the prompt-writer model (PromptForge Model), the story, a seed, and optionally `decode_vae` (e.g. the Loader's 2x `extra_vae`). Defaults apply for every settings node you leave out.
- **How each part is written:** from its own box, with the general prompt as context and the other parts listed as things to keep out; an empty box continues the story. The written prompt appears on the node before the part is sampled; with `review_prompts` the run waits for you to edit, *Approve*, *Regenerate* or *Approve this and all following*.
- **Continuation:** *carry the last frames forward* anchors the last frames and audio of the previous part at the start of the next with H3's guide mechanism (cut again on join); *reference video* hands over the ending as a reference video; *independent* uses only the references.
- **Join smoothing** evens out the brief light flash at a join. **Join check** regenerates a part whose first frame jumps from the previous one (off by default; each retry costs a part).
- **Memory:** everything is unloaded between prompt writer, sampler and decoder. Finished parts are kept as float16 on the CPU and joined at the end with one continuous audio track.
- **Live preview and finished parts** show in the node; text boxes can be resized by dragging their corners.
- The Enhancer has two optional inputs for this, `continuation` and `context_video`.

### PromptForge Preview Override (`PROMPTFORGE_PreviewOverride`)
Live preview for any sampler, shown in the node itself. Put it between the model and the sampler. While sampling, every step's latent is turned into a short animation (H.264 mp4, animated WebP without PyAV) of `preview_frames` latent frames (up to 99999 = all) played at `preview_fps`, at most `max_resolution` pixels on the long side. `tiny_vae` picks what draws it: `latent2rgb` (instant, no file) or a tiny decoder from `models/vae_approx` (`taeh3-approx` for MiniMax H3; flat TAESD-style decoders are rebuilt from the file). `suppress_default_preview` switches ComfyUI's own preview off while it runs. It works for audio+video models too (the video stream is unpacked from the packed latent). Behaviour follows the preview override of KJNodes; the code is new. The Endless Video node uses the same patch internally.

### Dialogue in video prompts

On the Prompt Enhancer, the MiniMax H3 presets accept `dialogue` (what the spoken lines should be about), `dialogue_language` and `dialogue_language_other` (for example Polish). Writing “no dialogue” (or “just sounds”, “no words”, “only moans”) makes it a sounds-only video: no lines, tags or speaker IDs are required and the dialogue check is skipped. Otherwise the enhancer writes the lines in that language inside the dialogue tags with the language tag, keeps everything else in English, and sizes the dialogue to `video_seconds`. The instruction block spells out the exact line shape (`speaker (S1) says: <d>[Language] words</d>`), the rules and the common mistakes, and the output is checked: if a `<d>` tag has no `[Language]` tag, no closing `</d>`, or the wrong language name, the enhancer retries with the problem named (twice at most, with `auto_retry` on) (about 2.3 words or 4 characters per second after a one second lead-in, with pauses between speakers). The language list is what MiniMax H3 speaks reliably according to third-party model cards (Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish); other languages are partially supported.

### Wiring nodes together

The Loader's PIPE output carries the model's family. Wire it to the LoRA Loader (the list switches to that family), the Empty Latent (family), the Sampler (recommended settings) and the Prompt Enhancer (`enhancement_preset = auto` picks the family's preset, for example Krea 2 T2I, MOSS Image T2I, Anima Tags, MiniMax H3 T2VA/Ref2VA).

### PromptForge LoRA Loader (`PROMPTFORGE_LoraLoader`)

One node instead of a chain of LoRA loaders. The node shows every LoRA in your loras folders; nothing has to be added by hand.

- **Search and filter**: full-text search over the path and trigger words, and a family filter (SDXL, Krea 2, MiniMax H3, Anima...). A LoRA's family comes from the LoRA Manager sidecar (`.metadata.json`), the training metadata in the file, its key names, then its folder and file name.
- **Stable list**: rows stay where they are when toggled; `on only` shows just the enabled ones. Number fields change by 0.05 with the mouse wheel. Hovering a cut-off name shows the full path and trigger words, and the (i) button opens the CivitAI information stored in the sidecar (preview, trigger words, usage tips, description, link). The family filter has a `Follow the connected loader` option you pick yourself; nothing is switched automatically.
- **Per LoRA**: on/off, model strength and clip strength (blank = same as model).
- **Random strength** (dice button): draws a strength between min and max, seeded by the node `seed`.
- **Random groups**: LoRAs sharing a group name are alternatives, and one of the enabled ones is picked per run.
- **skip_mismatched**: skips LoRAs whose family differs from the connected model's.
- **Outputs**: MODEL, CLIP, TRIGGER_WORDS (from the sidecars) and a REPORT of what was applied.
- All LoRAs are applied in one pass: the key maps are built once instead of once per LoRA.
- **Layer scope** (video / audio): on audio+video models (MiniMax H3, LTX Video) a LoRA row has a select to apply it to all layers, only the video layers or only the audio layers.
- **Stack report**: with several LoRAs on, the REPORT lists every LoRA's layers and rank, how much each pair overlaps (similarity, from the low-rank factors, no full weight matrices) and how hard they push together compared with the strongest one alone.
- **balance** (off by default): when they push harder than the strongest alone, every strength is lowered by the same factor. Image models can go down to 0.85 of the set strengths; video and audio models (turbo and distillation LoRAs keep their published strength) are left alone unless `balance_floor` is set.
- **Training info**: the (i) dialog also shows what the file itself says (rank, alpha, trained-on model, resolution, epochs ...) and its most common training tags, so LoRAs without a CivitAI sidecar still show something to copy.

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

`presets/promptforge_presets.json` — one crafted template per model family and task: **Krea 2** T2I / edit, **MOSS Image** T2I / edit, **Qwen Image** T2I / edit, **Anima**, **MiniMax H3** (the five modes), **ACE-Step**, plus generic styles. Each follows what its model reads: booru tags for Anima, long ordered captions for MOSS and Qwen, tight visible-scene paragraphs for Krea, H3's required section format for video.

Every preset is NSFW capable — there is no separate SFW/NSFW copy. Each writes explicit nudity, anatomy, positions and choreography when asked, without safety boilerplate. Image presets also carry an image-only guard, and the node strips any leaked audio/soundscape/music/shot fields from image outputs. The old `NSFW › …` and renamed preset values still validate and map to the current preset.

Output length is capped per family, because a good Krea prompt is not as long as a MiniMax H3 prompt:

| Preset | Cap |
|---|---|
| Krea 2 T2I / Edit | 220 words |
| MOSS Image T2I | 320 words |
| MOSS Image Edit | 80 words |
| Qwen Image T2I | 380 words |
| Qwen Image Edit | 160 words |
| Anima Tags | 150 tags |
| MiniMax H3 (all modes) | 750 words up to 10 s, 900 above |
| Enhance / generic styles | 220 / 600-700 words |
| ACE-Step 1.5 | no word cap |

The cap is injected into the request and sets `max_tokens` for that family; the answer is not cut afterwards. Set `length_words` on the Prompt Enhancer to override it; `0` uses the table.

## Models catalog

`catalog/models.json` — heretic VL-capable HF models (4B / 9B / 27B, free, image-capable or text-only) and GGUF variants, plus local scan dirs for text models.

## Dependencies

PromptForge does not need any other custom node pack. Its Python requirements are in `pyproject.toml` (`transformers`, `llama-cpp-python`, `bitsandbytes`, `accelerate`, `huggingface-hub`, `psutil`, `Pillow`, `numpy`, `imageio-ffmpeg`, `gguf`, `flash-linear-attention`); `flash-linear-attention` gives Qwen3.5 model folders (transformers route) their fast kernels, about twice as fast; the first prompt written with such a model compiles them once (around 40 s). ComfyUI itself provides the rest (`comfy_kitchen`, `safetensors`, the MiniMax H3, LTX and frame interpolation nodes). `ffmpeg` should be on the PATH for the video nodes (`imageio-ffmpeg` is the fallback).

GGUF diffusion models and text encoders in the Loader are loaded by `gguf_native/`, which is the loading code of [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) by City96 (Apache-2.0, license in `gguf_native/LICENSE-ComfyUI-GGUF`) included in this pack. It adds no nodes, so it can sit next to the original pack without clashing.

## Install

Drop this folder into a ComfyUI `custom_nodes` directory. Install the requirements into the Python environment that runs ComfyUI (not the system Python): torch, transformers, bitsandbytes, accelerate, llama-cpp-python with CUDA (for the GGUF engine), huggingface-hub, psutil. Text-only models that have no `AutoModelForCausalLM` mapping (Qwen3-VL, Qwen2.5-VL) load through the vision class, so text mode works for every catalog family.

Qwen3.5/Qwen3-Next hybrid attention uses reference kernels unless matching optional CUDA extensions are available. The node does not install packages at runtime; on this torch/CUDA build the available `flash-linear-attention` wheels are not recognized by transformers 5.16 and `causal-conv1d` has no matching wheel, so those transformers warnings are expected. For speed on a 16 GB GPU the Model node now selects **q4** for 9B safetensors models, keeps quantized models fully on the GPU, and the GGUF version is faster still.
