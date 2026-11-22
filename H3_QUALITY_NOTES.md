# MiniMax H3 quality and memory notes

These settings are what the working two-pass workflow uses. They were isolated by A/B runs on the same
checkpoint, seed and prompt.

## Keep

- First pass: `euler` / `beta57`, 8 steps, cfg 1, shifts `0 / 0`.
- Second pass: fp32 `minimax_h3_latent_upscaler_3d_conv_v1_fp32.pth`, 1 MP, 3 steps, start sigma 0.9035, `er_sde`.
- Performance: `comfy kitchen attention`, `ffn_chunks` 2, sparse attention off.
- Save: NVENC `slow`, quality, CQ 19, 8-bit, 48 fps.
- Loader: `quality`, explicit `minimax` clip type.
- LoRAs: free to toggle. The reference stack is Realism 1.5, Age -0.5, Flat Chested 0.75,
  Ref2VA Motion 0.75, GalaxyAce 1.0, Moawxx audio 1.0, SexGod 1.0.

## What makes it worse

| Setting | Effect |
|---|---|
| `er_sde` + `simple` | Coloured chroma speckle over skin and flat surfaces |
| fp16 latent upscaler | Washed contrast, low skin texture, plastic look |
| sparse attention on | Smeared fine detail |
| `ffn_chunks` 1 | Second-stage out-of-memory on 16 GB |
| NVENC fastest/fast/balanced | Bands on gradients and flat walls |
| shift 12/3 at 8 steps | Harsher, less stable result |

## Memory

More enabled LoRAs use more VRAM. If the second stage OOMs, raise `ffn_chunks` to 3 or 4. Do not turn on
sparse attention as the first fix; it costs detail. The node also unloads everything between the first and
second pass and decodes the first pass only after both sampling passes.

## Headless

Use `PromptForge_H3_AllInOne_Modes_v5.api.json`. It is already in `/prompt` format. The normal workflow
file stores the image picker as an extra leading widget on Load Image; order-based converters that ignore
that read the filename as the description and send no picture.

## Text writer sampling (Qwen3-VL)

Use the model card values: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.0-1.5.
Thinking mode would be 0.6 / 0.95 / 20 / 1.0, but the writer runs with thinking off.

Bad: temperature near 0 or greedy, no_repeat_ngram_size 4 forced on, repetition penalty above 1.15, top_p above
0.95 for long structured prompts, and stopping the sampler on the first repeated phrase. Those cause the
"starts fine, then rambles and never ends" failure.

Presence 1.5 is the Qwen serving value, but HF counts the whole prompt as "seen", so on very long prompts it
can be too strong. 1.0 is the default here; drop to 0.5 or 0 if the prose starts sounding forced.
