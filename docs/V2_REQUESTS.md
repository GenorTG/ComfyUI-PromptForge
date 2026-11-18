# V2 request backlog

Every item is required. Keep the status current; do not drop one silently.
`[ ]` open · `[~]` in progress · `[x]` done · `[!]` blocked

## LoRA stack node (done)
- [x] Pipette the model family from the incoming pipe and show it on the node (never "unknown" when the loader already named it).
- [x] Picker menu: inner padding, dark background, legible text, clearly its own popup.
- [x] Enable/disable is a tick toggle, not an "Add" button.
- [x] Strength values stay editable while the row is disabled.
- [x] Rows reorder by dragging.
- [x] Info popup (Power Lora Loader parity): only when `comfyui-lora-manager` is present, data from that pack, tabs, fixed size, fits without vertical scroll.

## Sidebar "Prompt Forge"
- [x] Frontend-only sidebar button and panel, independent from the node code.
- [x] Tab: outputs browser, recursive over the whole folder and subfolders.
- [x] Tab: inputs browser, recursive, with classify, vision analysis, NSFW filter toggle and search.
- [x] Tab: workflow templates, curated from the pack, fully dry (no prompts, no input images, no references, no personal data).
- [x] Template picking loads the graph so the user can save it as their own workflow.

## Saver nodes
- [x] Video saver exposes encoder, bitrate, quality and the rest, with a sane preset applied on a fresh drop.
- [x] Both savers support date/time wildcards in folder and filename (`%date:...%` style) so daily folders and timestamps need no manual edits.

## Prompt node
- [x] Add a generation-mode choice (image / minimax) that supersedes, hiding `soundscape` and `music` for image modes.
- [x] Replace the flat action dropdown with a searchable, filterable picker that shows each preset's description and metadata.

## Writer and model loader
- [x] Writer thinking toggle (default off).
- [x] Writer retries: "retry count" off/1/2/3, first passing try kept, nothing else written.
- [x] Writer node exposes the preset, default `auto (from pipe)`; a manual pick overrides only the writer template, never the pipe.
- [x] Model loader filters checkpoint / diffusion lists to the chosen family; `auto` shows every model.
- [x] Model rows get the same info window as loras (sample images, recommended settings, author text) and only when `comfyui-lora-manager` is present.

## Format node
- [x] Expose a sizing mode: preset resolutions per family, generic resolutions, megapixels + aspect, or custom width/height. Only the widgets of the active mode are shown.

## Reports and prompt inspection
- [x] Report panes render formatted (badges, validation, sections, copy button) instead of raw JSON.
- [x] Anima enhancement verified end to end; tag rules, `full body` and no realistic tags fixed the couple body horror.
- [x] Camera is a combo of the family's camera list, not a free text field.

## Workflow test quality
- [x] Multipart references: first pass back to 0.5 MP and overlap 8 so the join stops ghosting.

## Resource readout
- [x] Own top-bar system readout (CPU, RAM, VRAM, GPU, disk) built into the pack, no third-party pack needed.
- [x] Settings toggle to hide it entirely and to choose which meters show, in which order.

## Per-model generation presets
- [x] Model loader offers "apply saved generation settings for this model": writes sampler, scheduler, steps, cfg and clip skip from the preset store for that checkpoint so switching models is one click. The preset store STAYS in its own node (history lives there); the loader only applies. Reason: merging would duplicate storage and bloat the loader, while keeping it separate also serves workflows that never touch the loader.

## Resource bar polish
- [x] Centre the bar, widen the gaps, and give each meter a pill/border so it separates from the background.

## First pass visibility
- [x] The first pass is encoded and saved before the second pass so it survives a crash and can be judged while the refine runs; the final save stays the workflow result asset.

## Seeds
- [x] Writer sampling: top_p, top_k, repetition and presence penalty exposed (temperature alone caused gibberish tails).
- [x] One seed field only: drop the duplicate on the sampler node, keep it on the sampling settings.

## Naming and wiring
- [x] Single-route sampling nodes are titled "Sampling"; first/refine stays only where both passes exist.
- [x] Show Text node added; every `triggers`/`report` output is wired to one.

## Optional / research
- [x] Latent upscale research: image families need a 2D upscaler (SesquiLSR `upscaler_Wan21` for Anima/Krea 2, `upscaler_Flux2` for MOSS); H3's 3D model is 24-channel and cannot be reused. Refine denoise 0.20-0.25, never raise CFG.
- [x] Implement optional two-stage for the image families: 2D latent upscale node + refine stage, settings mirroring the H3 refine section.

## Ongoing
- [x] Finish and test every H3 mode workflow (t2v, i2v, fl2v, l2v, continue, ref2v, multipart t2v, multipart references) and keep the visual pass on each.
