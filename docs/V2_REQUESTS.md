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
- [ ] Writer node exposes the preset, default `auto (from pipe)`; a manual pick overrides only the writer template, never the pipe.
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
- [ ] Own top-bar system readout (CPU, RAM, VRAM, GPU, disk) built into the pack, no third-party pack needed.
- [ ] Settings toggle to hide it entirely and to choose which meters show, in which order.

## Seeds
- [x] Writer sampling: top_p, top_k, repetition and presence penalty exposed (temperature alone caused gibberish tails).
- [ ] One seed field only: drop the duplicate on the sampler node, keep it on the sampling settings.

## Naming and wiring
- [x] Single-route sampling nodes are titled "Sampling"; first/refine stays only where both passes exist.
- [x] Show Text node added; every `triggers`/`report` output is wired to one.

## Optional / research
- [ ] Look into latent upscale models for anima/krea/moss so a two-stage pass makes sense there too.

## Ongoing
- [x] Finish and test every H3 mode workflow (t2v, i2v, fl2v, l2v, continue, ref2v, multipart t2v, multipart references) and keep the visual pass on each.
