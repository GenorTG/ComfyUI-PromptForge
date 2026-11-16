# Legacy line and the new node set

**1.30.0 freezes this node set.** Every node that existed up to 1.30.0 is marked **legacy**: same
behaviour, no new features, bug fixes only. The picker shows them as `PromptForge Legacy · …` and
every description starts with `Legacy node, replaced by …`. Class keys never change, so saved
workflows keep loading and running; the new nodes are registered under new keys next to them.

The reason is configuration, not behaviour. Nodes like `PromptForge Model`, `PromptForge Loader`
and `PromptForge Prompt Enhancer` accept every model family at once, so each workflow has to carry
the right text encoder, VAE, CFG, sampler and template by hand. The new set knows the family:

- one node set per family, so a Krea 2 workflow cannot pick a MiniMax encoder or an ACE VAE;
- defaults that are correct for that family's distilled CFG, sampler, sigma shift and template;
- names that say what family and what job, in a picker that is not one flat list.

## Families and their node sets

Each family gets the same small, predictable shape. Only the nodes that family actually needs exist.

| Family | Nodes |
| --- | --- |
| **Krea 2** (`PROMPTFORGE_KREA_*`) | Model, Prompt, Enhance, Encode, Empty Latent, Sampler, NAG, Edit, Edit Encode, Edit Patch, Decode, Save |
| **Anima** (`PROMPTFORGE_ANIMA_*`) | Model, Prompt, Enhance, Encode (tags + negative), Empty Latent, Sampler, Decode, Save |
| **MOSS Image** (`PROMPTFORGE_MOSS_*`) | Model, Prompt, Enhance, Encode, Edit Encode, Empty Latent, Sampler, Decode, Save |
| **MiniMax H3** (`PROMPTFORGE_H3_*`) | Model, Prompt, Action, Parts, References, Continue, Format, Joins, Sampler, Upscale, Preview, Speed, Decode, Generate, Save Video |
| **ACE-Step** (`PROMPTFORGE_ACE_*`) | Model, Song, Encode, Sampler, Save Audio |
| **Shared** (`PROMPTFORGE_*`) | Prompt Library, Prompt Model (writer), Image/Video/Audio Load, Image Scale, Frame Interpolate, Preview Override, Quantize Model, Quantize Text Encoder |

Rules for the new set:

- Family in the display name: `PromptForge Krea 2 · Sampler`, never a bare `PromptForge Sampler`.
- One job per node, one output. No settings node that only another node understands.
- The family fixes the encoder, VAE and template. The model node takes the checkpoint and the
  LoRA stack; nothing else asks for a CLIP.
- Categories mirror the family: `PromptForge/Krea 2`, `PromptForge/Anima`, `PromptForge/MOSS`,
  `PromptForge/MiniMax H3`, `PromptForge/ACE`, `PromptForge/Shared`.
- Prompt copy is per family and per job (still, edit, tags, video, song), not one blob with switches.
- H3 keeps its parts because long video needs them; the Generate node hides the mode internals the
  old Modes node exposed.

## Migration

- Nothing is removed at once: legacy nodes stay registered until their family set is complete and
  tested, then they are removed in one release with a note in the README.
- New nodes are wired into one example workflow per family before the old ones go.
- The enhancer, model picker, quantization and preset catalogue are reused by the new nodes, not
  copied, so both sets share the same fixes while the transition runs.

## Work order

1. Shared foundation: `legacy.py`, the `PROMPTFORGE_*` category scheme, the family registry.
2. Krea 2 (used most, smallest surface): model, prompt, encode, sampler, decode.
3. Anima, MOSS Image, ACE-Step.
4. MiniMax H3 last: it is the biggest surface and the only one with long-video parts.
5. Remove the legacy set and rewrite the README around the family nodes.
