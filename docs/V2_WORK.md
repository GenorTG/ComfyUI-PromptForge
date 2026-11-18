# V2 completion work

Scope: make every V2 node stand on its own, look and behave like the legacy node it
replaces, and stop depending on hidden constants. Code first, tests only after the
whole set is converted.

## Rules

- The pipe is optional everywhere. Real sockets carry the work; the pipe only chains
  state (presets, metadata, inspect) and is merged when connected.
- No hidden behaviour: anything that can be switched is a widget with a default, not a
  hardcoded branch in `execute()`. Family tables stay as defaults, always overridable.
- Every node that shows something gets the space to show it: viewers, previews and
  lists set `getMinHeight` for their DOM panel and a matching canvas size.
- Workflows are laid out by size: no overlapping boxes, groups in columns, spacing from
  each node's own minimum size.

## Order

1. **Pipe-optional foundation** (`contracts.ensure_pipe`, family from the model artifact,
   per-node settings fallback to widgets). Every schema flips `pipe` to optional and every
   `execute` starts from `ensure_pipe`.
2. **Node-by-node quality pass**, family by family, in this order: shared, media, masking,
   prompting, image, H3, performance.
   - restore what legacy had: vision analysis, tags, NSFW blur, description save on the
     image loader; ideas/saved-prompt bar with save, load, edit, delete; preview panels on
     samplers, decoders and savers; result viewer on the saver.
   - restore per-node settings that V2 dropped (step cache, sampler presets, batch counts,
     decode output device, preview quality, and the rest of the legacy widget sets).
   - sizes: LoRA stack and library lists wide and tall; loader/picker/preview nodes sized
     for the asset; long combo values wrapped.
3. **Workflow layout**: rewrite the generator to place nodes by measured size per group
   column, then regenerate every workflow in the user folder.
4. **Docs**: node reference matches the restored widgets.
5. **Tests**: schema audit, instantiation probe, workflow load probe, then live
   generations per family and mode with visual review.

## Acceptance

- Every node executes with no pipe connected (model, encoder, sampler, decoder, saver at
  minimum), and the pipe path still works.
- No node has a size below the widget/preview it owns; the overlap check on every
  workflow is clean.
- Legacy parity list from `docs/V2.md` is satisfied or a widget explains its absence.
