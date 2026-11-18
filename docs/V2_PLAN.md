# PromptForge V2: image and video suite

Status: implementation reference, based on the local source and workflow audit on
2026-11-16. The initial plan has been implemented and refined against live ComfyUI
registration and browser-loaded workflows. See [the shipped node catalog,
templates and verification limits](V2.md). ACE and audio-only workflows remain
out of scope; H3 audio remains part of video generation.

The shipped architecture differs from some early inventory suggestions: shared
image nodes serve Krea 2, MOSS Image and Anima; H3 has native stages for six
modes plus optional refinement, references and multipart continuity. Current
templates are in `example_workflows/v2`; do not treat old proposed node counts or
per-module names below as a second, competing implementation spec.

## Requirements that take precedence over the previous plan

- Keep every existing node key, input schema and execution behavior compatible.
  Keep old workflow files in place. No scheduled removal of legacy nodes.
- Register V2 under new keys, with family and task in their display names.
- New workflows require only PromptForge, a supported ComfyUI installation,
  declared Python dependencies, and the selected model artifacts. No other custom pack.
- Show the order of work. Prompt writing, encoding, sampling, refinement and
  decoding must be independently inspectable stages.
- One clear owner for each setting. Group related settings together in workflows.
- Positive generation text describes the desired result affirmatively. Writer
  exclusions, actual negative conditioning, and diagnostic text stay separate.
- Raw mode preserves user text and does not load or run a prompt writer.
- Publish new blank templates; never copy user prompts, media paths, prompt
  libraries, credentials or embedded reference data from existing workflows.

## Audit findings

The source currently registers the legacy set only. `legacy.py` decorates names
and descriptions; it does not provide replacement implementations.

| Evidence | Consequence for V2 |
| --- | --- |
| `__init__.py` combines writer loading, generation, templates, validation and routes in roughly 3,000 lines | Extract a writer service with small family/task adapters; avoid copying this file per family |
| `unified_loader.py` creates a pipe with family, model filename and folder | The existing pipe is a hint, not an end-to-end state contract |
| Enhancer, sampler and latent nodes consume hints without forwarding a complete pipe | Introduce a new versioned contract; do not reinterpret the legacy socket |
| `h3_generate.py` writes, encodes, samples twice, decodes and joins within one call | Expose these stages for single clips; give long-video orchestration a narrower role |
| `_generate_checked` returns the last rejected attempt on exhaustion | Validation must have an explicit failure result that blocks encoding |
| H3 transition instructions include “No cut in the first shot” | Rewrite instruction sources as affirmative goals as well as validating outputs |
| `moss_nodes.py` resolves Krea edit/grounding/NAG classes through the global node registry | Implement owned equivalents or use verified core APIs; another pack cannot satisfy V2 dependencies |
| Krea workflows contain external GGUF, masking and text-viewer nodes | Bundle required loading/masking/inspection behavior or use core nodes |
| H3 v6 has 42 nodes and five groups, with a central generator receiving many independent settings | Use a sequential configuration chain and dedicated execution stages |
| `H3_QUALITY_NOTES.md` documents a different first-pass scheduler from the generator default | Treat these as named, checkpoint-specific presets, not a universal family default |

Local ComfyUI inspected: commit `63e5bf59`. Its core extras include H3 reference,
latent-upscale and conditioning support and MOSS support. This does not establish
compatibility with every stock ComfyUI release. Record the minimum tested version
when adapters have been exercised in an isolated installation.

Observed workflow scope: Krea T2I, Krea identity edit, Krea transfer, MOSS T2I,
MOSS edit, H3 single/multipart video. Anima is named in the previous design but
has no saved workflow in the inspected main workflow directory. Other entries
in `families.py` describe detection coverage, not completed workflow support.

## Architecture and ownership

Use `v2/` for new contracts, nodes and family adapters. Keep legacy registration
intact and merge V2 registrations separately after legacy decoration. New frontend
extensions target V2 class keys only. Shared changes must not alter legacy defaults.

Suggested modules: `contracts.py`, `presets.py`, `prompting/`, `shared/`, and
`families/{krea2,moss,anima,h3}/`. Split further only when an actual responsibility
needs its own module. Pure configuration/validation must import without GPU libraries.

Display example: `PromptForge V2 · MiniMax H3 · Encode References`.
Key example: `PROMPTFORGE_V2_H3_ENCODE_REFERENCES`.
Categories: `PromptForge V2/<family>` and `PromptForge V2/Shared`.

### Pipe and normal sockets

Use a new `PROMPTFORGE_V2_PIPE` socket with a schema version. Every stage on the
main chain accepts and returns it; starting configuration nodes can create it.
Use copy-on-write sections so branches never mutate their upstream configuration.

The pipe contains family/task identity, resolved configuration, prompt state,
artifact references, stage status and provenance. Separate sections own `models`,
`loras`, `prompt`, `format`, `sampling.first`, `sampling.second`, `performance`,
`references`, `continuity`, `decode`, and `output`. Do not serialize live model
objects or tensors into presets or saved run metadata.

Execution adapters receive the narrow fields they need. MODEL/CLIP/VAE,
CONDITIONING/LATENT/IMAGE/AUDIO/MASK remain available as standard sockets at
stage boundaries for interoperability and debugging. A model bundle can travel
in the pipe to reduce repeated loader wiring; large outputs are referenced, not copied.

Each node declares sections it owns and prerequisites it consumes. Missing
required artifacts, incompatible families/tasks and stale artifacts fail with a
message naming the stage and remedy. Updating a prompt invalidates downstream
conditioning; changing dimensions invalidates latents. Preserve unrelated sections.

No hidden frontend-only defaults: queued/API execution resolves the same values.
Value precedence: family baseline, selected preset, node-local values. Preset
application updates widgets before queueing; execution uses the resulting named
inputs. Connected inputs win and are shown as connected in the preset preview.

### Preset controller

A shared Presets node sits anywhere along the pipe and returns its input unchanged
during execution. Before queueing, its frontend controls traverse upstream pipe
connections and provide Capture, Preview changes, Apply, Restore and family/task
filtering for connected V2 nodes. Unrelated nodes elsewhere in the workflow are
excluded. At branches, collect reachable upstream participants once and report
ambiguous roles. Identify participants by stable per-workflow role identifiers,
not only numeric node IDs. Capture supported named inputs from V2 nodes; multiple
samplers have distinct first/second roles. Do not capture prompts/media by default.

Apply changes as one undoable frontend transaction. Show missing roles, incompatible
schemas and connected inputs before application. A preset stores versioned JSON;
headless users can materialize the same named-input changes before submitting.
The backend must not mutate nodes upstream during execution. Metadata saving at
the end records effective settings; it is a separate operation from preset control.

## Proposed node inventory

This is the initial concrete inventory: 64 node types across Shared (18), Krea (11),
MOSS (9), Anima (8), and H3 (18). Family adapters share implementations when the
behavior is actually the same. Avoid generating extra wrappers just for symmetry.

### Shared: 18 nodes

| Node | Responsibility |
| --- | --- |
| Workflow Presets | Capture/apply settings across supported V2 nodes before queueing |
| Run Inspect | Show resolved configuration, stage status and provenance; pass pipe through |
| Prompt Inspect | Show raw/final text, validation, token budgets and attempts; pass pipe through |
| Prompt Library | Select/save user text independently of enhancement |
| Writer Model | Select/load the local language/vision writer; separate from diffusion models |
| Load Image | Image/mask input and source metadata |
| Load Video | Frames, rate, timing and optional synchronized audio |
| Load Audio | Reference audio for video; no audio-only generation suite |
| Image Scale | Resize/crop with dimensions and crop semantics reported |
| Mask Prepare | Manual/input masks, feather, expand, combine; required core edit path |
| Subject Mask | Owned automatic subject/face masking with explicit required weights |
| Image Composite | Blend generated/source regions using a mask |
| Frame Interpolate | Interpolate while preserving duration and updating rate |
| Preview Settings | Configure live preview without changing saved generation results |
| Performance Settings | Memory/offload controls; family adapters expose applicable choices |
| Quantize Model | Explicit diffusion-weight conversion task |
| Quantize Text Encoder | Explicit encoder conversion task |
| Save Run Metadata | Save effective settings and provenance with optional text inclusion |

Subject Mask requires an implementation/license/model-artifact audit before choosing
a backend. Manual masks alone do not fulfill the existing automatic face-transfer use
case. Do not label transfer complete until its masking and compositing are owned.

### Krea 2: 11 nodes

Model, LoRA Stack, Prompt, Enhance, Format, Sampling Settings, Encode, Edit Prepare,
Sample, Decode, Save Image.

- Model owns diffusion checkpoint, encoder and VAE selection/compatibility.
- LoRA Stack owns ordered toggles, strengths, seeds, trigger suggestions and report.
- Prompt owns raw text, action/idea selection and raw/enhance route.
- Enhance exposes only Krea still/edit writer controls; writes validated prompt state.
- Format owns aspect/dimensions and target/reference sizing.
- Sampling Settings owns steps, CFG, schedule, denoise, shift and optional NAG.
- Encode produces conditioning; edit encoding uses the prepared references.
- Edit Prepare owns identity/reference patching, source latent preparation, reference
  placement and mask strength. It does not write prompts or sample.
- Sample owns execution only, consuming resolved sampling configuration.
- Decode and Save Image own conversion and persistence respectively.

Keep NAG and edit model patching as separately testable internal adapters; expose
additional nodes only if independent reuse warrants them. Preserve two-reference
transfer, mask controls and effective reference placement when NAG is enabled.

### MOSS Image: 9 nodes

Model, LoRA Stack, Prompt, Enhance, Format, Sampling Settings, Encode, Sample,
Decode. Saving uses the core `SaveImage` node.

Encode supports explicit still/edit tasks and reference latents. Keep the edit
reference distinct from the target noise latent. Sampling Settings exposes MOSS
guidance/skip-layer options and checkpoint-specific DMD presets. Saving uses core
SaveImage until a distinct MOSS persistence responsibility exists.

### Anima: 8 nodes

Model, LoRA Stack, Prompt, Enhance, Format, Encode, Sample, Decode.

Prompt supports affirmative tags and separate negative conditioning. Sample owns
its sampling widgets until a real two-pass use case needs a separate settings node.
Enhance uses tag-aware validation. Saving uses core SaveImage. Confirm exact encoder
and sampling behavior against local artifacts before declaring Anima supported.

### MiniMax H3: 18 nodes

| Node | Responsibility |
| --- | --- |
| Model | Diffusion, text encoder, video/audio VAE and optional detail decoder |
| LoRA Stack | Ordered model/video/audio scope, strengths, seeded choices and report |
| Prompt | Raw/idea/action selection; raw mode bypasses writer execution |
| Action | Structured subjects, action and camera intent; no sampling controls |
| Enhance | Mode-specific affirmative writing, repair, validation and review output |
| Format | Aspect, pixels, requested duration, native FPS and valid frame grid |
| References | Mode, first/last frames, labeled images/video/audio and role validation |
| Parts | Ordered part intents, durations and seed policy |
| Continuity | Carry-forward context, source continuation, join/transition policy |
| Sampling Settings | First and optional second pass, sigma shift and seed policy |
| Encode | Mode-specific conditioning and initial audiovisual latents |
| Sample | First-pass audiovisual sampling only |
| Latent Upscale | Spatial latent upscale; no implicit resampling or decoding |
| Refine | Optional second sampling pass; explicit start sigma and conditioning |
| Decode | Tiled video/audio decode, optional detail decoder and stage choice |
| Join | Assemble clips/audio using explicit overlap and transition settings |
| Sequence | Execute a multipart plan through the same tested stage services |
| Save Video | Mux and persist with codec/rate/quality controls and metadata |

Encode uses one mode selector via References, with only relevant ports/widgets
shown. Start with dedicated workflow templates for each mode, rather than five
near-identical encode node types. H3 mode names map explicitly to T2VA, I2VA,
L2VA, FL2VA and Ref2VA model capabilities; reject unsupported combinations early.

Sequence is the justified orchestration exception: future parts can depend on
the generated tail of the previous part. It consumes a compiled, inspected plan
and reuses explicit stage services, reports each part/stage, supports interruption
and resumable outputs, and never secretly invents prompts or sampling settings.
Single-clip templates do not use Sequence. Prototype native graph expansion before
choosing it; use an owned sequential executor only if expansion cannot express
the required dynamic continuity and review behavior reliably.

## Prompt correctness contract

Keep separate fields for user intent, writer exclusions, model-negative text,
structured reference/dialogue data, final positive text, and diagnostics.

1. Compile affirmative family/task instructions. Examples: “the entire face is
   visible in the frame” and “one continuous shot” instead of anti-artifact lists.
2. Generate with a recorded seed and family/task token budget.
3. Validate format, reference labels, repetition, reasoning leakage, instruction
   leakage and negative directives in descriptive positive fields.
4. Repair with specific feedback and a bounded attempt count; validate again.
5. If still invalid, block encoding and expose the attempted output and reasons.
   An explicitly selected raw route remains available; never silently substitute it.

Do not delete every occurrence of “no”, “not” or “without”. Quoted dialogue, song
or sign text inside video, proper names, and model-required sentinel fields need
context-aware handling. Do not silently rewrite raw user text. In raw mode show
lint findings, preserving the requested bypass behavior. Separate model-negative
conditioning is used only where the family/guidance method actually consumes it.

Prompt Inspect shows exact text that will be encoded, actual tokenizer counts
where available, budget, attempts, seed, reference mapping and validation status.
H3 inspection exposes each structured section. Review/edit/revalidate must happen
before expensive sampling; API workflows have an explicit noninteractive policy.

## Workflow catalogue and layout

Create `workflows/v2/` with blank editable UI JSON, keyed-input API JSON and a
small artifact/compatibility manifest for each supported template.

Initial catalogue: Krea T2I, identity edit, two-image transfer; MOSS T2I and edit;
Anima T2I; H3 T2VA, I2VA, L2VA, FL2VA, Ref2VA, two-pass Ref2VA, multipart
continuous video, multipart shot plan and continue-existing-video. Fifteen templates.
Raw/idea/action are supported routes inside applicable templates; they do not
multiply the catalogue. Single-pass/refinement stages are explicit and labeled.

Layout groups progress left to right, with two or three rows to control width:

```text
01 Models & LoRAs        02 Intent & References       03 Prompt & Review
                        04 Format & Sampling         05 Encode & Sample
                                                     06 Refine & Decode
                                                     07 Inspect & Save
```

The pipe links the responsible configuration nodes in order; media/reference
wires stay close to their consumers. No unused reference loaders in a T2VA file,
no distant scalar settings, no crossing a whole graph for debug text. Avoid hidden
wire extensions or frontend conveniences supplied by external packs.

## Delivery phases and acceptance

1. **Inventory and contracts:** machine-readable node/workflow dependency report,
   settings ownership, new-key convention, immutable pipe, artifact invalidation,
   preset schema and prompt failure contract. No legacy execution changes.
2. **First complete image path:** Krea T2I with raw/enhanced routes, inspection,
   owned NAG, clean loading, saved image and blank UI/API templates. Exercise pipe
   branching, deterministic seeds and preset changes end to end.
3. **Image editing:** owned Krea edit/grounding/reference patching, automatic masks,
   transfer/composite, then MOSS T2I/edit and Anima. Verify all formerly external
   behavior rather than replacing it with a non-equivalent fallback.
4. **H3 single clip:** prioritize the five modes, optional writer, reference labels,
   inspectable first pass, refinement and decode, synchronized sound and saving.
5. **H3 sequences:** parts, continuation, joins, review/resume, cancellation and
   per-stage memory lifecycle; preserve the documented two-pass quality preset.
6. **Release verification:** isolated ComfyUI with only this pack, node registration,
   required artifact diagnostics, real smoke generations, blank template audit,
   frontend preset undo/redo and API parity. Record hardware and model revisions.

Run meaningful CPU contract tests for pipe branch isolation, preset precedence,
failure propagation, route bypass and workflow links. Test prompt fixtures that
reproduce “no cut off face”, negative instructions nested in H3 fields, quoted
dialogue, reference labels, exhausted repairs and budget overflow. Real model
quality needs fixed-seed comparisons and visual inspection; text lint alone does
not establish anatomy, identity, motion or sound quality.

Capability support is earned per task, not inferred from a filename. Optional
writer/quantization dependencies must not prevent raw generation nodes registering.
Audit licenses before adapting external algorithms and preserve required notices.
Record model dependencies explicitly; standalone does not mean bundling weights.

## Remaining design questions

- Confirm whether families beyond Krea/MOSS/H3/Anima are actively used before
  adding dedicated suites. Keep existing broad legacy support intact.
- Choose the automatic masking implementation after inspecting available source,
  licenses and required weights; do not promise equivalence prematurely.
- Validate the multipart executor choice with an actual continuity prototype.
- The 64-node inventory is a proposal to review; task coverage and comprehensible
  controls matter more than maintaining an arbitrary node count.
