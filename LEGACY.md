# Legacy compatibility and the V2 suite

The 1.30.0 node set remains registered under its existing class keys. Existing
workflows stay in place and keep their input schemas and execution behavior.
There is no planned removal of legacy nodes. New work uses separate V2 keys.

The picker labels legacy nodes `PromptForge Legacy · …`. New V2 nodes use new
class keys and do not replace legacy implementations. The V2 suite is implemented
and documented in [docs/V2.md](docs/V2.md).

The architecture decisions and source audit are in [docs/V2_PLAN.md](docs/V2_PLAN.md).
The shipped suite follows these constraints:

- Image and video workflows are in scope; ACE and audio-only workflows stay as is.
- H3 retains synchronized audio as part of video generation.
- New workflows use only PromptForge and supported ComfyUI core nodes.
- Prompt writing, encoding, sampling, refinement and decoding are visible stages.
- Versioned pipe connections carry settings and stage state through the workflow.
- A pipe-connected preset controller captures/restores upstream node settings
  before queueing and passes its input through unchanged during execution.
- New blank workflows are published separately, without modifying old workflows.
- Existing class keys, workflow files, and behavior are preserved; ACE and
  audio-only functionality are unchanged.

Legacy implementation changes require a specific compatibility or bug-fix reason;
the V2 refactor must not silently change old defaults or writer behavior.
