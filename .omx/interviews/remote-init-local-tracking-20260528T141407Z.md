# Deep Interview Transcript: Remote Initial Segmentation + Local Tracking

## Metadata
- Profile: standard
- Context type: brownfield
- Final ambiguity: 0.12
- Threshold: 0.20
- Context snapshot: `.omx/context/remote-init-local-tracking-20260528T135451Z.md`

## Clarified Outcome
Add a hybrid FoundationPose live tracking mode where segmentation for initialization is computed on the server, but FoundationPose tracking runs locally after initialization. The first pass should also add stage-by-stage timing and VRAM diagnostics so 1280x720 local OOM failures can be attributed to the correct stage.

## Resolved Decisions
- Mode shape: server segmentation for initialization, local FoundationPose tracking afterward.
- Server segmentation calls: first frame and manual `R` reinitialization only.
- Auto reinitialization: do not route through the server in the first pass.
- Success criteria: 1280x720 should avoid the local SAM3 OOM path when possible, and logs must show timing/VRAM by stage.
- Entrypoints: add a GUI `Track Hybrid` button or option and matching CLI flags.
- Existing modes: preserve current Track Local and Track Remote behavior.
- Non-goals: remote tracking optimization, FoundationPose internal OOM optimization, auto server reinit, and GUI redesign are all excluded.

## Transcript

### Round 1
Question: At 1280x720 local, where exactly does OOM happen?

Answer: Logs cannot distinguish the stage.

### Round 2
Question: Should the first stage prioritize diagnostics, hybrid mode, or both together?

Answer: Both together: add the hybrid mode and stage-by-stage VRAM/timing logs.

### Round 3
Question: When should server segmentation be called in the hybrid mode?

Answer: First frame and manual `R` only.

### Round 4
Question: What is the first-pass success criterion?

Answer: It should run without OOM if SAM3 initialization was the problem, and it must log stage-level VRAM/timing either way.

### Round 5
Question: Which items are explicitly out of scope?

Answer: Exclude auto server reinit, remote tracking optimization, FoundationPose internal OOM optimization, and GUI redesign.

### Round 6
Question: Where should this hybrid mode be executable from?

Answer: GUI `Track Hybrid` button or option plus CLI flags.

## Pressure Pass Finding
The initial idea assumed that remote segmentation would solve 1280x720 local OOM. The interview kept that as a hypothesis, not a fact. The implementation must therefore log VRAM/timing around segmentation, registration, and tracking stages. If local tracking or registration still OOMs after server-side initialization, the mode is still useful only as a diagnostic and does not satisfy the full success criterion.

## Brownfield Evidence
- `fp_track_live.py` already supports `--init-mask` through `PrecomputedMaskProvider`, which proves the local tracker can initialize from a supplied mask.
- `FoundationPoseLiveTracker.process_frame()` accepts an explicit mask and uses segmentation only during initialization/reinitialization.
- Once initialized, local tracking calls FoundationPose tracking without invoking SAM3.
- The v2 server already has a segmentation endpoint used by segmentation sanity checks, but it is not wired as a tracking initializer.
- The GUI already has separate local/remote tracking concepts, so adding a distinct hybrid entrypoint preserves existing workflows.
