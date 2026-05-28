# Deep Interview Spec: Remote Initial Segmentation + Local FoundationPose Tracking

## Metadata
- Profile: standard
- Context type: brownfield
- Final ambiguity: 0.12
- Threshold: 0.20
- Context snapshot: `.omx/context/remote-init-local-tracking-20260528T135451Z.md`
- Transcript: `.omx/interviews/remote-init-local-tracking-20260528T141407Z.md`

## Intent
The operator wants to avoid the suspected local SAM3 memory spike at 1280x720 while also avoiding the latency of full remote tracking, where every RGBD frame is sent over the network. The desired compromise is to use the server only for initialization segmentation and keep per-frame FoundationPose tracking local.

## Desired Outcome
Add a hybrid live tracking path:

1. The server computes the object mask for the first initialization frame.
2. Local FoundationPose uses that mask to initialize/register the object.
3. Subsequent tracking frames run locally without per-frame server calls.
4. Pressing manual reinit (`R`) asks the server for a fresh segmentation mask and then reinitializes locally.
5. The GUI exposes this as a separate `Track Hybrid` button or option, and the same mode is available from CLI.
6. The mode logs timing and VRAM usage by stage so OOM causes are visible.

## In Scope
- Add CLI support to `visual_servoing.scripts.fp_track_live` for hybrid tracking.
- Add GUI support in `gui_app.py` for a `Track Hybrid` button or equivalent option.
- Use the existing v2 server segmentation capability as the remote mask source where possible.
- Send only the data required for server segmentation during initialization or manual reinit.
- Feed the returned mask into local FoundationPose initialization.
- Preserve local tracking after initialization without calling the server each frame.
- Add manual `R` handling so reinitialization requests a new server mask.
- Add stage-level timing and VRAM logging around:
  - frame capture/preprocess
  - remote segmentation request
  - local FoundationPose initialization/register
  - local FoundationPose track step
- Add tests or smoke coverage for command construction, mode selection, and no-per-frame-server-call behavior where feasible.

## Out Of Scope / Non-goals
- Do not change v1 client/server.
- Do not optimize full remote tracking transport or serialization.
- Do not change FoundationPose internals to reduce memory use.
- Do not add server-backed auto-reinit in the first pass.
- Do not redesign the GUI beyond adding the hybrid entrypoint/status text.
- Do not remove or alter existing Track Local and Track Remote behavior.
- Do not add new dependencies unless unavoidable and explicitly justified.

## Decision Boundaries
OMX may decide without further confirmation:
- Exact CLI flag names for the hybrid mode and remote segmentation server address.
- Exact GUI placement and label for `Track Hybrid`.
- Exact internal class/function structure for a remote mask provider.
- Exact log field names, as long as timing and VRAM stage attribution are visible.
- Whether to reuse `/foundationpose/v2/segmentation` directly or add a narrow helper endpoint if the existing endpoint cannot return a mask suitable for initialization.
- Error handling text for server unavailable, no mask, timeout, or local OOM.

OMX should ask before:
- Changing the meaning of existing Track Local or Track Remote buttons.
- Enabling server calls on automatic reinit.
- Downscaling/cropping 1280x720 as part of this feature.
- Adding a persistent cache protocol for masks.
- Modifying FoundationPose model internals.

## Constraints
- Keep the first pass small and reversible.
- Preserve current local and remote tracking commands.
- The hybrid mode should not send RGBD frames to the server every frame.
- Manual `R` is the only server-backed reinit path in the first pass.
- If local FoundationPose register or track still OOMs, the logs must identify the stage rather than hiding the failure.
- The default ZED tracking behavior should remain compatible with the existing GUI and CLI options.

## Acceptance Criteria
- CLI can launch hybrid mode with a remote segmentation server and local FoundationPose tracking.
- GUI can launch the same mode via `Track Hybrid` or an equivalent clearly separated control.
- On startup, the hybrid mode obtains a server segmentation mask and initializes local FoundationPose from that mask.
- During normal tracking after initialization, no per-frame server segmentation or remote tracking call is made.
- Pressing `R` requests a new server mask and reinitializes locally.
- Existing Track Local and Track Remote still launch their previous paths.
- Logs include stage-level timing and VRAM/memory information sufficient to distinguish:
  - remote segmentation latency
  - local initialization/register cost
  - local per-frame tracking cost
  - the stage where OOM occurs, if it still occurs
- Server unavailable, no-mask, and timeout cases report clear errors instead of silently falling back to local SAM3.
- Existing relevant tests pass, and new targeted tests cover command/mode wiring where practical.

## Assumptions Exposed And Resolved
- Assumption: 1280x720 local OOM is caused by SAM3.
  - Resolution: Treat as a hypothesis. Hybrid mode must include diagnostics because FoundationPose register/track may still be the failing stage.
- Assumption: server segmentation should be part of automatic reinitialization.
  - Resolution: Excluded for the first pass. Only first frame and manual `R` use the server.
- Assumption: a GUI-only feature is enough.
  - Resolution: Add both GUI and CLI entrypoints so the mode is testable and scriptable.

## Technical Context Findings
- `visual_servoing/scripts/fp_track_live.py` has an existing `--init-mask` path through `PrecomputedMaskProvider`.
- `visual_servoing/foundationpose_model_free/tracker.py` supports explicit masks in `process_frame()`.
- Local tracking after initialization does not need SAM3 every frame.
- `visual_servoing/visual_servo_server_v2.py` already exposes v2 server functionality including segmentation-related flows.
- `visual_servoing/foundationpose_model_free/gui_app.py` already launches local and remote tracking commands and is the natural GUI integration point.

## Recommended Handoff
Proceed with:

```text
$ralplan .omx/specs/deep-interview-remote-init-local-tracking.md
```

Use the resulting plan with `$ralph` or direct execution. Do not reopen the requirement interview unless the hybrid mode scope expands beyond the boundaries above.
