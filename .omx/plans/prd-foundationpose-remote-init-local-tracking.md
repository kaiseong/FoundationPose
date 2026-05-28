# PRD: FoundationPose Remote Initial Segmentation + Local Tracking

## Metadata
- Source spec: `.omx/specs/deep-interview-remote-init-local-tracking.md`
- Context snapshot: `.omx/context/remote-init-local-tracking-20260528T135451Z.md`
- Planning mode: ralplan consensus, short RALPLAN-DR
- Status: approved by Architect and Critic

## Requirements Summary
Add a third FoundationPose live tracking mode, `Track Hybrid`, where the server produces SAM3 segmentation masks only for initialization and manual reinitialization while local FoundationPose performs registration and subsequent tracking. The mode must be available from both CLI and GUI, preserve current `Track Local` and `Track Remote` behavior, and expose stage-level timing/VRAM diagnostics so 1280x720 OOM failures can be attributed to SAM3, registration, or tracking.

## Current Evidence
- `visual_servoing/scripts/fp_track_live.py:47` already accepts `--init-mask`, and `visual_servoing/scripts/fp_track_live.py:127` selects `PrecomputedMaskProvider` or `Sam3MaskProvider`.
- `visual_servoing/scripts/fp_track_live.py:183` currently calls `tracker.process_frame(...)` without explicit masks during the live loop.
- `visual_servoing/scripts/fp_track_live.py:252` handles manual `R` by calling `tracker.request_reinit()`.
- `visual_servoing/foundationpose_model_free/tracker.py:75` accepts an explicit `mask`, `tracker.py:97` enters registration/reinit, and `tracker.py:121` calls `adapter.track_one(...)` after initialization.
- `visual_servoing/visual_servo_server_v2.py:470` already exposes server-side segmentation and returns `mask_png_b64` at `visual_servoing/visual_servo_server_v2.py:523`.
- `visual_servoing/visual_servo_protocol_v2.py:109` already encodes segmentation RGBD requests.
- `visual_servoing/foundationpose_model_free/gui_app.py:713` and `gui_app.py:714` expose separate `Track Local` and `Track Remote` buttons.
- `visual_servoing/foundationpose_model_free/gui_app.py:2015` has a `remote_segmentation_sanity(...)` helper for posting segmentation requests.

## RALPLAN-DR Summary

### Principles
- Preserve existing behavior: `Track Local`, `Track Remote`, v1 client/server, and current automatic reinit semantics remain unchanged unless the user explicitly expands scope.
- Make server use explicit: hybrid mode calls the server only for initial mask acquisition and manual `R` reinitialization.
- Fail loudly rather than falling back silently: if the remote mask request fails, do not hide the failure by using local SAM3 in hybrid mode.
- Diagnose the real bottleneck: stage-level timing and CUDA memory logs must distinguish segmentation, registration, and tracking.
- Keep the first pass reversible: prefer a small client-side provider and GUI launch wiring over protocol or FoundationPose internals changes.

### Decision Drivers
- Avoid local SAM3 OOM at 1280x720 without sending every frame to the server.
- Keep frame-to-frame tracking latency closer to local tracking than remote tracking.
- Preserve a clear operator workflow in the existing GUI.

### Viable Options

#### Option A: RemoteSegmentationMaskProvider inside local tracking
Approach: add a `MaskProvider` implementation that posts the current RGBD frame to `/foundationpose/v2/segmentation`, decodes `mask_png_b64`, and returns a `MaskResult`. `fp_track_live.py` uses it instead of local SAM3 in hybrid mode.

Pros:
- Fits the existing `FoundationPoseLiveTracker` contract with minimal changes.
- Server calls naturally occur only when the tracker needs a mask.
- Manual `R` can reuse the existing reinit flow.
- Does not mix full remote tracking and local tracking responsibilities.

Cons:
- Requires careful error handling so automatic reinit does not call local SAM3.
- Requires new diagnostics around provider/register/track stages.

#### Option B: Fetch one temporary mask file and pass `--init-mask`
Approach: obtain a remote mask before launching local tracking and write it to a temporary file for the existing `PrecomputedMaskProvider`.

Pros:
- Smallest initial integration.
- Reuses current `--init-mask` path.

Cons:
- Does not support manual `R` cleanly without restarting or rewriting the file.
- Does not help distinguish runtime reinit behavior.
- Encourages a one-off path instead of a reusable provider.

#### Option C: Extend `visual_servo_client_v2` into a hybrid client
Approach: add hybrid behavior to the existing remote tracking client.

Pros:
- Reuses remote client command-line structure and URL parsing.
- Keeps remote-related transport in one executable.

Cons:
- Blurs the separation between full remote tracking and local tracking.
- Higher risk of accidentally sending per-frame requests.
- More likely to affect current Track Remote behavior.

### Decision
Choose Option A. Implement a remote mask provider used by `fp_track_live.py` in a distinct hybrid mode, then add GUI command wiring for `Track Hybrid`.

## In Scope
- CLI flags in `visual_servoing/scripts/fp_track_live.py` for remote initialization masks.
- A reusable remote segmentation mask provider or helper, preferably near existing mask-provider code or live tracking code.
- Decoding and validation of server-returned `mask_png_b64` masks.
- GUI command-builder and button wiring for `Track Hybrid`.
- Stage-level timing and CUDA memory diagnostics for camera read, remote mask request, registration, tracking, and display/overlay.
- Tests for provider selection, remote mask request construction/decoding, GUI wiring, and no per-frame server calls after initialization.
- Hybrid-mode recovery must force `auto_reinit=False` regardless of GUI checkbox or CLI `--auto-reinit`; the runtime should report that automatic reinit is disabled because server-backed auto reinit is out of scope.

## Out Of Scope
- No v1 client/server changes.
- No per-frame remote tracking optimization.
- No FoundationPose internals rewrite.
- No server-backed automatic reinit in the first pass.
- No downscaling/cropping fallback for 1280x720.
- No GUI redesign beyond the new tracking entrypoint and concise status/log text.
- No new dependency unless absolutely required.

## Implementation Plan

1. Add a remote segmentation mask provider.
   - Create a small provider that implements `get_mask(image_rgb, depth_m, object_name)`.
   - Use `encode_foundationpose_segmentation_request(...)` from `visual_servoing/visual_servo_protocol_v2.py:109`.
   - POST to `/foundationpose/v2/segmentation`, decode JSON response, decode `mask_png_b64`, validate mask shape, and return `MaskResult(source="remote_segmentation", metadata=...)`.
   - Surface server failures, no-mask responses, timeout, and bad shape as explicit errors.

2. Wire hybrid mode into `fp_track_live.py`.
   - Add flags such as `--remote-init-mask-server`, `--remote-init-mask-timeout-s`, `--remote-init-mask-device`, and `--remote-init-mask-resolution`.
   - Provider selection order:
     - `--init-mask` keeps using `PrecomputedMaskProvider`.
     - `--remote-init-mask-server` uses the remote provider.
     - otherwise local `Sam3MaskProvider`.
   - In hybrid mode, do not silently create a local SAM3 fallback provider.
   - In hybrid mode, force `TrackingRecoveryConfig(auto_reinit=False, ...)` even when `--auto-reinit` is passed. Manual `R` remains supported because `request_reinit()` is separate from auto reinit in `tracker.py:188`.
   - Print or include a clear status/metadata note such as `hybrid_auto_reinit_disabled=true` when a user requested auto reinit in hybrid mode.
   - Preserve the existing manual `R` path at `visual_servoing/scripts/fp_track_live.py:252`; the tracker should request a fresh provider mask on the next reinit frame.
   - Extract parser construction and provider/recovery-config selection into testable helper functions rather than keeping all branching inline in `main()` at `fp_track_live.py:40`.

3. Add stage-level diagnostics.
   - Measure remote segmentation request latency inside the provider and expose it via `MaskResult.metadata`.
   - Split `tracker.process_frame(...)` metadata to distinguish registration and tracking time, or add instrumentation at the adapter boundary.
   - Promote `remote_segmentation_ms`, `register_ms`, `track_one_ms`, and CUDA memory snapshots into the top-level `timing_ms` mapping consumed by `draw_status_overlay(...)` at `visual_servoing/point_pose/overlay.py:121` and `emit_json(...)` in `fp_track_live.py`.
   - Update `_timing_summary(...)` at `visual_servoing/point_pose/overlay.py:172` so the new stage fields can be seen in the CV2 window.
   - Add optional CUDA memory snapshots using guarded `torch.cuda` imports so CPU-only tests do not require torch CUDA.
   - Ensure `--print-timing` and JSON output include the stage data.

4. Add GUI entrypoint.
   - Add `Track Hybrid` next to `Track Local` and `Track Remote` in `visual_servoing/foundationpose_model_free/gui_app.py:713`.
   - Add a `GuiCommandBuilder.track_hybrid_live(...)` method based on `track_live(...)` at `gui_app.py:333`.
   - Include current server host/port, prompt, SAM settings, ZED settings, iterations, data root, and FoundationPose root.
   - Reuse `_download_remote_model_for_local_tracking(...)` if local `model.obj` is missing, because tracking remains local.
   - Set `_last_tracking_mode = "hybrid"` and update `reinitialize_tracking_event` so it restarts hybrid mode when appropriate.

5. Preserve server API and tests.
   - Reuse `FoundationPoseV2Service.segment(...)` at `visual_servoing/visual_servo_server_v2.py:470` unless a narrow helper is strictly necessary.
   - Do not alter `/foundationpose/v2/track` behavior.
   - Keep segmentation threshold and SAM options explicit in the request metadata.

## Acceptance Criteria
- CLI starts hybrid mode against a remote segmentation server and local profile mesh.
- GUI exposes `Track Hybrid` without changing `Track Local` or `Track Remote`.
- On the first frame, hybrid mode obtains a server mask and local registration succeeds or fails with a clear stage-attributed message.
- During normal tracking frames after initialization, no server request is made.
- Pressing `R` causes exactly one fresh remote segmentation request for the next reinitialization.
- Hybrid mode does not silently fall back to local SAM3 when the remote server fails.
- Hybrid mode ignores/disables automatic reinit and reports that server-backed auto reinit is excluded from the first pass.
- Timing/diagnostic output identifies at least remote segmentation latency, registration time, tracking time, and frame total time.
- The CV2 overlay timing line and JSON output can show the new stage timings, not just coarse `pose_estimation_ms`.
- CUDA memory fields are present when CUDA is available and absent or clearly marked unavailable otherwise.
- Existing local/remote tracking commands generated by the GUI remain unchanged.

## Risks And Mitigations
- Risk: local FoundationPose registration or tracking, not SAM3, is the real OOM source.
  - Mitigation: add stage-level VRAM logs before claiming the hybrid mode fixes OOM.
- Risk: automatic reinit accidentally triggers local SAM3 in hybrid mode.
  - Mitigation: force `auto_reinit=False` in hybrid mode, explicitly report that behavior, and test hybrid provider call counts.
- Risk: remote segmentation response shape or encoding mismatch breaks initialization.
  - Mitigation: validate shape and add provider decode tests using small synthetic masks.
- Risk: GUI restart/reinit state becomes ambiguous with three tracking modes.
  - Mitigation: update `_last_tracking_mode` handling and add source-inspection/unit tests.
- Risk: server endpoint changes destabilize existing debug/segmentation sanity flows.
  - Mitigation: reuse existing endpoint and keep server changes minimal.

## ADR

### Decision
Implement hybrid tracking as local `fp_track_live.py` plus a remote segmentation mask provider, exposed through both CLI and GUI.

### Drivers
- Lower per-frame latency than full remote tracking.
- Lower local initialization memory pressure than local SAM3.
- Minimal disruption to current local/remote modes.

### Alternatives Considered
- Temporary precomputed mask file: rejected because manual `R` support is weak.
- Extending full remote tracking client: rejected because it mixes responsibilities and increases regression risk.

### Why Chosen
The existing tracker already separates mask acquisition from tracking. A `MaskProvider` is the smallest integration point that can swap local SAM3 for remote segmentation while leaving FoundationPose tracking local.

### Consequences
- Hybrid still sends one full RGBD frame to the server during init and manual reinit.
- If registration or tracking OOMs locally, hybrid will diagnose but not solve that failure.
- The GUI gains a third mode that must be kept distinct from local and remote tracking.

### Follow-ups
- If hybrid proves stable but latency remains high on reinit, evaluate mask-only/cropped/ROI transport.
- If local registration still OOMs, plan a separate FoundationPose memory-reduction pass.
- If operators need automatic server-backed reinit, plan it separately after first-pass behavior is validated.

## Available Agent Types Roster
- `explore`: codebase lookup and file/symbol mapping.
- `architect`: design review and boundary validation.
- `critic`: plan and risk review.
- `executor`: implementation and refactoring.
- `test-engineer`: targeted test design and verification.
- `verifier`: completion evidence and acceptance validation.
- `code-reviewer`: post-change review.

## Follow-up Staffing Guidance

### Ralph
Use `$ralph .omx/plans/final-foundationpose-remote-init-local-tracking.md` for a single-owner implementation loop.
- Suggested lanes: one `executor` for implementation, one `test-engineer` for targeted tests after the first patch, one `verifier` for final evidence.
- Reasoning: medium for executor, medium for test-engineer, high for verifier.

### Team
Use `$team .omx/plans/final-foundationpose-remote-init-local-tracking.md` if parallel execution is preferred.
- Lane 1 `executor`: remote mask provider and CLI wiring.
- Lane 2 `executor`: GUI command/button wiring.
- Lane 3 `test-engineer`: tests for provider, CLI/GUI command construction, and call-count behavior.
- Lane 4 `verifier`: smoke commands and final acceptance evidence.
- Keep write sets disjoint until integration.

## Launch Hints
```bash
$ralph .omx/plans/final-foundationpose-remote-init-local-tracking.md
$team .omx/plans/final-foundationpose-remote-init-local-tracking.md
```

## Team Verification Path
- Team verifies unit tests for provider, tracker reinit call counts, GUI command construction, and protocol decoding.
- Team runs `py_compile` on touched Python modules.
- Team records any unavailable hardware/GPU validation as an explicit gap.
- A final Ralph/verifier pass should inspect that normal tracking frames do not call the server after initialization.

## Goal-Mode Follow-up Suggestions
- `$ultragoal`: suitable if this should become a durable implementation goal with checkpointed evidence.
- `$performance-goal`: suitable after implementation if the next target is measurable FPS/latency/VRAM improvement.
- `$autoresearch-goal`: not the best fit here because this is implementation delivery, not a research mission.
