# Final Plan: FoundationPose Remote Initial Segmentation + Local Tracking

## Consensus Status
- Planner iteration: 2
- Architect verdict: APPROVE
- Critic verdict: APPROVE
- Source spec: `.omx/specs/deep-interview-remote-init-local-tracking.md`
- PRD: `.omx/plans/prd-foundationpose-remote-init-local-tracking.md`
- Test spec: `.omx/plans/test-spec-foundationpose-remote-init-local-tracking.md`

## Decision
Implement `Track Hybrid` as a local FoundationPose tracking mode that uses the v2 server only as a remote segmentation mask provider for initialization and manual `R` reinitialization.

## RALPLAN-DR Summary

### Principles
- Preserve `Track Local`, `Track Remote`, and v1 behavior.
- Server calls occur only for initial mask acquisition and manual `R`.
- Hybrid must fail loudly rather than falling back silently to local SAM3.
- Timing/VRAM diagnostics must identify segmentation, registration, and tracking stages.
- Keep the first pass small and reversible.

### Decision Drivers
- Avoid local SAM3 OOM at 1280x720 when SAM3 is the failing stage.
- Avoid full remote tracking latency from per-frame RGBD transfer.
- Keep the operator workflow explicit in the GUI.

### Alternatives
- Temporary precomputed mask file: rejected because manual `R` support is weak.
- Extend full remote tracking client: rejected because it mixes remote/local responsibilities and increases per-frame server-call regression risk.
- Remote `MaskProvider` in local tracking: chosen because it matches the existing tracker boundary.

## Implementation Steps

1. Add remote segmentation mask provider.
   - Implement a `MaskProvider`-compatible provider that posts RGBD data through `encode_foundationpose_segmentation_request(...)`.
   - Decode `mask_png_b64`, validate shape, return `MaskResult(source="remote_segmentation", metadata=...)`.
   - Surface timeout, no-mask, bad-shape, and server-error cases explicitly.

2. Wire `fp_track_live.py`.
   - Add remote mask flags, including server address and timeout/SAM options.
   - Extract parser/provider/recovery selection into testable helpers.
   - Preserve `--init-mask` precedence, then remote provider, then local SAM3.
   - In hybrid mode, force `TrackingRecoveryConfig(auto_reinit=False, ...)` and report that automatic reinit is disabled.
   - Preserve manual `R` through `tracker.request_reinit()`.

3. Add diagnostics.
   - Emit `remote_segmentation_ms`, `register_ms`, `track_one_ms`, `frame_total_ms`.
   - Guard CUDA memory snapshots so CPU-only hosts do not initialize CUDA.
   - Merge stage fields into top-level `timing_ms` before JSON and overlay emission.
   - Update overlay timing summary to show the new stage fields.

4. Add GUI entrypoint.
   - Add `Track Hybrid` beside existing tracking buttons.
   - Add `GuiCommandBuilder.track_hybrid_live(...)`.
   - Use current server host/port, prompt, SAM settings, ZED settings, iterations, data root, and FoundationPose root.
   - Reuse remote model download for local tracking if local `model.obj` is missing.
   - Track `_last_tracking_mode = "hybrid"` for restart/reinit dispatch.

5. Preserve server behavior.
   - Reuse `/foundationpose/v2/segmentation`.
   - Do not alter `/foundationpose/v2/track`.
   - Do not change v1 files.

## Acceptance Criteria
- CLI launches hybrid mode against a remote segmentation server and local FoundationPose mesh.
- GUI launches the same mode through `Track Hybrid`.
- First frame obtains one server mask and initializes locally.
- Normal tracking frames make no server request.
- Manual `R` obtains exactly one fresh remote mask for reinit.
- Hybrid does not silently instantiate local SAM3 fallback after remote failure.
- Hybrid forces/disables automatic reinit and reports that choice.
- JSON and overlay timing expose `remote_segmentation_ms`, `register_ms`, `track_one_ms`, and `frame_total_ms`.
- CUDA memory diagnostics are present when available and safe/absent/unavailable on CPU-only hosts.

## Verification Plan
Run after implementation:

```bash
python -m py_compile \
  visual_servoing/scripts/fp_track_live.py \
  visual_servoing/foundationpose_model_free/mask_provider.py \
  visual_servoing/foundationpose_model_free/gui_app.py \
  visual_servoing/point_pose/overlay.py

PYTHONPATH=/home/kgs/FoundationPose PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
  visual_servoing/tests/foundationpose_model_free/test_mask_provider.py \
  visual_servoing/tests/foundationpose_model_free/test_tracker.py \
  visual_servoing/tests/foundationpose_model_free/test_gui_app.py \
  visual_servoing/tests/foundationpose_model_free/test_track_live_output.py \
  visual_servoing/tests/test_phone_pose_overlay.py \
  visual_servoing/tests/test_visual_servo_server_v2.py \
  -q
```

Manual hardware smoke, when available:

```bash
python -m visual_servoing.scripts.fp_track_live \
  --object multimeter_zed_1 \
  --prompt multimeter \
  --foundationpose-root /home/kgs/FoundationPose \
  --camera zed \
  --width 1280 \
  --height 720 \
  --fps 15 \
  --zed-depth-mode NEURAL \
  --remote-init-mask-server 192.168.0.3:8081 \
  --print-timing
```

## ADR

### Decision
Use Option A: remote segmentation mask provider inside local live tracking.

### Drivers
- Lower local memory pressure than local SAM3 initialization.
- Lower per-frame latency than full remote tracking.
- Minimal regression risk for existing modes.

### Alternatives Considered
- One-shot `--init-mask` file: rejected because it does not support manual `R`.
- Hybrid inside `visual_servo_client_v2`: rejected because it couples full remote and local tracking.

### Why Chosen
`FoundationPoseLiveTracker` already separates mask acquisition from tracking. A remote `MaskProvider` fits this boundary and keeps server calls restricted to reinitialization events.

### Consequences
- Hybrid still sends one full RGBD frame during init/manual reinit.
- If FoundationPose registration/tracking is the real OOM source, this mode diagnoses rather than fixes it.
- GUI has a third tracking mode that must stay semantically distinct.

### Follow-ups
- Evaluate cropped/mask-only transport if reinit latency is still high.
- Plan FoundationPose memory optimization separately if register/track OOMs.
- Plan server-backed automatic reinit separately only after first-pass behavior is validated.

## Available Agent Types Roster
- `explore`: codebase lookup and symbol mapping.
- `architect`: design and boundary review.
- `critic`: plan/test/risk review.
- `executor`: implementation.
- `test-engineer`: targeted tests.
- `verifier`: final evidence and acceptance validation.
- `code-reviewer`: post-change review.

## Follow-up Staffing

### Ralph
```bash
$ralph .omx/plans/final-foundationpose-remote-init-local-tracking.md
```

Recommended staffing: one `executor`, one `test-engineer`, one `verifier`.

### Team
```bash
$team .omx/plans/final-foundationpose-remote-init-local-tracking.md
```

Suggested lanes:
- `executor`: remote mask provider and CLI wiring.
- `executor`: GUI command/button wiring.
- `test-engineer`: provider, CLI/GUI, tracker call-count, overlay/JSON tests.
- `verifier`: py_compile, pytest, diff check, manual smoke gap report.

## Team Verification Path
- Team proves no per-frame server calls after initialization using provider call-count tests.
- Team proves auto reinit is disabled in hybrid.
- Team proves overlay/JSON stage timing visibility.
- Team records ZED/GPU smoke as run or explicitly not run.

## Goal-Mode Follow-up Suggestions
- `$ultragoal`: useful for durable implementation tracking.
- `$performance-goal`: useful after implementation for FPS/latency/VRAM optimization.
- `$autoresearch-goal`: not recommended for this implementation task.
