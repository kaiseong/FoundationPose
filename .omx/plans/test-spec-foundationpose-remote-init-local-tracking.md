# Test Spec: FoundationPose Remote Initial Segmentation + Local Tracking

## Metadata
- PRD: `.omx/plans/prd-foundationpose-remote-init-local-tracking.md`
- Source spec: `.omx/specs/deep-interview-remote-init-local-tracking.md`
- Status: approved by Architect and Critic

## Test Strategy
Focus on deterministic unit and integration-style tests that do not require a real ZED, GPU, SAM3, or FoundationPose runtime. Hardware validation remains a manual smoke gap unless the execution environment has the device stack active.

## Unit Tests

### Remote Mask Provider
- Given a small RGB/depth frame and fake server response with `mask_png_b64`, provider returns a bool mask with the expected shape and `source="remote_segmentation"`.
- Provider request uses `/foundationpose/v2/segmentation` and `REQUEST_CONTENT_TYPE`.
- Encoded request decodes through `decode_foundationpose_segmentation_request(...)` with correct prompt, RGB shape, depth shape, and mask options.
- Provider raises a clear error when:
  - response `ok` is false
  - `mask_png_b64` is missing
  - decoded mask shape differs from RGB shape
  - URL open times out or fails

### Tracker / Provider Interaction
- Initial `process_frame(...)` with remote provider calls provider once and adapter `register(...)` once.
- Second tracking frame calls adapter `track_one(...)` and does not call provider.
- After `request_reinit()`, next frame calls provider exactly once and adapter `register(...)` again.
- Hybrid recovery config forces `auto_reinit=False`, so lost-frame automatic reinit does not call the remote provider.
- If the user/GUI asks for auto reinit while launching hybrid, the generated command or runtime metadata reports that hybrid disables it.

### CLI Selection
- `fp_track_live.py` chooses:
  - `PrecomputedMaskProvider` when `--init-mask` is set
  - remote provider when `--remote-init-mask-server` is set
  - `Sam3MaskProvider` otherwise
- Hybrid mode does not instantiate `Sam3MaskProvider` as a fallback after remote failures.
- `--mock` or parser-level tests cover new flags without requiring camera access.
- Parser/provider selection helper tests verify remote server flags are clear and `--init-mask` precedence remains deterministic.

### Diagnostics
- Stage metadata includes remote segmentation latency after provider use.
- Registration and tracking stage timings are separately visible in emitted `timing_ms` fields.
- JSON output tests must assert top-level `timing_ms` includes at minimum `remote_segmentation_ms`, `register_ms`, `track_one_ms`, and `frame_total_ms` when those fields are supplied.
- Overlay tests must assert the CV2 status timing summary can display the new stage fields rather than only coarse `pose_estimation_ms`.
- CUDA memory snapshot tests must assert CPU-only hosts do not import/initialize CUDA as a side effect and either omit CUDA fields or mark them unavailable.
- A mocked CUDA-available path should assert CUDA memory fields are emitted when CUDA is available.

### GUI Command Construction
- `_build()` source contains `Track Hybrid`.
- `GuiCommandBuilder.track_hybrid_live(...)` emits `python -m visual_servoing.scripts.fp_track_live` with remote mask flags and existing camera/ZED/data-root options.
- `run_tracking_hybrid()` downloads remote `model.obj` when local mesh is missing, mirroring local tracking behavior.
- `run_tracking_hybrid()` or its command builder does not propagate GUI `Auto Reinit` into server-backed automatic reinit.
- `_last_tracking_mode` can represent `"hybrid"` and restart/reinit dispatch preserves that mode.
- Existing tests asserting `Track Local` and `Track Remote` still pass.

## Integration Tests
- Use a fake HTTP server or monkeypatched `urlopen` to verify provider request and response decoding end to end.
- Use `StubFoundationPoseAdapter` and a fake provider to run a three-frame init/track/reinit sequence.
- Use existing `test_visual_servo_server_v2.py` segmentation endpoint tests to ensure server behavior remains compatible.

## Manual Smoke Tests

### Server
```bash
python -m visual_servoing.visual_servo_server_v2 --host 0.0.0.0 --port 8081
```

### CLI Hybrid
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

Expected:
- First frame logs one remote segmentation stage.
- Subsequent frames log local tracking without remote segmentation.
- Pressing `R` logs one additional remote segmentation stage.

### GUI Hybrid
- Start GUI.
- Connect to server.
- Select profile with local or downloadable `model.obj`.
- Press `Track Hybrid`.
- Confirm window appears and status/logs show hybrid command.
- Press `R` and confirm reinit uses remote segmentation once.

## Verification Commands
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

## Acceptance Evidence Required
- Test output showing targeted tests pass.
- `git diff --check` clean.
- At least one dry-run or mocked command showing `Track Hybrid` builds the expected CLI.
- If hardware smoke cannot run, final report must state that ZED/GPU live validation is not executed.
