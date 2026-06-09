# Test Spec: v2 Track Hybrid EEF Follow

## Unit Tests

### Transform Math

- Identity object pose:
  - input: `t5_T_object = I`
  - expected target translation: `[-0.2, 0.0, 0.0]`
  - expected target rotation: `Rx(pi)`
- Rotated/transformed object pose:
  - verify the `-0.2 m` offset is applied along the object x-axis, not T5 x.
  - verify target rotation is `t5_R_object @ Rx(pi)`.
- Validate non-finite or malformed transforms are rejected.

### Step Limiting

- Current EE far from target:
  - command translation norm must be `<= 0.03 m`.
- Current EE within tolerance:
  - command should not be recommended.
- Rotation far from target:
  - orientation step must be capped by configured max rotation step.
- Rotation within tolerance:
  - orientation command should not rotate further.

### Safety Gating

- LOST state:
  - command skipped.
  - active command stream cancellation is requested in execute mode.
- REINIT state:
  - first-pass default skips command.
- TRACKING with fresh pose:
  - command may be planned.
- Pose jump over threshold:
  - command skipped with explicit reason.

### CLI Parsing

- `fp_track_live.py --eef-follow` enables dry-run EEF target logging.
- `--execute` requires `--address`.
- `--execute` cannot send commands unless EEF follow is enabled or execute implies EEF follow by documented parser behavior.
- Defaults include:
  - offset x `-0.2`
  - object x rotation `180 deg`
  - translation step `0.03 m`

## Integration Tests

### Hybrid Tracking Loop

Use fake camera, fake tracker, and fake robot context.

- When tracker returns TRACKING pose:
  - loop computes `t5_T_object`.
  - loop computes `target_t5_T_ee`.
  - fake robot receives a Cartesian command in execute mode.
- When tracker returns LOST:
  - fake robot receives no send command.
  - fake robot cancellation path is called if command stream was active.
- Dry-run:
  - no fake robot send occurs.
  - JSON/log output includes EEF fields.

### GUI Command Builder

- `GuiCommandBuilder.track_hybrid_live()` appends EEF follow flags only when GUI EEF follow is enabled.
- Execute checkbox/address fields append robot execution flags.
- Existing Track Local and Track Remote command-builder tests remain unchanged.

## Manual Verification

### Dry-run

Run Hybrid without execute and confirm logs:

```bash
python -m visual_servoing.scripts.fp_track_live \
  --object multimeter_zed_1 \
  --prompt multimeter \
  --foundationpose-root /home/kgs/FoundationPose \
  --camera zed \
  --zed-depth-mode NEURAL \
  --remote-register-server 192.168.0.3:8081 \
  --eef-follow \
  --print-json \
  --print-timing
```

Expected:

- cv2 tracking window shows axes.
- stdout includes EEF target/current/delta fields.
- no robot movement.

### Real Robot

Run from GUI `Track Hybrid` with EEF follow and execute enabled, or equivalent CLI:

```bash
python -m visual_servoing.scripts.fp_track_live \
  --object multimeter_zed_1 \
  --prompt multimeter \
  --foundationpose-root /home/kgs/FoundationPose \
  --camera zed \
  --zed-depth-mode NEURAL \
  --remote-register-server 192.168.0.3:8081 \
  --eef-follow \
  --execute \
  --address 192.168.30.1:50051 \
  --max-translation-step-m 0.03 \
  --print-json \
  --print-timing
```

Expected:

- EE moves no more than 3 cm per command.
- EE target stays at object x-axis `-0.2 m` offset.
- EE orientation follows object orientation with x-axis 180 degree flip.
- LOST/jump frames skip command.

## Verification Commands

Run at minimum:

```bash
python -m py_compile \
  visual_servoing/scripts/fp_track_live.py \
  visual_servoing/foundationpose_model_free/gui_app.py

python -m pytest \
  visual_servoing/tests/foundationpose_model_free/test_track_live_output.py \
  visual_servoing/tests/foundationpose_model_free/test_gui_app.py \
  visual_servoing/tests/test_visual_servo_client.py
```

Add any new helper-module tests to the pytest command.
