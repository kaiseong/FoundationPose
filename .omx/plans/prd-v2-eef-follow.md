# PRD: v2 Track Hybrid EEF Follow

## Objective

Add real-robot right-hand EEF following to the v2 `Track Hybrid` workflow. The Hybrid flow should continue to use server-side FoundationPose register and local tracking, then optionally command `ee_right` in T5 Cartesian space using the tracked object pose.

## Current State

- GUI `Track Hybrid` launches `visual_servoing.scripts.fp_track_live`.
- `fp_track_live.py` returns and displays `camera_T_object` from `FoundationPoseLiveTracker`.
- Hybrid mode initializes/registers through `RemoteFoundationPoseRegisterProvider` and then tracks locally.
- v1 robot command code already exists in `visual_servoing.visual_servo_client`:
  - `RobotContext`
  - live FK for `t5_T_camera`
  - `send_right_arm_cartesian()`
  - command stream based `rby.CartesianCommandBuilder`
- `cartesian_keyboard_test.py` is the user-validated coordinate reference for the right hand.

## User Workflow

1. Start the v2 server.
2. Open the GUI.
3. Select an object profile with built assets.
4. Enable EEF follow options for `Track Hybrid`.
5. Use `Track Hybrid`.
6. In dry-run, inspect target/current/delta logs.
7. With explicit execute enabled and robot address configured, the right hand follows the tracked pose.

## Functional Requirements

- Add EEF-follow capability to `Track Hybrid` only.
- Do not change v1 behavior.
- Do not add EEF-follow support to `Track Remote` or `Track Local` in this first pass.
- Convert pose before commanding:

```python
t5_T_object = t5_T_camera @ camera_T_object
target_t5_T_ee = t5_T_object @ object_T_ee
```

- Use the fixed user-specified `object_T_ee`:
  - translation: `[-0.2, 0.0, 0.0]` meters in the object frame
  - rotation: 180 degrees about object x
- The intended equivalent target is:

```python
target_t5_T_ee = t5_T_object @ Tx(-0.2, 0, 0) @ Rx(pi)
```

- Use the right-hand coordinate convention validated by `visual_servoing/cartesian_keyboard_test.py`.
- Add CLI options to `fp_track_live.py` for:
  - enabling EEF follow
  - enabling real robot execute
  - robot address/model/power/servo
  - command timing/limits
  - target offset and rotation defaults
  - camera mount/head pose options required for `t5_T_camera`
- Add minimal GUI controls and command-builder propagation for the Hybrid path.
- Include dry-run logs even when robot execution is not enabled.

## Safety Requirements

- No robot command is sent unless execution is explicitly enabled.
- Default per-command translation step limit: `0.03 m`.
- Default per-command orientation step limit should be conservative, around `5 deg`.
- Command only on valid fresh tracking poses, not on LOST state.
- Prefer skipping commands on REINIT state for first-pass safety; begin command after normal TRACKING resumes unless tests show this blocks usability.
- Skip and cancel active command stream when pose jump gates fail.
- Include clear skip reasons in logs.
- Keep `move_to_ready_on_connect` explicit and default off unless the existing local pattern requires otherwise.

## Non-goals

- No v1 behavior changes.
- No automatic offset capture from initial hand/object pose.
- No GUI redesign.
- No Track Remote EEF follow in this pass.
- No Track Local EEF follow in this pass.
- No bypass of pose safety gates for speed.

## Acceptance Criteria

- GUI `Track Hybrid` can launch the Hybrid tracker with EEF-follow flags.
- Dry-run output includes:
  - object xyz in T5 if available
  - target EE xyz in T5
  - current EE xyz
  - command delta
  - command sent/skipped reason
- With execute enabled, the command target is generated from:

```python
target_t5_T_ee = t5_T_object @ Tx(-0.2, 0, 0) @ Rx(pi)
```

- EE translation command step is limited to `0.03 m`.
- Orientation step is limited by the configured rotation cap.
- LOST or implausible jump poses are skipped.
- Existing v1 tests remain unaffected.
- New unit tests cover transform math, gating, step limiting, CLI parsing, and GUI command construction.

## Implementation Notes

- Prefer a small helper module for EEF-follow math and gating so `fp_track_live.py` remains readable.
- Reuse v1 `RobotContext` and camera FK utilities where practical instead of duplicating robot SDK command builders.
- If the current worktree contains interrupted Track Remote EEF-follow edits, keep the final implementation scoped to Hybrid and avoid committing broken partial Track Remote behavior.
