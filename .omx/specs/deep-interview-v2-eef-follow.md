# Deep Interview Spec: v2 Track Hybrid EEF Follow

## Metadata

- Profile: standard
- Context type: brownfield
- Final ambiguity: 0.15
- Threshold: 0.20
- Context snapshot: `.omx/context/v2-eef-follow-20260608T070515Z.md`
- Transcript: `.omx/interviews/v2-eef-follow-20260608T070515Z.md`

## Intent

Restore v1-style robot end-effector following in the v2 FoundationPose workflow, using the more capable v2 model-free object pose pipeline while still commanding the real right hand in T5 Cartesian space.

## Desired Outcome

When the GUI `Track Hybrid` flow is running and EEF follow execution is explicitly enabled, the robot `ee_right` should follow the tracked object pose from FoundationPose. The target hand pose is derived from the tracked object pose with a fixed object-frame offset and orientation rule.

## In Scope

- Add EEF-follow support to the `Track Hybrid` path only.
- Use the existing server-side register plus local tracking path from `visual_servoing.scripts.fp_track_live`.
- Reuse the existing v1 robot command pattern where practical:
  - `RobotContext`
  - `send_right_arm_cartesian()`
  - T5 root frame / `ee_right`
  - Cartesian position command
- Compute:

```python
target_t5_T_ee = t5_T_object @ object_T_ee
```

- Implement `object_T_ee` as the user's fixed rule:
  - offset: `[-0.2, 0.0, 0.0]` meters in the object frame
  - rotation: 180 degrees about the object x-axis
- Interpret the rule according to the right-hand coordinate convention verified with `visual_servoing/cartesian_keyboard_test.py`.
- Add dry-run output/logging for target pose, current pose, command delta, and skip reasons.
- Add real robot execute support for the Hybrid path.
- Expose minimal GUI options needed to run this from `Track Hybrid`.

## Out Of Scope

- Do not modify v1 behavior.
- Do not add EEF follow to `Track Remote` in the first pass.
- Do not add EEF follow to `Track Local` in the first pass.
- Do not add automatic offset capture from the initial EE/object pose.
- Do not redesign the GUI layout.
- Do not bypass pose safety gates for speed.

## Decision Boundaries

OMX may decide implementation details for:

- Argument names and defaults.
- Whether the reusable code lives in a helper module or directly in `fp_track_live.py`.
- Test factoring and fake robot context design.
- Minimal GUI control placement.

OMX should not change without user confirmation:

- The object-frame offset and orientation rule.
- The first target path (`Track Hybrid`).
- The real-robot acceptance target.
- v1 behavior.

## Constraints

- Default per-command EE translation step: `0.03 m`.
- Recommended per-command EE rotation step: bounded and conservative; use a default around `5 deg` unless implementation evidence requires adjustment.
- Command only when tracking state is valid.
- Skip commands on implausible pose jumps.
- Execute mode must be explicit; no robot command should be sent from a normal tracking run unless EEF follow execute is enabled.
- Preserve existing Hybrid tracking/reinit behavior.

## Acceptance Criteria

- GUI `Track Hybrid` can launch Hybrid tracking with EEF-follow options.
- In dry-run mode, logs show:
  - `t5_T_object` / object xyz
  - `target_t5_T_ee` / commanded xyz
  - current EE xyz if robot context is available
  - command delta
  - command skipped/sent reason
- With real robot execute enabled, `ee_right` moves toward:

```python
target_t5_T_ee = t5_T_object @ Tx(-0.2, 0, 0) @ Rx(pi)
```

- EE translation command step is limited to `0.03 m`.
- Commands are skipped when tracking is lost or pose jump checks fail.
- Target transform math has unit tests.
- Robot command integration has tests using fake `RobotContext` / fake SDK surfaces where possible.
- Existing v1 tests remain passing or unaffected.

## Handoff Recommendation

Use `$ralplan .omx/specs/deep-interview-v2-eef-follow.md` before implementation because this touches real robot motion and must preserve existing Track Hybrid behavior.
