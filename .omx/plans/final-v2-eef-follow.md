# Final Plan: v2 Track Hybrid EEF Follow

## RALPLAN-DR Summary

### Principles

1. Keep the first implementation scoped to `Track Hybrid`.
2. Reuse proven v1 robot command surfaces instead of rewriting SDK command logic.
3. Treat real robot motion as safety-critical: explicit execute, bounded steps, clear skips.
4. Make transform math testable and visible in logs.
5. Preserve existing tracking behavior when EEF follow is disabled.

### Decision Drivers

1. User requires real robot execution, not only dry-run visualization.
2. User selected `Track Hybrid` as the target path.
3. The target pose rule is fixed: object-frame x offset `-0.2 m` and object x rotation `180 deg`.

### Viable Options

#### Option A: Add EEF follow directly to `fp_track_live.py`

- Pros: shortest path, fewer files.
- Cons: tracking loop becomes harder to maintain and robot safety logic may sprawl.
- Verdict: viable only for a very small patch.

#### Option B: Add a focused EEF-follow helper and call it from `fp_track_live.py`

- Pros: transform math, step limits, and safety gates become testable; Hybrid loop stays readable.
- Cons: one new module or helper surface to maintain.
- Verdict: chosen.

#### Option C: Route Hybrid pose into `visual_servo_client_v2.py`

- Pros: keeps robot control in one client-like path.
- Cons: violates the user's `Track Hybrid` target and blurs Track Remote/Hybrid boundaries.
- Verdict: rejected.

#### Option D: Auto-capture initial `object_T_ee`

- Pros: flexible for arbitrary grasp poses.
- Cons: explicitly rejected by the user; wrong for the fixed hand/object rule.
- Verdict: rejected.

## Consensus Decision

Implement EEF follow as a small reusable helper called from `visual_servoing.scripts.fp_track_live` only when the Hybrid path is running and EEF follow is enabled. Reuse v1 `RobotContext` and FK utilities for real robot commands. Add minimal GUI controls that append the new Hybrid flags.

## Planner Plan

1. Normalize the working tree before implementation.
   - Inspect current uncommitted EEF-follow drafts.
   - Preserve unrelated user/other-agent changes.
   - Do not commit broken Track Remote EEF-follow behavior.

2. Add EEF-follow helper logic.
   - Build `object_T_ee` from object-frame offset and x-axis flip.
   - Convert `camera_T_object` to `t5_T_object` using v1-compatible `t5_T_camera`.
   - Plan bounded translation and rotation toward `target_t5_T_ee`.
   - Gate commands on tracking state, freshness, and pose jump limits.
   - Return structured diagnostics for logs/JSON.

3. Integrate helper into `fp_track_live.py`.
   - Add CLI args for EEF follow, execute, robot connection, command limits, camera mount/head pose, and target transform defaults.
   - Create `RobotContext` only when EEF follow is enabled or execute is requested.
   - On each frame, call the EEF helper after successful pose tracking.
   - Include EEF diagnostics in `emit_json()` and status/log fields.
   - Ensure disabled mode preserves current behavior.

4. Add GUI controls for Hybrid only.
   - Add EEF follow checkbox.
   - Add execute checkbox and robot address field.
   - Add step limit field defaulting to `0.03`.
   - Append flags only to `track_hybrid_live()`.
   - Leave Track Local/Remote unchanged.

5. Add tests.
   - Transform math.
   - Step limiting.
   - LOST/REINIT/jump skip gates.
   - CLI parse behavior.
   - Hybrid run loop using fake tracker/robot where feasible.
   - GUI command-builder propagation.

6. Verify.
   - Run targeted py_compile and pytest commands from the test spec.
   - Run dry-run CLI if environment supports camera-free fake/mocked mode.
   - Leave real robot execution as manual validation unless the user explicitly runs it.

## Architect Review

### Steelman Antithesis

Full wrist orientation following is riskier than position-only tracking because FoundationPose orientation can jump, especially after reinit or poor depth. A safer architecture would ship position-only first and defer wrist orientation.

### Synthesis

The user explicitly requires wrist following, so the plan accepts orientation following but constrains it with step limits, state gates, and jump skips. This preserves the desired behavior while reducing single-frame pose failure risk.

### Tradeoff Tension

Importing v1 `RobotContext` into the model-free tracking script couples v2 tracking to v1 command utilities. Rewriting robot control would reduce coupling, but it would duplicate safety-critical SDK code. Reuse is the better tradeoff for this pass.

### Architecture Verdict

APPROVE with the condition that robot command code remains disabled unless explicitly enabled and that the target transform math is unit-tested.

## Critic Review

### Criteria Check

- Scope consistency: APPROVE. Track Hybrid only.
- User intent: APPROVE. Real robot, wrist follows, fixed object-frame rule.
- Safety: APPROVE with required gates and bounded steps.
- Testability: APPROVE. Math and gating are separable.
- Regression risk: ACCEPTABLE if v1 and Track Remote/Local remain untouched.

### Critic Verdict

APPROVE.

## ADR

### Decision

Add real robot EEF follow to the v2 `Track Hybrid` path by integrating a tested EEF-follow helper into `fp_track_live.py` and exposing minimal GUI controls.

### Drivers

- User needs `Track Hybrid`, not Track Remote.
- User needs real robot execution.
- User requires object-orientation wrist following.
- Existing v1 Cartesian command code is already validated on the robot.

### Alternatives Considered

- Track Remote integration: rejected because it does not satisfy the chosen Hybrid workflow.
- Position-only: rejected because the user requested wrist following.
- Initial relative-pose capture: rejected because the user specified a fixed object-frame rule.
- Rewriting robot SDK command code: rejected because v1 already has the safer proven path.

### Consequences

- Hybrid path gains robot-control responsibility behind explicit flags.
- Tests must protect transform math and disabled-by-default behavior.
- Future Track Remote EEF follow can reuse the helper later, but remains out of scope.

### Follow-ups

- After real robot validation, consider optional Track Remote support.
- Consider a GUI dry-run target overlay if visual debugging is insufficient.
- Consider configurable object-frame offsets per profile after the fixed multimeter use case works.

## Available Agent Types

- `explore`: verify local symbols and existing tests.
- `executor`: implement helper, CLI, GUI, tests.
- `test-engineer`: expand fake robot/tracker coverage.
- `critic`: review safety gates and transform semantics.
- `verifier`: run final checks and summarize evidence.

## Recommended Execution Lane

Use `$ralph .omx/plans/final-v2-eef-follow.md` for sequential implementation and verification. This is safety-sensitive and should be completed by one owner with a tight feedback loop.

If using `$team`, suggested staffing:

- Executor lane: helper + `fp_track_live.py`
- GUI/test lane: GUI command builder + tests
- Verifier lane: safety/transform review and command output audit

## Team Verification Path

1. Unit tests for helper math and gates.
2. GUI command-builder tests.
3. Existing v1 robot-command tests.
4. Py_compile changed modules.
5. Dry-run Hybrid command output review.
6. User-run real robot validation with `--execute`.

## Goal-Mode Follow-up Suggestions

- `$ultragoal` is optional if this should become a durable multi-step objective.
- `$performance-goal` is not the primary fit because this task is robot behavior/safety, not throughput optimization.
- `$autoresearch-goal` is not applicable.
