# Architect Review: FoundationPose Remote Initial Segmentation + Local Tracking

## Verdict
ITERATE before Critic review.

## Steelman Antithesis
Option A adds HTTP transport, remote segmentation policy, diagnostics, and GUI command construction to the local tracking path. Extending `visual_servo_client_v2.py` could keep remote communication near existing v2 transport code, which already owns server URL parsing, request timeout handling, and manual `R` request dispatch.

## Tradeoff Tension
- Option A is the smallest behavioral change and uses the existing `MaskProvider` boundary, but it must explicitly suppress automatic server-backed reinit and surface diagnostics outside nested metadata.
- Extending the remote client centralizes HTTP concerns, but risks blurring full remote tracking and local tracking and increases the chance of accidental per-frame server calls.
- Explicit mask orchestration in `run_live()` could distinguish manual from automatic reinit directly, but is more invasive than using the existing tracker/provider contract.

## Required Revisions
- Hybrid mode must force `auto_reinit=False` regardless of GUI checkbox or CLI `--auto-reinit`; manual `R` remains supported via `tracker.request_reinit()`.
- The plan must say that hybrid reports/logs automatic reinit being disabled.
- Diagnostics must be promoted into the top-level `timing_ms` payload used by JSON/overlay, not only stored in `TrackingFrameResult.metadata`.
- Add tests proving hybrid does not propagate GUI `Auto Reinit`.
- Add tracker/provider call-count tests for init, normal tracking, and manual `R`.
- Extract or isolate parser/provider selection in `fp_track_live.py` for testability.

## Applied
These revisions were incorporated into the PRD and test spec before Critic review.
