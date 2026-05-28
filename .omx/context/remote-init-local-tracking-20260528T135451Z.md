# Deep Interview Context: remote-init-local-tracking

Task statement: User is considering adding a mode where only the initial segmentation/mask is computed on the server and subsequent FoundationPose tracking runs locally.

Desired outcome: Decide whether this mode is worth implementing and clarify exact requirements before planning/execution.

Stated solution: Server performs first segmentation only; local machine performs tracking after initialization.

Probable intent hypothesis: Avoid local VRAM OOM at 1280x720 while avoiding full remote tracking latency caused by sending full RGBD payload every frame.

Known facts/evidence:
- Previous latency probes showed ZED NEURAL depth is not the main bottleneck.
- Remote tracking bottleneck is large RGBD request I/O/serialization, not the core FoundationPose tracking step.
- `fp_track_live.py` currently supports `--init-mask` via `PrecomputedMaskProvider`; otherwise it creates local `Sam3MaskProvider`.
- `FoundationPoseLiveTracker.process_frame()` accepts an explicit `mask` and uses mask/mask_provider only for initialization or reinitialization.
- After initialization, `FoundationPoseLiveTracker` calls `adapter.track_one(rgb, depth_m, intrinsics)` without SAM3.
- GUI already has a remote segmentation sanity endpoint, but it is a check/preview, not wired as a tracking initializer.

Constraints:
- Deep-interview mode only; do not implement directly in this mode.
- Preserve existing local and remote tracking modes unless clarified otherwise.
- Need to avoid worsening latency/VRAM behavior.
- Existing production code has ongoing uncommitted changes unrelated to this interview.

Unknowns/open questions:
- Whether local OOM is caused by SAM3 initialization or FoundationPose adapter/model rendering/tracking at 1280x720.
- Whether remote segmentation should run only once, on manual R, and/or on auto-reinit.
- Whether local tracking must remain at 1280x720 or can downsample/crop after mask acquisition.
- Whether server should return just a mask PNG/array or also an initial pose.

Decision-boundary unknowns:
- Whether implementation should prioritize minimal new mode versus broader transport/ROI optimization.
- Whether it is acceptable to keep auto-reinit local-only or disable it.

Likely codebase touchpoints:
- `visual_servoing/scripts/fp_track_live.py`
- `visual_servoing/foundationpose_model_free/tracker.py`
- `visual_servoing/foundationpose_model_free/gui_app.py`
- `visual_servoing/visual_servo_server_v2.py`
- `visual_servoing/visual_servo_protocol_v2.py`

Prompt-safe initial-context summary status: not_needed
