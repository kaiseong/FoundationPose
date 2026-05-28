# Critic Review: FoundationPose Remote Initial Segmentation + Local Tracking

## Verdict
ITERATE.

## Required Revisions
- Replace permissive diagnostics language with hard tests for CV2 overlay timing summary and JSON top-level `timing_ms`.
- Include `visual_servoing/point_pose/overlay.py` in `py_compile`.
- Include `visual_servoing/tests/test_phone_pose_overlay.py` and `visual_servoing/tests/foundationpose_model_free/test_track_live_output.py` in verification commands.
- Name expected timing fields: `remote_segmentation_ms`, `register_ms`, `track_one_ms`, and `frame_total_ms`.
- Add CPU-only CUDA-memory expectations: no CUDA initialization/import side effects on CPU-only hosts; mocked CUDA path can prove fields are emitted.

## Applied
These revisions were incorporated into the test spec before the second Architect/Critic loop.
