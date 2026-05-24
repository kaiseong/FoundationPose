"""Replay recorded RGB-D frames through FoundationPose tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from visual_servoing.foundationpose_model_free.foundationpose_adapter import (
    FoundationPoseAdapter,
    FoundationPoseConfig,
    StubFoundationPoseAdapter,
)
from visual_servoing.foundationpose_model_free.mask_provider import Sam3MaskProvider
from visual_servoing.foundationpose_model_free.recorded_tracking import (
    RecordedTrackingReplayConfig,
    replay_recorded_tracking,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--session-id")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Use the deterministic stub adapter.")
    parser.add_argument("--full-frame-initial-mask", action="store_true")
    parser.add_argument("--foundationpose-root")
    parser.add_argument("--mesh-path")
    parser.add_argument("--prompt")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        profile = ObjectProfileRegistry(args.data_root).get(args.object)
        adapter = _adapter_from_args(args, profile)
        mask_provider = None
        if not args.full_frame_initial_mask:
            mask_provider = Sam3MaskProvider(
                prompt=args.prompt or profile.prompt,
                device=args.device,
                confidence_threshold=args.threshold,
                resolution=args.sam_resolution,
            )
        report = replay_recorded_tracking(
            profile,
            adapter=adapter,
            mask_provider=mask_provider,
            config=RecordedTrackingReplayConfig(
                session_id=args.session_id,
                max_frames=args.max_frames,
                full_frame_initial_mask=args.full_frame_initial_mask,
            ),
        )
        payload = {
            **report,
            "ok": report["tracking_frames"] > 0,
            "returncode": 0 if report["tracking_frames"] > 0 else 1,
            "mode": "track_recording",
        }
    except Exception as exc:
        payload = {"ok": False, "returncode": 2, "mode": "track_recording", "error": str(exc)}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_summary(payload))
    return int(payload.get("returncode", 0))


def _adapter_from_args(args: argparse.Namespace, profile):
    if args.mock:
        return StubFoundationPoseAdapter()
    return FoundationPoseAdapter(
        FoundationPoseConfig(
            foundationpose_root=Path(args.foundationpose_root).expanduser() if args.foundationpose_root else None,
            mesh_path=_mesh_path_from_args(args, profile),
        )
    )


def _mesh_path_from_args(args: argparse.Namespace, profile) -> Path | None:
    if args.mesh_path:
        return Path(args.mesh_path).expanduser()
    for candidate in (
        profile.assets_dir / "model" / "model.obj",
        profile.refs_dir / "model" / "model.obj",
    ):
        if candidate.exists():
            return candidate
    return None


def _human_summary(payload: dict) -> str:
    if payload.get("ok"):
        return (
            f"track_recording: ok processed={payload.get('processed_frames')} "
            f"tracking={payload.get('tracking_frames')} lost={payload.get('lost_frames')}"
        )
    return f"track_recording: failed {payload.get('error', '')}"


if __name__ == "__main__":
    raise SystemExit(main())
