"""Create an object profile and optionally capture live RGB-D reference frames."""

from __future__ import annotations

import argparse

from visual_servoing.foundationpose_model_free.capture_reference import (
    ReferenceCaptureConfig,
    capture_reference_frames,
)
from visual_servoing.foundationpose_model_free.mask_provider import (
    PrecomputedMaskProvider,
    Sam3MaskProvider,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.realsense_d405 import SUPPORTED_LIVE_CAMERA_MODELS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--frames", type=int, default=0, help="0 creates the profile without live capture.")
    parser.add_argument("--data-root")
    parser.add_argument("--mask-provider", choices=["sam3", "precomputed"], default="sam3")
    parser.add_argument("--mask")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--frame-interval-s", type=float, default=0.0)
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d405")
    parser.add_argument("--serial", default=None, help="Optional camera serial number for live capture.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    registry = ObjectProfileRegistry(args.data_root)
    profile = registry.create(args.name, prompt=args.prompt, exist_ok=True)
    if args.frames <= 0:
        print(f"created profile: {profile.name} at {profile.root}")
        return 0

    if args.mask_provider == "precomputed":
        if not args.mask:
            raise SystemExit("--mask is required with --mask-provider precomputed")
        provider = PrecomputedMaskProvider(args.mask)
    else:
        provider = Sam3MaskProvider(
            prompt=args.prompt,
            device=args.device,
            confidence_threshold=args.threshold,
        )
    capture_reference_frames(
        profile,
        mask_provider=provider,
        config=ReferenceCaptureConfig(
            frames=args.frames,
            frame_interval_s=args.frame_interval_s,
            camera_model=args.camera,
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ),
    )
    print(f"captured {profile.reference_count} reference frames for {profile.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
