"""Set FoundationPose reference cam_in_ob poses for one object profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from visual_servoing.foundationpose_model_free.reference_pose import (
    copy_reference_poses,
    generate_turntable_cam_in_obs,
    pose_depth_sanity_report,
    reference_indices,
    write_reference_poses,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--data-root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pose-dir", help="Directory containing 000000.txt style cam_in_ob matrices.")
    mode.add_argument("--turntable", action="store_true", help="Generate approximate fixed-camera turntable poses.")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--start-deg", type=float, default=0.0)
    parser.add_argument("--step-deg", type=float)
    parser.add_argument("--distance-m", type=float, help="Shortcut for --translation 0 0 DISTANCE.")
    parser.add_argument("--translation", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--first-camera-t-object", help="Optional 4x4 first camera_T_object matrix.")
    args = parser.parse_args()

    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    if args.pose_dir:
        copy_reference_poses(profile, args.pose_dir)
        print(f"copied {len(reference_indices(profile))} reference poses for {profile.name}")
        return 0

    translation = tuple(args.translation) if args.translation else None
    if translation is None and args.distance_m is not None:
        translation = (0.0, 0.0, float(args.distance_m))
    first_pose = np.loadtxt(args.first_camera_t_object).reshape(4, 4) if args.first_camera_t_object else None
    if first_pose is None and translation is None:
        raise SystemExit("--turntable requires --distance-m, --translation, or --first-camera-t-object")
    poses = generate_turntable_cam_in_obs(
        count=len(reference_indices(profile)),
        axis=args.axis,
        start_deg=args.start_deg,
        step_deg=args.step_deg,
        camera_t_object0=first_pose,
        translation_xyz_m=translation,
    )
    step_deg = args.step_deg if args.step_deg is not None else (360.0 / max(len(poses), 1))
    sanity = pose_depth_sanity_report(profile, expected_distance_m=args.distance_m)
    write_reference_poses(
        profile,
        poses,
        pose_source="approximate_turntable",
        pose_provenance={
            "approximate": True,
            "axis": args.axis,
            "start_deg": args.start_deg,
            "step_deg": step_deg,
            "distance_m": args.distance_m,
            "translation_xyz_m": translation,
            "first_camera_t_object": str(Path(args.first_camera_t_object).expanduser().resolve())
            if args.first_camera_t_object
            else None,
            "warning": "Generated turntable poses are approximate and assume fixed camera, centered object, and uniform object rotation.",
            "pose_depth_sanity_report": sanity,
        },
    )
    print(
        f"wrote {len(poses)} approximate turntable reference poses for {profile.name}; "
        "axis/distance assumptions were stored in the profile manifest"
    )
    if sanity.get("warnings"):
        print("warning: " + "; ".join(str(item) for item in sanity["warnings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
