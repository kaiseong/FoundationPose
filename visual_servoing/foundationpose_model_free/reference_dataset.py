"""Reference-view dataset helpers for model-free onboarding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics

from .profile_schema import ObjectProfile, ProfileStatus


def save_reference_frame(
    profile: ObjectProfile,
    index: int,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    cam_in_ob: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    profile.ensure_dirs()
    cv2 = _require_cv2()
    rgb_path = profile.rgb_dir / f"{index:06d}.png"
    depth_path = profile.depth_dir / f"{index:06d}.npy"
    depth_enhanced_path = profile.depth_enhanced_dir / f"{index:06d}.png"
    mask_path = profile.mask_dir / f"{index:06d}.png"

    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb_u8.shape}")
    depth = np.asarray(depth_m, dtype=np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    if depth.shape != rgb_u8.shape[:2] or mask_bool.shape != rgb_u8.shape[:2]:
        raise ValueError("depth and mask must match RGB image size")

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    np.save(depth_path, depth)
    cv2.imwrite(str(depth_enhanced_path), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
    cv2.imwrite(str(mask_path), mask_bool.astype(np.uint8) * 255)
    np.savetxt(profile.refs_dir / "K.txt", intrinsics.as_matrix())
    _write_select_frames(profile)
    (profile.refs_dir / "intrinsics.json").write_text(
        json.dumps(
            {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
                "distortion_coeffs": list(intrinsics.distortion_coeffs)
                if intrinsics.distortion_coeffs is not None
                else None,
                "distortion_model": intrinsics.distortion_model,
                "distortion_policy": "camera_coefficients"
                if intrinsics.distortion_coeffs is not None
                else "zero_unavailable",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if cam_in_ob is not None:
        np.savetxt(profile.cam_in_ob_dir / f"{index:06d}.txt", np.asarray(cam_in_ob, dtype=np.float64))
    if metadata:
        (profile.refs_dir / f"{index:06d}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    profile.reference_count = max(profile.reference_count, count_reference_frames(profile))
    profile.status = ProfileStatus.CAPTURED
    profile.touch()
    from .profile_manifest import mark_assets_stale

    mark_assets_stale(profile, f"reference frame {index:06d} saved")


def count_reference_frames(profile: ObjectProfile) -> int:
    return min(
        len(list(profile.rgb_dir.glob("*.png"))),
        len(list(profile.depth_dir.glob("*.npy"))),
        len(list(profile.mask_dir.glob("*.png"))),
    )


def count_foundationpose_reference_frames(profile: ObjectProfile) -> int:
    return min(
        len(list(profile.rgb_dir.glob("*.png"))),
        len(list(profile.depth_enhanced_dir.glob("*.png"))),
        len(list(profile.mask_dir.glob("*.png"))),
    )


def has_reference_poses(profile: ObjectProfile) -> bool:
    return len(list(profile.cam_in_ob_dir.glob("*.txt"))) >= count_reference_frames(profile)


def foundationpose_ref_view_dir(profile: ObjectProfile) -> Path:
    return profile.refs_dir


def _write_select_frames(profile: ObjectProfile) -> None:
    indices = [int(path.stem) for path in sorted(profile.rgb_dir.glob("*.png")) if path.stem.isdigit()]
    lines = ["frames:"]
    lines.extend(f"  - {index}" for index in indices)
    (profile.refs_dir / "select_frames.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required to write reference RGB/mask frames.") from exc
    return cv2
