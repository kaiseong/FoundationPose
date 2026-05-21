"""Reference pose helpers for FoundationPose model-free onboarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .profile_schema import ObjectProfile


def reference_indices(profile: ObjectProfile) -> list[int]:
    return [int(path.stem) for path in sorted(profile.rgb_dir.glob("*.png")) if path.stem.isdigit()]


def write_reference_poses(
    profile: ObjectProfile,
    cam_in_obs: list[np.ndarray],
    *,
    pose_source: str = "manual",
    pose_provenance: dict[str, Any] | None = None,
) -> None:
    indices = reference_indices(profile)
    if len(cam_in_obs) != len(indices):
        raise ValueError(f"pose count {len(cam_in_obs)} does not match reference count {len(indices)}")
    profile.cam_in_ob_dir.mkdir(parents=True, exist_ok=True)
    for index, cam_in_ob in zip(indices, cam_in_obs):
        matrix = np.asarray(cam_in_ob, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"cam_in_ob must have shape (4, 4), got {matrix.shape}")
        np.savetxt(profile.cam_in_ob_dir / f"{index:06d}.txt", matrix)
    profile.metadata["pose_source"] = pose_source
    profile.metadata["pose_provenance"] = pose_provenance or {}
    profile.touch()
    from .profile_manifest import mark_assets_stale

    mark_assets_stale(profile, "reference poses updated")


def copy_reference_poses(profile: ObjectProfile, pose_dir: str | Path) -> None:
    pose_dir = Path(pose_dir).expanduser().resolve()
    indices = reference_indices(profile)
    profile.cam_in_ob_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        source = pose_dir / f"{index:06d}.txt"
        if not source.exists():
            raise FileNotFoundError(f"missing reference pose: {source}")
        matrix = np.loadtxt(source).reshape(4, 4)
        np.savetxt(profile.cam_in_ob_dir / f"{index:06d}.txt", matrix)
    profile.metadata["pose_source"] = "imported_cam_in_ob"
    profile.metadata["pose_provenance"] = {"pose_dir": str(pose_dir), "approximate": False}
    profile.touch()
    from .profile_manifest import mark_assets_stale

    mark_assets_stale(profile, "reference poses imported")


def pose_depth_sanity_report(profile: ObjectProfile, *, expected_distance_m: float | None = None) -> dict[str, Any]:
    medians: list[float] = []
    for index in reference_indices(profile):
        depth_path = profile.depth_dir / f"{index:06d}.npy"
        mask_path = profile.mask_dir / f"{index:06d}.png"
        if not depth_path.exists() or not mask_path.exists():
            continue
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        mask = _load_mask(mask_path)
        if depth.shape != mask.shape:
            continue
        valid = depth[mask & np.isfinite(depth) & (depth > 0.0)]
        if valid.size:
            medians.append(float(np.median(valid)))
    report: dict[str, Any] = {
        "ok": True,
        "warnings": [],
        "masked_depth_median_m": None,
        "expected_distance_m": expected_distance_m,
    }
    if medians:
        observed = float(np.median(np.asarray(medians, dtype=np.float64)))
        report["masked_depth_median_m"] = observed
        if expected_distance_m is not None and abs(observed - float(expected_distance_m)) > 0.08:
            report["ok"] = False
            report["warnings"] = [
                f"turntable distance differs from captured masked depth median by {abs(observed - float(expected_distance_m)):.3f}m"
            ]
    return report


def generate_turntable_cam_in_obs(
    *,
    count: int,
    axis: str = "y",
    start_deg: float = 0.0,
    step_deg: float | None = None,
    camera_t_object0: np.ndarray | None = None,
    translation_xyz_m: tuple[float, float, float] | None = None,
) -> list[np.ndarray]:
    """Generate approximate cam_in_ob poses for fixed-camera object rotation.

    The model assumes each captured reference is separated by a known turntable
    rotation. It is a practical registration helper, not a substitute for a
    calibrated reference-pose source when high-accuracy reconstruction is needed.
    """

    if count < 1:
        raise ValueError("count must be >= 1")
    step = 360.0 / count if step_deg is None else float(step_deg)
    base = np.eye(4, dtype=np.float64) if camera_t_object0 is None else np.asarray(camera_t_object0, dtype=np.float64)
    if base.shape != (4, 4):
        raise ValueError(f"camera_t_object0 must have shape (4, 4), got {base.shape}")
    if translation_xyz_m is not None:
        base = base.copy()
        base[:3, 3] = np.asarray(translation_xyz_m, dtype=np.float64)

    poses = []
    for frame_id in range(count):
        angle = start_deg + frame_id * step
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = rotation_matrix(axis, np.deg2rad(angle))
        camera_t_object = base @ delta
        poses.append(np.linalg.inv(camera_t_object))
    return poses


def rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError("axis must be one of x, y, z")


def _load_mask(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("OpenCV is required to read reference masks.") from exc
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return image > 0
