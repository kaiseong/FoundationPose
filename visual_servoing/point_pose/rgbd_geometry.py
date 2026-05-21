"""RGB-D point geometry for an observed object coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None
    distortion_coeffs: tuple[float, ...] | None = None
    distortion_model: str | None = None

    @classmethod
    def from_mapping(cls, data: dict) -> "CameraIntrinsics":
        distortion_coeffs = _distortion_coeffs_from_mapping(data)
        if "camera_matrix" in data:
            matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            return cls(
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
                width=data.get("width"),
                height=data.get("height"),
                distortion_coeffs=distortion_coeffs,
                distortion_model=data.get("distortion_model"),
            )
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=data.get("width"),
            height=data.get("height"),
            distortion_coeffs=distortion_coeffs,
            distortion_model=data.get("distortion_model"),
        )

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def as_distortion_coeffs(self, *, fallback_zeros: bool = False) -> np.ndarray | None:
        if self.distortion_coeffs is None:
            if not fallback_zeros:
                return None
            return np.zeros((5, 1), dtype=np.float64)
        return np.asarray(self.distortion_coeffs, dtype=np.float64).reshape(-1, 1)


def _distortion_coeffs_from_mapping(data: dict) -> tuple[float, ...] | None:
    for key in ("distortion_coeffs", "dist_coeffs", "coeffs", "distortion"):
        if key not in data or data[key] is None:
            continue
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)
        return tuple(float(value) for value in values)
    return None


def normalize_vector(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def backproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project masked depth pixels into camera-frame XYZ points."""

    depth = np.asarray(depth_m, dtype=np.float64)
    mask_bool = np.asarray(mask).astype(bool)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth.shape}")
    if mask_bool.shape != depth.shape:
        raise ValueError(f"mask shape {mask_bool.shape} does not match depth {depth.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    valid = mask_bool & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    rows, cols = np.nonzero(valid)
    if stride > 1:
        rows = rows[::stride]
        cols = cols[::stride]
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z)).astype(np.float64)
    pixels = np.column_stack((cols, rows)).astype(np.float64)
    return points, pixels


def robust_center(points: np.ndarray) -> np.ndarray:
    points = _require_points(points)
    return np.median(points, axis=0)


def fit_plane_normal(points: np.ndarray, *, center: np.ndarray | None = None) -> np.ndarray:
    """Fit a plane normal and orient it toward the camera origin."""

    points = _require_points(points, min_points=3)
    if center is None:
        center = robust_center(points)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(vh[-1])
    if float(np.dot(normal, center)) > 0.0:
        normal = -normal
    return normal


def estimate_long_edge_axis(
    points: np.ndarray,
    normal: np.ndarray,
    *,
    center: np.ndarray | None = None,
    previous_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the phone's long edge direction from in-plane point variance."""

    points = _require_points(points, min_points=3)
    normal = normalize_vector(normal)
    if center is None:
        center = robust_center(points)
    centered = points - center
    in_plane = centered - np.outer(centered @ normal, normal)
    _, singular_values, vh = np.linalg.svd(in_plane, full_matrices=False)
    if singular_values[0] < 1e-9:
        axis = _fallback_axis(normal)
    else:
        axis = vh[0]
        axis = axis - np.dot(axis, normal) * normal
        if np.linalg.norm(axis) < 1e-9:
            axis = _fallback_axis(normal)
    axis = normalize_vector(axis)
    if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0.0:
        axis = -axis
    return axis


def estimate_image_long_edge_axis(
    pixels_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
    center: np.ndarray,
    normal: np.ndarray,
    *,
    previous_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the visible long edge from mask pixels and lift it to camera space."""

    pixels = _require_pixels(pixels_xy, min_points=3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    normal = normalize_vector(normal)
    pixel_center = np.median(pixels, axis=0)
    centered_pixels = pixels - pixel_center
    _, singular_values, vh = np.linalg.svd(centered_pixels, full_matrices=False)
    if singular_values[0] < 1e-9:
        return estimate_long_edge_axis(
            np.array(
                [
                    center + np.array([1e-3, 0.0, 0.0]),
                    center - np.array([1e-3, 0.0, 0.0]),
                    center + np.array([0.0, 1e-3, 0.0]),
                ]
            ),
            normal,
            center=center,
            previous_axis=previous_axis,
        )
    direction_uv = normalize_vector(vh[0])
    z = max(float(center[2]), 1e-6)
    axis = np.array(
        [
            direction_uv[0] * z / intrinsics.fx,
            direction_uv[1] * z / intrinsics.fy,
            0.0,
        ],
        dtype=np.float64,
    )
    axis = axis - np.dot(axis, normal) * normal
    if np.linalg.norm(axis) < 1e-9:
        axis = _fallback_axis(normal)
    axis = normalize_vector(axis)
    if previous_axis is not None and float(np.dot(axis, previous_axis)) < 0.0:
        axis = -axis
    return axis


def construct_transform(origin: np.ndarray, z_axis: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    """Construct a right-handed camera-to-phone transform."""

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    z_axis = normalize_vector(z_axis)
    x_axis = np.asarray(x_axis, dtype=np.float64).reshape(3)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize_vector(x_axis)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def construct_transform_from_y_axis(
    origin: np.ndarray,
    z_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    """Construct a right-handed transform where local Y is the phone long edge."""

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    z_axis = normalize_vector(z_axis)
    y_axis = np.asarray(y_axis, dtype=np.float64).reshape(3)
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_axis = normalize_vector(y_axis)
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    y_axis = normalize_vector(np.cross(z_axis, x_axis))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def stabilize_transform(transform: np.ndarray, previous_transform: np.ndarray | None) -> np.ndarray:
    if previous_transform is None:
        return transform
    current = np.array(transform, dtype=np.float64, copy=True)
    previous = np.asarray(previous_transform, dtype=np.float64)
    for axis_index in (0, 1, 2):
        if float(np.dot(current[:3, axis_index], previous[:3, axis_index])) < 0.0:
            current[:3, axis_index] *= -1.0
    if np.linalg.det(current[:3, :3]) < 0.0:
        current[:3, 1] *= -1.0
    return current


def estimate_phone_pose(
    points: np.ndarray,
    *,
    pixels_xy: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    previous_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the observed phone frame from segmented camera-frame points."""

    points = _require_points(points, min_points=6)
    center = robust_center(points)
    normal = fit_plane_normal(points, center=center)
    previous_y = None if previous_transform is None else np.asarray(previous_transform)[:3, 1]
    if pixels_xy is not None and intrinsics is not None:
        y_axis = estimate_image_long_edge_axis(
            pixels_xy,
            intrinsics,
            center,
            normal,
            previous_axis=previous_y,
        )
    else:
        y_axis = estimate_long_edge_axis(points, normal, center=center, previous_axis=previous_y)
    transform = construct_transform_from_y_axis(center, normal, y_axis)
    return stabilize_transform(transform, previous_transform)


def _require_points(points: np.ndarray, *, min_points: int = 1) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] < min_points:
        raise ValueError(f"Need at least {min_points} finite 3D points, got {points.shape[0]}")
    return points


def _require_pixels(pixels_xy: np.ndarray, *, min_points: int = 1) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape (N, 2), got {pixels.shape}")
    finite = np.all(np.isfinite(pixels), axis=1)
    pixels = pixels[finite]
    if pixels.shape[0] < min_points:
        raise ValueError(f"Need at least {min_points} finite pixels, got {pixels.shape[0]}")
    return pixels


def _fallback_axis(normal: np.ndarray) -> np.ndarray:
    camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis = camera_x - np.dot(camera_x, normal) * normal
    if np.linalg.norm(axis) < 1e-9:
        camera_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = camera_y - np.dot(camera_y, normal) * normal
    return normalize_vector(axis)
