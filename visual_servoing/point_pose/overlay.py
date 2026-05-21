"""OpenCV visualization for the observed point-geometry object frame."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .rgbd_geometry import CameraIntrinsics


def project_points(points_3d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points.shape}")
    z = points[:, 2]
    if np.any(z <= 0) or np.any(~np.isfinite(z)):
        raise ValueError("All projected points must have positive finite z.")
    u = intrinsics.fx * points[:, 0] / z + intrinsics.cx
    v = intrinsics.fy * points[:, 1] / z + intrinsics.cy
    return np.column_stack((u, v))


def axis_points(transform: np.ndarray, axis_length_m: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    origin = transform[:3, 3]
    return np.vstack(
        (
            origin,
            origin + transform[:3, 0] * axis_length_m,
            origin + transform[:3, 1] * axis_length_m,
            origin + transform[:3, 2] * axis_length_m,
        )
    )


def draw_phone_pose_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    transform: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    axis_length_m: float = 0.05,
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw mask contour and XYZ axes on an RGB image."""

    cv2 = _require_cv2()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {image.shape}")
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask_bool.shape} does not match image {image.shape[:2]}")

    rendered = image.copy()
    overlay = rendered.copy()
    overlay[mask_bool] = (255, 210, 0)
    rendered = cv2.addWeighted(overlay, alpha, rendered, 1.0 - alpha, 0)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(rendered, contours, -1, (255, 255, 255), 2)

    pixels = project_points(axis_points(transform, axis_length_m), intrinsics)
    origin = tuple(np.round(pixels[0]).astype(int))
    colors = ((255, 0, 0), (0, 220, 0), (0, 80, 255))
    labels = ("X", "Y", "Z")
    for endpoint, color, label in zip(pixels[1:], colors, labels):
        end = tuple(np.round(endpoint).astype(int))
        cv2.line(rendered, origin, end, color, 3, cv2.LINE_AA)
        cv2.circle(rendered, end, 4, color, -1, cv2.LINE_AA)
        cv2.putText(rendered, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.circle(rendered, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
    return rendered


def draw_axes_overlay(
    image_rgb: np.ndarray,
    transform: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    axis_length_m: float = 0.05,
) -> np.ndarray:
    """Draw XYZ axes on an RGB image without requiring a segmentation mask."""

    cv2 = _require_cv2()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {image.shape}")

    rendered = image.copy()
    pixels = project_points(axis_points(transform, axis_length_m), intrinsics)
    origin = tuple(np.round(pixels[0]).astype(int))
    colors = ((255, 0, 0), (0, 220, 0), (0, 80, 255))
    labels = ("X", "Y", "Z")
    for endpoint, color, label in zip(pixels[1:], colors, labels):
        end = tuple(np.round(endpoint).astype(int))
        cv2.line(rendered, origin, end, color, 3, cv2.LINE_AA)
        cv2.circle(rendered, end, 4, color, -1, cv2.LINE_AA)
        cv2.putText(rendered, label, end, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.circle(rendered, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)
    return rendered


def draw_status_overlay(
    image_rgb: np.ndarray,
    *,
    status: str,
    prompt: str,
    frame_index: int,
    fps: float | None = None,
    message: str | None = None,
    timing_ms: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Draw compact live-tracking status text on an RGB image."""

    cv2 = _require_cv2()
    rendered = np.asarray(image_rgb).copy()
    if rendered.ndim != 3 or rendered.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {rendered.shape}")

    height, width = rendered.shape[:2]
    lines = [_status_header(status=status, prompt=prompt, frame_index=frame_index, fps=fps)]
    if timing_ms:
        lines.append(_timing_summary(timing_ms))
    if message:
        lines.append(message[:110])
    bar_height = 10 + 23 * len(lines)
    cv2.rectangle(rendered, (0, 0), (width, min(height, bar_height)), (0, 0, 0), -1)

    status_color = _status_color(status)
    for index, line in enumerate(lines):
        color = status_color if index == 0 else (230, 230, 230)
        thickness = 2 if index == 0 else 1
        font_scale = 0.58 if index == 0 else 0.45
        cv2.putText(
            rendered,
            line,
            (10, 23 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return rendered


def _status_header(*, status: str, prompt: str, frame_index: int, fps: float | None) -> str:
    fps_text = "--" if fps is None else f"{fps:4.1f}"
    return f"{status} | frame {frame_index} | fps {fps_text} | prompt: {prompt}"


def _status_color(status: str) -> tuple[int, int, int]:
    if status in {"POSE OK", "TRACKING"}:
        return (40, 230, 80)
    if status == "REINIT":
        return (255, 220, 60)
    if status == "LOST":
        return (255, 90, 60)
    return (255, 190, 40)


def _timing_summary(timing_ms: Mapping[str, float]) -> str:
    fields = (
        ("cap", "camera_read_ms"),
        ("seg", "segmentation_ms"),
        ("bp", "backproject_ms"),
        ("pose", "pose_estimation_ms"),
        ("draw", "pose_overlay_ms"),
        ("disp", "display_ms"),
        ("total", "frame_total_ms"),
    )
    parts = []
    if "sam3_model_init_ms" in timing_ms:
        parts.append(f"init:{float(timing_ms['sam3_model_init_ms']):.1f}")
    for label, key in fields:
        if key in timing_ms:
            parts.append(f"{label}:{float(timing_ms[key]):.1f}")
    return "ms " + " | ".join(parts)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required for overlay rendering.") from exc
    return cv2
