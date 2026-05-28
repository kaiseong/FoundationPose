"""Wire protocol for FoundationPose pose-only remote tracking."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from typing import Any

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_core import require_transform
from visual_servoing.visual_servo_protocol import intrinsics_to_mapping


PROTOCOL_VERSION = 2
REQUEST_CONTENT_TYPE = "application/x-foundationpose-rgbd-npz"
RESPONSE_CONTENT_TYPE = "application/json"
DEFAULT_MAX_CONTENT_LENGTH = 64 * 1024 * 1024


@dataclass(frozen=True)
class FoundationPoseTrackRequest:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    request_id: str
    frame_index: int
    capture_monotonic_ns: int
    t5_T_camera: np.ndarray
    profile: str
    foundationpose_root: str | None
    refine_iterations: int
    track_iterations: int
    reinit: bool
    mask_options: dict[str, Any]
    recovery_options: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FoundationPoseSegmentationRequest:
    rgb: np.ndarray
    depth_m: np.ndarray
    request_id: str
    capture_monotonic_ns: int
    prompt: str
    mask_options: dict[str, Any]
    metadata: dict[str, Any]


def encode_foundationpose_track_request(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    request_id: str,
    frame_index: int,
    capture_monotonic_ns: int,
    t5_T_camera: np.ndarray,
    profile: str,
    foundationpose_root: str | None = None,
    refine_iterations: int = 5,
    track_iterations: int = 2,
    reinit: bool = False,
    mask_options: dict[str, Any] | None = None,
    recovery_options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    rgb = _validate_rgb(rgb)
    depth_m = _validate_depth(depth_m, rgb_shape=rgb.shape[:2])
    t5_T_camera = require_transform(t5_T_camera, "t5_T_camera")
    request_id = str(request_id)
    profile = str(profile).strip()
    if not request_id:
        raise ValueError("request_id is required")
    if not profile:
        raise ValueError("profile is required")

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "frame_index": int(frame_index),
            "capture_monotonic_ns": int(capture_monotonic_ns),
            "profile": profile,
            "foundationpose_root": foundationpose_root,
            "intrinsics": intrinsics_to_mapping(intrinsics),
            "refine_iterations": int(refine_iterations),
            "track_iterations": int(track_iterations),
            "reinit": bool(reinit),
            "mask_options": dict(mask_options or {}),
            "recovery_options": dict(recovery_options or {}),
        }
    )
    buffer = BytesIO()
    np.savez(
        buffer,
        rgb=rgb,
        depth_m=depth_m,
        t5_T_camera=t5_T_camera,
        metadata_json=np.array(json.dumps(payload_metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


def encode_foundationpose_segmentation_request(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    request_id: str,
    capture_monotonic_ns: int,
    prompt: str,
    mask_options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    rgb = _validate_rgb(rgb)
    depth_m = _validate_depth(depth_m, rgb_shape=rgb.shape[:2])
    request_id = str(request_id)
    prompt = str(prompt).strip()
    if not request_id:
        raise ValueError("request_id is required")
    if not prompt:
        raise ValueError("prompt is required")
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "capture_monotonic_ns": int(capture_monotonic_ns),
            "prompt": prompt,
            "mask_options": dict(mask_options or {}),
        }
    )
    buffer = BytesIO()
    np.savez(
        buffer,
        rgb=rgb,
        depth_m=depth_m,
        metadata_json=np.array(json.dumps(payload_metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


def decode_foundationpose_track_request(
    data: bytes,
    *,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
) -> FoundationPoseTrackRequest:
    if len(data) > int(max_content_length):
        raise ValueError(f"request body exceeds {max_content_length} bytes")
    try:
        archive = np.load(BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise ValueError("invalid FoundationPose track npz payload") from exc
    with archive:
        required = {"rgb", "depth_m", "t5_T_camera", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"FoundationPose track payload missing fields: {sorted(missing)}")
        metadata = _decode_metadata(archive["metadata_json"])
        if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported FoundationPose protocol version")
        request_id = str(metadata.get("request_id", ""))
        if not request_id:
            raise ValueError("request_id is required")
        if "capture_monotonic_ns" not in metadata:
            raise ValueError("capture_monotonic_ns is required")
        profile = str(metadata.get("profile", "")).strip()
        if not profile:
            raise ValueError("profile is required")
        rgb = _validate_rgb(archive["rgb"])
        depth_m = _validate_depth(archive["depth_m"], rgb_shape=rgb.shape[:2])
        intrinsics = _validate_intrinsics(metadata.get("intrinsics"), rgb_shape=rgb.shape[:2])
        return FoundationPoseTrackRequest(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            request_id=request_id,
            frame_index=int(metadata.get("frame_index", -1)),
            capture_monotonic_ns=int(metadata["capture_monotonic_ns"]),
            t5_T_camera=require_transform(archive["t5_T_camera"], "t5_T_camera"),
            profile=profile,
            foundationpose_root=_optional_string(metadata.get("foundationpose_root")),
            refine_iterations=_positive_int(metadata.get("refine_iterations", 5), "refine_iterations"),
            track_iterations=_positive_int(metadata.get("track_iterations", 2), "track_iterations"),
            reinit=bool(metadata.get("reinit", False)),
            mask_options=_mapping(metadata.get("mask_options"), "mask_options"),
            recovery_options=_mapping(metadata.get("recovery_options"), "recovery_options"),
            metadata=metadata,
        )


def decode_foundationpose_segmentation_request(
    data: bytes,
    *,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
) -> FoundationPoseSegmentationRequest:
    if len(data) > int(max_content_length):
        raise ValueError(f"request body exceeds {max_content_length} bytes")
    try:
        archive = np.load(BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise ValueError("invalid FoundationPose segmentation npz payload") from exc
    with archive:
        required = {"rgb", "depth_m", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"FoundationPose segmentation payload missing fields: {sorted(missing)}")
        metadata = _decode_metadata(archive["metadata_json"])
        if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported FoundationPose protocol version")
        request_id = str(metadata.get("request_id", ""))
        if not request_id:
            raise ValueError("request_id is required")
        if "capture_monotonic_ns" not in metadata:
            raise ValueError("capture_monotonic_ns is required")
        prompt = str(metadata.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        rgb = _validate_rgb(archive["rgb"])
        depth_m = _validate_depth(archive["depth_m"], rgb_shape=rgb.shape[:2])
        return FoundationPoseSegmentationRequest(
            rgb=rgb,
            depth_m=depth_m,
            request_id=request_id,
            capture_monotonic_ns=int(metadata["capture_monotonic_ns"]),
            prompt=prompt,
            mask_options=_mapping(metadata.get("mask_options"), "mask_options"),
            metadata=metadata,
        )


def encode_foundationpose_response(response: dict[str, Any]) -> bytes:
    return json.dumps(response, separators=(",", ":")).encode("utf-8")


def decode_foundationpose_response(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("FoundationPose response must be a JSON object")
    return value


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
    return np.ascontiguousarray(rgb)


def _validate_depth(depth_m: np.ndarray, *, rgb_shape: tuple[int, int]) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got {depth_m.shape}")
    if depth_m.shape != tuple(rgb_shape):
        raise ValueError(f"depth_m shape {depth_m.shape} does not match rgb shape {rgb_shape}")
    return np.ascontiguousarray(depth_m)


def _validate_intrinsics(value: Any, *, rgb_shape: tuple[int, int]) -> CameraIntrinsics:
    if not isinstance(value, dict):
        raise ValueError("intrinsics is required")
    try:
        intrinsics = CameraIntrinsics.from_mapping(value)
    except Exception as exc:
        raise ValueError("intrinsics must contain fx, fy, cx, and cy") from exc
    floats = np.asarray([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy], dtype=np.float64)
    if not np.all(np.isfinite(floats)):
        raise ValueError("intrinsics contains non-finite values")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("intrinsics fx/fy must be positive")
    height, width = rgb_shape
    if intrinsics.width is not None and int(intrinsics.width) != int(width):
        raise ValueError(f"intrinsics width {intrinsics.width} does not match rgb width {width}")
    if intrinsics.height is not None and int(intrinsics.height) != int(height):
        raise ValueError(f"intrinsics height {intrinsics.height} does not match rgb height {height}")
    return intrinsics


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    raw = value.item() if getattr(value, "shape", ()) == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    metadata = json.loads(str(raw))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object")
    return metadata


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed
