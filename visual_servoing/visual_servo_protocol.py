"""Dependency-light wire protocol for remote visual servo requests."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from typing import Any

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_core import require_transform


PROTOCOL_VERSION = 1
REQUEST_CONTENT_TYPE = "application/x-visual-servo-npz"
RESPONSE_CONTENT_TYPE = "application/json"
DEFAULT_MAX_CONTENT_LENGTH = 64 * 1024 * 1024


@dataclass(frozen=True)
class VisualServoRequest:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    request_id: str
    frame_index: int
    capture_monotonic_ns: int
    t5_T_camera: np.ndarray
    current_t5_T_ee: np.ndarray
    object_T_offset: np.ndarray
    metadata: dict[str, Any]


def encode_visual_servo_request(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    request_id: str,
    frame_index: int,
    capture_monotonic_ns: int,
    t5_T_camera: np.ndarray,
    current_t5_T_ee: np.ndarray,
    object_T_offset: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    rgb = _validate_rgb(rgb)
    depth_m = _validate_depth(depth_m, rgb_shape=rgb.shape[:2])
    t5_T_camera = require_transform(t5_T_camera, "t5_T_camera")
    current_t5_T_ee = require_transform(current_t5_T_ee, "current_t5_T_ee")
    object_T_offset = require_transform(object_T_offset, "object_T_offset")
    if not str(request_id):
        raise ValueError("request_id is required")

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": str(request_id),
            "frame_index": int(frame_index),
            "capture_monotonic_ns": int(capture_monotonic_ns),
            "intrinsics": intrinsics_to_mapping(intrinsics),
        }
    )
    buffer = BytesIO()
    np.savez(
        buffer,
        rgb=rgb,
        depth_m=depth_m,
        t5_T_camera=t5_T_camera,
        current_t5_T_ee=current_t5_T_ee,
        object_T_offset=object_T_offset,
        metadata_json=np.array(json.dumps(payload_metadata, separators=(",", ":"))),
    )
    return buffer.getvalue()


def decode_visual_servo_request(data: bytes, *, max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> VisualServoRequest:
    if len(data) > int(max_content_length):
        raise ValueError(f"request body exceeds {max_content_length} bytes")
    try:
        archive = np.load(BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise ValueError("invalid visual servo npz payload") from exc
    with archive:
        required = {"rgb", "depth_m", "t5_T_camera", "current_t5_T_ee", "object_T_offset", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"visual servo payload missing fields: {sorted(missing)}")
        metadata = _decode_metadata(archive["metadata_json"])
        if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported visual servo protocol version")
        request_id = str(metadata.get("request_id", ""))
        if not request_id:
            raise ValueError("request_id is required")
        if "capture_monotonic_ns" not in metadata:
            raise ValueError("capture_monotonic_ns is required")
        rgb = _validate_rgb(archive["rgb"])
        depth_m = _validate_depth(archive["depth_m"], rgb_shape=rgb.shape[:2])
        intrinsics = CameraIntrinsics.from_mapping(metadata["intrinsics"])
        return VisualServoRequest(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            request_id=request_id,
            frame_index=int(metadata.get("frame_index", -1)),
            capture_monotonic_ns=int(metadata["capture_monotonic_ns"]),
            t5_T_camera=require_transform(archive["t5_T_camera"], "t5_T_camera"),
            current_t5_T_ee=require_transform(archive["current_t5_T_ee"], "current_t5_T_ee"),
            object_T_offset=require_transform(archive["object_T_offset"], "object_T_offset"),
            metadata=metadata,
        )


def encode_visual_servo_response(response: dict[str, Any]) -> bytes:
    return json.dumps(response, separators=(",", ":")).encode("utf-8")


def decode_visual_servo_response(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("visual servo response must be a JSON object")
    return value


def intrinsics_to_mapping(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "width": intrinsics.width,
        "height": intrinsics.height,
        "distortion_coeffs": list(intrinsics.distortion_coeffs) if intrinsics.distortion_coeffs is not None else None,
        "distortion_model": intrinsics.distortion_model,
    }


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


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    raw = value.item() if getattr(value, "shape", ()) == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    metadata = json.loads(str(raw))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object")
    return metadata

