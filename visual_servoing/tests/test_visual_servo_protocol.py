from __future__ import annotations

import io
import json

import numpy as np
import pytest

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_protocol import (
    decode_visual_servo_request,
    decode_visual_servo_response,
    encode_visual_servo_request,
    encode_visual_servo_response,
)


def _request_bytes(**overrides):
    rgb = overrides.pop("rgb", np.zeros((4, 5, 3), dtype=np.uint8))
    depth_m = overrides.pop("depth_m", np.ones((4, 5), dtype=np.float32))
    return encode_visual_servo_request(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=CameraIntrinsics(fx=10.0, fy=11.0, cx=2.0, cy=2.5, width=5, height=4),
        request_id=overrides.pop("request_id", "req-1"),
        frame_index=overrides.pop("frame_index", 7),
        capture_monotonic_ns=overrides.pop("capture_monotonic_ns", 123),
        t5_T_camera=np.eye(4),
        current_t5_T_ee=np.eye(4),
        object_T_offset=np.eye(4),
        metadata=overrides,
    )


def _mutate_metadata(data: bytes, mutator) -> bytes:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        mutator(metadata)
        buffer = io.BytesIO()
        np.savez(
            buffer,
            rgb=archive["rgb"],
            depth_m=archive["depth_m"],
            t5_T_camera=archive["t5_T_camera"],
            current_t5_T_ee=archive["current_t5_T_ee"],
            object_T_offset=archive["object_T_offset"],
            metadata_json=np.array(json.dumps(metadata)),
        )
        return buffer.getvalue()


def test_request_round_trip_preserves_arrays_and_metadata():
    decoded = decode_visual_servo_request(_request_bytes(prompt="object"))

    assert decoded.request_id == "req-1"
    assert decoded.frame_index == 7
    assert decoded.capture_monotonic_ns == 123
    assert decoded.rgb.shape == (4, 5, 3)
    assert decoded.depth_m.shape == (4, 5)
    assert decoded.metadata["prompt"] == "object"
    assert decoded.intrinsics.width == 5


def test_request_rejects_bad_rgb_shape():
    with pytest.raises(ValueError, match="rgb must have shape"):
        _request_bytes(rgb=np.zeros((4, 5), dtype=np.uint8))


def test_request_rejects_depth_shape_mismatch():
    with pytest.raises(ValueError, match="does not match rgb"):
        _request_bytes(depth_m=np.ones((4, 4), dtype=np.float32))


def test_request_rejects_non_finite_transform():
    data = _request_bytes()

    archive = np.load(io.BytesIO(data), allow_pickle=False)
    with archive:
        metadata = str(archive["metadata_json"].item())
        t5_T_camera = np.array(archive["t5_T_camera"])
        t5_T_camera[0, 0] = np.nan
        buffer = io.BytesIO()
        np.savez(
            buffer,
            rgb=archive["rgb"],
            depth_m=archive["depth_m"],
            t5_T_camera=t5_T_camera,
            current_t5_T_ee=archive["current_t5_T_ee"],
            object_T_offset=archive["object_T_offset"],
            metadata_json=np.array(metadata),
        )

    with pytest.raises(ValueError, match="non-finite"):
        decode_visual_servo_request(buffer.getvalue())


def test_request_rejects_missing_request_id():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata.pop("request_id"))

    with pytest.raises(ValueError, match="request_id"):
        decode_visual_servo_request(data)


def test_request_rejects_missing_capture_monotonic_ns():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata.pop("capture_monotonic_ns"))

    with pytest.raises(ValueError, match="capture_monotonic_ns"):
        decode_visual_servo_request(data)


def test_request_rejects_unsupported_protocol_version():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata.__setitem__("protocol_version", 999))

    with pytest.raises(ValueError, match="unsupported"):
        decode_visual_servo_request(data)


def test_response_round_trip():
    response = {"ok": True, "status": "tracking", "action": {"target_t5_T_ee": np.eye(4).tolist()}}

    decoded = decode_visual_servo_response(encode_visual_servo_response(response))

    assert decoded == response
