from __future__ import annotations

import io
import json

import numpy as np
import pytest

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_protocol_v2 import (
    decode_foundationpose_track_request,
    decode_foundationpose_response,
    encode_foundationpose_track_request,
    encode_foundationpose_response,
)


def _request_bytes(**overrides):
    rgb = overrides.pop("rgb", np.zeros((4, 5, 3), dtype=np.uint8))
    depth_m = overrides.pop("depth_m", np.ones((4, 5), dtype=np.float32))
    t5_T_camera = np.eye(4, dtype=np.float64)
    t5_T_camera[:3, 3] = [0.1, 0.2, 0.3]
    return encode_foundationpose_track_request(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=overrides.pop(
            "intrinsics",
            CameraIntrinsics(fx=10.0, fy=11.0, cx=2.0, cy=2.5, width=5, height=4),
        ),
        request_id=overrides.pop("request_id", "req-1"),
        frame_index=overrides.pop("frame_index", 7),
        capture_monotonic_ns=overrides.pop("capture_monotonic_ns", 123),
        t5_T_camera=overrides.pop("t5_T_camera", t5_T_camera),
        profile=overrides.pop("profile", "phone"),
        foundationpose_root=overrides.pop("foundationpose_root", "/fp"),
        refine_iterations=overrides.pop("refine_iterations", 3),
        track_iterations=overrides.pop("track_iterations", 2),
        reinit=overrides.pop("reinit", True),
        mask_options=overrides.pop("mask_options", {"threshold": 0.4}),
        recovery_options=overrides.pop("recovery_options", {"auto_reinit": True}),
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
            metadata_json=np.array(json.dumps(metadata)),
        )
        return buffer.getvalue()


def test_request_round_trip_preserves_v2_pose_fields():
    decoded = decode_foundationpose_track_request(_request_bytes())

    assert decoded.request_id == "req-1"
    assert decoded.frame_index == 7
    assert decoded.capture_monotonic_ns == 123
    assert decoded.profile == "phone"
    assert decoded.foundationpose_root == "/fp"
    assert decoded.rgb.shape == (4, 5, 3)
    assert decoded.depth_m.shape == (4, 5)
    assert decoded.intrinsics.width == 5
    assert decoded.refine_iterations == 3
    assert decoded.track_iterations == 2
    assert decoded.reinit is True
    assert decoded.mask_options["threshold"] == 0.4
    assert decoded.recovery_options["auto_reinit"] is True
    np.testing.assert_allclose(decoded.t5_T_camera[:3, 3], [0.1, 0.2, 0.3])


def test_request_npz_fields_are_pose_only():
    with np.load(io.BytesIO(_request_bytes()), allow_pickle=False) as archive:
        assert set(archive.files) == {"rgb", "depth_m", "t5_T_camera", "metadata_json"}
        assert "current_t5_T_ee" not in archive.files
        assert "object_T_offset" not in archive.files


def test_request_rejects_bad_rgb_shape():
    with pytest.raises(ValueError, match="rgb must have shape"):
        _request_bytes(rgb=np.zeros((4, 5), dtype=np.uint8))


def test_request_rejects_depth_shape_mismatch():
    with pytest.raises(ValueError, match="does not match rgb"):
        _request_bytes(depth_m=np.ones((4, 4), dtype=np.float32))


def test_request_rejects_non_finite_t5_camera():
    transform = np.eye(4)
    transform[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _request_bytes(t5_T_camera=transform)


@pytest.mark.parametrize("field", ["request_id", "capture_monotonic_ns", "profile", "intrinsics"])
def test_request_rejects_missing_required_metadata(field):
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata.pop(field))

    with pytest.raises(ValueError, match=field):
        decode_foundationpose_track_request(data)


def test_request_rejects_unsupported_protocol_version():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata.__setitem__("protocol_version", 999))

    with pytest.raises(ValueError, match="unsupported"):
        decode_foundationpose_track_request(data)


def test_request_rejects_malformed_intrinsics():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata["intrinsics"].pop("fx"))

    with pytest.raises(ValueError, match="intrinsics"):
        decode_foundationpose_track_request(data)


def test_request_rejects_intrinsics_shape_mismatch():
    data = _mutate_metadata(_request_bytes(), lambda metadata: metadata["intrinsics"].__setitem__("width", 99))

    with pytest.raises(ValueError, match="intrinsics width"):
        decode_foundationpose_track_request(data)


def test_response_round_trip():
    response = {"ok": True, "tracking_state": "TRACKING", "camera_T_object": np.eye(4).tolist()}

    decoded = decode_foundationpose_response(encode_foundationpose_response(response))

    assert decoded == response
