from __future__ import annotations

from http.server import ThreadingHTTPServer
import subprocess
import sys
import threading
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

from visual_servoing.point_pose.sam3_phone_segmenter import MaskSelection
from visual_servoing.visual_servo_core import POSITION_ONLY_ORIENTATION_POLICY, make_transform_from_xyz_rpy
from visual_servoing.visual_servo_protocol import (
    REQUEST_CONTENT_TYPE,
    decode_visual_servo_request,
    decode_visual_servo_response,
    encode_visual_servo_request,
)
from visual_servoing.visual_servo_server import VisualServoService, make_handler
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


class FakeSegmenter:
    def segment(self, image_rgb):
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:8, 2:8] = True
        return MaskSelection(mask=mask, index=0, score=0.9, area=int(mask.sum()), box_xyxy=[2, 2, 7, 7])


def _request_body():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    return _encode_request(rgb, depth, metadata={"ee_link": "link_right_arm_6", "max_translation_step_m": 0.02})


def _request_body_with_metadata(metadata):
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    return _encode_request(rgb, depth, metadata=metadata)


def _encode_request(rgb, depth, *, metadata, current_t5_T_ee=None):
    return encode_visual_servo_request(
        rgb=rgb,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=5.0, cy=5.0, width=10, height=10),
        request_id="req-1",
        frame_index=4,
        capture_monotonic_ns=123,
        t5_T_camera=np.eye(4),
        current_t5_T_ee=np.eye(4) if current_t5_T_ee is None else current_t5_T_ee,
        object_T_offset=np.eye(4),
        metadata=metadata,
    )


def test_server_handles_one_valid_request_without_robot_sdk():
    service = VisualServoService(segmenter_factory=FakeSegmenter)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/visual-servo/action"
        req = urllib_request.Request(url, data=_request_body(), headers={"Content-Type": REQUEST_CONTENT_TYPE})
        with urllib_request.urlopen(req, timeout=2.0) as response:
            payload = decode_visual_servo_response(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert payload["ok"] is True
    assert payload["request_id"] == "req-1"
    assert payload["action"]["root_link"] == "link_torso_5"
    assert payload["action"]["control_mode"] == "right_arm_cartesian"
    assert "address" not in payload["action"]
    assert "power" not in payload["action"]
    assert "servo" not in payload["action"]


def test_server_returns_t5_position_only_action_with_current_rotation_preserved():
    current = make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 45.0])
    metadata = {
        "ee_link": "link_right_arm_6",
        "max_translation_step_m": 2.0,
        "target_offset_t5_m": [0.1, -0.2, 0.3],
    }
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    request = decode_visual_servo_request(_encode_request(rgb, depth, metadata=metadata, current_t5_T_ee=current))
    payload = VisualServoService(segmenter_factory=FakeSegmenter).handle(request)

    assert payload["ok"] is True
    assert payload["offset_frame"] == "link_torso_5"
    assert payload["orientation_policy"] == POSITION_ONLY_ORIENTATION_POLICY
    assert payload["target_offset_t5_m"] == [0.1, -0.2, 0.3]
    assert payload["action"]["orientation_policy"] == POSITION_ONLY_ORIENTATION_POLICY
    target = np.asarray(payload["action"]["target_t5_T_ee"], dtype=np.float64)
    np.testing.assert_allclose(target[:3, :3], current[:3, :3], atol=1e-12)
    object_centroid = np.asarray(payload["observation"]["t5_T_object"], dtype=np.float64)[:3, 3]
    np.testing.assert_allclose(
        payload["servo_step"]["desired_position_t5_m"],
        object_centroid + np.array([0.1, -0.2, 0.3]),
        atol=1e-12,
    )
    assert payload["servo_step"]["wrist_step_rad"] == 0.0


def test_server_module_import_does_not_require_robot_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['rby1_sdk']=None; import visual_servoing.visual_servo_server; print('ok')",
        ],
        cwd="/home/kgs/FoundationPose",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_server_rejects_invalid_ee_link_before_action_payload():
    service = VisualServoService(segmenter_factory=FakeSegmenter)
    payload = service.handle(
        decode_visual_servo_request(
            _request_body_with_metadata({"ee_link": "link_left_arm_6", "max_translation_step_m": 0.02})
        )
    )

    assert payload["ok"] is False
    assert payload["status"] == "skipped"
    assert "right-arm EE link" in payload["reason"]
    assert "action" not in payload


def test_server_rejects_bad_content_type():
    service = VisualServoService(segmenter_factory=FakeSegmenter)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/visual-servo/action"
        req = urllib_request.Request(url, data=b"bad", headers={"Content-Type": "application/octet-stream"})
        try:
            urllib_request.urlopen(req, timeout=2.0)
            raise AssertionError("expected HTTP error")
        except Exception as exc:
            assert "HTTP Error 415" in str(exc)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_server_rejects_oversized_payload():
    service = VisualServoService(segmenter_factory=FakeSegmenter)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, max_content_length=4))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/visual-servo/action"
        req = urllib_request.Request(url, data=b"too large", headers={"Content-Type": REQUEST_CONTENT_TYPE})
        try:
            urllib_request.urlopen(req, timeout=2.0)
            raise AssertionError("expected HTTP error")
        except urllib_error.HTTPError as exc:
            assert exc.code == 413
            payload = decode_visual_servo_response(exc.read())
            assert "too large" in payload["reason"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_server_rejects_invalid_payload():
    service = VisualServoService(segmenter_factory=FakeSegmenter)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/visual-servo/action"
        req = urllib_request.Request(url, data=b"not npz", headers={"Content-Type": REQUEST_CONTENT_TYPE})
        try:
            urllib_request.urlopen(req, timeout=2.0)
            raise AssertionError("expected HTTP error")
        except urllib_error.HTTPError as exc:
            assert exc.code == 400
            payload = decode_visual_servo_response(exc.read())
            assert "invalid visual servo npz payload" in payload["reason"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
