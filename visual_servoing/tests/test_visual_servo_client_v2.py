from __future__ import annotations

import subprocess
import sys

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_client_v2 import (
    build_parser,
    build_tracking_request_body,
    mask_options_from_args,
    parse_args,
    pose_matrix,
    render_response_overlay,
    response_timing_ms,
    send_track_request,
)
from visual_servoing.visual_servo_protocol_v2 import decode_foundationpose_track_request


class FakeFrame:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth_m = np.ones((4, 5), dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=10.0, fy=11.0, cx=2.0, cy=2.0, width=5, height=4)


def test_v2_client_import_does_not_require_robot_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['rby1_sdk']=None; import visual_servoing.visual_servo_client_v2; print('ok')",
        ],
        cwd="/home/kgs/FoundationPose",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_v2_server_import_does_not_require_robot_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['rby1_sdk']=None; import visual_servoing.visual_servo_server_v2; print('ok')",
        ],
        cwd="/home/kgs/FoundationPose",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_v2_client_parser_has_no_robot_command_options():
    parser = build_parser()
    option_names = {option for action in parser._actions for option in action.option_strings}

    forbidden = {"--execute", "--address", "--power", "--control-root-link", "--ee-link", "--target-offset-t5"}
    assert forbidden.isdisjoint(option_names)


def test_v2_client_parser_defaults_to_window_and_zed_neural():
    args = parse_args(["--object", "phone"])
    explicit = parse_args(["--object", "phone", "--zed-depth-mode", "ULTRA", "--no-window"])

    assert args.no_window is False
    assert args.zed_depth_mode == "NEURAL"
    assert explicit.zed_depth_mode == "ULTRA"
    assert explicit.no_window is True


def test_tracking_request_metadata_is_pose_only():
    args = parse_args(
        [
            "--object",
            "phone",
            "--foundationpose-root",
            "/fp",
            "--prompt",
            "mobile phone",
            "--refine-iterations",
            "3",
            "--track-iterations",
            "2",
            "--reinit",
        ]
    )

    body = build_tracking_request_body(
        frame=FakeFrame(),
        args=args,
        frame_index=9,
        request_id="req-9",
        capture_monotonic_ns=1234,
    )
    decoded = decode_foundationpose_track_request(body)

    assert decoded.profile == "phone"
    assert decoded.foundationpose_root == "/fp"
    assert decoded.refine_iterations == 3
    assert decoded.track_iterations == 2
    assert decoded.reinit is True
    assert mask_options_from_args(args)["prompt"] == "mobile phone"
    for forbidden in ("ee_link", "current_t5_T_ee", "target_t5_T_ee", "command_recommended"):
        assert forbidden not in decoded.metadata


def test_tracking_request_reinit_can_be_overridden_for_one_frame():
    args = parse_args(["--object", "phone"])

    body = build_tracking_request_body(
        frame=FakeFrame(),
        args=args,
        frame_index=1,
        request_id="req-1",
        capture_monotonic_ns=1234,
        reinit=True,
    )
    decoded = decode_foundationpose_track_request(body)

    assert decoded.reinit is True


def test_send_track_request_uses_v2_endpoint_and_content_type(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.headers, timeout))
        return Response()

    monkeypatch.setattr("visual_servoing.visual_servo_client_v2.urllib_request.urlopen", fake_urlopen)

    response = send_track_request("http://127.0.0.1:8081", b"payload", timeout_s=7.0)

    assert response == {"ok": True}
    assert calls[0][0].endswith("/foundationpose/v2/track")
    assert calls[0][1]["Content-type"] == "application/x-foundationpose-rgbd-npz"
    assert calls[0][2] == 7.0


def test_remote_tracking_overlay_draws_camera_pose_axes():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    intrinsics = CameraIntrinsics(fx=80.0, fy=80.0, cx=50.0, cy=40.0, width=100, height=80)
    camera_t_object = np.eye(4, dtype=np.float64)
    camera_t_object[:3, 3] = [0.0, 0.0, 1.0]

    overlay = render_response_overlay(
        rgb,
        intrinsics,
        {
            "ok": True,
            "tracking_state": "TRACKING",
            "camera_T_object": camera_t_object.tolist(),
        },
        prompt="phone",
        frame_index=3,
        fps=12.3,
        timing_ms={"camera_read_ms": 1.0, "pose_estimation_ms": 2.0},
        axis_length_m=0.05,
    )

    assert overlay.shape == rgb.shape
    assert int(overlay.sum()) > 0
    np.testing.assert_allclose(pose_matrix(camera_t_object.tolist()), camera_t_object)


def test_response_timing_extracts_server_tracking_fields():
    timing = response_timing_ms({"server_timing_ms": {"tracking_ms": 12.5, "session_ms": 1.5}})

    assert timing == {"remote_tracking_ms": 12.5, "remote_session_ms": 1.5}
