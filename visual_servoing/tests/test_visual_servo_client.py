from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from visual_servoing.visual_servo_client import (
    DEFAULT_CAMERA_MOUNT_LINK,
    DEFAULT_RIGHT_ARM_EE_LINK,
    DEFAULT_T5_HEAD_XYZ_RPY,
    REALSENSE_HEAD_TO_CAMERA_XYZ_RPY,
    ZED_HEAD_TO_CAMERA_XYZ_RPY,
    LiveCameraPreview,
    RIGHT_ARM_CONTROL_ROOT_LINK,
    RobotContext,
    ServoLimits,
    clamp_translation_step,
    decode_mask_preview,
    emit_iteration_output,
    format_iteration_summary,
    make_transform_from_xyz_rpy,
    plan_visual_servo_step,
    process_remote_servo_iteration,
    parse_args,
    current_t5_T_camera,
    fixed_t5_T_camera,
    live_should_stop_after_result,
    remote_request_metadata,
    run_remote_fixture,
    send_remote_visual_servo_request,
    signed_angle_about_axis,
    strip_mask_preview_for_logging,
    synthetic_rgbd_fixture,
    validate_args,
)
from visual_servoing.visual_servo_core import POSITION_ONLY_ORIENTATION_POLICY, REMOTE_ACTION_CONTROL_MODE
from visual_servoing.visual_servo_protocol import decode_visual_servo_request


class FakeRobotContext:
    def __init__(self, *, execute: bool = True, current_pose: np.ndarray | None = None):
        self.execute = execute
        self._current_pose = np.eye(4) if current_pose is None else np.asarray(current_pose, dtype=np.float64).copy()
        self.sent_targets: list[np.ndarray] = []
        self.cancel_reasons: list[str] = []

    def current_ee_pose(self):
        return self._current_pose.copy()

    def send_right_arm_cartesian(self, target_t5_T_ee):
        self.sent_targets.append(np.asarray(target_t5_T_ee, dtype=np.float64).copy())
        return {"finish_code": "ok"}

    def cancel_command_stream(self, reason: str):
        self.cancel_reasons.append(str(reason))
        return {"transport": "command_stream", "cancelled": True, "control_cancelled": True, "reason": str(reason)}


class FakeBuilder:
    def __getattr__(self, name):
        if name.startswith(("add_", "set_")):
            return self._chain
        raise AttributeError(name)

    def _chain(self, *args, **kwargs):
        del args, kwargs
        return self


class FakeCartesianImpedanceControlCommandBuilder(FakeBuilder):
    instances: list["FakeCartesianImpedanceControlCommandBuilder"] = []

    def __init__(self):
        self.targets = []
        self.instances.append(self)

    def add_target(self, *args):
        self.targets.append(args)
        return self


class FakeCommandHandler:
    def get(self):
        return SimpleNamespace(finish_code="ok")


class FakeCommandStream:
    def __init__(self):
        self.send_args = None
        self.cancel_calls = 0
        self.done = False

    def send_command(self, *args):
        self.send_args = args
        return SimpleNamespace(finish_code="stream-ok", status="streaming", valid=True)

    def is_done(self):
        return self.done

    def cancel(self):
        self.cancel_calls += 1
        self.done = True


class FakeCommandRobot:
    def __init__(self):
        self.send_args = None
        self.cancel_calls = 0
        self.wait_calls: list[int] = []
        self.manager_state = FakeRby.ControlManagerState.State.Enabled
        self.control_state = FakeRby.ControlManagerState.ControlState.Idle
        self.streams: list[FakeCommandStream] = []
        self.reset_fault_calls = 0
        self.enable_calls = 0

    def send_command(self, *args):
        self.send_args = args
        return FakeCommandHandler()

    def create_command_stream(self, *args, **kwargs):
        del args, kwargs
        stream = FakeCommandStream()
        self.streams.append(stream)
        return stream

    def get_control_manager_state(self):
        return SimpleNamespace(state=self.manager_state, control_state=self.control_state)

    def reset_fault_control_manager(self):
        self.reset_fault_calls += 1
        self.manager_state = FakeRby.ControlManagerState.State.Enabled
        return True

    def enable_control_manager(self):
        self.enable_calls += 1
        self.manager_state = FakeRby.ControlManagerState.State.Enabled
        return True

    def cancel_control(self):
        self.cancel_calls += 1

    def wait_for_control_ready(self, timeout_ms):
        self.wait_calls.append(int(timeout_ms))
        return True


class FakeRby:
    class ControlManagerState:
        class State:
            Enabled = "enabled"
            MajorFault = "major_fault"
            MinorFault = "minor_fault"

        class ControlState:
            Idle = "idle"
            Running = "running"

    CartesianImpedanceControlCommandBuilder = FakeCartesianImpedanceControlCommandBuilder
    JointPositionCommandBuilder = FakeBuilder
    CommandHeaderBuilder = FakeBuilder
    RobotCommandBuilder = FakeBuilder
    ComponentBasedCommandBuilder = FakeBuilder
    BodyComponentBasedCommandBuilder = FakeBuilder


def _remote_args(*, execute: bool = True):
    argv = ["--live", "--remote-server", "127.0.0.1:8080"]
    if execute:
        argv += ["--execute", "--address", "127.0.0.1:50051"]
    else:
        argv += ["--no-execute"]
    args = parse_args(argv)
    args.max_translation_step_m = 0.02
    args.max_wrist_step_deg = 5.0
    return args


def _remote_fixture_call(args, robot_context):
    rgb, depth_m, intrinsics = synthetic_rgbd_fixture()
    return process_remote_servo_iteration(
        args,
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        t5_T_camera=np.eye(4),
        current_t5_T_ee=np.eye(4),
        robot_context=robot_context,
        frame_index=2,
    )


def _tracking_response(body: bytes, *, target: np.ndarray | None = None, **overrides):
    request = decode_visual_servo_request(body)
    payload = {
        "ok": True,
        "status": "tracking",
        "request_id": request.request_id,
        "frame_index": request.frame_index,
        "server_timing_ms": {"planning_ms": 0.1},
        "action": {
            "root_link": RIGHT_ARM_CONTROL_ROOT_LINK,
            "ee_link": "link_right_arm_6",
            "control_mode": REMOTE_ACTION_CONTROL_MODE,
            "target_t5_T_ee": (target if target is not None else np.eye(4)).tolist(),
            "command_recommended": True,
        },
    }
    action_overrides = overrides.pop("action", None)
    payload.update(overrides)
    if action_overrides:
        payload["action"].update(action_overrides)
    return payload


def test_make_transform_from_xyz_rpy_identity():
    transform = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0])

    np.testing.assert_allclose(transform, np.eye(4), atol=1e-12)


def test_make_transform_from_xyz_rpy_translation_and_rotation():
    transform = make_transform_from_xyz_rpy([0.1, 0.2, 0.3, 0, 0, 90])

    np.testing.assert_allclose(transform[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(transform[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_default_fixed_camera_pose_uses_head_1_45_degree_basis():
    args = parse_args(["--live"])

    assert args.camera_mount_link == DEFAULT_CAMERA_MOUNT_LINK
    assert tuple(args.t5_head_pose) == DEFAULT_T5_HEAD_XYZ_RPY
    assert tuple(args.head_camera_pose) == REALSENSE_HEAD_TO_CAMERA_XYZ_RPY
    assert args.camera_pose_preset_resolved == "realsense"
    expected = make_transform_from_xyz_rpy(DEFAULT_T5_HEAD_XYZ_RPY) @ make_transform_from_xyz_rpy(
        args.head_camera_pose
    )
    np.testing.assert_allclose(fixed_t5_T_camera(args), expected, atol=1e-12)


def test_live_zed_uses_zed_camera_pose_preset_by_default():
    args = parse_args(["--live-zed"])

    assert tuple(args.head_camera_pose) == ZED_HEAD_TO_CAMERA_XYZ_RPY
    assert args.camera_pose_preset_resolved == "zed"
    t5_T_camera = fixed_t5_T_camera(args)
    camera_x_axis_t5 = t5_T_camera[:3, :3] @ np.array([1.0, 0.0, 0.0])
    camera_y_axis_t5 = t5_T_camera[:3, :3] @ np.array([0.0, 1.0, 0.0])
    camera_z_axis_t5 = t5_T_camera[:3, :3] @ np.array([0.0, 0.0, 1.0])

    assert camera_x_axis_t5[1] < 0.0
    np.testing.assert_allclose(camera_x_axis_t5, [0.0, -1.0, 0.0], atol=1e-8)
    assert camera_y_axis_t5[2] < 0.0
    np.testing.assert_allclose(camera_y_axis_t5, [-0.70710678, 0.0, -0.70710678], atol=1e-8)
    assert camera_z_axis_t5[0] > 0.0
    np.testing.assert_allclose(camera_z_axis_t5, [0.70710678, 0.0, -0.70710678], atol=1e-8)


def test_camera_pose_preset_and_explicit_pose_override_zed_default():
    realsense_args = parse_args(["--live-zed", "--camera-pose-preset", "realsense"])
    custom_args = parse_args(["--live-zed", "--head-camera-pose", "0", "0", "0", "0", "0", "0"])

    assert tuple(realsense_args.head_camera_pose) == REALSENSE_HEAD_TO_CAMERA_XYZ_RPY
    assert realsense_args.camera_pose_preset_resolved == "realsense"
    assert tuple(custom_args.head_camera_pose) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert custom_args.camera_pose_preset_resolved == "custom"


def test_zed_depth_mode_cli_defaults_to_neural_and_accepts_ultra():
    default_args = parse_args(["--live-zed"])
    explicit_args = parse_args(["--live-zed", "--zed-depth-mode", "ULTRA"])

    assert default_args.zed_depth_mode == "NEURAL"
    assert explicit_args.zed_depth_mode == "ULTRA"


def test_current_camera_pose_uses_mount_link_fk_when_robot_executes(monkeypatch):
    args = parse_args(["--live", "--camera-mount-link", "link_head_2"])
    mount_t5 = make_transform_from_xyz_rpy([0.1, 0.2, 0.3, 0, 10, 0])
    calls = []

    def fake_compute_fk(robot, ee_link, base_link):
        calls.append((robot, ee_link, base_link))
        return mount_t5

    monkeypatch.setattr("visual_servoing.visual_servo_client.compute_fk", fake_compute_fk)
    robot = object()
    context = SimpleNamespace(execute=True, robot=robot)

    expected = mount_t5 @ make_transform_from_xyz_rpy(args.head_camera_pose)

    np.testing.assert_allclose(current_t5_T_camera(args, context), expected, atol=1e-12)
    assert calls == [(robot, "link_head_2", RIGHT_ARM_CONTROL_ROOT_LINK)]


def test_current_camera_pose_falls_back_to_fixed_pose_for_dry_run(monkeypatch):
    args = parse_args(["--live"])

    def fail_compute_fk(*args, **kwargs):
        raise AssertionError("dry-run camera pose must not read robot FK")

    monkeypatch.setattr("visual_servoing.visual_servo_client.compute_fk", fail_compute_fk)
    context = SimpleNamespace(execute=False, robot=None)

    np.testing.assert_allclose(current_t5_T_camera(args, context), fixed_t5_T_camera(args), atol=1e-12)


def test_camera_preview_disabled_by_default():
    args = parse_args(["--live"])

    assert LiveCameraPreview(args).show(np.zeros((2, 2, 3), dtype=np.uint8)) is True


def test_camera_preview_shows_rgb_frame_and_allows_q_to_stop(monkeypatch):
    class FakeCv2:
        COLOR_RGB2BGR = 1
        WINDOW_NORMAL = 2
        INTER_NEAREST = 3

        def __init__(self):
            self.windows = []
            self.images = []
            self.destroyed = []

        def namedWindow(self, name, flag):
            self.windows.append((name, flag))

        def cvtColor(self, image, code):
            assert code == self.COLOR_RGB2BGR
            return image[..., ::-1]

        def resize(self, image, *args, **kwargs):
            del args, kwargs
            return image

        def imshow(self, name, image):
            self.images.append((name, image.copy()))

        def waitKey(self, delay_ms):
            assert delay_ms == 1
            return ord("q")

        def destroyWindow(self, name):
            self.destroyed.append(name)

    fake_cv2 = FakeCv2()
    import visual_servoing.visual_servo_client as client

    monkeypatch.setattr(client, "require_cv2", lambda: fake_cv2)
    args = parse_args(["--live", "--show-camera-window", "--camera-window-scale", "2.0"])
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)

    with LiveCameraPreview(args) as preview:
        keep_running = preview.show(rgb)

    assert keep_running is False
    assert fake_cv2.windows == [("visual_servo_client", fake_cv2.WINDOW_NORMAL)]
    assert fake_cv2.destroyed == ["visual_servo_client"]
    assert fake_cv2.images[0][0] == "visual_servo_client"
    np.testing.assert_array_equal(fake_cv2.images[0][1], np.array([[[3, 2, 1]]], dtype=np.uint8))


def test_no_window_disables_camera_preview_even_when_requested(monkeypatch):
    import visual_servoing.visual_servo_client as client

    monkeypatch.setattr(client, "require_cv2", lambda: pytest.fail("cv2 should not be loaded"))
    args = parse_args(["--live", "--show-camera-window", "--no-window"])

    with LiveCameraPreview(args) as preview:
        assert preview.show(np.zeros((2, 2, 3), dtype=np.uint8)) is True


def test_show_mask_window_requests_remote_mask_preview():
    args = parse_args(["--live", "--remote-server", "127.0.0.1:8080", "--show-mask-window"])

    assert LiveCameraPreview(args).enabled is True
    assert remote_request_metadata(args)["return_mask_preview"] is True


def test_live_converged_result_keeps_tracking_by_default():
    args = parse_args(["--live"])

    assert live_should_stop_after_result(args, {"ok": True, "status": "converged"}) is False


def test_live_converged_result_can_stop_when_requested():
    args = parse_args(["--live", "--stop-on-converged"])

    assert live_should_stop_after_result(args, {"ok": True, "status": "converged"}) is True
    assert live_should_stop_after_result(args, {"ok": False, "status": "converged"}) is False


def test_no_window_disables_remote_mask_preview_request():
    args = parse_args(["--live", "--remote-server", "127.0.0.1:8080", "--show-mask-window", "--no-window"])

    assert LiveCameraPreview(args).enabled is False
    assert remote_request_metadata(args)["return_mask_preview"] is False


def test_camera_preview_overlays_mask_preview(monkeypatch):
    class FakeCv2:
        COLOR_RGB2BGR = 1
        WINDOW_NORMAL = 2
        INTER_NEAREST = 3

        def __init__(self):
            self.images = []

        def namedWindow(self, name, flag):
            del name, flag

        def cvtColor(self, image, code):
            assert code == self.COLOR_RGB2BGR
            return image[..., ::-1]

        def resize(self, image, *args, **kwargs):
            del args, kwargs
            return image

        def imshow(self, name, image):
            self.images.append((name, image.copy()))

        def waitKey(self, delay_ms):
            assert delay_ms == 1
            return -1

        def destroyWindow(self, name):
            del name

    fake_cv2 = FakeCv2()
    import visual_servoing.visual_servo_client as client

    monkeypatch.setattr(client, "require_cv2", lambda: fake_cv2)
    args = parse_args(["--live", "--show-mask-window", "--mask-overlay-alpha", "1.0"])
    mask = np.array([[True, False], [False, False]], dtype=bool)
    preview_payload = {
        "encoding": "packbits-b64-v1",
        "shape": [2, 2],
        "data": base64.b64encode(np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()).decode("ascii"),
    }

    with LiveCameraPreview(args) as preview:
        keep_running = preview.show_result(
            np.zeros((2, 2, 3), dtype=np.uint8),
            {"mask": {"preview": preview_payload}},
        )

    assert keep_running is True
    assert fake_cv2.images[0][0] == "visual_servo_client"
    np.testing.assert_array_equal(fake_cv2.images[0][1][0, 0], [0, 255, 0])
    np.testing.assert_array_equal(fake_cv2.images[0][1][0, 1], [0, 0, 0])


def test_camera_preview_draws_estimated_depth_overlay(monkeypatch):
    class FakeCv2:
        COLOR_RGB2BGR = 1
        WINDOW_NORMAL = 2
        INTER_NEAREST = 3
        FONT_HERSHEY_SIMPLEX = 4
        LINE_AA = 5

        def __init__(self):
            self.images = []
            self.texts = []
            self.rectangles = []

        def namedWindow(self, name, flag):
            del name, flag

        def cvtColor(self, image, code):
            assert code == self.COLOR_RGB2BGR
            return image[..., ::-1]

        def resize(self, image, *args, **kwargs):
            del args, kwargs
            return image

        def rectangle(self, image, pt1, pt2, color, thickness):
            del image
            self.rectangles.append((pt1, pt2, color, thickness))

        def putText(self, image, text, org, font, scale, color, thickness, line_type):
            del image, org, font, scale, color, thickness, line_type
            self.texts.append(text)

        def imshow(self, name, image):
            self.images.append((name, image.copy()))

        def waitKey(self, delay_ms):
            assert delay_ms == 1
            return -1

        def destroyWindow(self, name):
            del name

    fake_cv2 = FakeCv2()
    import visual_servoing.visual_servo_client as client

    monkeypatch.setattr(client, "require_cv2", lambda: fake_cv2)
    args = parse_args(["--live", "--show-camera-window"])

    with LiveCameraPreview(args) as preview:
        keep_running = preview.show_result(
            np.zeros((40, 120, 3), dtype=np.uint8),
            {"observation": {"centroid_camera_m": [0.0, 0.0, 0.2134]}},
        )
        keep_running = keep_running and preview.show(np.zeros((40, 120, 3), dtype=np.uint8))
        keep_running = keep_running and preview.show_result(
            np.zeros((40, 120, 3), dtype=np.uint8),
            {"ok": False, "reason": "No usable object mask was produced."},
        )
        keep_running = keep_running and preview.show_result(
            np.zeros((40, 120, 3), dtype=np.uint8),
            {"observation": {"centroid_camera_m": [0.0, 0.0, 0.4172]}},
        )

    assert keep_running is True
    assert fake_cv2.texts == [
        "depth z=0.213 m",
        "depth z=0.213 m",
        "depth z=0.213 m",
        "depth z=0.417 m",
    ]
    assert fake_cv2.rectangles


def test_decode_mask_preview_round_trip():
    mask = np.array([[False, True, False], [True, True, False]], dtype=bool)
    preview = {
        "encoding": "packbits-b64-v1",
        "shape": [2, 3],
        "data": base64.b64encode(np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()).decode("ascii"),
    }

    np.testing.assert_array_equal(decode_mask_preview(preview), mask)


def test_strip_mask_preview_for_logging_keeps_mask_metadata_without_blob():
    payload = {"mask": {"score": 0.9, "preview": {"data": "large"}}}

    sanitized = strip_mask_preview_for_logging(payload)

    assert sanitized == {"mask": {"score": 0.9}}
    assert payload["mask"]["preview"] == {"data": "large"}


def test_clamp_translation_step_limits_norm():
    step = clamp_translation_step(np.array([3.0, 4.0, 0.0]), max_step_m=0.5)

    np.testing.assert_allclose(np.linalg.norm(step), 0.5)
    np.testing.assert_allclose(step / np.linalg.norm(step), [0.6, 0.8, 0.0])


def test_clamp_translation_step_keeps_small_error():
    error = np.array([0.001, 0.002, 0.0])

    step = clamp_translation_step(error, max_step_m=0.5)

    np.testing.assert_allclose(step, error)


def test_signed_angle_about_axis():
    angle = signed_angle_about_axis(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    assert math.isclose(angle, math.pi / 2.0, abs_tol=1e-12)


def test_plan_visual_servo_step_dry_run_fields():
    current = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0])
    visual = make_transform_from_xyz_rpy([0.1, 0, 0, 0, 0, 0])
    limits = ServoLimits(max_translation_step_m=0.02, max_wrist_step_rad=math.radians(5))

    step = plan_visual_servo_step(
        current_t5_T_ee=current,
        visual_target_t5=visual,
        ee_offset=make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0]),
        ee_offset_rpy_deg=(0.0, 0.0, 0.0),
        limits=limits,
        object_grasp_axis_t5=np.array([0.0, 1.0, 0.0]),
    )

    assert step.status == "tracking"
    np.testing.assert_allclose(step.position_error_m, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(step.translation_step_m, [0.02, 0.0, 0.0])
    assert step.command_recommended is True


def test_plan_visual_servo_step_clamps_wrist():
    current = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0])
    visual = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0])
    limits = ServoLimits(max_translation_step_m=0.02, max_wrist_step_rad=math.radians(3))

    step = plan_visual_servo_step(
        current_t5_T_ee=current,
        visual_target_t5=visual,
        ee_offset=make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0]),
        ee_offset_rpy_deg=(0.0, 0.0, 0.0),
        limits=limits,
        object_grasp_axis_t5=np.array([-1.0, 0.0, 0.0]),
    )

    assert math.isclose(abs(step.wrist_step_rad), math.radians(3), abs_tol=1e-12)


def test_plan_visual_servo_step_converged():
    current = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0])
    visual = make_transform_from_xyz_rpy([0.001, 0, 0, 0, 0, 0])
    limits = ServoLimits(
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5),
        position_tolerance_m=0.005,
        wrist_tolerance_rad=math.radians(2),
    )

    step = plan_visual_servo_step(
        current_t5_T_ee=current,
        visual_target_t5=visual,
        ee_offset=make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 0]),
        ee_offset_rpy_deg=(0.0, 0.0, 0.0),
        limits=limits,
        object_grasp_axis_t5=np.array([0.0, 1.0, 0.0]),
    )

    assert step.status == "converged"
    assert step.command_recommended is False


def test_plan_visual_servo_step_uses_current_ee_frame_offset_translation():
    current = make_transform_from_xyz_rpy([0, 0, 0, 0, 0, 90])
    visual = make_transform_from_xyz_rpy([1, 1, 0, 0, 0, 0])
    offset = make_transform_from_xyz_rpy([0.1, 0, 0, 0, 0, 0])
    limits = ServoLimits(max_translation_step_m=2.0)

    step = plan_visual_servo_step(
        current_t5_T_ee=current,
        visual_target_t5=visual,
        ee_offset=offset,
        ee_offset_rpy_deg=(0.0, 0.0, 0.0),
        limits=limits,
        object_grasp_axis_t5=np.array([0.0, 1.0, 0.0]),
    )

    np.testing.assert_allclose(step.desired_position_t5_m, [1.0, 1.1, 0.0], atol=1e-12)


def test_dry_run_context_does_not_import_robot_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "rby1_sdk", None)
    args = SimpleNamespace(current_ee_pose=[0, 0, 0, 0, 0, 0])

    pose = RobotContext.dry_run(args).current_ee_pose()

    np.testing.assert_allclose(pose, np.eye(4))


def test_remote_request_timeout_has_actionable_message(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("timed out")

    monkeypatch.setattr("visual_servoing.visual_servo_client.urllib_request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="timed out after 2.0s"):
        send_remote_visual_servo_request("127.0.0.1:8080", b"body", timeout_s=2.0)


def test_send_right_arm_cartesian_uses_command_stream_with_timeout():
    args = parse_args(["--live", "--command-priority", "7", "--command-timeout-s", "1.25"])
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)
    target = make_transform_from_xyz_rpy([0.1, -0.2, 0.3, 0, 0, 0])

    feedback = context.send_right_arm_cartesian(target)

    assert feedback == {
        "finish_code": "stream-ok",
        "status": "streaming",
        "valid": "True",
        "transport": "command_stream",
        "stream_done": False,
    }
    assert robot.send_args is None
    assert len(robot.streams) == 1
    assert robot.streams[0].send_args is not None
    assert len(robot.streams[0].send_args) == 2
    assert robot.streams[0].send_args[1] == 1250
    assert robot.wait_calls == []
    target_args = FakeCartesianImpedanceControlCommandBuilder.instances[-1].targets[-1]
    assert target_args[0] == args.control_root_link
    assert target_args[1] == args.ee_link
    np.testing.assert_allclose(target_args[2], target)
    assert target_args[3:] == (
        float(args.linear_limit),
        float(args.angular_limit),
        float(args.linear_gain),
        float(args.angular_gain),
    )


def test_send_right_arm_cartesian_cancels_active_control_before_waiting():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    robot.control_state = FakeRby.ControlManagerState.ControlState.Running
    context = RobotContext(args, robot=robot, rby=FakeRby)

    context.send_right_arm_cartesian(np.eye(4))

    assert robot.cancel_calls == 0
    assert robot.wait_calls == []


def test_send_right_arm_cartesian_reopens_done_stream_after_waiting():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)
    context.open_command_stream()
    robot.streams[0].done = True

    context.send_right_arm_cartesian(np.eye(4))

    assert len(robot.streams) == 2
    assert robot.wait_calls == [2500]


def test_wait_for_control_ready_recovers_control_manager_fault():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    robot.manager_state = FakeRby.ControlManagerState.State.MinorFault
    context = RobotContext(args, robot=robot, rby=FakeRby)

    context.wait_for_control_ready()

    assert robot.reset_fault_calls == 1
    assert robot.enable_calls == 1
    assert robot.wait_calls == [2500]


def test_done_stream_reopen_recovers_control_manager_fault_before_waiting():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)
    context.open_command_stream()
    robot.streams[0].done = True
    robot.manager_state = FakeRby.ControlManagerState.State.MajorFault

    context.send_right_arm_cartesian(np.eye(4))

    assert robot.reset_fault_calls == 1
    assert robot.enable_calls == 1
    assert robot.wait_calls == [2500]
    assert len(robot.streams) == 2


def test_cancel_command_stream_stops_stream_and_control_until_next_valid_command():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)
    context.open_command_stream()

    feedback = context.cancel_command_stream("lost target")

    assert feedback["cancelled"] is True
    assert feedback["control_cancelled"] is True
    assert feedback["reason"] == "lost target"
    assert robot.streams[0].cancel_calls == 1
    assert robot.cancel_calls == 1
    assert context.command_stream is None

    robot.control_state = FakeRby.ControlManagerState.ControlState.Running
    context.send_right_arm_cartesian(np.eye(4))

    assert len(robot.streams) == 2
    assert robot.wait_calls == [2500]
    assert robot.cancel_calls == 2


def test_move_right_arm_to_ready_pose_sends_joint_command_and_waits():
    args = parse_args(
        [
            "--live",
            "--command-priority",
            "9",
            "--ready-min-time-s",
            "2.5",
            "--ready-hold-time-s",
            "3.5",
        ]
    )
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)

    feedback = context.move_right_arm_to_ready_pose()

    assert feedback == {"finish_code": "ok"}
    assert robot.send_args is not None
    assert len(robot.send_args) == 2
    assert robot.send_args[1] == 9
    assert robot.wait_calls == [1000, 1000]


def test_close_cancels_command_stream_before_disconnect():
    args = parse_args(["--live"])
    robot = FakeCommandRobot()
    robot.disconnect_calls = 0

    def disconnect():
        robot.disconnect_calls += 1

    robot.disconnect = disconnect
    context = RobotContext(args, robot=robot, rby=FakeRby)
    context.open_command_stream()

    context.close()

    assert robot.streams[0].cancel_calls == 1
    assert robot.disconnect_calls == 1


def test_validate_execute_rejects_non_right_arm_ee_link():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--ee-link", "link_left_arm_6"])

    with pytest.raises(SystemExit, match="right-arm EE links"):
        validate_args(args)


def test_validate_execute_defaults_to_m_model_and_all_power_servo():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051"])

    assert args.execute is True
    assert args.width == 1280
    assert args.height == 720
    assert args.fps == 15
    assert args.max_iterations == 0
    assert args.max_translation_step_m == 0.06
    assert args.remote_timeout_s == 2.0
    assert args.stale_action_max_age_s == 1.0
    assert args.target_offset_t5 == (0.0, 0.0, 0.0)
    assert args.model == "m"
    assert args.ee_link == DEFAULT_RIGHT_ARM_EE_LINK
    assert args.power == ".*"
    assert args.servo == ".*"
    assert args.move_to_ready_on_connect is True
    validate_args(args)


def test_no_execute_and_no_ready_flags_disable_robot_defaults():
    args = parse_args(["--live", "--no-execute", "--no-move-to-ready-on-connect"])

    assert args.execute is False
    assert args.move_to_ready_on_connect is False
    validate_args(args)


def test_validate_execute_rejects_non_m_model():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--model", "a"])

    with pytest.raises(SystemExit, match="--model 'm'"):
        validate_args(args)


def test_validate_execute_rejects_empty_power_or_servo_pattern():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--power", ""])

    with pytest.raises(SystemExit, match="--power cannot be empty"):
        validate_args(args)

    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--servo", ""])

    with pytest.raises(SystemExit, match="--servo cannot be empty"):
        validate_args(args)


def test_validate_execute_rejects_invalid_ready_or_priority_values():
    for flag, value in [
        ("--command-priority", "-1"),
        ("--command-hold-time-s", "0"),
        ("--command-timeout-s", "0"),
        ("--ready-min-time-s", "0"),
        ("--ready-hold-time-s", "0"),
    ]:
        args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", flag, value])
        with pytest.raises(SystemExit):
            validate_args(args)


def test_validate_rejects_invalid_camera_window_scale():
    args = parse_args(["--live", "--show-camera-window", "--camera-window-scale", "0"])

    with pytest.raises(SystemExit):
        validate_args(args)


def test_validate_execute_rejects_non_t5_root_link():
    args = parse_args(
        [
            "--live",
            "--execute",
            "--address",
            "127.0.0.1:50051",
            "--control-root-link",
            "base",
        ]
    )

    with pytest.raises(SystemExit, match="--control-root-link 'link_torso_5'"):
        validate_args(args)


def test_validate_execute_accepts_robot_safe_defaults():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051"])

    validate_args(args)


def test_remote_target_offset_t5_metadata_is_explicit_t5_frame():
    args = parse_args(
        [
            "--live",
            "--remote-server",
            "127.0.0.1:8080",
            "--target-offset-t5",
            "0.1",
            "-0.2",
            "0.3",
        ]
    )

    metadata = remote_request_metadata(args)

    assert metadata["target_offset_t5_m"] == [0.1, -0.2, 0.3]
    assert metadata["offset_frame"] == "link_torso_5"
    assert metadata["orientation_policy"] == "preserve_current_ee_rotation"
    assert metadata["servo_dofs"] == "xyz_position_only"
    assert metadata["camera_pose_preset"] == "realsense"
    assert metadata["head_camera_pose"] == list(REALSENSE_HEAD_TO_CAMERA_XYZ_RPY)


def test_remote_rejects_object_frame_offset_to_avoid_axis_spin():
    args = parse_args(
        [
            "--live",
            "--remote-server",
            "127.0.0.1:8080",
            "--object-offset",
            "0.1",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
        ]
    )

    with pytest.raises(SystemExit, match="--target-offset-t5"):
        validate_args(args)


def test_validate_remote_fixture_request_requires_remote_server():
    args = parse_args(["--remote-fixture-request"])

    with pytest.raises(SystemExit, match="requires --remote-server"):
        validate_args(args)


def test_validate_remote_fixture_request_rejects_execute():
    args = parse_args(
        [
            "--remote-fixture-request",
            "--remote-server",
            "127.0.0.1:8080",
            "--execute",
            "--address",
            "127.0.0.1:50051",
        ]
    )

    with pytest.raises(SystemExit, match="cannot be used with --execute"):
        validate_args(args)


def test_remote_iteration_executes_only_valid_tracking_action(monkeypatch):
    args = _remote_args(execute=True)
    args.target_offset_t5 = (0.1, -0.2, 0.3)
    robot_context = FakeRobotContext(execute=True)
    target = make_transform_from_xyz_rpy([0.01, 0.0, 0.0, 0.0, 0.0, 2.0])

    def fake_send(server, body, *, timeout_s):
        assert server == "127.0.0.1:8080"
        assert timeout_s == args.remote_timeout_s
        request = decode_visual_servo_request(body)
        np.testing.assert_allclose(request.object_T_offset, np.eye(4), atol=1e-12)
        assert request.metadata["target_offset_t5_m"] == [0.1, -0.2, 0.3]
        assert request.metadata["offset_frame"] == "link_torso_5"
        assert request.metadata["orientation_policy"] == "preserve_current_ee_rotation"
        np.testing.assert_allclose(request.metadata["target_t5_R_ee"], np.eye(3), atol=1e-12)
        return _tracking_response(body, target=target)

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, next_pose = _remote_fixture_call(args, robot_context)

    assert result["command_sent"] is True
    assert result["remote"]["action_executable"] is True
    assert len(robot_context.sent_targets) == 1
    np.testing.assert_allclose(robot_context.sent_targets[0], target)
    np.testing.assert_allclose(next_pose, target)


def test_remote_iteration_preserves_current_rotation_for_legacy_server_response(monkeypatch):
    args = _remote_args(execute=True)
    rgb, depth_m, intrinsics = synthetic_rgbd_fixture()
    current = make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 45.0])
    robot_context = FakeRobotContext(execute=True, current_pose=current)
    target_from_legacy_server = np.eye(4)
    target_from_legacy_server[:3, 3] = [0.01, 0.0, 0.0]
    expected_target = current.copy()
    expected_target[:3, 3] = target_from_legacy_server[:3, 3]

    def fake_send(server, body, *, timeout_s):
        del server, timeout_s
        request = decode_visual_servo_request(body)
        np.testing.assert_allclose(request.metadata["target_t5_R_ee"], current[:3, :3], atol=1e-12)
        return _tracking_response(
            body,
            target=target_from_legacy_server,
            action={
                "orientation_policy": "fixed_t5_rpy_zero",
            },
        )

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, next_pose = process_remote_servo_iteration(
        args,
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        t5_T_camera=np.eye(4),
        current_t5_T_ee=np.eye(4),
        robot_context=robot_context,
        frame_index=2,
    )

    assert result["command_sent"] is True
    assert result["remote"]["action_executable"] is True
    assert result["orientation_policy"] == POSITION_ONLY_ORIENTATION_POLICY
    np.testing.assert_allclose(robot_context.sent_targets[0], expected_target)
    np.testing.assert_allclose(next_pose, expected_target)


def test_remote_iteration_rejects_stale_before_command_path(monkeypatch):
    args = _remote_args(execute=True)
    args.stale_action_max_age_s = 0.001
    robot_context = FakeRobotContext(execute=True)

    def fake_send(server, body, *, timeout_s):
        del server, timeout_s
        import time

        time.sleep(0.02)
        return _tracking_response(body, target=make_transform_from_xyz_rpy([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, next_pose = _remote_fixture_call(args, robot_context)

    assert result["command_sent"] is False
    assert result["remote"]["stale"] is True
    assert result["remote"]["action_executable"] is False
    assert "stale" in result["reason"]
    assert robot_context.sent_targets == []
    assert len(robot_context.cancel_reasons) == 1
    assert "stale" in robot_context.cancel_reasons[0]
    assert result["command_feedback"]["cancelled"] is True
    np.testing.assert_allclose(next_pose, np.eye(4))


@pytest.mark.parametrize("status", ["converged", "skipped", "error"])
def test_remote_iteration_no_command_statuses_never_execute(monkeypatch, status):
    args = _remote_args(execute=True)
    robot_context = FakeRobotContext(execute=True)

    def fake_send(server, body, *, timeout_s):
        del server, timeout_s
        return _tracking_response(body, status=status)

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, next_pose = _remote_fixture_call(args, robot_context)

    assert result["command_sent"] is False
    assert result["remote"]["action_executable"] is False
    assert robot_context.sent_targets == []
    assert len(robot_context.cancel_reasons) == 1
    assert result["command_feedback"]["cancelled"] is True
    np.testing.assert_allclose(next_pose, np.eye(4))


def test_remote_iteration_wrong_root_never_executes(monkeypatch):
    args = _remote_args(execute=True)
    robot_context = FakeRobotContext(execute=True)

    def fake_send(server, body, *, timeout_s):
        del server, timeout_s
        return _tracking_response(body, action={"root_link": "base"})

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, _next_pose = _remote_fixture_call(args, robot_context)

    assert result["command_sent"] is False
    assert result["remote"]["action_executable"] is False
    assert "root_link" in result["reason"]
    assert robot_context.sent_targets == []
    assert len(robot_context.cancel_reasons) == 1
    assert "root_link" in robot_context.cancel_reasons[0]


def test_remote_iteration_ok_false_never_executes(monkeypatch):
    args = _remote_args(execute=True)
    robot_context = FakeRobotContext(execute=True)

    def fake_send(server, body, *, timeout_s):
        del server, timeout_s
        return _tracking_response(body, ok=False, status="skipped", reason="segmentation failed")

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, _next_pose = _remote_fixture_call(args, robot_context)

    assert result["ok"] is False
    assert result["command_sent"] is False
    assert robot_context.sent_targets == []
    assert len(robot_context.cancel_reasons) == 1
    assert "segmentation failed" in robot_context.cancel_reasons[0]


def test_remote_fixture_does_not_stop_on_invalid_converged_status(monkeypatch, capsys):
    args = parse_args(
        [
            "--remote-fixture-request",
            "--remote-server",
            "127.0.0.1:8080",
            "--max-iterations",
            "2",
            "--debug",
        ]
    )
    calls = {"count": 0}

    def fake_process(*args_, **kwargs):
        del args_, kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return {"ok": False, "status": "converged", "frame_index": 0}, np.eye(4)
        return {"ok": True, "status": "converged", "frame_index": 1}, np.eye(4)

    monkeypatch.setattr("visual_servoing.visual_servo_client.process_remote_servo_iteration", fake_process)

    assert run_remote_fixture(args) == 0

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls["count"] == 2
    assert [line["ok"] for line in lines] == [False, True]


def test_default_output_is_concise_timing_summary():
    frame_start = 100.0
    result = {
        "ok": False,
        "status": "skipped",
        "frame_index": 205,
        "command_sent": False,
        "reason": "remote response not ok: No usable object mask was produced.",
        "remote": {
            "round_trip_ms": 172.149,
            "request_encode_ms": 6.845,
        },
        "observation": {
            "centroid_camera_m": [0.120039346, 0.085829463, 0.698559344],
        },
        "servo_step": {
            "desired_position_t5_m": [0.767559344, -0.111039346, 0.171243989],
            "current_t5_T_ee": [
                [1.0, 0.0, 0.0, 0.396052219],
                [0.0, 1.0, 0.0, -0.253822572],
                [0.0, 0.0, 1.0, -0.329319159],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "target_t5_T_ee": [
                [1.0, 0.0, 0.0, 0.413480065],
                [0.0, 1.0, 0.0, -0.247124440],
                [0.0, 0.0, 1.0, -0.305837140],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }

    summary = format_iteration_summary(result, frame_start)

    assert summary.startswith("frame=205 status=skipped ok=false")
    assert "action_latency_ms=172.1" in summary
    assert "encode_ms=6.8" in summary
    assert "cam_xyz_m=(0.120,0.086,0.699)" in summary
    assert "target_t5_xyz_m=(0.768,-0.111,0.171)" in summary
    assert "current_t5_xyz_m=(0.396,-0.254,-0.329)" in summary
    assert "ee_cmd_t5_xyz_m=(0.413,-0.247,-0.306)" in summary
    assert "cmd_delta_t5_m=(0.017,0.007,0.023)" in summary
    assert "command=skip" in summary
    assert "No usable object mask" in summary
    assert not summary.startswith("{")


def test_debug_output_prints_json(capsys):
    args = parse_args(["--live", "--debug"])
    result = {
        "ok": False,
        "status": "skipped",
        "frame_index": 1,
        "command_sent": False,
        "reason": "no mask",
        "mask": {"preview": {"encoding": "packbits-b64-v1"}, "area": 10},
    }

    emit_iteration_output(args, result, time.perf_counter())

    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_index"] == 1
    assert payload["mask"] == {"area": 10}


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "visual_servoing/visual_servo_client.py", "--help"],
        cwd="/home/kgs/FoundationPose",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--execute" in result.stdout
