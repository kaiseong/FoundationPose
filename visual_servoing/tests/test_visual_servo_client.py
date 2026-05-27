from __future__ import annotations

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from visual_servoing.visual_servo_client import (
    RIGHT_ARM_CONTROL_ROOT_LINK,
    RobotContext,
    ServoLimits,
    clamp_translation_step,
    make_transform_from_xyz_rpy,
    plan_visual_servo_step,
    process_remote_servo_iteration,
    parse_args,
    remote_request_metadata,
    run_remote_fixture,
    signed_angle_about_axis,
    synthetic_rgbd_fixture,
    validate_args,
)
from visual_servoing.visual_servo_core import REMOTE_ACTION_CONTROL_MODE
from visual_servoing.visual_servo_protocol import decode_visual_servo_request


class FakeRobotContext:
    def __init__(self, *, execute: bool = True):
        self.execute = execute
        self.sent_targets: list[np.ndarray] = []

    def current_ee_pose(self):
        return np.eye(4)

    def send_right_arm_cartesian(self, target_t5_T_ee):
        self.sent_targets.append(np.asarray(target_t5_T_ee, dtype=np.float64).copy())
        return {"finish_code": "ok"}


class FakeBuilder:
    def __getattr__(self, name):
        if name.startswith(("add_", "set_")):
            return self._chain
        raise AttributeError(name)

    def _chain(self, *args, **kwargs):
        del args, kwargs
        return self


class FakeCommandHandler:
    def get(self):
        return SimpleNamespace(finish_code="ok")


class FakeCommandRobot:
    def __init__(self):
        self.send_args = None
        self.cancel_calls = 0
        self.wait_calls: list[int] = []
        self.control_state = FakeRby.ControlManagerState.ControlState.Idle

    def send_command(self, *args):
        self.send_args = args
        return FakeCommandHandler()

    def get_control_manager_state(self):
        return SimpleNamespace(control_state=self.control_state)

    def cancel_control(self):
        self.cancel_calls += 1

    def wait_for_control_ready(self, timeout_ms):
        self.wait_calls.append(int(timeout_ms))
        return True


class FakeRby:
    class ControlManagerState:
        class ControlState:
            Idle = "idle"
            Running = "running"

    CartesianImpedanceControlCommandBuilder = FakeBuilder
    CommandHeaderBuilder = FakeBuilder
    RobotCommandBuilder = FakeBuilder
    ComponentBasedCommandBuilder = FakeBuilder
    BodyComponentBasedCommandBuilder = FakeBuilder


def _remote_args(*, execute: bool = True):
    argv = ["--live", "--remote-server", "127.0.0.1:8080"]
    if execute:
        argv += ["--execute", "--address", "127.0.0.1:50051"]
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


def test_send_right_arm_cartesian_uses_sdk_send_command_without_timeout_argument():
    args = parse_args(["--live"])
    robot = FakeCommandRobot()
    context = RobotContext(args, robot=robot, rby=FakeRby)

    feedback = context.send_right_arm_cartesian(np.eye(4))

    assert feedback == {"finish_code": "ok"}
    assert robot.send_args is not None
    assert len(robot.send_args) == 1
    assert robot.wait_calls == [1000]


def test_send_right_arm_cartesian_cancels_active_control_before_waiting():
    args = parse_args(["--live", "--control-ready-timeout-ms", "2500"])
    robot = FakeCommandRobot()
    robot.control_state = FakeRby.ControlManagerState.ControlState.Running
    context = RobotContext(args, robot=robot, rby=FakeRby)

    context.send_right_arm_cartesian(np.eye(4))

    assert robot.cancel_calls == 1
    assert robot.wait_calls == [2500]


def test_validate_execute_rejects_non_right_arm_ee_link():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--ee-link", "link_left_arm_6"])

    with pytest.raises(SystemExit, match="right-arm EE links"):
        validate_args(args)


def test_validate_execute_defaults_to_m_model_and_all_power_servo():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051"])

    assert args.model == "m"
    assert args.power == ".*"
    assert args.servo == ".*"
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
        return _tracking_response(body, target=target)

    monkeypatch.setattr("visual_servoing.visual_servo_client.send_remote_visual_servo_request", fake_send)

    result, next_pose = _remote_fixture_call(args, robot_context)

    assert result["command_sent"] is True
    assert result["remote"]["action_executable"] is True
    assert len(robot_context.sent_targets) == 1
    np.testing.assert_allclose(robot_context.sent_targets[0], target)
    np.testing.assert_allclose(next_pose, target)


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


def test_remote_fixture_does_not_stop_on_invalid_converged_status(monkeypatch, capsys):
    args = parse_args(
        [
            "--remote-fixture-request",
            "--remote-server",
            "127.0.0.1:8080",
            "--max-iterations",
            "2",
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
