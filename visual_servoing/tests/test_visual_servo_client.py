from __future__ import annotations

import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from visual_servoing.visual_servo_client import (
    RobotContext,
    ServoLimits,
    clamp_translation_step,
    make_transform_from_xyz_rpy,
    plan_visual_servo_step,
    parse_args,
    signed_angle_about_axis,
    validate_args,
)


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


def test_validate_execute_rejects_non_right_arm_ee_link():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--ee-link", "link_left_arm_6"])

    with pytest.raises(SystemExit, match="right-arm EE links"):
        validate_args(args)


def test_validate_execute_rejects_broad_power_servo():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--power", ".*"])

    with pytest.raises(SystemExit, match="--power must use a strict right-arm-only pattern"):
        validate_args(args)


def test_validate_execute_rejects_loose_right_power_pattern():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--power", "right.*"])

    with pytest.raises(SystemExit, match="--power must use a strict right-arm-only pattern"):
        validate_args(args)


def test_validate_execute_rejects_bright_substring_power_pattern():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--power", "bright_component"])

    with pytest.raises(SystemExit, match="--power must use a strict right-arm-only pattern"):
        validate_args(args)


def test_validate_execute_rejects_loose_right_servo_pattern():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051", "--servo", "right.*"])

    with pytest.raises(SystemExit, match="--servo must use a strict right-arm-only pattern"):
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


def test_validate_execute_accepts_right_arm_safe_defaults():
    args = parse_args(["--live", "--execute", "--address", "127.0.0.1:50051"])

    validate_args(args)


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
