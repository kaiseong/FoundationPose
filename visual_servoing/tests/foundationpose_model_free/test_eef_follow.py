from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visual_servoing.foundationpose_model_free.eef_follow import (
    EefFollowState,
    apply_eef_follow,
    object_T_ee_from_args,
    plan_full_pose_eef_step,
)
from visual_servoing.scripts.fp_track_live import parse_args
from visual_servoing.visual_servo_core import axis_angle_rotation


class FakeRobotContext:
    def __init__(self, current_t5_T_ee: np.ndarray) -> None:
        self._current_t5_T_ee = np.asarray(current_t5_T_ee, dtype=np.float64)
        self.sent_targets: list[np.ndarray] = []
        self.cancel_reasons: list[str] = []

    def current_ee_pose(self) -> np.ndarray:
        return self._current_t5_T_ee.copy()

    def send_right_arm_cartesian(self, target_t5_T_ee: np.ndarray) -> dict[str, object]:
        self.sent_targets.append(np.asarray(target_t5_T_ee, dtype=np.float64).copy())
        return {"status": "fake_sent"}

    def cancel_command_stream(self, reason: str) -> dict[str, object]:
        self.cancel_reasons.append(reason)
        return {"status": "fake_cancelled", "reason": reason}


def test_default_object_to_ee_rule_offsets_negative_object_x_and_flips_x_rotation():
    args = parse_args(["--object", "meter", "--remote-register-server", "192.168.0.3:8081", "--eef-follow"])

    object_T_ee = object_T_ee_from_args(args)

    assert np.allclose(object_T_ee[:3, 3], [-0.2, 0.0, 0.0], atol=1e-9)
    assert np.allclose(object_T_ee[:3, :3], axis_angle_rotation(np.array([1.0, 0.0, 0.0]), np.pi), atol=1e-9)


def test_full_pose_step_uses_object_frame_offset_and_clamps_translation_step():
    t5_T_object = np.eye(4, dtype=np.float64)
    t5_T_object[:3, :3] = axis_angle_rotation(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    t5_T_object[:3, 3] = [1.0, 2.0, 3.0]
    object_T_ee = np.eye(4, dtype=np.float64)
    object_T_ee[:3, 3] = [-0.2, 0.0, 0.0]
    object_T_ee[:3, :3] = axis_angle_rotation(np.array([1.0, 0.0, 0.0]), np.pi)
    current_t5_T_ee = np.eye(4, dtype=np.float64)
    current_t5_T_ee[:3, 3] = [1.0, 2.0, 3.0]

    step = plan_full_pose_eef_step(
        current_t5_T_ee=current_t5_T_ee,
        t5_T_object=t5_T_object,
        object_T_ee=object_T_ee,
        max_translation_step_m=0.03,
        max_rotation_step_rad=np.deg2rad(5.0),
        position_tolerance_m=0.005,
        rotation_tolerance_rad=np.deg2rad(2.0),
    )

    assert np.allclose(step.desired_t5_T_ee[:3, 3], [1.0, 1.8, 3.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(step.translation_step_m), 0.03, atol=1e-9)
    assert np.allclose(step.translation_step_m, [0.0, -0.03, 0.0], atol=1e-9)
    assert np.isclose(step.wrist_step_rad, np.deg2rad(5.0), atol=1e-9)


def test_apply_eef_follow_dry_run_plans_without_sending_robot_command():
    args = parse_args(["--object", "meter", "--remote-register-server", "192.168.0.3:8081", "--eef-follow"])
    robot_context = FakeRobotContext(np.eye(4, dtype=np.float64))

    payload = apply_eef_follow(
        args=args,
        robot_context=robot_context,  # type: ignore[arg-type]
        state=EefFollowState(),
        camera_T_object=np.eye(4, dtype=np.float64),
        t5_T_camera=np.eye(4, dtype=np.float64),
        tracking_state="TRACKING",
        pose_is_fresh=True,
    )

    assert payload is not None
    assert payload["ok"] is True
    assert payload["command_sent"] is False
    assert payload["execute"] is False
    assert "dry-run" in str(payload["reason"])
    assert robot_context.sent_targets == []
    step = payload["servo_step"]
    assert isinstance(step, dict)
    assert np.allclose(step["desired_position_t5_m"], [-0.2, 0.0, 0.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(step["translation_step_m"]), 0.03, atol=1e-9)


def test_apply_eef_follow_skips_when_tracking_is_not_fresh():
    args = SimpleNamespace(eef_follow=True, execute=True)
    robot_context = FakeRobotContext(np.eye(4, dtype=np.float64))

    payload = apply_eef_follow(
        args=args,  # type: ignore[arg-type]
        robot_context=robot_context,  # type: ignore[arg-type]
        state=EefFollowState(),
        camera_T_object=np.eye(4, dtype=np.float64),
        t5_T_camera=np.eye(4, dtype=np.float64),
        tracking_state="LOST",
        pose_is_fresh=True,
    )

    assert payload is not None
    assert payload["ok"] is False
    assert payload["command_sent"] is False
    assert "tracking state LOST" in str(payload["reason"])
    assert robot_context.sent_targets == []
    assert robot_context.cancel_reasons == ["tracking state LOST: no EEF command"]


def test_parse_args_rejects_eef_follow_outside_track_hybrid():
    try:
        parse_args(["--object", "meter", "--eef-follow"])
        raise AssertionError("expected --eef-follow without Track Hybrid to fail")
    except SystemExit as exc:
        assert "--eef-follow is supported only with Track Hybrid" in str(exc)
