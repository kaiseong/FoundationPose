from __future__ import annotations

import math

import numpy as np

from visual_servoing.visual_servo_core import (
    POSITION_ONLY_ORIENTATION_POLICY,
    REMOTE_ACTION_CONTROL_MODE,
    REMOTE_OFFSET_FRAME,
    RIGHT_ARM_CONTROL_ROOT_LINK,
    ServoLimits,
    apply_object_offset,
    make_transform_from_xyz_rpy,
    plan_t5_position_servo_action,
    plan_visual_servo_action,
    validate_remote_action,
)


def _response(target: np.ndarray, **overrides):
    payload = {
        "ok": True,
        "status": "tracking",
        "request_id": "req-1",
        "frame_index": 3,
        "action": {
            "root_link": RIGHT_ARM_CONTROL_ROOT_LINK,
            "ee_link": "link_right_arm_6",
            "control_mode": REMOTE_ACTION_CONTROL_MODE,
            "target_t5_T_ee": target.tolist(),
            "command_recommended": True,
        },
    }
    payload.update(overrides)
    return payload


def test_apply_object_offset_uses_object_frame_translation():
    t5_T_object = make_transform_from_xyz_rpy([1.0, 2.0, 3.0, 0.0, 0.0, 90.0])
    object_T_offset = make_transform_from_xyz_rpy([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    target = apply_object_offset(t5_T_object, object_T_offset)

    np.testing.assert_allclose(target[:3, 3], [1.0, 2.1, 3.0], atol=1e-12)


def test_plan_visual_servo_action_applies_object_offset_not_current_ee_frame():
    current = make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 90.0])
    t5_T_object = make_transform_from_xyz_rpy([1.0, 1.0, 0.0, 0.0, 0.0, 90.0])
    object_T_offset = make_transform_from_xyz_rpy([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    step = plan_visual_servo_action(
        current_t5_T_ee=current,
        t5_T_object=t5_T_object,
        object_T_offset=object_T_offset,
        limits=ServoLimits(max_translation_step_m=2.0),
        object_grasp_axis_t5=np.array([0.0, 1.0, 0.0]),
    )

    np.testing.assert_allclose(step.desired_position_t5_m, [1.0, 1.1, 0.0], atol=1e-12)


def test_plan_t5_position_servo_action_uses_t5_offset_and_fixed_zero_rpy():
    current = make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 90.0])

    step = plan_t5_position_servo_action(
        current_t5_T_ee=current,
        object_centroid_t5=np.array([1.0, 1.0, 0.0]),
        target_offset_t5=np.array([0.1, -0.2, 0.3]),
        limits=ServoLimits(max_translation_step_m=2.0),
    )

    assert REMOTE_OFFSET_FRAME == RIGHT_ARM_CONTROL_ROOT_LINK
    assert POSITION_ONLY_ORIENTATION_POLICY == "fixed_t5_rpy_zero"
    np.testing.assert_allclose(step.desired_position_t5_m, [1.1, 0.8, 0.3], atol=1e-12)
    np.testing.assert_allclose(step.target_t5_T_ee[:3, 3], [1.1, 0.8, 0.3], atol=1e-12)
    np.testing.assert_allclose(step.target_t5_T_ee[:3, :3], np.eye(3), atol=1e-12)
    assert math.isclose(step.wrist_error_rad, math.radians(90.0), abs_tol=1e-12)
    assert math.isclose(step.wrist_step_rad, math.radians(90.0), abs_tol=1e-12)


def test_plan_t5_position_servo_action_converges_on_position_and_zero_rpy():
    current = make_transform_from_xyz_rpy([1.001, 1.0, 1.0, 0.0, 0.0, 0.0])

    step = plan_t5_position_servo_action(
        current_t5_T_ee=current,
        object_centroid_t5=np.array([1.0, 1.0, 1.0]),
        target_offset_t5=np.zeros(3),
        limits=ServoLimits(position_tolerance_m=0.005),
    )

    assert step.status == "converged"
    assert step.command_recommended is False
    np.testing.assert_allclose(step.target_t5_T_ee[:3, :3], np.eye(3), atol=1e-12)


def test_validate_remote_action_accepts_tracking_with_bounded_target():
    current = np.eye(4)
    target = make_transform_from_xyz_rpy([0.01, 0.0, 0.0, 0.0, 0.0, 2.0])

    validation = validate_remote_action(
        _response(target),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.2,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=current,
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is True
    assert validation.executable is True
    np.testing.assert_allclose(validation.target_t5_T_ee, target)


def test_validate_remote_action_rejects_stale_response():
    validation = validate_remote_action(
        _response(np.eye(4)),
        request_id="req-1",
        frame_index=3,
        round_trip_s=1.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False
    assert "stale" in validation.reason


def test_validate_remote_action_rejects_mismatched_request_id():
    validation = validate_remote_action(
        _response(np.eye(4)),
        request_id="other",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "request_id" in validation.reason


def test_validate_remote_action_rejects_mismatched_frame_index():
    validation = validate_remote_action(
        _response(np.eye(4)),
        request_id="req-1",
        frame_index=4,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "frame_index" in validation.reason


def test_validate_remote_action_rejects_invalid_frame_index():
    validation = validate_remote_action(
        _response(np.eye(4), frame_index="not-an-int"),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "frame_index" in validation.reason


def test_validate_remote_action_uses_client_round_trip_not_server_timestamps():
    response = _response(make_transform_from_xyz_rpy([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    response["server_received_monotonic_ns"] = 0
    response["server_completed_monotonic_ns"] = 0

    validation = validate_remote_action(
        response,
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is True
    assert validation.executable is True


def test_validate_remote_action_rejects_wrong_root():
    response = _response(np.eye(4))
    response["action"]["root_link"] = "base"

    validation = validate_remote_action(
        response,
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False


def test_validate_remote_action_rejects_non_right_arm_ee_link():
    response = _response(np.eye(4))
    response["action"]["ee_link"] = "link_left_arm_6"

    validation = validate_remote_action(
        response,
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "right-arm" in validation.reason


def test_validate_remote_action_rejects_oversized_translation():
    validation = validate_remote_action(
        _response(make_transform_from_xyz_rpy([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "translation" in validation.reason


def test_validate_remote_action_rejects_oversized_wrist_step():
    validation = validate_remote_action(
        _response(make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 30.0])),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "wrist" in validation.reason


def test_validate_remote_action_accepts_fixed_zero_rpy_policy_without_wrist_step_limit():
    current = make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 90.0])
    response = _response(np.eye(4))
    response["action"]["orientation_policy"] = POSITION_ONLY_ORIENTATION_POLICY

    validation = validate_remote_action(
        response,
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=current,
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is True
    assert validation.executable is True
    np.testing.assert_allclose(validation.target_t5_T_ee[:3, :3], np.eye(3), atol=1e-12)


def test_validate_remote_action_rejects_fixed_zero_rpy_policy_with_non_identity_rotation():
    response = _response(make_transform_from_xyz_rpy([0.0, 0.0, 0.0, 0.0, 0.0, 10.0]))
    response["action"]["orientation_policy"] = POSITION_ONLY_ORIENTATION_POLICY

    validation = validate_remote_action(
        response,
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "identity rotation" in validation.reason


def test_validate_remote_action_rejects_non_finite_target():
    target = np.eye(4)
    target[0, 3] = np.nan

    validation = validate_remote_action(
        _response(target),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False


def test_validate_remote_action_rejects_non_rigid_target():
    target = np.eye(4)
    target[0, 0] = 2.0

    validation = validate_remote_action(
        _response(target),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert "orthonormal" in validation.reason


def test_validate_remote_action_rejects_ok_false_response():
    validation = validate_remote_action(
        _response(np.eye(4), ok=False, reason="segmentation failed"),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False
    assert "not ok" in validation.reason


def test_validate_remote_action_converged_is_no_command():
    validation = validate_remote_action(
        _response(np.eye(4), status="converged"),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is True
    assert validation.executable is False


def test_validate_remote_action_rejects_stale_converged_response():
    validation = validate_remote_action(
        _response(np.eye(4), status="converged"),
        request_id="req-1",
        frame_index=3,
        round_trip_s=1.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False
    assert "stale" in validation.reason


def test_validate_remote_action_rejects_mismatched_converged_response():
    validation = validate_remote_action(
        _response(np.eye(4), status="converged", request_id="old-request"),
        request_id="req-1",
        frame_index=3,
        round_trip_s=0.1,
        stale_action_max_age_s=1.0,
        current_t5_T_ee=np.eye(4),
        max_translation_step_m=0.02,
        max_wrist_step_rad=math.radians(5.0),
    )

    assert validation.ok is False
    assert validation.executable is False
    assert "request_id" in validation.reason


def test_validate_remote_action_skipped_and_error_are_no_command():
    for status in ("skipped", "error"):
        validation = validate_remote_action(
            _response(np.eye(4), status=status),
            request_id="req-1",
            frame_index=3,
            round_trip_s=0.1,
            stale_action_max_age_s=1.0,
            current_t5_T_ee=np.eye(4),
            max_translation_step_m=0.02,
            max_wrist_step_rad=math.radians(5.0),
        )

        assert validation.ok is True
        assert validation.executable is False
