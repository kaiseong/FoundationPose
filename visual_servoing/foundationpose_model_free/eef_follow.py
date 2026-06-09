"""Right-arm EEF follow helpers for FoundationPose pose tracking clients."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

import visual_servoing.visual_servo_core as servo_core
from visual_servoing.visual_servo_client import (
    CAMERA_POSE_PRESET_CHOICES,
    DEFAULT_CAMERA_MOUNT_LINK,
    DEFAULT_RIGHT_ARM_EE_LINK,
    DEFAULT_T5_HEAD_XYZ_RPY,
    ROBOT_MODEL,
    RIGHT_ARM_CONTROL_ROOT_LINK,
    RobotContext,
    current_t5_T_camera,
    fixed_t5_T_camera,
    resolve_head_camera_pose_args,
    validate_execute_safety,
)


TRACKING_STATE = "TRACKING"


@dataclass
class EefFollowState:
    last_t5_T_object: np.ndarray | None = None


@dataclass(frozen=True)
class EefFollowStep:
    status: str
    current_t5_T_ee: np.ndarray
    desired_t5_T_ee: np.ndarray
    target_t5_T_ee: np.ndarray
    t5_T_object: np.ndarray
    position_error_m: np.ndarray
    translation_step_m: np.ndarray
    wrist_error_rad: float
    wrist_step_rad: float
    command_recommended: bool


def add_eef_follow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eef-follow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plan right-arm EE targets from the tracked FoundationPose object pose. Default is off.",
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow real robot right-arm Cartesian commands when --eef-follow is enabled. Default is dry-run/off.",
    )
    parser.add_argument("--address", help="Robot address; required with --execute.")
    parser.add_argument("--model", default=ROBOT_MODEL, help="Robot model. Execute mode is fixed to m.")
    parser.add_argument("--power", default=".*", help="Power-on component pattern. Defaults to all components.")
    parser.add_argument("--servo", default=".*", help="Servo-on component pattern. Defaults to all components.")
    parser.add_argument("--control-root-link", default=RIGHT_ARM_CONTROL_ROOT_LINK)
    parser.add_argument("--ee-link", default=DEFAULT_RIGHT_ARM_EE_LINK)
    parser.add_argument("--command-min-time-s", type=float, default=0.25)
    parser.add_argument("--command-hold-time-s", type=float, default=0.5)
    parser.add_argument("--command-timeout-s", type=float, default=2.0)
    parser.add_argument("--command-priority", type=int, default=10)
    parser.add_argument("--control-ready-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--move-to-ready-on-connect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move only the right arm to a Cartesian-ready bent pose before tracking commands.",
    )
    parser.add_argument("--ready-min-time-s", type=float, default=3.0)
    parser.add_argument("--ready-hold-time-s", type=float, default=4.0)
    parser.add_argument("--linear-limit", type=float, default=1.0)
    parser.add_argument("--angular-limit", type=float, default=np.pi / 2.0)
    parser.add_argument("--acceleration-limit-scaling", type=float, default=1.0)
    parser.add_argument("--max-translation-step-m", type=float, default=0.03)
    parser.add_argument("--max-rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--max-object-pose-jump-m", type=float, default=0.05)
    parser.add_argument("--max-object-pose-jump-deg", type=float, default=20.0)
    parser.add_argument(
        "--target-offset-object",
        type=float,
        nargs=3,
        default=(-0.2, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Object-frame offset from object origin to desired EE target. Default is -0.2 m along object x.",
    )
    parser.add_argument(
        "--target-rpy-object-deg",
        type=float,
        nargs=3,
        default=(180.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Object-frame RPY rotation from object frame to desired EE frame. Default is Rx(180 deg).",
    )
    parser.add_argument(
        "--current-ee-pose",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Dry-run current EE pose in T5 frame: meters and degrees.",
    )
    parser.add_argument(
        "--t5-head-pose",
        type=float,
        nargs=6,
        default=DEFAULT_T5_HEAD_XYZ_RPY,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Fixed T5-to-head pose for non-execute camera geometry.",
    )
    parser.add_argument(
        "--camera-pose-preset",
        choices=CAMERA_POSE_PRESET_CHOICES,
        default="auto",
        help="Camera mount pose preset. auto selects zed for --camera zed and realsense otherwise.",
    )
    parser.add_argument(
        "--head-camera-pose",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Override camera-mount-link-to-camera pose: meters and degrees.",
    )
    parser.add_argument(
        "--camera-mount-link",
        default=DEFAULT_CAMERA_MOUNT_LINK,
        help="Robot link the camera bracket is mounted to. Execute mode uses live FK from this link to T5.",
    )


def resolve_and_validate_eef_follow_args(args: argparse.Namespace) -> None:
    resolve_head_camera_pose_args(args)
    if bool(getattr(args, "execute", False)) and not bool(getattr(args, "eef_follow", False)):
        raise SystemExit("--execute for fp_track_live requires --eef-follow")
    if bool(getattr(args, "execute", False)):
        validate_execute_safety(args)
    if float(getattr(args, "max_translation_step_m", 0.0)) <= 0.0:
        raise SystemExit("--max-translation-step-m must be positive")
    if float(getattr(args, "max_rotation_step_deg", 0.0)) <= 0.0:
        raise SystemExit("--max-rotation-step-deg must be positive")
    if float(getattr(args, "position_tolerance_m", 0.0)) < 0.0:
        raise SystemExit("--position-tolerance-m must be non-negative")
    if float(getattr(args, "rotation_tolerance_deg", 0.0)) < 0.0:
        raise SystemExit("--rotation-tolerance-deg must be non-negative")
    if float(getattr(args, "max_object_pose_jump_m", 0.0)) <= 0.0:
        raise SystemExit("--max-object-pose-jump-m must be positive")
    if float(getattr(args, "max_object_pose_jump_deg", 0.0)) <= 0.0:
        raise SystemExit("--max-object-pose-jump-deg must be positive")


def make_robot_context(args: argparse.Namespace) -> RobotContext:
    return RobotContext.connect(args) if bool(getattr(args, "execute", False)) else RobotContext.dry_run(args)


def t5_T_camera_for_frame(
    args: argparse.Namespace,
    robot_context: RobotContext,
    *,
    explicit_t5_T_camera: np.ndarray | None = None,
    default_identity: bool = False,
) -> np.ndarray:
    if explicit_t5_T_camera is not None:
        return explicit_t5_T_camera
    if bool(getattr(args, "execute", False)):
        return current_t5_T_camera(args, robot_context)
    if default_identity:
        return np.eye(4, dtype=np.float64)
    return fixed_t5_T_camera(args)


def parse_transform_json(raw: str | None) -> np.ndarray | None:
    if raw is None or not str(raw).strip():
        return None
    matrix = np.asarray(json.loads(str(raw)), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"t5_T_camera must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("t5_T_camera contains non-finite values")
    return matrix


def object_T_ee_from_args(args: argparse.Namespace) -> np.ndarray:
    offset = np.asarray(getattr(args, "target_offset_object", (-0.2, 0.0, 0.0)), dtype=np.float64).reshape(3)
    rpy = np.asarray(getattr(args, "target_rpy_object_deg", (180.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
    return servo_core.make_transform_from_xyz_rpy((*offset.tolist(), *rpy.tolist()))


def apply_eef_follow(
    *,
    args: argparse.Namespace,
    robot_context: RobotContext,
    state: EefFollowState,
    camera_T_object: np.ndarray | None,
    t5_T_camera: np.ndarray,
    tracking_state: str,
    pose_is_fresh: bool,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "eef_follow", False)):
        return None
    if str(tracking_state) != TRACKING_STATE:
        return _skip_with_cancel(args, robot_context, f"tracking state {tracking_state}: no EEF command")
    if camera_T_object is None:
        return _skip_with_cancel(args, robot_context, "tracked object pose unavailable")
    if not pose_is_fresh:
        return _skip_with_cancel(args, robot_context, "tracked object pose is not fresh")

    try:
        t5_T_object = servo_core.require_transform(t5_T_camera, "t5_T_camera") @ servo_core.require_transform(
            camera_T_object,
            "camera_T_object",
        )
        object_T_ee = object_T_ee_from_args(args)
    except ValueError as exc:
        return _skip_with_cancel(args, robot_context, str(exc))

    jump_reason = object_pose_jump_reason(args, state.last_t5_T_object, t5_T_object)
    if jump_reason is not None:
        return _skip_with_cancel(args, robot_context, jump_reason, t5_T_object=t5_T_object)

    current_t5_T_ee = robot_context.current_ee_pose()
    step = plan_full_pose_eef_step(
        current_t5_T_ee=current_t5_T_ee,
        t5_T_object=t5_T_object,
        object_T_ee=object_T_ee,
        max_translation_step_m=float(args.max_translation_step_m),
        max_rotation_step_rad=math.radians(float(args.max_rotation_step_deg)),
        position_tolerance_m=float(args.position_tolerance_m),
        rotation_tolerance_rad=math.radians(float(args.rotation_tolerance_deg)),
    )
    state.last_t5_T_object = t5_T_object.copy()

    command_sent = False
    command_feedback = None
    if step.command_recommended and bool(getattr(args, "execute", False)):
        command_feedback = robot_context.send_right_arm_cartesian(step.target_t5_T_ee)
        command_sent = True
        reason = "right-arm Cartesian command sent"
    elif step.command_recommended:
        reason = "dry-run right-arm Cartesian target planned"
    else:
        reason = "converged within configured tolerances"
        if bool(getattr(args, "execute", False)):
            command_feedback = robot_context.cancel_command_stream(reason)

    payload: dict[str, Any] = {
        "ok": True,
        "command_sent": command_sent,
        "execute": bool(getattr(args, "execute", False)),
        "reason": reason,
        "servo_step": eef_step_payload(step),
    }
    if command_feedback is not None:
        payload["command_feedback"] = command_feedback
    return payload


def object_pose_jump_reason(args: argparse.Namespace, previous: np.ndarray | None, current: np.ndarray) -> str | None:
    if previous is None:
        return None
    previous = servo_core.require_transform(previous, "previous_t5_T_object")
    current = servo_core.require_transform(current, "current_t5_T_object")
    translation_jump = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    rotation_jump = servo_core.rotation_angle(previous[:3, :3].T @ current[:3, :3])
    if translation_jump > float(args.max_object_pose_jump_m) + 1e-9:
        return f"object pose translation jump {translation_jump:.3f}m exceeds limit"
    if rotation_jump > math.radians(float(args.max_object_pose_jump_deg)) + 1e-9:
        return f"object pose rotation jump {math.degrees(rotation_jump):.1f}deg exceeds limit"
    return None


def plan_full_pose_eef_step(
    *,
    current_t5_T_ee: np.ndarray,
    t5_T_object: np.ndarray,
    object_T_ee: np.ndarray,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> EefFollowStep:
    current_t5_T_ee = servo_core.require_transform(current_t5_T_ee, "current_t5_T_ee")
    t5_T_object = servo_core.require_transform(t5_T_object, "t5_T_object")
    object_T_ee = servo_core.require_transform(object_T_ee, "object_T_ee")
    desired_t5_T_ee = t5_T_object @ object_T_ee

    position_error_m = desired_t5_T_ee[:3, 3] - current_t5_T_ee[:3, 3]
    translation_step_m = servo_core.clamp_translation_step(position_error_m, max_step_m=max_translation_step_m)

    current_rotation = current_t5_T_ee[:3, :3]
    desired_rotation = desired_t5_T_ee[:3, :3]
    relative_rotation = current_rotation.T @ desired_rotation
    wrist_error_rad = servo_core.rotation_angle(relative_rotation)
    wrist_step_rad = min(abs(float(max_rotation_step_rad)), wrist_error_rad)

    target_t5_T_ee = np.eye(4, dtype=np.float64)
    target_t5_T_ee[:3, 3] = current_t5_T_ee[:3, 3] + translation_step_m
    if wrist_error_rad <= rotation_tolerance_rad:
        target_t5_T_ee[:3, :3] = current_rotation
        wrist_step_rad = 0.0
    elif wrist_step_rad >= wrist_error_rad - 1e-12:
        target_t5_T_ee[:3, :3] = desired_rotation
    else:
        axis = rotation_axis(relative_rotation, wrist_error_rad)
        target_t5_T_ee[:3, :3] = current_rotation @ servo_core.axis_angle_rotation(axis, wrist_step_rad)

    converged = (
        float(np.linalg.norm(position_error_m)) <= float(position_tolerance_m)
        and wrist_error_rad <= float(rotation_tolerance_rad)
    )
    command_recommended = not converged and (
        float(np.linalg.norm(translation_step_m)) > 1e-9 or abs(wrist_step_rad) > 1e-9
    )
    return EefFollowStep(
        status="converged" if converged else "tracking",
        current_t5_T_ee=current_t5_T_ee.copy(),
        desired_t5_T_ee=desired_t5_T_ee,
        target_t5_T_ee=target_t5_T_ee,
        t5_T_object=t5_T_object.copy(),
        position_error_m=position_error_m,
        translation_step_m=translation_step_m,
        wrist_error_rad=float(wrist_error_rad),
        wrist_step_rad=float(wrist_step_rad),
        command_recommended=bool(command_recommended),
    )


def rotation_axis(rotation: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(math.pi - angle) < 1e-5:
        diag = np.diag(rotation)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0))
        index = int(np.argmax(axis))
        if axis[index] < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if index == 0:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif index == 1:
            axis[0] = math.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        else:
            axis[0] = math.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = math.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        return servo_core.normalize_vector(axis)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return servo_core.normalize_vector(axis)


def _skip_with_cancel(
    args: argparse.Namespace,
    robot_context: RobotContext,
    reason: str,
    *,
    t5_T_object: np.ndarray | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "command_sent": False,
        "execute": bool(getattr(args, "execute", False)),
        "reason": reason,
    }
    if t5_T_object is None:
        payload["t5_T_object"] = None
    else:
        payload["t5_T_object"] = servo_core.matrix_list(t5_T_object)
    if bool(getattr(args, "execute", False)):
        payload["command_feedback"] = robot_context.cancel_command_stream(reason)
    return payload


def eef_step_payload(step: EefFollowStep) -> dict[str, Any]:
    return {
        "t5_T_object": servo_core.matrix_list(step.t5_T_object),
        "desired_t5_T_ee": servo_core.matrix_list(step.desired_t5_T_ee),
        "position_error_m": step.position_error_m.tolist(),
        "translation_step_m": step.translation_step_m.tolist(),
        "current_t5_T_ee": servo_core.matrix_list(step.current_t5_T_ee),
        "target_t5_T_ee": servo_core.matrix_list(step.target_t5_T_ee),
        "desired_position_t5_m": step.desired_t5_T_ee[:3, 3].tolist(),
        "status": step.status,
        "wrist_error_rad": step.wrist_error_rad,
        "wrist_step_rad": step.wrist_step_rad,
        "command_recommended": bool(step.command_recommended),
    }
