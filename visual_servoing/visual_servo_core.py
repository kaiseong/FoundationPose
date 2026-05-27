"""Pure visual-servo geometry, planning, and remote-action validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import (
    CameraIntrinsics,
    backproject_masked_depth,
    estimate_phone_pose,
    normalize_vector,
)


RIGHT_ARM_CONTROL_ROOT_LINK = "link_torso_5"
DEFAULT_RIGHT_ARM_EE_LINK = "ee_right"
RIGHT_ARM_EE_LINKS = frozenset({"link_right_arm_6", DEFAULT_RIGHT_ARM_EE_LINK})
REMOTE_ACTION_CONTROL_MODE = "right_arm_cartesian"
REMOTE_OFFSET_FRAME = RIGHT_ARM_CONTROL_ROOT_LINK
POSITION_ONLY_ORIENTATION_POLICY = "fixed_t5_rpy_zero"
EXECUTABLE_REMOTE_STATUS = "tracking"
NO_COMMAND_REMOTE_STATUSES = frozenset({"converged", "skipped", "error"})


@dataclass(frozen=True)
class ServoLimits:
    max_translation_step_m: float = 0.01
    max_wrist_step_rad: float = math.radians(5.0)
    position_tolerance_m: float = 0.005
    wrist_tolerance_rad: float = math.radians(2.0)


@dataclass(frozen=True)
class VisualObservation:
    camera_T_object: np.ndarray
    t5_T_object: np.ndarray
    centroid_camera_m: np.ndarray
    object_long_axis_t5: np.ndarray
    object_grasp_axis_t5: np.ndarray
    masked_points: int


@dataclass(frozen=True)
class ServoStep:
    status: str
    current_t5_T_ee: np.ndarray
    target_t5_T_ee: np.ndarray
    desired_position_t5_m: np.ndarray
    position_error_m: np.ndarray
    translation_step_m: np.ndarray
    wrist_error_rad: float
    wrist_step_rad: float
    command_recommended: bool
    ignored_offset_rpy_deg: tuple[float, float]


@dataclass(frozen=True)
class RemoteActionValidation:
    ok: bool
    executable: bool
    reason: str
    target_t5_T_ee: np.ndarray | None = None


def estimate_visual_observation(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    t5_T_camera: np.ndarray,
    previous_transform: np.ndarray | None,
    min_depth_m: float,
    max_depth_m: float,
) -> VisualObservation:
    points, pixels = backproject_masked_depth(
        depth_m,
        mask,
        intrinsics,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    camera_T_object = estimate_phone_pose(
        points,
        pixels_xy=pixels,
        intrinsics=intrinsics,
        previous_transform=previous_transform,
    )
    t5_T_object = require_transform(t5_T_camera, "t5_T_camera") @ camera_T_object
    object_long_axis_t5 = normalize_vector(t5_T_object[:3, 1])
    object_grasp_axis_t5 = normalize_vector(t5_T_object[:3, 0])
    return VisualObservation(
        camera_T_object=camera_T_object,
        t5_T_object=t5_T_object,
        centroid_camera_m=camera_T_object[:3, 3].copy(),
        object_long_axis_t5=object_long_axis_t5,
        object_grasp_axis_t5=object_grasp_axis_t5,
        masked_points=int(points.shape[0]),
    )


def apply_object_offset(t5_T_object: np.ndarray, object_T_offset: np.ndarray) -> np.ndarray:
    return require_transform(t5_T_object, "t5_T_object") @ require_transform(object_T_offset, "object_T_offset")


def plan_t5_position_servo_action(
    *,
    current_t5_T_ee: np.ndarray,
    object_centroid_t5: np.ndarray,
    target_offset_t5: np.ndarray,
    limits: ServoLimits,
) -> ServoStep:
    current_t5_T_ee = require_transform(current_t5_T_ee, "current_t5_T_ee")
    object_centroid_t5 = require_vector3(object_centroid_t5, "object_centroid_t5")
    target_offset_t5 = require_vector3(target_offset_t5, "target_offset_t5")

    desired_position_t5_m = object_centroid_t5 + target_offset_t5
    position_error_m = desired_position_t5_m - current_t5_T_ee[:3, 3]
    translation_step_m = clamp_translation_step(
        position_error_m,
        max_step_m=limits.max_translation_step_m,
    )

    target_rotation = np.eye(3, dtype=np.float64)
    target_t5_T_ee = np.eye(4, dtype=np.float64)
    target_t5_T_ee[:3, 3] = current_t5_T_ee[:3, 3] + translation_step_m
    target_t5_T_ee[:3, :3] = target_rotation

    wrist_error_rad = rotation_angle(current_t5_T_ee[:3, :3].T @ target_rotation)
    converged = (
        float(np.linalg.norm(position_error_m)) <= limits.position_tolerance_m
        and abs(wrist_error_rad) <= limits.wrist_tolerance_rad
    )
    status = "converged" if converged else "tracking"
    command_recommended = not converged and (
        float(np.linalg.norm(translation_step_m)) > 1e-9 or abs(wrist_error_rad) > 1e-9
    )
    return ServoStep(
        status=status,
        current_t5_T_ee=current_t5_T_ee,
        target_t5_T_ee=target_t5_T_ee,
        desired_position_t5_m=desired_position_t5_m,
        position_error_m=position_error_m,
        translation_step_m=translation_step_m,
        wrist_error_rad=float(wrist_error_rad),
        wrist_step_rad=float(wrist_error_rad),
        command_recommended=command_recommended,
        ignored_offset_rpy_deg=(0.0, 0.0),
    )


def plan_visual_servo_action(
    *,
    current_t5_T_ee: np.ndarray,
    t5_T_object: np.ndarray,
    object_T_offset: np.ndarray,
    limits: ServoLimits,
    object_grasp_axis_t5: np.ndarray | None = None,
    ee_align_axis: str = "y",
    wrist_axis: str = "z",
) -> ServoStep:
    target_t5 = apply_object_offset(t5_T_object, object_T_offset)
    target_axis = object_grasp_axis_t5
    if target_axis is None:
        target_axis = target_t5[:3, 0]
    return plan_visual_servo_step(
        current_t5_T_ee=current_t5_T_ee,
        visual_target_t5=target_t5,
        ee_offset=np.eye(4, dtype=np.float64),
        ee_offset_rpy_deg=(0.0, 0.0, 0.0),
        limits=limits,
        object_grasp_axis_t5=target_axis,
        ee_align_axis=ee_align_axis,
        wrist_axis=wrist_axis,
    )


def plan_visual_servo_step(
    *,
    current_t5_T_ee: np.ndarray,
    visual_target_t5: np.ndarray,
    ee_offset: np.ndarray,
    ee_offset_rpy_deg: tuple[float, float, float],
    limits: ServoLimits,
    object_grasp_axis_t5: np.ndarray | None = None,
    ee_align_axis: str = "y",
    wrist_axis: str = "z",
) -> ServoStep:
    current_t5_T_ee = require_transform(current_t5_T_ee, "current_t5_T_ee")
    visual_target_t5 = require_transform(visual_target_t5, "visual_target_t5")
    ee_offset = require_transform(ee_offset, "ee_offset")

    current_rotation = current_t5_T_ee[:3, :3]
    offset_translation_t5 = current_rotation @ ee_offset[:3, 3]
    desired_position_t5_m = visual_target_t5[:3, 3] + offset_translation_t5
    position_error_m = desired_position_t5_m - current_t5_T_ee[:3, 3]
    translation_step_m = clamp_translation_step(
        position_error_m,
        max_step_m=limits.max_translation_step_m,
    )

    target_axis = object_grasp_axis_t5
    if target_axis is None:
        target_axis = visual_target_t5[:3, 0]
    target_axis = normalize_vector(np.asarray(target_axis, dtype=np.float64))
    current_axis = current_rotation @ local_axis(ee_align_axis)
    wrist_axis_t5 = current_rotation @ local_axis(wrist_axis)
    wrist_error_rad = signed_angle_about_axis(current_axis, target_axis, wrist_axis_t5)
    wrist_error_rad += math.radians(float(ee_offset_rpy_deg[2]))
    wrist_step_rad = clamp_scalar(wrist_error_rad, limits.max_wrist_step_rad)

    target_t5_T_ee = current_t5_T_ee.copy()
    target_t5_T_ee[:3, 3] = current_t5_T_ee[:3, 3] + translation_step_m
    target_t5_T_ee[:3, :3] = current_rotation @ axis_angle_rotation(local_axis(wrist_axis), wrist_step_rad)

    converged = (
        float(np.linalg.norm(position_error_m)) <= limits.position_tolerance_m
        and abs(wrist_error_rad) <= limits.wrist_tolerance_rad
    )
    status = "converged" if converged else "tracking"
    command_recommended = not converged and (
        float(np.linalg.norm(translation_step_m)) > 1e-9 or abs(wrist_step_rad) > 1e-9
    )
    return ServoStep(
        status=status,
        current_t5_T_ee=current_t5_T_ee,
        target_t5_T_ee=target_t5_T_ee,
        desired_position_t5_m=desired_position_t5_m,
        position_error_m=position_error_m,
        translation_step_m=translation_step_m,
        wrist_error_rad=float(wrist_error_rad),
        wrist_step_rad=float(wrist_step_rad),
        command_recommended=command_recommended,
        ignored_offset_rpy_deg=(float(ee_offset_rpy_deg[0]), float(ee_offset_rpy_deg[1])),
    )


def validate_remote_action(
    response: dict[str, Any],
    *,
    request_id: str,
    frame_index: int,
    round_trip_s: float,
    stale_action_max_age_s: float,
    current_t5_T_ee: np.ndarray,
    max_translation_step_m: float,
    max_wrist_step_rad: float,
    expected_root_link: str = RIGHT_ARM_CONTROL_ROOT_LINK,
    allowed_ee_links: frozenset[str] = RIGHT_ARM_EE_LINKS,
    rotation_tolerance: float = 1e-5,
) -> RemoteActionValidation:
    if not bool(response.get("ok", False)):
        return RemoteActionValidation(False, False, f"remote response not ok: {response.get('reason') or response.get('error')}")

    status = str(response.get("status", ""))
    if status != EXECUTABLE_REMOTE_STATUS and status not in NO_COMMAND_REMOTE_STATUSES:
        return RemoteActionValidation(False, False, f"remote status {status!r} is not executable")

    if str(response.get("request_id", "")) != str(request_id):
        return RemoteActionValidation(False, False, "remote request_id mismatch")
    try:
        response_frame_index = int(response.get("frame_index", -1))
    except (TypeError, ValueError):
        return RemoteActionValidation(False, False, "remote frame_index is invalid")
    if response_frame_index != int(frame_index):
        return RemoteActionValidation(False, False, "remote frame_index mismatch")
    if not math.isfinite(round_trip_s) or float(round_trip_s) > float(stale_action_max_age_s):
        return RemoteActionValidation(False, False, "remote action stale by client round-trip timing")

    if status in NO_COMMAND_REMOTE_STATUSES:
        return RemoteActionValidation(True, False, f"remote status {status}: no command")

    action = response.get("action")
    if not isinstance(action, dict):
        return RemoteActionValidation(False, False, "remote action missing")
    if action.get("control_mode") != REMOTE_ACTION_CONTROL_MODE:
        return RemoteActionValidation(False, False, "remote action control_mode mismatch")
    if action.get("root_link") != expected_root_link:
        return RemoteActionValidation(False, False, "remote action root_link mismatch")
    if action.get("ee_link") not in allowed_ee_links:
        return RemoteActionValidation(False, False, "remote action ee_link is not right-arm")
    if not bool(action.get("command_recommended", False)):
        return RemoteActionValidation(True, False, "remote action did not recommend command")

    try:
        current = require_transform(current_t5_T_ee, "current_t5_T_ee")
        target = require_rigid_transform(action.get("target_t5_T_ee"), "target_t5_T_ee", tolerance=rotation_tolerance)
    except ValueError as exc:
        return RemoteActionValidation(False, False, str(exc))

    translation_delta = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    if translation_delta > float(max_translation_step_m) + 1e-9:
        return RemoteActionValidation(False, False, "remote action translation step exceeds limit")

    orientation_policy = str(action.get("orientation_policy", ""))
    if orientation_policy == POSITION_ONLY_ORIENTATION_POLICY:
        if not np.allclose(target[:3, :3], np.eye(3), atol=rotation_tolerance):
            return RemoteActionValidation(False, False, "remote action fixed_t5_rpy_zero target is not identity rotation")
    else:
        rotation_delta = rotation_angle(current[:3, :3].T @ target[:3, :3])
        if rotation_delta > float(max_wrist_step_rad) + 1e-9:
            return RemoteActionValidation(False, False, "remote action wrist step exceeds limit")

    return RemoteActionValidation(True, True, "remote action validated", target_t5_T_ee=target)


def require_rigid_transform(value: Any, name: str, *, tolerance: float = 1e-5) -> np.ndarray:
    transform = require_transform(value, name)
    if not np.allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0], atol=tolerance):
        raise ValueError(f"{name} final row is not homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError(f"{name} rotation is not orthonormal")
    det = float(np.linalg.det(rotation))
    if abs(det - 1.0) > tolerance:
        raise ValueError(f"{name} rotation determinant is not +1")
    return transform


def require_transform(transform: Any, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    return transform


def require_vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def make_transform_from_xyz_rpy(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy(
        math.radians(float(values[3])),
        math.radians(float(values[4])),
        math.radians(float(values[5])),
    )
    transform[:3, 3] = values[:3]
    return transform


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [sy * cp, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize_vector(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def local_axis(name: str) -> np.ndarray:
    sign = -1.0 if name.startswith("-") else 1.0
    axis = name[1:] if name.startswith("-") else name
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    return sign * mapping[axis]


def clamp_translation_step(error: np.ndarray, *, max_step_m: float) -> np.ndarray:
    error = np.asarray(error, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(error))
    if norm <= max_step_m or norm < 1e-12:
        return error.copy()
    return error * (float(max_step_m) / norm)


def clamp_scalar(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def signed_angle_about_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    axis = normalize_vector(axis)
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source = source - axis * float(np.dot(source, axis))
    target = target - axis * float(np.dot(target, axis))
    if np.linalg.norm(source) < 1e-9 or np.linalg.norm(target) < 1e-9:
        return 0.0
    source = normalize_vector(source)
    target = normalize_vector(target)
    sine = float(np.dot(axis, np.cross(source, target)))
    cosine = float(np.dot(source, target))
    return math.atan2(sine, cosine)


def rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def to_list(vector: np.ndarray) -> list[float]:
    return np.asarray(vector, dtype=float).reshape(-1).tolist()
