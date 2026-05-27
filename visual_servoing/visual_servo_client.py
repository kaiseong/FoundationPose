#!/usr/bin/env python3
"""Baseline point-cloud visual servo client.

The default path is intentionally dry-run only. Real robot motion is enabled
only when --execute is provided.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_ADDED_PACKAGE_PARENT: str | None = None
if __package__ in (None, ""):
    _PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
        _ADDED_PACKAGE_PARENT = _PACKAGE_PARENT

from visual_servoing.point_pose.live_camera_config import SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera
from visual_servoing.point_pose.rgbd_geometry import (
    CameraIntrinsics,
    backproject_masked_depth,
    estimate_phone_pose,
    normalize_vector,
)
from visual_servoing.point_pose.sam3_phone_segmenter import Sam3PhoneSegmenter, load_mask

if _ADDED_PACKAGE_PARENT is not None:
    sys.path.remove(_ADDED_PACKAGE_PARENT)


HEAD_TO_CAMERA_XYZ_RPY = (0.047, 0.009, 0.057, -90.0, 0.0, -90.0)
DEFAULT_RIGHT_ARM_STIFFNESS = (90.0, 90.0, 90.0, 70.0, 70.0, 70.0, 70.0)
DEFAULT_RIGHT_ARM_TORQUE_LIMIT = (40.0, 40.0, 40.0, 30.0, 30.0, 30.0, 30.0)
RIGHT_ARM_CONTROL_ROOT_LINK = "link_torso_5"
RIGHT_ARM_EE_LINKS = frozenset({"link_right_arm_6", "ee_right"})
RIGHT_ARM_POWER_SERVO_PATTERNS = frozenset({"right_arm.*", "^right_arm.*$"})


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-cloud baseline visual servo client.")
    live = parser.add_mutually_exclusive_group()
    live.add_argument("--live", action="store_true", help="Use live RGB-D camera selected by --camera.")
    live.add_argument("--live-d405", action="store_true", help="Use live D405 camera.")
    live.add_argument("--live-d435", action="store_true", help="Use live D435 camera.")
    live.add_argument("--live-zed", action="store_true", help="Use live ZED camera.")

    parser.add_argument("--rgb", help="Offline RGB image path. Optional when --mask is provided.")
    parser.add_argument("--depth", help="Offline depth path (.npy meters, or image scaled by --depth-scale).")
    parser.add_argument("--mask", help="Offline binary mask path. Required for offline mode.")
    parser.add_argument("--intrinsics", help="Offline intrinsics JSON path.")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Image depth unit to meters.")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.0)

    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--prompt", default="object", help="SAM3 prompt for live segmentation.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="SAM3 device.")
    parser.add_argument("--threshold", type=float, default=0.5, help="SAM3 confidence threshold.")
    parser.add_argument("--sam-resolution", type=int, default=1008)

    parser.add_argument("--max-iterations", type=int, default=1, help="0 means run until interrupted.")
    parser.add_argument("--loop-sleep-s", type=float, default=0.0)
    parser.add_argument("--print-json", action="store_true", help="Accepted for compatibility; JSON is always printed.")
    parser.add_argument("--no-window", action="store_true", help="Accepted for compatibility; this client prints diagnostics.")

    parser.add_argument(
        "--ee-offset",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Current-EE-frame relative offset: meters and degrees.",
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
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Fixed T5-to-head pose for dry-run/live geometry. Defaults to identity.",
    )
    parser.add_argument(
        "--head-camera-pose",
        type=float,
        nargs=6,
        default=HEAD_TO_CAMERA_XYZ_RPY,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Head-to-camera pose: meters and degrees.",
    )
    parser.add_argument("--max-translation-step-m", type=float, default=0.01)
    parser.add_argument("--max-wrist-step-deg", type=float, default=5.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--wrist-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--ee-align-axis", default="y", choices=["x", "y", "z", "-x", "-y", "-z"])
    parser.add_argument("--wrist-axis", default="z", choices=["x", "y", "z", "-x", "-y", "-z"])

    parser.add_argument("--execute", action="store_true", help="Allow real robot right-arm Cartesian commands.")
    parser.add_argument("--address", help="Robot address; required with --execute.")
    parser.add_argument("--model", default="a", help="Robot model.")
    parser.add_argument("--power", default="right_arm.*")
    parser.add_argument("--servo", default="right_arm.*")
    parser.add_argument("--control-root-link", default=RIGHT_ARM_CONTROL_ROOT_LINK)
    parser.add_argument("--ee-link", default="link_right_arm_6")
    parser.add_argument("--command-min-time-s", type=float, default=0.25)
    parser.add_argument("--command-timeout-s", type=float, default=2.0)
    parser.add_argument("--linear-limit", type=float, default=1.0)
    parser.add_argument("--angular-limit", type=float, default=math.pi / 2.0)
    parser.add_argument("--linear-gain", type=float, default=50.0)
    parser.add_argument("--angular-gain", type=float, default=math.pi * 20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    validate_args(args)
    if is_live_mode(args):
        return run_live(args)
    return run_offline(args)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_iterations < 0:
        raise SystemExit("--max-iterations must be >= 0")
    if args.execute:
        validate_execute_safety(args)
    if not is_live_mode(args):
        missing = [name for name in ("depth", "mask", "intrinsics") if not getattr(args, name)]
        if missing:
            raise SystemExit(f"offline mode requires: {', '.join('--' + name for name in missing)}")


def validate_execute_safety(args: argparse.Namespace) -> None:
    if not args.address:
        raise SystemExit("--address is required with --execute")
    if args.control_root_link != RIGHT_ARM_CONTROL_ROOT_LINK:
        raise SystemExit(
            f"--execute is restricted to --control-root-link {RIGHT_ARM_CONTROL_ROOT_LINK!r}; "
            "torso/head/base frames are out of scope for this baseline"
        )
    if args.ee_link not in RIGHT_ARM_EE_LINKS:
        allowed = ", ".join(sorted(RIGHT_ARM_EE_LINKS))
        raise SystemExit(f"--execute is restricted to right-arm EE links: {allowed}")
    validate_right_arm_regex("--power", args.power)
    validate_right_arm_regex("--servo", args.servo)


def validate_right_arm_regex(flag: str, value: str) -> None:
    normalized = str(value).strip().lower()
    if normalized not in RIGHT_ARM_POWER_SERVO_PATTERNS:
        allowed = ", ".join(sorted(RIGHT_ARM_POWER_SERVO_PATTERNS))
        raise SystemExit(
            f"{flag} must use a strict right-arm-only pattern when --execute is used; "
            f"got {value!r}; allowed: {allowed}"
        )


def is_live_mode(args: argparse.Namespace) -> bool:
    return bool(args.live or args.live_d405 or args.live_d435 or args.live_zed)


def run_offline(args: argparse.Namespace) -> int:
    rgb = read_rgb(args.rgb) if args.rgb else None
    depth_m = read_depth(args.depth, depth_scale=args.depth_scale)
    mask_shape = rgb.shape[:2] if rgb is not None else depth_m.shape[:2]
    mask = load_mask(args.mask, shape=mask_shape)
    intrinsics = read_intrinsics(args.intrinsics)
    robot_context = RobotContext.connect(args) if args.execute else RobotContext.dry_run(args)
    try:
        previous_object_transform = None
        current_t5_T_ee = robot_context.current_ee_pose()
        t5_T_camera = fixed_t5_T_camera(args)

        for frame_index in iteration_range(args.max_iterations):
            result, previous_object_transform, current_t5_T_ee = process_servo_iteration(
                args,
                depth_m=depth_m,
                mask=mask,
                intrinsics=intrinsics,
                t5_T_camera=t5_T_camera,
                current_t5_T_ee=current_t5_T_ee,
                previous_object_transform=previous_object_transform,
                robot_context=robot_context,
                frame_index=frame_index,
            )
            print(json.dumps(result, separators=(",", ":")))
            if result.get("status") == "converged":
                break
            if args.loop_sleep_s > 0.0:
                time.sleep(args.loop_sleep_s)
    finally:
        robot_context.close()
    return 0


def run_live(args: argparse.Namespace) -> int:
    camera_model = selected_camera_model(args)
    segmenter = Sam3PhoneSegmenter(
        prompt=args.prompt,
        device=args.device,
        confidence_threshold=args.threshold,
        resolution=args.sam_resolution,
    )
    robot_context = RobotContext.connect(args) if args.execute else RobotContext.dry_run(args)
    try:
        previous_object_transform = None
        current_t5_T_ee = robot_context.current_ee_pose()
        t5_T_camera = fixed_t5_T_camera(args)
        with LiveRgbdCamera(
            model=camera_model,
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ) as camera:
            for frame_index in iteration_range(args.max_iterations):
                frame = camera.read(timeout_ms=args.frame_timeout_ms)
                try:
                    selection = segmenter.segment(frame.rgb)
                    result, previous_object_transform, current_t5_T_ee = process_servo_iteration(
                        args,
                        depth_m=frame.depth_m,
                        mask=selection.mask,
                        intrinsics=frame.intrinsics,
                        t5_T_camera=t5_T_camera,
                        current_t5_T_ee=current_t5_T_ee,
                        previous_object_transform=previous_object_transform,
                        robot_context=robot_context,
                        frame_index=frame_index,
                    )
                    result["mask"] = {
                        "index": selection.index,
                        "score": selection.score,
                        "area": selection.area,
                        "box_xyxy": selection.box_xyxy,
                    }
                except Exception as exc:
                    result = skipped_result(args, frame_index, str(exc))
                print(json.dumps(result, separators=(",", ":")))
                if result.get("status") == "converged":
                    break
                if args.loop_sleep_s > 0.0:
                    time.sleep(args.loop_sleep_s)
    finally:
        robot_context.close()
    return 0


def process_servo_iteration(
    args: argparse.Namespace,
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    t5_T_camera: np.ndarray,
    current_t5_T_ee: np.ndarray,
    previous_object_transform: np.ndarray | None,
    robot_context: "RobotContext",
    frame_index: int,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray]:
    try:
        observation = estimate_visual_observation(
            depth_m,
            mask,
            intrinsics,
            t5_T_camera=t5_T_camera,
            previous_transform=previous_object_transform,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if robot_context.execute:
            current_t5_T_ee = robot_context.current_ee_pose()
        limits = ServoLimits(
            max_translation_step_m=args.max_translation_step_m,
            max_wrist_step_rad=math.radians(args.max_wrist_step_deg),
            position_tolerance_m=args.position_tolerance_m,
            wrist_tolerance_rad=math.radians(args.wrist_tolerance_deg),
        )
        ee_offset = make_transform_from_xyz_rpy(args.ee_offset)
        step = plan_visual_servo_step(
            current_t5_T_ee=current_t5_T_ee,
            visual_target_t5=observation.t5_T_object,
            ee_offset=ee_offset,
            ee_offset_rpy_deg=tuple(float(value) for value in args.ee_offset[3:6]),
            limits=limits,
            object_grasp_axis_t5=observation.object_grasp_axis_t5,
            ee_align_axis=args.ee_align_axis,
            wrist_axis=args.wrist_axis,
        )
        command_sent = False
        command_feedback: dict[str, Any] | None = None
        reason = command_reason(args, step)
        if args.execute and step.command_recommended:
            command_feedback = robot_context.send_right_arm_cartesian(step.target_t5_T_ee)
            command_sent = True
            reason = "right-arm Cartesian command sent"
        result = diagnostic_payload(
            args,
            frame_index=frame_index,
            observation=observation,
            step=step,
            command_sent=command_sent,
            reason=reason,
            command_feedback=command_feedback,
        )
        return result, observation.camera_T_object, step.target_t5_T_ee
    except Exception as exc:
        return skipped_result(args, frame_index, str(exc)), previous_object_transform, current_t5_T_ee


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
    t5_T_object = t5_T_camera @ camera_T_object
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


def command_reason(args: argparse.Namespace, step: ServoStep) -> str:
    if step.status == "converged":
        return "converged within configured tolerances"
    if not args.execute:
        return "dry-run: --execute not set"
    if not step.command_recommended:
        return "no command recommended"
    return "ready to send right-arm Cartesian command"


def diagnostic_payload(
    args: argparse.Namespace,
    *,
    frame_index: int,
    observation: VisualObservation,
    step: ServoStep,
    command_sent: bool,
    reason: str,
    command_feedback: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "status": step.status,
        "frame_index": frame_index,
        "execute": bool(args.execute),
        "command_sent": command_sent,
        "reason": reason,
        "control": {
            "root_link": args.control_root_link,
            "ee_link": args.ee_link,
            "dofs": "xyz_plus_wrist_only",
            "ee_align_axis": args.ee_align_axis,
            "wrist_axis": args.wrist_axis,
        },
        "observation": {
            "masked_points": observation.masked_points,
            "centroid_camera_m": to_list(observation.centroid_camera_m),
            "object_long_axis_t5": to_list(observation.object_long_axis_t5),
            "object_grasp_axis_t5": to_list(observation.object_grasp_axis_t5),
            "camera_T_object": matrix_list(observation.camera_T_object),
            "t5_T_object": matrix_list(observation.t5_T_object),
        },
        "servo_step": {
            "desired_position_t5_m": to_list(step.desired_position_t5_m),
            "position_error_m": to_list(step.position_error_m),
            "translation_step_m": to_list(step.translation_step_m),
            "wrist_error_rad": step.wrist_error_rad,
            "wrist_step_rad": step.wrist_step_rad,
            "ignored_offset_rpy_deg": list(step.ignored_offset_rpy_deg),
            "current_t5_T_ee": matrix_list(step.current_t5_T_ee),
            "target_t5_T_ee": matrix_list(step.target_t5_T_ee),
        },
    }
    if command_feedback is not None:
        payload["command_feedback"] = command_feedback
    return payload


def skipped_result(args: argparse.Namespace, frame_index: int, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "skipped",
        "frame_index": frame_index,
        "execute": bool(args.execute),
        "command_sent": False,
        "reason": reason,
    }


class RobotContext:
    def __init__(self, args: argparse.Namespace, *, robot=None, rby=None) -> None:
        self.args = args
        self.robot = robot
        self.rby = rby

    @classmethod
    def dry_run(cls, args: argparse.Namespace) -> "RobotContext":
        return cls(args)

    @classmethod
    def connect(cls, args: argparse.Namespace) -> "RobotContext":
        rby = require_rby()
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"Failed to connect robot {args.address}")
        if not robot.is_power_on(args.power) and not robot.power_on(args.power):
            raise RuntimeError(f"Failed to turn power ({args.power}) on")
        if not robot.is_servo_on(args.servo) and not robot.servo_on(args.servo):
            raise RuntimeError(f"Failed to servo ({args.servo}) on")
        if robot.get_control_manager_state().state in [
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ]:
            if not robot.reset_fault_control_manager():
                raise RuntimeError("Failed to reset control manager")
        if not robot.enable_control_manager():
            raise RuntimeError("Failed to enable control manager")
        return cls(args, robot=robot, rby=rby)

    @property
    def execute(self) -> bool:
        return self.robot is not None

    def current_ee_pose(self) -> np.ndarray:
        if self.robot is None:
            return make_transform_from_xyz_rpy(self.args.current_ee_pose)
        return compute_fk(self.robot, self.args.ee_link, self.args.control_root_link)

    def send_right_arm_cartesian(self, target_t5_T_ee: np.ndarray) -> dict[str, Any]:
        if self.robot is None or self.rby is None:
            raise RuntimeError("Robot is not connected.")
        rby = self.rby
        builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(30))
            .set_minimum_time(float(self.args.command_min_time_s))
            .set_joint_stiffness(list(DEFAULT_RIGHT_ARM_STIFFNESS))
            .set_joint_torque_limit(list(DEFAULT_RIGHT_ARM_TORQUE_LIMIT))
            .add_joint_limit("right_arm_3", -2.6, -0.5)
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(0.6)
        )
        builder.add_target(
            self.args.control_root_link,
            self.args.ee_link,
            target_t5_T_ee,
            float(self.args.linear_limit),
            float(self.args.angular_limit),
            float(self.args.linear_gain),
            float(self.args.angular_gain),
        )
        command = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder().set_right_arm_command(builder)
            )
        )
        feedback = self.robot.send_command(command, float(self.args.command_timeout_s)).get()
        finish_code = getattr(feedback, "finish_code", None)
        return {"finish_code": str(finish_code)}

    def close(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()


def compute_fk(robot, ee_link: str, base_link: str) -> np.ndarray:
    robot_state = robot.get_state()
    q_full = robot_state.position
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state([base_link, ee_link], robot.model().robot_joint_names)
    dyn_state.set_q(q_full)
    dyn_model.compute_forward_kinematics(dyn_state)
    return np.asarray(dyn_model.compute_transformation(dyn_state, 0, 1), dtype=np.float64)


def require_rby():
    try:
        import rby1_sdk as rby  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on robot SDK environment
        raise RuntimeError("rby1_sdk is required only when --execute is used.") from exc
    return rby


def fixed_t5_T_camera(args: argparse.Namespace) -> np.ndarray:
    return make_transform_from_xyz_rpy(args.t5_head_pose) @ make_transform_from_xyz_rpy(args.head_camera_pose)


def selected_camera_model(args: argparse.Namespace) -> str:
    if args.camera:
        return args.camera
    if args.live_zed:
        return "zed"
    if args.live_d435:
        return "d435"
    return "d405"


def iteration_range(max_iterations: int):
    index = 0
    while max_iterations == 0 or index < max_iterations:
        yield index
        index += 1


def read_rgb(path: str | Path) -> np.ndarray:
    cv2 = require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_depth(path: str | Path, *, depth_scale: float) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    cv2 = require_cv2()
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    return depth_raw.astype(np.float32) * float(depth_scale)


def read_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CameraIntrinsics.from_mapping(data)


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional OpenCV install
        raise RuntimeError("OpenCV is required for image/mask file I/O.") from exc
    return cv2


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


def require_transform(transform: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    return transform


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def to_list(vector: np.ndarray) -> list[float]:
    return np.asarray(vector, dtype=float).reshape(-1).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
