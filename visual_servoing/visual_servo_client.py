#!/usr/bin/env python3
"""Baseline point-cloud visual servo client.

The default path is intentionally dry-run only. Real robot motion is enabled
only when --execute is provided.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

_ADDED_PACKAGE_PARENT: str | None = None
if __package__ in (None, ""):
    _PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
        _ADDED_PACKAGE_PARENT = _PACKAGE_PARENT

from visual_servoing.point_pose.live_camera_config import SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.point_pose.sam3_phone_segmenter import Sam3PhoneSegmenter, load_mask
from visual_servoing.visual_servo_protocol import (
    REQUEST_CONTENT_TYPE,
    decode_visual_servo_response,
    encode_visual_servo_request,
)
import visual_servoing.visual_servo_core as servo_core

if _ADDED_PACKAGE_PARENT is not None:
    sys.path.remove(_ADDED_PACKAGE_PARENT)


DEFAULT_T5_HEAD_XYZ_RPY = (0.0, 0.0, 0.0, 0.0, 45.0, 0.0)
HEAD_TO_CAMERA_XYZ_RPY = (0.047, 0.009, 0.057, -90.0, 0.0, -90.0)
DEFAULT_RIGHT_ARM_STIFFNESS = (90.0, 90.0, 90.0, 70.0, 70.0, 70.0, 70.0)
DEFAULT_RIGHT_ARM_TORQUE_LIMIT = (40.0, 40.0, 40.0, 30.0, 30.0, 30.0, 30.0)
RIGHT_ARM_CARTESIAN_READY_POSE_DEG = (0.0, -5.0, 0.0, -120.0, 0.0, 40.0, 0.0)
RIGHT_ARM_CONTROL_ROOT_LINK = servo_core.RIGHT_ARM_CONTROL_ROOT_LINK
RIGHT_ARM_EE_LINKS = servo_core.RIGHT_ARM_EE_LINKS
REMOTE_OFFSET_FRAME = servo_core.REMOTE_OFFSET_FRAME
POSITION_ONLY_ORIENTATION_POLICY = servo_core.POSITION_ONLY_ORIENTATION_POLICY
ROBOT_MODEL = "m"

ServoLimits = servo_core.ServoLimits
VisualObservation = servo_core.VisualObservation
ServoStep = servo_core.ServoStep
estimate_visual_observation = servo_core.estimate_visual_observation
plan_visual_servo_step = servo_core.plan_visual_servo_step
make_transform_from_xyz_rpy = servo_core.make_transform_from_xyz_rpy
rotation_matrix_from_rpy = servo_core.rotation_matrix_from_rpy
axis_angle_rotation = servo_core.axis_angle_rotation
local_axis = servo_core.local_axis
clamp_translation_step = servo_core.clamp_translation_step
clamp_scalar = servo_core.clamp_scalar
signed_angle_about_axis = servo_core.signed_angle_about_axis
require_transform = servo_core.require_transform
matrix_list = servo_core.matrix_list
to_list = servo_core.to_list


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
    parser.add_argument("--remote-server", help="Remote visual servo server as HOST:PORT or URL.")
    parser.add_argument("--remote-timeout-s", type=float, default=2.0)
    parser.add_argument("--stale-action-max-age-s", type=float, default=1.0)
    parser.add_argument(
        "--remote-fixture-request",
        action="store_true",
        help="Send one synthetic RGB-D request to --remote-server without a camera or robot.",
    )

    parser.add_argument("--max-iterations", type=int, default=1, help="0 means run until interrupted.")
    parser.add_argument("--loop-sleep-s", type=float, default=0.0)
    parser.add_argument("--print-json", action="store_true", help="Accepted for compatibility; JSON is always printed.")
    parser.add_argument("--no-window", action="store_true", help="Disable any local OpenCV preview window.")
    parser.add_argument(
        "--show-camera-window",
        action="store_true",
        help="Show the current UPC-side live RGB frame in an OpenCV window. Press q or Esc to stop.",
    )
    parser.add_argument("--camera-window-name", default="visual_servo_client")
    parser.add_argument("--camera-window-scale", type=float, default=1.0)
    parser.add_argument(
        "--show-mask-window",
        action="store_true",
        help="Request the selected remote SAM mask and overlay it in the OpenCV preview window.",
    )
    parser.add_argument("--mask-overlay-alpha", type=float, default=0.45)

    parser.add_argument(
        "--ee-offset",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Current-EE-frame relative offset: meters and degrees.",
    )
    parser.add_argument(
        "--object-offset",
        type=float,
        nargs=6,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Deprecated for remote visual servo; use --target-offset-t5 for position-only t5 offsets.",
    )
    parser.add_argument(
        "--target-offset-t5",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Remote visual servo target offset in link_torso_5/T5 frame: meters.",
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
        help="Fixed T5-to-head pose for dry-run/live geometry. Defaults to head_1=45 deg.",
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
    parser.add_argument("--model", default=ROBOT_MODEL, help="Robot model. Execute mode is fixed to m.")
    parser.add_argument("--power", default=".*", help="Power-on component pattern. Defaults to all components.")
    parser.add_argument("--servo", default=".*", help="Servo-on component pattern. Defaults to all components.")
    parser.add_argument("--control-root-link", default=RIGHT_ARM_CONTROL_ROOT_LINK)
    parser.add_argument("--ee-link", default="link_right_arm_6")
    parser.add_argument("--command-min-time-s", type=float, default=0.25)
    parser.add_argument("--command-hold-time-s", type=float, default=0.5)
    parser.add_argument("--command-timeout-s", type=float, default=2.0)
    parser.add_argument("--command-priority", type=int, default=10)
    parser.add_argument("--control-ready-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--move-to-ready-on-connect",
        action="store_true",
        help="Move only the right arm to a Cartesian-ready bent pose before visual servo commands.",
    )
    parser.add_argument("--ready-min-time-s", type=float, default=3.0)
    parser.add_argument("--ready-hold-time-s", type=float, default=4.0)
    parser.add_argument("--linear-limit", type=float, default=1.0)
    parser.add_argument("--angular-limit", type=float, default=math.pi / 2.0)
    parser.add_argument("--linear-gain", type=float, default=50.0)
    parser.add_argument("--angular-gain", type=float, default=math.pi * 20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    validate_args(args)
    if args.remote_fixture_request:
        return run_remote_fixture(args)
    if is_live_mode(args):
        return run_live(args)
    return run_offline(args)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_iterations < 0:
        raise SystemExit("--max-iterations must be >= 0")
    if args.remote_timeout_s <= 0.0:
        raise SystemExit("--remote-timeout-s must be > 0")
    if args.stale_action_max_age_s <= 0.0:
        raise SystemExit("--stale-action-max-age-s must be > 0")
    if args.remote_fixture_request and not args.remote_server:
        raise SystemExit("--remote-fixture-request requires --remote-server")
    if args.remote_fixture_request and args.execute:
        raise SystemExit("--remote-fixture-request cannot be used with --execute")
    if args.remote_server and not (is_live_mode(args) or args.remote_fixture_request):
        raise SystemExit("--remote-server requires a live camera mode or --remote-fixture-request")
    if args.remote_server and any(abs(float(value)) > 1e-12 for value in args.object_offset):
        raise SystemExit("--object-offset is object-frame SE(3); use --target-offset-t5 X Y Z for remote position-only servo")
    if args.camera_window_scale <= 0.0:
        raise SystemExit("--camera-window-scale must be > 0")
    if not 0.0 <= float(args.mask_overlay_alpha) <= 1.0:
        raise SystemExit("--mask-overlay-alpha must be between 0 and 1")
    if args.execute:
        validate_execute_safety(args)
    if not is_live_mode(args) and not args.remote_fixture_request:
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
    if args.model != ROBOT_MODEL:
        raise SystemExit(f"--execute is fixed to --model {ROBOT_MODEL!r}")
    if args.ee_link not in RIGHT_ARM_EE_LINKS:
        allowed = ", ".join(sorted(RIGHT_ARM_EE_LINKS))
        raise SystemExit(f"--execute is restricted to right-arm EE links: {allowed}")
    validate_component_pattern("--power", args.power)
    validate_component_pattern("--servo", args.servo)
    if args.command_priority < 0:
        raise SystemExit("--command-priority must be >= 0")
    if args.command_hold_time_s <= 0.0:
        raise SystemExit("--command-hold-time-s must be > 0")
    if args.command_timeout_s <= 0.0:
        raise SystemExit("--command-timeout-s must be > 0")
    if args.ready_min_time_s <= 0.0:
        raise SystemExit("--ready-min-time-s must be > 0")
    if args.ready_hold_time_s <= 0.0:
        raise SystemExit("--ready-hold-time-s must be > 0")


def validate_component_pattern(flag: str, value: str) -> None:
    if not str(value).strip():
        raise SystemExit(f"{flag} cannot be empty")


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
    if args.remote_server:
        return run_remote_live(args)
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
        with LiveCameraPreview(args) as preview, LiveRgbdCamera(
            model=camera_model,
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ) as camera:
            for frame_index in iteration_range(args.max_iterations):
                frame = camera.read(timeout_ms=args.frame_timeout_ms)
                if not preview.show(frame.rgb):
                    break
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
                    if args.show_mask_window and not preview.show(
                        frame.rgb,
                        mask=selection.mask,
                        box_xyxy=selection.box_xyxy,
                    ):
                        break
                except Exception as exc:
                    result = skipped_result(args, frame_index, str(exc))
                    if args.execute:
                        result["command_feedback"] = robot_context.cancel_command_stream(
                            f"local segmentation skipped: {exc}"
                        )
                print(json.dumps(result, separators=(",", ":")))
                if result.get("ok") is True and result.get("status") == "converged":
                    break
                if args.loop_sleep_s > 0.0:
                    time.sleep(args.loop_sleep_s)
    finally:
        robot_context.close()
    return 0


def run_remote_live(args: argparse.Namespace) -> int:
    camera_model = selected_camera_model(args)
    robot_context = RobotContext.connect(args) if args.execute else RobotContext.dry_run(args)
    try:
        current_t5_T_ee = robot_context.current_ee_pose()
        t5_T_camera = fixed_t5_T_camera(args)
        with LiveCameraPreview(args) as preview, LiveRgbdCamera(
            model=camera_model,
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ) as camera:
            for frame_index in iteration_range(args.max_iterations):
                frame = camera.read(timeout_ms=args.frame_timeout_ms)
                if not preview.show(frame.rgb):
                    break
                result, current_t5_T_ee = process_remote_servo_iteration(
                    args,
                    rgb=frame.rgb,
                    depth_m=frame.depth_m,
                    intrinsics=frame.intrinsics,
                    t5_T_camera=t5_T_camera,
                    current_t5_T_ee=current_t5_T_ee,
                    robot_context=robot_context,
                    frame_index=frame_index,
                )
                if args.show_mask_window and not preview.show_result(frame.rgb, result):
                    break
                print(json.dumps(strip_mask_preview_for_logging(result), separators=(",", ":")))
                if result.get("ok") is True and result.get("status") == "converged":
                    break
                if args.loop_sleep_s > 0.0:
                    time.sleep(args.loop_sleep_s)
    finally:
        robot_context.close()
    return 0


def run_remote_fixture(args: argparse.Namespace) -> int:
    rgb, depth_m, intrinsics = synthetic_rgbd_fixture()
    robot_context = RobotContext.dry_run(args)
    current_t5_T_ee = robot_context.current_ee_pose()
    t5_T_camera = fixed_t5_T_camera(args)
    for frame_index in iteration_range(args.max_iterations):
        result, current_t5_T_ee = process_remote_servo_iteration(
            args,
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            t5_T_camera=t5_T_camera,
            current_t5_T_ee=current_t5_T_ee,
            robot_context=robot_context,
            frame_index=frame_index,
        )
        print(json.dumps(result, separators=(",", ":")))
        if result.get("ok") is True and result.get("status") == "converged":
            break
    return 0


def process_remote_servo_iteration(
    args: argparse.Namespace,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    t5_T_camera: np.ndarray,
    current_t5_T_ee: np.ndarray,
    robot_context: "RobotContext",
    frame_index: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if robot_context.execute:
        current_t5_T_ee = robot_context.current_ee_pose()
    request_id = f"{time.monotonic_ns()}-{frame_index}"
    object_T_offset = np.eye(4, dtype=np.float64)
    capture_monotonic_ns = time.monotonic_ns()
    metadata = remote_request_metadata(args)
    metadata["ee_link"] = args.ee_link

    encode_start = time.perf_counter()
    body = encode_visual_servo_request(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        request_id=request_id,
        frame_index=frame_index,
        capture_monotonic_ns=capture_monotonic_ns,
        t5_T_camera=t5_T_camera,
        current_t5_T_ee=current_t5_T_ee,
        object_T_offset=object_T_offset,
        metadata=metadata,
    )
    encode_ms = (time.perf_counter() - encode_start) * 1000.0
    send_start = time.perf_counter()
    try:
        response = send_remote_visual_servo_request(args.remote_server, body, timeout_s=args.remote_timeout_s)
        round_trip_s = time.perf_counter() - send_start
        validation = servo_core.validate_remote_action(
            response,
            request_id=request_id,
            frame_index=frame_index,
            round_trip_s=round_trip_s,
            stale_action_max_age_s=args.stale_action_max_age_s,
            current_t5_T_ee=current_t5_T_ee,
            max_translation_step_m=args.max_translation_step_m,
            max_wrist_step_rad=math.radians(args.max_wrist_step_deg),
            expected_root_link=args.control_root_link,
            allowed_ee_links=RIGHT_ARM_EE_LINKS,
        )
    except Exception as exc:
        response = {"ok": False, "status": "skipped", "request_id": request_id, "frame_index": frame_index, "reason": str(exc)}
        round_trip_s = time.perf_counter() - send_start
        validation = servo_core.RemoteActionValidation(False, False, str(exc))

    command_sent = False
    command_feedback: dict[str, Any] | None = None
    next_t5_T_ee = current_t5_T_ee
    if validation.executable and validation.target_t5_T_ee is not None:
        next_t5_T_ee = validation.target_t5_T_ee
        if args.execute:
            command_feedback = robot_context.send_right_arm_cartesian(validation.target_t5_T_ee)
            command_sent = True
    elif args.execute:
        command_feedback = robot_context.cancel_command_stream(validation.reason)
    result = remote_diagnostic_payload(
        args,
        frame_index=frame_index,
        request_id=request_id,
        response=response,
        validation=validation,
        command_sent=command_sent,
        command_feedback=command_feedback,
        encode_ms=encode_ms,
        round_trip_s=round_trip_s,
    )
    return result, next_t5_T_ee


def send_remote_visual_servo_request(server: str, body: bytes, *, timeout_s: float) -> dict[str, Any]:
    url = normalize_remote_server_url(server)
    request = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": REQUEST_CONTENT_TYPE},
    )
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_s)) as response:
            data = response.read()
    except urllib_error.HTTPError as exc:
        data = exc.read()
        try:
            payload = decode_visual_servo_response(data)
        except Exception:
            raise RuntimeError(f"remote visual servo server returned HTTP {exc.code}") from exc
        raise RuntimeError(payload.get("reason") or payload.get("error") or f"HTTP {exc.code}") from exc
    return decode_visual_servo_response(data)


def normalize_remote_server_url(server: str) -> str:
    value = str(server).strip()
    if not value:
        raise ValueError("--remote-server cannot be empty")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/") + "/visual-servo/action"


def remote_request_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_depth_m": float(args.min_depth_m),
        "max_depth_m": float(args.max_depth_m),
        "max_translation_step_m": float(args.max_translation_step_m),
        "max_wrist_step_deg": float(args.max_wrist_step_deg),
        "position_tolerance_m": float(args.position_tolerance_m),
        "wrist_tolerance_deg": float(args.wrist_tolerance_deg),
        "prompt": args.prompt,
        "threshold": float(args.threshold),
        "sam_resolution": int(args.sam_resolution),
        "target_offset_t5_m": [float(value) for value in args.target_offset_t5],
        "offset_frame": REMOTE_OFFSET_FRAME,
        "orientation_policy": POSITION_ONLY_ORIENTATION_POLICY,
        "servo_dofs": "xyz_position_only",
        "ee_align_axis": args.ee_align_axis,
        "wrist_axis": args.wrist_axis,
        "control_root_link": args.control_root_link,
        "return_mask_preview": bool(args.show_mask_window and not args.no_window),
    }


def remote_diagnostic_payload(
    args: argparse.Namespace,
    *,
    frame_index: int,
    request_id: str,
    response: dict[str, Any],
    validation: servo_core.RemoteActionValidation,
    command_sent: bool,
    command_feedback: dict[str, Any] | None,
    encode_ms: float,
    round_trip_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(response.get("ok", False)) and validation.ok,
        "status": response.get("status", "skipped"),
        "frame_index": frame_index,
        "execute": bool(args.execute),
        "command_sent": command_sent,
        "reason": validation.reason,
        "remote": {
            "enabled": True,
            "server": args.remote_server,
            "request_id": request_id,
            "round_trip_ms": float(round_trip_s) * 1000.0,
            "stale_action_max_age_s": float(args.stale_action_max_age_s),
            "stale": float(round_trip_s) > float(args.stale_action_max_age_s),
            "request_encode_ms": float(encode_ms),
            "action_valid": validation.ok,
            "action_executable": validation.executable,
        },
        "server_timing_ms": response.get("server_timing_ms", {}),
    }
    for key in ("action", "observation", "servo_step", "mask"):
        if key in response:
            payload[key] = response[key]
    for key in ("offset_frame", "orientation_policy", "target_offset_t5_m"):
        if key in response:
            payload[key] = response[key]
    if command_feedback is not None:
        payload["command_feedback"] = command_feedback
    return payload


def synthetic_rgbd_fixture() -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    height = 32
    width = 32
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[8:24, 10:22] = np.array([220, 220, 220], dtype=np.uint8)
    depth_m = np.full((height, width), 0.5, dtype=np.float32)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
    return rgb, depth_m, intrinsics


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
        elif args.execute:
            command_feedback = robot_context.cancel_command_stream(reason)
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
        result = skipped_result(args, frame_index, str(exc))
        if args.execute:
            result["command_feedback"] = robot_context.cancel_command_stream(f"local servo skipped: {exc}")
        return result, previous_object_transform, current_t5_T_ee


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


def strip_mask_preview_for_logging(payload: dict[str, Any]) -> dict[str, Any]:
    mask = payload.get("mask")
    if not isinstance(mask, dict) or "preview" not in mask:
        return payload
    sanitized = dict(payload)
    sanitized_mask = dict(mask)
    sanitized_mask.pop("preview", None)
    sanitized["mask"] = sanitized_mask
    return sanitized


def decode_mask_preview(preview: dict[str, Any]) -> np.ndarray:
    if preview.get("encoding") != "packbits-b64-v1":
        raise ValueError(f"unsupported mask preview encoding: {preview.get('encoding')!r}")
    shape = preview.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("mask preview shape must be [height, width]")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("mask preview shape must be positive")
    raw = base64.b64decode(str(preview.get("data", "")).encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed, count=height * width)
    return bits.reshape((height, width)).astype(bool)


class RobotContext:
    def __init__(self, args: argparse.Namespace, *, robot=None, rby=None) -> None:
        self.args = args
        self.robot = robot
        self.rby = rby
        self.command_stream = None
        self.command_stream_cancelled_for_safety = False

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
        context = cls(args, robot=robot, rby=rby)
        recovered_fault = context.reset_fault_control_manager_if_needed()
        if not recovered_fault and not robot.enable_control_manager():
            raise RuntimeError("Failed to enable control manager")
        context.wait_for_control_ready()
        if args.move_to_ready_on_connect:
            context.move_right_arm_to_ready_pose()
        context.open_command_stream()
        return context

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
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(self.args.command_hold_time_s))
            )
            .set_minimum_time(float(self.args.command_min_time_s))
            .set_joint_stiffness(np.asarray(DEFAULT_RIGHT_ARM_STIFFNESS, dtype=np.float64))
            .set_joint_torque_limit(np.asarray(DEFAULT_RIGHT_ARM_TORQUE_LIMIT, dtype=np.float64))
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
        stream = self.ensure_command_stream()
        feedback = stream.send_command(command, self.command_timeout_ms())
        payload = self._feedback_payload(feedback)
        payload["transport"] = "command_stream"
        payload["stream_done"] = bool(stream.is_done()) if hasattr(stream, "is_done") else False
        return payload

    def open_command_stream(self):
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        self.command_stream = self.robot.create_command_stream(priority=int(self.args.command_priority))
        return self.command_stream

    def ensure_command_stream(self):
        if self.command_stream is None:
            if self.command_stream_cancelled_for_safety:
                self.wait_for_control_ready()
                self.command_stream_cancelled_for_safety = False
            return self.open_command_stream()
        if hasattr(self.command_stream, "is_done") and self.command_stream.is_done():
            self.command_stream = None
            self.wait_for_control_ready()
            self.command_stream_cancelled_for_safety = False
            return self.open_command_stream()
        return self.command_stream

    def cancel_command_stream(self, reason: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "transport": "command_stream",
            "cancelled": False,
            "control_cancelled": False,
            "reason": str(reason),
        }
        if self.command_stream is not None and hasattr(self.command_stream, "cancel"):
            self.command_stream.cancel()
            payload["cancelled"] = True
            if hasattr(self.command_stream, "is_done"):
                payload["stream_done"] = bool(self.command_stream.is_done())
        self.command_stream = None
        self.command_stream_cancelled_for_safety = True
        if self.robot is not None and hasattr(self.robot, "cancel_control"):
            self.robot.cancel_control()
            payload["control_cancelled"] = True
        return payload

    def reset_fault_control_manager_if_needed(self) -> bool:
        if self.robot is None or self.rby is None:
            return False
        state = self.robot.get_control_manager_state()
        manager_state = getattr(state, "state", None)
        state_enum = getattr(self.rby.ControlManagerState, "State", None)
        if state_enum is None:
            return False
        fault_states = tuple(
            getattr(state_enum, name)
            for name in ("MajorFault", "MinorFault")
            if hasattr(state_enum, name)
        )
        if manager_state not in fault_states:
            return False
        if not self.robot.reset_fault_control_manager():
            raise RuntimeError("Failed to reset control manager")
        if not self.robot.enable_control_manager():
            raise RuntimeError("Failed to enable control manager")
        return True

    def command_timeout_ms(self) -> int:
        return max(1, int(round(float(self.args.command_timeout_s) * 1000.0)))

    def move_right_arm_to_ready_pose(self) -> dict[str, Any]:
        if self.robot is None or self.rby is None:
            raise RuntimeError("Robot is not connected.")
        self.wait_for_control_ready()
        rby = self.rby
        ready_q = np.deg2rad(np.asarray(RIGHT_ARM_CARTESIAN_READY_POSE_DEG, dtype=np.float64))
        command = rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
                    rby.JointPositionCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder().set_control_hold_time(float(self.args.ready_hold_time_s))
                    )
                    .set_minimum_time(float(self.args.ready_min_time_s))
                    .set_position(ready_q)
                )
            )
        )
        feedback = self.robot.send_command(command, int(self.args.command_priority)).get()
        payload = self._feedback_payload(feedback)
        finish_code = payload.get("finish_code")
        if finish_code not in {"FinishCode.Ok", "ok"}:
            raise RuntimeError(f"right-arm ready pose command failed: {payload}")
        self.wait_for_control_ready()
        return payload

    @staticmethod
    def _feedback_payload(feedback) -> dict[str, Any]:
        payload = {"finish_code": str(getattr(feedback, "finish_code", None))}
        for name in ("status", "valid"):
            if hasattr(feedback, name):
                payload[name] = str(getattr(feedback, name))
        return payload

    def wait_for_control_ready(self) -> None:
        if self.robot is None or self.rby is None:
            return
        self.reset_fault_control_manager_if_needed()
        state = self.robot.get_control_manager_state()
        control_state = getattr(state, "control_state", None)
        idle_state = getattr(self.rby.ControlManagerState.ControlState, "Idle", None)
        if idle_state is not None and control_state != idle_state and hasattr(self.robot, "cancel_control"):
            self.robot.cancel_control()
        if hasattr(self.robot, "wait_for_control_ready"):
            ready = self.robot.wait_for_control_ready(int(self.args.control_ready_timeout_ms))
            if not ready:
                raise RuntimeError("wait_for_control_ready timed out before Cartesian command")

    def close(self) -> None:
        if self.command_stream is not None and hasattr(self.command_stream, "cancel"):
            self.command_stream.cancel()
            self.command_stream = None
        self.command_stream_cancelled_for_safety = False
        if self.robot is not None:
            self.robot.disconnect()


class LiveCameraPreview:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.enabled = bool((args.show_camera_window or args.show_mask_window) and not args.no_window)
        self.cv2 = None
        self.window_name = str(args.camera_window_name)

    def __enter__(self) -> "LiveCameraPreview":
        if not self.enabled:
            return self
        self.cv2 = require_cv2()
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.cv2 is not None:
            self.cv2.destroyWindow(self.window_name)

    def show(
        self,
        rgb: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        mask_preview: dict[str, Any] | None = None,
        box_xyxy: list[float] | tuple[float, float, float, float] | None = None,
    ) -> bool:
        if not self.enabled:
            return True
        if self.cv2 is None:
            self.cv2 = require_cv2()
        image_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        if mask_preview is not None:
            try:
                mask = decode_mask_preview(mask_preview)
            except Exception:
                mask = None
        if mask is not None:
            image_rgb = self._overlay_mask(image_rgb, mask)
        if box_xyxy is not None:
            image_rgb = self._draw_box(image_rgb, box_xyxy)
        image = self.cv2.cvtColor(image_rgb, self.cv2.COLOR_RGB2BGR)
        scale = float(self.args.camera_window_scale)
        if abs(scale - 1.0) > 1e-12:
            image = self.cv2.resize(image, None, fx=scale, fy=scale, interpolation=self.cv2.INTER_NEAREST)
        self.cv2.imshow(self.window_name, image)
        key = self.cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def show_result(self, rgb: np.ndarray, result: dict[str, Any]) -> bool:
        mask_payload = result.get("mask")
        if not isinstance(mask_payload, dict):
            return self.show(rgb)
        return self.show(
            rgb,
            mask_preview=mask_payload.get("preview"),
            box_xyxy=mask_payload.get("box_xyxy"),
        )

    def _overlay_mask(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_bool = self._mask_for_image(mask, image_rgb.shape[:2])
        if not np.any(mask_bool):
            return image_rgb
        alpha = float(self.args.mask_overlay_alpha)
        color = np.array([0, 255, 0], dtype=np.float32)
        overlay = image_rgb.astype(np.float32)
        overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color * alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _mask_for_image(self, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        if mask_u8.shape[:2] == shape:
            return mask_u8.astype(bool)
        if self.cv2 is None:
            self.cv2 = require_cv2()
        resized = self.cv2.resize(mask_u8, (int(shape[1]), int(shape[0])), interpolation=self.cv2.INTER_NEAREST)
        return np.asarray(resized, dtype=np.uint8).astype(bool)

    @staticmethod
    def _draw_box(
        image_rgb: np.ndarray,
        box_xyxy: list[float] | tuple[float, float, float, float],
    ) -> np.ndarray:
        if len(box_xyxy) != 4:
            return image_rgb
        height, width = image_rgb.shape[:2]
        x0, y0, x1, y1 = [int(round(float(value))) for value in box_xyxy]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        if x1 < x0 or y1 < y0:
            return image_rgb
        image_rgb[y0 : y1 + 1, x0] = [255, 0, 0]
        image_rgb[y0 : y1 + 1, x1] = [255, 0, 0]
        image_rgb[y0, x0 : x1 + 1] = [255, 0, 0]
        image_rgb[y1, x0 : x1 + 1] = [255, 0, 0]
        return image_rgb


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


if __name__ == "__main__":
    raise SystemExit(main())
