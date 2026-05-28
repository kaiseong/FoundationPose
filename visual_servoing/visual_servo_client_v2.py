"""Thin FoundationPose v2 pose-only remote tracking client."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib import request as urllib_request

import numpy as np

from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    decode_foundationpose_response,
    encode_foundationpose_track_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoundationPose v2 pose-only remote tracking client.")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8081)
    parser.add_argument("--remote-server", default=None, help="Optional full server base URL or host:port override.")
    parser.add_argument("--object", "--profile", dest="profile", required=True)
    parser.add_argument("--prompt", default=None, help="Prompt override carried in mask options for server-side SAM3.")
    parser.add_argument("--foundationpose-root", default=None)
    parser.add_argument("--data-root", default=None, help="Reserved for local profile parity; not sent as a robot command.")
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d405")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--zed-depth-mode", default="NEURAL")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--reinit", action="store_true")
    parser.add_argument("--auto-reinit", action="store_true")
    parser.add_argument("--auto-reinit-after-lost-frames", type=int, default=5)
    parser.add_argument("--hold-last-pose-frames", type=int, default=0)
    parser.add_argument("--enable-depth-lost-check", action="store_true")
    parser.add_argument("--max-pose-jump-m", type=float, default=None)
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument(
        "--t5-T-camera-json",
        default=None,
        help="Optional JSON 4x4 transform. Defaults to identity for pose validation.",
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def server_base_url(args: argparse.Namespace) -> str:
    raw = args.remote_server or f"{args.server_host}:{int(args.server_port)}"
    raw = str(raw).strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")
    return f"http://{raw}".rstrip("/")


def send_track_request(
    server_url: str,
    body: bytes,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib_request.Request(
        f"{server_url.rstrip('/')}/foundationpose/v2/track",
        data=body,
        headers={"Content-Type": REQUEST_CONTENT_TYPE},
    )
    with urllib_request.urlopen(request, timeout=float(timeout_s)) as response:
        return decode_foundationpose_response(response.read())


def build_tracking_request_body(
    *,
    frame,
    args: argparse.Namespace,
    frame_index: int,
    request_id: str,
    capture_monotonic_ns: int,
) -> bytes:
    return encode_foundationpose_track_request(
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        intrinsics=frame.intrinsics,
        request_id=request_id,
        frame_index=frame_index,
        capture_monotonic_ns=capture_monotonic_ns,
        t5_T_camera=parse_t5_T_camera(args.t5_T_camera_json),
        profile=args.profile,
        foundationpose_root=args.foundationpose_root,
        refine_iterations=args.refine_iterations,
        track_iterations=args.track_iterations,
        reinit=args.reinit,
        mask_options=mask_options_from_args(args),
        recovery_options=recovery_options_from_args(args),
        metadata={"client": "visual_servo_client_v2"},
    )


def mask_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "device": args.sam_device,
        "threshold": float(args.threshold),
        "resolution": int(args.sam_resolution),
    }
    if args.prompt:
        options["prompt"] = args.prompt
    return options


def recovery_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "hold_last_pose_frames": int(args.hold_last_pose_frames),
        "auto_reinit": bool(args.auto_reinit),
        "auto_reinit_after_lost_frames": int(args.auto_reinit_after_lost_frames),
        "verify_pose_depth": bool(args.enable_depth_lost_check),
    }
    if args.max_pose_jump_m is not None:
        options["max_pose_jump_m"] = float(args.max_pose_jump_m)
    return options


def parse_t5_T_camera(raw: str | None) -> np.ndarray:
    if raw is None or not str(raw).strip():
        return np.eye(4, dtype=np.float64)
    value = json.loads(raw)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"t5_T_camera must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("t5_T_camera contains non-finite values")
    return matrix


def run_live(args: argparse.Namespace) -> int:
    server_url = server_base_url(args)
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        zed_depth_mode=args.zed_depth_mode,
    ) as camera:
        frame_index = 0
        while True:
            capture_ns = time.monotonic_ns()
            frame = camera.read(timeout_ms=args.frame_timeout_ms)
            request_id = f"{capture_ns}-{frame_index}"
            body = build_tracking_request_body(
                frame=frame,
                args=args,
                frame_index=frame_index,
                request_id=request_id,
                capture_monotonic_ns=capture_ns,
            )
            response = send_track_request(server_url, body, timeout_s=args.request_timeout_s)
            print(format_response_line(response) if not args.print_json else json.dumps(response, separators=(",", ":")))
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    return 0


def format_response_line(response: dict[str, Any]) -> str:
    fields = [
        f"frame={response.get('frame_index')}",
        f"ok={str(response.get('ok')).lower()}",
        f"state={response.get('tracking_state')}",
    ]
    camera_xyz = _pose_xyz(response.get("camera_T_object"))
    t5_xyz = _pose_xyz(response.get("t5_T_object"))
    if camera_xyz is not None:
        fields.append(f"camera_xyz_m={camera_xyz}")
    if t5_xyz is not None:
        fields.append(f"t5_xyz_m={t5_xyz}")
    message = response.get("message")
    if message:
        fields.append(f"message={message}")
    return " ".join(str(field) for field in fields)


def _pose_xyz(value: Any) -> str | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    xyz = matrix[:3, 3]
    return f"({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
