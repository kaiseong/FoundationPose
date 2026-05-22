#!/usr/bin/env python3
"""Run phone frame visualization from live RGB-D input or offline RGB-D files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ADDED_PACKAGE_PARENT: str | None = None
if __package__ in (None, ""):
    _PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
        _ADDED_PACKAGE_PARENT = _PACKAGE_PARENT

from visual_servoing.point_pose.overlay import draw_phone_pose_overlay, draw_status_overlay
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.point_pose.rgbd_geometry import (
    CameraIntrinsics,
    backproject_masked_depth,
    estimate_phone_pose,
)
from visual_servoing.point_pose.sam3_phone_segmenter import Sam3PhoneSegmenter, load_mask

if _ADDED_PACKAGE_PARENT is not None:
    sys.path.remove(_ADDED_PACKAGE_PARENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Read live RGB-D frames from --camera.")
    mode.add_argument("--live-d405", action="store_true", help="Read live RGB-D frames from a D405.")
    mode.add_argument("--live-d435", action="store_true", help="Read live RGB-D frames from a D435.")
    mode.add_argument("--live-zed", action="store_true", help="Read live RGB-D frames from a ZED camera.")
    mode.add_argument("--rgb", help="Offline RGB image path.")
    parser.add_argument("--depth", help="Offline depth path (.npy meters, or image scaled by --depth-scale).")
    parser.add_argument("--mask", help="Offline binary phone mask path. Skips SAM3.")
    parser.add_argument("--intrinsics", help="Offline intrinsics JSON path.")
    parser.add_argument("--prompt", default="phone", help="SAM3 text prompt.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="SAM3 device.")
    parser.add_argument("--threshold", type=float, default=0.5, help="SAM3 confidence threshold.")
    parser.add_argument(
        "--camera",
        choices=SUPPORTED_LIVE_CAMERA_MODELS,
        default=None,
        help="Live RGB-D camera model. Defaults to the selected --live-* flag.",
    )
    parser.add_argument("--serial", default=None, help="Optional camera serial number.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Image depth unit to meters.")
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--save-overlay", help="Write latest overlay image to this path.")
    parser.add_argument("--no-window", action="store_true", help="Do not open an OpenCV window.")
    parser.add_argument("--print-json", action="store_true", help="Print transform JSON lines.")
    parser.add_argument("--print-timing", action="store_true", help="Include per-frame timing JSON.")
    parser.add_argument("--max-frames", type=int, default=0, help="Live frames to process; 0 means until quit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.live or args.live_d405 or args.live_d435 or args.live_zed:
        return run_live(args)
    return run_offline(args)


def run_offline(args: argparse.Namespace) -> int:
    frame_start = time.perf_counter()
    timing_ms: dict[str, float] = {}
    if not args.depth or not args.intrinsics:
        raise SystemExit("--depth and --intrinsics are required with --rgb.")
    start = time.perf_counter()
    image = read_rgb(args.rgb)
    timing_ms["rgb_read_ms"] = elapsed_ms(start)
    start = time.perf_counter()
    depth = read_depth(args.depth, depth_scale=args.depth_scale)
    timing_ms["depth_read_ms"] = elapsed_ms(start)
    start = time.perf_counter()
    intrinsics = read_intrinsics(args.intrinsics)
    timing_ms["intrinsics_read_ms"] = elapsed_ms(start)
    if args.mask:
        start = time.perf_counter()
        mask = load_mask(args.mask, shape=image.shape[:2])
        timing_ms["mask_read_ms"] = elapsed_ms(start)
    else:
        start = time.perf_counter()
        mask = segment_image(args, image).mask
        timing_ms["segmentation_ms"] = elapsed_ms(start)
    transform, overlay, process_timing = process_frame(
        args,
        image,
        depth,
        mask,
        intrinsics,
        previous_transform=None,
    )
    timing_ms.update(process_timing)
    timing_ms["frame_total_ms"] = elapsed_ms(frame_start)
    emit_outputs(args, transform, overlay, timing_ms=timing_ms)
    return 0


def run_live(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    camera_model = selected_camera_model(args)
    window_title = f"{camera_model.upper()} phone pose"
    segmenter = Sam3PhoneSegmenter(
        prompt=args.prompt,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    previous_transform = None
    last_overlay = None
    previous_frame_time = time.monotonic()
    fps = None
    with LiveRgbdCamera(
        model=camera_model,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        frame_index = 0
        if not args.no_window:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        while True:
            frame_start = time.perf_counter()
            timing_ms: dict[str, float] = {}
            transform = None
            start = time.perf_counter()
            frame = camera.read(timeout_ms=args.frame_timeout_ms)
            timing_ms["camera_read_ms"] = elapsed_ms(start)
            message = None
            status = "NO POSE"
            try:
                start = time.perf_counter()
                selection = segmenter.segment(frame.rgb)
                timing_ms["segmentation_ms"] = elapsed_ms(start)
                model_load_ms = segmenter.pop_last_model_load_ms()
                if model_load_ms is not None:
                    timing_ms["sam3_model_init_ms"] = model_load_ms
                    timing_ms["sam3_inference_ms"] = max(
                        timing_ms["segmentation_ms"] - model_load_ms,
                        0.0,
                    )
                transform, overlay, process_timing = process_frame(
                    args,
                    frame.rgb,
                    frame.depth_m,
                    selection.mask,
                    frame.intrinsics,
                    previous_transform=previous_transform,
                )
                timing_ms.update(process_timing)
                previous_transform = transform
            except ValueError as exc:
                if "segmentation_ms" not in timing_ms:
                    timing_ms["segmentation_ms"] = elapsed_ms(start)
                overlay = frame.rgb.copy()
                message = str(exc)
            else:
                status = "POSE OK"
            now = time.monotonic()
            elapsed = max(now - previous_frame_time, 1e-9)
            current_fps = 1.0 / elapsed
            fps = current_fps if fps is None else (0.85 * fps + 0.15 * current_fps)
            previous_frame_time = now
            timing_ms["fps"] = fps
            start = time.perf_counter()
            overlay = draw_status_overlay(
                overlay,
                status=status,
                prompt=args.prompt,
                frame_index=frame_index,
                fps=fps,
                message=message,
                timing_ms=timing_ms,
            )
            timing_ms["status_overlay_ms"] = elapsed_ms(start)
            last_overlay = overlay
            key = None
            if not args.no_window:
                start = time.perf_counter()
                cv2.imshow(window_title, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                timing_ms["display_ms"] = elapsed_ms(start)
            timing_ms["frame_total_ms"] = elapsed_ms(frame_start)
            if args.print_json:
                if transform is None:
                    print(
                        status_json(
                            "no_pose",
                            message or "No pose.",
                            timing_ms=timing_ms if args.print_timing else None,
                        )
                    )
                else:
                    print(transform_json(transform, timing_ms=timing_ms if args.print_timing else None))
            elif args.print_timing:
                print(status_json(status, message or "", timing_ms=timing_ms))
            if key in (ord("q"), 27):
                break
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    if args.save_overlay and last_overlay is not None:
        write_rgb(args.save_overlay, last_overlay)
    return 0


def selected_camera_model(args: argparse.Namespace) -> str:
    if args.camera:
        return args.camera
    if getattr(args, "live_zed", False):
        return "zed"
    if getattr(args, "live_d435", False):
        return "d435"
    return "d405"


def process_frame(
    args: argparse.Namespace,
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    previous_transform: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    timing_ms: dict[str, float] = {}
    start = time.perf_counter()
    points, _ = backproject_masked_depth(
        depth_m,
        mask,
        intrinsics,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        stride=1,
    )
    timing_ms["backproject_ms"] = elapsed_ms(start)
    timing_ms["masked_points"] = float(points.shape[0])
    start = time.perf_counter()
    transform = estimate_phone_pose(
        points,
        pixels_xy=mask_pixels_xy(mask),
        intrinsics=intrinsics,
        previous_transform=previous_transform,
    )
    timing_ms["pose_estimation_ms"] = elapsed_ms(start)
    start = time.perf_counter()
    overlay = draw_phone_pose_overlay(
        image_rgb,
        mask,
        transform,
        intrinsics,
        axis_length_m=args.axis_length_m,
    )
    timing_ms["pose_overlay_ms"] = elapsed_ms(start)
    return transform, overlay, timing_ms


def mask_pixels_xy(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(np.asarray(mask).astype(bool))
    return np.column_stack((cols, rows)).astype(np.float64)


def segment_image(args: argparse.Namespace, image_rgb: np.ndarray):
    return Sam3PhoneSegmenter(
        prompt=args.prompt,
        device=args.device,
        confidence_threshold=args.threshold,
    ).segment(image_rgb)


def emit_outputs(
    args: argparse.Namespace,
    transform: np.ndarray,
    overlay: np.ndarray,
    *,
    timing_ms: dict[str, float] | None = None,
) -> None:
    if args.print_json:
        print(transform_json(transform, timing_ms=timing_ms if args.print_timing else None))
    elif args.print_timing and timing_ms is not None:
        print(status_json("POSE OK", "", timing_ms=timing_ms))
    if args.save_overlay:
        write_rgb(args.save_overlay, overlay)
    if not args.no_window:
        cv2 = require_cv2()
        cv2.imshow("phone pose", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)


def transform_json(
    transform: np.ndarray,
    *,
    timing_ms: dict[str, float] | None = None,
) -> str:
    payload = {"camera_T_phone": np.asarray(transform, dtype=float).tolist()}
    if timing_ms is not None:
        payload["timing_ms"] = rounded_timing(timing_ms)
    return json.dumps(payload, separators=(",", ":"))


def status_json(
    status: str,
    message: str,
    *,
    timing_ms: dict[str, float] | None = None,
) -> str:
    payload: dict[str, object] = {"status": status, "message": message}
    if timing_ms is not None:
        payload["timing_ms"] = rounded_timing(timing_ms)
    return json.dumps(payload, separators=(",", ":"))


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def rounded_timing(timing_ms: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 3) for key, value in timing_ms.items()}


def read_rgb(path: str | Path) -> np.ndarray:
    cv2 = require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    cv2 = require_cv2()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not write overlay image: {path}")


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
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required for this command.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
