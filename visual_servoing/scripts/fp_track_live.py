"""Live RGB-D tracking command for a selected FoundationPose model-free profile."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

from visual_servoing.foundationpose_model_free.asset_builder import find_generated_mesh
from visual_servoing.foundationpose_model_free.foundationpose_adapter import (
    FoundationPoseAdapter,
    FoundationPoseConfig,
    StubFoundationPoseAdapter,
)
from visual_servoing.foundationpose_model_free.mask_provider import (
    PrecomputedMaskProvider,
    Sam3MaskProvider,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.foundationpose_model_free.tracker import (
    FoundationPoseLiveTracker,
    TrackingRecoveryConfig,
    TrackingState,
)
from visual_servoing.point_pose.overlay import (
    draw_axes_overlay,
    draw_phone_pose_overlay,
    draw_status_overlay,
)
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--prompt", help="Override the profile prompt for SAM3 initialization.")
    parser.add_argument("--data-root")
    parser.add_argument("--foundationpose-root", default=None)
    parser.add_argument("--mesh", help="Override generated profile mesh path.")
    parser.add_argument("--init-mask", help="Use a precomputed first-frame mask instead of SAM3.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d405")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug-dir")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/ESC.")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--print-timing", action="store_true")
    parser.add_argument(
        "--expected-distance-m",
        type=float,
        default=None,
        help="Optional measured camera-to-object distance; JSON output includes distance error.",
    )
    parser.add_argument("--mock", action="store_true", help="Validate profile loading without camera/FoundationPose.")
    parser.add_argument("--hold-last-pose-frames", type=int, default=0)
    parser.add_argument("--auto-reinit", action="store_true")
    parser.add_argument("--auto-reinit-after-lost-frames", type=int, default=5)
    parser.add_argument(
        "--enable-depth-lost-check",
        action="store_true",
        help="Mark tracking LOST when pose origin disagrees with image/depth. Disabled by default.",
    )
    parser.add_argument(
        "--disable-depth-lost-check",
        action="store_true",
        help="Deprecated compatibility flag; depth lost check is disabled by default.",
    )
    parser.add_argument(
        "--warn-initial-pose-mask-alignment",
        action="store_true",
        help="Show a warning when initial pose origin is far from the SAM mask center.",
    )
    parser.add_argument("--pose-depth-tolerance-m", type=float, default=0.18)
    parser.add_argument("--pose-depth-window-radius-px", type=int, default=7)
    parser.add_argument("--max-pose-jump-m", type=float, default=None)
    parser.add_argument("--implausible-lost-threshold", type=int, default=1)
    args = parser.parse_args()

    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    if args.prompt:
        profile.prompt = args.prompt
    if args.mock:
        adapter = StubFoundationPoseAdapter()
        print(f"mock tracking ready for {profile.name}: refs={profile.reference_count}, assets={profile.asset_status}")
        return 0
    mesh_path = Path(args.mesh).expanduser().resolve() if args.mesh else find_generated_mesh(profile)
    if mesh_path is None:
        raise SystemExit(
            f"profile {profile.name} has no generated mesh; run fp_build_assets.py --execute first "
            "or pass --mesh /path/to/model.obj"
        )
    adapter = FoundationPoseAdapter(
        FoundationPoseConfig(
            foundationpose_root=Path(args.foundationpose_root).expanduser().resolve()
            if args.foundationpose_root
            else None,
            mesh_path=mesh_path,
            debug_dir=Path(args.debug_dir).expanduser().resolve() if args.debug_dir else profile.logs_dir / "debug",
            debug=args.debug,
            refinement_iterations=args.refine_iterations,
            tracking_iterations=args.track_iterations,
        )
    )
    mask_provider = (
        PrecomputedMaskProvider(args.init_mask)
        if args.init_mask
        else Sam3MaskProvider(prompt=profile.prompt, device=args.device, confidence_threshold=args.threshold)
    )
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        mask_provider=mask_provider,
        recovery_config=TrackingRecoveryConfig(
            hold_last_pose_frames=args.hold_last_pose_frames,
            auto_reinit=args.auto_reinit,
            auto_reinit_after_lost_frames=args.auto_reinit_after_lost_frames,
            verify_pose_depth=bool(args.enable_depth_lost_check and not args.disable_depth_lost_check),
            warn_initial_pose_mask_alignment=bool(args.warn_initial_pose_mask_alignment),
            pose_depth_tolerance_m=args.pose_depth_tolerance_m,
            pose_depth_window_radius_px=args.pose_depth_window_radius_px,
            max_pose_jump_m=args.max_pose_jump_m,
            implausible_lost_threshold=args.implausible_lost_threshold,
        ),
    )
    return run_live(args, profile.prompt, tracker)


def run_live(args: argparse.Namespace, prompt: str, tracker: FoundationPoseLiveTracker) -> int:
    cv2 = require_cv2()
    previous_frame_time = time.monotonic()
    fps_smooth = None
    last_overlay = None
    window_title = f"{args.camera.upper()} FoundationPose"
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        if not args.no_window:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        frame_index = 0
        while True:
            frame_start = time.perf_counter()
            timing_ms: dict[str, float] = {}
            start = time.perf_counter()
            frame = camera.read(timeout_ms=args.frame_timeout_ms)
            timing_ms["camera_read_ms"] = elapsed_ms(start)

            status = TrackingState.LOST
            message = None
            pose = None
            result = None
            result_metadata = None
            overlay = frame.rgb.copy()
            try:
                start = time.perf_counter()
                result = tracker.process_frame(rgb=frame.rgb, depth_m=frame.depth_m, intrinsics=frame.intrinsics)
                timing_ms["pose_estimation_ms"] = elapsed_ms(start)
                pose = result.pose.camera_T_object if result.pose is not None else None
                status = result.state
                message = result.message
                result_metadata = result.metadata
                if pose is not None:
                    start = time.perf_counter()
                    if result.mask is not None:
                        overlay = draw_phone_pose_overlay(
                            frame.rgb,
                            result.mask,
                            pose,
                            frame.intrinsics,
                            axis_length_m=args.axis_length_m,
                        )
                    else:
                        overlay = draw_axes_overlay(
                            frame.rgb,
                            pose,
                            frame.intrinsics,
                            axis_length_m=args.axis_length_m,
                        )
                    timing_ms["pose_overlay_ms"] = elapsed_ms(start)
                    if result.held_pose and not message:
                        message = "holding last pose; press R to reinitialize"
                    message = combine_status_messages(
                        message,
                        pose_status_message(pose, expected_distance_m=args.expected_distance_m),
                    )
            except Exception as exc:
                timing_ms["pose_estimation_ms"] = timing_ms.get("pose_estimation_ms", elapsed_ms(start))
                message = str(exc)

            now = time.monotonic()
            current_fps = 1.0 / max(now - previous_frame_time, 1e-9)
            fps_smooth = current_fps if fps_smooth is None else 0.85 * fps_smooth + 0.15 * current_fps
            previous_frame_time = now
            timing_ms["fps"] = fps_smooth
            start = time.perf_counter()
            overlay = draw_status_overlay(
                overlay,
                status=status,
                prompt=prompt,
                frame_index=frame_index,
                fps=fps_smooth,
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
            emit_json(
                args,
                status=status,
                message=message,
                pose=pose,
                timing_ms=timing_ms,
                metadata=result_metadata,
            )
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                tracker.request_reinit()
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
    if last_overlay is not None and not args.no_window:
        cv2.destroyWindow(window_title)
    return 0


def emit_json(
    args: argparse.Namespace,
    *,
    status: str,
    message: str | None,
    pose: np.ndarray | None,
    timing_ms: dict[str, float],
    metadata: dict[str, object] | None = None,
) -> None:
    if not args.print_json and not args.print_timing:
        return
    payload: dict[str, object] = {"status": status}
    payload["tracking_state"] = status
    if message:
        payload["message"] = message
    if pose is not None:
        pose_array = np.asarray(pose, dtype=float)
        payload["camera_T_object"] = pose_array.tolist()
        payload.update(pose_distance_payload(pose_array, expected_distance_m=args.expected_distance_m))
    if args.print_json or args.print_timing:
        payload["timing_ms"] = {key: round(float(value), 3) for key, value in timing_ms.items()}
    if metadata:
        payload["tracking_metadata"] = metadata
    print(json.dumps(payload, separators=(",", ":")))


def pose_distance_payload(pose: np.ndarray, *, expected_distance_m: float | None = None) -> dict[str, object]:
    translation = np.asarray(pose, dtype=float)[:3, 3]
    distance_m = float(np.linalg.norm(translation))
    payload: dict[str, object] = {
        "object_position_m": {
            "x": round(float(translation[0]), 6),
            "y": round(float(translation[1]), 6),
            "z": round(float(translation[2]), 6),
        },
        "object_distance_m": round(distance_m, 6),
        "object_z_m": round(float(translation[2]), 6),
    }
    if expected_distance_m is not None:
        error_m = distance_m - float(expected_distance_m)
        payload["expected_distance_m"] = round(float(expected_distance_m), 6)
        payload["distance_error_m"] = round(error_m, 6)
        payload["distance_abs_error_m"] = round(abs(error_m), 6)
    return payload


def pose_status_message(pose: np.ndarray, *, expected_distance_m: float | None = None) -> str:
    translation = np.asarray(pose, dtype=float)[:3, 3]
    distance_m = float(np.linalg.norm(translation))
    parts = [
        f"x:{translation[0]:+.3f}m",
        f"y:{translation[1]:+.3f}m",
        f"z:{translation[2]:+.3f}m",
        f"dist:{distance_m:.3f}m",
    ]
    if expected_distance_m is not None:
        parts.append(f"err:{distance_m - float(expected_distance_m):+.3f}m")
    return " | ".join(parts)


def combine_status_messages(*messages: str | None) -> str | None:
    return " | ".join(message for message in messages if message) or None


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required for live visualization.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
