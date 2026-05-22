"""Generate FoundationPose cam_in_ob reference poses from ChArUco detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from visual_servoing.foundationpose_model_free.charuco_reference import (
    BoardObjectTransform,
    CharucoBoardSpec,
    CharucoQualityConfig,
    detect_charuco_pose,
    draw_charuco_detection_debug_bgr,
    draw_charuco_axes_overlay_bgr,
    generate_charuco_reference_poses,
    record_charuco_pose_provenance,
    write_charuco_reference_poses,
)
from visual_servoing.foundationpose_model_free.mask_provider import Sam3MaskProvider
from visual_servoing.foundationpose_model_free.reference_processing import (
    ReferenceProcessingConfig,
    evaluate_recorded_references,
    process_recorded_references,
    reselect_recorded_references,
)
from visual_servoing.foundationpose_model_free.reference_recording import (
    ReferenceRecordingConfig,
    ReferenceRecordingSession,
)
from visual_servoing.foundationpose_model_free.reference_dataset import count_reference_frames, save_reference_frame
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline-generate", action="store_true", help="Generate cam_in_ob for existing refs/rgb frames.")
    mode.add_argument("--detect-only", action="store_true", help="Detect and report ChArUco pose without writing poses.")
    mode.add_argument("--live-capture", action="store_true", help="Open the legacy direct RGB-D capture window.")
    mode.add_argument("--record", action="store_true", help="Record raw RGB-D frames for later offline processing.")
    mode.add_argument(
        "--process-recordings",
        action="store_true",
        help="Process raw recording sessions into accepted FoundationPose references.",
    )
    mode.add_argument(
        "--reselect-recordings",
        action="store_true",
        help="Republish references from the latest processing cache without rerunning SAM.",
    )
    parser.add_argument("--object", help="Object profile name.")
    parser.add_argument("--data-root")
    parser.add_argument("--rgb", help="RGB image path for --detect-only.")
    parser.add_argument("--intrinsics", help="Intrinsics JSON or K.txt path for --detect-only with --rgb.")
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d435")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--record-interval-s", type=float, default=0.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=8)
    parser.add_argument("--square-length-m", type=float, default=0.030)
    parser.add_argument("--marker-length-m", type=float, default=0.022)
    parser.add_argument("--dictionary", default="auto")
    parser.add_argument("--legacy-pattern", action="store_true")
    parser.add_argument("--board-t-object", help="Path to 4x4 txt or JSON board_T_object.")
    parser.add_argument("--object-xyz-m", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--object-rpy-deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("R", "P", "Y"))
    parser.add_argument("--capture-once", action="store_true", help="For --detect-only, grab one live frame and exit.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Discard this many live frames before --capture-once to let camera auto exposure/white balance settle.",
    )
    parser.add_argument("--preview-output", help="Write a ChArUco board-axis overlay preview image.")
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--min-corners", type=int, default=6)
    parser.add_argument("--min-markers", type=int, default=2)
    parser.add_argument("--max-reprojection-error-px", type=float, default=4.0)
    parser.add_argument("--min-image-coverage-fraction", type=float, default=0.005)
    parser.add_argument("--required-keyframes", type=int, default=16)
    parser.add_argument("--max-keyframes", type=int, default=32)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.0005)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.05)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        board_spec = CharucoBoardSpec(
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_m=args.square_length_m,
            marker_length_m=args.marker_length_m,
            dictionary=args.dictionary,
            legacy_pattern=args.legacy_pattern,
        )
        quality_config = CharucoQualityConfig(
            min_corners=args.min_corners,
            min_markers=args.min_markers,
            max_reprojection_error_px=args.max_reprojection_error_px,
            min_image_coverage_fraction=args.min_image_coverage_fraction,
        )
        board_object = _board_object_from_args(args, required=not args.detect_only)
        if args.offline_generate:
            payload = _offline_generate(args, board_spec, quality_config, board_object)
        elif args.detect_only:
            payload = _detect_only(args, board_spec, quality_config, board_object)
        elif args.live_capture:
            payload = _live_capture(args, board_spec, quality_config, board_object)
        elif args.record:
            payload = _record_raw(args, board_spec, board_object)
        elif args.process_recordings:
            payload = _process_recordings(args, board_spec, quality_config, board_object)
        else:
            payload = _reselect_recordings(args, board_spec, quality_config, board_object)
    except Exception as exc:
        payload = {"ok": False, "returncode": 2, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_summary(payload))
    return int(payload.get("returncode", 0))


def _offline_generate(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if not args.object:
        raise ValueError("--object is required with --offline-generate")
    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    results = generate_charuco_reference_poses(
        profile,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
    )
    write_charuco_reference_poses(profile, results, board_object=board_object)
    return {
        "ok": True,
        "returncode": 0,
        "mode": "offline_generate",
        "object": profile.name,
        "frame_count": len(results),
        "selected_dictionaries": [result.selected_dictionary for result in results],
        "median_reprojection_error_px": _median_reprojection_error(results),
    }


def _detect_only(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if args.rgb:
        image = _load_rgb(args.rgb)
        intrinsics = _load_intrinsics_file(args.intrinsics) if args.intrinsics else _default_intrinsics_for_image(image)
        result = detect_charuco_pose(
            image,
            intrinsics,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
        )
        preview_path = _write_axis_preview_if_requested(args, image, intrinsics, result)
        return {
            "ok": result.ok,
            "returncode": 0 if result.ok else 1,
            "mode": "detect_only",
            "preview_path": preview_path,
            "result": result.to_metadata(),
        }
    return _live_detect_only(args, board_spec, quality_config, board_object)


def _live_detect_only(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    cv2 = _require_cv2()
    accepted = 0
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        if args.capture_once:
            frame = _read_after_warmup(camera, args.warmup_frames)
            result = detect_charuco_pose(
                frame.rgb,
                frame.intrinsics,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
            )
            preview_path = _write_axis_preview_if_requested(args, frame.rgb, frame.intrinsics, result)
            return {
                "ok": result.ok,
                "returncode": 0 if result.ok else 1,
                "mode": "detect_only_capture_once",
                "preview_path": preview_path,
                "result": result.to_metadata(),
            }
        while True:
            frame = camera.read()
            result = detect_charuco_pose(
                frame.rgb,
                frame.intrinsics,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
            )
            accepted += int(result.ok)
            if result.ok:
                preview = draw_charuco_axes_overlay_bgr(
                    frame.rgb,
                    frame.intrinsics,
                    result,
                    axis_length_m=args.axis_length_m,
                )
                preview = _draw_status_on_bgr(preview, result.to_metadata())
            else:
                preview = _draw_live_status(frame.rgb, result.to_metadata())
            cv2.imshow("ChArUco Detect Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
    cv2.destroyWindow("ChArUco Detect Preview")
    return {"ok": True, "returncode": 0, "mode": "detect_only_live", "accepted_preview_frames": accepted}


def _read_after_warmup(camera, warmup_frames: int):
    frame = None
    for _ in range(max(int(warmup_frames), 0) + 1):
        frame = camera.read()
    if frame is None:
        raise RuntimeError("camera did not provide a frame")
    return frame


def _live_capture(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if not args.object:
        raise ValueError("--object is required with --live-capture")
    cv2 = _require_cv2()
    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    provider = Sam3MaskProvider(
        prompt=args.prompt or profile.prompt,
        device=args.device,
        confidence_threshold=args.threshold,
        resolution=args.sam_resolution,
    )
    accepted = 0
    rejected = 0
    records: list[dict[str, Any]] = []
    accepted_results = []
    accepted_indices: list[int] = []
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        while accepted < args.frames:
            frame = camera.read()
            preview = _draw_live_status(frame.rgb, {"ok": None, "message": "press C/Space to capture, Q to quit"})
            cv2.imshow("ChArUco Reference Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key not in {ord("c"), ord("C"), ord(" ")}:
                continue
            start = time.perf_counter()
            result = detect_charuco_pose(
                frame.rgb,
                frame.intrinsics,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
            )
            timing_ms = {"charuco_pose_ms": (time.perf_counter() - start) * 1000.0}
            if not result.ok or result.cam_in_ob is None:
                rejected += 1
                records.append({"accepted": False, "reason": result.reject_reasons, "charuco_pose": result.to_metadata()})
                continue
            start = time.perf_counter()
            try:
                mask = provider.get_mask(frame.rgb, depth_m=frame.depth_m, object_name=profile.prompt)
            except Exception as exc:
                rejected += 1
                records.append(
                    {
                        "accepted": False,
                        "reason": [f"object mask rejected: {exc}"],
                        "charuco_pose": result.to_metadata(),
                    }
                )
                continue
            timing_ms["segmentation_ms"] = (time.perf_counter() - start) * 1000.0
            index = count_reference_frames(profile)
            save_reference_frame(
                profile,
                index,
                rgb=frame.rgb,
                depth_m=frame.depth_m,
                mask=mask.mask,
                intrinsics=frame.intrinsics,
                cam_in_ob=result.cam_in_ob,
                metadata={
                    "capture_mode": "charuco_live_event",
                    "mask_source": mask.source,
                    "mask_metadata": mask.metadata,
                    "charuco_pose": result.to_metadata(),
                    "timing_ms": timing_ms,
                },
            )
            accepted += 1
            accepted_results.append(result)
            accepted_indices.append(index)
            records.append({"accepted": True, "index": index, "charuco_pose": result.to_metadata(), "mask_source": mask.source})
    cv2.destroyWindow("ChArUco Reference Capture")
    if accepted:
        record_charuco_pose_provenance(
            profile,
            accepted_results,
            board_object=board_object,
            indices=accepted_indices,
        )
    release = getattr(provider, "release", None)
    if callable(release):
        release()
    return {
        "ok": accepted > 0,
        "returncode": 0 if accepted > 0 else 1,
        "mode": "live_capture",
        "object": profile.name,
        "accepted": accepted,
        "rejected": rejected,
        "records": records,
    }


def _record_raw(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if not args.object:
        raise ValueError("--object is required with --record")
    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    config = ReferenceRecordingConfig(
        camera_model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        board_spec=board_spec,
        board_object=board_object,
        sam_device=args.device,
        sam_resolution=args.sam_resolution,
        sam_confidence_threshold=args.threshold,
    )
    start = time.perf_counter()
    frame_limit = max(int(args.frames), 0)
    duration_s = max(float(args.duration_s), 0.0)
    interval_s = max(float(args.record_interval_s), 0.0)
    records = []
    with ReferenceRecordingSession(profile, config=config) as session:
        while True:
            if frame_limit and len(records) >= frame_limit:
                break
            if duration_s and time.perf_counter() - start >= duration_s:
                break
            if not frame_limit and not duration_s and records:
                break
            record = session.record_next_frame()
            records.append(record.to_dict())
            if interval_s > 0:
                time.sleep(interval_s)
        info = session.info()
    return {
        "ok": bool(records),
        "returncode": 0 if records else 1,
        "mode": "record",
        "object": profile.name,
        "session_id": info.session_id,
        "session_dir": str(info.session_dir),
        "recorded": len(records),
        "records": records,
    }


def _process_recordings(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if not args.object:
        raise ValueError("--object is required with --process-recordings")
    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    provider = Sam3MaskProvider(
        prompt=args.prompt or profile.prompt,
        device=args.device,
        confidence_threshold=args.threshold,
        resolution=args.sam_resolution,
    )
    config = ReferenceProcessingConfig(
        required_keyframes=args.required_keyframes,
        max_keyframes=args.max_keyframes,
        min_mask_area_fraction=args.min_mask_area_fraction,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        publish=not args.evaluate_only,
    )
    if args.evaluate_only:
        report = evaluate_recorded_references(
            profile,
            mask_provider=provider,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
            config=config,
        )
    else:
        report = process_recorded_references(
            profile,
            mask_provider=provider,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
            config=config,
        )
    payload = report.to_dict()
    payload["ok"] = report.ok
    payload["returncode"] = 0 if report.accepted > 0 else 1
    payload["mode"] = "process_recordings"
    return payload


def _reselect_recordings(
    args: argparse.Namespace,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> dict[str, Any]:
    if not args.object:
        raise ValueError("--object is required with --reselect-recordings")
    profile = ObjectProfileRegistry(args.data_root).get(args.object)
    config = ReferenceProcessingConfig(
        required_keyframes=args.required_keyframes,
        max_keyframes=args.max_keyframes,
        min_mask_area_fraction=args.min_mask_area_fraction,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        publish=not args.evaluate_only,
    )
    report = reselect_recorded_references(
        profile,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=config,
    )
    payload = report.to_dict()
    payload["ok"] = report.ok
    payload["returncode"] = 0 if report.accepted > 0 else 1
    payload["mode"] = "reselect_recordings"
    return payload


def _board_object_from_args(args: argparse.Namespace, *, required: bool) -> BoardObjectTransform:
    if args.board_t_object:
        return BoardObjectTransform.load(args.board_t_object)
    if args.object_xyz_m is None:
        if not required:
            return BoardObjectTransform.identity()
        raise ValueError("--board-t-object or --object-xyz-m is required")
    return BoardObjectTransform.from_xyz_rpy_deg(
        tuple(float(v) for v in args.object_xyz_m),
        tuple(float(v) for v in args.object_rpy_deg),
    )


def _load_rgb(path: str) -> np.ndarray:
    cv2 = _require_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_intrinsics_file(path: str | None) -> CameraIntrinsics:
    if path is None:
        raise ValueError("--intrinsics is required for RGB detect-only")
    source = str(path)
    if source.lower().endswith(".json"):
        return CameraIntrinsics.from_mapping(json.loads(Path(source).read_text(encoding="utf-8")))
    matrix = np.loadtxt(source).reshape(3, 3)
    return CameraIntrinsics.from_mapping({"camera_matrix": matrix.tolist()})


def _default_intrinsics_for_image(image: np.ndarray) -> CameraIntrinsics:
    height, width = image.shape[:2]
    focal = float(max(width, height))
    return CameraIntrinsics(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def _median_reprojection_error(results) -> float | None:
    values = [
        result.best_candidate.reprojection_error_px
        for result in results
        if result.best_candidate is not None and result.best_candidate.reprojection_error_px is not None
    ]
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _write_axis_preview_if_requested(
    args: argparse.Namespace,
    image_rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    result,
) -> str | None:
    if not args.preview_output:
        return None
    cv2 = _require_cv2()
    output_path = Path(args.preview_output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result.ok:
        overlay = draw_charuco_axes_overlay_bgr(
            image_rgb,
            intrinsics,
            result,
            axis_length_m=args.axis_length_m,
        )
    else:
        overlay = draw_charuco_detection_debug_bgr(image_rgb, result)
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)


def _draw_live_status(image_rgb: np.ndarray, metadata: dict[str, Any]):
    cv2 = _require_cv2()
    image = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return _draw_status_on_bgr(image, metadata)


def _draw_status_on_bgr(image, metadata: dict[str, Any]):
    cv2 = _require_cv2()
    ok = metadata.get("ok")
    text = f"ChArUco: {'OK' if ok else 'WAIT' if ok is None else 'REJECT'}"
    if metadata.get("selected_dictionary"):
        text += f" {metadata['selected_dictionary']}"
    if metadata.get("reject_reasons"):
        text += " | " + "; ".join(str(item) for item in metadata["reject_reasons"][:2])
    if metadata.get("message"):
        text = str(metadata["message"])
    cv2.putText(image, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def _human_summary(payload: dict[str, Any]) -> str:
    if payload.get("ok"):
        return f"{payload.get('mode', 'charuco')}: ok"
    return f"{payload.get('mode', 'charuco')}: failed {payload.get('error', '')}"


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("OpenCV is required for ChArUco reference capture.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
