"""Save one or more live RGB-D frames for camera pipeline diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from visual_servoing.common.paths import data_root
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d435")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir, args.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        for _ in range(max(args.warmup, 0)):
            camera.read()
        for index in range(args.frames):
            start = time.perf_counter()
            frame = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0
            frame_dir = output_dir / f"frame_{index:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            summary = save_debug_frame(
                frame_dir,
                rgb=frame.rgb,
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                camera=args.camera,
                read_ms=read_ms,
            )
            summaries.append(summary)

    payload = {"output_dir": str(output_dir), "frames": summaries}
    print(json.dumps(payload, indent=2 if args.json else None, sort_keys=True))
    return 0


def resolve_output_dir(output_dir: str | None, root: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return data_root(root) / "rgbd_debug" / stamp


def save_debug_frame(
    output_dir: Path,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics,
    camera: str,
    read_ms: float,
) -> dict[str, object]:
    cv2 = require_cv2()
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)

    rgb_path = output_dir / "rgb.png"
    depth_npy_path = output_dir / "depth_m.npy"
    depth_mm_path = output_dir / "depth_mm.png"
    depth_color_path = output_dir / "depth_colormap.png"
    valid_path = output_dir / "depth_valid_mask.png"
    preview_path = output_dir / "preview_rgb_depth_valid.png"
    stats_path = output_dir / "stats.json"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    np.save(depth_npy_path, depth)
    cv2.imwrite(str(depth_mm_path), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
    depth_color = colorize_depth(depth)
    cv2.imwrite(str(depth_color_path), depth_color)
    cv2.imwrite(str(valid_path), valid.astype(np.uint8) * 255)
    valid_rgb = np.repeat((valid.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    preview = np.concatenate(
        [
            cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
            depth_color,
            valid_rgb,
        ],
        axis=1,
    )
    cv2.imwrite(str(preview_path), preview)

    summary = {
        "camera": camera,
        "read_ms": round(float(read_ms), 3),
        "rgb_shape": list(rgb_u8.shape),
        "depth_shape": list(depth.shape),
        "intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "depth_stats_m": depth_stats(depth),
        "files": {
            "rgb": str(rgb_path),
            "depth_m": str(depth_npy_path),
            "depth_mm": str(depth_mm_path),
            "depth_colormap": str(depth_color_path),
            "depth_valid_mask": str(valid_path),
            "preview": str(preview_path),
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def depth_stats(depth_m: np.ndarray) -> dict[str, object]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    height, width = depth.shape[:2]
    y0 = max(height // 2 - 10, 0)
    y1 = min(height // 2 + 11, height)
    x0 = max(width // 2 - 10, 0)
    x1 = min(width // 2 + 11, width)
    center_window = depth[y0:y1, x0:x1]
    center_valid = center_window[np.isfinite(center_window) & (center_window > 0.0)]
    if valid.size == 0:
        return {
            "valid_ratio": 0.0,
            "valid_count": 0,
            "center_21x21_median": None,
        }
    percentiles = np.percentile(valid, [1, 5, 50, 95, 99])
    return {
        "valid_ratio": round(float(valid.size / depth.size), 6),
        "valid_count": int(valid.size),
        "min": round(float(valid.min()), 6),
        "p01": round(float(percentiles[0]), 6),
        "p05": round(float(percentiles[1]), 6),
        "median": round(float(percentiles[2]), 6),
        "p95": round(float(percentiles[3]), 6),
        "p99": round(float(percentiles[4]), 6),
        "max": round(float(valid.max()), 6),
        "center_pixel": _finite_depth_or_none(depth[height // 2, width // 2]),
        "center_21x21_median": None if center_valid.size == 0 else round(float(np.median(center_valid)), 6),
        "center_21x21_valid_ratio": round(float(center_valid.size / center_window.size), 6),
    }


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1e-3
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    normalized[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~np.isfinite(depth) | (depth <= 0.0)] = 0
    return color


def _finite_depth_or_none(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    return round(value, 6)


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required for this command.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
