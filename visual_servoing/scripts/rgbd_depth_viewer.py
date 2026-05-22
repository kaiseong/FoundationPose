"""OpenCV live RGB-D viewer for RealSense camera diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from visual_servoing.common.paths import data_root
from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, SUPPORTED_LIVE_CAMERA_MODELS
from visual_servoing.scripts.rgbd_capture_debug import colorize_depth, depth_stats, require_cv2, save_debug_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=SUPPORTED_LIVE_CAMERA_MODELS, default="d435")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--data-root")
    parser.add_argument("--snapshot-dir")
    args = parser.parse_args()

    cv2 = require_cv2()
    snapshot_dir = resolve_snapshot_dir(args.snapshot_dir, args.data_root)
    resolution_label = "native" if args.width is None or args.height is None else f"{args.width}x{args.height}"
    window_title = f"RGB-D Viewer | {args.camera} {resolution_label}"

    frame_count = 0
    last_time = time.perf_counter()
    fps_ema = 0.0
    with LiveRgbdCamera(
        model=args.camera,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        for _ in range(max(args.warmup, 0)):
            camera.read()
        while True:
            start = time.perf_counter()
            frame = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            fps_ema = inst_fps if fps_ema <= 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)
            last_time = now

            view = build_view(
                frame.rgb,
                frame.depth_m,
                camera_label=args.camera,
                read_ms=read_ms,
                fps=fps_ema,
            )
            cv2.imshow(window_title, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                frame_dir = snapshot_dir / time.strftime("%Y%m%d_%H%M%S")
                frame_dir.mkdir(parents=True, exist_ok=True)
                summary = save_debug_frame(
                    frame_dir,
                    rgb=frame.rgb,
                    depth_m=frame.depth_m,
                    intrinsics=frame.intrinsics,
                    camera=args.camera,
                    read_ms=read_ms,
                )
                print(json.dumps({"snapshot": summary["files"]}, sort_keys=True))
            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    cv2.destroyWindow(window_title)
    return 0


def build_view(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    camera_label: str,
    read_ms: float,
    fps: float,
) -> np.ndarray:
    cv2 = require_cv2()
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    depth_color = colorize_depth(depth)
    valid_bgr = np.repeat((valid.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    view = np.concatenate([rgb_bgr, depth_color, valid_bgr], axis=1)

    stats = depth_stats(depth)
    text_lines = [
        f"{camera_label.upper()} read={read_ms:.1f}ms fps={fps:.1f}",
        f"shape rgb={rgb_u8.shape[1]}x{rgb_u8.shape[0]} depth={depth.shape[1]}x{depth.shape[0]}",
        f"valid={float(stats['valid_ratio']) * 100.0:.1f}% median={stats.get('median')}m center={stats.get('center_pixel')}m",
        "left: RGB | middle: depth colormap | right: valid depth mask | q/esc: quit | s: save",
    ]
    for i, line in enumerate(text_lines):
        y = 24 + i * 24
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def resolve_snapshot_dir(snapshot_dir: str | None, root: str | None) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir).expanduser().resolve()
    return data_root(root) / "rgbd_viewer_snapshots"


if __name__ == "__main__":
    raise SystemExit(main())
