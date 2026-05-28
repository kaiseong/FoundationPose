"""Live tracking loop primitives for one selected FoundationPose profile."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics

from .foundationpose_adapter import PoseEstimate
from .mask_provider import MaskProvider
from .metrics import TrackingMetrics
from .profile_schema import ObjectProfile


class TrackingState:
    TRACKING = "TRACKING"
    LOST = "LOST"
    REINIT = "REINIT"


@dataclass(frozen=True)
class TrackingRecoveryConfig:
    hold_last_pose_frames: int = 0
    auto_reinit: bool = False
    auto_reinit_after_lost_frames: int = 30
    verify_pose_depth: bool = False
    warn_initial_pose_mask_alignment: bool = False
    pose_depth_tolerance_m: float = 0.18
    pose_depth_window_radius_px: int = 7
    max_pose_jump_m: float | None = None
    implausible_lost_threshold: int = 1


@dataclass
class TrackingFrameResult:
    pose: PoseEstimate | None
    initialized: bool
    metrics: dict[str, float]
    state: str = TrackingState.LOST
    fresh_pose: bool = False
    held_pose: bool = False
    message: str | None = None
    mask: np.ndarray | None = None
    metadata: dict[str, object] | None = None


class FoundationPoseLiveTracker:
    def __init__(
        self,
        *,
        profile: ObjectProfile,
        adapter,
        mask_provider: MaskProvider | None = None,
        recovery_config: TrackingRecoveryConfig | None = None,
    ) -> None:
        self.profile = profile
        self.adapter = adapter
        self.mask_provider = mask_provider
        self.recovery_config = recovery_config or TrackingRecoveryConfig()
        self.metrics = TrackingMetrics()
        self.initialized = False
        self.state = TrackingState.LOST
        self.lost_frames = 0
        self.implausible_frames = 0
        self._last_pose: PoseEstimate | None = None
        self._reinit_requested = False

    def request_reinit(self) -> None:
        self._reinit_requested = True

    def process_frame(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        pose = None
        state = TrackingState.TRACKING
        fresh_pose = True
        held_pose = False
        message = None
        init_mask = None
        metadata: dict[str, object] = {
            "manual_reinit_requested": self._reinit_requested,
            "auto_reinit_enabled": self.recovery_config.auto_reinit,
            "hold_last_pose_frames": self.recovery_config.hold_last_pose_frames,
        }
        try:
            should_reinit = self._should_reinit(mask is not None)
            metadata["should_reinit"] = should_reinit
            if not self.initialized or should_reinit:
                state = TrackingState.REINIT
                if mask is None:
                    if self.mask_provider is None:
                        raise ValueError("initialization requires a mask or mask_provider")
                    start = time.perf_counter()
                    mask_result = self.mask_provider.get_mask(
                        rgb,
                        depth_m=depth_m,
                        object_name=self.profile.prompt,
                    )
                    mask_provider_ms = elapsed_ms(start)
                    mask = mask_result.mask
                    metadata["mask_provider_source"] = mask_result.source
                    metadata["mask_provider_ms"] = mask_provider_ms
                    if mask_result.source == "remote_segmentation":
                        metadata["remote_segmentation_ms"] = float(
                            mask_result.metadata.get("remote_segmentation_ms", mask_provider_ms)
                        )
                    if mask_result.confidence is not None:
                        metadata["mask_confidence"] = float(mask_result.confidence)
                    metadata["mask_provider_metadata"] = mask_result.metadata
                    release = getattr(self.mask_provider, "release", None)
                    if callable(release):
                        release()
                init_mask = np.asarray(mask).astype(bool)
                start = time.perf_counter()
                pose = self.adapter.register(
                    rgb=rgb,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    mask=mask,
                )
                metadata["register_ms"] = elapsed_ms(start)
                self.initialized = True
                self._reinit_requested = False
            else:
                start = time.perf_counter()
                pose = self.adapter.track_one(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics)
                metadata["track_one_ms"] = elapsed_ms(start)
        except Exception as exc:
            message = str(exc) or "tracking/reinitialization failed"
            self.metrics.update(None)
            return self._lost_result(message=message, mask=init_mask, metadata=metadata)
        if not _valid_pose(pose.camera_T_object):
            self.metrics.update(None)
            metadata["invalid_pose_reason"] = "non_finite"
            return self._lost_result(message="pose contains non-finite values", metadata=metadata)
        if state != TrackingState.REINIT and self.recovery_config.verify_pose_depth:
            depth_report = _pose_depth_visibility_report(
                pose.camera_T_object,
                depth_m,
                intrinsics,
                tolerance_m=self.recovery_config.pose_depth_tolerance_m,
                radius_px=self.recovery_config.pose_depth_window_radius_px,
            )
            metadata["pose_depth_report"] = depth_report
            message = depth_report["message"]
            if message is not None:
                metadata["invalid_pose_reason"] = "pose_depth_visibility"
                return self._implausible_result(pose=pose, message=message, metadata=metadata)
        if state != TrackingState.REINIT and self.recovery_config.max_pose_jump_m is not None and self._last_pose:
            pose_jump_m = float(
                np.linalg.norm(
                    np.asarray(pose.camera_T_object, dtype=np.float64)[:3, 3]
                    - np.asarray(self._last_pose.camera_T_object, dtype=np.float64)[:3, 3]
                )
            )
            metadata["pose_jump_m"] = pose_jump_m
            metadata["max_pose_jump_m"] = float(self.recovery_config.max_pose_jump_m)
            if pose_jump_m > float(self.recovery_config.max_pose_jump_m):
                metadata["invalid_pose_reason"] = "pose_jump"
                return self._implausible_result(
                    pose=pose,
                    message=f"pose jump {pose_jump_m:.3f}m exceeds {float(self.recovery_config.max_pose_jump_m):.3f}m",
                    metadata=metadata,
                )
        if init_mask is not None and self.recovery_config.warn_initial_pose_mask_alignment:
            message = _pose_mask_alignment_message(
                pose.camera_T_object,
                init_mask,
                intrinsics,
            )
        self.metrics.update(pose.camera_T_object)
        self._last_pose = pose
        self.lost_frames = 0
        self.implausible_frames = 0
        self.state = state if state == TrackingState.REINIT else TrackingState.TRACKING
        metadata["consecutive_implausible_frames"] = 0
        return TrackingFrameResult(
            pose=pose,
            initialized=self.initialized,
            metrics=self.metrics.summary(),
            state=self.state,
            fresh_pose=fresh_pose,
            held_pose=held_pose,
            message=message,
            mask=init_mask,
            metadata=metadata,
        )

    def append_metrics_log(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.metrics.summary(), sort_keys=True) + "\n")

    def _should_reinit(self, has_explicit_mask: bool) -> bool:
        if self._reinit_requested or has_explicit_mask:
            return True
        return (
            self.recovery_config.auto_reinit
            and self.initialized
            and self.mask_provider is not None
            and self.lost_frames >= self.recovery_config.auto_reinit_after_lost_frames
        )

    def _lost_result(
        self,
        *,
        message: str,
        mask: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrackingFrameResult:
        self.lost_frames += 1
        if metadata is None:
            metadata = {}
        metadata["lost_frames"] = self.lost_frames
        self.state = TrackingState.LOST
        held_pose = (
            self._last_pose
            if self._last_pose is not None
            and self.lost_frames <= max(self.recovery_config.hold_last_pose_frames, 0)
            else None
        )
        return TrackingFrameResult(
            pose=held_pose,
            initialized=self.initialized,
            metrics=self.metrics.summary(),
            state=TrackingState.LOST,
            fresh_pose=False,
            held_pose=held_pose is not None,
            message=message,
            mask=mask,
            metadata=metadata,
        )

    def _implausible_result(
        self,
        *,
        pose: PoseEstimate,
        message: str,
        metadata: dict[str, object],
    ) -> TrackingFrameResult:
        self.implausible_frames += 1
        metadata["consecutive_implausible_frames"] = self.implausible_frames
        threshold = max(int(self.recovery_config.implausible_lost_threshold), 1)
        metadata["implausible_lost_threshold"] = threshold
        if self.implausible_frames >= threshold:
            self.metrics.update(None)
            return self._lost_result(message=message, metadata=metadata)
        self.metrics.update(pose.camera_T_object)
        self._last_pose = pose
        self.lost_frames = 0
        self.state = TrackingState.TRACKING
        return TrackingFrameResult(
            pose=pose,
            initialized=self.initialized,
            metrics=self.metrics.summary(),
            state=TrackingState.TRACKING,
            fresh_pose=True,
            held_pose=False,
            message=message,
            metadata=metadata,
        )


def _valid_pose(matrix: np.ndarray) -> bool:
    pose = np.asarray(matrix)
    return pose.shape == (4, 4) and bool(np.all(np.isfinite(pose)))


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _pose_mask_alignment_message(
    camera_t_object: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    warn_px: float = 80.0,
) -> str | None:
    mask_bool = np.asarray(mask).astype(bool)
    rows, cols = np.nonzero(mask_bool)
    if rows.size == 0:
        return "initialization mask is empty"
    origin = np.asarray(camera_t_object, dtype=np.float64)[:3, 3]
    if origin[2] <= 1e-6:
        return "pose origin is behind the camera"
    pose_uv = np.array(
        [
            intrinsics.fx * origin[0] / origin[2] + intrinsics.cx,
            intrinsics.fy * origin[1] / origin[2] + intrinsics.cy,
        ],
        dtype=np.float64,
    )
    mask_uv = np.array([np.median(cols), np.median(rows)], dtype=np.float64)
    error_px = float(np.linalg.norm(pose_uv - mask_uv))
    if error_px <= warn_px:
        return None
    return f"pose origin is {error_px:.0f}px from SAM3 mask center; press R to reinitialize"


def _pose_depth_visibility_message(
    camera_t_object: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    tolerance_m: float,
    radius_px: int,
) -> str | None:
    return _pose_depth_visibility_report(
        camera_t_object,
        depth_m,
        intrinsics,
        tolerance_m=tolerance_m,
        radius_px=radius_px,
    )["message"]


def _pose_depth_visibility_report(
    camera_t_object: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    tolerance_m: float,
    radius_px: int,
) -> dict[str, object]:
    origin = np.asarray(camera_t_object, dtype=np.float64)[:3, 3]
    report: dict[str, object] = {
        "message": None,
        "pose_depth_m": float(origin[2]),
        "projected_u_px": None,
        "projected_v_px": None,
        "image_width_px": int(np.asarray(depth_m).shape[1]) if np.asarray(depth_m).ndim >= 2 else None,
        "image_height_px": int(np.asarray(depth_m).shape[0]) if np.asarray(depth_m).ndim >= 2 else None,
        "projected_in_bounds": False,
        "local_depth_total_px": 0,
        "local_depth_valid_px": 0,
        "local_depth_valid_ratio": 0.0,
        "observed_depth_m": None,
        "depth_error_m": None,
    }
    if origin[2] <= 1e-6:
        report["message"] = "pose origin is behind the camera; tracking lost"
        return report
    u = intrinsics.fx * origin[0] / origin[2] + intrinsics.cx
    v = intrinsics.fy * origin[1] / origin[2] + intrinsics.cy
    report["projected_u_px"] = float(u)
    report["projected_v_px"] = float(v)
    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape[:2]
    if u < 0 or u >= width or v < 0 or v >= height:
        report["message"] = "pose origin projected outside image; tracking lost"
        return report
    report["projected_in_bounds"] = True

    radius = max(int(radius_px), 0)
    col = int(round(u))
    row = int(round(v))
    x0 = max(col - radius, 0)
    x1 = min(col + radius + 1, width)
    y0 = max(row - radius, 0)
    y1 = min(row + radius + 1, height)
    local_depth = depth[y0:y1, x0:x1]
    report["local_depth_total_px"] = int(local_depth.size)
    valid = local_depth[np.isfinite(local_depth) & (local_depth > 0.0)]
    report["local_depth_valid_px"] = int(valid.size)
    report["local_depth_valid_ratio"] = float(valid.size / max(int(local_depth.size), 1))
    if valid.size == 0:
        report["message"] = "no valid depth near projected pose origin; tracking lost"
        return report
    observed_depth = float(np.median(valid))
    report["observed_depth_m"] = observed_depth
    error_m = abs(observed_depth - float(origin[2]))
    report["depth_error_m"] = float(error_m)
    if error_m > float(tolerance_m):
        report["message"] = f"pose/depth mismatch {error_m:.3f}m near projected origin; tracking lost"
    return report
