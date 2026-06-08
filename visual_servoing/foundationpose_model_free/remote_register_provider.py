"""Remote FoundationPose register provider for local tracking bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any
from urllib import request as urllib_request

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    decode_foundationpose_response,
    encode_foundationpose_track_request,
)

from .foundationpose_adapter import PoseEstimate


FOUNDATIONPOSE_TRACK_PATH = "/foundationpose/v2/track"


class RemoteRegisterError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteRegisterResult:
    pose: PoseEstimate
    metadata: dict[str, Any] = field(default_factory=dict)


class RemoteFoundationPoseRegisterProvider:
    def __init__(
        self,
        server: str,
        *,
        profile: str,
        foundationpose_root: str | None = None,
        refine_iterations: int = 5,
        track_iterations: int = 2,
        prompt: str = "object",
        device: str = "cuda",
        threshold: float = 0.3,
        resolution: int = 1008,
        timeout_s: float = 10.0,
        recovery_options: dict[str, Any] | None = None,
    ) -> None:
        self.server_url = _normalize_foundationpose_server_url(server)
        self.profile = str(profile).strip()
        if not self.profile:
            raise ValueError("remote register profile is required")
        self.foundationpose_root = foundationpose_root
        self.refine_iterations = int(refine_iterations)
        self.track_iterations = int(track_iterations)
        self.prompt = str(prompt).strip() or "object"
        self.device = str(device).strip() or "cuda"
        self.threshold = float(threshold)
        self.resolution = int(resolution)
        self.timeout_s = float(timeout_s)
        self.recovery_options = dict(recovery_options or {})

    def register(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> RemoteRegisterResult:
        request_id = f"remote-register-{time.monotonic_ns()}"
        body = encode_foundationpose_track_request(
            rgb=np.asarray(rgb),
            depth_m=np.asarray(depth_m, dtype=np.float32),
            intrinsics=intrinsics,
            request_id=request_id,
            frame_index=-1,
            capture_monotonic_ns=time.monotonic_ns(),
            t5_T_camera=np.eye(4, dtype=np.float64),
            profile=self.profile,
            foundationpose_root=self.foundationpose_root,
            refine_iterations=self.refine_iterations,
            track_iterations=self.track_iterations,
            reinit=True,
            mask_options={
                "prompt": self.prompt,
                "device": self.device,
                "threshold": self.threshold,
                "resolution": self.resolution,
            },
            recovery_options=self.recovery_options,
            metadata={"source": "fp_track_live_remote_register"},
        )
        request = urllib_request.Request(
            f"{self.server_url}{FOUNDATIONPOSE_TRACK_PATH}",
            data=body,
            headers={"Content-Type": REQUEST_CONTENT_TYPE},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                payload = decode_foundationpose_response(response.read())
        except Exception as exc:
            raise RemoteRegisterError(f"remote register request failed: {exc}") from exc
        remote_register_ms = (time.perf_counter() - start) * 1000.0
        if payload.get("ok") is not True:
            reason = payload.get("message") or payload.get("reason") or payload.get("status") or "remote response not ok"
            raise RemoteRegisterError(f"remote register failed: {reason}")
        pose = np.asarray(payload.get("camera_T_object"), dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise RemoteRegisterError("remote register response missing a finite camera_T_object")
        metadata = {
            "server": self.server_url,
            "request_id": request_id,
            "profile": self.profile,
            "prompt": self.prompt,
            "device": self.device,
            "resolution": self.resolution,
            "threshold": self.threshold,
            "remote_register_ms": remote_register_ms,
            "remote_tracking_state": payload.get("tracking_state"),
            "remote_status": payload.get("status"),
            "remote_tracker_session_id": payload.get("tracker_session_id"),
            "remote_tracker_metadata": payload.get("tracker_metadata"),
            "remote_server_timing_ms": payload.get("server_timing_ms"),
            "remote_tracker_cache": payload.get("tracker_cache"),
        }
        return RemoteRegisterResult(
            PoseEstimate(pose, "remote_foundationpose_register", metadata.copy()),
            metadata,
        )


def _normalize_foundationpose_server_url(server: str) -> str:
    value = str(server).strip()
    if not value:
        raise ValueError("remote FoundationPose server is required")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")
