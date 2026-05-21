"""Live RGB-D capture boundaries for point-pose scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rgbd_geometry import CameraIntrinsics


@dataclass(frozen=True)
class RgbdFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics


class RealSenseUnavailableError(RuntimeError):
    pass


SUPPORTED_REALSENSE_MODELS = ("d405", "d435")
SUPPORTED_LIVE_CAMERA_MODELS = SUPPORTED_REALSENSE_MODELS


class LiveRgbdCamera:
    """Factory wrapper for supported live RGB-D camera backends."""

    def __new__(
        cls,
        *,
        model: str = "d405",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ):
        model = model.lower()
        if model in SUPPORTED_REALSENSE_MODELS:
            return RealSenseCamera(model=model, serial=serial, width=width, height=height, fps=fps)
        raise ValueError(f"unsupported RGB-D camera model: {model}")


class RealSenseCamera:
    def __init__(
        self,
        *,
        model: str = "d405",
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        model = model.lower()
        if model not in SUPPORTED_REALSENSE_MODELS:
            raise ValueError(f"unsupported RealSense model: {model}")
        self.model = model
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0

    def start(self) -> None:
        rs = self._import_rs()
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None

    def read(self, *, timeout_ms: int = 1000) -> RgbdFrame:
        if self._pipeline is None or self._align is None:
            raise RuntimeError(f"{self.label}.start() must be called before read().")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"{self.label} did not provide both color and aligned depth frames.")

        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        distortion_coeffs = tuple(float(value) for value in getattr(intr, "coeffs", []))
        intrinsics = CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
            distortion_coeffs=distortion_coeffs or None,
            distortion_model=str(getattr(intr, "model", "")) or None,
        )
        return RgbdFrame(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics)

    @property
    def label(self) -> str:
        return self.model.upper()

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _import_rs(self):
        if self._rs is not None:
            return self._rs
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RealSenseUnavailableError(
                "pyrealsense2 is required for live RealSense capture. Install Intel RealSense "
                "Python bindings or use offline mode with --rgb --depth --mask --intrinsics."
            ) from exc
        self._rs = rs
        return rs


class D405Camera(RealSenseCamera):
    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        super().__init__(model="d405", serial=serial, width=width, height=height, fps=fps)


class D435Camera(RealSenseCamera):
    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        super().__init__(model="d435", serial=serial, width=width, height=height, fps=fps)
