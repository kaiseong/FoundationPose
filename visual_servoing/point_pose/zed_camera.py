"""ZED live RGB-D camera backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .live_camera_config import (
    LiveCameraConfig,
    resolve_live_camera_config,
)
from .rgbd_geometry import CameraIntrinsics
from .realsense_d405 import RgbdFrame


ZED_CONFIDENCE_THRESHOLD = 95
DEFAULT_ZED_DEPTH_MODE = "NEURAL"
ZED_DEPTH_MODES = ("NEURAL", "ULTRA", "QUALITY", "PERFORMANCE")


class ZedUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZedBackendDiagnostic:
    ok: bool
    detail: str


def check_zed_backend() -> ZedBackendDiagnostic:
    try:
        sl = _import_sl()
    except ZedUnavailableError as exc:
        return ZedBackendDiagnostic(False, str(exc))
    device_count = "unknown"
    try:
        devices = sl.Camera.get_device_list()
        device_count = str(len(devices))
    except Exception:
        pass
    return ZedBackendDiagnostic(True, f"ZED SDK Python bindings available; detected devices={device_count}")


class ZedCamera:
    def __init__(
        self,
        *,
        model: str = "zed",
        serial: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int = 15,
        depth_mode: str = DEFAULT_ZED_DEPTH_MODE,
    ) -> None:
        self.config: LiveCameraConfig = resolve_live_camera_config(
            model=model,
            serial=serial,
            width=width,
            height=height,
            fps=fps,
        )
        self.depth_mode = normalize_zed_depth_mode(depth_mode)
        self._sl = None
        self._camera = None
        self._runtime_params = None
        self._image_mat = None
        self._depth_mat = None

    def start(self) -> None:
        sl = self._import_sl()
        camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = zed_depth_mode_constant(sl, self.depth_mode)
        init_params.camera_fps = int(self.config.fps)
        if self.config.sdk_resolution:
            init_params.camera_resolution = getattr(sl.RESOLUTION, self.config.sdk_resolution)
        if self.config.serial:
            try:
                init_params.set_from_serial_number(int(self.config.serial))
            except ValueError as exc:
                raise ValueError("ZED serial must be numeric when provided") from exc

        status = camera.open(init_params)
        if not _is_success(sl, status):
            raise RuntimeError(zed_open_error_message(status, self.depth_mode))

        runtime_params = sl.RuntimeParameters()
        runtime_params.confidence_threshold = ZED_CONFIDENCE_THRESHOLD
        self._camera = camera
        self._runtime_params = runtime_params
        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.close()
        self._camera = None
        self._runtime_params = None
        self._image_mat = None
        self._depth_mat = None

    def read(self, *, timeout_ms: int = 1000) -> RgbdFrame:
        del timeout_ms
        if self._camera is None or self._runtime_params is None:
            raise RuntimeError("ZED.start() must be called before read().")
        sl = self._import_sl()
        status = self._camera.grab(self._runtime_params)
        if not _is_success(sl, status):
            raise RuntimeError(f"ZED camera grab failed: {status}")
        assert self._image_mat is not None
        assert self._depth_mat is not None
        _retrieve_image(self._camera, self._image_mat, sl, sl.VIEW.LEFT)
        _retrieve_measure(self._camera, self._depth_mat, sl, sl.MEASURE.DEPTH)

        rgb = _rgb_from_zed_image(self._image_mat.get_data())
        depth_m = np.asarray(self._depth_mat.get_data()).astype(np.float32, copy=True)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        if rgb.shape[:2] != depth_m.shape[:2]:
            raise RuntimeError(f"ZED RGB/depth shape mismatch: rgb={rgb.shape[:2]} depth={depth_m.shape[:2]}")
        intrinsics = _read_left_intrinsics(self._camera, frame_shape=rgb.shape[:2])
        return RgbdFrame(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics)

    @property
    def label(self) -> str:
        return "ZED"

    def __enter__(self) -> "ZedCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _import_sl(self):
        if self._sl is None:
            self._sl = _import_sl()
        return self._sl


def _import_sl():
    try:
        import pyzed.sl as sl  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional SDK install
        raise ZedUnavailableError(
            "pyzed.sl is required for live ZED capture. Install the Stereolabs ZED SDK "
            "Python bindings or select a RealSense camera/offline input."
        ) from exc
    return sl


def _is_success(sl, status: Any) -> bool:
    return status == getattr(sl.ERROR_CODE, "SUCCESS", 0)


def normalize_zed_depth_mode(depth_mode: str) -> str:
    normalized = str(depth_mode).strip().upper()
    if normalized not in ZED_DEPTH_MODES:
        allowed = ", ".join(ZED_DEPTH_MODES)
        raise ValueError(f"unsupported ZED depth mode: {depth_mode}; allowed: {allowed}")
    return normalized


def zed_depth_mode_constant(sl, depth_mode: str):
    normalized = normalize_zed_depth_mode(depth_mode)
    try:
        return getattr(sl.DEPTH_MODE, normalized)
    except AttributeError as exc:
        raise RuntimeError(f"ZED SDK does not expose DEPTH_MODE.{normalized}") from exc


def zed_open_error_message(status: Any, depth_mode: str) -> str:
    message = f"ZED camera open failed: {status}"
    if normalize_zed_depth_mode(depth_mode) == "NEURAL":
        message += (
            "; NEURAL depth requires a working ZED SDK TensorRT installation. "
            "Install/fix TensorRT for ZED, or retry with --zed-depth-mode ULTRA."
        )
    return message


def _retrieve_image(camera, mat, sl, view) -> None:
    mem_cpu = getattr(getattr(sl, "MEM", None), "CPU", None)
    if mem_cpu is None:
        camera.retrieve_image(mat, view)
    else:
        camera.retrieve_image(mat, view, mem_cpu)


def _retrieve_measure(camera, mat, sl, measure) -> None:
    mem_cpu = getattr(getattr(sl, "MEM", None), "CPU", None)
    if mem_cpu is None:
        camera.retrieve_measure(mat, measure)
    else:
        camera.retrieve_measure(mat, measure, mem_cpu)


def _rgb_from_zed_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise RuntimeError(f"ZED image must be HxWx3 or HxWx4, got shape {array.shape}")
    # ZED VIEW.LEFT is BGRA/BGR; the shared RgbdFrame contract is RGB.
    return array[..., [2, 1, 0]].astype(np.uint8, copy=True)


def _read_left_intrinsics(camera, *, frame_shape: tuple[int, int]) -> CameraIntrinsics:
    information = camera.get_camera_information()
    calibration = information.camera_configuration.calibration_parameters
    left = calibration.left_cam
    image_size = getattr(left, "image_size", None)
    width = int(getattr(image_size, "width", 0) or frame_shape[1])
    height = int(getattr(image_size, "height", 0) or frame_shape[0])
    distortion = getattr(left, "disto", None)
    distortion_coeffs = None
    if distortion is not None:
        coeffs = tuple(float(value) for value in np.asarray(distortion).ravel())
        distortion_coeffs = coeffs or None
    return CameraIntrinsics(
        fx=float(left.fx),
        fy=float(left.fy),
        cx=float(left.cx),
        cy=float(left.cy),
        width=width,
        height=height,
        distortion_coeffs=distortion_coeffs,
        distortion_model="zed_left",
    )
