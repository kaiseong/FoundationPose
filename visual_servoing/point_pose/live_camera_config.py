"""Shared live RGB-D camera defaults and validation."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_REALSENSE_MODELS = ("d405", "d435")
SUPPORTED_ZED_MODELS = ("zed",)
SUPPORTED_LIVE_CAMERA_MODELS = SUPPORTED_REALSENSE_MODELS + SUPPORTED_ZED_MODELS

DEFAULT_LIVE_CAMERA_MODEL = "d405"
DEFAULT_LIVE_CAMERA_FPS = 15
REALSENSE_DEFAULT_RESOLUTION = (640, 480)
ZED_DEFAULT_RESOLUTION = (672, 376)
ZED_DEFAULT_SDK_RESOLUTION = "VGA"

ZED_RESOLUTION_MODES: dict[tuple[int, int], str] = {
    (672, 376): "VGA",
    (960, 600): "SVGA",
    (1280, 720): "HD720",
    (1920, 1080): "HD1080",
    (1920, 1200): "HD1200",
    (2048, 1536): "HD1536",
    (2208, 1242): "HD2K",
    (2560, 1440): "QHDPLUS",
    (3840, 2160): "HD4K",
}


@dataclass(frozen=True)
class LiveCameraConfig:
    model: str
    serial: str | None
    width: int | None
    height: int | None
    fps: int
    sdk_resolution: str | None = None
    native_resolution: bool = False


def resolve_live_camera_config(
    *,
    model: str = DEFAULT_LIVE_CAMERA_MODEL,
    serial: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = DEFAULT_LIVE_CAMERA_FPS,
) -> LiveCameraConfig:
    model = normalize_camera_model(model)
    resolved_fps = DEFAULT_LIVE_CAMERA_FPS if fps is None else _positive_int("fps", fps)
    width, height = _normalize_dimensions(width, height)

    if model in SUPPORTED_REALSENSE_MODELS:
        if width is None:
            width, height = REALSENSE_DEFAULT_RESOLUTION
        return LiveCameraConfig(
            model=model,
            serial=_clean_serial(serial),
            width=width,
            height=height,
            fps=resolved_fps,
            sdk_resolution=None,
            native_resolution=False,
        )

    if model in SUPPORTED_ZED_MODELS:
        if width is None:
            return LiveCameraConfig(
                model=model,
                serial=_clean_serial(serial),
                width=None,
                height=None,
                fps=resolved_fps,
                sdk_resolution=ZED_DEFAULT_SDK_RESOLUTION,
                native_resolution=True,
            )
        resolution = ZED_RESOLUTION_MODES.get((width, height))
        if resolution is None:
            raise ValueError(
                f"unsupported ZED resolution {width}x{height}; supported native modes: "
                f"{format_supported_zed_resolutions()}"
            )
        return LiveCameraConfig(
            model=model,
            serial=_clean_serial(serial),
            width=width,
            height=height,
            fps=resolved_fps,
            sdk_resolution=resolution,
            native_resolution=True,
        )

    raise ValueError(f"unsupported RGB-D camera model: {model}")


def normalize_camera_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized not in SUPPORTED_LIVE_CAMERA_MODELS:
        raise ValueError(f"unsupported RGB-D camera model: {model}")
    return normalized


def default_camera_resolution(model: str) -> tuple[int, int]:
    model = normalize_camera_model(model)
    if model in SUPPORTED_ZED_MODELS:
        return ZED_DEFAULT_RESOLUTION
    return REALSENSE_DEFAULT_RESOLUTION


def is_default_camera_resolution(model: str, width: int | None, height: int | None) -> bool:
    width, height = _normalize_dimensions(width, height)
    if width is None:
        return True
    return (width, height) == default_camera_resolution(model)


def format_supported_zed_resolutions() -> str:
    return ", ".join(
        f"{width}x{height}({mode})"
        for (width, height), mode in sorted(ZED_RESOLUTION_MODES.items(), key=lambda item: (item[0][1], item[0][0]))
    )


def _normalize_dimensions(width: int | None, height: int | None) -> tuple[int | None, int | None]:
    if width is None and height is None:
        return None, None
    if width is None or height is None:
        raise ValueError("width and height must be provided together or both omitted")
    return _positive_int("width", width), _positive_int("height", height)


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _clean_serial(serial: str | None) -> str | None:
    if serial is None:
        return None
    serial = str(serial).strip()
    return serial or None
