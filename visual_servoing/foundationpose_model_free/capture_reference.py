"""Live RGB-D reference capture workflow for object onboarding."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

from visual_servoing.point_pose.realsense_d405 import LiveRgbdCamera, RealSenseCamera, RgbdFrame

from .mask_provider import MaskProvider
from .profile_schema import ObjectProfile
from .reference_dataset import count_reference_frames, save_reference_frame


@dataclass(frozen=True)
class ReferenceCaptureConfig:
    frames: int = 16
    frame_interval_s: float = 0.0
    frame_timeout_ms: int = 5000
    camera_model: str = "d405"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 15


@dataclass(frozen=True)
class ManualReferenceCaptureConfig:
    target_frames: int = 16
    frame_timeout_ms: int = 5000
    camera_model: str = "d405"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 15


@dataclass(frozen=True)
class ManualReferenceCaptureResult:
    index: int
    captured_count: int
    target_frames: int
    mask_source: str
    rgb_path: str | None = None
    mask_path: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.captured_count >= self.target_frames


class ManualReferenceCaptureSession:
    """Event-driven live RGB-D reference capture for GUI/button workflows."""

    def __init__(
        self,
        profile: ObjectProfile,
        *,
        mask_provider: MaskProvider,
        config: ManualReferenceCaptureConfig | None = None,
        camera: RealSenseCamera | None = None,
    ) -> None:
        self.profile = profile
        self.mask_provider = mask_provider
        self.config = config or ManualReferenceCaptureConfig()
        if self.config.target_frames < 1:
            raise ValueError("target_frames must be >= 1")
        self._camera = camera
        self._owned_camera = camera is None
        self._started = False
        self._camera_lock = threading.Lock()
        self._latest_frame: RgbdFrame | None = None
        self._latest_frame_time_s = 0.0

    @property
    def captured_count(self) -> int:
        return count_reference_frames(self.profile)

    @property
    def complete(self) -> bool:
        return self.captured_count >= self.config.target_frames

    def start(self) -> None:
        if self._started:
            return
        if self._camera is None:
            self._camera = LiveRgbdCamera(
                model=self.config.camera_model,
                serial=self.config.serial,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )
            self._owned_camera = True
        self._camera.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._owned_camera and self._camera is not None:
                self._camera.stop()
        finally:
            release = getattr(self.mask_provider, "release", None)
            if callable(release):
                release()
            self._latest_frame = None
            self._latest_frame_time_s = 0.0
        self._started = False

    def read_preview_frame(self) -> RgbdFrame:
        """Read one live frame for an operator preview window."""

        if not self._started or self._camera is None:
            raise RuntimeError("ManualReferenceCaptureSession.start() must be called before read_preview_frame().")
        with self._camera_lock:
            frame = self._camera.read(timeout_ms=self.config.frame_timeout_ms)
            self._latest_frame = frame
            self._latest_frame_time_s = time.perf_counter()
            return frame

    def capture_once(self) -> ManualReferenceCaptureResult:
        """Capture and save exactly one reference frame for one user event."""

        if self.complete:
            raise RuntimeError("manual reference capture target is already complete")
        if not self._started or self._camera is None:
            raise RuntimeError("ManualReferenceCaptureSession.start() must be called before capture_once().")

        index = self.captured_count
        timing_ms: dict[str, float] = {}
        start = time.perf_counter()
        frame = self._capture_frame()
        timing_ms["camera_read_ms"] = _elapsed_ms(start)
        start = time.perf_counter()
        mask = self.mask_provider.get_mask(
            frame.rgb,
            depth_m=frame.depth_m,
            object_name=self.profile.prompt,
        )
        timing_ms["segmentation_ms"] = _elapsed_ms(start)
        start = time.perf_counter()
        save_reference_frame(
            self.profile,
            index,
            rgb=frame.rgb,
            depth_m=frame.depth_m,
            mask=mask.mask,
            intrinsics=frame.intrinsics,
            metadata={
                "capture_mode": "manual_event",
                "mask_source": mask.source,
                "mask_metadata": mask.metadata,
                "timing_ms": timing_ms,
            },
        )
        timing_ms["save_reference_ms"] = _elapsed_ms(start)
        return ManualReferenceCaptureResult(
            index=index,
            captured_count=self.captured_count,
            target_frames=self.config.target_frames,
            mask_source=mask.source,
            rgb_path=str(self.profile.rgb_dir / f"{index:06d}.png"),
            mask_path=str(self.profile.mask_dir / f"{index:06d}.png"),
            timing_ms=timing_ms,
        )

    def _capture_frame(self) -> RgbdFrame:
        with self._camera_lock:
            if self._latest_frame is not None and time.perf_counter() - self._latest_frame_time_s < 0.75:
                frame = self._latest_frame
                self._latest_frame = None
                self._latest_frame_time_s = 0.0
                return frame
            assert self._camera is not None
            frame = self._camera.read(timeout_ms=self.config.frame_timeout_ms)
            self._latest_frame = None
            self._latest_frame_time_s = 0.0
            return frame

    def __enter__(self) -> "ManualReferenceCaptureSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def capture_reference_frames(
    profile: ObjectProfile,
    *,
    mask_provider: MaskProvider,
    config: ReferenceCaptureConfig,
    camera: RealSenseCamera | None = None,
) -> ObjectProfile:
    if config.frames < 1:
        raise ValueError("frames must be >= 1")
    owned_camera = camera is None
    if camera is None:
        camera = LiveRgbdCamera(
            model=config.camera_model,
            serial=config.serial,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
    if owned_camera:
        camera.start()
    try:
        for index in range(config.frames):
            frame = camera.read(timeout_ms=config.frame_timeout_ms)
            mask = mask_provider.get_mask(
                frame.rgb,
                depth_m=frame.depth_m,
                object_name=profile.prompt,
            )
            save_reference_frame(
                profile,
                index,
                rgb=frame.rgb,
                depth_m=frame.depth_m,
                mask=mask.mask,
                intrinsics=frame.intrinsics,
                metadata={"mask_source": mask.source, "mask_metadata": mask.metadata},
            )
            if config.frame_interval_s > 0 and index + 1 < config.frames:
                time.sleep(config.frame_interval_s)
    finally:
        if owned_camera:
            camera.stop()
    return profile


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
