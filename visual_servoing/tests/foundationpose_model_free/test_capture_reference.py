from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visual_servoing.foundationpose_model_free.capture_reference import (
    ManualReferenceCaptureConfig,
    ManualReferenceCaptureSession,
)
from visual_servoing.foundationpose_model_free.mask_provider import MaskResult
from visual_servoing.foundationpose_model_free.reference_dataset import count_reference_frames
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.realsense_d405 import RgbdFrame
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.reads = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self, *, timeout_ms: int = 1000) -> RgbdFrame:
        if not self.started:
            raise RuntimeError("not started")
        self.reads += 1
        rgb = np.full((8, 10, 3), self.reads, dtype=np.uint8)
        depth = np.ones((8, 10), dtype=np.float32)
        intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=5.0, cy=4.0, width=10, height=8)
        return RgbdFrame(rgb=rgb, depth_m=depth, intrinsics=intr)


class FakeMaskProvider:
    def get_mask(self, image_rgb, *, depth_m=None, object_name=None) -> MaskResult:
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:6, 3:8] = True
        return MaskResult(mask=mask, source="fake", metadata={"object_name": object_name})


class ReleasableMaskProvider(FakeMaskProvider):
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


def test_manual_capture_session_saves_nothing_before_event(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    session = ManualReferenceCaptureSession(
        profile,
        mask_provider=FakeMaskProvider(),
        camera=FakeCamera(),
    )

    session.start()

    assert count_reference_frames(profile) == 0
    session.stop()


def test_manual_capture_session_saves_one_reference_per_event(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    camera = FakeCamera()
    session = ManualReferenceCaptureSession(
        profile,
        mask_provider=FakeMaskProvider(),
        config=ManualReferenceCaptureConfig(target_frames=2),
        camera=camera,
    )

    session.start()
    first = session.capture_once()
    second = session.capture_once()

    assert first.index == 0
    assert second.index == 1
    assert camera.reads == 2
    assert count_reference_frames(profile) == 2
    assert first.rgb_path is not None
    assert first.mask_path is not None
    assert second.rgb_path is not None
    assert second.mask_path is not None
    assert Path(first.rgb_path).exists()
    assert Path(first.mask_path).exists()
    assert second.complete is True
    session.stop()


def test_manual_capture_session_captures_latest_preview_frame(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    camera = FakeCamera()
    session = ManualReferenceCaptureSession(
        profile,
        mask_provider=FakeMaskProvider(),
        camera=camera,
    )

    session.start()
    preview_frame = session.read_preview_frame()
    result = session.capture_once()

    assert camera.reads == 1
    assert int(preview_frame.rgb[0, 0, 0]) == 1
    assert result.index == 0
    assert count_reference_frames(profile) == 1
    session.stop()


def test_manual_capture_session_rejects_capture_after_target(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    session = ManualReferenceCaptureSession(
        profile,
        mask_provider=FakeMaskProvider(),
        config=ManualReferenceCaptureConfig(target_frames=1),
        camera=FakeCamera(),
    )

    session.start()
    session.capture_once()

    with pytest.raises(RuntimeError, match="target"):
        session.capture_once()
    session.stop()


def test_manual_capture_session_releases_mask_provider_on_stop(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    provider = ReleasableMaskProvider()
    session = ManualReferenceCaptureSession(
        profile,
        mask_provider=provider,
        camera=FakeCamera(),
    )

    session.start()
    session.stop()
    session.stop()

    assert provider.release_calls == 1
