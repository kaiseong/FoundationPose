from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.charuco_reference import (
    BoardObjectTransform,
    CharucoBoardSpec,
    CharucoPoseResult,
    CharucoQualityConfig,
    DictionaryCandidateResult,
)
from visual_servoing.foundationpose_model_free.mask_provider import MaskResult
from visual_servoing.foundationpose_model_free.reference_dataset import count_reference_frames
from visual_servoing.foundationpose_model_free.reference_processing import (
    READINESS_NEED_MORE_RECORDING,
    READINESS_NO_RECORDINGS,
    READINESS_READY,
    ReferenceProcessingConfig,
    compute_mask_depth_stats,
    evaluate_recorded_references,
    latest_processing_report,
    process_recorded_references,
)
from visual_servoing.foundationpose_model_free.reference_recording import ReferenceRecordingSession
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.realsense_d405 import RgbdFrame
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


class FakeCamera:
    def __init__(self, *, zero_depth_after: int | None = None) -> None:
        self.started = False
        self.reads = 0
        self.zero_depth_after = zero_depth_after

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self, *, timeout_ms: int = 1000) -> RgbdFrame:
        if not self.started:
            raise RuntimeError("not started")
        self.reads += 1
        rgb = np.full((12, 16, 3), self.reads, dtype=np.uint8)
        depth = np.ones((12, 16), dtype=np.float32)
        if self.zero_depth_after is not None and self.reads > self.zero_depth_after:
            depth[:] = 0.0
        intrinsics = CameraIntrinsics(fx=100.0, fy=101.0, cx=8.0, cy=6.0, width=16, height=12)
        return RgbdFrame(rgb=rgb, depth_m=depth, intrinsics=intrinsics)


class FakeMaskProvider:
    def __init__(self, *, fail_on_value: int | None = None) -> None:
        self.fail_on_value = fail_on_value
        self.released = False

    def get_mask(self, image_rgb, *, depth_m=None, object_name=None) -> MaskResult:
        if self.fail_on_value is not None and int(image_rgb[0, 0, 0]) == self.fail_on_value:
            raise RuntimeError("planned mask failure")
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[2:10, 3:13] = True
        return MaskResult(mask=mask, source="fake", metadata={"object_name": object_name})

    def release(self) -> None:
        self.released = True


def test_processing_publishes_references_and_is_idempotent(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=16)

    config = ReferenceProcessingConfig(required_keyframes=16, max_keyframes=32)
    first_provider = FakeMaskProvider()
    first = process_recorded_references(
        profile,
        mask_provider=first_provider,
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=config,
        pose_detector=_fake_pose_detector,
    )
    second = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=config,
        pose_detector=_fake_pose_detector,
    )

    assert first.readiness == READINESS_READY
    assert second.readiness == READINESS_READY
    assert first_provider.released is True
    assert first.accepted == 16
    assert second.accepted == 16
    assert count_reference_frames(profile) == 16
    assert len(list(profile.rgb_dir.glob("*.png"))) == 16
    assert latest_processing_report(profile)["readiness"] == READINESS_READY
    assert (profile.cam_in_ob_dir / "000000.txt").exists()


def test_processing_need_more_recording_uses_appended_sessions(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=8)

    partial = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=16, max_keyframes=32),
        pose_detector=_fake_pose_detector,
    )
    assert partial.readiness == READINESS_NEED_MORE_RECORDING
    assert partial.force_build_allowed is True
    assert partial.accepted == 8

    _record_frames(profile, count=8)
    ready = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=16, max_keyframes=32),
        pose_detector=_fake_pose_detector,
    )
    assert ready.readiness == READINESS_READY
    assert ready.accepted == 16
    assert count_reference_frames(profile) == 16


def test_processing_rejects_invalid_depth(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4, zero_depth_after=2)

    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4, min_valid_depth_ratio=0.5),
        pose_detector=_fake_pose_detector,
    )

    assert report.accepted == 2
    assert report.rejected == 2
    assert any("valid depth ratio" in "; ".join(record["reasons"]) for record in report.records)


def test_processing_evaluate_only_does_not_publish_refs(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)

    report = evaluate_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4, publish=True),
        pose_detector=_fake_pose_detector,
    )

    assert report.readiness == READINESS_READY
    assert report.accepted == 4
    assert report.published is False
    assert count_reference_frames(profile) == 0


def test_processing_reports_no_recordings_without_publishing(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")

    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )

    assert report.readiness == READINESS_NO_RECORDINGS
    assert report.force_build_allowed is False
    assert report.accepted == 0
    assert report.rejected == 0
    assert count_reference_frames(profile) == 0


def test_processing_records_charuco_and_mask_rejection_reasons(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)

    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(fail_on_value=3),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector_reject_value_2,
    )

    reason_text = "\n".join("; ".join(record["reasons"]) for record in report.records)
    assert report.accepted == 2
    assert report.rejected == 2
    assert "charuco rejected" in reason_text
    assert "mask rejected" in reason_text


def test_processing_rejects_pose_detector_exception_per_frame(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)

    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=3, max_keyframes=4),
        pose_detector=_fake_pose_detector_raise_value_2,
    )

    reason_text = "\n".join("; ".join(record["reasons"]) for record in report.records)
    assert report.readiness == READINESS_READY
    assert report.accepted == 3
    assert report.rejected == 1
    assert "charuco rejected: camera_T_board must contain only finite values" in reason_text


def test_mask_depth_stats_handles_empty_mask():
    stats = compute_mask_depth_stats(
        np.ones((3, 4), dtype=np.float32),
        np.zeros((3, 4), dtype=bool),
        min_depth_m=0.05,
        max_depth_m=3.0,
    )

    assert stats["mask_pixels"] == 0
    assert stats["valid_depth_ratio"] == 0.0


def _record_frames(profile, *, count: int, zero_depth_after: int | None = None) -> None:
    with ReferenceRecordingSession(profile, camera=FakeCamera(zero_depth_after=zero_depth_after)) as session:
        for _ in range(count):
            session.record_next_frame()


def _fake_pose_detector(image_rgb, intrinsics, *, board_spec, quality_config, board_object):
    value = int(image_rgb[0, 0, 0])
    yaw_rad = np.deg2rad((value * 22.5) % 360.0)
    cam_in_ob = np.eye(4, dtype=np.float64)
    cam_in_ob[:3, 3] = np.array([np.sin(yaw_rad), 0.0, np.cos(yaw_rad) + 2.0])
    camera_t_board = np.eye(4, dtype=np.float64)
    camera_t_object = np.linalg.inv(cam_in_ob)
    candidate = DictionaryCandidateResult(
        dictionary="DICT_5X5_100",
        ok=True,
        squares_x=board_spec.squares_x,
        squares_y=board_spec.squares_y,
        corner_count=28,
        marker_count=20,
        reprojection_error_px=0.25,
        image_coverage_fraction=0.1,
        camera_T_board=camera_t_board,
        camera_T_object=camera_t_object,
        cam_in_ob=cam_in_ob,
        distortion_policy="zero_unavailable",
    )
    return CharucoPoseResult(
        ok=True,
        selected_dictionary="DICT_5X5_100",
        candidates=[candidate],
        board_spec=board_spec,
        quality_config=quality_config,
        opencv_version="test",
        board_coordinate_convention="opencv_charuco_board",
        legacy_pattern=False,
        camera_T_board=camera_t_board,
        camera_T_object=camera_t_object,
        cam_in_ob=cam_in_ob,
    )


def _fake_pose_detector_reject_value_2(image_rgb, intrinsics, *, board_spec, quality_config, board_object):
    if int(image_rgb[0, 0, 0]) == 2:
        return CharucoPoseResult(
            ok=False,
            selected_dictionary=None,
            candidates=[],
            board_spec=board_spec,
            quality_config=quality_config,
            opencv_version="test",
            board_coordinate_convention="opencv_charuco_board",
            legacy_pattern=False,
            camera_T_board=None,
            camera_T_object=None,
            cam_in_ob=None,
            reject_reasons=["planned charuco failure"],
        )
    return _fake_pose_detector(
        image_rgb,
        intrinsics,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
    )


def _fake_pose_detector_raise_value_2(image_rgb, intrinsics, *, board_spec, quality_config, board_object):
    if int(image_rgb[0, 0, 0]) == 2:
        raise ValueError("camera_T_board must contain only finite values")
    return _fake_pose_detector(
        image_rgb,
        intrinsics,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
    )
