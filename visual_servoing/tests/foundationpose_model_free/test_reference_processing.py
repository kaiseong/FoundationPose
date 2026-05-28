from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from visual_servoing.foundationpose_model_free.charuco_reference import (
    BoardObjectTransform,
    CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
    CharucoBoardSpec,
    CharucoDetectorConfig,
    CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
    CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT,
    CharucoPoseResult,
    CharucoQualityConfig,
    DictionaryCandidateResult,
    charuco_origin_offset_board_m,
    effective_board_T_object,
    normalize_charuco_origin_convention,
)
from visual_servoing.foundationpose_model_free.mask_provider import MaskResult
from visual_servoing.foundationpose_model_free.reference_dataset import count_reference_frames
from visual_servoing.foundationpose_model_free.reference_processing import (
    READINESS_NEED_MORE_RECORDING,
    READINESS_NO_RECORDINGS,
    READINESS_READY,
    EvaluatedCandidate,
    ReferenceProcessingConfig,
    RecordedCandidate,
    compute_mask_depth_stats,
    evaluate_recorded_references,
    index_recorded_frame_records,
    latest_processing_report,
    normalize_excluded_candidate_ids,
    profile_excluded_candidate_ids,
    process_recorded_references,
    reselect_recorded_references,
    select_view_diverse_candidates,
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
        self.calls = 0

    def get_mask(self, image_rgb, *, depth_m=None, object_name=None) -> MaskResult:
        self.calls += 1
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


def test_processing_writes_charuco_axes_previews_for_detected_frames(tmp_path, monkeypatch):
    from visual_servoing.foundationpose_model_free import reference_processing as reference_processing_module

    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=3)

    def fake_draw_axes(image_rgb, intrinsics, result):
        del intrinsics, result
        return np.zeros_like(image_rgb)

    monkeypatch.setattr(reference_processing_module, "draw_charuco_axes_overlay_bgr", fake_draw_axes)
    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=3, max_keyframes=3),
        pose_detector=_fake_pose_detector,
    )

    cache_path = Path(str(report.processing_cache_path))
    previews = sorted((cache_path / "charuco_axes").rglob("*.png"))
    records = json.loads((cache_path / "records.json").read_text(encoding="utf-8"))["records"]
    assert len(previews) == 3
    assert report.processing_summary["charuco_axes_preview_count"] == 3
    assert all(record.get("charuco_axes_preview_path") for record in records)
    assert all((cache_path / str(record["charuco_axes_preview_path"])).exists() for record in records)


def test_reselect_recorded_references_reuses_processing_cache_without_sam(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=8)
    board_spec = CharucoBoardSpec()
    quality_config = CharucoQualityConfig()
    board_object = BoardObjectTransform.identity()

    processed = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )
    reselected = reselect_recorded_references(
        profile,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=2, max_keyframes=2),
    )

    assert processed.processing_cache_path
    assert reselected.readiness == READINESS_READY
    assert reselected.accepted == 2
    assert count_reference_frames(profile) == 2
    metadata = (profile.refs_dir / "000000.json").read_text(encoding="utf-8")
    assert "recording_processing_cache_reselect" in metadata


def test_processing_reuses_existing_cache_for_appended_recordings(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    board_spec = CharucoBoardSpec()
    quality_config = CharucoQualityConfig()
    board_object = BoardObjectTransform.identity()
    _record_frames(profile, count=4)
    first_provider = FakeMaskProvider()

    first = process_recorded_references(
        profile,
        mask_provider=first_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=8),
        pose_detector=_fake_pose_detector,
    )
    _record_frames(profile, count=3)
    second_provider = FakeMaskProvider()
    second = process_recorded_references(
        profile,
        mask_provider=second_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=8),
        pose_detector=_fake_pose_detector,
    )

    assert first_provider.calls == 4
    assert first.accepted == 4
    assert second_provider.calls == 3
    assert second.accepted == 7
    assert second.processing_summary["cache_mode"] == "incremental"
    assert second.processing_summary["reused_cached_records"] == 4
    assert second.processing_summary["newly_processed_candidates"] == 3
    assert count_reference_frames(profile) == 7


def test_processing_cache_is_partitioned_by_charuco_detector_preset(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    board_spec = CharucoBoardSpec()
    quality_config = CharucoQualityConfig()
    board_object = BoardObjectTransform.identity()
    _record_frames(profile, count=4)

    default_provider = FakeMaskProvider()
    process_recorded_references(
        profile,
        mask_provider=default_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        detector_config=CharucoDetectorConfig(CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT),
        pose_detector=_fake_pose_detector,
    )
    tuned_provider = FakeMaskProvider()
    tuned = process_recorded_references(
        profile,
        mask_provider=tuned_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        detector_config=CharucoDetectorConfig(CHARUCO_DETECTOR_PRESET_CONSERVATIVE),
        pose_detector=_fake_pose_detector,
    )
    reused_provider = FakeMaskProvider()
    reused = process_recorded_references(
        profile,
        mask_provider=reused_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        detector_config=CharucoDetectorConfig(CHARUCO_DETECTOR_PRESET_CONSERVATIVE),
        pose_detector=_fake_pose_detector,
    )

    assert default_provider.calls == 4
    assert tuned_provider.calls == 4
    assert tuned.processing_summary["reused_cached_records"] == 0
    assert tuned.processing_summary["detector_preset"] == CHARUCO_DETECTOR_PRESET_CONSERVATIVE
    assert reused_provider.calls == 0
    assert reused.processing_summary["reused_cached_records"] == 4


def test_processing_cache_is_partitioned_by_charuco_origin_convention(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    board_spec = CharucoBoardSpec()
    quality_config = CharucoQualityConfig()
    board_object = BoardObjectTransform.identity()
    _record_frames(profile, count=4)

    default_provider = FakeMaskProvider()
    process_recorded_references(
        profile,
        mask_provider=default_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )
    opencv_provider = FakeMaskProvider()
    opencv = process_recorded_references(
        profile,
        mask_provider=opencv_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        charuco_origin_convention=CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
        pose_detector=_fake_pose_detector,
    )
    reused_provider = FakeMaskProvider()
    reused = process_recorded_references(
        profile,
        mask_provider=reused_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        charuco_origin_convention=CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
        pose_detector=_fake_pose_detector,
    )

    assert default_provider.calls == 4
    assert opencv_provider.calls == 4
    assert opencv.processing_summary["charuco_origin_convention"] == CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD
    assert reused_provider.calls == 0
    assert reused.processing_summary["reused_cached_records"] == 4


def test_reselect_rejects_changed_charuco_origin_convention(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)

    process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )

    with np.testing.assert_raises_regex(ValueError, "different ChArUco origin convention"):
        reselect_recorded_references(
            profile,
            board_spec=CharucoBoardSpec(),
            quality_config=CharucoQualityConfig(),
            board_object=BoardObjectTransform.identity(),
            config=ReferenceProcessingConfig(required_keyframes=2, max_keyframes=2),
            charuco_origin_convention=CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
        )


def test_reselect_excludes_candidate_ids_and_persists_profile_setting(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=6)
    board_spec = CharucoBoardSpec()
    quality_config = CharucoQualityConfig()
    board_object = BoardObjectTransform.identity()

    processed = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )
    excluded_id = next(record["candidate_id"] for record in processed.records if record.get("selected_index") == 0)
    reselected = reselect_recorded_references(
        profile,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        excluded_candidate_ids=excluded_id,
    )

    selected_ids = {record["candidate_id"] for record in reselected.records if record.get("accepted")}
    excluded_records = [record for record in reselected.records if record.get("excluded")]
    assert excluded_id not in selected_ids
    assert excluded_records and excluded_records[0]["candidate_id"] == excluded_id
    assert excluded_records[0]["reasons"] == ["accepted cached candidate excluded by user"]
    assert profile_excluded_candidate_ids(profile) == (excluded_id,)
    assert reselected.processing_summary["excluded_count"] == 1
    assert count_reference_frames(profile) == 4


def test_processing_exclusions_can_make_ready_recording_need_more_frames(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)
    excluded_ids = sorted(index_recorded_frame_records(profile))[:2]

    report = process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        excluded_candidate_ids=excluded_ids,
        pose_detector=_fake_pose_detector,
    )

    assert report.readiness == READINESS_NEED_MORE_RECORDING
    assert report.accepted == 2
    assert report.force_build_allowed is True
    assert report.processing_summary["eligible_count"] == 4
    assert report.processing_summary["excluded_count"] == 2
    assert report.processing_summary["selected_count"] == 2
    assert set(report.processing_summary["excluded_candidate_ids"]) == set(excluded_ids)


def test_reselect_recorded_references_rejects_changed_object_transform(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    _record_frames(profile, count=4)

    process_recorded_references(
        profile,
        mask_provider=FakeMaskProvider(),
        board_spec=CharucoBoardSpec(),
        quality_config=CharucoQualityConfig(),
        board_object=BoardObjectTransform.identity(),
        config=ReferenceProcessingConfig(required_keyframes=4, max_keyframes=4),
        pose_detector=_fake_pose_detector,
    )

    with np.testing.assert_raises_regex(ValueError, "different Obj XYZ/RPY"):
        reselect_recorded_references(
            profile,
            board_spec=CharucoBoardSpec(),
            quality_config=CharucoQualityConfig(),
            board_object=BoardObjectTransform.from_xyz_rpy_deg((0.1, 0.0, 0.0)),
            config=ReferenceProcessingConfig(required_keyframes=2, max_keyframes=2),
        )


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


def test_view_diverse_selection_balances_oversampled_angle_bins():
    candidates = [
        *_evaluated_candidates_for_bin(0, count=20, start_index=0, score_start=100.0),
        *_evaluated_candidates_for_bin(1, count=3, start_index=100, score_start=70.0),
        *_evaluated_candidates_for_bin(2, count=3, start_index=200, score_start=60.0),
    ]

    selected = select_view_diverse_candidates(
        candidates,
        config=ReferenceProcessingConfig(required_keyframes=3, max_keyframes=9),
    )

    assert len(selected) == 9
    assert Counter(item.view_bin for item in selected) == {0: 3, 1: 3, 2: 3}


def test_view_diverse_selection_skips_excluded_candidates():
    candidates = _evaluated_candidates_for_bin(0, count=4, start_index=0, score_start=100.0)

    selected = select_view_diverse_candidates(
        candidates,
        config=ReferenceProcessingConfig(required_keyframes=3, max_keyframes=3),
        excluded_candidate_ids="session:000000",
    )

    assert [item.candidate.candidate_id for item in selected] == [
        "session:000001",
        "session:000002",
        "session:000003",
    ]


def test_normalize_excluded_candidate_ids_dedupes_common_separators():
    assert normalize_excluded_candidate_ids("a:000001, b:000002\na:000001; c:000003") == (
        "a:000001",
        "b:000002",
        "c:000003",
    )


def test_mask_depth_stats_handles_empty_mask():
    stats = compute_mask_depth_stats(
        np.ones((3, 4), dtype=np.float32),
        np.zeros((3, 4), dtype=bool),
        min_depth_m=0.05,
        max_depth_m=3.0,
    )

    assert stats["mask_pixels"] == 0
    assert stats["valid_depth_ratio"] == 0.0


def _evaluated_candidates_for_bin(
    bin_id: int,
    *,
    count: int,
    start_index: int,
    score_start: float,
) -> list[EvaluatedCandidate]:
    return [
        EvaluatedCandidate(
            candidate=RecordedCandidate(
                session_id="session",
                frame_index=start_index + index,
                session_dir=Path("."),
                frame_record=None,  # type: ignore[arg-type]
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth_m=np.ones((2, 2), dtype=np.float32),
                intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
            ),
            accepted=True,
            reasons=[],
            score=score_start - index,
            view_yaw_deg=float(bin_id),
            view_bin=bin_id,
        )
        for index in range(count)
    ]


def _record_frames(profile, *, count: int, zero_depth_after: int | None = None) -> None:
    with ReferenceRecordingSession(profile, camera=FakeCamera(zero_depth_after=zero_depth_after)) as session:
        for _ in range(count):
            session.record_next_frame()


def _fake_pose_detector(
    image_rgb,
    intrinsics,
    *,
    board_spec,
    quality_config,
    board_object,
    detector_config=None,
    charuco_origin_convention=None,
):
    detector_config = detector_config or CharucoDetectorConfig()
    origin_convention = normalize_charuco_origin_convention(charuco_origin_convention)
    value = int(image_rgb[0, 0, 0])
    yaw_rad = np.deg2rad((value * 22.5) % 360.0)
    cam_in_ob = np.eye(4, dtype=np.float64)
    cam_in_ob[:3, 3] = np.array([np.sin(yaw_rad), 0.0, np.cos(yaw_rad) + 2.0])
    camera_t_board = np.eye(4, dtype=np.float64)
    camera_t_object = np.linalg.inv(cam_in_ob)
    effective = effective_board_T_object(
        board_spec,
        board_object.board_T_object,
        charuco_origin_convention=origin_convention,
    )
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
        detector_preset=detector_config.preset,
        detector_parameters=detector_config.parameter_summary(),
        charuco_origin_convention=origin_convention,
        charuco_origin_offset_board_m=charuco_origin_offset_board_m(board_spec, origin_convention),
        effective_board_T_object=effective,
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
        charuco_origin_convention=origin_convention,
        charuco_origin_offset_board_m=charuco_origin_offset_board_m(board_spec, origin_convention),
        camera_T_board=camera_t_board,
        camera_T_object=camera_t_object,
        cam_in_ob=cam_in_ob,
        user_board_T_object=board_object.board_T_object,
        effective_board_T_object=effective,
        detector_preset=detector_config.preset,
        detector_parameters=detector_config.parameter_summary(),
    )


def _fake_pose_detector_reject_value_2(
    image_rgb,
    intrinsics,
    *,
    board_spec,
    quality_config,
    board_object,
    detector_config=None,
):
    if int(image_rgb[0, 0, 0]) == 2:
        detector_config = detector_config or CharucoDetectorConfig()
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
            detector_preset=detector_config.preset,
            detector_parameters=detector_config.parameter_summary(),
        )
    return _fake_pose_detector(
        image_rgb,
        intrinsics,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        detector_config=detector_config,
    )


def _fake_pose_detector_raise_value_2(
    image_rgb,
    intrinsics,
    *,
    board_spec,
    quality_config,
    board_object,
    detector_config=None,
):
    if int(image_rgb[0, 0, 0]) == 2:
        raise ValueError("camera_T_board must contain only finite values")
    return _fake_pose_detector(
        image_rgb,
        intrinsics,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        detector_config=detector_config,
    )
