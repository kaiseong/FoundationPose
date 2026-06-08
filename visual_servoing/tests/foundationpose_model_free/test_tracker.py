from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.foundationpose_adapter import (
    FoundationPoseAdapter,
    FoundationPoseConfig,
    PoseEstimate,
    StubFoundationPoseAdapter,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.foundationpose_model_free.remote_register_provider import RemoteRegisterResult
from visual_servoing.foundationpose_model_free.tracker import (
    FoundationPoseLiveTracker,
    TrackingRecoveryConfig,
    TrackingState,
)
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_tracker_registers_once_then_tracks_with_stub_adapter(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=StubFoundationPoseAdapter())
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)

    first = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    second = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert first.initialized is True
    assert second.initialized is True
    assert first.pose is not None
    assert second.pose is not None
    assert first.state == TrackingState.REINIT
    assert second.state == TrackingState.TRACKING
    assert second.metrics["frames"] == 2.0


def test_tracker_enters_lost_and_holds_last_pose_after_failure(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=FailingTrackAdapter(),
        recovery_config=TrackingRecoveryConfig(hold_last_pose_frames=1),
    )
    rgb, depth, mask, intr = _frame_inputs()

    first = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    lost = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    expired = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert first.state == TrackingState.REINIT
    assert lost.state == TrackingState.LOST
    assert lost.held_pose is True
    assert lost.pose is not None
    assert expired.held_pose is False
    assert expired.pose is None


def test_tracker_manual_reinit_calls_register_again(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=adapter)
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    tracker.request_reinit()
    result = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)

    assert adapter.register_calls == 2
    assert adapter.track_calls == 1
    assert result.state == TrackingState.REINIT


def test_tracker_auto_reinit_is_disabled_by_default(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = FailingTrackAdapter()
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=adapter)
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=None)

    assert adapter.register_calls == 1


def test_tracker_auto_reinit_triggers_after_lost_threshold(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = FailingTrackAdapter()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        mask_provider=AlwaysMaskProvider(),
        recovery_config=TrackingRecoveryConfig(auto_reinit=True, auto_reinit_after_lost_frames=1),
    )
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    reinit = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert adapter.register_calls == 2
    assert reinit.state == TrackingState.REINIT


def test_tracker_releases_reinitialization_mask_provider(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    mask_provider = ReleasableMaskProvider()
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=adapter, mask_provider=mask_provider)
    rgb, depth, _mask, intr = _frame_inputs()

    result = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert result.state == TrackingState.REINIT
    assert result.mask is not None
    assert mask_provider.release_calls == 1


def test_tracker_calls_mask_provider_only_for_init_and_manual_reinit(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    mask_provider = CountingMaskProvider()
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=adapter, mask_provider=mask_provider)
    rgb, depth, _mask, intr = _frame_inputs()

    first = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    second = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    tracker.request_reinit()
    third = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert first.state == TrackingState.REINIT
    assert second.state == TrackingState.TRACKING
    assert third.state == TrackingState.REINIT
    assert mask_provider.get_mask_calls == 2
    assert adapter.register_calls == 2
    assert adapter.track_calls == 1
    assert first.metadata is not None
    assert first.metadata["mask_provider_source"] == "fake"
    assert "register_ms" in first.metadata
    assert "track_one_ms" in second.metadata


def test_tracker_uses_remote_register_provider_for_init_and_manual_reinit(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    remote_register_provider = CountingRemoteRegisterProvider()
    mask_provider = CountingMaskProvider()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        mask_provider=mask_provider,
        remote_register_provider=remote_register_provider,
    )
    rgb, depth, _mask, intr = _frame_inputs()

    first = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    second = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)
    tracker.request_reinit()
    third = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert first.state == TrackingState.REINIT
    assert second.state == TrackingState.TRACKING
    assert third.state == TrackingState.REINIT
    assert remote_register_provider.register_calls == 2
    assert adapter.initialize_from_pose_calls == 2
    assert adapter.register_calls == 0
    assert adapter.track_calls == 1
    assert mask_provider.get_mask_calls == 0
    assert first.mask is None
    assert first.pose is not None
    np.testing.assert_allclose(first.pose.camera_T_object, remote_register_provider.pose)
    assert first.metadata is not None
    assert first.metadata["remote_register_ms"] == 12.0
    assert "remote_register_provider_ms" in first.metadata
    assert "local_pose_seed_ms" in first.metadata
    assert first.metadata["remote_server_timing_ms"] == {"tracking_ms": 11.0}


def test_tracker_warns_when_initial_pose_origin_is_far_from_mask_center(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    adapter.pose[0, 3] = 1.0
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        recovery_config=TrackingRecoveryConfig(warn_initial_pose_mask_alignment=True),
    )
    rgb, depth, mask, intr = _frame_inputs()

    result = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)

    assert result.state == TrackingState.REINIT
    assert result.message is not None
    assert "from SAM3 mask center" in result.message


def test_tracker_marks_tracking_lost_when_pose_origin_leaves_image(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        recovery_config=TrackingRecoveryConfig(verify_pose_depth=True),
    )
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    adapter.pose[0, 3] = 10.0
    lost = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert lost.state == TrackingState.LOST
    assert lost.message is not None
    assert "outside image" in lost.message


def test_tracker_marks_tracking_lost_when_depth_disagrees_with_pose(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        recovery_config=TrackingRecoveryConfig(verify_pose_depth=True),
    )
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    depth[:, :] = 0.2
    lost = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert lost.state == TrackingState.LOST
    assert lost.message is not None
    assert "pose/depth mismatch" in lost.message
    assert lost.metadata is not None
    report = lost.metadata["pose_depth_report"]
    assert report["projected_in_bounds"] is True
    assert report["observed_depth_m"] == 0.20000000298023224
    assert report["depth_error_m"] > 0.7


def test_tracker_marks_tracking_lost_on_pose_jump_when_configured(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        recovery_config=TrackingRecoveryConfig(max_pose_jump_m=0.05),
    )
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    adapter.pose[0, 3] = 0.20
    lost = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert lost.state == TrackingState.LOST
    assert lost.message is not None
    assert "pose jump" in lost.message
    assert lost.metadata is not None
    assert lost.metadata["invalid_pose_reason"] == "pose_jump"


def test_tracker_pose_jump_gate_warns_before_threshold(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        recovery_config=TrackingRecoveryConfig(max_pose_jump_m=0.05, implausible_lost_threshold=2),
    )
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    adapter.pose[0, 3] = 0.20
    warning = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert warning.state == TrackingState.TRACKING
    assert warning.message is not None
    assert warning.metadata is not None
    assert warning.metadata["consecutive_implausible_frames"] == 1


def test_tracker_keeps_tracking_when_pose_depth_check_is_disabled_by_default(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    adapter = CountingAdapter()
    tracker = FoundationPoseLiveTracker(profile=profile, adapter=adapter)
    rgb, depth, mask, intr = _frame_inputs()

    tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    adapter.pose[0, 3] = 10.0
    result = tracker.process_frame(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert result.state == TrackingState.TRACKING
    assert result.pose is not None


def test_foundationpose_adapter_resets_torch_defaults_after_register_and_track(monkeypatch):
    reset_calls = []
    cuda_calls = []
    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.foundationpose_adapter.reset_torch_defaults_for_cpu_ops",
        lambda: reset_calls.append(True),
    )
    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.foundationpose_adapter.set_torch_defaults_for_cuda_ops",
        lambda: cuda_calls.append(True),
    )
    adapter = FoundationPoseAdapter(FoundationPoseConfig())
    adapter._estimator = CountingEstimator()
    rgb, depth, mask, intr = _frame_inputs()

    adapter.register(rgb=rgb, depth_m=depth, intrinsics=intr, mask=mask)
    adapter.track_one(rgb=rgb, depth_m=depth, intrinsics=intr)

    assert len(reset_calls) == 2
    assert len(cuda_calls) == 4


def _frame_inputs():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    return rgb, depth, mask, intr


class CountingAdapter:
    def __init__(self) -> None:
        self.register_calls = 0
        self.track_calls = 0
        self.initialize_from_pose_calls = 0
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[2, 3] = 1.0

    def register(self, *, rgb, depth_m, intrinsics, mask) -> PoseEstimate:
        self.register_calls += 1
        return PoseEstimate(self.pose.copy(), "counting_register", {})

    def track_one(self, *, rgb, depth_m, intrinsics) -> PoseEstimate:
        self.track_calls += 1
        return PoseEstimate(self.pose.copy(), "counting_track", {})

    def initialize_from_pose(self, *, camera_T_object) -> PoseEstimate:
        self.initialize_from_pose_calls += 1
        self.pose = np.asarray(camera_T_object, dtype=np.float64).copy()
        return PoseEstimate(self.pose.copy(), "counting_remote_seed", {"initialized_from_pose": True})


class FailingTrackAdapter(CountingAdapter):
    def track_one(self, *, rgb, depth_m, intrinsics) -> PoseEstimate:
        self.track_calls += 1
        raise RuntimeError("tracking failed")


class AlwaysMaskProvider:
    def get_mask(self, image_rgb, *, depth_m=None, object_name=None):
        from visual_servoing.foundationpose_model_free.mask_provider import MaskResult

        return MaskResult(mask=np.ones(image_rgb.shape[:2], dtype=bool), source="fake")


class ReleasableMaskProvider(AlwaysMaskProvider):
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class CountingMaskProvider(AlwaysMaskProvider):
    def __init__(self) -> None:
        self.get_mask_calls = 0

    def get_mask(self, image_rgb, *, depth_m=None, object_name=None):
        self.get_mask_calls += 1
        return super().get_mask(image_rgb, depth_m=depth_m, object_name=object_name)


class CountingRemoteRegisterProvider:
    def __init__(self) -> None:
        self.register_calls = 0
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = [0.1, -0.2, 0.7]

    def register(self, *, rgb, depth_m, intrinsics) -> RemoteRegisterResult:
        self.register_calls += 1
        metadata = {
            "remote_register_ms": 12.0,
            "remote_server_timing_ms": {"tracking_ms": 11.0},
        }
        return RemoteRegisterResult(PoseEstimate(self.pose.copy(), "fake_remote_register", metadata), metadata)


class CountingEstimator:
    def register(self, *, K, rgb, depth, ob_mask, iteration):
        return np.eye(4, dtype=np.float64)

    def track_one(self, *, rgb, depth, K, iteration):
        return np.eye(4, dtype=np.float64)
