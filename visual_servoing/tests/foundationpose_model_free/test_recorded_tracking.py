from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.foundationpose_adapter import StubFoundationPoseAdapter
from visual_servoing.foundationpose_model_free.recorded_tracking import (
    RecordedTrackingReplayConfig,
    replay_recorded_tracking,
)
from visual_servoing.foundationpose_model_free.reference_recording import ReferenceRecordingSession
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
        rgb = np.full((12, 16, 3), self.reads, dtype=np.uint8)
        depth = np.ones((12, 16), dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=100.0, fy=101.0, cx=8.0, cy=6.0, width=16, height=12)
        return RgbdFrame(rgb=rgb, depth_m=depth, intrinsics=intrinsics)


def test_recorded_tracking_replay_uses_live_tracker_boundary(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    with ReferenceRecordingSession(profile, camera=FakeCamera()) as session:
        for _ in range(3):
            session.record_next_frame()

    report = replay_recorded_tracking(
        profile,
        adapter=StubFoundationPoseAdapter(),
        config=RecordedTrackingReplayConfig(max_frames=3, full_frame_initial_mask=True),
    )

    assert report["processed_frames"] == 3
    assert report["tracking_frames"] == 3
    assert report["records"][0]["state"] == "REINIT"
    assert report["records"][1]["state"] == "TRACKING"
    assert report["records"][0]["pose_source"] == "stub_register"
    assert report["records"][1]["pose_source"] == "stub_track_one"
