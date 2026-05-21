from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.reference_dataset import count_reference_frames
from visual_servoing.foundationpose_model_free.reference_recording import (
    ReferenceRecordingSession,
    count_recorded_frames,
    list_recording_sessions,
    load_frame_records,
)
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
        intrinsics = CameraIntrinsics(fx=100.0, fy=101.0, cx=5.0, cy=4.0, width=10, height=8)
        return RgbdFrame(rgb=rgb, depth_m=depth, intrinsics=intrinsics)


def test_raw_recording_session_persists_frames_without_final_refs(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    camera = FakeCamera()
    session = ReferenceRecordingSession(profile, camera=camera)

    session.start()
    first = session.record_next_frame()
    second = session.record_next_frame()
    info = session.stop()

    assert first.index == 0
    assert second.index == 1
    assert info.frame_count == 2
    assert count_recorded_frames(info.session_dir) == 2
    assert count_reference_frames(profile) == 0
    assert not profile.refs_dir.joinpath("select_frames.yml").exists()
    assert not profile.refs_dir.joinpath("K.txt").exists()

    records = load_frame_records(info.session_dir)
    assert len(records) == 2
    assert info.session_dir.joinpath(records[0].rgb_path).exists()
    assert info.session_dir.joinpath(records[0].depth_path).exists()
    assert info.session_dir.joinpath(records[0].intrinsics_path).exists()


def test_raw_recording_sessions_append_under_profile(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")

    with ReferenceRecordingSession(profile, camera=FakeCamera()) as first:
        first.record_next_frame()
        first_id = first.session_id
    with ReferenceRecordingSession(profile, camera=FakeCamera()) as second:
        second.record_next_frame()
        second.record_next_frame()
        second_id = second.session_id

    sessions = list_recording_sessions(profile)
    assert first_id != second_id
    assert [session.frame_count for session in sessions] == [1, 2]
    assert count_reference_frames(profile) == 0
