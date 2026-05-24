"""Replay recorded RGB-D sessions through the FoundationPose live tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mask_provider import MaskProvider
from .profile_schema import ObjectProfile
from .reference_processing import load_recorded_candidate
from .reference_recording import list_recording_sessions, load_frame_records
from .tracker import FoundationPoseLiveTracker, TrackingRecoveryConfig


@dataclass(frozen=True)
class RecordedTrackingReplayConfig:
    session_id: str | None = None
    max_frames: int = 0
    full_frame_initial_mask: bool = False


def replay_recorded_tracking(
    profile: ObjectProfile,
    *,
    adapter,
    config: RecordedTrackingReplayConfig | None = None,
    mask_provider: MaskProvider | None = None,
    recovery_config: TrackingRecoveryConfig | None = None,
) -> dict[str, Any]:
    """Run recorded frames through the same tracker boundary used by live tracking."""

    config = config or RecordedTrackingReplayConfig()
    tracker = FoundationPoseLiveTracker(
        profile=profile,
        adapter=adapter,
        mask_provider=mask_provider,
        recovery_config=recovery_config,
    )
    frame_records = _selected_recorded_frames(profile, session_id=config.session_id)
    if config.max_frames > 0:
        frame_records = frame_records[: int(config.max_frames)]
    records: list[dict[str, Any]] = []
    for output_index, (session_dir, frame_record) in enumerate(frame_records):
        candidate = load_recorded_candidate(session_dir, frame_record)
        mask = None
        if output_index == 0 and config.full_frame_initial_mask:
            mask = np.ones(candidate.rgb.shape[:2], dtype=bool)
        result = tracker.process_frame(
            rgb=candidate.rgb,
            depth_m=candidate.depth_m,
            intrinsics=candidate.intrinsics,
            mask=mask,
        )
        pose_matrix = result.pose.camera_T_object.tolist() if result.pose is not None else None
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "session_id": candidate.session_id,
                "frame_index": int(candidate.frame_index),
                "state": result.state,
                "initialized": bool(result.initialized),
                "fresh_pose": bool(result.fresh_pose),
                "held_pose": bool(result.held_pose),
                "message": result.message,
                "pose_source": result.pose.source if result.pose is not None else None,
                "camera_T_object": pose_matrix,
                "metrics": dict(result.metrics),
                "metadata": dict(result.metadata or {}),
            }
        )
    tracking_frames = sum(1 for record in records if record["camera_T_object"] is not None)
    lost_frames = sum(1 for record in records if record["state"] == "LOST")
    return {
        "object": profile.name,
        "session_id": config.session_id,
        "processed_frames": len(records),
        "tracking_frames": tracking_frames,
        "lost_frames": lost_frames,
        "records": records,
    }


def _selected_recorded_frames(profile: ObjectProfile, *, session_id: str | None):
    selected = []
    for session in list_recording_sessions(profile):
        if session_id is not None and session.session_id != session_id:
            continue
        for record in load_frame_records(session.session_dir):
            selected.append((session.session_dir, record))
    return sorted(selected, key=lambda item: (item[1].session_id, item[1].index))
