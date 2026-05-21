from __future__ import annotations

import argparse
import json

import numpy as np

from visual_servoing.scripts.fp_track_live import (
    combine_status_messages,
    emit_json,
    pose_distance_payload,
    pose_status_message,
)


def test_pose_distance_payload_reports_position_distance_and_error():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.03, -0.04, 0.30]

    payload = pose_distance_payload(pose, expected_distance_m=0.31)

    assert payload["object_position_m"] == {"x": 0.03, "y": -0.04, "z": 0.3}
    assert payload["object_distance_m"] == 0.304138
    assert payload["object_z_m"] == 0.3
    assert payload["expected_distance_m"] == 0.31
    assert payload["distance_error_m"] == -0.005862
    assert payload["distance_abs_error_m"] == 0.005862


def test_emit_json_includes_distance_fields(capsys):
    args = argparse.Namespace(print_json=True, print_timing=False, expected_distance_m=1.0)
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.0, 0.0, 1.1]

    emit_json(args, status="TRACKING", message=None, pose=pose, timing_ms={})

    payload = json.loads(capsys.readouterr().out)
    assert payload["camera_T_object"] == pose.tolist()
    assert payload["object_distance_m"] == 1.1
    assert payload["distance_error_m"] == 0.1


def test_pose_status_message_formats_overlay_distance_text():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.03, -0.04, 0.30]

    assert pose_status_message(pose, expected_distance_m=0.31) == (
        "x:+0.030m | y:-0.040m | z:+0.300m | dist:0.304m | err:-0.006m"
    )


def test_combine_status_messages_keeps_error_and_pose_text():
    assert combine_status_messages("tracking lost", "x:+0.1m") == "tracking lost | x:+0.1m"
    assert combine_status_messages(None, "x:+0.1m") == "x:+0.1m"
