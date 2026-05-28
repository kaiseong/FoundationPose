from __future__ import annotations

import argparse
import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np

from visual_servoing.foundationpose_model_free.mask_provider import (
    PrecomputedMaskProvider,
    RemoteSegmentationMaskProvider,
    Sam3MaskProvider,
)
from visual_servoing.foundationpose_model_free.tracker import TrackingRecoveryConfig
from visual_servoing.scripts.fp_track_live import (
    build_mask_provider,
    build_recovery_config,
    combine_status_messages,
    cuda_memory_snapshot,
    emit_json,
    merge_timing_metadata,
    parse_args,
    pose_distance_payload,
    pose_status_message,
    run_live,
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


def test_build_mask_provider_prefers_init_mask_then_remote_then_sam3(tmp_path):
    profile = SimpleNamespace(prompt="multimeter")
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.ones((4, 5), dtype=bool))

    init_args = parse_args(
        [
            "--object",
            "meter",
            "--init-mask",
            str(mask_path),
            "--remote-init-mask-server",
            "192.168.0.3:8081",
        ]
    )
    remote_args = parse_args(
        [
            "--object",
            "meter",
            "--remote-init-mask-server",
            "192.168.0.3:8081",
            "--remote-init-mask-device",
            "cpu",
            "--remote-init-mask-resolution",
            "512",
            "--remote-init-mask-threshold",
            "0.2",
        ]
    )
    local_args = parse_args(["--object", "meter", "--device", "cpu"])

    assert isinstance(build_mask_provider(init_args, profile), PrecomputedMaskProvider)
    remote = build_mask_provider(remote_args, profile)
    assert isinstance(remote, RemoteSegmentationMaskProvider)
    assert remote.server_url == "http://192.168.0.3:8081"
    assert remote.device == "cpu"
    assert remote.resolution == 512
    assert remote.confidence_threshold == 0.2
    assert isinstance(build_mask_provider(local_args, profile), Sam3MaskProvider)


def test_hybrid_recovery_forces_auto_reinit_off():
    args = parse_args(["--object", "meter", "--remote-init-mask-server", "192.168.0.3:8081", "--auto-reinit"])

    config = build_recovery_config(args)

    assert isinstance(config, TrackingRecoveryConfig)
    assert config.auto_reinit is False
    assert args.hybrid_remote_init is True
    assert args.hybrid_auto_reinit_disabled is True


def test_merge_timing_metadata_promotes_stage_timings():
    timing_ms = {"frame_total_ms": 12.0}

    merge_timing_metadata(
        timing_ms,
        {
            "remote_segmentation_ms": 7.5,
            "register_ms": 4.0,
            "track_one_ms": 2.0,
            "ignored": "not numeric",
        },
    )

    assert timing_ms["remote_segmentation_ms"] == 7.5
    assert timing_ms["register_ms"] == 4.0
    assert timing_ms["track_one_ms"] == 2.0
    assert timing_ms["frame_total_ms"] == 12.0


def test_cuda_memory_snapshot_is_safe_without_importing_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert cuda_memory_snapshot() == {}


def test_cuda_memory_snapshot_reports_numeric_fields_when_torch_loaded(monkeypatch):
    class FakeCuda:
        def is_available(self):
            return True

        def memory_allocated(self):
            return 2 * 1024 * 1024

        def memory_reserved(self):
            return 3 * 1024 * 1024

        def max_memory_allocated(self):
            return 4 * 1024 * 1024

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))

    assert cuda_memory_snapshot() == {
        "cuda_allocated_mb": 2.0,
        "cuda_reserved_mb": 3.0,
        "cuda_max_allocated_mb": 4.0,
    }


def test_run_live_sets_frame_total_before_status_overlay():
    source = inspect.getsource(run_live)

    assert source.index('timing_ms["frame_total_ms"] = elapsed_ms(frame_start)') < source.index(
        "overlay = draw_status_overlay("
    )


def test_pose_status_message_formats_overlay_distance_text():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.03, -0.04, 0.30]

    assert pose_status_message(pose, expected_distance_m=0.31) == (
        "x:+0.030m | y:-0.040m | z:+0.300m | dist:0.304m | err:-0.006m"
    )


def test_combine_status_messages_keeps_error_and_pose_text():
    assert combine_status_messages("tracking lost", "x:+0.1m") == "tracking lost | x:+0.1m"
    assert combine_status_messages(None, "x:+0.1m") == "x:+0.1m"
