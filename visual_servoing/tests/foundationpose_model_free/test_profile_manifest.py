from __future__ import annotations

import json
import os

import numpy as np

from visual_servoing.foundationpose_model_free.profile_manifest import (
    ASSET_STATUS_READY,
    ASSET_STATUS_STALE,
    ManifestError,
    collect_artifact_records,
    compute_source_dependency_fingerprint,
    read_profile_manifest,
    record_asset_ready,
    refresh_profile_manifest,
)
from visual_servoing.foundationpose_model_free.charuco_reference import POSE_SOURCE
from visual_servoing.foundationpose_model_free.reference_dataset import save_reference_frame
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_manifest_contains_required_contract_fields(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")

    manifest = read_profile_manifest(profile)

    assert manifest["schema_version"] == 1
    assert manifest["source_dependency_fingerprint"]
    assert isinstance(manifest["artifact_records"], list)
    assert isinstance(manifest["deterministic_validation_report"], dict)
    assert isinstance(manifest["heuristic_policy"], dict)
    assert isinstance(manifest["heuristic_report"], dict)


def test_legacy_ready_profile_lazily_migrates_to_stale_without_deleting_assets(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    mesh = profile.assets_dir / "model" / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# obj\n", encoding="utf-8")
    profile.manifest_path.unlink()
    profile.asset_status = ASSET_STATUS_READY
    profile.generated_assets = [str(mesh)]
    profile.save()

    migrated = read_profile_manifest(profile)

    assert profile.asset_status == ASSET_STATUS_STALE
    assert mesh.exists()
    assert migrated["stale_reason"] == "legacy profile without manifest"
    assert collect_artifact_records(profile)[0]["exists"] is True


def test_corrupt_manifest_fails_closed(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    profile.manifest_path.write_text("{not-json", encoding="utf-8")

    try:
        read_profile_manifest(profile)
    except ManifestError as exc:
        assert "corrupt or torn" in str(exc)
    else:
        raise AssertionError("expected ManifestError")


def test_heuristic_policy_update_does_not_change_source_fingerprint(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    before = compute_source_dependency_fingerprint(profile)

    refresh_profile_manifest(
        profile,
        reason="heuristic_policy_update",
        heuristic_policy={"tracking_watchdogs_affect_assets": False, "max_pose_jump_m": 0.1},
        heuristic_report={"ok": True, "warnings": []},
    )

    after = json.loads(profile.manifest_path.read_text(encoding="utf-8"))["source_dependency_fingerprint"]
    assert before == after


def test_source_fingerprint_ignores_mtime_when_contents_are_unchanged(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)
    before = compute_source_dependency_fingerprint(profile)

    os.utime(profile.refs_dir / "K.txt", None)

    assert compute_source_dependency_fingerprint(profile) == before


def test_record_asset_ready_writes_fresh_manifest(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    mesh = profile.assets_dir / "model" / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# obj\n", encoding="utf-8")

    manifest = record_asset_ready(profile, generated_assets=[mesh])

    assert profile.asset_status == ASSET_STATUS_READY
    assert manifest["asset_status"] == ASSET_STATUS_READY
    assert manifest["artifact_records"][0]["exists"] is True


def test_charuco_pose_provenance_changes_source_fingerprint(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse")
    before = compute_source_dependency_fingerprint(profile)
    profile.metadata["pose_source"] = POSE_SOURCE
    profile.metadata["pose_provenance"] = {
        "pose_source": POSE_SOURCE,
        "board_spec": {"squares_x": 5, "squares_y": 8},
        "board_object_transform": {"board_T_object": np.eye(4).tolist()},
        "board_coordinate_convention": "opencv_charuco_board",
        "distortion_policy": {"zero_unavailable": 1},
    }

    after = compute_source_dependency_fingerprint(profile)

    assert after != before
