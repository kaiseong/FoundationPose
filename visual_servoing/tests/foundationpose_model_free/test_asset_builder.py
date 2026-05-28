from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visual_servoing.foundationpose_model_free.asset_builder import (
    FoundationPoseAssetBuilder,
    find_generated_mesh,
    profile_model_path,
)
from visual_servoing.foundationpose_model_free.charuco_reference import POSE_SOURCE
from visual_servoing.foundationpose_model_free.profile_manifest import record_asset_ready
from visual_servoing.foundationpose_model_free.reference_dataset import save_reference_frame
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_asset_builder_command_uses_profile_runner(tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("", encoding="utf-8")
    (foundationpose / "bundlesdf" / "config_ycbv.yml").write_text("{}", encoding="utf-8")
    profile = _profile_with_reference(tmp_path)

    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")
    command = builder.build_command(profile)

    assert "fp_run_profile_nerf.py" in Path(command[1]).name
    assert "--foundationpose-root" in command
    assert str(foundationpose.resolve()) in command
    assert "--output-dir" in command
    assert str(profile.assets_dir / "model") in command


def test_find_generated_mesh_prefers_profile_asset_path(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    mesh_path = profile_model_path(profile)
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("# obj\n", encoding="utf-8")
    record_asset_ready(profile, generated_assets=[mesh_path])

    assert find_generated_mesh(profile) == mesh_path


def test_find_generated_mesh_rejects_stale_profile_asset_path(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    mesh_path = profile_model_path(profile)
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("# obj\n", encoding="utf-8")

    assert find_generated_mesh(profile) is None
    assert find_generated_mesh(profile, require_fresh=False) == mesh_path


def test_asset_builder_dry_run_validates_missing_pose_files(tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("", encoding="utf-8")
    (foundationpose / "bundlesdf" / "config_ycbv.yml").write_text("{}", encoding="utf-8")
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")

    with pytest.raises(ValueError, match="cam_in_ob"):
        builder.build(profile, execute=False)


def test_asset_builder_dry_run_returns_validation_report_for_valid_profile(tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("", encoding="utf-8")
    (foundationpose / "bundlesdf" / "config_ycbv.yml").write_text("{}", encoding="utf-8")
    profile = _profile_with_reference(tmp_path)

    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")
    result = builder.build(profile, execute=False)

    assert result.executed is False
    assert result.returncode == 0
    assert result.validation_report is not None
    assert result.validation_report["ok"] is True


def test_asset_builder_dry_run_reports_charuco_provenance(tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("", encoding="utf-8")
    (foundationpose / "bundlesdf" / "config_ycbv.yml").write_text("{}", encoding="utf-8")
    profile = _profile_with_reference(tmp_path)
    profile.metadata["pose_source"] = POSE_SOURCE
    profile.metadata["pose_provenance"] = {
        "pose_source": POSE_SOURCE,
        "board_spec": {"squares_x": 5, "squares_y": 8},
        "selected_dictionaries": ["DICT_5X5_100"],
        "opencv_version": "test",
    }
    profile.save()

    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")
    result = builder.build(profile, execute=False)

    assert result.validation_report is not None
    assert result.validation_report["pose_source"] == POSE_SOURCE
    assert result.validation_report["charuco_provenance"]["selected_dictionaries"] == ["DICT_5X5_100"]


def test_asset_builder_fingerprints_deterministic_build_inputs(tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    run_nerf = foundationpose / "bundlesdf" / "run_nerf.py"
    config = foundationpose / "bundlesdf" / "config_ycbv.yml"
    run_nerf.write_text("print('v1')\n", encoding="utf-8")
    config.write_text("v: 1\n", encoding="utf-8")
    profile = _profile_with_reference(tmp_path)
    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")

    result = builder.build(profile, execute=False)

    assert result.validation_report is not None
    build_inputs = result.validation_report["deterministic_build_inputs"]
    assert build_inputs["run_nerf"]["sha256"]
    assert build_inputs["config_ycbv"]["sha256"]


def test_asset_builder_execute_uses_egl_for_headless_rendering(monkeypatch, tmp_path):
    foundationpose = tmp_path / "FoundationPose"
    (foundationpose / "bundlesdf").mkdir(parents=True)
    (foundationpose / "bundlesdf" / "run_nerf.py").write_text("print('runner')\n", encoding="utf-8")
    (foundationpose / "bundlesdf" / "config_ycbv.yml").write_text("v: 1\n", encoding="utf-8")
    profile = _profile_with_reference(tmp_path)
    captured_env = {}

    def fake_run(command, *, cwd, env, text, capture_output, check):
        del command, cwd, text, capture_output, check
        captured_env.update(env)
        return SimpleNamespace(returncode=1, stdout="", stderr="simulated failure")

    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.setattr("visual_servoing.foundationpose_model_free.asset_builder.subprocess.run", fake_run)

    builder = FoundationPoseAssetBuilder(foundationpose_root=foundationpose, python_executable="python")
    result = builder.build(profile, execute=True)

    assert result.returncode == 1
    assert captured_env["PYOPENGL_PLATFORM"] == "egl"


def _profile_with_reference(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone", prompt="mobile phone")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr, cam_in_ob=np.eye(4))
    return profile
