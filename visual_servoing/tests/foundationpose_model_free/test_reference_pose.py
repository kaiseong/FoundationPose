from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.reference_dataset import save_reference_frame
from visual_servoing.foundationpose_model_free.reference_pose import (
    generate_turntable_cam_in_obs,
    pose_depth_sanity_report,
    write_reference_poses,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_generate_turntable_cam_in_obs_starts_at_inverse_first_pose():
    poses = generate_turntable_cam_in_obs(count=4, axis="y", translation_xyz_m=(0.0, 0.0, 1.0))

    assert len(poses) == 4
    np.testing.assert_allclose(poses[0][:3, 3], [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(poses[0][:3, :3], np.eye(3), atol=1e-9)


def test_write_reference_poses_matches_saved_reference_indices(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    write_reference_poses(profile, [np.eye(4)])

    assert (profile.cam_in_ob_dir / "000000.txt").exists()
    assert profile.metadata["pose_source"] == "manual"


def test_write_reference_poses_stores_turntable_provenance_and_stales_assets(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)
    mesh = profile.assets_dir / "model" / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# obj\n", encoding="utf-8")
    profile.asset_status = "ready"
    profile.generated_assets = [str(mesh)]

    write_reference_poses(
        profile,
        [np.eye(4)],
        pose_source="approximate_turntable",
        pose_provenance={"axis": "y", "distance_m": 0.31, "approximate": True},
    )

    assert profile.asset_status == "stale"
    assert profile.metadata["pose_source"] == "approximate_turntable"
    assert profile.metadata["pose_provenance"]["axis"] == "y"


def test_pose_depth_sanity_warns_on_distance_mismatch(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("phone")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.full((10, 12), 0.5, dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    report = pose_depth_sanity_report(profile, expected_distance_m=0.31)

    assert report["ok"] is False
    assert report["warnings"]
