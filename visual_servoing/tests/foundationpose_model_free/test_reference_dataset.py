from __future__ import annotations

import numpy as np
import json

from visual_servoing.foundationpose_model_free.reference_dataset import (
    count_reference_frames,
    has_reference_poses,
    save_reference_frame,
)
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_save_reference_frame_writes_rgb_depth_mask_and_pose(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)

    save_reference_frame(
        profile,
        0,
        rgb=rgb,
        depth_m=depth,
        mask=mask,
        intrinsics=intr,
        cam_in_ob=np.eye(4),
    )

    assert count_reference_frames(profile) == 1
    assert has_reference_poses(profile)
    assert (profile.rgb_dir / "000000.png").exists()
    assert (profile.depth_dir / "000000.npy").exists()
    assert (profile.depth_enhanced_dir / "000000.png").exists()
    assert (profile.mask_dir / "000000.png").exists()
    assert (profile.cam_in_ob_dir / "000000.txt").exists()
    assert (profile.refs_dir / "select_frames.yml").exists()


def test_save_reference_frame_stores_distortion_coefficients_when_present(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(
        fx=100.0,
        fy=101.0,
        cx=6.0,
        cy=5.0,
        width=12,
        height=10,
        distortion_coeffs=(0.1, 0.2, 0.3, 0.4, 0.5),
        distortion_model="brown_conrady",
    )

    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    saved = json.loads((profile.refs_dir / "intrinsics.json").read_text(encoding="utf-8"))
    assert saved["distortion_coeffs"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert saved["distortion_model"] == "brown_conrady"
    assert saved["distortion_policy"] == "camera_coefficients"


def test_save_reference_frame_records_zero_distortion_fallback_when_coefficients_missing(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)

    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    saved = json.loads((profile.refs_dir / "intrinsics.json").read_text(encoding="utf-8"))
    assert saved["distortion_coeffs"] is None
    assert saved["distortion_policy"] == "zero_unavailable"


def test_save_reference_frame_marks_existing_assets_stale(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    mesh = profile.assets_dir / "model" / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# obj\n", encoding="utf-8")
    profile.asset_status = "ready"
    profile.generated_assets = [str(mesh)]
    profile.save()
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.ones((10, 12), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=6.0, cy=5.0, width=12, height=10)

    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    assert profile.asset_status == "stale"
    assert profile.metadata["asset_stale_reason"] == "reference frame 000000 saved"
