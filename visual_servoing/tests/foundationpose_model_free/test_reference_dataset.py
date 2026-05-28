from __future__ import annotations

import json

import numpy as np

from visual_servoing.foundationpose_model_free.reference_dataset import (
    count_reference_frames,
    depth_to_uint16_mm,
    has_reference_poses,
    save_reference_frame,
    sanitize_depth_m,
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


def test_sanitize_depth_m_replaces_invalid_depth_with_zero():
    depth = np.array([[1.0, np.nan], [np.inf, -0.2]], dtype=np.float32)

    sanitized, quality = sanitize_depth_m(depth)

    np.testing.assert_allclose(sanitized, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert quality["total_pixels"] == 4
    assert quality["valid_depth_pixels"] == 1
    assert quality["valid_depth_ratio"] == 0.25
    assert quality["invalid_depth_pixels"] == 3
    assert quality["invalid_depth_ratio"] == 0.75
    assert quality["nonfinite_depth_pixels"] == 2
    assert quality["nonpositive_depth_pixels"] == 1
    assert quality["valid_depth_min_m"] == 1.0
    assert quality["valid_depth_max_m"] == 1.0
    assert quality["valid_depth_median_m"] == 1.0


def test_depth_to_uint16_mm_writes_invalid_depth_as_zero():
    depth = np.array([[0.5, np.nan, np.inf], [0.0, -1.0, 2.0]], dtype=np.float32)

    depth_mm = depth_to_uint16_mm(depth)

    np.testing.assert_array_equal(
        depth_mm,
        np.array([[500, 0, 0], [0, 0, 2000]], dtype=np.uint16),
    )


def test_save_reference_frame_sanitizes_invalid_depth_and_records_quality(tmp_path):
    cv2 = __import__("cv2")
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.array([[0.5, np.nan, np.inf], [0.0, -1.0, 2.0]], dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=101.0, cx=1.0, cy=1.0, width=3, height=2)

    save_reference_frame(
        profile,
        0,
        rgb=rgb,
        depth_m=depth,
        mask=mask,
        intrinsics=intr,
        metadata={"source": "test"},
    )

    saved_depth = np.load(profile.depth_dir / "000000.npy")
    np.testing.assert_array_equal(
        saved_depth,
        np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32),
    )
    depth_png = cv2.imread(str(profile.depth_enhanced_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(
        depth_png,
        np.array([[500, 0, 0], [0, 0, 2000]], dtype=np.uint16),
    )
    metadata = json.loads((profile.refs_dir / "000000.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "test"
    assert metadata["depth_quality"]["valid_depth_pixels"] == 2
    assert metadata["depth_quality"]["invalid_depth_pixels"] == 4
    assert metadata["depth_quality"]["valid_depth_ratio"] == 2 / 6
    assert metadata["depth_quality"]["valid_depth_min_m"] == 0.5
    assert metadata["depth_quality"]["valid_depth_max_m"] == 2.0
