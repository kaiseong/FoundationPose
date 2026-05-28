from __future__ import annotations

import numpy as np

from visual_servoing.phone_pose.overlay import (
    draw_axes_overlay,
    draw_phone_pose_overlay,
    draw_status_overlay,
    project_points,
)
from visual_servoing.phone_pose.rgbd_geometry import CameraIntrinsics
from visual_servoing.point_pose.overlay import _timing_summary


def test_project_axes_uses_intrinsics():
    intr = CameraIntrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0)
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]])

    pixels = project_points(points, intr)

    np.testing.assert_allclose(pixels, [[10.0, 20.0], [20.0, 40.0]])


def test_draw_overlay_changes_image_pixels():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[35:45, 45:55] = True
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0)
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]

    rendered = draw_phone_pose_overlay(image, mask, transform, intr, axis_length_m=0.05)

    assert rendered.shape == image.shape
    assert int(rendered.sum()) > 0
    assert not np.array_equal(rendered, image)


def test_draw_axes_overlay_changes_image_pixels_without_mask():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0)
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 1.0]

    rendered = draw_axes_overlay(image, transform, intr, axis_length_m=0.05)

    assert rendered.shape == image.shape
    assert int(rendered.sum()) > 0
    assert not np.array_equal(rendered, image)


def test_draw_status_overlay_adds_live_status_text():
    image = np.zeros((80, 160, 3), dtype=np.uint8)

    rendered = draw_status_overlay(
        image,
        status="NO POSE",
        prompt="mobile phone",
        frame_index=3,
        fps=12.3,
        message="No usable phone mask was produced.",
    )

    assert rendered.shape == image.shape
    assert int(rendered.sum()) > 0
    assert not np.array_equal(rendered, image)


def test_timing_summary_includes_hybrid_tracking_fields():
    summary = _timing_summary(
        {
            "camera_read_ms": 1.0,
            "remote_segmentation_ms": 2.0,
            "register_ms": 3.0,
            "track_one_ms": 4.0,
            "frame_total_ms": 5.0,
            "cuda_allocated_mb": 10.0,
            "cuda_reserved_mb": 20.0,
        }
    )

    assert "rseg:2.0" in summary
    assert "reg:3.0" in summary
    assert "trk:4.0" in summary
    assert "total:5.0" in summary
    assert "cuda:10/20MB" in summary
