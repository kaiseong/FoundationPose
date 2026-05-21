from __future__ import annotations

import numpy as np

from visual_servoing.phone_pose.rgbd_geometry import (
    CameraIntrinsics,
    backproject_masked_depth,
    construct_transform,
    construct_transform_from_y_axis,
    estimate_phone_pose,
    fit_plane_normal,
)


def test_backproject_masked_depth_returns_camera_points():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.0, cy=1.0)
    depth = np.ones((3, 3), dtype=np.float32)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True

    points, pixels = backproject_masked_depth(depth, mask, intr)

    assert points.shape == (2, 3)
    assert pixels.tolist() == [[1.0, 1.0], [2.0, 1.0]]
    np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(points[1], [0.01, 0.0, 1.0])


def test_backproject_rejects_invalid_depth():
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)
    depth = np.array([[0.0, np.nan], [np.inf, 1.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    points, _ = backproject_masked_depth(depth, mask, intr, min_depth_m=0.05, max_depth_m=2.0)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [0.01, 0.01, 1.0])


def test_fit_phone_plane_normal_faces_camera():
    points = make_phone_points(z=0.6)

    normal = fit_plane_normal(points)

    assert normal[2] < -0.99


def test_estimate_phone_pose_long_edge_matches_synthetic_rectangle():
    points = make_phone_points(width=0.06, height=0.14, z=0.7)

    transform = estimate_phone_pose(points)

    y_axis = transform[:3, 1]
    assert abs(float(np.dot(y_axis, [0.0, 1.0, 0.0]))) > 0.98


def test_estimate_phone_pose_uses_mask_pixels_for_visible_long_edge():
    points = make_phone_points(width=0.06, height=0.14, z=0.7)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    cols, rows = np.meshgrid(np.arange(45, 56), np.arange(20, 81))
    pixels = np.column_stack((cols.ravel(), rows.ravel()))

    transform = estimate_phone_pose(points, pixels_xy=pixels, intrinsics=intr)

    y_axis = transform[:3, 1]
    assert abs(float(np.dot(y_axis, [0.0, 1.0, 0.0]))) > 0.98


def test_construct_transform_is_orthonormal_and_right_handed():
    transform = construct_transform(
        origin=np.array([0.1, 0.2, 0.7]),
        z_axis=np.array([0.0, 0.0, -1.0]),
        x_axis=np.array([1.0, 0.0, 0.0]),
    )

    rot = transform[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-8)
    np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_construct_transform_from_y_axis_is_orthonormal_and_right_handed():
    transform = construct_transform_from_y_axis(
        origin=np.array([0.1, 0.2, 0.7]),
        z_axis=np.array([0.0, 0.0, -1.0]),
        y_axis=np.array([0.0, 1.0, 0.0]),
    )

    rot = transform[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-8)
    np.testing.assert_allclose(transform[:3, 1], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_axis_sign_stabilization_prevents_frame_flip():
    points = make_phone_points(width=0.06, height=0.14, z=0.7)
    first = estimate_phone_pose(points)
    flipped_points = points[::-1].copy()

    second = estimate_phone_pose(flipped_points, previous_transform=first)

    assert float(np.dot(first[:3, 1], second[:3, 1])) > 0.0
    assert float(np.dot(first[:3, 2], second[:3, 2])) > 0.0


def make_phone_points(width: float = 0.06, height: float = 0.14, z: float = 0.7) -> np.ndarray:
    xs = np.linspace(-width / 2.0, width / 2.0, 9)
    ys = np.linspace(-height / 2.0, height / 2.0, 15)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
