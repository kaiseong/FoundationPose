from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from visual_servoing.foundationpose_model_free.reference_dataset import save_reference_frame
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_fp_charuco_reference_help_exits_successfully():
    completed = subprocess.run(
        [sys.executable, "-m", "visual_servoing.scripts.fp_charuco_reference", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--offline-generate" in completed.stdout
    assert "--live-capture" in completed.stdout
    assert "--record" in completed.stdout
    assert "--process-recordings" in completed.stdout
    assert "--reselect-recordings" in completed.stdout
    assert "--required-keyframes" in completed.stdout
    assert "--max-keyframes" in completed.stdout


def test_fp_charuco_reference_json_error_for_missing_board_object_transform():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_servoing.scripts.fp_charuco_reference",
            "--offline-generate",
            "--object",
            "mouse",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "--board-t-object or --object-xyz-m is required" in payload["error"]


def test_detect_only_writes_axis_preview_for_valid_rgb(tmp_path):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    rgb = _synthetic_charuco_rgb(cv2)
    rgb_path = tmp_path / "charuco.png"
    k_path = tmp_path / "K.txt"
    preview_path = tmp_path / "axis.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.savetxt(k_path, np.array([[600.0, 0.0, 250.0], [0.0, 600.0, 400.0], [0.0, 0.0, 1.0]]))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_servoing.scripts.fp_charuco_reference",
            "--detect-only",
            "--rgb",
            str(rgb_path),
            "--intrinsics",
            str(k_path),
            "--dictionary",
            "DICT_5X5_100",
            "--preview-output",
            str(preview_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["preview_path"] == str(preview_path)
    assert preview_path.exists()


def test_detect_only_writes_debug_preview_for_rejected_rgb(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    rgb_path = tmp_path / "blank.png"
    k_path = tmp_path / "K.txt"
    preview_path = tmp_path / "rejected.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.savetxt(k_path, np.array([[100.0, 0.0, 60.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_servoing.scripts.fp_charuco_reference",
            "--detect-only",
            "--rgb",
            str(rgb_path),
            "--intrinsics",
            str(k_path),
            "--dictionary",
            "DICT_5X5_100",
            "--preview-output",
            str(preview_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["preview_path"] == str(preview_path)
    assert preview_path.exists()


def test_offline_generate_writes_cam_in_ob_for_valid_saved_reference(tmp_path):
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = _synthetic_charuco_rgb(cv2)
    depth = np.ones(rgb.shape[:2], dtype=np.float32)
    mask = np.ones(rgb.shape[:2], dtype=bool)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=250.0, cy=400.0, width=500, height=800)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_servoing.scripts.fp_charuco_reference",
            "--offline-generate",
            "--object",
            "mouse",
            "--data-root",
            str(tmp_path),
            "--dictionary",
            "DICT_5X5_100",
            "--object-xyz-m",
            "0",
            "0",
            "0",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert (profile.cam_in_ob_dir / "000000.txt").exists()
    profile = ObjectProfileRegistry(tmp_path).get("mouse")
    assert profile.metadata["pose_source"] == "charuco_board_jig"


def test_offline_generate_rejects_missing_board_detection_without_writing_pose(tmp_path):
    profile = ObjectProfileRegistry(tmp_path).create("mouse", prompt="wireless mouse")
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    depth = np.ones((80, 100), dtype=np.float32)
    mask = np.ones((80, 100), dtype=bool)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)
    save_reference_frame(profile, 0, rgb=rgb, depth_m=depth, mask=mask, intrinsics=intr)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "visual_servoing.scripts.fp_charuco_reference",
            "--offline-generate",
            "--object",
            "mouse",
            "--data-root",
            str(tmp_path),
            "--dictionary",
            "DICT_5X5_100",
            "--object-xyz-m",
            "0",
            "0",
            "0",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert not (profile.cam_in_ob_dir / "000000.txt").exists()
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "rejected frame" in payload["error"]


def _synthetic_charuco_rgb(cv2):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 8), 0.030, 0.022, dictionary)
    gray = board.generateImage((500, 800))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
