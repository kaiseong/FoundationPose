from __future__ import annotations

from pathlib import Path

from visual_servoing.foundationpose_model_free import setup_check
from visual_servoing.foundationpose_model_free.setup_check import CheckResult, run_checks, summarize


def test_setup_check_reports_missing_foundationpose_root(tmp_path):
    summary = summarize(run_checks(foundationpose_path=tmp_path / "missing"))

    checks = {item["name"]: item for item in summary["checks"]}
    assert checks["foundationpose_root"]["ok"] is False
    assert "FoundationPose" in checks["foundationpose_root"]["detail"]


def test_setup_check_is_camera_aware(monkeypatch, tmp_path):
    monkeypatch.setattr(
        setup_check,
        "_zed_sdk_check",
        lambda *, required: CheckResult("zed_sdk", True, "mock zed diagnostic", required=required),
    )

    zed_summary = summarize(run_checks(foundationpose_path=tmp_path / "missing", camera="zed"))
    zed_checks = {item["name"]: item for item in zed_summary["checks"]}
    assert "pyrealsense2" not in zed_checks
    assert zed_checks["zed_sdk"]["required"] is True

    d435_summary = summarize(run_checks(foundationpose_path=tmp_path / "missing", camera="d435"))
    d435_checks = {item["name"]: item for item in d435_summary["checks"]}
    assert d435_checks["pyrealsense2"]["required"] is True
    assert "zed_sdk" not in d435_checks


def test_foundationpose_folder_does_not_import_forbidden_runtime_dependencies():
    root = Path(__file__).resolve().parents[2]
    files = list((root / "foundationpose_model_free").glob("*.py"))
    files.extend((root / "scripts").glob("fp_*.py"))
    forbidden_parts = [
        "li" + "lio",
        "li" + "lio_see",
        "li" + "lio_think",
        "example_" + "lili-o_think",
        "rospy",
        "rclpy",
        "sensor_msgs",
        "std_msgs",
        "py" + "zed",
    ]
    for path in files:
        lowered = path.read_text(encoding="utf-8").lower()
        for forbidden in forbidden_parts:
            assert forbidden.lower() not in lowered, f"{forbidden} found in {path}"


def test_point_pose_and_foundationpose_are_separate_folders():
    root = Path(__file__).resolve().parents[2]
    assert (root / "point_pose").is_dir()
    assert (root / "foundationpose_model_free").is_dir()
    assert (root / "phone_pose").is_dir()
