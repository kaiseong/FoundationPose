from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visual_servoing.run_d405_phone_pose import selected_camera_model, status_json
from visual_servoing.phone_pose.realsense_d405 import (
    D405Camera,
    D435Camera,
    LiveRgbdCamera,
    RealSenseUnavailableError,
    SUPPORTED_LIVE_CAMERA_MODELS,
    bgr_to_rgb,
)
from visual_servoing.phone_pose.sam3_phone_segmenter import select_single_mask


def test_select_single_phone_mask_prefers_score_then_area():
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :3, :3] = True
    masks[2, :4, :4] = True
    scores = np.array([0.9, 0.8, 0.9])

    selection = select_single_mask(masks, scores=scores, min_area=1)

    assert selection.index == 2
    assert selection.area == 16
    assert selection.score == 0.9


def test_missing_pyrealsense2_has_helpful_error(monkeypatch):
    camera = D405Camera()

    def fake_import(name, *args, **kwargs):
        if name == "pyrealsense2":
            raise ImportError("missing")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RealSenseUnavailableError, match="offline mode"):
        camera.start()


def test_d435_camera_wrapper_selects_d435_model():
    camera = D435Camera()

    assert camera.model == "d435"
    assert camera.label == "D435"


def test_realsense_bgr_frames_are_returned_as_rgb():
    image_bgr = np.array([[[10, 20, 30], [1, 2, 3]]], dtype=np.uint8)

    image_rgb = bgr_to_rgb(image_bgr)

    np.testing.assert_array_equal(image_rgb, np.array([[[30, 20, 10], [3, 2, 1]]], dtype=np.uint8))


def test_live_camera_model_selection_prefers_explicit_camera_flag():
    class Args:
        camera = "d435"
        live_d435 = False

    assert selected_camera_model(Args()) == "d435"


def test_live_camera_model_selection_supports_zed_legacy_alias():
    class Args:
        camera = None
        live_zed = True
        live_d435 = False

    assert selected_camera_model(Args()) == "zed"


def test_supported_live_camera_models_include_zed():
    assert "zed" in SUPPORTED_LIVE_CAMERA_MODELS


def test_no_forbidden_dependency_strings_in_implementation():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "phone_pose").glob("*.py")) + [root / "run_d405_phone_pose.py"]
    forbidden_parts = [
        "li" + "lio",
        "li" + "lio_see",
        "li" + "lio_think",
        "example_" + "lili-o_think",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for forbidden in forbidden_parts:
            assert forbidden.lower() not in lowered, f"{forbidden} found in {path}"


def test_status_json_reports_live_frame_failures():
    assert status_json("no_pose", "No usable phone mask was produced.") == (
        '{"status":"no_pose","message":"No usable phone mask was produced."}'
    )
