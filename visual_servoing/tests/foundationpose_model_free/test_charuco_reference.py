from __future__ import annotations

import numpy as np
import pytest

from visual_servoing.foundationpose_model_free.charuco_reference import (
    BoardObjectTransform,
    CharucoBoardSpec,
    CharucoDetectorConfig,
    CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
    CharucoQualityConfig,
    DictionaryCandidateResult,
    camera_T_object_from_board,
    cam_in_ob_from_camera_T_object,
    choose_best_dictionary_result,
    detect_charuco_pose,
    draw_charuco_axes_overlay_bgr,
    _create_charuco_detector,
)
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


def test_default_board_spec_matches_user_printed_charuco_board():
    spec = CharucoBoardSpec()

    assert spec.squares_x == 5
    assert spec.squares_y == 8
    assert spec.square_length_m == 0.030
    assert spec.marker_length_m == 0.022
    assert spec.candidate_dictionaries() == ("DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000")


def test_camera_T_object_is_camera_T_board_times_board_T_object():
    camera_T_board = np.eye(4)
    camera_T_board[:3, 3] = [0.1, 0.2, 0.3]
    board_T_object = BoardObjectTransform.from_xyz_rpy_deg((0.01, 0.02, 0.03)).board_T_object

    camera_T_object = camera_T_object_from_board(camera_T_board, board_T_object)

    assert np.allclose(camera_T_object, camera_T_board @ board_T_object)


def test_cam_in_ob_is_inverse_of_camera_T_object():
    camera_T_object = BoardObjectTransform.from_xyz_rpy_deg((0.1, 0.2, 0.5), (0.0, 10.0, 20.0)).board_T_object

    cam_in_ob = cam_in_ob_from_camera_T_object(camera_T_object)

    assert np.allclose(cam_in_ob, np.linalg.inv(camera_T_object))


def test_rejects_non_finite_or_wrong_shape_board_object_transform():
    with pytest.raises(ValueError, match="shape"):
        BoardObjectTransform(np.eye(3))
    bad = np.eye(4)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        BoardObjectTransform(bad)


def test_dictionary_auto_detection_prefers_more_valid_corners_then_lower_reprojection_error():
    candidates = [
        DictionaryCandidateResult("DICT_5X5_50", ok=True, corner_count=10, reprojection_error_px=0.5),
        DictionaryCandidateResult("DICT_5X5_100", ok=True, corner_count=12, reprojection_error_px=2.0),
        DictionaryCandidateResult("DICT_5X5_250", ok=True, corner_count=12, reprojection_error_px=0.2),
        DictionaryCandidateResult("DICT_5X5_1000", ok=False, corner_count=30, reprojection_error_px=0.1),
    ]

    best = choose_best_dictionary_result(candidates)

    assert best is not None
    assert best.dictionary == "DICT_5X5_250"


def test_unsupported_dictionary_name_reports_clear_reject_reason():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)

    result = detect_charuco_pose(
        image,
        intr,
        board_spec=CharucoBoardSpec(dictionary="DICT_5X5_DOES_NOT_EXIST"),
        quality_config=CharucoQualityConfig(min_corners=1, min_markers=1),
    )

    assert result.ok is False
    assert "unsupported ChArUco dictionary" in result.reject_reasons[0]


def test_conservative_detector_preset_is_reported_in_metadata():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)

    result = detect_charuco_pose(
        image,
        intr,
        board_spec=CharucoBoardSpec(dictionary="DICT_5X5_100"),
        quality_config=CharucoQualityConfig(min_corners=1, min_markers=1),
        detector_config=CharucoDetectorConfig(CHARUCO_DETECTOR_PRESET_CONSERVATIVE),
    )

    metadata = result.to_metadata()
    assert metadata["detector_preset"] == CHARUCO_DETECTOR_PRESET_CONSERVATIVE
    assert metadata["detector_parameters"]["cornerRefinementMethod"] == "CORNER_REFINE_SUBPIX"
    assert result.candidates
    assert result.candidates[0].detector_preset == CHARUCO_DETECTOR_PRESET_CONSERVATIVE


def test_detector_preset_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported ChArUco detector preset"):
        CharucoDetectorConfig("aggressive")


def test_conservative_detector_constructor_fallback_keeps_error_metadata():
    class FakeAruco:
        class DetectorParameters:
            pass

        class CharucoParameters:
            pass

        class RefineParameters:
            pass

        @staticmethod
        def CharucoDetector(*args):
            if len(args) > 1:
                raise RuntimeError("overload mismatch")
            return "detector"

    class FakeCv2:
        aruco = FakeAruco

    detector, metadata = _create_charuco_detector(
        FakeCv2,
        object(),
        CharucoDetectorConfig(CHARUCO_DETECTOR_PRESET_CONSERVATIVE),
    )

    assert detector == "detector"
    assert metadata["_constructor_fallback"] == "CharucoDetector(board)"
    assert "overload mismatch" in metadata["_constructor_error"]


def test_synthetic_charuco_board_detects_with_opencv_aruco():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 8), 0.030, 0.022, dictionary)
    gray = board.generateImage((500, 800))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=250.0, cy=400.0, width=500, height=800)

    result = detect_charuco_pose(
        rgb,
        intr,
        board_spec=CharucoBoardSpec(dictionary="DICT_5X5_100"),
        quality_config=CharucoQualityConfig(max_reprojection_error_px=1.0),
    )

    assert result.ok is True
    assert result.selected_dictionary == "DICT_5X5_100"
    assert result.cam_in_ob is not None
    assert result.best_candidate is not None
    assert result.best_candidate.reprojection_error_px is not None
    assert result.best_candidate.reprojection_error_px < 1.0


def test_charuco_axis_overlay_draws_visible_colored_axes():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 8), 0.030, 0.022, dictionary)
    gray = board.generateImage((500, 800))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=250.0, cy=400.0, width=500, height=800)
    result = detect_charuco_pose(rgb, intr, board_spec=CharucoBoardSpec(dictionary="DICT_5X5_100"))

    overlay_bgr = draw_charuco_axes_overlay_bgr(rgb, intr, result)

    red_pixels = (overlay_bgr[:, :, 2] > 180) & (overlay_bgr[:, :, 1] < 120) & (overlay_bgr[:, :, 0] < 120)
    green_pixels = (overlay_bgr[:, :, 1] > 180) & (overlay_bgr[:, :, 2] < 120) & (overlay_bgr[:, :, 0] < 120)
    blue_pixels = (overlay_bgr[:, :, 0] > 180) & (overlay_bgr[:, :, 1] < 120) & (overlay_bgr[:, :, 2] < 120)

    assert int(red_pixels.sum()) > 50
    assert int(green_pixels.sum()) > 50
    assert int(blue_pixels.sum()) > 50


def test_synthetic_legacy_charuco_board_is_detected_by_default_auto_fallback():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 8), 0.030, 0.022, dictionary)
    if not hasattr(board, "setLegacyPattern"):
        pytest.skip("OpenCV build does not expose ChArUco legacy pattern")
    board.setLegacyPattern(True)
    gray = board.generateImage((500, 800))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=250.0, cy=400.0, width=500, height=800)

    result = detect_charuco_pose(
        rgb,
        intr,
        board_spec=CharucoBoardSpec(dictionary="auto"),
        quality_config=CharucoQualityConfig(max_reprojection_error_px=1.0),
    )

    assert result.ok is True
    assert result.legacy_pattern is True
    assert result.best_candidate is not None
    assert result.best_candidate.legacy_pattern is True


def test_calibio_style_5x8_label_is_detected_with_swapped_opencv_board_size():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        pytest.skip("OpenCV aruco module unavailable")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((8, 5), 0.030, 0.022, dictionary)
    gray = board.generateImage((800, 500))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=400.0, cy=250.0, width=800, height=500)

    result = detect_charuco_pose(
        rgb,
        intr,
        board_spec=CharucoBoardSpec(squares_x=5, squares_y=8, dictionary="auto"),
        quality_config=CharucoQualityConfig(max_reprojection_error_px=1.0),
    )

    assert result.ok is True
    assert result.board_spec.squares_x == 8
    assert result.board_spec.squares_y == 5
    assert result.best_candidate is not None
    assert result.best_candidate.squares_x == 8
    assert result.best_candidate.squares_y == 5
