"""ChArUco-assisted reference-pose generation for model-free assets."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics

from .profile_schema import ObjectProfile, utc_now_iso
from .reference_pose import reference_indices, write_reference_poses


DICT_5X5_CANDIDATES = ("DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000")
POSE_SOURCE = "charuco_board_jig"
CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0 = "charuco_corner_id_0"
CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD = "opencv_board_origin"
DEFAULT_CHARUCO_ORIGIN_CONVENTION = CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0
CHARUCO_ORIGIN_CONVENTIONS = (
    CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0,
    CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
)
CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT = "opencv-default"
CHARUCO_DETECTOR_PRESET_CONSERVATIVE = "conservative-charuco"
CHARUCO_DETECTOR_PRESETS = (
    CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT,
    CHARUCO_DETECTOR_PRESET_CONSERVATIVE,
)
_CONSERVATIVE_CHARUCO_DETECTOR_PARAMETERS: dict[str, Any] = {
    "cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
    "cornerRefinementWinSize": 5,
    "cornerRefinementMaxIterations": 50,
    "cornerRefinementMinAccuracy": 0.01,
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 23,
    "adaptiveThreshWinSizeStep": 3,
    "adaptiveThreshConstant": 7,
    "polygonalApproxAccuracyRate": 0.01,
    "minDistanceToBorder": 3,
    "minMarkerPerimeterRate": 0.01,
    "perspectiveRemovePixelPerCell": 12,
}


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int = 5
    squares_y: int = 8
    square_length_m: float = 0.030
    marker_length_m: float = 0.022
    dictionary: str = "auto"
    legacy_pattern: bool = False

    def __post_init__(self) -> None:
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("ChArUco board needs at least 2x2 squares")
        if self.square_length_m <= 0.0:
            raise ValueError("square_length_m must be positive")
        if self.marker_length_m <= 0.0:
            raise ValueError("marker_length_m must be positive")
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")

    def candidate_dictionaries(self) -> tuple[str, ...]:
        dictionary = self.dictionary.strip().upper()
        if dictionary in {"AUTO", "DICT_5X5", "DICT_5X5_AUTO"}:
            return DICT_5X5_CANDIDATES
        return (dictionary,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": int(self.squares_x),
            "squares_y": int(self.squares_y),
            "square_length_m": float(self.square_length_m),
            "marker_length_m": float(self.marker_length_m),
            "dictionary": self.dictionary,
            "legacy_pattern": bool(self.legacy_pattern),
        }


@dataclass(frozen=True)
class CharucoQualityConfig:
    min_corners: int = 6
    min_markers: int = 2
    max_reprojection_error_px: float = 4.0
    min_image_coverage_fraction: float = 0.005

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_corners": int(self.min_corners),
            "min_markers": int(self.min_markers),
            "max_reprojection_error_px": float(self.max_reprojection_error_px),
            "min_image_coverage_fraction": float(self.min_image_coverage_fraction),
        }


@dataclass(frozen=True)
class CharucoDetectorConfig:
    preset: str = CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT

    def __post_init__(self) -> None:
        preset = str(self.preset).strip().lower()
        if preset not in CHARUCO_DETECTOR_PRESETS:
            raise ValueError(
                "unsupported ChArUco detector preset: "
                f"{self.preset}; expected one of {', '.join(CHARUCO_DETECTOR_PRESETS)}"
            )
        object.__setattr__(self, "preset", preset)

    def parameter_summary(self) -> dict[str, Any]:
        if self.preset == CHARUCO_DETECTOR_PRESET_CONSERVATIVE:
            return dict(_CONSERVATIVE_CHARUCO_DETECTOR_PARAMETERS)
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_preset": self.preset,
            "detector_parameters": self.parameter_summary(),
        }


@dataclass(frozen=True)
class BoardObjectTransform:
    board_T_object: np.ndarray
    source: str = "numeric_xyz_rpy"
    xyz_m: tuple[float, float, float] | None = None
    rpy_deg: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.board_T_object, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"board_T_object must have shape (4, 4), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("board_T_object must contain only finite values")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError("board_T_object last row must be [0, 0, 0, 1]")
        object.__setattr__(self, "board_T_object", matrix)

    @classmethod
    def identity(cls) -> "BoardObjectTransform":
        return cls(np.eye(4, dtype=np.float64), source="identity", xyz_m=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0))

    @classmethod
    def from_xyz_rpy_deg(
        cls,
        xyz_m: tuple[float, float, float],
        rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "BoardObjectTransform":
        matrix = np.eye(4, dtype=np.float64)
        roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
        matrix[:3, :3] = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
        matrix[:3, 3] = np.asarray(xyz_m, dtype=np.float64)
        return cls(
            matrix,
            source="numeric_xyz_rpy",
            xyz_m=tuple(float(v) for v in xyz_m),
            rpy_deg=tuple(float(v) for v in rpy_deg),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BoardObjectTransform":
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            matrix_data = data.get("board_T_object", data.get("matrix"))
            if matrix_data is None:
                raise ValueError(f"{path} must contain board_T_object or matrix")
            return cls(np.asarray(matrix_data, dtype=np.float64), source=str(path))
        return cls(np.loadtxt(path).reshape(4, 4), source=str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "board_T_object": self.board_T_object.tolist(),
            "xyz_m": list(self.xyz_m) if self.xyz_m is not None else None,
            "rpy_deg": list(self.rpy_deg) if self.rpy_deg is not None else None,
        }


@dataclass(frozen=True)
class DictionaryCandidateResult:
    dictionary: str
    ok: bool
    legacy_pattern: bool = False
    squares_x: int = 0
    squares_y: int = 0
    corner_count: int = 0
    marker_count: int = 0
    reprojection_error_px: float | None = None
    image_coverage_fraction: float = 0.0
    reject_reasons: list[str] = field(default_factory=list)
    camera_T_board: np.ndarray | None = None
    camera_T_object: np.ndarray | None = None
    cam_in_ob: np.ndarray | None = None
    distortion_policy: str = "unknown"
    detector_preset: str = CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT
    detector_parameters: dict[str, Any] = field(default_factory=dict)
    charuco_origin_convention: str = DEFAULT_CHARUCO_ORIGIN_CONVENTION
    charuco_origin_offset_board_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effective_board_T_object: np.ndarray | None = None

    def to_dict(self, *, include_matrices: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dictionary": self.dictionary,
            "ok": bool(self.ok),
            "legacy_pattern": bool(self.legacy_pattern),
            "squares_x": int(self.squares_x),
            "squares_y": int(self.squares_y),
            "corner_count": int(self.corner_count),
            "marker_count": int(self.marker_count),
            "reprojection_error_px": self.reprojection_error_px,
            "image_coverage_fraction": float(self.image_coverage_fraction),
            "reject_reasons": list(self.reject_reasons),
            "distortion_policy": self.distortion_policy,
            "detector_preset": self.detector_preset,
            "detector_parameters": dict(self.detector_parameters),
            "charuco_origin_convention": self.charuco_origin_convention,
            "charuco_origin_offset_board_m": list(self.charuco_origin_offset_board_m),
        }
        if include_matrices:
            data["camera_T_board"] = _matrix_to_list(self.camera_T_board)
            data["camera_T_object"] = _matrix_to_list(self.camera_T_object)
            data["cam_in_ob"] = _matrix_to_list(self.cam_in_ob)
            data["effective_board_T_object"] = _matrix_to_list(self.effective_board_T_object)
        return data


@dataclass(frozen=True)
class CharucoPoseResult:
    ok: bool
    selected_dictionary: str | None
    candidates: list[DictionaryCandidateResult]
    board_spec: CharucoBoardSpec
    quality_config: CharucoQualityConfig
    opencv_version: str
    board_coordinate_convention: str
    legacy_pattern: bool
    charuco_origin_convention: str = DEFAULT_CHARUCO_ORIGIN_CONVENTION
    charuco_origin_offset_board_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_T_board: np.ndarray | None = None
    camera_T_object: np.ndarray | None = None
    cam_in_ob: np.ndarray | None = None
    user_board_T_object: np.ndarray | None = None
    effective_board_T_object: np.ndarray | None = None
    reject_reasons: list[str] = field(default_factory=list)
    detector_preset: str = CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT
    detector_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def best_candidate(self) -> DictionaryCandidateResult | None:
        return choose_best_dictionary_result(self.candidates)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "selected_dictionary": self.selected_dictionary,
            "board_spec": self.board_spec.to_dict(),
            "quality_config": self.quality_config.to_dict(),
            "opencv_version": self.opencv_version,
            "board_coordinate_convention": self.board_coordinate_convention,
            "legacy_pattern": bool(self.legacy_pattern),
            "charuco_origin_convention": self.charuco_origin_convention,
            "charuco_origin_offset_board_m": list(self.charuco_origin_offset_board_m),
            "detector_preset": self.detector_preset,
            "detector_parameters": dict(self.detector_parameters),
            "camera_T_board": _matrix_to_list(self.camera_T_board),
            "camera_T_object": _matrix_to_list(self.camera_T_object),
            "cam_in_ob": _matrix_to_list(self.cam_in_ob),
            "user_board_T_object": _matrix_to_list(self.user_board_T_object),
            "effective_board_T_object": _matrix_to_list(self.effective_board_T_object),
            "reject_reasons": list(self.reject_reasons),
            "candidates": [candidate.to_dict(include_matrices=False) for candidate in self.candidates],
        }


def camera_T_object_from_board(camera_T_board: np.ndarray, board_T_object: np.ndarray) -> np.ndarray:
    camera_T_board = _require_transform(camera_T_board, "camera_T_board")
    board_T_object = _require_transform(board_T_object, "board_T_object")
    return camera_T_board @ board_T_object


def normalize_charuco_origin_convention(value: str | None = None) -> str:
    convention = str(value or DEFAULT_CHARUCO_ORIGIN_CONVENTION).strip().lower().replace("-", "_")
    aliases = {
        "corner_id_0": CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0,
        "charuco_corner0": CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0,
        "charuco_corner_id0": CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0,
        "charuco_corner_id_0": CHARUCO_ORIGIN_CONVENTION_CORNER_ID_0,
        "opencv": CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
        "opencv_board": CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
        "opencv_board_origin": CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD,
    }
    convention = aliases.get(convention, convention)
    if convention not in CHARUCO_ORIGIN_CONVENTIONS:
        raise ValueError(
            "unsupported ChArUco origin convention: "
            f"{value}; expected one of {', '.join(CHARUCO_ORIGIN_CONVENTIONS)}"
        )
    return convention


def charuco_origin_offset_board_m(
    board_spec: CharucoBoardSpec,
    charuco_origin_convention: str | None = None,
) -> tuple[float, float, float]:
    convention = normalize_charuco_origin_convention(charuco_origin_convention)
    if convention == CHARUCO_ORIGIN_CONVENTION_OPENCV_BOARD:
        return (0.0, 0.0, 0.0)
    square = float(board_spec.square_length_m)
    return (square, square, 0.0)


def effective_board_T_object(
    board_spec: CharucoBoardSpec,
    user_board_T_object: np.ndarray,
    *,
    charuco_origin_convention: str | None = None,
) -> np.ndarray:
    user_transform = _require_transform(user_board_T_object, "user_board_T_object")
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = np.asarray(
        charuco_origin_offset_board_m(board_spec, charuco_origin_convention),
        dtype=np.float64,
    )
    return offset @ user_transform


def cam_in_ob_from_camera_T_object(camera_T_object: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_require_transform(camera_T_object, "camera_T_object"))


def choose_best_dictionary_result(
    candidates: list[DictionaryCandidateResult],
) -> DictionaryCandidateResult | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            not item.ok,
            -int(item.corner_count),
            float("inf") if item.reprojection_error_px is None else float(item.reprojection_error_px),
            -float(item.image_coverage_fraction),
            item.dictionary,
        ),
    )[0]


def detect_charuco_pose(
    image_rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    board_spec: CharucoBoardSpec | None = None,
    quality_config: CharucoQualityConfig | None = None,
    board_object: BoardObjectTransform | None = None,
    detector_config: CharucoDetectorConfig | None = None,
    detector_preset: str | None = None,
    charuco_origin_convention: str | None = None,
) -> CharucoPoseResult:
    cv2 = _require_cv2()
    board_spec = board_spec or CharucoBoardSpec()
    quality_config = quality_config or CharucoQualityConfig()
    board_object = board_object or BoardObjectTransform.identity()
    detector_config = _resolve_detector_config(detector_config, detector_preset)
    origin_convention = normalize_charuco_origin_convention(charuco_origin_convention)
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {image.shape}")
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    candidates = []
    for candidate_board_spec in _candidate_board_specs(board_spec):
        for legacy_pattern in _candidate_legacy_patterns(candidate_board_spec):
            for dictionary_name in candidate_board_spec.candidate_dictionaries():
                candidates.append(
                    _detect_candidate(
                        cv2,
                        gray,
                        intrinsics,
                        board_spec=candidate_board_spec,
                        quality_config=quality_config,
                        dictionary_name=dictionary_name,
                        legacy_pattern=legacy_pattern,
                        board_T_object=board_object.board_T_object,
                        detector_config=detector_config,
                        charuco_origin_convention=origin_convention,
                    )
                )
    best = choose_best_dictionary_result(candidates)
    ok = bool(best and best.ok)
    selected_board_spec = _selected_board_spec(board_spec, best)
    user_board_T_object = board_object.board_T_object
    effective_transform = effective_board_T_object(
        selected_board_spec,
        user_board_T_object,
        charuco_origin_convention=origin_convention,
    )
    return CharucoPoseResult(
        ok=ok,
        selected_dictionary=best.dictionary if best and best.ok else None,
        candidates=candidates,
        board_spec=selected_board_spec,
        quality_config=quality_config,
        opencv_version=str(cv2.__version__),
        board_coordinate_convention="opencv_charuco_board",
        legacy_pattern=bool(best.legacy_pattern) if best else bool(board_spec.legacy_pattern),
        charuco_origin_convention=origin_convention,
        charuco_origin_offset_board_m=charuco_origin_offset_board_m(selected_board_spec, origin_convention),
        camera_T_board=best.camera_T_board if best and best.ok else None,
        camera_T_object=best.camera_T_object if best and best.ok else None,
        cam_in_ob=best.cam_in_ob if best and best.ok else None,
        user_board_T_object=user_board_T_object,
        effective_board_T_object=best.effective_board_T_object if best and best.ok else effective_transform,
        reject_reasons=[] if ok else (best.reject_reasons if best else ["no dictionary candidates"]),
        detector_preset=detector_config.preset,
        detector_parameters=detector_config.parameter_summary(),
    )


def generate_charuco_reference_poses(
    profile: ObjectProfile,
    *,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    detector_config: CharucoDetectorConfig | None = None,
    charuco_origin_convention: str | None = None,
) -> list[CharucoPoseResult]:
    intrinsics = load_reference_intrinsics(profile)
    indices = reference_indices(profile)
    results: list[CharucoPoseResult] = []
    for index in indices:
        image = _load_reference_rgb(profile, index)
        result = detect_charuco_pose(
            image,
            intrinsics,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
            detector_config=detector_config,
            charuco_origin_convention=charuco_origin_convention,
        )
        results.append(result)
    failures = [(index, result.reject_reasons) for index, result in zip(indices, results) if not result.ok]
    if failures:
        details = "; ".join(f"{index:06d}: {', '.join(reasons)}" for index, reasons in failures)
        raise ValueError(f"ChArUco pose generation rejected frame(s): {details}")
    return results


def write_charuco_reference_poses(
    profile: ObjectProfile,
    results: list[CharucoPoseResult],
    *,
    board_object: BoardObjectTransform,
) -> None:
    cam_in_obs = []
    for result in results:
        if not result.ok or result.cam_in_ob is None:
            raise ValueError("all ChArUco pose results must be valid before writing")
        cam_in_obs.append(result.cam_in_ob)
    indices = reference_indices(profile)
    pose_provenance = build_charuco_pose_provenance(
        profile,
        results,
        board_object=board_object,
        indices=indices,
    )
    write_reference_poses(profile, cam_in_obs, pose_source=POSE_SOURCE, pose_provenance=pose_provenance)
    for index, result in zip(indices, results):
        _merge_reference_metadata(profile, index, {"charuco_pose": result.to_metadata()})


def record_charuco_pose_provenance(
    profile: ObjectProfile,
    results: list[CharucoPoseResult],
    *,
    board_object: BoardObjectTransform,
    indices: list[int],
) -> None:
    profile.metadata["pose_source"] = POSE_SOURCE
    profile.metadata["pose_provenance"] = build_charuco_pose_provenance(
        profile,
        results,
        board_object=board_object,
        indices=indices,
    )
    profile.touch()
    profile.save()
    from .profile_manifest import mark_assets_stale

    mark_assets_stale(profile, "ChArUco pose provenance updated")


def build_charuco_pose_provenance(
    profile: ObjectProfile,
    results: list[CharucoPoseResult],
    *,
    board_object: BoardObjectTransform,
    indices: list[int],
) -> dict[str, Any]:
    first = results[0] if results else None
    return {
        "approximate": False,
        "pose_source": POSE_SOURCE,
        "board_spec": first.board_spec.to_dict() if first else {},
        "quality_config": first.quality_config.to_dict() if first else {},
        "board_object_transform": board_object.to_dict(),
        "selected_dictionaries": [result.selected_dictionary for result in results],
        "opencv_version": first.opencv_version if first else None,
        "board_coordinate_convention": first.board_coordinate_convention if first else None,
        "charuco_origin_convention": first.charuco_origin_convention if first else None,
        "charuco_origin_offset_board_m": list(first.charuco_origin_offset_board_m) if first else None,
        "legacy_pattern": first.legacy_pattern if first else None,
        "detector_preset": first.detector_preset if first else None,
        "detector_parameters": dict(first.detector_parameters) if first else {},
        "distortion_policy": _summarize_distortion_policy(results),
        "frame_quality": [
            {
                "index": index,
                "selected_dictionary": result.selected_dictionary,
                "detector_preset": result.detector_preset,
                "corner_count": result.best_candidate.corner_count if result.best_candidate else 0,
                "marker_count": result.best_candidate.marker_count if result.best_candidate else 0,
                "reprojection_error_px": result.best_candidate.reprojection_error_px if result.best_candidate else None,
                "image_coverage_fraction": result.best_candidate.image_coverage_fraction if result.best_candidate else 0.0,
            }
            for index, result in zip(indices, results)
        ],
        "updated_at": utc_now_iso(),
    }


def load_reference_intrinsics(profile: ObjectProfile) -> CameraIntrinsics:
    intrinsics_json = profile.refs_dir / "intrinsics.json"
    if intrinsics_json.exists():
        return CameraIntrinsics.from_mapping(json.loads(intrinsics_json.read_text(encoding="utf-8")))
    k_path = profile.refs_dir / "K.txt"
    if not k_path.exists():
        raise FileNotFoundError(f"reference intrinsics not found: {intrinsics_json} or {k_path}")
    matrix = np.loadtxt(k_path).reshape(3, 3)
    return CameraIntrinsics.from_mapping({"camera_matrix": matrix.tolist()})


def provenance_summary(profile: ObjectProfile) -> dict[str, Any] | None:
    provenance = profile.metadata.get("pose_provenance")
    if not isinstance(provenance, dict) or provenance.get("pose_source") != POSE_SOURCE:
        return None
    return {
        "pose_source": POSE_SOURCE,
        "board_spec": provenance.get("board_spec"),
        "selected_dictionaries": provenance.get("selected_dictionaries"),
        "opencv_version": provenance.get("opencv_version"),
        "board_coordinate_convention": provenance.get("board_coordinate_convention"),
        "charuco_origin_convention": provenance.get("charuco_origin_convention"),
        "charuco_origin_offset_board_m": provenance.get("charuco_origin_offset_board_m"),
        "legacy_pattern": provenance.get("legacy_pattern"),
        "detector_preset": provenance.get("detector_preset"),
        "detector_parameters": provenance.get("detector_parameters"),
        "distortion_policy": provenance.get("distortion_policy"),
        "frame_quality": provenance.get("frame_quality"),
    }


def draw_charuco_axes_overlay_bgr(
    image_rgb: np.ndarray,
    intrinsics: CameraIntrinsics,
    result: CharucoPoseResult,
    *,
    axis_length_m: float = 0.05,
) -> np.ndarray:
    """Draw the raw board outline and corrected object-origin +X/+Y/+Z axes."""

    if not result.ok or result.camera_T_board is None or result.camera_T_object is None:
        raise ValueError("valid ChArUco board and object poses are required to draw axes")
    cv2 = _require_cv2()
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    camera_T_board = np.asarray(result.camera_T_board, dtype=np.float64)
    camera_T_object = np.asarray(result.camera_T_object, dtype=np.float64)
    board_rvec, _ = cv2.Rodrigues(camera_T_board[:3, :3])
    board_tvec = camera_T_board[:3, 3].reshape(3, 1)
    object_rvec, _ = cv2.Rodrigues(camera_T_object[:3, :3])
    object_tvec = camera_T_object[:3, 3].reshape(3, 1)
    dist_coeffs = intrinsics.as_distortion_coeffs(fallback_zeros=True)
    length = float(axis_length_m)
    board_width_m = float(result.board_spec.squares_x) * float(result.board_spec.square_length_m)
    board_height_m = float(result.board_spec.squares_y) * float(result.board_spec.square_length_m)

    board_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float32,
    )
    object_axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.0, 0.0, length],
        ],
        dtype=np.float32,
    )
    board_projected, _ = cv2.projectPoints(
        board_points,
        board_rvec,
        board_tvec,
        intrinsics.as_matrix(),
        dist_coeffs,
    )
    object_projected, _ = cv2.projectPoints(
        object_axis_points,
        object_rvec,
        object_tvec,
        intrinsics.as_matrix(),
        dist_coeffs,
    )
    board_origin, board_x, board_xy, board_y = board_projected.reshape(-1, 2).astype(int)
    origin, x_end, y_end, z_end = object_projected.reshape(-1, 2).astype(int)
    _draw_polyline_with_shadow(
        image_bgr,
        [board_origin, board_x, board_xy, board_y, board_origin],
        (255, 255, 255),
        thickness=2,
    )
    _draw_axis_arrow(image_bgr, origin, x_end, (0, 0, 255), "+X")
    _draw_axis_arrow(image_bgr, origin, y_end, (0, 255, 0), "+Y")
    _draw_axis_arrow(image_bgr, origin, z_end, (255, 0, 0), "+Z")
    _draw_origin_marker(image_bgr, origin)
    return image_bgr


def draw_charuco_detection_debug_bgr(image_rgb: np.ndarray, result: CharucoPoseResult) -> np.ndarray:
    """Draw detected ArUco markers plus the ChArUco rejection reason."""

    cv2 = _require_cv2()
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    candidate = result.best_candidate
    dictionary_name = candidate.dictionary if candidate is not None else result.board_spec.candidate_dictionaries()[0]
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        detector = cv2.aruco.ArucoDetector(dictionary)
        gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_bgr, marker_corners, marker_ids)
    except Exception:
        pass

    return image_bgr


def _draw_axis_arrow(image_bgr: np.ndarray, start: np.ndarray, end: np.ndarray, color_bgr: tuple[int, int, int], label: str) -> None:
    cv2 = _require_cv2()
    start_xy = tuple(int(v) for v in start)
    end_xy = tuple(int(v) for v in end)
    cv2.arrowedLine(image_bgr, start_xy, end_xy, (0, 0, 0), 8, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(image_bgr, start_xy, end_xy, color_bgr, 4, cv2.LINE_AA, tipLength=0.18)
    label_xy = _label_position(start, end)
    cv2.putText(image_bgr, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image_bgr, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_bgr, 3, cv2.LINE_AA)


def _draw_origin_marker(image_bgr: np.ndarray, origin: np.ndarray) -> None:
    cv2 = _require_cv2()
    origin_xy = tuple(int(v) for v in origin)
    cv2.circle(image_bgr, origin_xy, 11, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image_bgr, origin_xy, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image_bgr, origin_xy, 8, (0, 0, 255), 2, cv2.LINE_AA)
    text_xy = (origin_xy[0] + 12, origin_xy[1] - 12)
    cv2.putText(image_bgr, "O", text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(image_bgr, "O", text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)


def _draw_polyline_with_shadow(
    image_bgr: np.ndarray,
    points: list[np.ndarray],
    color_bgr: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    cv2 = _require_cv2()
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_bgr, [pts], isClosed=True, color=(0, 0, 0), thickness=thickness + 4, lineType=cv2.LINE_AA)
    cv2.polylines(image_bgr, [pts], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


def _label_position(start: np.ndarray, end: np.ndarray) -> tuple[int, int]:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0:
        vector = np.array([24.0, -24.0], dtype=np.float64)
    else:
        vector = vector / norm * 16.0
    label = np.asarray(end, dtype=np.float64) + vector + np.array([4.0, -4.0], dtype=np.float64)
    return tuple(int(round(v)) for v in label)


def _detect_candidate(
    cv2,
    gray: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    dictionary_name: str,
    legacy_pattern: bool,
    board_T_object: np.ndarray,
    detector_config: CharucoDetectorConfig,
    charuco_origin_convention: str,
) -> DictionaryCandidateResult:
    reject_reasons: list[str] = []
    detector_parameters = detector_config.parameter_summary()
    origin_convention = normalize_charuco_origin_convention(charuco_origin_convention)
    origin_offset = charuco_origin_offset_board_m(board_spec, origin_convention)
    effective_transform = effective_board_T_object(
        board_spec,
        board_T_object,
        charuco_origin_convention=origin_convention,
    )

    def candidate_result(**overrides) -> DictionaryCandidateResult:
        data: dict[str, Any] = {
            "dictionary": dictionary_name,
            "ok": False,
            "legacy_pattern": legacy_pattern,
            "squares_x": board_spec.squares_x,
            "squares_y": board_spec.squares_y,
            "detector_preset": detector_config.preset,
            "detector_parameters": detector_parameters,
            "charuco_origin_convention": origin_convention,
            "charuco_origin_offset_board_m": origin_offset,
        }
        data.update(overrides)
        return DictionaryCandidateResult(**data)

    try:
        board = _create_board(cv2, board_spec, dictionary_name, legacy_pattern=legacy_pattern)
    except Exception as exc:
        return candidate_result(reject_reasons=[str(exc)])

    try:
        detector, detector_metadata = _create_charuco_detector(cv2, board, detector_config)
        detector_parameters = detector_metadata
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    except Exception as exc:
        return candidate_result(reject_reasons=[f"ChArUco detection failed: {exc}"])

    corner_count = 0 if charuco_corners is None else int(len(charuco_corners))
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    if corner_count < quality_config.min_corners:
        reject_reasons.append(f"corner count {corner_count} below minimum {quality_config.min_corners}")
    if marker_count < quality_config.min_markers:
        reject_reasons.append(f"marker count {marker_count} below minimum {quality_config.min_markers}")
    if charuco_corners is None or charuco_ids is None or corner_count == 0:
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            reject_reasons=reject_reasons or ["no ChArUco corners detected"],
        )

    try:
        object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    except Exception as exc:
        reject_reasons.append(f"matchImagePoints failed: {exc}")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            reject_reasons=reject_reasons,
        )
    image_points_xy = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    coverage = _image_coverage_fraction(image_points_xy, image_shape=gray.shape[:2])
    if coverage < quality_config.min_image_coverage_fraction:
        reject_reasons.append(
            f"image coverage {coverage:.6f} below minimum {quality_config.min_image_coverage_fraction:.6f}"
        )
    if not _has_non_collinear_points(image_points_xy):
        reject_reasons.append("detected ChArUco corners are nearly collinear")

    dist_coeffs = intrinsics.as_distortion_coeffs(fallback_zeros=True)
    distortion_policy = "camera_coefficients" if intrinsics.distortion_coeffs is not None else "zero_unavailable"
    flags = getattr(cv2, "SOLVEPNP_IPPE", getattr(cv2, "SOLVEPNP_ITERATIVE", 0))
    try:
        pnp_ok, rvec, tvec = cv2.solvePnP(
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32),
            intrinsics.as_matrix(),
            dist_coeffs,
            flags=flags,
        )
    except Exception as exc:
        reject_reasons.append(f"solvePnP failed: {exc}")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            distortion_policy=distortion_policy,
        )
    if not pnp_ok:
        reject_reasons.append("solvePnP returned false")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            distortion_policy=distortion_policy,
        )
    try:
        rotation, _ = cv2.Rodrigues(rvec)
    except Exception as exc:
        reject_reasons.append(f"Rodrigues failed: {exc}")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            distortion_policy=distortion_policy,
        )
    camera_T_board = np.eye(4, dtype=np.float64)
    camera_T_board[:3, :3] = rotation
    camera_T_board[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(camera_T_board)):
        reject_reasons.append("camera_T_board contains non-finite values")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            camera_T_board=camera_T_board,
            distortion_policy=distortion_policy,
        )
    if camera_T_board[2, 3] <= 0.0:
        reject_reasons.append(f"board is not in front of camera: z={camera_T_board[2, 3]:.6f}")

    try:
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float32),
            rvec,
            tvec,
            intrinsics.as_matrix(),
            dist_coeffs,
        )
    except Exception as exc:
        reject_reasons.append(f"projectPoints failed: {exc}")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            camera_T_board=camera_T_board,
            distortion_policy=distortion_policy,
        )
    projected_xy = np.asarray(projected).reshape(-1, 2)
    if not np.all(np.isfinite(projected_xy)):
        reject_reasons.append("projected ChArUco corners contain non-finite values")
        return candidate_result(
            corner_count=corner_count,
            marker_count=marker_count,
            image_coverage_fraction=coverage,
            reject_reasons=reject_reasons,
            camera_T_board=camera_T_board,
            distortion_policy=distortion_policy,
        )
    reprojection_error = _rms_reprojection_error(image_points_xy, projected_xy)
    if reprojection_error > quality_config.max_reprojection_error_px:
        reject_reasons.append(
            f"reprojection error {reprojection_error:.3f}px above maximum {quality_config.max_reprojection_error_px:.3f}px"
        )

    camera_T_object = camera_T_object_from_board(camera_T_board, effective_transform)
    cam_in_ob = cam_in_ob_from_camera_T_object(camera_T_object)
    ok = not reject_reasons
    return candidate_result(
        ok=ok,
        corner_count=corner_count,
        marker_count=marker_count,
        reprojection_error_px=float(reprojection_error),
        image_coverage_fraction=coverage,
        reject_reasons=reject_reasons,
        camera_T_board=camera_T_board,
        camera_T_object=camera_T_object,
        cam_in_ob=cam_in_ob,
        effective_board_T_object=effective_transform,
        distortion_policy=distortion_policy,
    )


def _resolve_detector_config(
    detector_config: CharucoDetectorConfig | None,
    detector_preset: str | None,
) -> CharucoDetectorConfig:
    if detector_config is not None and detector_preset is not None:
        requested = CharucoDetectorConfig(detector_preset)
        if requested.preset != detector_config.preset:
            raise ValueError(
                "detector_config and detector_preset disagree: "
                f"{detector_config.preset} != {requested.preset}"
            )
    if detector_config is not None:
        return detector_config
    return CharucoDetectorConfig(detector_preset or CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT)


def _create_charuco_detector(cv2, board, detector_config: CharucoDetectorConfig):
    if detector_config.preset == CHARUCO_DETECTOR_PRESET_OPENCV_DEFAULT:
        return cv2.aruco.CharucoDetector(board), detector_config.parameter_summary()

    detector_parameters = cv2.aruco.DetectorParameters()
    applied: dict[str, Any] = {}
    unsupported: list[str] = []
    for name, value in detector_config.parameter_summary().items():
        if not hasattr(detector_parameters, name):
            unsupported.append(name)
            continue
        applied[name] = _set_detector_parameter(cv2, detector_parameters, name, value)

    metadata: dict[str, Any] = dict(applied)
    if unsupported:
        metadata["_unsupported_parameters"] = unsupported

    charuco_parameters = cv2.aruco.CharucoParameters() if hasattr(cv2.aruco, "CharucoParameters") else None
    refine_parameters = cv2.aruco.RefineParameters() if hasattr(cv2.aruco, "RefineParameters") else None
    try:
        if charuco_parameters is not None and refine_parameters is not None:
            detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters, refine_parameters)
        elif charuco_parameters is not None:
            detector = cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)
        else:
            detector = cv2.aruco.CharucoDetector(board)
            metadata["_constructor_fallback"] = "CharucoDetector(board)"
    except Exception as exc:
        detector = cv2.aruco.CharucoDetector(board)
        metadata["_constructor_fallback"] = "CharucoDetector(board)"
        metadata["_constructor_error"] = str(exc)
    return detector, metadata


def _set_detector_parameter(cv2, detector_parameters, name: str, value: Any) -> Any:
    if name == "cornerRefinementMethod" and value == "CORNER_REFINE_SUBPIX":
        actual = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    else:
        actual = value
    setattr(detector_parameters, name, actual)
    return value


def _candidate_legacy_patterns(spec: CharucoBoardSpec) -> tuple[bool, ...]:
    if spec.legacy_pattern:
        return (True,)
    return (False, True)


def _candidate_board_specs(spec: CharucoBoardSpec) -> tuple[CharucoBoardSpec, ...]:
    if spec.squares_x == spec.squares_y:
        return (spec,)
    swapped = CharucoBoardSpec(
        squares_x=spec.squares_y,
        squares_y=spec.squares_x,
        square_length_m=spec.square_length_m,
        marker_length_m=spec.marker_length_m,
        dictionary=spec.dictionary,
        legacy_pattern=spec.legacy_pattern,
    )
    return (spec, swapped)


def _selected_board_spec(spec: CharucoBoardSpec, candidate: DictionaryCandidateResult | None) -> CharucoBoardSpec:
    if candidate is None:
        return spec
    return CharucoBoardSpec(
        squares_x=int(candidate.squares_x or spec.squares_x),
        squares_y=int(candidate.squares_y or spec.squares_y),
        square_length_m=spec.square_length_m,
        marker_length_m=spec.marker_length_m,
        dictionary=spec.dictionary,
        legacy_pattern=bool(candidate.legacy_pattern),
    )


def _create_board(cv2, spec: CharucoBoardSpec, dictionary_name: str, *, legacy_pattern: bool | None = None):
    dictionary_attr = dictionary_name.strip().upper()
    if not hasattr(cv2.aruco, dictionary_attr):
        raise ValueError(f"unsupported ChArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_attr))
    board = cv2.aruco.CharucoBoard(
        (int(spec.squares_x), int(spec.squares_y)),
        float(spec.square_length_m),
        float(spec.marker_length_m),
        dictionary,
    )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(spec.legacy_pattern if legacy_pattern is None else legacy_pattern))
    return board


def _load_reference_rgb(profile: ObjectProfile, index: int) -> np.ndarray:
    cv2 = _require_cv2()
    path = profile.rgb_dir / f"{index:06d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read reference RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _merge_reference_metadata(profile: ObjectProfile, index: int, update: dict[str, Any]) -> None:
    path = profile.refs_dir / f"{index:06d}.json"
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data.update(update)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_distortion_policy(results: list[CharucoPoseResult]) -> dict[str, Any]:
    policies: dict[str, int] = {}
    for result in results:
        candidate = result.best_candidate
        if candidate is None:
            continue
        policies[candidate.distortion_policy] = policies.get(candidate.distortion_policy, 0) + 1
    return policies


def _image_coverage_fraction(points_xy: np.ndarray, *, image_shape: tuple[int, int]) -> float:
    if points_xy.size == 0:
        return 0.0
    min_xy = np.min(points_xy, axis=0)
    max_xy = np.max(points_xy, axis=0)
    bbox_area = max(float(max_xy[0] - min_xy[0]), 0.0) * max(float(max_xy[1] - min_xy[1]), 0.0)
    return float(bbox_area / max(int(image_shape[0]) * int(image_shape[1]), 1))


def _has_non_collinear_points(points_xy: np.ndarray) -> bool:
    if points_xy.shape[0] < 3:
        return False
    centered = points_xy - np.mean(points_xy, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return bool(singular_values.shape[0] >= 2 and singular_values[1] > 1e-3)


def _rms_reprojection_error(observed_xy: np.ndarray, projected_xy: np.ndarray) -> float:
    diff = np.asarray(observed_xy, dtype=np.float64) - np.asarray(projected_xy, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _require_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _matrix_to_list(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float64).tolist()


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV with cv2.aruco is required for ChArUco reference poses.") from exc
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("OpenCV aruco/CharucoDetector support is required for ChArUco reference poses.")
    return cv2
