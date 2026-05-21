"""Offline processing from raw RGB-D recordings into FoundationPose references."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import numpy as np

from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics

from .charuco_reference import (
    BoardObjectTransform,
    CharucoBoardSpec,
    CharucoPoseResult,
    CharucoQualityConfig,
    detect_charuco_pose,
    record_charuco_pose_provenance,
)
from .mask_provider import MaskProvider, MaskResult
from .profile_manifest import mark_assets_stale
from .profile_schema import ObjectProfile, ProfileStatus, utc_now_iso
from .reference_dataset import count_reference_frames, save_reference_frame
from .reference_recording import RecordedFrameRecord, list_recording_sessions, load_frame_records


READINESS_READY = "ready"
READINESS_NEED_MORE_RECORDING = "need_more_recording"
READINESS_NO_RECORDINGS = "no_recordings"
REPORT_FILENAME = "reference_processing_latest.json"


@dataclass(frozen=True)
class ReferenceProcessingConfig:
    required_keyframes: int = 16
    max_keyframes: int = 32
    min_mask_area_fraction: float = 0.0005
    min_valid_depth_ratio: float = 0.05
    min_depth_m: float = 0.05
    max_depth_m: float = 3.0
    publish: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_keyframes": int(self.required_keyframes),
            "max_keyframes": int(self.max_keyframes),
            "min_mask_area_fraction": float(self.min_mask_area_fraction),
            "min_valid_depth_ratio": float(self.min_valid_depth_ratio),
            "min_depth_m": float(self.min_depth_m),
            "max_depth_m": float(self.max_depth_m),
            "publish": bool(self.publish),
        }


@dataclass(frozen=True)
class RecordedCandidate:
    session_id: str
    frame_index: int
    session_dir: Path
    frame_record: RecordedFrameRecord
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics

    @property
    def candidate_id(self) -> str:
        return f"{self.session_id}:{self.frame_index:06d}"


@dataclass
class EvaluatedCandidate:
    candidate: RecordedCandidate
    accepted: bool
    reasons: list[str]
    charuco_pose: CharucoPoseResult | None = None
    mask: MaskResult | None = None
    depth_stats: dict[str, float | int] = field(default_factory=dict)
    score: float = 0.0
    view_yaw_deg: float | None = None
    view_bin: int | None = None
    selected_index: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "session_id": self.candidate.session_id,
            "frame_index": self.candidate.frame_index,
            "accepted": bool(self.accepted),
            "selected_index": self.selected_index,
            "score": float(self.score),
            "view_yaw_deg": self.view_yaw_deg,
            "view_bin": self.view_bin,
            "reasons": list(self.reasons),
            "charuco_pose": self.charuco_pose.to_metadata() if self.charuco_pose is not None else None,
            "mask_source": self.mask.source if self.mask is not None else None,
            "mask_metadata": self.mask.metadata if self.mask is not None else None,
            "depth_stats": dict(self.depth_stats),
        }


@dataclass(frozen=True)
class ReferenceProcessingReport:
    object_name: str
    run_id: str
    readiness: str
    accepted: int
    rejected: int
    processed_candidates: int
    required_keyframes: int
    force_build_allowed: bool
    published: bool
    recording_sessions: list[dict[str, Any]]
    thresholds: dict[str, Any]
    records: list[dict[str, Any]]
    output_reference_count: int
    report_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.readiness == READINESS_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object_name,
            "run_id": self.run_id,
            "readiness": self.readiness,
            "accepted": int(self.accepted),
            "rejected": int(self.rejected),
            "processed_candidates": int(self.processed_candidates),
            "required_keyframes": int(self.required_keyframes),
            "force_build_allowed": bool(self.force_build_allowed),
            "published": bool(self.published),
            "recording_sessions": list(self.recording_sessions),
            "thresholds": dict(self.thresholds),
            "records": list(self.records),
            "output_reference_count": int(self.output_reference_count),
            "report_path": self.report_path,
        }


PoseDetector = Callable[..., CharucoPoseResult]


def process_recorded_references(
    profile: ObjectProfile,
    *,
    mask_provider: MaskProvider,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    config: ReferenceProcessingConfig | None = None,
    pose_detector: PoseDetector = detect_charuco_pose,
) -> ReferenceProcessingReport:
    config = config or ReferenceProcessingConfig()
    run_id = _new_processing_run_id()
    try:
        candidates = load_recorded_candidates(profile)
        sessions = [session for session in list_recording_sessions(profile)]
        evaluated = [
            evaluate_candidate(
                candidate,
                mask_provider=mask_provider,
                board_spec=board_spec,
                quality_config=quality_config,
                board_object=board_object,
                config=config,
                pose_detector=pose_detector,
            )
            for candidate in candidates
        ]
        selected = select_view_diverse_candidates(evaluated, config=config)
        for index, item in enumerate(selected):
            item.selected_index = index
        readiness = _readiness_for_selection(selected, candidates, config)
        force_build_allowed = readiness == READINESS_NEED_MORE_RECORDING and bool(selected)
        published = False
        if config.publish and selected:
            publish_selected_references(
                profile,
                selected,
                board_object=board_object,
                run_id=run_id,
            )
            published = True
        report = _build_report(
            profile,
            run_id=run_id,
            readiness=readiness,
            selected=selected,
            evaluated=evaluated,
            sessions=sessions,
            config=config,
            force_build_allowed=force_build_allowed,
            published=published,
        )
        return write_processing_report(profile, report)
    finally:
        release = getattr(mask_provider, "release", None)
        if callable(release):
            release()


def evaluate_recorded_references(
    profile: ObjectProfile,
    *,
    mask_provider: MaskProvider,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    config: ReferenceProcessingConfig | None = None,
    pose_detector: PoseDetector = detect_charuco_pose,
) -> ReferenceProcessingReport:
    config = config or ReferenceProcessingConfig(publish=False)
    if config.publish:
        config = ReferenceProcessingConfig(
            required_keyframes=config.required_keyframes,
            max_keyframes=config.max_keyframes,
            min_mask_area_fraction=config.min_mask_area_fraction,
            min_valid_depth_ratio=config.min_valid_depth_ratio,
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
            publish=False,
        )
    return process_recorded_references(
        profile,
        mask_provider=mask_provider,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
        config=config,
        pose_detector=pose_detector,
    )


def load_recorded_candidates(profile: ObjectProfile) -> list[RecordedCandidate]:
    candidates: list[RecordedCandidate] = []
    cv2 = _require_cv2()
    for session in list_recording_sessions(profile):
        for record in load_frame_records(session.session_dir):
            rgb_path = session.session_dir / record.rgb_path
            depth_path = session.session_dir / record.depth_path
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read recorded RGB frame: {rgb_path}")
            depth = np.load(depth_path).astype(np.float32)
            intrinsics = CameraIntrinsics.from_mapping(record.intrinsics)
            candidates.append(
                RecordedCandidate(
                    session_id=record.session_id,
                    frame_index=record.index,
                    session_dir=session.session_dir,
                    frame_record=record,
                    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    depth_m=depth,
                    intrinsics=intrinsics,
                )
            )
    return sorted(candidates, key=lambda item: (item.session_id, item.frame_index))


def evaluate_candidate(
    candidate: RecordedCandidate,
    *,
    mask_provider: MaskProvider,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    config: ReferenceProcessingConfig,
    pose_detector: PoseDetector = detect_charuco_pose,
) -> EvaluatedCandidate:
    start = time.perf_counter()
    try:
        pose = pose_detector(
            candidate.rgb,
            candidate.intrinsics,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
        )
    except Exception as exc:
        return EvaluatedCandidate(
            candidate=candidate,
            accepted=False,
            reasons=[f"charuco rejected: {exc}"],
            depth_stats={"charuco_pose_ms": _elapsed_ms(start)},
        )
    pose_ms = _elapsed_ms(start)
    if not pose.ok or pose.cam_in_ob is None:
        return EvaluatedCandidate(
            candidate=candidate,
            accepted=False,
            reasons=["charuco rejected: " + "; ".join(pose.reject_reasons or ["pose unavailable"])],
            charuco_pose=pose,
            depth_stats={"charuco_pose_ms": pose_ms},
        )
    start = time.perf_counter()
    try:
        mask = mask_provider.get_mask(
            candidate.rgb,
            depth_m=candidate.depth_m,
            object_name=None,
        )
    except Exception as exc:
        return EvaluatedCandidate(
            candidate=candidate,
            accepted=False,
            reasons=[f"mask rejected: {exc}"],
            charuco_pose=pose,
            depth_stats={"charuco_pose_ms": pose_ms, "segmentation_ms": _elapsed_ms(start)},
        )
    segmentation_ms = _elapsed_ms(start)
    depth_stats = compute_mask_depth_stats(
        candidate.depth_m,
        mask.mask,
        min_depth_m=config.min_depth_m,
        max_depth_m=config.max_depth_m,
    )
    depth_stats["charuco_pose_ms"] = pose_ms
    depth_stats["segmentation_ms"] = segmentation_ms
    reasons = _candidate_quality_reasons(
        mask.mask,
        image_shape=candidate.rgb.shape[:2],
        depth_stats=depth_stats,
        config=config,
    )
    score = _candidate_score(pose, depth_stats)
    view_yaw = _view_yaw_deg(pose)
    view_bin = _view_bin(view_yaw, bins=max(config.required_keyframes, 1))
    return EvaluatedCandidate(
        candidate=candidate,
        accepted=not reasons,
        reasons=reasons,
        charuco_pose=pose,
        mask=mask,
        depth_stats=depth_stats,
        score=score,
        view_yaw_deg=view_yaw,
        view_bin=view_bin,
    )


def compute_mask_depth_stats(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, float | int]:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    mask_pixels = int(np.count_nonzero(mask_bool))
    if mask_pixels == 0:
        return {
            "mask_pixels": 0,
            "valid_depth_pixels": 0,
            "valid_depth_ratio": 0.0,
            "median_depth_m": 0.0,
        }
    valid = mask_bool & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    valid_values = depth[valid]
    valid_count = int(valid_values.size)
    return {
        "mask_pixels": mask_pixels,
        "valid_depth_pixels": valid_count,
        "valid_depth_ratio": float(valid_count / max(mask_pixels, 1)),
        "median_depth_m": float(np.median(valid_values)) if valid_count else 0.0,
    }


def select_view_diverse_candidates(
    evaluated: list[EvaluatedCandidate],
    *,
    config: ReferenceProcessingConfig,
) -> list[EvaluatedCandidate]:
    accepted = [item for item in evaluated if item.accepted]
    if not accepted:
        return []
    accepted.sort(key=lambda item: (-item.score, item.candidate.session_id, item.candidate.frame_index))
    selected: list[EvaluatedCandidate] = []
    used_ids: set[str] = set()
    bins: dict[int, list[EvaluatedCandidate]] = {}
    for item in accepted:
        bins.setdefault(int(item.view_bin or 0), []).append(item)
    for bin_id in sorted(bins):
        if len(selected) >= config.max_keyframes:
            break
        item = bins[bin_id][0]
        selected.append(item)
        used_ids.add(item.candidate.candidate_id)
    if len(selected) < min(config.required_keyframes, config.max_keyframes):
        for item in accepted:
            if len(selected) >= config.max_keyframes:
                break
            if item.candidate.candidate_id in used_ids:
                continue
            selected.append(item)
            used_ids.add(item.candidate.candidate_id)
    return sorted(selected, key=lambda item: (item.candidate.session_id, item.candidate.frame_index))


def publish_selected_references(
    profile: ObjectProfile,
    selected: list[EvaluatedCandidate],
    *,
    board_object: BoardObjectTransform,
    run_id: str,
) -> None:
    if not selected:
        return
    staging_root = profile.root / f".refs_staging_{run_id}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_profile = ObjectProfile(
        name=profile.name,
        root=staging_root,
        prompt=profile.prompt,
    )
    for output_index, item in enumerate(selected):
        if item.mask is None or item.charuco_pose is None or item.charuco_pose.cam_in_ob is None:
            raise ValueError("selected candidates must have mask and ChArUco pose")
        save_reference_frame(
            staging_profile,
            output_index,
            rgb=item.candidate.rgb,
            depth_m=item.candidate.depth_m,
            mask=item.mask.mask,
            intrinsics=item.candidate.intrinsics,
            cam_in_ob=item.charuco_pose.cam_in_ob,
            metadata={
                "capture_mode": "recording_processing",
                "processing_run_id": run_id,
                "raw_candidate_id": item.candidate.candidate_id,
                "raw_session_id": item.candidate.session_id,
                "raw_frame_index": item.candidate.frame_index,
                "mask_source": item.mask.source,
                "mask_metadata": item.mask.metadata,
                "charuco_pose": item.charuco_pose.to_metadata(),
                "depth_stats": item.depth_stats,
                "score": item.score,
                "view_yaw_deg": item.view_yaw_deg,
                "view_bin": item.view_bin,
            },
        )
    backup_dir = profile.root / f"refs_backup_{run_id}"
    old_refs = profile.refs_dir
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if old_refs.exists():
            os.replace(old_refs, backup_dir)
        os.replace(staging_profile.refs_dir, old_refs)
    except Exception:
        if not old_refs.exists() and backup_dir.exists():
            os.replace(backup_dir, old_refs)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    profile.reference_count = count_reference_frames(profile)
    profile.status = ProfileStatus.CAPTURED
    profile.metadata["reference_workflow"] = "recording_processing"
    profile.metadata["last_processing_run_id"] = run_id
    profile.touch()
    profile.save()
    record_charuco_pose_provenance(
        profile,
        [item.charuco_pose for item in selected if item.charuco_pose is not None],
        board_object=board_object,
        indices=list(range(len(selected))),
    )
    mark_assets_stale(profile, "recording processing published references")


def write_processing_report(profile: ObjectProfile, report: ReferenceProcessingReport) -> ReferenceProcessingReport:
    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    path = profile.logs_dir / REPORT_FILENAME
    payload = report.to_dict()
    payload["report_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile.metadata["reference_processing"] = payload
    profile.touch()
    profile.save()
    return ReferenceProcessingReport(
        object_name=report.object_name,
        run_id=report.run_id,
        readiness=report.readiness,
        accepted=report.accepted,
        rejected=report.rejected,
        processed_candidates=report.processed_candidates,
        required_keyframes=report.required_keyframes,
        force_build_allowed=report.force_build_allowed,
        published=report.published,
        recording_sessions=report.recording_sessions,
        thresholds=report.thresholds,
        records=report.records,
        output_reference_count=report.output_reference_count,
        report_path=str(path),
    )


def latest_processing_report(profile: ObjectProfile) -> dict[str, Any] | None:
    path = profile.logs_dir / REPORT_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    report = profile.metadata.get("reference_processing")
    return dict(report) if isinstance(report, dict) else None


def _build_report(
    profile: ObjectProfile,
    *,
    run_id: str,
    readiness: str,
    selected: list[EvaluatedCandidate],
    evaluated: list[EvaluatedCandidate],
    sessions,
    config: ReferenceProcessingConfig,
    force_build_allowed: bool,
    published: bool,
) -> ReferenceProcessingReport:
    selected_ids = {item.candidate.candidate_id for item in selected}
    records = []
    for item in evaluated:
        record = item.to_record()
        if item.accepted and item.candidate.candidate_id not in selected_ids:
            record["accepted"] = False
            record["reasons"] = ["accepted candidate not selected for publish window"]
        records.append(record)
    return ReferenceProcessingReport(
        object_name=profile.name,
        run_id=run_id,
        readiness=readiness,
        accepted=len(selected),
        rejected=len([item for item in evaluated if not item.accepted]),
        processed_candidates=len(evaluated),
        required_keyframes=config.required_keyframes,
        force_build_allowed=force_build_allowed,
        published=published,
        recording_sessions=[
            {
                "session_id": session.session_id,
                "session_dir": str(session.session_dir),
                "frame_count": int(session.frame_count),
                "metadata": dict(session.metadata),
            }
            for session in sessions
        ],
        thresholds=config.to_dict(),
        records=records,
        output_reference_count=count_reference_frames(profile),
    )


def _readiness_for_selection(
    selected: list[EvaluatedCandidate],
    candidates: list[RecordedCandidate],
    config: ReferenceProcessingConfig,
) -> str:
    if not candidates:
        return READINESS_NO_RECORDINGS
    if len(selected) >= config.required_keyframes:
        return READINESS_READY
    return READINESS_NEED_MORE_RECORDING


def _candidate_quality_reasons(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    depth_stats: dict[str, float | int],
    config: ReferenceProcessingConfig,
) -> list[str]:
    reasons: list[str] = []
    area_fraction = float(np.count_nonzero(mask) / max(int(image_shape[0]) * int(image_shape[1]), 1))
    if area_fraction < config.min_mask_area_fraction:
        reasons.append(
            f"mask area fraction {area_fraction:.6f} below minimum {config.min_mask_area_fraction:.6f}"
        )
    valid_ratio = float(depth_stats.get("valid_depth_ratio", 0.0))
    if valid_ratio < config.min_valid_depth_ratio:
        reasons.append(f"valid depth ratio {valid_ratio:.3f} below minimum {config.min_valid_depth_ratio:.3f}")
    return reasons


def _candidate_score(pose: CharucoPoseResult, depth_stats: dict[str, float | int]) -> float:
    best = pose.best_candidate
    reproj = float(best.reprojection_error_px if best and best.reprojection_error_px is not None else 999.0)
    coverage = float(best.image_coverage_fraction if best else 0.0)
    valid_ratio = float(depth_stats.get("valid_depth_ratio", 0.0))
    corners = float(best.corner_count if best else 0)
    return corners + coverage * 100.0 + valid_ratio * 10.0 - reproj


def _view_yaw_deg(pose: CharucoPoseResult) -> float | None:
    if pose.cam_in_ob is None:
        return None
    camera_in_object = np.asarray(pose.cam_in_ob, dtype=np.float64)
    vector = camera_in_object[:3, 3]
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(vector[0], vector[2])))


def _view_bin(view_yaw_deg: float | None, *, bins: int) -> int | None:
    if view_yaw_deg is None:
        return None
    width = 360.0 / max(int(bins), 1)
    return int(np.floor((view_yaw_deg + 180.0) / width)) % max(int(bins), 1)


def _new_processing_run_id() -> str:
    stamp = utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"process-{stamp}-{time.time_ns() % 1_000_000_000:09d}"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("OpenCV is required to process raw reference recordings.") from exc
    return cv2
