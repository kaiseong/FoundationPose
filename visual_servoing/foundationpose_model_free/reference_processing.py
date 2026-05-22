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
    DictionaryCandidateResult,
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
PROCESSING_CACHE_DIRNAME = "processing_cache"
PROCESSING_CACHE_POINTER = "latest.json"
PROCESSING_CACHE_RECORDS = "records.json"
PROCESSING_CACHE_VERSION = 1


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
    processing_cache_path: str | None = None

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
            "processing_cache_path": self.processing_cache_path,
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
        cache_path = write_processing_cache(
            profile,
            evaluated,
            board_spec=board_spec,
            quality_config=quality_config,
            board_object=board_object,
            config=config,
            run_id=run_id,
            mask_provider=mask_provider,
        )
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
            processing_cache_path=str(cache_path),
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


def reselect_recorded_references(
    profile: ObjectProfile,
    *,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    config: ReferenceProcessingConfig | None = None,
    cache_dir: str | Path | None = None,
) -> ReferenceProcessingReport:
    """Republish references from the latest processing cache without rerunning SAM."""

    config = config or ReferenceProcessingConfig()
    run_id = _new_processing_run_id()
    cache_path, cache_payload = load_processing_cache(profile, cache_dir=cache_dir)
    _validate_processing_cache(
        cache_payload,
        board_spec=board_spec,
        quality_config=quality_config,
        board_object=board_object,
    )
    cached_records = list(cache_payload.get("records", []))
    selected_records = select_view_diverse_records(cached_records, config=config)
    for index, record in enumerate(selected_records):
        record["selected_index"] = index
    sessions = [session for session in list_recording_sessions(profile)]
    readiness = _readiness_for_cached_selection(selected_records, cached_records, config)
    force_build_allowed = readiness == READINESS_NEED_MORE_RECORDING and bool(selected_records)
    published = False
    if config.publish and selected_records:
        publish_selected_cached_references(
            profile,
            selected_records,
            cache_dir=cache_path,
            board_object=board_object,
            run_id=run_id,
            source_run_id=str(cache_payload.get("run_id") or ""),
        )
        published = True
    report = _build_report_from_cached_records(
        profile,
        run_id=run_id,
        readiness=readiness,
        selected_records=selected_records,
        cached_records=cached_records,
        sessions=sessions,
        config=config,
        force_build_allowed=force_build_allowed,
        published=published,
        processing_cache_path=str(cache_path),
    )
    return write_processing_report(profile, report)


def write_processing_cache(
    profile: ObjectProfile,
    evaluated: list[EvaluatedCandidate],
    *,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
    config: ReferenceProcessingConfig,
    run_id: str,
    mask_provider: MaskProvider | None = None,
) -> Path:
    cache_root = profile.root / PROCESSING_CACHE_DIRNAME
    run_dir = cache_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    records = []
    cv2 = _require_cv2()
    for item in evaluated:
        record = item.to_record()
        if item.mask is not None:
            mask_rel = Path("masks") / item.candidate.session_id / f"{item.candidate.frame_index:06d}.png"
            mask_path = run_dir / mask_rel
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), np.asarray(item.mask.mask).astype(np.uint8) * 255)
            record["cached_mask_path"] = mask_rel.as_posix()
        records.append(record)
    payload = {
        "version": PROCESSING_CACHE_VERSION,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "object": profile.name,
        "board_spec": board_spec.to_dict(),
        "quality_config": quality_config.to_dict(),
        "board_object": board_object.to_dict(),
        "thresholds": config.to_dict(),
        "mask_provider": _mask_provider_metadata(mask_provider),
        "records": records,
    }
    (run_dir / PROCESSING_CACHE_RECORDS).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / PROCESSING_CACHE_POINTER).write_text(
        json.dumps({"run_id": run_id, "cache_dir": str(run_dir)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def load_processing_cache(
    profile: ObjectProfile,
    *,
    cache_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if cache_dir is None:
        pointer_path = profile.root / PROCESSING_CACHE_DIRNAME / PROCESSING_CACHE_POINTER
        if not pointer_path.exists():
            raise FileNotFoundError("no processing cache found; run Processing first")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        cache_path = Path(str(pointer.get("cache_dir") or ""))
        if not cache_path.is_absolute():
            cache_path = pointer_path.parent / cache_path
    else:
        cache_path = Path(cache_dir).expanduser()
    records_path = cache_path / PROCESSING_CACHE_RECORDS
    if not records_path.exists():
        raise FileNotFoundError(f"processing cache records not found: {records_path}")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != PROCESSING_CACHE_VERSION:
        raise ValueError(f"unsupported processing cache version: {payload.get('version')}")
    return cache_path, payload


def load_recorded_candidates(profile: ObjectProfile) -> list[RecordedCandidate]:
    candidates: list[RecordedCandidate] = []
    for session in list_recording_sessions(profile):
        for record in load_frame_records(session.session_dir):
            candidates.append(load_recorded_candidate(session.session_dir, record))
    return sorted(candidates, key=lambda item: (item.session_id, item.frame_index))


def load_recorded_candidate(session_dir: Path, record: RecordedFrameRecord) -> RecordedCandidate:
    cv2 = _require_cv2()
    rgb_path = session_dir / record.rgb_path
    depth_path = session_dir / record.depth_path
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read recorded RGB frame: {rgb_path}")
    depth = np.load(depth_path).astype(np.float32)
    intrinsics = CameraIntrinsics.from_mapping(record.intrinsics)
    return RecordedCandidate(
        session_id=record.session_id,
        frame_index=record.index,
        session_dir=session_dir,
        frame_record=record,
        rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        depth_m=depth,
        intrinsics=intrinsics,
    )


def index_recorded_frame_records(profile: ObjectProfile) -> dict[str, tuple[Path, RecordedFrameRecord]]:
    records: dict[str, tuple[Path, RecordedFrameRecord]] = {}
    for session in list_recording_sessions(profile):
        for record in load_frame_records(session.session_dir):
            records[f"{record.session_id}:{record.index:06d}"] = (session.session_dir, record)
    return records


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
    max_keyframes = max(int(config.max_keyframes), 0)
    if max_keyframes <= 0:
        return []
    selected: list[EvaluatedCandidate] = []
    used_ids: set[str] = set()
    bins: dict[int, list[EvaluatedCandidate]] = {}
    for item in accepted:
        bin_id = int(item.view_bin) if item.view_bin is not None else 0
        bins.setdefault(bin_id, []).append(item)
    for items in bins.values():
        items.sort(key=lambda item: (-item.score, item.candidate.session_id, item.candidate.frame_index))

    bin_ids = sorted(bins)
    per_bin_cap = int(np.ceil(max_keyframes / max(len(bin_ids), 1)))
    per_bin_counts = {bin_id: 0 for bin_id in bin_ids}
    per_bin_offsets = {bin_id: 0 for bin_id in bin_ids}

    # Max is an upper bound, not a target. Keep selected references balanced across
    # observed view bins instead of padding oversampled angles with many near-duplicates.
    while len(selected) < max_keyframes:
        progressed = False
        for bin_id in bin_ids:
            if len(selected) >= max_keyframes:
                break
            if per_bin_counts[bin_id] >= per_bin_cap:
                continue
            items = bins[bin_id]
            offset = per_bin_offsets[bin_id]
            if offset >= len(items):
                continue
            item = items[offset]
            per_bin_offsets[bin_id] = offset + 1
            if item.candidate.candidate_id in used_ids:
                continue
            selected.append(item)
            used_ids.add(item.candidate.candidate_id)
            per_bin_counts[bin_id] += 1
            progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: (item.candidate.session_id, item.candidate.frame_index))


def select_view_diverse_records(
    records: list[dict[str, Any]],
    *,
    config: ReferenceProcessingConfig,
) -> list[dict[str, Any]]:
    accepted = [
        dict(record)
        for record in records
        if record.get("accepted")
        and record.get("cached_mask_path")
        and _record_cam_in_ob(record) is not None
    ]
    if not accepted:
        return []
    max_keyframes = max(int(config.max_keyframes), 0)
    if max_keyframes <= 0:
        return []
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    bins: dict[int, list[dict[str, Any]]] = {}
    for record in accepted:
        bin_id = int(record["view_bin"]) if record.get("view_bin") is not None else 0
        bins.setdefault(bin_id, []).append(record)
    for items in bins.values():
        items.sort(
            key=lambda record: (
                -float(record.get("score", 0.0)),
                str(record.get("session_id", "")),
                int(record.get("frame_index", 0)),
            )
        )

    bin_ids = sorted(bins)
    per_bin_cap = int(np.ceil(max_keyframes / max(len(bin_ids), 1)))
    per_bin_counts = {bin_id: 0 for bin_id in bin_ids}
    per_bin_offsets = {bin_id: 0 for bin_id in bin_ids}

    while len(selected) < max_keyframes:
        progressed = False
        for bin_id in bin_ids:
            if len(selected) >= max_keyframes:
                break
            if per_bin_counts[bin_id] >= per_bin_cap:
                continue
            items = bins[bin_id]
            offset = per_bin_offsets[bin_id]
            if offset >= len(items):
                continue
            record = items[offset]
            per_bin_offsets[bin_id] = offset + 1
            candidate_id = str(record.get("candidate_id", ""))
            if candidate_id in used_ids:
                continue
            selected.append(record)
            used_ids.add(candidate_id)
            per_bin_counts[bin_id] += 1
            progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda record: (str(record.get("session_id", "")), int(record.get("frame_index", 0))))


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


def publish_selected_cached_references(
    profile: ObjectProfile,
    selected_records: list[dict[str, Any]],
    *,
    cache_dir: Path,
    board_object: BoardObjectTransform,
    run_id: str,
    source_run_id: str,
) -> None:
    if not selected_records:
        return
    record_index = index_recorded_frame_records(profile)
    staging_root = profile.root / f".refs_staging_{run_id}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_profile = ObjectProfile(
        name=profile.name,
        root=staging_root,
        prompt=profile.prompt,
    )
    pose_results = []
    cv2 = _require_cv2()
    for output_index, record in enumerate(selected_records):
        candidate_id = str(record.get("candidate_id"))
        if candidate_id not in record_index:
            raise FileNotFoundError(f"recorded frame for cached candidate not found: {candidate_id}")
        session_dir, frame_record = record_index[candidate_id]
        candidate = load_recorded_candidate(session_dir, frame_record)
        mask_rel = record.get("cached_mask_path")
        if not mask_rel:
            raise FileNotFoundError(f"cached mask path missing for candidate: {candidate_id}")
        mask_path = cache_dir / str(mask_rel)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise FileNotFoundError(f"cached mask image not found: {mask_path}")
        if mask_image.shape != candidate.rgb.shape[:2]:
            mask_image = cv2.resize(
                mask_image,
                (candidate.rgb.shape[1], candidate.rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        cam_in_ob = _record_cam_in_ob(record)
        if cam_in_ob is None:
            raise ValueError(f"cached ChArUco pose missing cam_in_ob for candidate: {candidate_id}")
        save_reference_frame(
            staging_profile,
            output_index,
            rgb=candidate.rgb,
            depth_m=candidate.depth_m,
            mask=mask_image > 0,
            intrinsics=candidate.intrinsics,
            cam_in_ob=cam_in_ob,
            metadata={
                "capture_mode": "recording_processing_cache_reselect",
                "processing_run_id": run_id,
                "source_processing_run_id": source_run_id,
                "raw_candidate_id": candidate.candidate_id,
                "raw_session_id": candidate.session_id,
                "raw_frame_index": candidate.frame_index,
                "mask_source": record.get("mask_source"),
                "mask_metadata": record.get("mask_metadata"),
                "charuco_pose": record.get("charuco_pose"),
                "depth_stats": record.get("depth_stats"),
                "score": record.get("score"),
                "view_yaw_deg": record.get("view_yaw_deg"),
                "view_bin": record.get("view_bin"),
            },
        )
        pose_results.append(_cached_pose_result(record))
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
    profile.metadata["reference_workflow"] = "recording_processing_cache_reselect"
    profile.metadata["last_processing_run_id"] = run_id
    profile.metadata["source_processing_run_id"] = source_run_id
    profile.touch()
    profile.save()
    record_charuco_pose_provenance(
        profile,
        pose_results,
        board_object=board_object,
        indices=list(range(len(selected_records))),
    )
    mark_assets_stale(profile, "recording processing cache reselected references")


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
        processing_cache_path=report.processing_cache_path,
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
    processing_cache_path: str | None = None,
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
        processing_cache_path=processing_cache_path,
    )


def _build_report_from_cached_records(
    profile: ObjectProfile,
    *,
    run_id: str,
    readiness: str,
    selected_records: list[dict[str, Any]],
    cached_records: list[dict[str, Any]],
    sessions,
    config: ReferenceProcessingConfig,
    force_build_allowed: bool,
    published: bool,
    processing_cache_path: str | None,
) -> ReferenceProcessingReport:
    selected_ids = {str(record.get("candidate_id")) for record in selected_records}
    report_records = []
    for source in cached_records:
        record = dict(source)
        candidate_id = str(record.get("candidate_id"))
        if record.get("accepted") and candidate_id not in selected_ids:
            record["accepted"] = False
            record["reasons"] = ["accepted cached candidate not selected for publish window"]
        elif candidate_id in selected_ids:
            record["accepted"] = True
            record["reasons"] = []
            selected_index = next(
                (
                    selected.get("selected_index")
                    for selected in selected_records
                    if str(selected.get("candidate_id")) == candidate_id
                ),
                None,
            )
            record["selected_index"] = selected_index
        report_records.append(record)
    return ReferenceProcessingReport(
        object_name=profile.name,
        run_id=run_id,
        readiness=readiness,
        accepted=len(selected_records),
        rejected=len([record for record in cached_records if not record.get("accepted")]),
        processed_candidates=len(cached_records),
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
        records=report_records,
        output_reference_count=count_reference_frames(profile),
        processing_cache_path=processing_cache_path,
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


def _readiness_for_cached_selection(
    selected_records: list[dict[str, Any]],
    cached_records: list[dict[str, Any]],
    config: ReferenceProcessingConfig,
) -> str:
    if not cached_records:
        return READINESS_NO_RECORDINGS
    if len(selected_records) >= config.required_keyframes:
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


def _record_cam_in_ob(record: dict[str, Any]) -> np.ndarray | None:
    pose = record.get("charuco_pose")
    if not isinstance(pose, dict):
        return None
    value = pose.get("cam_in_ob")
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _cached_pose_result(record: dict[str, Any]) -> CharucoPoseResult:
    pose = record.get("charuco_pose")
    if not isinstance(pose, dict):
        raise ValueError("cached record missing charuco_pose")
    board_spec_data = pose.get("board_spec") if isinstance(pose.get("board_spec"), dict) else {}
    quality_data = pose.get("quality_config") if isinstance(pose.get("quality_config"), dict) else {}
    board_spec = CharucoBoardSpec(**_filter_kwargs(board_spec_data, CharucoBoardSpec))
    quality_config = CharucoQualityConfig(**_filter_kwargs(quality_data, CharucoQualityConfig))
    candidates = [
        DictionaryCandidateResult(
            dictionary=str(candidate.get("dictionary", "")),
            ok=bool(candidate.get("ok", False)),
            legacy_pattern=bool(candidate.get("legacy_pattern", False)),
            squares_x=int(candidate.get("squares_x", 0)),
            squares_y=int(candidate.get("squares_y", 0)),
            corner_count=int(candidate.get("corner_count", 0)),
            marker_count=int(candidate.get("marker_count", 0)),
            reprojection_error_px=candidate.get("reprojection_error_px"),
            image_coverage_fraction=float(candidate.get("image_coverage_fraction", 0.0)),
            reject_reasons=list(candidate.get("reject_reasons", [])),
            distortion_policy=str(candidate.get("distortion_policy", "unknown")),
        )
        for candidate in pose.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    return CharucoPoseResult(
        ok=bool(pose.get("ok", False)),
        selected_dictionary=pose.get("selected_dictionary"),
        candidates=candidates,
        board_spec=board_spec,
        quality_config=quality_config,
        opencv_version=str(pose.get("opencv_version", "unknown")),
        board_coordinate_convention=str(pose.get("board_coordinate_convention", "opencv_charuco_board")),
        legacy_pattern=bool(pose.get("legacy_pattern", False)),
        camera_T_board=_matrix_from_metadata(pose.get("camera_T_board")),
        camera_T_object=_matrix_from_metadata(pose.get("camera_T_object")),
        cam_in_ob=_matrix_from_metadata(pose.get("cam_in_ob")),
        reject_reasons=list(pose.get("reject_reasons", [])),
    )


def _filter_kwargs(data: dict[str, Any], cls) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {key: value for key, value in data.items() if key in fields}


def _matrix_from_metadata(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _validate_processing_cache(
    payload: dict[str, Any],
    *,
    board_spec: CharucoBoardSpec,
    quality_config: CharucoQualityConfig,
    board_object: BoardObjectTransform,
) -> None:
    if payload.get("object") is None:
        raise ValueError("processing cache is missing object metadata")
    cached_board = payload.get("board_spec")
    cached_quality = payload.get("quality_config")
    cached_object = payload.get("board_object")
    if cached_board != board_spec.to_dict():
        raise ValueError("cached processing used different ChArUco board settings; run Processing again")
    if cached_quality != quality_config.to_dict():
        raise ValueError("cached processing used different ChArUco quality settings; run Processing again")
    cached_matrix = None
    if isinstance(cached_object, dict):
        cached_matrix = cached_object.get("board_T_object")
    if cached_matrix is None or not np.allclose(
        np.asarray(cached_matrix, dtype=np.float64),
        board_object.board_T_object,
        atol=1e-9,
    ):
        raise ValueError("cached processing used different Obj XYZ/RPY; run Processing again")


def _mask_provider_metadata(mask_provider: MaskProvider | None) -> dict[str, Any]:
    if mask_provider is None:
        return {}
    return {
        "class": type(mask_provider).__name__,
        "prompt": getattr(mask_provider, "prompt", None),
        "device": getattr(mask_provider, "device", None),
        "resolution": getattr(mask_provider, "resolution", None),
        "confidence_threshold": getattr(mask_provider, "confidence_threshold", None),
        "autocast_dtype": getattr(mask_provider, "autocast_dtype", None),
    }


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
