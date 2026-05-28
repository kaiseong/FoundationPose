"""Debug artifact packaging for model-free FoundationPose processing."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile
from typing import Any

import numpy as np

from .profile_schema import ObjectProfile
from .reference_processing import latest_processing_report, index_recorded_frame_records


DEBUG_ARTIFACT_MANIFEST = "manifest.json"


def build_processing_debug_artifacts_zip(profile: ObjectProfile) -> tuple[bytes, dict[str, Any]]:
    """Return a zip containing selected-candidate Processing debug artifacts."""

    report = latest_processing_report(profile)
    if not report:
        raise FileNotFoundError("no Processing report found; run Processing first")
    cache_path = _processing_cache_path(profile, report)
    selected = _selected_processing_records(report)
    if not selected:
        raise ValueError("latest Processing report has no selected candidates")
    frame_index = index_recorded_frame_records(profile)
    processing_summary = dict(report.get("processing_summary")) if isinstance(report.get("processing_summary"), dict) else {}
    thresholds = dict(report.get("thresholds")) if isinstance(report.get("thresholds"), dict) else {}
    min_depth_m = float(thresholds.get("min_depth_m", 0.0))
    max_depth_m = float(thresholds.get("max_depth_m", np.inf))
    cv2 = _require_cv2()

    manifest: dict[str, Any] = {
        "profile": profile.name,
        "run_id": report.get("run_id"),
        "processing_cache_path": str(cache_path),
        "candidate_count": len(selected),
        "accepted_count": int(report.get("accepted", len(selected))),
        "eligible_count": processing_summary.get("eligible_count"),
        "excluded_count": processing_summary.get("excluded_count"),
        "selected_count": processing_summary.get("selected_count", len(selected)),
        "excluded_candidate_ids": list(processing_summary.get("excluded_candidate_ids") or []),
        "charuco_origin_convention": processing_summary.get("charuco_origin_convention"),
        "charuco_origin_offset_board_m": processing_summary.get("charuco_origin_offset_board_m"),
        "candidates": [],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for record in selected:
            candidate_id = str(record.get("candidate_id") or "")
            if candidate_id not in frame_index:
                raise FileNotFoundError(f"recorded frame for selected candidate not found: {candidate_id}")
            session_dir, frame_record = frame_index[candidate_id]
            selected_index = int(record.get("selected_index"))
            stem = _artifact_stem(selected_index, str(record.get("session_id") or frame_record.session_id), int(record.get("frame_index", frame_record.index)))
            candidate_manifest: dict[str, Any] = {
                "candidate_id": candidate_id,
                "exclude_id": candidate_id,
                "artifact_stem": stem,
                "selected_index": selected_index,
                "session_id": str(record.get("session_id") or frame_record.session_id),
                "frame_index": int(record.get("frame_index", frame_record.index)),
                "charuco_axes": None,
                "mask": None,
                "depth_colormap": None,
                "source": {
                    "charuco_axes_preview_path": record.get("charuco_axes_preview_path"),
                    "cached_mask_path": record.get("cached_mask_path"),
                    "depth_path": frame_record.depth_path,
                },
            }

            axes_rel = record.get("charuco_axes_preview_path")
            if axes_rel:
                axes_path = cache_path / str(axes_rel)
                if axes_path.exists():
                    archive_name = f"charuco_axes/{stem}.png"
                    zf.write(axes_path, archive_name)
                    candidate_manifest["charuco_axes"] = archive_name

            mask_rel = record.get("cached_mask_path")
            if not mask_rel:
                raise FileNotFoundError(f"selected candidate is missing cached mask path: {candidate_id}")
            mask_path = cache_path / str(mask_rel)
            if not mask_path.exists():
                raise FileNotFoundError(f"selected candidate mask not found: {mask_path}")
            mask_png = _binary_mask_png(mask_path, cv2=cv2)
            mask_archive_name = f"masks/{stem}.png"
            zf.writestr(mask_archive_name, mask_png)
            candidate_manifest["mask"] = mask_archive_name

            depth_path = session_dir / frame_record.depth_path
            if not depth_path.exists():
                raise FileNotFoundError(f"selected candidate depth not found: {depth_path}")
            depth = np.load(depth_path).astype(np.float32)
            depth_png = _depth_colormap_png(depth, min_depth_m=min_depth_m, max_depth_m=max_depth_m, cv2=cv2)
            depth_archive_name = f"depth_colormap/{stem}.png"
            zf.writestr(depth_archive_name, depth_png)
            candidate_manifest["depth_colormap"] = depth_archive_name

            manifest["candidates"].append(candidate_manifest)
        zf.writestr(DEBUG_ARTIFACT_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    archive = buffer.getvalue()
    return archive, {
        "profile": profile.name,
        "run_id": report.get("run_id"),
        "candidate_count": len(selected),
        "archive_bytes": len(archive),
    }


def _processing_cache_path(profile: ObjectProfile, report: dict[str, Any]) -> Path:
    raw_path = report.get("processing_cache_path")
    if not raw_path:
        raise FileNotFoundError("Processing report does not reference a processing cache")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = profile.root / path
    if not path.exists():
        raise FileNotFoundError(f"Processing cache not found: {path}")
    return path


def _selected_processing_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        return []
    selected = [
        dict(record)
        for record in records
        if isinstance(record, dict) and record.get("accepted") and record.get("selected_index") is not None
    ]
    return sorted(selected, key=lambda record: int(record.get("selected_index", 0)))


def _artifact_stem(selected_index: int, session_id: str, frame_index: int) -> str:
    safe_session = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in session_id)
    return f"{selected_index:03d}_{safe_session}_{frame_index:06d}"


def _binary_mask_png(mask_path: Path, *, cv2) -> bytes:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read cached mask image: {mask_path}")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise RuntimeError(f"failed to encode mask image: {mask_path}")
    return encoded.tobytes()


def _depth_colormap_png(depth_m: np.ndarray, *, min_depth_m: float, max_depth_m: float, cv2) -> bytes:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if np.isfinite(min_depth_m):
        valid &= depth >= float(min_depth_m)
    if np.isfinite(max_depth_m):
        valid &= depth <= float(max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    valid_values = depth[valid]
    if valid_values.size:
        lo = float(np.percentile(valid_values, 2.0))
        hi = float(np.percentile(valid_values, 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = (np.clip(depth, lo, hi) - lo) / (hi - lo)
        normalized[valid] = np.clip(scaled[valid] * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color[~valid] = 0
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise RuntimeError("failed to encode depth colormap image")
    return encoded.tobytes()


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("OpenCV is required to package debug artifacts") from exc
    return cv2
