"""Profile manifest and generated-asset freshness helpers."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .profile_schema import ObjectProfile, ProfileStatus, utc_now_iso


MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
ASSET_STATUS_MISSING = "missing"
ASSET_STATUS_STALE = "stale"
ASSET_STATUS_READY = "ready"
ASSET_STATUS_FAILED = "failed"


class ManifestError(RuntimeError):
    pass


def manifest_path(profile: ObjectProfile) -> Path:
    return profile.root / MANIFEST_FILE


def read_profile_manifest(profile: ObjectProfile, *, migrate: bool = True) -> dict[str, Any]:
    path = manifest_path(profile)
    if not path.exists():
        if not migrate:
            raise FileNotFoundError(f"profile manifest not found: {path}")
        if _has_generated_artifact(profile) or profile.asset_status == ASSET_STATUS_READY:
            profile.asset_status = ASSET_STATUS_STALE
            if profile.status == ProfileStatus.ASSETS_READY:
                profile.status = ProfileStatus.CAPTURED
            profile.metadata["asset_stale_reason"] = "legacy profile without manifest"
            profile.metadata["asset_stale_at"] = utc_now_iso()
            profile.touch()
            profile.save()
        manifest = build_profile_manifest(
            profile,
            reason="lazy_migration",
            stale_reason=profile.metadata.get("asset_stale_reason"),
        )
        write_profile_manifest(profile, manifest)
        return manifest
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestError(f"profile manifest is corrupt or torn: {path}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"profile manifest must be a JSON object: {path}")
    missing = [key for key in _required_manifest_fields() if key not in data]
    if missing:
        raise ManifestError(f"profile manifest missing required fields {missing}: {path}")
    if int(data.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"profile manifest schema {data.get('schema_version')} is unsupported; "
            f"expected {MANIFEST_SCHEMA_VERSION}: {path}"
        )
    return data


def write_profile_manifest(profile: ObjectProfile, manifest: dict[str, Any]) -> None:
    profile.ensure_dirs()
    path = manifest_path(profile)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_directory(path.parent)


def build_profile_manifest(
    profile: ObjectProfile,
    *,
    reason: str,
    deterministic_validation_report: dict[str, Any] | None = None,
    heuristic_policy: dict[str, Any] | None = None,
    heuristic_report: dict[str, Any] | None = None,
    stale_reason: str | None = None,
    deterministic_build_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_name": profile.name,
        "prompt": profile.prompt,
        "asset_status": profile.asset_status,
        "manifest_reason": reason,
        "manifest_updated_at": utc_now_iso(),
        "source_dependency_fingerprint": compute_source_dependency_fingerprint(
            profile,
            deterministic_build_inputs=deterministic_build_inputs,
        ),
        "source_dependency_records": collect_source_dependency_records(profile),
        "artifact_records": collect_artifact_records(profile),
        "deterministic_build_inputs": deterministic_build_inputs or {},
        "deterministic_validation_report": deterministic_validation_report
        or _empty_validation_report(),
        "heuristic_policy": heuristic_policy or _default_heuristic_policy(),
        "heuristic_report": heuristic_report or _default_heuristic_report(),
        "stale_reason": stale_reason,
    }


def refresh_profile_manifest(
    profile: ObjectProfile,
    *,
    reason: str,
    deterministic_validation_report: dict[str, Any] | None = None,
    heuristic_policy: dict[str, Any] | None = None,
    heuristic_report: dict[str, Any] | None = None,
    stale_reason: str | None = None,
    deterministic_build_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_profile_manifest(
        profile,
        reason=reason,
        deterministic_validation_report=deterministic_validation_report,
        heuristic_policy=heuristic_policy,
        heuristic_report=heuristic_report,
        stale_reason=stale_reason,
        deterministic_build_inputs=deterministic_build_inputs,
    )
    write_profile_manifest(profile, manifest)
    return manifest


def mark_assets_stale(profile: ObjectProfile, reason: str) -> dict[str, Any]:
    if _has_generated_artifact(profile) or profile.asset_status in {
        ASSET_STATUS_READY,
        ASSET_STATUS_STALE,
        ASSET_STATUS_FAILED,
    }:
        profile.asset_status = ASSET_STATUS_STALE
        if profile.status == ProfileStatus.ASSETS_READY:
            profile.status = ProfileStatus.CAPTURED
    else:
        profile.asset_status = ASSET_STATUS_MISSING
    profile.metadata["asset_stale_reason"] = reason
    profile.metadata["asset_stale_at"] = utc_now_iso()
    profile.touch()
    profile.save()
    return refresh_profile_manifest(profile, reason="asset_stale", stale_reason=reason)


def record_asset_ready(
    profile: ObjectProfile,
    *,
    generated_assets: Iterable[str | Path],
    deterministic_validation_report: dict[str, Any] | None = None,
    deterministic_build_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile.asset_status = ASSET_STATUS_READY
    profile.status = ProfileStatus.ASSETS_READY
    profile.generated_assets = [str(path) for path in generated_assets]
    profile.metadata.pop("asset_stale_reason", None)
    profile.metadata.pop("asset_stale_at", None)
    profile.touch()
    profile.save()
    return refresh_profile_manifest(
        profile,
        reason="asset_ready",
        deterministic_validation_report=deterministic_validation_report,
        deterministic_build_inputs=deterministic_build_inputs,
    )


def ensure_asset_freshness(
    profile: ObjectProfile,
    *,
    deterministic_build_inputs: dict[str, Any] | None = None,
) -> bool:
    manifest = read_profile_manifest(profile)
    build_inputs = (
        deterministic_build_inputs
        if deterministic_build_inputs is not None
        else dict(manifest.get("deterministic_build_inputs", {}))
    )
    current_fingerprint = compute_source_dependency_fingerprint(
        profile,
        deterministic_build_inputs=build_inputs,
    )
    stored_fingerprint = str(manifest.get("source_dependency_fingerprint", ""))
    if profile.asset_status == ASSET_STATUS_READY and stored_fingerprint != current_fingerprint:
        mark_assets_stale(profile, "source dependency fingerprint changed")
        return False
    return profile.asset_status == ASSET_STATUS_READY and stored_fingerprint == current_fingerprint


def compute_source_dependency_fingerprint(
    profile: ObjectProfile,
    *,
    deterministic_build_inputs: dict[str, Any] | None = None,
) -> str:
    payload = {
        "profile_name": profile.name,
        "prompt": profile.prompt,
        "reference_count": profile.reference_count,
        "pose_source": profile.metadata.get("pose_source"),
        "pose_provenance": profile.metadata.get("pose_provenance"),
        "turntable_assumptions": profile.metadata.get("turntable_assumptions"),
        "deterministic_build_inputs": deterministic_build_inputs or {},
        "source_records": [_fingerprint_record(record) for record in collect_source_dependency_records(profile)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_source_dependency_records(profile: ObjectProfile) -> list[dict[str, Any]]:
    files: list[Path] = [
        profile.refs_dir / "K.txt",
        profile.refs_dir / "select_frames.yml",
        profile.refs_dir / "intrinsics.json",
    ]
    files.extend(_sorted_files(profile.rgb_dir, "*.png"))
    files.extend(_sorted_files(profile.depth_dir, "*.npy"))
    files.extend(_sorted_files(profile.depth_enhanced_dir, "*.png"))
    files.extend(_sorted_files(profile.mask_dir, "*.png"))
    files.extend(_sorted_files(profile.cam_in_ob_dir, "*.txt"))
    return [_file_record(profile.root, path) for path in files]


def collect_artifact_records(profile: ObjectProfile) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for asset in profile.generated_assets:
        path = Path(asset)
        candidates.append(path if path.is_absolute() else profile.root / path)
    candidates.extend(
        [
            profile.assets_dir / "model" / "model.obj",
            profile.refs_dir / "model" / "model.obj",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = resolved if resolved.is_absolute() else profile.root / resolved
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return [_file_record(profile.root, path) for path in unique]


def _has_generated_artifact(profile: ObjectProfile) -> bool:
    return any(record["exists"] for record in collect_artifact_records(profile))


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    if not path.exists():
        return {"path": relative, "exists": False}
    stat = path.stat()
    return {
        "path": relative,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _fingerprint_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key in {"path", "exists", "size", "sha256"}
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def _required_manifest_fields() -> tuple[str, ...]:
    return (
        "schema_version",
        "source_dependency_fingerprint",
        "artifact_records",
        "deterministic_validation_report",
        "heuristic_policy",
        "heuristic_report",
    )


def _empty_validation_report() -> dict[str, Any]:
    return {
        "ok": None,
        "errors": [],
        "warnings": [],
        "checked_at": utc_now_iso(),
    }


def _default_heuristic_policy() -> dict[str, Any]:
    return {
        "asset_freshness_policy": "deterministic_dependencies_only",
        "mask_quality_severity": "block",
        "tracking_watchdogs_affect_assets": False,
    }


def _default_heuristic_report() -> dict[str, Any]:
    return {
        "ok": None,
        "warnings": [],
        "checked_at": utc_now_iso(),
    }


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
