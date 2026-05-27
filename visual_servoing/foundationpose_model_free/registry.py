"""Registry for saved FoundationPose object profiles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from visual_servoing.common.paths import object_profiles_root

from .profile_schema import MANIFEST_FILE, ObjectProfile, PROFILE_FILE, validate_profile_name
from .reference_processing import PROCESSING_CACHE_DIRNAME, PROCESSING_CACHE_POINTER, REPORT_FILENAME


class ObjectProfileRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = object_profiles_root(root)

    def create(self, name: str, *, prompt: str = "object", exist_ok: bool = False) -> ObjectProfile:
        name = validate_profile_name(name)
        profile_dir = self.root / name
        profile_path = profile_dir / PROFILE_FILE
        if profile_path.exists() and not exist_ok:
            raise FileExistsError(f"Object profile already exists: {name}")
        profile = ObjectProfile(name=name, root=profile_dir, prompt=prompt)
        if profile_path.exists() and exist_ok:
            existing = ObjectProfile.load(profile_path)
            requested_prompt = prompt or existing.prompt
            if requested_prompt != existing.prompt:
                existing.prompt = requested_prompt
                from .profile_manifest import mark_assets_stale

                mark_assets_stale(existing, "profile prompt changed")
            else:
                from .profile_manifest import read_profile_manifest

                read_profile_manifest(existing)
                existing.touch()
                existing.save()
            return existing
        profile.save()
        from .profile_manifest import refresh_profile_manifest

        refresh_profile_manifest(profile, reason="profile_created")
        return profile

    def get(self, name: str) -> ObjectProfile:
        name = validate_profile_name(name)
        profile_path = self.root / name / PROFILE_FILE
        if not profile_path.exists():
            raise FileNotFoundError(f"Object profile not found: {name}")
        profile = ObjectProfile.load(profile_path)
        from .profile_manifest import read_profile_manifest

        read_profile_manifest(profile)
        return profile

    def list(self) -> list[ObjectProfile]:
        if not self.root.exists():
            return []
        profiles = []
        for profile_path in sorted(self.root.glob(f"*/{PROFILE_FILE}")):
            profile = ObjectProfile.load(profile_path)
            from .profile_manifest import read_profile_manifest

            read_profile_manifest(profile)
            profiles.append(profile)
        return profiles

    def delete(self, name: str, *, confirm: bool = False) -> None:
        name = validate_profile_name(name)
        if not confirm:
            raise ValueError("delete requires confirm=True")
        profile_dir = self.root / name
        if not profile_dir.exists():
            raise FileNotFoundError(f"Object profile not found: {name}")
        external_processing_caches = _discover_external_processing_cache_roots(profile_dir, name)
        shutil.rmtree(profile_dir)
        for cache_root in external_processing_caches:
            if cache_root.exists():
                shutil.rmtree(cache_root)

    def select(self, name: str) -> ObjectProfile:
        selected = self.get(name)
        for profile in self.list():
            profile.selected = profile.name == selected.name
            profile.touch()
            profile.save()
            if profile.name == selected.name:
                selected = profile
        return selected


def _discover_external_processing_cache_roots(profile_dir: Path, profile_name: str) -> list[Path]:
    """Find processing cache roots referenced by this profile before deleting it.

    Normal current-layout caches live inside ``profile_dir`` and are removed by
    deleting the profile itself. This also cleans up legacy absolute cache paths
    recorded in profile metadata or reports, while refusing paths that do not
    look like ``.../object_profiles/<profile>/processing_cache``.
    """

    profile_dir = profile_dir.resolve()
    roots: set[Path] = set()
    for cache_path in _iter_referenced_processing_cache_paths(profile_dir):
        cache_root = _processing_cache_root_for_profile(cache_path, profile_name)
        if cache_root is None:
            continue
        if _is_relative_to(cache_root, profile_dir):
            continue
        roots.add(cache_root)
    return sorted(roots, key=lambda path: str(path), reverse=True)


def _iter_referenced_processing_cache_paths(profile_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for payload_path in (
        profile_dir / PROFILE_FILE,
        profile_dir / MANIFEST_FILE,
        profile_dir / "logs" / REPORT_FILENAME,
        profile_dir / PROCESSING_CACHE_DIRNAME / PROCESSING_CACHE_POINTER,
    ):
        payload = _load_json_if_present(payload_path)
        if payload is None:
            continue
        paths.extend(_cache_paths_from_payload(payload, base_dir=payload_path.parent))
    return paths


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_paths_from_payload(payload: dict[str, Any], *, base_dir: Path) -> list[Path]:
    raw_values: list[Any] = []
    raw_values.append(payload.get("processing_cache_path"))
    raw_values.append(payload.get("source_processing_cache_path"))
    raw_values.append(payload.get("cache_dir"))

    summary = payload.get("processing_summary")
    if isinstance(summary, dict):
        raw_values.append(summary.get("source_processing_cache_path"))

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        reference_processing = metadata.get("reference_processing")
        if isinstance(reference_processing, dict):
            raw_values.extend(_cache_paths_from_payload(reference_processing, base_dir=base_dir))

    paths: list[Path] = []
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        paths.append(path.resolve(strict=False))
    return paths


def _processing_cache_root_for_profile(path: Path, profile_name: str) -> Path | None:
    parts = path.resolve(strict=False).parts
    for index, part in enumerate(parts):
        if part != "object_profiles":
            continue
        if index + 2 >= len(parts):
            continue
        if parts[index + 1] == profile_name and parts[index + 2] == PROCESSING_CACHE_DIRNAME:
            return Path(*parts[: index + 3])
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True
