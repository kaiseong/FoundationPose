"""Registry for saved FoundationPose object profiles."""

from __future__ import annotations

import shutil
from pathlib import Path

from visual_servoing.common.paths import object_profiles_root

from .profile_schema import ObjectProfile, PROFILE_FILE, validate_profile_name


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
        shutil.rmtree(profile_dir)

    def select(self, name: str) -> ObjectProfile:
        selected = self.get(name)
        for profile in self.list():
            profile.selected = profile.name == selected.name
            profile.touch()
            profile.save()
            if profile.name == selected.name:
                selected = profile
        return selected
