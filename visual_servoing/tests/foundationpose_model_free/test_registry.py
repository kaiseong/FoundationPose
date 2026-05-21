from __future__ import annotations

import pytest

from visual_servoing.foundationpose_model_free.profile_schema import ProfileStatus
from visual_servoing.foundationpose_model_free.registry import ObjectProfileRegistry


def test_registry_creates_lists_selects_and_deletes_profile(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)

    profile = registry.create("phone_black", prompt="mobile phone")

    assert profile.name == "phone_black"
    assert profile.prompt == "mobile phone"
    assert profile.status == ProfileStatus.CREATED
    assert profile.profile_path.exists()
    assert [item.name for item in registry.list()] == ["phone_black"]

    selected = registry.select("phone_black")
    assert selected.selected is True
    assert registry.get("phone_black").selected is True

    with pytest.raises(ValueError, match="confirm"):
        registry.delete("phone_black")
    registry.delete("phone_black", confirm=True)
    assert registry.list() == []


def test_registry_rejects_invalid_profile_name(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)

    with pytest.raises(ValueError, match="profile name"):
        registry.create("../bad")


def test_registry_prompt_update_marks_existing_assets_stale(tmp_path):
    registry = ObjectProfileRegistry(tmp_path)
    profile = registry.create("mouse", prompt="mouse")
    mesh = profile.assets_dir / "model" / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# obj\n", encoding="utf-8")
    profile.asset_status = "ready"
    profile.generated_assets = [str(mesh)]
    profile.save()

    updated = registry.create("mouse", prompt="wireless mouse", exist_ok=True)

    assert updated.prompt == "wireless mouse"
    assert updated.asset_status == "stale"
    assert updated.metadata["asset_stale_reason"] == "profile prompt changed"
