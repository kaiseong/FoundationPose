from __future__ import annotations

import json

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


def test_registry_delete_removes_legacy_processing_cache_references(tmp_path):
    registry = ObjectProfileRegistry(tmp_path / "current")
    profile = registry.create("multimeter", prompt="multimeter")
    legacy_cache = tmp_path / "legacy" / "visual_servoing_data" / "object_profiles" / "multimeter" / "processing_cache"
    legacy_run = legacy_cache / "process-old"
    legacy_run.mkdir(parents=True)
    (legacy_run / "records.json").write_text("{}", encoding="utf-8")
    unrelated_cache = tmp_path / "legacy" / "visual_servoing_data" / "object_profiles" / "other" / "processing_cache"
    unrelated_cache.mkdir(parents=True)

    profile.metadata["reference_processing"] = {
        "processing_cache_path": str(legacy_run),
        "processing_summary": {
            "source_processing_cache_path": str(unrelated_cache),
        },
    }
    profile.save()
    report_dir = profile.logs_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "reference_processing_latest.json").write_text(
        json.dumps({"processing_cache_path": str(legacy_run)}, indent=2),
        encoding="utf-8",
    )

    registry.delete("multimeter", confirm=True)

    assert not profile.root.exists()
    assert not legacy_cache.exists()
    assert unrelated_cache.exists()
