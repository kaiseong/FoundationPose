"""Object profile schema for FoundationPose model-free assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


PROFILE_FILE = "profile.json"
MANIFEST_FILE = "manifest.json"
VALID_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ProfileStatus:
    CREATED = "created"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    BUILDING = "building"
    ASSETS_READY = "assets_ready"
    FAILED = "failed"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_profile_name(name: str) -> str:
    name = str(name).strip()
    if not VALID_PROFILE_NAME.match(name):
        raise ValueError(
            "profile name must start with an alphanumeric character and contain "
            "only letters, numbers, '_', '-', or '.', with max length 64"
        )
    return name


@dataclass
class ObjectProfile:
    name: str
    root: Path
    prompt: str = "object"
    status: str = ProfileStatus.CREATED
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    reference_count: int = 0
    asset_status: str = "missing"
    selected: bool = False
    generated_assets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = validate_profile_name(self.name)
        self.root = Path(self.root)

    @property
    def profile_path(self) -> Path:
        return self.root / PROFILE_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def refs_dir(self) -> Path:
        return self.root / "refs"

    @property
    def rgb_dir(self) -> Path:
        return self.refs_dir / "rgb"

    @property
    def depth_dir(self) -> Path:
        return self.refs_dir / "depth"

    @property
    def depth_enhanced_dir(self) -> Path:
        return self.refs_dir / "depth_enhanced"

    @property
    def mask_dir(self) -> Path:
        return self.refs_dir / "mask"

    @property
    def cam_in_ob_dir(self) -> Path:
        return self.refs_dir / "cam_in_ob"

    @property
    def assets_dir(self) -> Path:
        return self.root / "foundationpose"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.rgb_dir,
            self.depth_dir,
            self.depth_enhanced_dir,
            self.mask_dir,
            self.cam_in_ob_dir,
            self.assets_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reference_count": int(self.reference_count),
            "asset_status": self.asset_status,
            "selected": bool(self.selected),
            "generated_assets": list(self.generated_assets),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, root: Path, data: dict[str, Any]) -> "ObjectProfile":
        return cls(
            name=str(data["name"]),
            root=Path(root),
            prompt=str(data.get("prompt", "object")),
            status=str(data.get("status", ProfileStatus.CREATED)),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            reference_count=int(data.get("reference_count", 0)),
            asset_status=str(data.get("asset_status", "missing")),
            selected=bool(data.get("selected", False)),
            generated_assets=list(data.get("generated_assets", [])),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObjectProfile":
        profile_path = Path(path)
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        return cls.from_dict(profile_path.parent, data)

    def save(self) -> None:
        self.ensure_dirs()
        self.profile_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
