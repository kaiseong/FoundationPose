"""Filesystem layout helpers for visual servoing data."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_DIR_NAME = "visual_servoing_data"
ENV_DATA_DIR = "VISUAL_SERVOING_DATA_DIR"


def data_root(root: str | Path | None = None) -> Path:
    """Return the base directory for generated object profiles and logs."""

    if root is not None:
        return Path(root).expanduser().resolve()
    configured = os.environ.get(ENV_DATA_DIR)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / DEFAULT_DATA_DIR_NAME).resolve()


def object_profiles_root(root: str | Path | None = None) -> Path:
    return data_root(root) / "object_profiles"
