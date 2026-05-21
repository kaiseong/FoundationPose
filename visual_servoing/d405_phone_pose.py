#!/usr/bin/env python3
"""Standalone script entrypoint for D405 phone pose visualization."""

from __future__ import annotations

from pathlib import Path
import sys

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
_ADDED_PACKAGE_PARENT = False
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)
    _ADDED_PACKAGE_PARENT = True

from visual_servoing.run_d405_phone_pose import main

if _ADDED_PACKAGE_PARENT:
    sys.path.remove(_PACKAGE_PARENT)


if __name__ == "__main__":
    raise SystemExit(main())
