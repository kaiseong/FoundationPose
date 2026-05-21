"""Run FoundationPose BundleSDF model-free mesh build for one local profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ADDED_PACKAGE_PARENT: str | None = None
if __package__ in (None, ""):
    _PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
        _ADDED_PACKAGE_PARENT = _PACKAGE_PARENT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-root", required=True)
    parser.add_argument("--ref_view_dir", "--ref-view-dir", dest="ref_view_dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    foundationpose_root = Path(args.foundationpose_root).expanduser().resolve()
    bundlesdf_root = foundationpose_root / "bundlesdf"
    for path in (str(bundlesdf_root), str(foundationpose_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import yaml  # type: ignore
    from run_nerf import run_one_ob  # type: ignore

    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    mesh = run_one_ob(base_dir=str(Path(args.ref_view_dir).expanduser().resolve()), cfg=cfg)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model.obj"
    mesh.export(output_file)
    print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
