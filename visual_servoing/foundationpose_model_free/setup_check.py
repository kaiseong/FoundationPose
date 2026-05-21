"""Environment checks for the FoundationPose model-free stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Iterable


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "required": self.required,
        }


def run_checks(*, foundationpose_path: str | Path | None = None) -> list[CheckResult]:
    results = [
        _python_check(),
        _conda_env_check(),
        _module_check("numpy", required=True),
        _module_check("torch", required=True),
        _module_check("cv2", required=True, package_hint="opencv-python"),
        _module_check("pyrealsense2", required=True),
        _module_check("sam3", required=False),
        _import_check("nvdiffrast.torch", required=True, package_hint="nvdiffrast"),
        _import_check("pytorch3d", required=True),
        _module_check("kaolin", required=False),
    ]
    results.extend(_foundationpose_checks(foundationpose_path))
    return results


def summarize(results: Iterable[CheckResult]) -> dict[str, object]:
    items = list(results)
    return {
        "ok": all(item.ok for item in items if item.required),
        "checks": [item.to_dict() for item in items],
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }


def _python_check() -> CheckResult:
    version = sys.version_info
    ok = (version.major, version.minor) == (3, 11)
    detail = (
        f"{version.major}.{version.minor}.{version.micro}; cloned FoundationPose "
        "environment.yml pins python=3.11"
    )
    return CheckResult("python_version", ok, detail, required=True)


def _conda_env_check() -> CheckResult:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    ok = env == "visual"
    detail = env or "not running inside a named conda env"
    return CheckResult("conda_env_visual", ok, detail, required=False)


def _module_check(name: str, *, required: bool, package_hint: str | None = None) -> CheckResult:
    spec = importlib.util.find_spec(name)
    hint = package_hint or name
    if spec is None:
        return CheckResult(name, False, f"missing; install {hint}", required=required)
    origin = spec.origin or "namespace/package"
    return CheckResult(name, True, origin, required=required)


def _import_check(name: str, *, required: bool, package_hint: str | None = None) -> CheckResult:
    hint = package_hint or name
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return CheckResult(name, False, f"missing or failed import; install {hint}: {exc}", required=required)
    origin = getattr(module, "__file__", None) or "namespace/package"
    return CheckResult(name, True, str(origin), required=required)


def _foundationpose_checks(path: str | Path | None) -> list[CheckResult]:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("FOUNDATIONPOSE_ROOT")
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([Path.cwd() / "FoundationPose", Path.home() / "FoundationPose"])

    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        estimater = candidate / "estimater.py"
        run_nerf = candidate / "bundlesdf" / "run_nerf.py"
        if estimater.exists():
            ok = run_nerf.exists()
            detail = f"{candidate}; run_nerf={'yes' if ok else 'missing'}"
            mycpp_files = list((candidate / "mycpp" / "build").glob("mycpp*.so"))
            weights_ok, weights_detail = _weights_check(candidate)
            return [
                CheckResult("foundationpose_root", True, str(candidate), required=True),
                CheckResult("foundationpose_model_free_tools", ok, detail, required=True),
                CheckResult(
                    "foundationpose_mycpp",
                    bool(mycpp_files),
                    str(mycpp_files[0]) if mycpp_files else "missing; run build_all_conda.sh",
                    required=True,
                ),
                CheckResult("foundationpose_weights", weights_ok, weights_detail, required=True),
            ]
    return [
        CheckResult(
            "foundationpose_root",
            False,
            "missing; clone NVlabs/FoundationPose and set FOUNDATIONPOSE_ROOT",
            required=True,
        ),
        CheckResult("foundationpose_model_free_tools", False, "not checked without root", required=True),
        CheckResult("foundationpose_mycpp", False, "not checked without root", required=True),
        CheckResult("foundationpose_weights", False, "not checked without root", required=True),
    ]


def _weights_check(root: Path) -> tuple[bool, str]:
    expected = [
        root / "weights" / "2024-01-11-20-02-45" / "model_best.pth",
        root / "weights" / "2024-01-11-20-02-45" / "config.yml",
        root / "weights" / "2023-10-28-18-33-37" / "model_best.pth",
        root / "weights" / "2023-10-28-18-33-37" / "config.yml",
    ]
    missing = [path.relative_to(root) for path in expected if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        return False, f"missing FoundationPose scorer/refiner weights: {joined}"
    return True, str(root / "weights")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationpose-path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    summary = summarize(run_checks(foundationpose_path=args.foundationpose_path))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for item in summary["checks"]:
            marker = "OK" if item["ok"] else "MISS"
            req = "required" if item["required"] else "optional"
            print(f"{marker:4} {item['name']} ({req}) - {item['detail']}")
    return 1 if args.strict and not summary["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
