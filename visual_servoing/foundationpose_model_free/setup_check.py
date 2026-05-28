"""Environment checks for the FoundationPose model-free stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from visual_servoing.point_pose.live_camera_config import SUPPORTED_LIVE_CAMERA_MODELS, SUPPORTED_REALSENSE_MODELS


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


def run_checks(*, foundationpose_path: str | Path | None = None, camera: str = "all") -> list[CheckResult]:
    camera = camera.lower()
    if camera != "all" and camera not in SUPPORTED_LIVE_CAMERA_MODELS:
        raise ValueError(f"unsupported camera: {camera}")
    check_realsense = camera == "all" or camera in SUPPORTED_REALSENSE_MODELS
    check_zed = camera == "all" or camera == "zed"
    results = [
        _python_check(),
        _conda_env_check(),
        _module_check("numpy", required=True),
        _module_check("torch", required=True),
        _module_check("cv2", required=True, package_hint="opencv-python"),
        _module_check("sam3", required=False),
        _import_check("nvdiffrast.torch", required=True, package_hint="nvdiffrast"),
        _import_check("pytorch3d", required=True),
        _module_check("trimesh", required=True),
        _module_check("xatlas", required=True),
        _module_check("rtree", required=True),
        _trimesh_proximity_check(required=True),
        _pyopengl_version_check(required=True),
        _pyrender_egl_texture_check(required=True),
        _module_check("kaolin", required=False),
    ]
    if check_realsense:
        results.append(_module_check("pyrealsense2", required=True))
    if check_zed:
        results.append(_zed_sdk_check(required=camera == "zed"))
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
    ok = (version.major, version.minor) in {(3, 11), (3, 12)}
    detail = (
        f"{version.major}.{version.minor}.{version.micro}; FoundationPose environment.yml "
        "pins python=3.11, unified SAM3/FoundationPose envs may use python=3.12"
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


def _pyopengl_version_check(*, required: bool) -> CheckResult:
    name = "pyopengl"
    try:
        version = importlib.metadata.version("PyOpenGL")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(name, False, "missing; install PyOpenGL>=3.1.10", required=required)
    parts = tuple(int(part) for part in version.split(".")[:3] if part.isdigit())
    ok = parts >= (3, 1, 10)
    detail = f"{version}; Python 3.12/EGL textured rendering needs PyOpenGL>=3.1.10"
    return CheckResult(name, ok, detail, required=required)


def _trimesh_proximity_check(*, required: bool) -> CheckResult:
    try:
        import numpy as np
        import trimesh

        mesh = trimesh.creation.box()
        points = np.array([[0.25, 0.25, 1.0]], dtype=np.float64)
        locations, distances, triangle_ids = trimesh.proximity.closest_point(mesh, points)
        if locations.shape != (1, 3) or distances.shape != (1,) or triangle_ids.shape != (1,):
            raise RuntimeError("unexpected closest_point output shapes")
    except Exception as exc:
        return CheckResult(
            "trimesh_proximity",
            False,
            f"failed; install rtree for trimesh.proximity.closest_point: {exc}",
            required=required,
        )
    return CheckResult("trimesh_proximity", True, "closest_point OK", required=required)


def _pyrender_egl_texture_check(*, required: bool) -> CheckResult:
    previous_platform = os.environ.get("PYOPENGL_PLATFORM")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import numpy as np
        import pyrender
        import trimesh

        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            image=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        scene.add(pyrender.PerspectiveCamera(yfov=1.0), pose=np.eye(4))
        renderer = pyrender.OffscreenRenderer(8, 8)
        try:
            renderer.render(scene)
        finally:
            renderer.delete()
    except Exception as exc:
        return CheckResult(
            "pyrender_egl_texture",
            False,
            f"failed; set PYOPENGL_PLATFORM=egl and install PyOpenGL>=3.1.10: {exc}",
            required=required,
        )
    finally:
        if previous_platform is None:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ["PYOPENGL_PLATFORM"] = previous_platform
    return CheckResult("pyrender_egl_texture", True, "textured EGL offscreen render OK", required=required)


def _zed_sdk_check(*, required: bool) -> CheckResult:
    try:
        from visual_servoing.point_pose.zed_camera import check_zed_backend
    except Exception as exc:
        return CheckResult("zed_sdk", False, f"diagnostic unavailable: {exc}", required=required)
    diagnostic = check_zed_backend()
    return CheckResult("zed_sdk", diagnostic.ok, diagnostic.detail, required=required)


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
    parser.add_argument("--camera", choices=(*SUPPORTED_LIVE_CAMERA_MODELS, "all"), default="all")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    summary = summarize(run_checks(foundationpose_path=args.foundationpose_path, camera=args.camera))
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
