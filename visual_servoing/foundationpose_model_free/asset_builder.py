"""FoundationPose model-free asset build wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .profile_manifest import (
    ASSET_STATUS_READY,
    ASSET_STATUS_STALE,
    ensure_asset_freshness,
    read_profile_manifest,
    record_asset_ready,
    refresh_profile_manifest,
)
from .profile_schema import ObjectProfile, ProfileStatus
from .reference_dataset import foundationpose_ref_view_dir
from .charuco_reference import provenance_summary as charuco_provenance_summary


MIN_REFERENCE_FRAMES = 1
MIN_MASK_DEPTH_VALID_RATIO = 0.01


@dataclass(frozen=True)
class AssetBuildResult:
    command: list[str]
    returncode: int
    elapsed_ms: float
    stdout: str = ""
    stderr: str = ""
    executed: bool = False
    validation_report: dict[str, Any] | None = None


class FoundationPoseAssetBuilder:
    def __init__(
        self,
        *,
        foundationpose_root: str | Path,
        python_executable: str | Path | None = None,
        min_reference_frames: int = MIN_REFERENCE_FRAMES,
    ) -> None:
        self.foundationpose_root = Path(foundationpose_root).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.min_reference_frames = max(int(min_reference_frames), 1)

    @property
    def run_nerf_path(self) -> Path:
        return self.foundationpose_root / "bundlesdf" / "run_nerf.py"

    @property
    def profile_runner_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "fp_run_profile_nerf.py"

    def build_command(self, profile: ObjectProfile) -> list[str]:
        return [
            self.python_executable,
            str(self.profile_runner_path),
            "--foundationpose-root",
            str(self.foundationpose_root),
            "--ref_view_dir",
            str(foundationpose_ref_view_dir(profile)),
            "--output-dir",
            str(profile_model_dir(profile)),
            "--config",
            str(self.foundationpose_root / "bundlesdf" / "config_ycbv.yml"),
        ]

    def validate_inputs(self, profile: ObjectProfile) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        frame_records: list[dict[str, Any]] = []
        build_inputs = self.deterministic_build_inputs()
        if not self.run_nerf_path.exists():
            errors.append(f"FoundationPose run_nerf.py not found: {self.run_nerf_path}")
        config_path = self.foundationpose_root / "bundlesdf" / "config_ycbv.yml"
        if not config_path.exists():
            errors.append(f"FoundationPose BundleSDF config not found: {config_path}")
        if not self.profile_runner_path.exists():
            errors.append(f"profile run_nerf wrapper not found: {self.profile_runner_path}")

        rgb_indices = _frame_indices(profile.rgb_dir, "*.png")
        depth_indices = _frame_indices(profile.depth_enhanced_dir, "*.png")
        mask_indices = _frame_indices(profile.mask_dir, "*.png")
        pose_indices = _frame_indices(profile.cam_in_ob_dir, "*.txt")
        all_index_sets = {
            "rgb": rgb_indices,
            "depth_enhanced": depth_indices,
            "mask": mask_indices,
            "cam_in_ob": pose_indices,
        }
        if len(rgb_indices) < self.min_reference_frames:
            errors.append(
                f"profile {profile.name} has {len(rgb_indices)} RGB reference frame(s); "
                f"minimum is {self.min_reference_frames}"
            )
        if len(rgb_indices) < 16:
            warnings.append("fewer than 16 reference views; reconstruction quality may be weak")
        mismatched = {
            name: sorted(indices)
            for name, indices in all_index_sets.items()
            if indices != rgb_indices
        }
        if mismatched:
            errors.append(
                "reference frame index mismatch: "
                + ", ".join(f"{name}={indices}" for name, indices in mismatched.items())
            )

        select_frames = profile.refs_dir / "select_frames.yml"
        if not select_frames.exists():
            errors.append("FoundationPose model-free asset build needs refs/select_frames.yml")
        k_path = profile.refs_dir / "K.txt"
        if not k_path.exists():
            errors.append("FoundationPose model-free asset build needs refs/K.txt")
        else:
            try:
                k_matrix = np.loadtxt(k_path)
                if k_matrix.shape != (3, 3) or not np.all(np.isfinite(k_matrix)):
                    errors.append(f"refs/K.txt must be a finite 3x3 matrix, got shape {k_matrix.shape}")
            except Exception as exc:
                errors.append(f"could not read refs/K.txt: {exc}")

        cv2 = _optional_cv2()
        if rgb_indices and not mismatched:
            for index in sorted(rgb_indices):
                frame_report = _validate_reference_frame(profile, index, cv2=cv2)
                frame_records.append(frame_report)
                errors.extend(frame_report["errors"])
                warnings.extend(frame_report["warnings"])

        manifest = read_profile_manifest(profile)
        if profile.asset_status == ASSET_STATUS_READY:
            ensure_asset_freshness(profile, deterministic_build_inputs=build_inputs)
            if profile.asset_status == ASSET_STATUS_STALE:
                warnings.append("generated assets are stale and will be rebuilt")
        report: dict[str, Any] = {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "frame_count": len(rgb_indices),
            "frame_records": frame_records,
            "manifest_asset_status": manifest.get("asset_status"),
            "current_asset_status": profile.asset_status,
            "checked_at": time.time(),
            "deterministic_build_inputs": build_inputs,
            "pose_source": profile.metadata.get("pose_source"),
            "charuco_provenance": charuco_provenance_summary(profile),
        }
        refresh_profile_manifest(
            profile,
            reason="build_validation",
            deterministic_validation_report=report,
            deterministic_build_inputs=build_inputs,
        )
        if errors:
            raise ValueError("Invalid FoundationPose profile inputs: " + "; ".join(errors))
        return report

    def build(self, profile: ObjectProfile, *, execute: bool = False) -> AssetBuildResult:
        command = self.build_command(profile)
        start = time.perf_counter()
        validation_report = self.validate_inputs(profile)
        if not execute:
            return AssetBuildResult(
                command=command,
                returncode=0,
                elapsed_ms=0.0,
                executed=False,
                validation_report=validation_report,
            )
        profile.status = ProfileStatus.BUILDING
        profile.touch()
        profile.save()
        env = os.environ.copy()
        conda_lib = str(Path(sys.prefix) / "lib")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            command,
            cwd=str(self.foundationpose_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = AssetBuildResult(
            command=command,
            returncode=int(completed.returncode),
            elapsed_ms=elapsed_ms,
            stdout=completed.stdout,
            stderr=completed.stderr,
            executed=True,
            validation_report=validation_report,
        )
        profile.logs_dir.mkdir(parents=True, exist_ok=True)
        with (profile.logs_dir / "build.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.__dict__, sort_keys=True) + "\n")
        mesh = find_generated_mesh(profile, require_fresh=False)
        if completed.returncode == 0 and mesh is not None:
            record_asset_ready(
                profile,
                generated_assets=[mesh],
                deterministic_validation_report=validation_report,
                deterministic_build_inputs=self.deterministic_build_inputs(),
            )
        else:
            profile.status = ProfileStatus.FAILED
            profile.asset_status = "failed"
            if completed.returncode == 0 and mesh is None:
                validation_report["warnings"].append("BundleSDF returned success but no model.obj was found")
            profile.touch()
            profile.save()
            refresh_profile_manifest(
                profile,
                reason="asset_build_failed",
                deterministic_validation_report=validation_report,
                deterministic_build_inputs=self.deterministic_build_inputs(),
            )
        return result

    def deterministic_build_inputs(self) -> dict[str, Any]:
        return {
            "foundationpose_root": str(self.foundationpose_root),
            "python_executable": str(self.python_executable),
            "run_nerf": _external_file_record(self.run_nerf_path),
            "config_ycbv": _external_file_record(self.foundationpose_root / "bundlesdf" / "config_ycbv.yml"),
            "profile_runner": _external_file_record(self.profile_runner_path),
        }


def profile_model_dir(profile: ObjectProfile) -> Path:
    return profile.assets_dir / "model"


def profile_model_path(profile: ObjectProfile) -> Path:
    return profile_model_dir(profile) / "model.obj"


def find_generated_mesh(profile: ObjectProfile, *, require_fresh: bool = True) -> Path | None:
    if require_fresh and not ensure_asset_freshness(profile):
        return None
    candidates: list[Path] = []
    for asset in profile.generated_assets:
        path = Path(asset)
        candidates.append(path if path.is_absolute() else profile.root / path)
    candidates.extend(
        [
            profile_model_path(profile),
            profile.refs_dir / "model" / "model.obj",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _frame_indices(directory: Path, pattern: str) -> set[int]:
    return {int(path.stem) for path in directory.glob(pattern) if path.stem.isdigit()}


def _validate_reference_frame(profile: ObjectProfile, index: int, *, cv2) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    mask_path = profile.mask_dir / f"{index:06d}.png"
    depth_path = profile.depth_enhanced_dir / f"{index:06d}.png"
    pose_path = profile.cam_in_ob_dir / f"{index:06d}.txt"
    mask_area = 0
    valid_depth_ratio = 0.0
    if cv2 is None:
        errors.append("OpenCV is required to validate reference masks/depth images")
    else:
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            errors.append(f"could not read mask image: {mask_path}")
        if depth_image is None:
            errors.append(f"could not read depth image: {depth_path}")
        if mask_image is not None:
            mask = mask_image > 0
            mask_area = int(mask.sum())
            if mask_area == 0:
                errors.append(f"reference mask is empty: {mask_path}")
        if mask_image is not None and depth_image is not None and mask_area > 0:
            if depth_image.shape[:2] != mask_image.shape[:2]:
                errors.append(f"depth/mask shape mismatch for frame {index:06d}")
            else:
                valid_depth = depth_image[mask_image > 0]
                valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0)]
                valid_depth_ratio = float(valid_depth.size / max(mask_area, 1))
                if valid_depth_ratio < MIN_MASK_DEPTH_VALID_RATIO:
                    errors.append(
                        f"usable depth coverage is {valid_depth_ratio:.3f} for frame {index:06d}; "
                        f"minimum is {MIN_MASK_DEPTH_VALID_RATIO:.3f}"
                    )
                elif valid_depth_ratio < 0.25:
                    warnings.append(f"low masked depth coverage {valid_depth_ratio:.3f} for frame {index:06d}")
    try:
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            errors.append(f"cam_in_ob/{index:06d}.txt must be a 4x4 matrix, got shape {pose.shape}")
        elif not np.all(np.isfinite(pose)):
            errors.append(f"cam_in_ob/{index:06d}.txt contains non-finite values")
    except Exception as exc:
        errors.append(f"could not read cam_in_ob/{index:06d}.txt: {exc}")
    return {
        "index": index,
        "mask_area_px": mask_area,
        "valid_depth_ratio": valid_depth_ratio,
        "errors": errors,
        "warnings": warnings,
    }


def _optional_cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def _external_file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
