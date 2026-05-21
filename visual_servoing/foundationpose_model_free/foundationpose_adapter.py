"""Thin adapter around FoundationPose register/track APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from visual_servoing.common.torch_state import reset_torch_defaults_for_cpu_ops, set_torch_defaults_for_cuda_ops
from visual_servoing.point_pose.rgbd_geometry import CameraIntrinsics


class FoundationPoseUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class FoundationPoseConfig:
    foundationpose_root: Path | None = None
    mesh_path: Path | None = None
    debug_dir: Path | None = None
    debug: int = 0
    refinement_iterations: int = 5
    tracking_iterations: int = 2


@dataclass(frozen=True)
class PoseEstimate:
    camera_T_object: np.ndarray
    source: str
    metadata: dict[str, object]


class FoundationPoseAdapter:
    """Runtime wrapper that keeps heavy FoundationPose imports lazy."""

    def __init__(self, config: FoundationPoseConfig) -> None:
        self.config = config
        self._estimator: Any | None = None
        self._initialized = False

    def register(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: np.ndarray,
    ) -> PoseEstimate:
        set_torch_defaults_for_cuda_ops()
        estimator = self._get_estimator()
        set_torch_defaults_for_cuda_ops()
        try:
            pose = estimator.register(
                K=intrinsics.as_matrix(),
                rgb=np.asarray(rgb),
                depth=np.asarray(depth_m),
                ob_mask=np.asarray(mask).astype(bool),
                iteration=self.config.refinement_iterations,
            )
        finally:
            reset_torch_defaults_for_cpu_ops()
        self._initialized = True
        return PoseEstimate(np.asarray(pose, dtype=np.float64), "foundationpose_register", {})

    def track_one(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> PoseEstimate:
        if not self._initialized:
            raise RuntimeError("FoundationPoseAdapter.track_one() called before register().")
        set_torch_defaults_for_cuda_ops()
        estimator = self._get_estimator()
        set_torch_defaults_for_cuda_ops()
        try:
            pose = estimator.track_one(
                rgb=np.asarray(rgb),
                depth=np.asarray(depth_m),
                K=intrinsics.as_matrix(),
                iteration=self.config.tracking_iterations,
            )
        finally:
            reset_torch_defaults_for_cpu_ops()
        return PoseEstimate(np.asarray(pose, dtype=np.float64), "foundationpose_track_one", {})

    def _get_estimator(self):
        if self._estimator is not None:
            return self._estimator
        if self.config.foundationpose_root is not None:
            root = str(Path(self.config.foundationpose_root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            import estimater  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on FoundationPose install
            raise FoundationPoseUnavailableError(
                "FoundationPose is not importable. Set FOUNDATIONPOSE_ROOT or run setup_check."
            ) from exc
        set_torch_defaults_for_cuda_ops()
        try:
            self._estimator = self._construct_estimator(estimater)
        finally:
            reset_torch_defaults_for_cpu_ops()
        return self._estimator

    def _construct_estimator(self, estimater_module):
        mesh_path = self.config.mesh_path
        if mesh_path is None:
            raise FoundationPoseUnavailableError(
                "A generated model-free mesh/object asset is required before live tracking."
            )
        try:
            import trimesh  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on FoundationPose install
            raise FoundationPoseUnavailableError("trimesh is required to load generated object assets.") from exc
        mesh = trimesh.load(str(mesh_path), force="mesh")
        scorer = estimater_module.ScorePredictor()
        refiner = estimater_module.PoseRefinePredictor()
        glctx = estimater_module.dr.RasterizeCudaContext()
        debug_dir = str(self.config.debug_dir) if self.config.debug_dir else "debug_foundationpose"
        return estimater_module.FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=debug_dir,
            debug=self.config.debug,
            glctx=glctx,
        )


class StubFoundationPoseAdapter:
    """Deterministic adapter for tests and GUI mock mode."""

    def __init__(self) -> None:
        self._initialized = False
        self._pose = np.eye(4, dtype=np.float64)

    def register(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: np.ndarray,
    ) -> PoseEstimate:
        self._initialized = True
        self._pose = np.eye(4, dtype=np.float64)
        self._pose[2, 3] = float(np.nanmedian(depth_m[np.asarray(mask).astype(bool)])) if np.any(mask) else 1.0
        return PoseEstimate(self._pose.copy(), "stub_register", {})

    def track_one(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> PoseEstimate:
        if not self._initialized:
            raise RuntimeError("Stub adapter called before register().")
        return PoseEstimate(self._pose.copy(), "stub_track_one", {})
