"""Tracking timing and pose-stability metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.asarray(a, dtype=np.float64)[:3, :3]
    rb = np.asarray(b, dtype=np.float64)[:3, :3]
    delta = ra.T @ rb
    cosine = (float(np.trace(delta)) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def translation_delta_m(a: np.ndarray, b: np.ndarray) -> float:
    ta = np.asarray(a, dtype=np.float64)[:3, 3]
    tb = np.asarray(b, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(tb - ta))


@dataclass
class TrackingMetrics:
    started_at: float = field(default_factory=time.perf_counter)
    frames: int = 0
    drops: int = 0
    translation_jitter_m: list[float] = field(default_factory=list)
    rotation_jitter_deg: list[float] = field(default_factory=list)
    last_pose: np.ndarray | None = None

    def update(self, pose: np.ndarray | None) -> None:
        self.frames += 1
        if pose is None:
            self.drops += 1
            return
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"pose must have shape (4, 4), got {pose.shape}")
        if self.last_pose is not None:
            self.translation_jitter_m.append(translation_delta_m(self.last_pose, pose))
            self.rotation_jitter_deg.append(rotation_angle_deg(self.last_pose, pose))
        self.last_pose = pose.copy()

    def summary(self) -> dict[str, float]:
        elapsed_s = max(time.perf_counter() - self.started_at, 1e-9)
        return {
            "frames": float(self.frames),
            "drops": float(self.drops),
            "drop_rate": float(self.drops / self.frames) if self.frames else 0.0,
            "fps": float(self.frames / elapsed_s),
            "translation_jitter_m_mean": _mean(self.translation_jitter_m),
            "rotation_jitter_deg_mean": _mean(self.rotation_jitter_deg),
        }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))
