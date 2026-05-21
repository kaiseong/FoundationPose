from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.metrics import (
    TrackingMetrics,
    rotation_angle_deg,
    translation_delta_m,
)


def test_pose_delta_helpers_report_translation_and_rotation():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.1
    second[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert translation_delta_m(first, second) == 0.1
    assert abs(rotation_angle_deg(first, second) - 90.0) < 1e-6


def test_tracking_metrics_counts_drops_and_jitter():
    metrics = TrackingMetrics()
    first = np.eye(4)
    second = np.eye(4)
    second[2, 3] = 0.02

    metrics.update(first)
    metrics.update(None)
    metrics.update(second)
    summary = metrics.summary()

    assert summary["frames"] == 3.0
    assert summary["drops"] == 1.0
    assert summary["drop_rate"] == 1.0 / 3.0
    assert summary["translation_jitter_m_mean"] == 0.02
