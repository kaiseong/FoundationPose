"""Pluggable object-mask providers for FoundationPose initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from pathlib import Path
from typing import Protocol

import numpy as np


class MaskProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaskQualityConfig:
    min_area_fraction: float = 0.0005
    max_area_fraction: float = 0.85
    min_confidence: float | None = None
    min_bbox_width_px: int = 2
    min_bbox_height_px: int = 2
    min_largest_component_fraction: float = 0.50
    min_depth_valid_ratio: float = 0.01
    temporal_min_iou: float | None = None
    temporal_max_centroid_shift_px: float | None = None
    severity: str = "block"

    def __post_init__(self) -> None:
        if self.severity not in {"block", "warn", "operator_override"}:
            raise ValueError("mask quality severity must be block, warn, or operator_override")


@dataclass(frozen=True)
class MaskQualityResult:
    ok: bool
    severity: str
    reasons: list[str]
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    source: str
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class MaskProvider(Protocol):
    def get_mask(
        self,
        image_rgb: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> MaskResult:
        ...


class PrecomputedMaskProvider:
    def __init__(self, mask_path: str | Path, *, quality_config: MaskQualityConfig | None = None) -> None:
        self.mask_path = Path(mask_path)
        self.quality_config = quality_config or MaskQualityConfig()
        self._previous_mask: np.ndarray | None = None

    def get_mask(
        self,
        image_rgb: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> MaskResult:
        mask = load_binary_mask(self.mask_path, shape=np.asarray(image_rgb).shape[:2])
        quality = validate_mask_quality(
            mask,
            image_shape=np.asarray(image_rgb).shape[:2],
            depth_m=depth_m,
            previous_mask=self._previous_mask,
            config=self.quality_config,
        )
        _raise_for_quality_failure(quality)
        self._previous_mask = mask.copy()
        return MaskResult(
            mask=mask,
            source="precomputed",
            metadata={"path": str(self.mask_path), "mask_quality": quality.__dict__},
        )


class ManualPolygonMaskProvider:
    def __init__(
        self,
        points_xy: np.ndarray | list[tuple[float, float]],
        *,
        quality_config: MaskQualityConfig | None = None,
    ) -> None:
        self.points_xy = np.asarray(points_xy, dtype=np.int32)
        self.quality_config = quality_config or MaskQualityConfig()
        self._previous_mask: np.ndarray | None = None
        if self.points_xy.ndim != 2 or self.points_xy.shape[1] != 2 or self.points_xy.shape[0] < 3:
            raise ValueError("manual polygon requires at least three (x, y) points")

    def get_mask(
        self,
        image_rgb: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> MaskResult:
        cv2 = _require_cv2()
        height, width = np.asarray(image_rgb).shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.points_xy.reshape(-1, 1, 2)], 255)
        mask_bool = mask > 0
        quality = validate_mask_quality(
            mask_bool,
            image_shape=np.asarray(image_rgb).shape[:2],
            depth_m=depth_m,
            previous_mask=self._previous_mask,
            config=self.quality_config,
        )
        _raise_for_quality_failure(quality)
        self._previous_mask = mask_bool.copy()
        return MaskResult(mask=mask_bool, source="manual_polygon", metadata={"mask_quality": quality.__dict__})


class Sam3MaskProvider:
    def __init__(
        self,
        *,
        prompt: str = "object",
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        resolution: int = 1008,
        autocast_dtype: str | None = "float32",
        fallback_to_cpu_on_cuda_oom: bool = True,
        quality_config: MaskQualityConfig | None = None,
    ) -> None:
        self.prompt = prompt
        self.device = _resolve_sam3_device(device)
        self.confidence_threshold = confidence_threshold
        self.resolution = int(resolution)
        self.autocast_dtype = autocast_dtype
        self.fallback_to_cpu_on_cuda_oom = fallback_to_cpu_on_cuda_oom
        self.quality_config = quality_config or MaskQualityConfig(min_confidence=confidence_threshold)
        self._segmenter = None
        self._previous_mask: np.ndarray | None = None

    def get_mask(
        self,
        image_rgb: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        object_name: str | None = None,
    ) -> MaskResult:
        prompt = object_name or self.prompt
        fallback_metadata: dict[str, object] = {}
        try:
            selection = self._segment_once(prompt, image_rgb)
        except Exception as exc:  # pragma: no cover - depends on SAM3 runtime
            detail = str(exc)
            if not detail and exc.__cause__ is not None:
                detail = str(exc.__cause__)
            if self._can_retry_without_autocast(detail):
                fallback_metadata = {
                    "fallback_from_device": self.device,
                    "fallback_reason": detail,
                    "fallback_action": "disabled_cuda_autocast",
                }
                self.release()
                self.autocast_dtype = "float32"
                try:
                    selection = self._segment_once(prompt, image_rgb)
                except Exception as retry_exc:
                    retry_detail = str(retry_exc)
                    if not retry_detail and retry_exc.__cause__ is not None:
                        retry_detail = str(retry_exc.__cause__)
                    if not self._can_retry_on_cpu(retry_detail):
                        message = "SAM3 failed to produce an initialization mask after disabling CUDA autocast."
                        if retry_detail:
                            message = f"{message} Root cause: {retry_detail}"
                        raise MaskProviderError(message) from retry_exc
                    fallback_metadata["fallback_reason_after_autocast_retry"] = retry_detail
                    self.release()
                    self.device = "cpu"
                    try:
                        selection = self._segment_once(prompt, image_rgb)
                    except Exception as cpu_exc:
                        cpu_detail = str(cpu_exc)
                        if not cpu_detail and cpu_exc.__cause__ is not None:
                            cpu_detail = str(cpu_exc.__cause__)
                        message = "SAM3 failed to produce an initialization mask after CPU fallback."
                        if cpu_detail:
                            message = f"{message} Root cause: {cpu_detail}"
                        raise MaskProviderError(message) from cpu_exc
            elif self._can_retry_on_cpu(detail):
                fallback_metadata = {
                    "fallback_from_device": self.device,
                    "fallback_reason": detail,
                }
                self.release()
                self.device = "cpu"
                try:
                    selection = self._segment_once(prompt, image_rgb)
                except Exception as cpu_exc:
                    cpu_detail = str(cpu_exc)
                    if not cpu_detail and cpu_exc.__cause__ is not None:
                        cpu_detail = str(cpu_exc.__cause__)
                    message = "SAM3 failed to produce an initialization mask after CPU fallback."
                    if cpu_detail:
                        message = f"{message} Root cause: {cpu_detail}"
                    raise MaskProviderError(message) from cpu_exc
            else:
                message = "SAM3 failed to produce an initialization mask."
                if detail:
                    message = f"{message} Root cause: {detail}"
                raise MaskProviderError(message) from exc
        quality = validate_mask_quality(
            selection.mask,
            image_shape=np.asarray(image_rgb).shape[:2],
            confidence=selection.score,
            depth_m=depth_m,
            previous_mask=self._previous_mask,
            config=self.quality_config,
        )
        _raise_for_quality_failure(quality)
        self._previous_mask = np.asarray(selection.mask).astype(bool).copy()
        return MaskResult(
            mask=selection.mask,
            source="sam3",
            confidence=selection.score,
            metadata={
                "area": selection.area,
                "prompt": prompt,
                "index": selection.index,
                "device": self.device,
                "resolution": self.resolution,
                "autocast_dtype": self.autocast_dtype,
                **fallback_metadata,
                "mask_quality": quality.__dict__,
            },
        )

    def _segment_once(self, prompt: str, image_rgb: np.ndarray):
        segmenter = self._get_segmenter(prompt)
        return segmenter.segment(image_rgb)

    def _can_retry_on_cpu(self, detail: str) -> bool:
        return (
            self.fallback_to_cpu_on_cuda_oom
            and self.device.split(":", 1)[0] == "cuda"
            and _looks_like_cuda_recoverable_failure(detail)
        )

    def _can_retry_without_autocast(self, detail: str) -> bool:
        return (
            self.device.split(":", 1)[0] == "cuda"
            and str(self.autocast_dtype or "").lower() in {"bfloat16", "bf16", "float16", "fp16", "half"}
            and _looks_like_autocast_dtype_mismatch(detail)
        )

    def _get_segmenter(self, prompt: str):
        if self._segmenter is not None and self._segmenter.prompt == prompt:
            return self._segmenter
        try:
            from visual_servoing.point_pose.sam3_phone_segmenter import Sam3PhoneSegmenter
        except Exception as exc:  # pragma: no cover - depends on local package state
            raise MaskProviderError("SAM3 mask provider requires the point_pose SAM3 wrapper.") from exc
        self._segmenter = Sam3PhoneSegmenter(
            prompt=prompt,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            resolution=self.resolution,
            autocast_dtype=self.autocast_dtype,
        )
        return self._segmenter

    def release(self) -> None:
        self._segmenter = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return


def load_binary_mask(path: str | Path, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        mask = np.load(path).astype(bool)
    else:
        cv2 = _require_cv2()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        mask = image > 0
    if shape is not None and mask.shape != shape:
        cv2 = _require_cv2()
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def _resolve_sam3_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested != "auto":
        return requested or "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _looks_like_cuda_recoverable_failure(detail: str) -> bool:
    text = str(detail).lower()
    if "no usable object mask" in text or "no candidate mask" in text:
        return False
    return any(
        marker in text
        for marker in (
            "sam initialize failed",
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
            "failed to allocate",
            "driver",
            "device-side assert",
            "no kernel image",
        )
    )


def _looks_like_autocast_dtype_mismatch(detail: str) -> bool:
    text = str(detail).lower()
    return (
        "must have the same dtype" in text
        and ("bfloat16" in text or "float16" in text or "half" in text)
        and "float" in text
    )


def validate_mask_quality(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    confidence: float | None = None,
    depth_m: np.ndarray | None = None,
    previous_mask: np.ndarray | None = None,
    config: MaskQualityConfig | None = None,
) -> MaskQualityResult:
    cfg = config or MaskQualityConfig()
    mask_bool = np.asarray(mask).astype(bool)
    height, width = int(image_shape[0]), int(image_shape[1])
    reasons: list[str] = []
    if mask_bool.shape != (height, width):
        reasons.append(f"mask shape {mask_bool.shape} does not match image shape {(height, width)}")
    area_px = int(mask_bool.sum()) if mask_bool.shape == (height, width) else 0
    image_area = max(height * width, 1)
    area_fraction = float(area_px / image_area)
    bbox_width = 0
    bbox_height = 0
    centroid_x: float | None = None
    centroid_y: float | None = None
    if area_px <= 0:
        reasons.append("mask is empty")
    else:
        rows, cols = np.nonzero(mask_bool)
        bbox_width = int(cols.max() - cols.min() + 1)
        bbox_height = int(rows.max() - rows.min() + 1)
        centroid_x = float(np.mean(cols))
        centroid_y = float(np.mean(rows))
        if area_fraction < cfg.min_area_fraction:
            reasons.append(f"mask area fraction {area_fraction:.6f} below minimum {cfg.min_area_fraction:.6f}")
        if area_fraction > cfg.max_area_fraction:
            reasons.append(f"mask area fraction {area_fraction:.3f} above maximum {cfg.max_area_fraction:.3f}")
        if bbox_width < cfg.min_bbox_width_px or bbox_height < cfg.min_bbox_height_px:
            reasons.append(f"mask bbox {bbox_width}x{bbox_height}px is too small")
    if confidence is not None and cfg.min_confidence is not None and float(confidence) < float(cfg.min_confidence):
        reasons.append(f"mask confidence {float(confidence):.3f} below minimum {float(cfg.min_confidence):.3f}")

    largest_component_fraction = _largest_component_fraction(mask_bool) if area_px > 0 else 0.0
    if area_px > 0 and largest_component_fraction < cfg.min_largest_component_fraction:
        reasons.append(
            f"largest connected component fraction {largest_component_fraction:.3f} below minimum "
            f"{cfg.min_largest_component_fraction:.3f}"
        )

    depth_valid_ratio: float | None = None
    if depth_m is not None and area_px > 0 and mask_bool.shape == np.asarray(depth_m).shape[:2]:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = depth[mask_bool]
        valid = valid[np.isfinite(valid) & (valid > 0.0)]
        depth_valid_ratio = float(valid.size / max(area_px, 1))
        if depth_valid_ratio < cfg.min_depth_valid_ratio:
            reasons.append(
                f"mask depth support {depth_valid_ratio:.3f} below minimum {cfg.min_depth_valid_ratio:.3f}"
            )

    temporal_iou: float | None = None
    centroid_shift_px: float | None = None
    if previous_mask is not None and area_px > 0:
        previous = np.asarray(previous_mask).astype(bool)
        if previous.shape == mask_bool.shape and previous.any():
            union = np.logical_or(previous, mask_bool).sum()
            temporal_iou = float(np.logical_and(previous, mask_bool).sum() / max(int(union), 1))
            previous_rows, previous_cols = np.nonzero(previous)
            prev_centroid = np.array([np.mean(previous_cols), np.mean(previous_rows)], dtype=float)
            current_centroid = np.array([centroid_x or 0.0, centroid_y or 0.0], dtype=float)
            centroid_shift_px = float(np.linalg.norm(current_centroid - prev_centroid))
            if cfg.temporal_min_iou is not None and temporal_iou < cfg.temporal_min_iou:
                reasons.append(f"temporal mask IoU {temporal_iou:.3f} below minimum {cfg.temporal_min_iou:.3f}")
            if (
                cfg.temporal_max_centroid_shift_px is not None
                and centroid_shift_px > cfg.temporal_max_centroid_shift_px
            ):
                reasons.append(
                    f"temporal mask centroid shift {centroid_shift_px:.1f}px above maximum "
                    f"{cfg.temporal_max_centroid_shift_px:.1f}px"
                )

    ok = not reasons or cfg.severity != "block"
    return MaskQualityResult(
        ok=ok,
        severity=cfg.severity,
        reasons=reasons,
        metrics={
            "area_px": area_px,
            "area_fraction": area_fraction,
            "bbox_width_px": bbox_width,
            "bbox_height_px": bbox_height,
            "confidence": None if confidence is None else float(confidence),
            "largest_component_fraction": largest_component_fraction,
            "depth_valid_ratio": depth_valid_ratio,
            "temporal_iou": temporal_iou,
            "centroid_shift_px": centroid_shift_px,
        },
    )


def _raise_for_quality_failure(quality: MaskQualityResult) -> None:
    if not quality.ok:
        raise MaskProviderError("mask quality failed: " + "; ".join(quality.reasons))


def _largest_component_fraction(mask: np.ndarray) -> float:
    try:
        cv2 = _require_cv2()
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    except Exception:
        return 1.0
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(np.max(areas) / max(int(mask.sum()), 1))


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise MaskProviderError("OpenCV is required for this mask provider.") from exc
    return cv2
