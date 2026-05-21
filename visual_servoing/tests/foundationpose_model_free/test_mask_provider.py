from __future__ import annotations

import numpy as np

from visual_servoing.foundationpose_model_free.mask_provider import (
    ManualPolygonMaskProvider,
    MaskQualityConfig,
    MaskProviderError,
    PrecomputedMaskProvider,
    Sam3MaskProvider,
    validate_mask_quality,
)


def test_precomputed_mask_provider_loads_npy_mask(tmp_path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    result = PrecomputedMaskProvider(path).get_mask(image)

    assert result.source == "precomputed"
    assert result.mask.dtype == bool
    assert int(result.mask.sum()) == 6


def test_manual_polygon_mask_provider_rasterizes_polygon():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    provider = ManualPolygonMaskProvider([(5, 5), (15, 5), (10, 15)])

    result = provider.get_mask(image)

    assert result.source == "manual_polygon"
    assert result.mask.shape == image.shape[:2]
    assert int(result.mask.sum()) > 0


def test_sam3_mask_provider_reports_root_cause():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    provider = Sam3MaskProvider(prompt="wireless mouse")
    provider._segmenter = FailingSegmenter(prompt="wireless mouse")

    try:
        provider.get_mask(image)
    except MaskProviderError as exc:
        assert "Root cause: no candidate mask" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


def test_sam3_mask_provider_falls_back_to_cpu_on_cuda_oom():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    provider = CpuFallbackProvider(prompt="wireless mouse", device="cuda", resolution=512)

    result = provider.get_mask(image, object_name="wireless mouse")

    assert result.source == "sam3"
    assert result.metadata["device"] == "cpu"
    assert result.metadata["resolution"] == 512
    assert result.metadata["fallback_from_device"] == "cuda"
    assert provider.devices == ["cuda", "cpu"]


def test_sam3_mask_provider_falls_back_to_cpu_on_sam_initialize_failed():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    provider = CpuFallbackProvider(
        prompt="wireless mouse",
        device="cuda",
        resolution=512,
        cuda_error="SAM Initialize failed",
    )

    result = provider.get_mask(image, object_name="wireless mouse")

    assert result.metadata["device"] == "cpu"
    assert result.metadata["fallback_from_device"] == "cuda"


def test_mask_quality_rejects_tiny_mask_by_default():
    image_shape = (100, 100)
    mask = np.zeros(image_shape, dtype=bool)
    mask[10, 10] = True

    result = validate_mask_quality(mask, image_shape=image_shape)

    assert result.ok is False
    assert any("area fraction" in reason or "bbox" in reason for reason in result.reasons)


def test_mask_quality_warn_mode_does_not_block_result(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    result = PrecomputedMaskProvider(
        path,
        quality_config=MaskQualityConfig(severity="warn"),
    ).get_mask(image)

    assert result.source == "precomputed"
    quality = result.metadata["mask_quality"]
    assert quality["ok"] is True
    assert quality["reasons"]


def test_precomputed_mask_provider_blocks_bad_quality_mask(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    path = tmp_path / "mask.npy"
    np.save(path, mask)

    with np.testing.assert_raises(MaskProviderError):
        PrecomputedMaskProvider(path).get_mask(image)


def test_precomputed_mask_provider_applies_temporal_consistency(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first = np.zeros((100, 100), dtype=bool)
    first[10:30, 10:30] = True
    second = np.zeros((100, 100), dtype=bool)
    second[70:90, 70:90] = True
    path = tmp_path / "mask.npy"
    np.save(path, first)
    provider = PrecomputedMaskProvider(path, quality_config=MaskQualityConfig(temporal_min_iou=0.5))

    provider.get_mask(image)
    np.save(path, second)

    try:
        provider.get_mask(image)
    except MaskProviderError as exc:
        assert "temporal mask IoU" in str(exc)
    else:
        raise AssertionError("expected temporal quality failure")


class FailingSegmenter:
    def __init__(self, *, prompt: str) -> None:
        self.prompt = prompt

    def segment(self, image_rgb):
        raise ValueError("no candidate mask")


class OomSegmenter:
    def __init__(self, *, prompt: str, device: str, cuda_error: str) -> None:
        self.prompt = prompt
        self.device = device
        self.cuda_error = cuda_error

    def segment(self, image_rgb):
        if self.device == "cuda":
            raise RuntimeError(self.cuda_error)
        mask = np.zeros(np.asarray(image_rgb).shape[:2], dtype=bool)
        mask[5:15, 5:15] = True
        return FakeSelection(mask=mask)


class FakeSelection:
    def __init__(self, *, mask: np.ndarray) -> None:
        self.mask = mask
        self.score = 0.99
        self.area = int(mask.sum())
        self.index = 0


class CpuFallbackProvider(Sam3MaskProvider):
    def __init__(self, *, cuda_error: str = "CUDA out of memory. Tried to allocate 20.00 MiB.", **kwargs) -> None:
        super().__init__(**kwargs)
        self.devices = []
        self.cuda_error = cuda_error

    def _get_segmenter(self, prompt: str):
        self.devices.append(self.device)
        return OomSegmenter(prompt=prompt, device=self.device, cuda_error=self.cuda_error)
