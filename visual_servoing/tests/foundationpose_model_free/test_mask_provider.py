from __future__ import annotations

import base64
import json

import numpy as np

from visual_servoing.foundationpose_model_free.mask_provider import (
    ManualPolygonMaskProvider,
    MaskQualityConfig,
    MaskProviderError,
    PrecomputedMaskProvider,
    RemoteSegmentationMaskProvider,
    Sam3MaskProvider,
    validate_mask_quality,
)
from visual_servoing.visual_servo_protocol_v2 import (
    REQUEST_CONTENT_TYPE,
    decode_foundationpose_segmentation_request,
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


def test_remote_segmentation_mask_provider_posts_rgbd_and_decodes_png(monkeypatch):
    cv2 = _require_test_cv2()
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    depth = np.ones((20, 20), dtype=np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "ok": True,
                    "status": "segmented",
                    "mask": {"area": int(mask.sum() > 0), "confidence": 0.91},
                    "mask_png_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "mask_source": "sam3",
                    "mask_metadata": {"index": 0},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.mask_provider.urllib_request.urlopen",
        fake_urlopen,
    )
    provider = RemoteSegmentationMaskProvider(
        "192.168.0.3:8081",
        prompt="multimeter",
        device="cpu",
        confidence_threshold=0.2,
        resolution=512,
        timeout_s=7.0,
    )

    result = provider.get_mask(image, depth_m=depth, object_name="meter")

    assert result.source == "remote_segmentation"
    assert result.mask.dtype == bool
    assert result.mask.shape == image.shape[:2]
    assert int(result.mask.sum()) == 100
    assert result.confidence == 0.91
    assert result.metadata["server"] == "http://192.168.0.3:8081"
    assert result.metadata["mask_source"] == "sam3"
    assert result.metadata["remote_segmentation_ms"] >= 0.0
    request, timeout = calls[0]
    assert request.full_url == "http://192.168.0.3:8081/foundationpose/v2/segmentation"
    assert timeout == 7.0
    assert dict(request.header_items())["Content-type"] == REQUEST_CONTENT_TYPE
    decoded = decode_foundationpose_segmentation_request(request.data)
    assert decoded.prompt == "meter"
    assert decoded.rgb.shape == (20, 20, 3)
    assert decoded.depth_m.shape == (20, 20)
    assert decoded.mask_options["device"] == "cpu"
    assert decoded.mask_options["threshold"] == 0.2
    assert decoded.mask_options["resolution"] == 512


def test_remote_segmentation_mask_provider_requires_depth():
    provider = RemoteSegmentationMaskProvider("127.0.0.1:8081", prompt="meter")

    try:
        provider.get_mask(np.zeros((20, 20, 3), dtype=np.uint8))
    except MaskProviderError as exc:
        assert "requires depth_m" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


def test_remote_segmentation_mask_provider_surfaces_server_failure(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": False, "reason": "no usable mask"}).encode("utf-8")

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.mask_provider.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = RemoteSegmentationMaskProvider("127.0.0.1:8081", prompt="meter")

    try:
        provider.get_mask(np.zeros((20, 20, 3), dtype=np.uint8), depth_m=np.ones((20, 20), dtype=np.float32))
    except MaskProviderError as exc:
        assert "remote segmentation failed: no usable mask" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


def test_remote_segmentation_mask_provider_surfaces_request_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.mask_provider.urllib_request.urlopen",
        fake_urlopen,
    )
    provider = RemoteSegmentationMaskProvider("127.0.0.1:8081", prompt="meter")

    try:
        provider.get_mask(np.zeros((20, 20, 3), dtype=np.uint8), depth_m=np.ones((20, 20), dtype=np.float32))
    except MaskProviderError as exc:
        assert "remote segmentation request failed" in str(exc)
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


def test_remote_segmentation_mask_provider_rejects_missing_mask_png(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True, "status": "segmented"}).encode("utf-8")

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.mask_provider.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = RemoteSegmentationMaskProvider("127.0.0.1:8081", prompt="meter")

    try:
        provider.get_mask(np.zeros((20, 20, 3), dtype=np.uint8), depth_m=np.ones((20, 20), dtype=np.float32))
    except MaskProviderError as exc:
        assert "missing mask_png_b64" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


def test_remote_segmentation_mask_provider_rejects_bad_mask_shape(monkeypatch):
    cv2 = _require_test_cv2()
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "ok": True,
                    "mask_png_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "visual_servoing.foundationpose_model_free.mask_provider.urllib_request.urlopen",
        lambda request, timeout: FakeResponse(),
    )
    provider = RemoteSegmentationMaskProvider("127.0.0.1:8081", prompt="meter")

    try:
        provider.get_mask(np.zeros((20, 20, 3), dtype=np.uint8), depth_m=np.ones((20, 20), dtype=np.float32))
    except MaskProviderError as exc:
        assert "does not match image shape" in str(exc)
    else:
        raise AssertionError("expected MaskProviderError")


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


def test_sam3_mask_provider_retries_dtype_mismatch_without_autocast():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    provider = DtypeFallbackProvider(prompt="wireless mouse", device="cuda", autocast_dtype="bfloat16")

    result = provider.get_mask(image, object_name="wireless mouse")

    assert result.source == "sam3"
    assert result.metadata["device"] == "cuda"
    assert result.metadata["autocast_dtype"] == "float32"
    assert result.metadata["fallback_action"] == "disabled_cuda_autocast"
    assert provider.autocast_history == ["bfloat16", "float32"]


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


class DtypeMismatchSegmenter:
    def __init__(self, *, prompt: str, autocast_dtype: str | None) -> None:
        self.prompt = prompt
        self.autocast_dtype = autocast_dtype

    def segment(self, image_rgb):
        if str(self.autocast_dtype).lower() in {"bfloat16", "bf16"}:
            raise RuntimeError("mat1 and mat2 must have the same dtype, but got BFloat16 and Float")
        mask = np.zeros(np.asarray(image_rgb).shape[:2], dtype=bool)
        mask[5:15, 5:15] = True
        return FakeSelection(mask=mask)


class DtypeFallbackProvider(Sam3MaskProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.autocast_history = []

    def _get_segmenter(self, prompt: str):
        self.autocast_history.append(self.autocast_dtype)
        return DtypeMismatchSegmenter(prompt=prompt, autocast_dtype=self.autocast_dtype)


def _require_test_cv2():
    import pytest

    return pytest.importorskip("cv2")
