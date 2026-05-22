"""Lazy SAM3 wrapper for selecting one object mask for point-pose scripts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from visual_servoing.common.torch_state import reset_torch_defaults_for_cpu_ops


@dataclass(frozen=True)
class MaskSelection:
    mask: np.ndarray
    index: int
    score: float
    area: int
    box_xyxy: list[float] | None = None


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "detach"):
        value = value.detach()
        if str(getattr(value, "dtype", "")).endswith("bfloat16"):
            value = value.float()
        value = value.cpu()
    return np.asarray(value)


def normalize_masks(value: Any, height: int, width: int) -> np.ndarray:
    masks = to_numpy(value).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim != 3:
        masks = np.zeros((0, height, width), dtype=bool)
    if masks.shape[-2:] != (height, width):
        raise ValueError(f"mask shape {masks.shape[-2:]} does not match {(height, width)}")
    return masks


def select_single_mask(
    masks: Any,
    scores: Any | None = None,
    boxes: Any | None = None,
    *,
    min_area: int = 16,
) -> MaskSelection:
    masks_np = to_numpy(masks).astype(bool)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks_np.shape}")

    scores_np = to_numpy(scores).astype(np.float64).reshape(-1) if scores is not None else np.array([])
    boxes_np = to_numpy(boxes).astype(np.float64) if boxes is not None else np.array([])

    candidates: list[tuple[float, int, int]] = []
    for index, mask in enumerate(masks_np):
        area = int(mask.sum())
        if area < min_area:
            continue
        score = float(scores_np[index]) if index < scores_np.shape[0] else 0.0
        candidates.append((score, area, index))
    if not candidates:
        raise ValueError("No usable object mask was produced.")

    score, area, index = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    box = boxes_np[index].tolist() if boxes_np.ndim == 2 and index < boxes_np.shape[0] else None
    return MaskSelection(mask=masks_np[index].copy(), index=index, score=score, area=area, box_xyxy=box)


def load_mask(path: str | Path, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    cv2 = _require_cv2()
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


class Sam3PhoneSegmenter:
    def __init__(
        self,
        *,
        prompt: str = "phone",
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        resolution: int = 1008,
        autocast_dtype: str | None = "float32",
    ) -> None:
        self.prompt = prompt
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.autocast_dtype = autocast_dtype
        self._processor = None
        self._last_model_load_ms: float | None = None

    def segment(self, image_rgb: np.ndarray) -> MaskSelection:
        reset_torch_defaults_for_cpu_ops()
        processor = self._get_processor()
        torch = _require_torch()
        height, width = image_rgb.shape[:2]
        pil_image = _to_pil_image(image_rgb)
        with torch.inference_mode(), _autocast_context(torch, self.device, self.autocast_dtype):
            state = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=state, prompt=self.prompt)
        masks = normalize_masks(output.get("masks"), height, width)
        return select_single_mask(masks, output.get("scores"), output.get("boxes"))

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        start = time.perf_counter()
        try:
            _prefer_local_sam3_package()
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "SAM3 is required for live segmentation. Use --mask for offline mode."
            ) from exc
        _patch_sam3_float32_fused_mlp()
        model = build_sam3_image_model(device=self.device)
        kwargs = {
            "resolution": self.resolution,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
        }
        try:
            self._processor = Sam3Processor(model, **kwargs)
        except TypeError:
            self._processor = Sam3Processor(model)
        self._last_model_load_ms = (time.perf_counter() - start) * 1000.0
        return self._processor

    def pop_last_model_load_ms(self) -> float | None:
        value = self._last_model_load_ms
        self._last_model_load_ms = None
        return value


def _to_pil_image(image_rgb: np.ndarray):
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Pillow is required for SAM3 image conversion.") from exc
    return Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")


def _prefer_local_sam3_package() -> None:
    local_sam3_parent = Path("/home/kgs/sam3")
    if local_sam3_parent.exists():
        local_path = str(local_sam3_parent)
        if local_path in sys.path:
            sys.path.remove(local_path)
        sys.path.insert(0, local_path)

    loaded = sys.modules.get("sam3")
    if loaded is not None and getattr(loaded, "__file__", None) is None:
        for name in [key for key in sys.modules if key == "sam3" or key.startswith("sam3.")]:
            sys.modules.pop(name, None)


def _patch_sam3_float32_fused_mlp() -> None:
    """Keep SAM3's ViT MLP in float32 on GPUs where its BF16 fused op mismatches weights."""

    try:
        import torch
        import sam3.model.vitdet as vitdet
        import sam3.perflib.fused as fused
    except Exception:
        return

    if getattr(fused.addmm_act, "_visual_servoing_float32_safe", False):
        return

    def addmm_act_float32_safe(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        input_dtype = mat1.dtype
        weight_dtype = linear.weight.dtype
        x = mat1.to(weight_dtype) if mat1.dtype != weight_dtype else mat1
        y = linear(x)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            y = torch.nn.functional.relu(y)
        elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            y = torch.nn.functional.gelu(y)
        else:
            raise ValueError(f"Unexpected activation {activation}")
        if input_dtype.is_floating_point and input_dtype != torch.bfloat16 and y.dtype != input_dtype:
            y = y.to(input_dtype)
        return y

    addmm_act_float32_safe._visual_servoing_float32_safe = True
    fused.addmm_act = addmm_act_float32_safe
    vitdet.addmm_act = addmm_act_float32_safe


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyTorch is required for SAM3 segmentation.") from exc
    return torch


def _autocast_context(torch, device: str, autocast_dtype: str | None = "float32"):
    device_type = str(device).split(":", 1)[0]
    dtype_name = str(autocast_dtype or "float32").lower()
    if dtype_name in {"none", "off", "false", "0", "float", "float32", "fp32"}:
        return nullcontext()
    if device_type == "cuda" and torch.cuda.is_available() and dtype_name in {"bfloat16", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cuda" and torch.cuda.is_available() and dtype_name in {"float16", "fp16", "half"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required for mask loading.") from exc
    return cv2
