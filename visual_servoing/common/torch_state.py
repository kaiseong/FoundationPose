"""Helpers for isolating PyTorch global state changed by third-party models."""

from __future__ import annotations


def reset_torch_defaults_for_cpu_ops() -> None:
    """Keep CPU-side preprocessing safe after libraries switch defaults to CUDA."""

    try:
        import torch
    except Exception:
        return

    try:
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
    except Exception:
        try:
            torch.set_default_tensor_type(torch.FloatTensor)
        except Exception:
            return


def set_torch_defaults_for_cuda_ops() -> None:
    """Match FoundationPose's CUDA-default tensor assumptions during inference."""

    try:
        import torch
    except Exception:
        return

    try:
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)
    except Exception:
        try:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        except Exception:
            return
