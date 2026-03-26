"""Centralised device management — single source of truth for GPU/CPU selection."""

from __future__ import annotations

import sys

import torch

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_gpu_type: str = "cpu"       # "cuda" | "mps" | "cpu"
_gpu_available: bool = False
_use_gpu: bool = False


def _detect_gpu() -> tuple[str, bool]:
    """Detect the best available GPU backend.

    Returns:
        (device_type, is_available) — e.g. ("cuda", True) or ("mps", True).
    """
    if sys.platform == "darwin":
        # macOS: check for Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", True
        return "cpu", False

    # Windows / Linux: check for NVIDIA CUDA
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


def init() -> None:
    """Detect hardware and set platform-appropriate defaults.

    Call once at application startup, before creating the UI.
    """
    global _gpu_type, _gpu_available, _use_gpu

    _gpu_type, _gpu_available = _detect_gpu()

    if sys.platform == "darwin":
        _use_gpu = False              # Mac: off by default
    else:
        _use_gpu = _gpu_available     # Windows/Linux: on if available


# ---------------------------------------------------------------------------
# Public queries
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the device that should be used right now."""
    if _use_gpu and _gpu_available:
        return torch.device(_gpu_type)
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """Whether any GPU backend was detected at startup."""
    return _gpu_available


def is_gpu_enabled() -> bool:
    """Whether the user has opted in to GPU usage."""
    return _use_gpu


def gpu_type() -> str:
    """The GPU backend type ("cuda", "mps", or "cpu" if none)."""
    return _gpu_type


def default_use_gpu() -> bool:
    """The platform-appropriate default for the Use GPU checkbox."""
    return _use_gpu


# ---------------------------------------------------------------------------
# Runtime toggle
# ---------------------------------------------------------------------------
def set_use_gpu(enabled: bool) -> None:
    """Toggle GPU usage at runtime.

    Moves any cached segmentation models to the new device.
    """
    global _use_gpu
    _use_gpu = enabled and _gpu_available

    # Move cached segmentation models to the new device
    from semantic_grain.segmentation.backend import move_models_to_device
    move_models_to_device(get_device())
