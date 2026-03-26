"""Registry of available segmentation methods."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationMethod:
    """Describes a segmentation model and its requirements."""

    key: str
    display_name: str
    model_id: str
    backend: str  # "standard" | "mask2former"
    inference_size: int
    required_packages: tuple[str, ...]


METHODS: dict[str, SegmentationMethod] = {
    "segformer_b5": SegmentationMethod(
        key="segformer_b5",
        display_name="SegFormer-B5 (Best Quality)",
        model_id="nvidia/segformer-b5-finetuned-ade-640-640",
        backend="standard",
        inference_size=640,
        required_packages=(),
    ),
    "segformer_b2": SegmentationMethod(
        key="segformer_b2",
        display_name="SegFormer-B2 (Balanced)",
        model_id="nvidia/segformer-b2-finetuned-ade-512-512",
        backend="standard",
        inference_size=512,
        required_packages=(),
    ),
    "segformer_b0": SegmentationMethod(
        key="segformer_b0",
        display_name="SegFormer-B0 (Fastest)",
        model_id="nvidia/segformer-b0-finetuned-ade-512-512",
        backend="standard",
        inference_size=512,
        required_packages=(),
    ),
    "mask2former": SegmentationMethod(
        key="mask2former",
        display_name="Mask2Former Swin-L (SOTA)",
        model_id="facebook/mask2former-swin-large-ade-semantic",
        backend="mask2former",
        inference_size=640,
        required_packages=("timm",),
    ),
}

DEFAULT_METHOD = "segformer_b5"


def _is_package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def is_method_available(key: str) -> bool:
    """Check whether all required packages for a method are installed."""
    method = METHODS[key]
    return all(_is_package_available(p) for p in method.required_packages)


def get_available_methods() -> dict[str, SegmentationMethod]:
    """Return only methods whose dependencies are satisfied."""
    return {k: v for k, v in METHODS.items() if is_method_available(k)}


def get_dropdown_choices() -> list[tuple[str, str]]:
    """Build (display_name, key) pairs for Gradio dropdown.

    Unavailable methods are suffixed with their missing dependency.
    """
    choices = []
    for key, method in METHODS.items():
        if is_method_available(key):
            choices.append((method.display_name, key))
        else:
            missing = [p for p in method.required_packages if not _is_package_available(p)]
            suffix = f" (requires {', '.join(missing)})"
            choices.append((method.display_name + suffix, key))
    return choices


def check_and_install(key: str) -> tuple[bool, str]:
    """Install missing packages for a method. Returns (success, message)."""
    method = METHODS[key]
    missing = [p for p in method.required_packages if not _is_package_available(p)]
    if not missing:
        return True, "All dependencies available."

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True, f"Installed: {', '.join(missing)}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to install {', '.join(missing)}: {e}"
