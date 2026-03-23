"""Image output: TIFF-16, PNG, JPG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_tiff16(image: np.ndarray, path: str | Path) -> Path:
    """Save float32 [0,1] grayscale or RGB as 16-bit TIFF."""
    path = _ensure_dir(Path(path))
    arr = np.clip(image, 0, 1)
    arr = (arr * 65535).astype(np.uint16)
    tifffile.imwrite(str(path), arr)
    return path


def save_jpg(image: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    """Save float32 [0,1] image as JPEG."""
    path = _ensure_dir(Path(path))
    arr = np.clip(image, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    img.save(str(path), quality=quality)
    return path


def save_png(image: np.ndarray, path: str | Path) -> Path:
    """Save float32 [0,1] image as PNG."""
    path = _ensure_dir(Path(path))
    arr = np.clip(image, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    img.save(str(path))
    return path
