"""Image loading: JPG/PNG/TIFF → float32 [0,1] with EXIF orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path) -> np.ndarray:
    """Load an image and return float32 RGB array in [0, 1].

    Handles EXIF orientation so rotated JPEGs display correctly.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0
