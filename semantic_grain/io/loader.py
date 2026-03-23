"""Image loading: RAW, HEIF, JPG/PNG/TIFF → float32 [0,1] with EXIF orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Supported extension sets (single source of truth)
# ---------------------------------------------------------------------------
RAW_EXTENSIONS = frozenset({
    ".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".orf", ".rw2", ".pef", ".srw", ".3fr", ".iiq",
    ".x3f", ".mrw", ".erf", ".kdc",
})

HEIF_EXTENSIONS = frozenset({".heif", ".heic", ".avif"})

PILLOW_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".webp", ".gif",
})

ALL_SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | PILLOW_EXTENSIONS

# ---------------------------------------------------------------------------
# Register pillow-heif as a Pillow plugin (if installed)
# ---------------------------------------------------------------------------
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# RAW loader (via rawpy / LibRaw)
# ---------------------------------------------------------------------------
def _load_raw(path: Path) -> np.ndarray:
    """Load a camera RAW file and return float32 RGB in [0, 1]."""
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            f"Cannot open RAW file '{path.name}': rawpy is not installed.\n"
            "Install it with:  pip install rawpy"
        )

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            no_auto_bright=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    return rgb.astype(np.float32) / 65535.0


# ---------------------------------------------------------------------------
# Pillow loader (JPG, PNG, TIFF, HEIF, WebP, …)
# ---------------------------------------------------------------------------
def _load_pillow(path: Path) -> np.ndarray:
    """Load an image via Pillow and return float32 RGB in [0, 1]."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Load an image and return float32 RGB array in [0, 1].

    Dispatches based on file extension:
    - Camera RAW formats → rawpy (16-bit, camera white balance)
    - HEIF/HEIC/AVIF    → Pillow with pillow-heif plugin
    - Everything else    → Pillow (handles EXIF orientation)
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in RAW_EXTENSIONS:
        return _load_raw(path)

    if ext in HEIF_EXTENSIONS:
        try:
            from pillow_heif import register_heif_opener  # noqa: F811
        except ImportError:
            raise ImportError(
                f"Cannot open HEIF file '{path.name}': pillow-heif is not installed.\n"
                "Install it with:  pip install pillow-heif"
            )

    return _load_pillow(path)
