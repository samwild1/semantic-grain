"""Channel-weighted RGB → monochrome conversion."""

from __future__ import annotations

import numpy as np

from semantic_grain.config import DEFAULT_BW_MIX


def rgb_to_mono(
    image: np.ndarray,
    weights: tuple[float, float, float] = DEFAULT_BW_MIX,
) -> np.ndarray:
    """Convert float32 RGB [0,1] to grayscale using channel weights.

    Args:
        image: (H, W, 3) float32 in [0, 1].
        weights: (R, G, B) mixing weights — will be normalized to sum to 1.

    Returns:
        (H, W) float32 grayscale in [0, 1].
    """
    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    return np.dot(image, w)
