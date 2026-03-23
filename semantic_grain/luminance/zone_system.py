"""Luminance zone soft masks: shadow, midtone, highlight."""

from __future__ import annotations

import numpy as np

from semantic_grain.config import ZONE_SHADOW_MID, ZONE_MID_HIGH, ZONE_SOFTNESS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_zone_masks(
    luminance: np.ndarray,
    shadow_mid: float = ZONE_SHADOW_MID,
    mid_high: float = ZONE_MID_HIGH,
    softness: float = ZONE_SOFTNESS,
) -> dict[str, np.ndarray]:
    """Compute soft zone masks from a luminance map.

    Returns dict with keys 'shadow', 'midtone', 'highlight',
    each a float32 array in [0, 1] that sum to ~1.0 at every pixel.
    """
    k = 1.0 / max(softness, 1e-6)

    # shadow: high where luminance is below shadow_mid
    shadow = 1.0 - _sigmoid((luminance - shadow_mid) * k)

    # highlight: high where luminance is above mid_high
    highlight = _sigmoid((luminance - mid_high) * k)

    # midtone: what's left
    midtone = 1.0 - shadow - highlight
    midtone = np.clip(midtone, 0, 1)

    return {
        "shadow": shadow.astype(np.float32),
        "midtone": midtone.astype(np.float32),
        "highlight": highlight.astype(np.float32),
    }
