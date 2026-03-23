"""Parametric film S-curve (toe compression + shoulder rolloff)."""

from __future__ import annotations

import numpy as np


def apply_tone_curve(
    image: np.ndarray,
    toe: float = 0.3,
    shoulder: float = 0.3,
) -> np.ndarray:
    """Apply an S-shaped tone curve to a grayscale image.

    Uses a smooth sigmoid-like remapping:
      - `toe` controls shadow compression (higher = more compressed blacks)
      - `shoulder` controls highlight rolloff (higher = softer highlights)

    Args:
        image: (H, W) float32 in [0, 1].
        toe: Shadow compression strength [0, 1].
        shoulder: Highlight rolloff strength [0, 1].

    Returns:
        (H, W) float32 in [0, 1].
    """
    x = np.clip(image, 0, 1)

    # Lift shadows (toe): raise blacks slightly then re-expand
    if toe > 0:
        # Apply a power curve < 1 to shadows, scaled by toe
        shadow_mask = 1.0 - x  # stronger effect in darks
        lift = toe * 0.15 * shadow_mask * shadow_mask
        x = x + lift

    # Compress highlights (shoulder): pull bright values down
    if shoulder > 0:
        highlight_mask = x * x  # stronger effect in brights
        compress = shoulder * 0.15 * highlight_mask
        x = x - compress

    # Final sigmoid nudge for S-shape
    mid_strength = (toe + shoulder) * 0.5
    if mid_strength > 0:
        # Soft sigmoid centered at 0.5
        x = x + mid_strength * 0.1 * (4.0 * (x - 0.5) ** 3)

    return np.clip(x, 0, 1)
