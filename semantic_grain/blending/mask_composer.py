"""Compose semantic + luminance masks and blend per-region grain layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from semantic_grain.config import GrainProfile, GRAIN_CATEGORIES
from semantic_grain.grain.generator import generate_grain, modulate_by_luminance

if TYPE_CHECKING:
    from semantic_grain.cache import ProcessingCache


def soften_masks(
    hard_masks: dict[str, np.ndarray],
    sigma: float = 21.0,
) -> dict[str, np.ndarray]:
    """Gaussian-blur binary masks to create soft transitions.

    Masks are normalized so they sum to 1.0 at every pixel.
    """
    soft: dict[str, np.ndarray] = {}
    for cat, mask in hard_masks.items():
        m = mask.astype(np.float32)
        ksize = int(sigma * 4) | 1  # must be odd
        m = cv2.GaussianBlur(m, (ksize, ksize), sigma)
        soft[cat] = m

    # Normalize to sum to 1
    total = sum(soft.values())
    total = np.maximum(total, 1e-8)
    for cat in soft:
        soft[cat] = soft[cat] / total

    return soft


def compose_grain(
    luminance: np.ndarray,
    soft_masks: dict[str, np.ndarray],
    profiles: dict[str, GrainProfile],
    global_strength: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate per-region grain, modulate by luminance, blend with soft masks.

    Args:
        luminance: (H, W) float32 in [0, 1] — the B&W image.
        soft_masks: Per-category soft float masks that sum to 1.
        profiles: Per-category grain profiles.
        global_strength: Overall grain multiplier.
        seed: Base random seed.

    Returns:
        (H, W) float32 blended grain layer (centered around 0).
    """
    h, w = luminance.shape
    blended = np.zeros((h, w), dtype=np.float32)

    for i, cat in enumerate(GRAIN_CATEGORIES):
        if cat not in soft_masks:
            continue

        mask = soft_masks[cat]
        if mask.max() < 1e-6:
            continue

        profile = profiles.get(cat, profiles["default"])

        # Generate unique grain per region
        grain = generate_grain((h, w), profile, seed=seed + i)

        # Modulate by luminance
        grain = modulate_by_luminance(
            grain, luminance,
            profile.shadow_boost, profile.highlight_rolloff,
        )

        blended += mask * grain * global_strength

    return blended


def compose_grain_cached(
    luminance: np.ndarray,
    soft_masks: dict[str, np.ndarray],
    profiles: dict[str, GrainProfile],
    global_strength: float,
    seed: int,
    cache: ProcessingCache,
    recompute_gen: set[str] | None,
    recompute_mod: set[str] | None,
) -> np.ndarray:
    """Cache-aware grain composition.

    Only regenerates FFT grains for regions in *recompute_gen*.
    Only re-modulates for regions in *recompute_mod*.
    Always re-blends (cheap weighted sum).

    Args:
        recompute_gen: Categories needing FFT regeneration, or None for all.
        recompute_mod: Categories needing luminance re-modulation, or None for all.
    """
    h, w = luminance.shape
    blended = np.zeros((h, w), dtype=np.float32)

    for i, cat in enumerate(GRAIN_CATEGORIES):
        if cat not in soft_masks:
            continue

        mask = soft_masks[cat]
        if mask.max() < 1e-6:
            continue

        profile = profiles.get(cat, profiles["default"])

        # Stage 3: generate raw grain (FFT) — use cache when valid
        need_gen = (
            recompute_gen is None
            or cat in recompute_gen
            or cat not in cache.raw_grains
        )
        if need_gen:
            cache.raw_grains[cat] = generate_grain((h, w), profile, seed=seed + i)

        # Stage 4: modulate by luminance — use cache when valid
        need_mod = (
            need_gen
            or recompute_mod is None
            or cat in recompute_mod
            or cat not in cache.mod_grains
        )
        if need_mod:
            cache.mod_grains[cat] = modulate_by_luminance(
                cache.raw_grains[cat], luminance,
                profile.shadow_boost, profile.highlight_rolloff,
            )

        blended += mask * cache.mod_grains[cat] * global_strength

    cache.blended_grain = blended
    return blended
