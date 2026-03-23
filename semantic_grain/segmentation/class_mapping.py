"""Map ADE20K 150 classes → 8 grain categories."""

from __future__ import annotations

import numpy as np

from semantic_grain.config import ADE20K_TO_GRAIN, GRAIN_CATEGORIES


def map_segmentation_to_grain(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Convert an ADE20K label map to per-category binary masks.

    Args:
        label_map: (H, W) int array with ADE20K class indices (0–149).

    Returns:
        Dict mapping each grain category to a bool mask (H, W).
        Every pixel belongs to exactly one category.
    """
    masks: dict[str, np.ndarray] = {}
    assigned = np.zeros(label_map.shape, dtype=bool)

    for cat in GRAIN_CATEGORIES:
        if cat == "default":
            continue
        # Collect all ADE20K class IDs that map to this category
        class_ids = [k for k, v in ADE20K_TO_GRAIN.items() if v == cat]
        mask = np.isin(label_map, class_ids)
        masks[cat] = mask
        assigned |= mask

    # Everything not assigned goes to default
    masks["default"] = ~assigned

    return masks
