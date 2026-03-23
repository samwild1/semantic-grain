"""HSV-based skin detection within person masks."""

from __future__ import annotations

import cv2
import numpy as np


def detect_skin(
    image_rgb: np.ndarray,
    person_mask: np.ndarray,
) -> np.ndarray:
    """Detect skin pixels within a person mask using HSV thresholds.

    Args:
        image_rgb: (H, W, 3) float32 in [0, 1].
        person_mask: (H, W) bool mask of person regions.

    Returns:
        (H, W) bool mask — True where skin is detected within person regions.
    """
    # Convert to uint8 BGR for OpenCV
    bgr = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad skin tone range in HSV
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)
    skin1 = cv2.inRange(hsv, lower, upper)

    # Wrap-around for reddish hues
    lower2 = np.array([165, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    skin2 = cv2.inRange(hsv, lower2, upper2)

    skin_hsv = (skin1 | skin2).astype(bool)

    # Intersect with person mask
    return skin_hsv & person_mask
