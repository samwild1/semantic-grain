"""Orchestrates the full semantic grain pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from semantic_grain.config import (
    DEFAULT_PROFILES,
    GrainProfile,
    GRAIN_CATEGORIES,
    ADE20K_TO_GRAIN,
)
from semantic_grain.color.bw_conversion import rgb_to_mono
from semantic_grain.luminance.tone_curve import apply_tone_curve
from semantic_grain.luminance.zone_system import compute_zone_masks
from semantic_grain.grain.generator import generate_grain, modulate_by_luminance
from semantic_grain.segmentation.segformer_seg import segment_image
from semantic_grain.segmentation.class_mapping import map_segmentation_to_grain
from semantic_grain.segmentation.skin_detector import detect_skin
from semantic_grain.blending.mask_composer import soften_masks, compose_grain
from semantic_grain.io.saver import save_tiff16


def run_segmentation(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Run segmentation + skin detection → hard binary masks per category.

    This is the expensive step — results should be cached per image.
    """
    label_map = segment_image(image_rgb)
    masks = map_segmentation_to_grain(label_map)

    # Refine: split person mask into skin vs non-skin
    person_ids = [k for k, v in ADE20K_TO_GRAIN.items() if v == "skin"]
    person_mask = np.isin(label_map, person_ids)

    if person_mask.any():
        skin_mask = detect_skin(image_rgb, person_mask)
        # Skin pixels come from skin category; non-skin person pixels go to default
        masks["skin"] = skin_mask
        non_skin_person = person_mask & ~skin_mask
        masks["default"] = masks["default"] | non_skin_person

    return masks


def _scale_profiles(
    profiles: dict[str, GrainProfile],
    scale: float,
) -> dict[str, GrainProfile]:
    """Scale frequency parameters so grain at preview res matches full-res appearance.

    At lower resolution, the same center_freq produces visually coarser grain
    because frequencies are in cycles/pixel. Multiplying by the downscale ratio
    keeps the visual grain size consistent.
    """
    from dataclasses import replace
    scaled = {}
    for cat, p in profiles.items():
        scaled[cat] = replace(p,
            center_freq=p.center_freq * scale,
            bandwidth=p.bandwidth * scale,
        )
    return scaled


def apply_grain(
    image_rgb: np.ndarray,
    hard_masks: dict[str, np.ndarray],
    profiles: dict[str, GrainProfile] | None = None,
    global_strength: float = 1.0,
    toe: float = 0.3,
    shoulder: float = 0.3,
    bw_mix: tuple[float, float, float] = (0.35, 0.45, 0.20),
    convert_bw: bool = True,
    seed: int = 42,
    preview_size: int | None = 1024,
) -> np.ndarray:
    """Apply semantic grain pipeline (fast path — uses cached segmentation).

    Args:
        image_rgb: (H, W, 3) float32 [0, 1].
        hard_masks: Pre-computed binary masks per grain category.
        profiles: Per-category grain profiles. Uses defaults if None.
        global_strength: Overall grain multiplier.
        toe: Tone curve shadow parameter.
        shoulder: Tone curve highlight parameter.
        bw_mix: RGB channel mix for B&W conversion.
        convert_bw: If True, output is grayscale. If False, grain is applied
            to luminance and recombined with original color.
        seed: Random seed.
        preview_size: If set, process at reduced resolution for speed.
            Grain frequency parameters are scaled so the preview matches
            full-resolution output.

    Returns:
        If convert_bw: (H, W) float32 grayscale in [0, 1].
        If not convert_bw: (H, W, 3) float32 RGB in [0, 1].
    """
    import cv2

    if profiles is None:
        profiles = dict(DEFAULT_PROFILES)

    h_orig, w_orig = image_rgb.shape[:2]
    freq_scale = 1.0

    # Optionally downscale for preview
    if preview_size and max(h_orig, w_orig) > preview_size:
        freq_scale = preview_size / max(h_orig, w_orig)
        new_h = int(h_orig * freq_scale)
        new_w = int(w_orig * freq_scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_masks = {}
        for cat, mask in hard_masks.items():
            resized_masks[cat] = cv2.resize(
                mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        hard_masks = resized_masks

    # Scale grain frequencies to match full-res appearance at preview res
    if freq_scale < 1.0:
        profiles = _scale_profiles(profiles, freq_scale)

    # Compute luminance for grain modulation (always needed)
    luminance = rgb_to_mono(image_rgb, bw_mix)

    # Soften masks
    soft_masks = soften_masks(hard_masks)

    # Compose grain (modulated by luminance)
    grain = compose_grain(luminance, soft_masks, profiles, global_strength, seed)

    if convert_bw:
        result = luminance + grain
        result = apply_tone_curve(result, toe, shoulder)
        return np.clip(result, 0, 1).astype(np.float32)
    else:
        # Color mode: apply grain to each RGB channel, then tone curve per-channel
        result = image_rgb.copy()
        for c in range(3):
            ch = result[:, :, c] + grain
            ch = apply_tone_curve(ch, toe, shoulder)
            result[:, :, c] = ch
        return np.clip(result, 0, 1).astype(np.float32)


def generate_mask_overlay(
    image_rgb: np.ndarray,
    hard_masks: dict[str, np.ndarray],
    preview_size: int | None = 1024,
) -> np.ndarray:
    """Create a colored overlay showing segmentation regions.

    Returns (H, W, 3) float32 RGB overlay image.
    """
    import cv2

    h_orig, w_orig = image_rgb.shape[:2]

    if preview_size and max(h_orig, w_orig) > preview_size:
        scale = preview_size / max(h_orig, w_orig)
        new_h = int(h_orig * scale)
        new_w = int(w_orig * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_masks = {}
        for cat, mask in hard_masks.items():
            resized_masks[cat] = cv2.resize(
                mask.astype(np.uint8), (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        hard_masks = resized_masks

    h, w = image_rgb.shape[:2]

    # Category → color (RGB)
    colors = {
        "skin":       np.array([1.0, 0.7, 0.6]),
        "sky":        np.array([0.4, 0.6, 1.0]),
        "vegetation": np.array([0.3, 0.8, 0.3]),
        "water":      np.array([0.2, 0.5, 0.9]),
        "concrete":   np.array([0.7, 0.7, 0.6]),
        "default":    np.array([0.5, 0.5, 0.5]),
    }

    overlay = image_rgb.copy()
    for cat, mask in hard_masks.items():
        if cat in colors and mask.any():
            color = colors[cat]
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5

    return overlay


def export_full_resolution(
    image_rgb: np.ndarray,
    hard_masks: dict[str, np.ndarray],
    output_path: str | Path,
    profiles: dict[str, GrainProfile] | None = None,
    global_strength: float = 1.0,
    toe: float = 0.3,
    shoulder: float = 0.3,
    bw_mix: tuple[float, float, float] = (0.35, 0.45, 0.20),
    convert_bw: bool = True,
    seed: int = 42,
) -> Path:
    """Process at full resolution and save as 16-bit TIFF."""
    result = apply_grain(
        image_rgb, hard_masks, profiles,
        global_strength, toe, shoulder, bw_mix, convert_bw, seed,
        preview_size=None,  # full resolution
    )
    return save_tiff16(result, output_path)
