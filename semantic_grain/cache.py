"""Tiered processing cache for interactive preview rendering.

Caches intermediate pipeline results so only stages affected by parameter
changes are recomputed.  All cached data is at full resolution — no quality
compromise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_grain.config import GrainProfile, GRAIN_CATEGORIES


# ---------------------------------------------------------------------------
# Immutable parameter snapshot (for diff detection)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionParams:
    """Hashable snapshot of one region's grain profile."""
    center_freq: float
    bandwidth: float
    spectral_slope: float
    amplitude: float
    shadow_boost: float
    highlight_rolloff: float

    @classmethod
    def from_profile(cls, p: GrainProfile) -> RegionParams:
        return cls(
            center_freq=p.center_freq,
            bandwidth=p.bandwidth,
            spectral_slope=p.spectral_slope,
            amplitude=p.amplitude,
            shadow_boost=p.shadow_boost,
            highlight_rolloff=p.highlight_rolloff,
        )


@dataclass(frozen=True)
class PipelineParams:
    """Immutable snapshot of all pipeline parameters."""
    strength: float
    toe: float
    shoulder: float
    bw_r: float
    bw_g: float
    bw_b: float
    convert_bw: bool
    # Per-region profiles stored as a tuple of (category, RegionParams)
    region_params: tuple[tuple[str, RegionParams], ...]

    @classmethod
    def build(
        cls,
        strength: float,
        toe: float,
        shoulder: float,
        bw_mix: tuple[float, float, float],
        convert_bw: bool,
        profiles: dict[str, GrainProfile],
    ) -> PipelineParams:
        region_params = tuple(
            (cat, RegionParams.from_profile(profiles[cat]))
            for cat in GRAIN_CATEGORIES
            if cat in profiles
        )
        return cls(
            strength=strength,
            toe=toe,
            shoulder=shoulder,
            bw_r=bw_mix[0],
            bw_g=bw_mix[1],
            bw_b=bw_mix[2],
            convert_bw=convert_bw,
            region_params=region_params,
        )

    def _region_dict(self) -> dict[str, RegionParams]:
        return dict(self.region_params)


# ---------------------------------------------------------------------------
# Processing cache
# ---------------------------------------------------------------------------
@dataclass
class ProcessingCache:
    """Holds cached intermediate results at full resolution."""

    # Image identity
    image_path: str | None = None
    seg_method: str | None = None
    image: np.ndarray | None = None
    masks: dict[str, np.ndarray] | None = None

    # Stage 1: soft masks (depends on hard masks only)
    soft_masks: dict[str, np.ndarray] | None = None

    # Stage 2: luminance (depends on image + bw_mix)
    luminance: np.ndarray | None = None

    # Stage 3: raw FFT grain per region (depends on shape + profile FFT params + seed)
    raw_grains: dict[str, np.ndarray] = field(default_factory=dict)

    # Stage 4: luminance-modulated grain per region
    mod_grains: dict[str, np.ndarray] = field(default_factory=dict)

    # Stage 5: blended grain layer
    blended_grain: np.ndarray | None = None

    # Stage 6: final result (before display downscale)
    result: np.ndarray | None = None

    # Previous parameters (for diff detection)
    prev_params: PipelineParams | None = None

    def invalidate_all(self) -> None:
        """Clear everything (new image loaded)."""
        self.soft_masks = None
        self.luminance = None
        self.raw_grains.clear()
        self.mod_grains.clear()
        self.blended_grain = None
        self.result = None
        self.prev_params = None

    def invalidate_from_stage(
        self, stage: int, categories: set[str] | None = None,
    ) -> None:
        """Invalidate a stage and all downstream stages.

        For stages 3-4, ``categories`` selects which regions to invalidate.
        ``None`` means all regions.
        """
        if stage <= 1:
            self.soft_masks = None
        if stage <= 2:
            self.luminance = None
            # Luminance change invalidates all modulated grains
            self.mod_grains.clear()
        if stage <= 3:
            if categories is None:
                self.raw_grains.clear()
                self.mod_grains.clear()
            else:
                for cat in categories:
                    self.raw_grains.pop(cat, None)
                    self.mod_grains.pop(cat, None)
        if stage <= 4:
            if categories is not None and stage == 4:
                for cat in categories:
                    self.mod_grains.pop(cat, None)
            # (If stage < 4, mod_grains already cleared above)
        if stage <= 5:
            self.blended_grain = None
        if stage <= 6:
            self.result = None


# ---------------------------------------------------------------------------
# Diff detection
# ---------------------------------------------------------------------------

# FFT-dependent profile fields (regeneration required)
_FFT_FIELDS = ("center_freq", "bandwidth", "spectral_slope", "amplitude")
# Modulation-only fields (no FFT, just re-modulate)
_MOD_FIELDS = ("shadow_boost", "highlight_rolloff")


def determine_invalidation(
    old: PipelineParams | None,
    new: PipelineParams,
) -> tuple[int, set[str] | None]:
    """Compare parameter snapshots and return the earliest invalidated stage.

    Returns:
        (stage, categories) where:
        - stage: earliest stage needing recomputation (1-6), or 7 if nothing changed.
        - categories: set of affected region names for stage 3/4 changes,
          or None to mean "all regions".
    """
    if old is None:
        return 1, None

    earliest = 7
    affected_cats: set[str] = set()

    # Stage 6: tone curve / convert_bw
    if old.toe != new.toe or old.shoulder != new.shoulder or old.convert_bw != new.convert_bw:
        earliest = min(earliest, 6)

    # Stage 5: strength
    if old.strength != new.strength:
        earliest = min(earliest, 5)

    # Stage 2: BW mix (cascades through 2→4→5→6)
    if (old.bw_r, old.bw_g, old.bw_b) != (new.bw_r, new.bw_g, new.bw_b):
        earliest = min(earliest, 2)

    # Per-region checks (stages 3-4)
    old_regions = old._region_dict()
    new_regions = new._region_dict()

    for cat in GRAIN_CATEGORIES:
        op = old_regions.get(cat)
        np_ = new_regions.get(cat)
        if op is None or np_ is None:
            if op != np_:
                earliest = min(earliest, 3)
                affected_cats.add(cat)
            continue

        # Check FFT params
        fft_changed = any(
            getattr(op, f) != getattr(np_, f) for f in _FFT_FIELDS
        )
        if fft_changed:
            earliest = min(earliest, 3)
            affected_cats.add(cat)
            continue

        # Check modulation params
        mod_changed = any(
            getattr(op, f) != getattr(np_, f) for f in _MOD_FIELDS
        )
        if mod_changed:
            earliest = min(earliest, 4)
            affected_cats.add(cat)

    # If BW mix changed, all regions need modulation refresh
    if earliest <= 2:
        return earliest, None

    return earliest, affected_cats if affected_cats else None
