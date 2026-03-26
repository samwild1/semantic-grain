"""Configuration: grain profiles, class mappings, zone thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Grain categories
# ---------------------------------------------------------------------------
GRAIN_CATEGORIES = ("skin", "sky", "vegetation", "water", "concrete", "default")


# ---------------------------------------------------------------------------
# GrainProfile
# ---------------------------------------------------------------------------
@dataclass
class GrainProfile:
    """Parameters controlling spectral grain synthesis for one region."""

    center_freq: float = 0.12
    bandwidth: float = 0.06
    spectral_slope: float = 0.3
    amplitude: float = 0.04
    shadow_boost: float = 1.3
    highlight_rolloff: float = 0.7


# ---------------------------------------------------------------------------
# Default per-region profiles (Delta 400 inspired)
# ---------------------------------------------------------------------------
DEFAULT_PROFILES: dict[str, GrainProfile] = {
    "skin": GrainProfile(
        center_freq=0.06, bandwidth=0.035, spectral_slope=0.45,
        amplitude=0.035, shadow_boost=1.25, highlight_rolloff=0.75,
    ),
    "sky": GrainProfile(
        center_freq=0.04, bandwidth=0.025, spectral_slope=0.60,
        amplitude=0.030, shadow_boost=1.05, highlight_rolloff=0.50,
    ),
    "concrete": GrainProfile(
        center_freq=0.08, bandwidth=0.04, spectral_slope=0.35,
        amplitude=0.045, shadow_boost=1.40, highlight_rolloff=0.70,
    ),
    "vegetation": GrainProfile(
        center_freq=0.07, bandwidth=0.035, spectral_slope=0.40,
        amplitude=0.040, shadow_boost=1.30, highlight_rolloff=0.70,
    ),
    "water": GrainProfile(
        center_freq=0.04, bandwidth=0.025, spectral_slope=0.55,
        amplitude=0.030, shadow_boost=1.10, highlight_rolloff=0.55,
    ),
    "default": GrainProfile(
        center_freq=0.06, bandwidth=0.035, spectral_slope=0.45,
        amplitude=0.040, shadow_boost=1.25, highlight_rolloff=0.70,
    ),
}


# ---------------------------------------------------------------------------
# Film stock grain types — physics-based presets (multi-octave synthesis)
# ---------------------------------------------------------------------------
# With multi-octave synthesis, parameters have these meanings:
#   center_freq    — dominant (coarsest) grain scale; higher octaves added
#                    automatically.  Lower = larger clumps.
#   bandwidth      — per-octave passband width.  Narrow = sharp grain edges,
#                    wide = softer/fuzzier grain.
#   spectral_slope — inter-octave falloff (1/2^(i*α)).  0 = all octaves
#                    equal (modern T-grain).  0.5+ = large clumps dominate
#                    (classic cubic grain).
#   amplitude      — NORMALISED to a uniform reference across all types so
#                    the Strength slider is the sole intensity control.
#   shadow_boost   — grain multiplier in shadows (film: heavier in shadows).
#   highlight_rolloff — grain multiplier in highlights (film: fades in highlights).
#
# Reference amplitude per region (constant across all grain types):
#   skin=0.035, sky=0.030, vegetation=0.040, water=0.030,
#   concrete=0.045, default=0.040

GRAIN_TYPES: dict[str, dict[str, GrainProfile]] = {
    # Sentinel: "Custom" means keep current slider values
    "Custom": {},

    # Classic cubic grain — photojournalism standard, strong organic clumping.
    # Large clumps dominate (high spectral_slope), wide bandwidth.
    "Kodak Tri-X 400": {
        "skin":       GrainProfile(0.05, 0.04, 0.60, 0.035, 1.35, 0.65),
        "sky":        GrainProfile(0.03, 0.03, 0.75, 0.030, 1.15, 0.45),
        "vegetation": GrainProfile(0.06, 0.04, 0.55, 0.040, 1.40, 0.65),
        "water":      GrainProfile(0.04, 0.03, 0.65, 0.030, 1.15, 0.50),
        "concrete":   GrainProfile(0.07, 0.05, 0.50, 0.045, 1.50, 0.65),
        "default":    GrainProfile(0.05, 0.04, 0.60, 0.040, 1.35, 0.65),
    },

    # Cubic grain, slightly more uniform than Tri-X — smoother transitions,
    # moderate spectral slope, slightly tighter bandwidth.
    "Ilford HP5 Plus 400": {
        "skin":       GrainProfile(0.06, 0.035, 0.50, 0.035, 1.30, 0.70),
        "sky":        GrainProfile(0.04, 0.025, 0.65, 0.030, 1.10, 0.50),
        "vegetation": GrainProfile(0.07, 0.035, 0.45, 0.040, 1.35, 0.70),
        "water":      GrainProfile(0.04, 0.03, 0.55, 0.030, 1.10, 0.55),
        "concrete":   GrainProfile(0.08, 0.04, 0.40, 0.045, 1.40, 0.70),
        "default":    GrainProfile(0.06, 0.035, 0.50, 0.040, 1.30, 0.70),
    },

    # T-grain (tabular) — very fine, very uniform, modern clean look.
    # Flat spectral slope = all octaves visible, narrow bandwidth.
    "Kodak T-Max 100": {
        "skin":       GrainProfile(0.12, 0.02, 0.15, 0.035, 1.10, 0.85),
        "sky":        GrainProfile(0.10, 0.015, 0.20, 0.030, 1.00, 0.80),
        "vegetation": GrainProfile(0.11, 0.02, 0.15, 0.040, 1.12, 0.82),
        "water":      GrainProfile(0.10, 0.015, 0.18, 0.030, 1.00, 0.80),
        "concrete":   GrainProfile(0.14, 0.025, 0.12, 0.045, 1.15, 0.85),
        "default":    GrainProfile(0.12, 0.02, 0.15, 0.040, 1.10, 0.82),
    },

    # High-speed tabular — dramatic large grain, heavy shadows.
    # Very low center_freq (huge clumps), high spectral slope.
    "Ilford Delta 3200": {
        "skin":       GrainProfile(0.025, 0.03, 0.80, 0.035, 1.65, 0.45),
        "sky":        GrainProfile(0.015, 0.02, 0.90, 0.030, 1.40, 0.30),
        "vegetation": GrainProfile(0.03, 0.03, 0.75, 0.040, 1.70, 0.50),
        "water":      GrainProfile(0.02, 0.02, 0.85, 0.030, 1.40, 0.35),
        "concrete":   GrainProfile(0.035, 0.04, 0.70, 0.045, 1.80, 0.50),
        "default":    GrainProfile(0.025, 0.03, 0.80, 0.040, 1.65, 0.45),
    },

    # Ultra-fine crystalline — minimal clumping, Japanese precision.
    # Very flat spectral slope, narrow bandwidth, high center_freq.
    "Fuji Neopan Acros 100 II": {
        "skin":       GrainProfile(0.14, 0.015, 0.08, 0.035, 1.05, 0.90),
        "sky":        GrainProfile(0.12, 0.012, 0.10, 0.030, 1.00, 0.85),
        "vegetation": GrainProfile(0.13, 0.015, 0.08, 0.040, 1.08, 0.88),
        "water":      GrainProfile(0.12, 0.012, 0.10, 0.030, 1.00, 0.85),
        "concrete":   GrainProfile(0.16, 0.018, 0.06, 0.045, 1.10, 0.90),
        "default":    GrainProfile(0.14, 0.015, 0.08, 0.040, 1.05, 0.88),
    },

    # Colour negative T-grain — smooth, fine, minimal clumping.
    # Similar to T-Max but slightly softer spectral slope.
    "Kodak Portra 400": {
        "skin":       GrainProfile(0.11, 0.02, 0.18, 0.035, 1.08, 0.88),
        "sky":        GrainProfile(0.09, 0.015, 0.22, 0.030, 1.00, 0.82),
        "vegetation": GrainProfile(0.10, 0.02, 0.18, 0.040, 1.10, 0.85),
        "water":      GrainProfile(0.09, 0.015, 0.20, 0.030, 1.02, 0.82),
        "concrete":   GrainProfile(0.12, 0.025, 0.15, 0.045, 1.12, 0.88),
        "default":    GrainProfile(0.10, 0.02, 0.18, 0.040, 1.08, 0.85),
    },

    # Classic medium-speed cubic — balanced workhorse.
    # Moderate spectral slope, moderate center_freq.
    "Ilford FP4 Plus 125": {
        "skin":       GrainProfile(0.08, 0.03, 0.35, 0.035, 1.18, 0.78),
        "sky":        GrainProfile(0.06, 0.02, 0.45, 0.030, 1.05, 0.60),
        "vegetation": GrainProfile(0.07, 0.03, 0.38, 0.040, 1.22, 0.75),
        "water":      GrainProfile(0.06, 0.025, 0.42, 0.030, 1.08, 0.62),
        "concrete":   GrainProfile(0.09, 0.035, 0.30, 0.045, 1.28, 0.78),
        "default":    GrainProfile(0.08, 0.03, 0.35, 0.040, 1.18, 0.75),
    },

    # Cinema stock — distinctive mid-century character.
    # Strong clumping, wide bandwidth, mid-frequency presence.
    "Kodak Double-X 5222": {
        "skin":       GrainProfile(0.045, 0.04, 0.55, 0.035, 1.35, 0.60),
        "sky":        GrainProfile(0.03, 0.03, 0.70, 0.030, 1.12, 0.42),
        "vegetation": GrainProfile(0.05, 0.04, 0.50, 0.040, 1.40, 0.62),
        "water":      GrainProfile(0.035, 0.03, 0.60, 0.030, 1.15, 0.48),
        "concrete":   GrainProfile(0.06, 0.05, 0.45, 0.045, 1.48, 0.62),
        "default":    GrainProfile(0.045, 0.04, 0.55, 0.040, 1.35, 0.60),
    },
}

DEFAULT_GRAIN_TYPE = "Custom"


def get_grain_type_choices() -> list[str]:
    """Return grain type names for the UI dropdown."""
    return list(GRAIN_TYPES.keys())


# ---------------------------------------------------------------------------
# ADE20K 150-class → grain category mapping
# ---------------------------------------------------------------------------
# ADE20K class indices (0-based) → grain category string
ADE20K_TO_GRAIN: dict[int, str] = {
    # sky
    2: "sky",
    # vegetation
    4: "vegetation",   # tree
    9: "vegetation",   # grass
    16: "vegetation",  # mountain
    17: "vegetation",  # plant
    66: "vegetation",  # palm
    73: "vegetation",  # flower
    # water
    21: "water",   # water
    26: "water",   # sea
    60: "water",   # river
    109: "water",  # waterfall
    128: "water",  # lake
    # concrete / built environment
    0: "concrete",   # wall
    1: "concrete",   # building
    6: "concrete",   # road
    8: "concrete",   # windowpane
    11: "concrete",  # sidewalk
    13: "concrete",  # earth
    25: "concrete",  # house
    28: "concrete",  # rock
    29: "concrete",  # bridge
    32: "concrete",  # fence
    34: "concrete",  # path
    46: "concrete",  # stairs
    48: "concrete",  # column
    51: "concrete",  # skyscraper
    79: "concrete",  # tower
    84: "concrete",  # pier
    # skin — handled specially (person mask + HSV filter)
    12: "skin",  # person — will be refined by skin detector
}


# ---------------------------------------------------------------------------
# Luminance zone thresholds (soft boundaries)
# ---------------------------------------------------------------------------
ZONE_SHADOW_MID = 0.25   # midpoint between shadow and midtone
ZONE_MID_HIGH = 0.70     # midpoint between midtone and highlight
ZONE_SOFTNESS = 0.12     # transition width


# ---------------------------------------------------------------------------
# Tone curve defaults
# ---------------------------------------------------------------------------
DEFAULT_TOE = 0.0
DEFAULT_SHOULDER = 0.0


# ---------------------------------------------------------------------------
# B&W channel mix defaults (warm / portrait-friendly)
# ---------------------------------------------------------------------------
DEFAULT_BW_MIX = (0.35, 0.45, 0.20)  # R, G, B


# ---------------------------------------------------------------------------
# Preset I/O
# ---------------------------------------------------------------------------
def profiles_to_dict(profiles: dict[str, GrainProfile]) -> dict[str, Any]:
    return {k: asdict(v) for k, v in profiles.items()}


def profiles_from_dict(d: dict[str, Any]) -> dict[str, GrainProfile]:
    return {k: GrainProfile(**v) for k, v in d.items()}


def load_preset(path: Path) -> dict[str, GrainProfile]:
    with open(path) as f:
        return profiles_from_dict(yaml.safe_load(f))


def save_preset(profiles: dict[str, GrainProfile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(profiles_to_dict(profiles), f, default_flow_style=False)
