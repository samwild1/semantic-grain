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
        center_freq=0.15, bandwidth=0.04, spectral_slope=0.3,
        amplitude=0.03, shadow_boost=1.2, highlight_rolloff=0.8,
    ),
    "sky": GrainProfile(
        center_freq=0.06, bandwidth=0.03, spectral_slope=0.5,
        amplitude=0.025, shadow_boost=1.0, highlight_rolloff=0.5,
    ),
    "concrete": GrainProfile(
        center_freq=0.20, bandwidth=0.08, spectral_slope=0.1,
        amplitude=0.05, shadow_boost=1.4, highlight_rolloff=0.7,
    ),
    "vegetation": GrainProfile(
        center_freq=0.12, bandwidth=0.05, spectral_slope=0.4,
        amplitude=0.04, shadow_boost=1.3, highlight_rolloff=0.7,
    ),
    "water": GrainProfile(
        center_freq=0.08, bandwidth=0.04, spectral_slope=0.6,
        amplitude=0.03, shadow_boost=1.1, highlight_rolloff=0.6,
    ),
    "default": GrainProfile(
        center_freq=0.12, bandwidth=0.06, spectral_slope=0.3,
        amplitude=0.04, shadow_boost=1.3, highlight_rolloff=0.7,
    ),
}


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
DEFAULT_TOE = 0.3
DEFAULT_SHOULDER = 0.3


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
