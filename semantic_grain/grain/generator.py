"""FFT spectral grain synthesis and luminance-modulated application."""

from __future__ import annotations

import numpy as np

from semantic_grain.config import GrainProfile


def _build_bandpass(
    shape: tuple[int, int],
    center_freq: float,
    bandwidth: float,
    spectral_slope: float,
) -> np.ndarray:
    """Build a radial bandpass filter in the frequency domain.

    Args:
        shape: (H, W) of the spatial image.
        center_freq: Center of the passband as fraction of Nyquist (0–1).
        bandwidth: Width of the passband.
        spectral_slope: 1/f^alpha falloff — higher = more organic/clumpy.

    Returns:
        (H, W) float32 filter magnitude.
    """
    h, w = shape
    # Frequency grid normalized to [0, 1] where 1 = Nyquist
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(fy_grid ** 2 + fx_grid ** 2)

    # Gaussian bandpass
    bp = np.exp(-0.5 * ((radius - center_freq) / max(bandwidth, 1e-6)) ** 2)

    # 1/f^alpha slope (avoid division by zero at DC)
    slope = 1.0 / np.maximum(radius, 1e-6) ** spectral_slope
    slope /= slope.max()

    return (bp * slope).astype(np.float32)


def generate_grain(
    shape: tuple[int, int],
    profile: GrainProfile,
    seed: int | None = None,
) -> np.ndarray:
    """Synthesize a grain texture via FFT spectral shaping.

    Args:
        shape: (H, W) output size.
        profile: Grain parameters.
        seed: Random seed for reproducibility.

    Returns:
        (H, W) float32 grain centered around 0, scaled by profile.amplitude.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)

    # FFT → shape spectrum → IFFT
    spectrum = np.fft.fft2(noise)
    filt = _build_bandpass(shape, profile.center_freq, profile.bandwidth, profile.spectral_slope)
    spectrum *= filt
    grain = np.fft.ifft2(spectrum).real

    # Normalize to unit variance then scale by amplitude
    std = grain.std()
    if std > 1e-8:
        grain /= std
    grain *= profile.amplitude

    return grain


def modulate_by_luminance(
    grain: np.ndarray,
    luminance: np.ndarray,
    shadow_boost: float,
    highlight_rolloff: float,
) -> np.ndarray:
    """Modulate grain amplitude by local luminance.

    Shadows get boosted, highlights get reduced — matching real film behavior.

    Args:
        grain: (H, W) grain texture.
        luminance: (H, W) in [0, 1].
        shadow_boost: Multiplier at luminance=0 (>1 = heavier shadow grain).
        highlight_rolloff: Multiplier at luminance=1 (<1 = less highlight grain).

    Returns:
        (H, W) modulated grain.
    """
    # Linear interpolation: at L=0 → shadow_boost, at L=1 → highlight_rolloff
    mod = shadow_boost + (highlight_rolloff - shadow_boost) * luminance
    return grain * mod
