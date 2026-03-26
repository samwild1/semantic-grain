"""Multi-octave FFT spectral grain synthesis and luminance modulation.

Grain is synthesised by summing bandpass-filtered noise at multiple frequency
octaves, producing the multi-scale clumping characteristic of real silver
halide film emulsions.  A subtle non-linear transfer adds particle-like
sparsity (Poisson-like amplitude distribution).
"""

from __future__ import annotations

import numpy as np
import torch

from semantic_grain.config import GrainProfile
from semantic_grain.device import get_device

# Number of frequency octaves for multi-scale grain
_N_OCTAVES = 4

# Non-linear transfer exponent (>1 = sparser, more particle-like grain)
_GRAIN_EXPONENT = 1.2


# ---------------------------------------------------------------------------
# CPU path (numpy)
# ---------------------------------------------------------------------------
def _build_multiscale_filter_cpu(
    shape: tuple[int, int],
    center_freq: float,
    bandwidth: float,
    spectral_slope: float,
) -> np.ndarray:
    """Build a multi-octave radial bandpass filter in the frequency domain.

    Sums Gaussian bandpass filters at geometrically-spaced frequencies,
    weighted by a 1/f^α inter-octave falloff.  This produces multi-scale
    grain structure: large clumps containing medium clusters containing
    fine grains.

    Args:
        shape: (H, W) of the spatial image.
        center_freq: Dominant (coarsest) grain scale as fraction of Nyquist.
        bandwidth: Per-octave passband width — controls grain sharpness.
        spectral_slope: Inter-octave falloff exponent.  0 = all octaves
            equal (uniform/modern), 0.5+ = dominant large clumps (organic).

    Returns:
        (H, W) float32 filter magnitude.
    """
    h, w = shape
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(fy_grid ** 2 + fx_grid ** 2)

    filt = np.zeros(shape, dtype=np.float32)
    for i in range(_N_OCTAVES):
        octave_freq = center_freq * (2.0 ** i)
        octave_bw = bandwidth * (1.5 ** i)
        octave_weight = 1.0 / (2.0 ** (i * spectral_slope))

        bp = np.exp(
            -0.5 * ((radius - octave_freq) / max(octave_bw, 1e-6)) ** 2
        )
        filt += bp * octave_weight

    return filt


def _generate_grain_cpu(
    shape: tuple[int, int],
    profile: GrainProfile,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)

    spectrum = np.fft.fft2(noise)
    filt = _build_multiscale_filter_cpu(
        shape, profile.center_freq, profile.bandwidth, profile.spectral_slope,
    )
    spectrum *= filt
    grain = np.fft.ifft2(spectrum).real

    # Normalise to unit variance
    std = grain.std()
    if std > 1e-8:
        grain /= std

    # Non-linear transfer: sparser, more particle-like grain
    grain = np.sign(grain) * np.abs(grain) ** _GRAIN_EXPONENT

    # Re-normalise after transfer and scale by amplitude
    std = grain.std()
    if std > 1e-8:
        grain /= std
    grain *= profile.amplitude

    return grain


# ---------------------------------------------------------------------------
# GPU path (torch.fft on CUDA or MPS)
# ---------------------------------------------------------------------------
def _build_multiscale_filter_gpu(
    shape: tuple[int, int],
    center_freq: float,
    bandwidth: float,
    spectral_slope: float,
    device: torch.device,
) -> torch.Tensor:
    """GPU version of multi-octave bandpass filter."""
    h, w = shape
    fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32)
    fx = torch.fft.fftfreq(w, device=device, dtype=torch.float32)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(fy_grid ** 2 + fx_grid ** 2)

    filt = torch.zeros(shape, device=device, dtype=torch.float32)
    for i in range(_N_OCTAVES):
        octave_freq = center_freq * (2.0 ** i)
        octave_bw = bandwidth * (1.5 ** i)
        octave_weight = 1.0 / (2.0 ** (i * spectral_slope))

        bp = torch.exp(
            -0.5 * ((radius - octave_freq) / max(octave_bw, 1e-6)) ** 2
        )
        filt = filt + bp * octave_weight

    return filt


def _generate_grain_gpu(
    shape: tuple[int, int],
    profile: GrainProfile,
    seed: int | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)

    spectrum = torch.fft.fft2(noise)
    filt = _build_multiscale_filter_gpu(
        shape, profile.center_freq, profile.bandwidth, profile.spectral_slope,
        device=device,
    )
    spectrum = spectrum * filt
    grain = torch.fft.ifft2(spectrum).real

    # Normalise to unit variance
    std = grain.std()
    if std > 1e-8:
        grain = grain / std

    # Non-linear transfer: sparser, more particle-like grain
    grain = torch.sign(grain) * torch.abs(grain) ** _GRAIN_EXPONENT

    # Re-normalise after transfer and scale by amplitude
    std = grain.std()
    if std > 1e-8:
        grain = grain / std
    grain = grain * profile.amplitude

    return grain.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------
def generate_grain(
    shape: tuple[int, int],
    profile: GrainProfile,
    seed: int | None = None,
) -> np.ndarray:
    """Synthesise a grain texture via multi-octave FFT spectral shaping.

    Uses GPU (torch.fft on CUDA or MPS) when enabled, otherwise falls back
    to numpy FFT on CPU.  Produces multi-scale grain with organic clumping
    characteristic of real silver halide film.

    Args:
        shape: (H, W) output size.
        profile: Grain parameters.
        seed: Random seed for reproducibility.

    Returns:
        (H, W) float32 grain centred around 0, scaled by profile.amplitude.
    """
    device = get_device()
    if device.type != "cpu":
        return _generate_grain_gpu(shape, profile, seed, device=device)
    return _generate_grain_cpu(shape, profile, seed)


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
