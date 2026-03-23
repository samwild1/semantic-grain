# Semantic Grain

**Context-aware film grain that understands your photograph.**

<p align="center">
  <img src="examples/semantic_grain.jpg" alt="Semantic Grain output" width="100%">
</p>

Semantic Grain uses AI semantic segmentation to identify regions in your photograph -- sky, water, skin, vegetation, concrete -- and applies unique FFT-spectral grain to each, modeling how real photographic emulsion responds differently across a scene. Shadows accumulate heavier grain. Highlights stay clean. Every region gets its own spectral character.

This is not a filter. It is a computational emulsion.

---

## Before | After | How It Sees

<table>
  <tr>
    <td align="center"><strong>Original</strong></td>
    <td align="center"><strong>Semantic Grain</strong></td>
    <td align="center"><strong>Segmentation Map</strong></td>
  </tr>
  <tr>
    <td><img src="examples/original.jpg" width="100%"></td>
    <td><img src="examples/semantic_grain.jpg" width="100%"></td>
    <td><img src="examples/segmentation_mask.jpg" width="100%"></td>
  </tr>
</table>

The segmentation map shows how the model understands the scene. Each colored region receives grain with distinct spectral characteristics -- different size, clumpiness, shadow response, and highlight rolloff.

| Color | Region | Grain character |
|-------|--------|-----------------|
| Peach | Skin | Fine, gentle, restrained in highlights |
| Blue | Sky | Very fine, smooth, minimal shadow boost |
| Green | Vegetation | Medium, moderate clumping |
| Dark blue | Water | Fine, smooth, strong spectral slope |
| Tan | Concrete | Coarse, pronounced, gritty |
| Grey | Default | Medium, balanced |

---

## Features

- **Semantic segmentation** via [SegFormer-B5](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) -- 150 ADE20K classes mapped to 6 grain categories
- **FFT spectral grain synthesis** -- frequency-domain grain shaped by bandpass filters with 1/f spectral slopes, not noise overlay
- **Luminance-aware modulation** -- heavier grain in shadows, lighter in highlights, matching real silver halide behavior
- **Per-region grain profiles** with 6 parameters each: center frequency, bandwidth, spectral slope, amplitude, shadow boost, highlight rolloff
- **Soft mask blending** -- Gaussian-softened region boundaries prevent visible seams
- **HSV skin detection** -- refines person segmentation to distinguish skin from clothing
- **Film tone curve** -- parametric S-curve with independent toe and shoulder control
- **Channel-weighted B&W conversion** -- customizable RGB mix (default: 35/45/20)
- **Interactive Gradio UI** with real-time parameter adjustment
- **YAML grain presets** -- ships with Ilford Delta 400-inspired defaults
- **16-bit TIFF export** for maximum tonal fidelity

---

## How It Works

```
Input Image
    |
    v
[SegFormer-B5] -----> 150-class Label Map -----> 6 Grain Category Masks
    |                                                      |
    v                                                      v
[B&W Conversion]                                [Gaussian Soft Blending]
    |                                                      |
    v                                                      v
[Luminance Map] ----------> [Per-Region FFT Grain Synthesis]
    |                                  |
    v                                  v
[Zone Masks] -----> [Luminance-Modulated Grain Compositing]
                                       |
                                       v
                              [Parametric Tone Curve]
                                       |
                                       v
                                    Output
```

Each grain category has its own spectral profile controlling grain size, clumpiness, and tonal response. Grain is synthesized in the frequency domain using shaped bandpass filters, then modulated by local luminance so shadows naturally accumulate more grain -- just like real film.

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU recommended (CPU works but segmentation will be slower)

### Setup with Conda (recommended)

```bash
git clone https://github.com/samwild1/semantic-grain.git
cd semantic-grain
conda env create -f environment.yml
conda activate sgrain
pip install -e .
```

### Setup with pip

```bash
git clone https://github.com/samwild1/semantic-grain.git
cd semantic-grain
pip install -e .
```

> On first run, SegFormer-B5 weights (~380 MB) are downloaded automatically from Hugging Face.

---

## Usage

### Launch the interactive UI

```bash
python -m semantic_grain
```

This opens a Gradio web interface where you can:
- Upload any photograph
- Adjust global grain strength, tone curve, and B&W channel mix
- Fine-tune per-region grain parameters (6 parameters x 6 regions)
- Visualize the segmentation mask
- Export as JPEG or 16-bit TIFF

### Windows quick launch

Double-click `Start Semantic Grain.bat` (assumes Conda is installed).

### Programmatic use

```python
from semantic_grain.io.loader import load_image
from semantic_grain.pipeline import run_segmentation, apply_grain

image = load_image("your_photo.jpg")
masks = run_segmentation(image)
result = apply_grain(image, masks, convert_bw=True, preview_size=None)
```

---

## Grain Presets

Grain profiles are stored as YAML in `presets/`. The default preset is inspired by Ilford Delta 400:

```yaml
skin:
  amplitude: 0.03
  center_freq: 0.15
  bandwidth: 0.04
  spectral_slope: 0.3
  shadow_boost: 1.2
  highlight_rolloff: 0.8
sky:
  amplitude: 0.025
  center_freq: 0.06
  ...
```

| Parameter | What it controls |
|-----------|-----------------|
| `center_freq` | Grain size (higher = finer) |
| `bandwidth` | Variation in grain clump sizes |
| `spectral_slope` | Organic feel (higher = more natural 1/f clumping) |
| `amplitude` | Overall grain intensity for this region |
| `shadow_boost` | Extra grain in dark areas (>1 = heavier shadows) |
| `highlight_rolloff` | Grain reduction in bright areas (<1 = cleaner highlights) |

---

## Project Structure

```
semantic-grain/
├── semantic_grain/
│   ├── app.py                # Gradio web interface
│   ├── pipeline.py           # Core processing orchestration
│   ├── config.py             # Profiles, mappings, defaults
│   ├── segmentation/         # SegFormer-B5 + skin detection
│   ├── grain/                # FFT spectral synthesis
│   ├── luminance/            # Tone curves + zone system
│   ├── blending/             # Soft mask compositing
│   ├── color/                # B&W conversion
│   └── io/                   # Image loading + saving
├── presets/                   # YAML grain presets
├── examples/                  # Example images
├── scripts/                   # Utility scripts
├── environment.yml            # Conda environment
└── pyproject.toml             # Package metadata
```

---

## License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You may use this software for personal, non-commercial purposes. Commercial use and derivative works are not permitted.

---

## Acknowledgments

- [NVIDIA SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) for semantic segmentation
- Grain synthesis is inspired by how real silver halide crystals respond to light, not by any existing digital filter

---

*Semantic Grain is the first module of a larger vision: building a native digital medium with the emotional intelligence of film.*
