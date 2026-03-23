"""Gradio UI for Semantic Grain."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np

from semantic_grain.config import (
    DEFAULT_PROFILES,
    GrainProfile,
    GRAIN_CATEGORIES,
    DEFAULT_TOE,
    DEFAULT_SHOULDER,
    DEFAULT_BW_MIX,
    load_preset,
    save_preset,
)
from semantic_grain.io.loader import load_image
from semantic_grain.io.saver import save_tiff16, save_jpg
from semantic_grain.pipeline import (
    run_segmentation,
    apply_grain,
    generate_mask_overlay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRESETS_DIR = PROJECT_ROOT / "presets"

# ---------------------------------------------------------------------------
# State held across callbacks
# ---------------------------------------------------------------------------
_cached_image: np.ndarray | None = None
_cached_masks: dict[str, np.ndarray] | None = None
_image_path: str | None = None


def _get_sample_images() -> list[str]:
    if SAMPLES_DIR.exists():
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        return sorted(
            str(p) for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts
        )
    return []


def _load_and_segment(image_path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load image and run segmentation (cached)."""
    global _cached_image, _cached_masks, _image_path

    if image_path == _image_path and _cached_image is not None:
        return _cached_image, _cached_masks

    img = load_image(image_path)
    masks = run_segmentation(img)

    _cached_image = img
    _cached_masks = masks
    _image_path = image_path

    return img, masks


def _build_profiles_from_sliders(slider_values: dict) -> dict[str, GrainProfile]:
    """Reconstruct GrainProfile dict from flat slider values."""
    profiles = {}
    for cat in GRAIN_CATEGORIES:
        profiles[cat] = GrainProfile(
            center_freq=slider_values.get(f"{cat}_center_freq", DEFAULT_PROFILES[cat].center_freq),
            bandwidth=slider_values.get(f"{cat}_bandwidth", DEFAULT_PROFILES[cat].bandwidth),
            spectral_slope=slider_values.get(f"{cat}_spectral_slope", DEFAULT_PROFILES[cat].spectral_slope),
            amplitude=slider_values.get(f"{cat}_amplitude", DEFAULT_PROFILES[cat].amplitude),
            shadow_boost=slider_values.get(f"{cat}_shadow_boost", DEFAULT_PROFILES[cat].shadow_boost),
            highlight_rolloff=slider_values.get(f"{cat}_highlight_rolloff", DEFAULT_PROFILES[cat].highlight_rolloff),
        )
    return profiles


# ---------------------------------------------------------------------------
# Callback: process image
# ---------------------------------------------------------------------------
def process_image(
    image_path,
    strength, toe, shoulder,
    bw_r, bw_g, bw_b,
    convert_bw,
    show_masks,
    # Per-region sliders (6 params × 6 categories = 36 values)
    *region_sliders,
):
    """Main processing callback."""
    if image_path is None:
        return None, "Upload an image first."

    # Handle Gradio file upload (returns path string or dict)
    if isinstance(image_path, dict):
        image_path = image_path.get("name", image_path.get("path", ""))
    image_path = str(image_path)

    try:
        img, masks = _load_and_segment(image_path)
    except Exception as e:
        return None, f"Error loading image: {e}"

    # Build profiles from slider values
    param_names = ["center_freq", "bandwidth", "spectral_slope", "amplitude", "shadow_boost", "highlight_rolloff"]
    slider_dict = {}
    idx = 0
    for cat in GRAIN_CATEGORIES:
        for param in param_names:
            slider_dict[f"{cat}_{param}"] = region_sliders[idx]
            idx += 1

    profiles = _build_profiles_from_sliders(slider_dict)

    if show_masks:
        overlay = generate_mask_overlay(img, masks)
        return (overlay * 255).astype(np.uint8), "Showing segmentation masks"

    t0 = time.time()
    result = apply_grain(
        img, masks, profiles,
        global_strength=strength,
        toe=toe, shoulder=shoulder,
        bw_mix=(bw_r, bw_g, bw_b),
        convert_bw=convert_bw,
        preview_size=None,  # full resolution for accurate grain preview
    )
    elapsed = time.time() - t0

    # Ensure RGB for Gradio display
    if result.ndim == 2:
        result_rgb = np.stack([result, result, result], axis=-1)
    else:
        result_rgb = result

    # Downscale for display only (grain was rendered at full res)
    import cv2
    h, w = result_rgb.shape[:2]
    display_size = 1536
    if max(h, w) > display_size:
        scale = display_size / max(h, w)
        result_rgb = cv2.resize(result_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return (result_rgb * 255).astype(np.uint8), f"Processed in {elapsed:.2f}s"


def export_image(
    image_path,
    strength, toe, shoulder,
    bw_r, bw_g, bw_b,
    convert_bw,
    export_format,
    *region_sliders,
):
    """Export at full resolution."""
    if image_path is None:
        return "Upload an image first."

    if isinstance(image_path, dict):
        image_path = image_path.get("name", image_path.get("path", ""))
    image_path = str(image_path)

    try:
        img, masks = _load_and_segment(image_path)
    except Exception as e:
        return f"Error: {e}"

    param_names = ["center_freq", "bandwidth", "spectral_slope", "amplitude", "shadow_boost", "highlight_rolloff"]
    slider_dict = {}
    idx = 0
    for cat in GRAIN_CATEGORIES:
        for param in param_names:
            slider_dict[f"{cat}_{param}"] = region_sliders[idx]
            idx += 1

    profiles = _build_profiles_from_sliders(slider_dict)

    stem = Path(image_path).stem
    ext = ".jpg" if export_format == "JPEG" else ".tiff"
    out_path = OUTPUT_DIR / f"{stem}_sgrain{ext}"

    t0 = time.time()
    result = apply_grain(
        img, masks, profiles,
        global_strength=strength,
        toe=toe, shoulder=shoulder,
        bw_mix=(bw_r, bw_g, bw_b),
        convert_bw=convert_bw,
        preview_size=None,
    )

    if export_format == "JPEG":
        save_jpg(result, out_path, quality=95)
    else:
        save_tiff16(result, out_path)

    elapsed = time.time() - t0
    return f"Exported to {out_path} ({elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------
def create_ui() -> gr.Blocks:
    sample_images = _get_sample_images()

    with gr.Blocks(
        title="Semantic Grain",
        theme=gr.themes.Base(
            primary_hue="stone",
            neutral_hue="stone",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .gr-button { border-radius: 2px !important; }
        """,
    ) as demo:
        gr.Markdown("# SEMANTIC GRAIN\n*Grain becomes a language, not a texture pack.*")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input Image",
                    type="filepath",
                    sources=["upload"],
                )
                if sample_images:
                    sample_dropdown = gr.Dropdown(
                        choices=sample_images,
                        label="Or select a sample",
                        interactive=True,
                    )
                    sample_dropdown.change(
                        fn=lambda x: x,
                        inputs=[sample_dropdown],
                        outputs=[image_input],
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(label="Output Preview", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Global controls
        with gr.Accordion("Global Controls", open=True):
            with gr.Row():
                strength = gr.Slider(0, 3.0, value=0.4, step=0.05, label="Strength")
                toe = gr.Slider(0, 1.0, value=DEFAULT_TOE, step=0.05, label="Tone Curve (toe)")
                shoulder = gr.Slider(0, 1.0, value=DEFAULT_SHOULDER, step=0.05, label="Tone Curve (shoulder)")
            with gr.Row():
                bw_r = gr.Slider(0, 1.0, value=DEFAULT_BW_MIX[0], step=0.05, label="B&W Red")
                bw_g = gr.Slider(0, 1.0, value=DEFAULT_BW_MIX[1], step=0.05, label="B&W Green")
                bw_b = gr.Slider(0, 1.0, value=DEFAULT_BW_MIX[2], step=0.05, label="B&W Blue")

        with gr.Row():
            convert_bw = gr.Checkbox(label="Convert to B&W", value=False)
            show_masks = gr.Checkbox(label="Show Masks", value=False)

        # Per-region grain controls
        region_sliders = []
        with gr.Accordion("Per-Region Grain", open=False):
            for cat in GRAIN_CATEGORIES:
                defaults = DEFAULT_PROFILES[cat]
                with gr.Tab(cat.capitalize()):
                    with gr.Row():
                        region_sliders.append(
                            gr.Slider(0.01, 0.5, value=defaults.center_freq, step=0.01,
                                      label=f"Grain Size (center_freq)")
                        )
                        region_sliders.append(
                            gr.Slider(0.01, 0.2, value=defaults.bandwidth, step=0.005,
                                      label=f"Clump Variation (bandwidth)")
                        )
                        region_sliders.append(
                            gr.Slider(0.0, 1.5, value=defaults.spectral_slope, step=0.05,
                                      label=f"Organic Feel (spectral_slope)")
                        )
                    with gr.Row():
                        region_sliders.append(
                            gr.Slider(0.0, 0.15, value=defaults.amplitude, step=0.005,
                                      label=f"Amplitude")
                        )
                        region_sliders.append(
                            gr.Slider(0.5, 2.0, value=defaults.shadow_boost, step=0.05,
                                      label=f"Shadow Boost")
                        )
                        region_sliders.append(
                            gr.Slider(0.1, 1.5, value=defaults.highlight_rolloff, step=0.05,
                                      label=f"Highlight Rolloff")
                        )

        # Action buttons
        with gr.Row():
            process_btn = gr.Button("Process", variant="primary")
            export_format = gr.Dropdown(choices=["JPEG", "TIFF-16"], value="JPEG", label="Export Format")
            export_btn = gr.Button("Export")

        # Wire up processing
        all_inputs = [
            image_input,
            strength, toe, shoulder,
            bw_r, bw_g, bw_b,
            convert_bw,
            show_masks,
            *region_sliders,
        ]

        process_btn.click(
            fn=process_image,
            inputs=all_inputs,
            outputs=[output_image, status_text],
        )

        # Auto-process on slider change
        for slider in [strength, toe, shoulder, bw_r, bw_g, bw_b, convert_bw, show_masks, *region_sliders]:
            slider.change(
                fn=process_image,
                inputs=all_inputs,
                outputs=[output_image, status_text],
            )

        # Export
        export_inputs = [
            image_input,
            strength, toe, shoulder,
            bw_r, bw_g, bw_b,
            convert_bw,
            export_format,
            *region_sliders,
        ]
        export_btn.click(
            fn=export_image,
            inputs=export_inputs,
            outputs=[status_text],
        )

    return demo
