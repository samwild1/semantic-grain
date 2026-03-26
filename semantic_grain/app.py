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
    GRAIN_TYPES,
    DEFAULT_GRAIN_TYPE,
    get_grain_type_choices,
    DEFAULT_TOE,
    DEFAULT_SHOULDER,
    DEFAULT_BW_MIX,
    load_preset,
    save_preset,
)
from semantic_grain.cache import ProcessingCache, PipelineParams
from semantic_grain.io.loader import load_image, ALL_SUPPORTED_EXTENSIONS
from semantic_grain.io.saver import save_tiff16, save_jpg
from semantic_grain.pipeline import (
    run_segmentation,
    apply_grain,
    apply_grain_cached,
    generate_mask_overlay,
)
from semantic_grain.segmentation.registry import (
    METHODS,
    DEFAULT_METHOD,
    get_dropdown_choices,
    is_method_available,
    check_and_install,
)
from semantic_grain.device import (
    is_gpu_available,
    is_gpu_enabled,
    default_use_gpu,
    gpu_type,
    gpu_name,
    gpu_status_label,
    set_use_gpu,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"
PRESETS_DIR = PROJECT_ROOT / "presets"

# ---------------------------------------------------------------------------
# State held across callbacks (module-level to avoid Gradio pickle overhead)
# ---------------------------------------------------------------------------
_cache = ProcessingCache()
_last_status: str = ""


def _get_sample_images() -> list[str]:
    if SAMPLES_DIR.exists():
        return sorted(
            str(p) for p in SAMPLES_DIR.iterdir()
            if p.suffix.lower() in ALL_SUPPORTED_EXTENSIONS
        )
    return []


def _load_and_segment(
    image_path: str,
    method_key: str = DEFAULT_METHOD,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load image and run segmentation (cached in module-level _cache)."""
    global _cache

    if (
        image_path == _cache.image_path
        and method_key == _cache.seg_method
        and _cache.image is not None
    ):
        return _cache.image, _cache.masks

    # Auto-install missing dependencies if needed
    if not is_method_available(method_key):
        ok, msg = check_and_install(method_key)
        if not ok:
            raise RuntimeError(f"Cannot use {METHODS[method_key].display_name}: {msg}")

    img = load_image(image_path)
    masks = run_segmentation(img, method_key=method_key)

    _cache.invalidate_all()
    _cache.image_path = image_path
    _cache.seg_method = method_key
    _cache.image = img
    _cache.masks = masks

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


def _apply_grain_type_and_process(
    grain_type,
    image_path, seg_method,
    strength, toe, shoulder,
    bw_r, bw_g, bw_b,
    convert_bw, show_masks,
    *current_region_sliders,
):
    """Apply grain type preset to all sliders and immediately re-process."""
    param_names = ["center_freq", "bandwidth", "spectral_slope",
                   "amplitude", "shadow_boost", "highlight_rolloff"]

    if grain_type == "Custom" or grain_type not in GRAIN_TYPES:
        slider_values = list(current_region_sliders)
    else:
        profiles = GRAIN_TYPES[grain_type]
        slider_values = []
        for cat in GRAIN_CATEGORIES:
            profile = profiles[cat]
            for param in param_names:
                slider_values.append(getattr(profile, param))

    result_image, status = process_image(
        image_path, seg_method,
        strength, toe, shoulder,
        bw_r, bw_g, bw_b,
        convert_bw, show_masks,
        *slider_values,
    )

    return [*slider_values, result_image, status]


# ---------------------------------------------------------------------------
# Callback: process image (cached — only recomputes changed stages)
# ---------------------------------------------------------------------------
def process_image(
    image_path,
    seg_method,
    strength, toe, shoulder,
    bw_r, bw_g, bw_b,
    convert_bw,
    show_masks,
    # Per-region sliders (6 params × 6 categories = 36 values)
    *region_sliders,
):
    """Main processing callback with tiered caching."""
    if image_path is None:
        return None, "Upload an image first."

    # Handle Gradio file upload (returns path string or dict)
    if isinstance(image_path, dict):
        image_path = image_path.get("name", image_path.get("path", ""))
    image_path = str(image_path)

    t_total = time.time()

    try:
        t0 = time.time()
        img, masks = _load_and_segment(image_path, method_key=seg_method)
        t_seg = time.time() - t0
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

    bw_mix = (bw_r, bw_g, bw_b)
    params = PipelineParams.build(
        strength=strength, toe=toe, shoulder=shoulder,
        bw_mix=bw_mix, convert_bw=convert_bw, profiles=profiles,
    )

    t0 = time.time()
    result = apply_grain_cached(
        img, masks, profiles,
        global_strength=strength,
        toe=toe, shoulder=shoulder,
        bw_mix=bw_mix,
        convert_bw=convert_bw,
        seed=42,
        cache=_cache,
        params=params,
    )
    t_grain = time.time() - t0

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

    global _last_status
    elapsed = time.time() - t_total

    if elapsed < 0.1 and _last_status:
        # Everything was cached — show the last real render time
        status = _last_status + " (cached)"
    elif t_seg > 0.1:
        # Fresh segmentation + grain render
        status = f"{elapsed:.1f}s (segmentation {t_seg:.1f}s + grain {t_grain:.1f}s) — {w}×{h}"
        _last_status = status
    else:
        # Cached segmentation, fresh grain render
        status = f"{t_grain:.1f}s grain — {w}×{h}"
        _last_status = status

    return (result_rgb * 255).astype(np.uint8), status


def export_image(
    image_path,
    seg_method,
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
        img, masks = _load_and_segment(image_path, method_key=seg_method)
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

    with gr.Blocks(title="Semantic Grain") as demo:
        gr.Markdown("# SEMANTIC GRAIN\n*Grain becomes a language, not a texture pack.*")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input Image",
                    type="filepath",
                    sources=["upload"],
                )
                seg_method_dropdown = gr.Dropdown(
                    choices=get_dropdown_choices(),
                    value=DEFAULT_METHOD,
                    label="Segmentation Model",
                    interactive=True,
                )
                grain_type_dropdown = gr.Dropdown(
                    choices=get_grain_type_choices(),
                    value=DEFAULT_GRAIN_TYPE,
                    label="Grain Type",
                    interactive=True,
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
                with gr.Row():
                    gr.HTML("")  # spacer
                    use_gpu = gr.Checkbox(
                        label=gpu_status_label(),
                        value=default_use_gpu(),
                        interactive=is_gpu_available(),
                    )
                output_image = gr.Image(label="Output Preview", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Global controls
        with gr.Accordion("Global Controls", open=True):
            with gr.Row():
                strength = gr.Slider(0, 3.0, value=0.4, step=0.05, label="Strength")
                toe = gr.Slider(0, 1.0, value=DEFAULT_TOE, step=0.05, label="Shadow Lift")
                shoulder = gr.Slider(0, 1.0, value=DEFAULT_SHOULDER, step=0.05, label="Highlight Compression")
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
            seg_method_dropdown,
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

        # Auto-process on global control change (region sliders use Process btn
        # or grain type dropdown to avoid 36× cascade re-renders)
        for slider in [seg_method_dropdown, strength, toe, shoulder, bw_r, bw_g, bw_b, convert_bw, show_masks]:
            slider.change(
                fn=process_image,
                inputs=all_inputs,
                outputs=[output_image, status_text],
            )

        # Grain type preset: updates all 36 sliders + re-processes in one call
        grain_type_inputs = [
            grain_type_dropdown,
            image_input, seg_method_dropdown,
            strength, toe, shoulder,
            bw_r, bw_g, bw_b,
            convert_bw, show_masks,
            *region_sliders,
        ]
        grain_type_dropdown.change(
            fn=_apply_grain_type_and_process,
            inputs=grain_type_inputs,
            outputs=[*region_sliders, output_image, status_text],
        )

        # GPU toggle
        def _toggle_gpu(enabled):
            set_use_gpu(enabled)
            _cache.invalidate_all()
            if enabled:
                return f"GPU enabled — {gpu_name()} ({gpu_type().upper()})"
            return "Using CPU"

        use_gpu.change(
            fn=_toggle_gpu,
            inputs=[use_gpu],
            outputs=[status_text],
        )

        # Export
        export_inputs = [
            image_input,
            seg_method_dropdown,
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

    # Limit concurrency so rapid slider changes don't queue stale renders
    demo.queue(default_concurrency_limit=1)

    return demo
