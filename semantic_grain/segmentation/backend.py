"""Unified segmentation backend — dispatches to the right model and post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from semantic_grain.device import get_device
from semantic_grain.segmentation.registry import METHODS, DEFAULT_METHOD

# ---------------------------------------------------------------------------
# Model cache: keeps loaded models on CPU, only one on GPU at a time.
# ---------------------------------------------------------------------------
_loaded: dict[str, tuple[Any, Any]] = {}  # key -> (processor, model)
_current_key: str | None = None


def _ensure_loaded(key: str) -> tuple[Any, Any]:
    """Load (or retrieve from cache) the processor and model for *key*.

    Only one model resides on the active device at a time.  Previously
    loaded models are kept on CPU so switching back is cheap.
    """
    global _current_key

    device = get_device()
    method = METHODS[key]

    # Already loaded — just make sure it's on the right device
    if key in _loaded:
        processor, model = _loaded[key]
        if _current_key != key:
            # Move the old model off device
            if _current_key is not None and _current_key in _loaded:
                _loaded[_current_key][1].cpu()
            model.to(device)
            _current_key = key
        return processor, model

    # Move old model off device before loading a new one
    if _current_key is not None and _current_key in _loaded:
        _loaded[_current_key][1].cpu()

    if method.backend == "mask2former":
        processor, model = _load_mask2former(method.model_id)
    else:
        processor, model = _load_standard(method.model_id)

    model.to(device)
    model.eval()
    _loaded[key] = (processor, model)
    _current_key = key
    return processor, model


# ---------------------------------------------------------------------------
# Device migration (called by device.set_use_gpu)
# ---------------------------------------------------------------------------
def move_models_to_device(device: torch.device) -> None:
    """Move the active model to a new device after a GPU toggle."""
    global _current_key

    if _current_key is not None and _current_key in _loaded:
        _, model = _loaded[_current_key]
        model.to(device)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_standard(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    return processor, model


def _load_mask2former(model_id: str) -> tuple[Any, Any]:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    return processor, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _infer_standard(
    processor: Any,
    model: Any,
    pil_img: Image.Image,
    h_orig: int,
    w_orig: int,
) -> np.ndarray:
    """Standard path: logits -> upsample -> argmax."""
    device = get_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, h', w')
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def _infer_mask2former(
    processor: Any,
    model: Any,
    pil_img: Image.Image,
    h_orig: int,
    w_orig: int,
) -> np.ndarray:
    """Mask2Former path: post_process_semantic_segmentation."""
    device = get_device()
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(h_orig, w_orig)]
    )[0]
    return result.cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def segment_image(
    image_rgb: np.ndarray,
    method_key: str = DEFAULT_METHOD,
) -> np.ndarray:
    """Run segmentation on an RGB float32 [0,1] image.

    Args:
        image_rgb: (H, W, 3) float32 in [0, 1].
        method_key: Key from ``registry.METHODS``.

    Returns:
        (H, W) int32 label map with ADE20K class indices (0-149).
    """
    method = METHODS[method_key]
    processor, model = _ensure_loaded(method_key)

    h_orig, w_orig = image_rgb.shape[:2]

    # Convert to PIL
    img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # Downscale for inference
    scale = method.inference_size / max(h_orig, w_orig)
    if scale < 1.0:
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if method.backend == "mask2former":
        return _infer_mask2former(processor, model, pil_img, h_orig, w_orig)
    else:
        return _infer_standard(processor, model, pil_img, h_orig, w_orig)
