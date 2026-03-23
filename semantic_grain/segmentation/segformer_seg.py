"""SegFormer-B5 (ADE20K) inference wrapper."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


_model = None
_processor = None
_device = None

MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model() -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation]:
    global _model, _processor, _device
    if _model is None:
        _device = _get_device()
        _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        _model.to(_device)
        _model.eval()
    return _processor, _model


def segment_image(image_rgb: np.ndarray, inference_size: int = 640) -> np.ndarray:
    """Run SegFormer segmentation on an RGB float32 [0,1] image.

    Args:
        image_rgb: (H, W, 3) float32 in [0, 1].
        inference_size: Downscale longest side to this for inference speed.

    Returns:
        (H, W) int32 label map with ADE20K class indices (0–149).
    """
    processor, model = _load_model()
    h_orig, w_orig = image_rgb.shape[:2]

    # Convert to PIL for the processor
    img_uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # Downscale for inference
    scale = inference_size / max(h_orig, w_orig)
    if scale < 1.0:
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, h', w')

    # Upsample logits to original resolution
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    )
    labels = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    return labels
