"""One-off script to generate downsized example images for the GitHub repo."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from semantic_grain.io.loader import load_image
from semantic_grain.pipeline import run_segmentation, generate_mask_overlay

INPUT_PATH = project_root / "samples" / "DSCF6498.jpg"
RENDER_PATH = project_root / "output" / "DSCF6498_sgrain.jpg"
OUTPUT_DIR = project_root / "examples"
TARGET_LONG_EDGE = 1200
JPEG_QUALITY = 85


def downsize_and_save(img_float32, out_path):
    """Resize float32 [0,1] image to TARGET_LONG_EDGE and save as JPEG."""
    h, w = img_float32.shape[:2]
    scale = TARGET_LONG_EDGE / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_float32 = cv2.resize(img_float32, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_uint8 = (np.clip(img_float32, 0, 1) * 255).astype(np.uint8)
    if img_uint8.ndim == 3:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_uint8

    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved {out_path.name} ({size_kb:.0f} KB)")


def downsize(img_float32):
    """Resize float32 image to TARGET_LONG_EDGE, return float32."""
    h, w = img_float32.shape[:2]
    scale = TARGET_LONG_EDGE / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_float32, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_float32


def main():
    print("Loading and downsizing original image...")
    original_full = load_image(INPUT_PATH)
    original = downsize(original_full)

    print("Loading and downsizing rendered image...")
    rendered_full = load_image(RENDER_PATH)
    rendered = downsize(rendered_full)

    print("Running segmentation on downsized image...")
    masks = run_segmentation(original)

    print("Generating mask overlay...")
    overlay = generate_mask_overlay(original, masks, preview_size=None)

    print("Saving examples...")
    downsize_and_save(original, OUTPUT_DIR / "original.jpg")
    downsize_and_save(rendered, OUTPUT_DIR / "semantic_grain.jpg")
    downsize_and_save(overlay, OUTPUT_DIR / "segmentation_mask.jpg")

    print("Done! Example images saved to examples/")


if __name__ == "__main__":
    main()
