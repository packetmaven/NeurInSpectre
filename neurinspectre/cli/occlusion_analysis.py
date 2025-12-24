#!/usr/bin/env python3
"""
NeurInSpectre Occlusion Analysis
Advanced adversarial vulnerability assessment through occlusion-based attribution.

This is a CLI-focused module. It intentionally avoids import-time side effects
(no top-level prints, no duplicated definitions) so importing `neurinspectre.cli`
does not unexpectedly emit output.
"""

from __future__ import annotations

from io import BytesIO
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    """Select the best available device for this workstation."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _require_pillow():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception as e:
        raise ImportError("Pillow is required for occlusion analysis. Install with `pip install pillow`.") from e


def _require_requests():
    try:
        import requests  # type: ignore
        return requests
    except Exception as e:
        raise ImportError("requests is required for URL image loading. Install with `pip install requests`.") from e


def _require_transformers():
    try:
        from transformers import AutoModelForImageClassification  # type: ignore
    except Exception as e:
        raise ImportError("transformers is required for occlusion analysis. Install with `pip install transformers`.") from e

    # Prefer AutoImageProcessor on newer transformers; fall back to AutoFeatureExtractor.
    processor_cls = None
    try:
        from transformers import AutoImageProcessor as _AutoProcessor  # type: ignore
        processor_cls = _AutoProcessor
    except Exception:
        from transformers import AutoFeatureExtractor as _AutoProcessor  # type: ignore
        processor_cls = _AutoProcessor

    return AutoModelForImageClassification, processor_cls


def load_image(*, image_path: Optional[str] = None, url: Optional[str] = None):
    """Load an image from path or URL and return a PIL Image in RGB."""
    Image = _require_pillow()

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        return img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img

    if url:
        requests = _require_requests()
        headers = {"User-Agent": "NeurInSpectre/1.0 (+https://github.com/)"}
        resp = requests.get(url, stream=True, timeout=15, headers=headers)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        return img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img

    raise ValueError("Either image_path or url must be provided")


def run_occlusion_analysis(
    model,
    processor,
    image,
    *,
    patch_size: int = 32,
    stride: int = 16,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    """Run occlusion analysis and return (occlusion_scores, original_class_index)."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers")

    width, height = image.size
    if patch_size > width or patch_size > height:
        raise ValueError(f"patch_size={patch_size} exceeds image dimensions {(width, height)}")

    # Convert image to tensor
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Baseline prediction
    with torch.no_grad():
        outputs = model(pixel_values)
        original_pred = torch.softmax(outputs.logits, dim=1)
        original_class = int(original_pred.argmax().item())
        original_prob = float(original_pred[0, original_class].item())

    print(f"🎯 Original prediction: {getattr(model.config, 'id2label', {}).get(original_class, str(original_class))} ({original_prob:.2f})")
    print("🔍 Running occlusion analysis (this may take a few minutes)...")

    grid_y = (height - patch_size) // stride + 1
    grid_x = (width - patch_size) // stride + 1
    occlusion_scores = np.zeros((grid_y, grid_x), dtype=np.float64)

    total_patches = grid_y * grid_x
    patch_count = 0

    for yi, top in enumerate(range(0, height - patch_size + 1, stride)):
        for xi, left in enumerate(range(0, width - patch_size + 1, stride)):
            occluded_image = image.copy()
            occluded_image.paste((0, 0, 0), (left, top, left + patch_size, top + patch_size))

            occ_inputs = processor(images=occluded_image, return_tensors="pt")
            with torch.no_grad():
                occ_outputs = model(occ_inputs.pixel_values.to(device))
                occ_pred = torch.softmax(occ_outputs.logits, dim=1)

            occlusion_scores[yi, xi] = original_prob - float(occ_pred[0, original_class].item())

            patch_count += 1
            if patch_count % 25 == 0 or patch_count == total_patches:
                progress = 100.0 * patch_count / max(1, total_patches)
                print(f"  📊 Progress: {progress:.1f}% ({patch_count}/{total_patches} patches)")

    return occlusion_scores, original_class


def plot_results(image, occlusion_scores: np.ndarray, model, original_class: int, *, output_file: str = "occlusion_analysis_results.png"):
    """Save a simple 2D heatmap visualization."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise ImportError("matplotlib is required for 2D plotting. Install with `pip install matplotlib`.") from e

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(occlusion_scores, cmap="viridis")
    axes[1].set_title("Occlusion impact (ΔP)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    label = getattr(model.config, "id2label", {}).get(original_class, str(original_class))
    fig.suptitle(f"Occlusion analysis — class {original_class} ({label})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"📊 2D analysis saved as: {output_file}")


def plot_3d_results(occlusion_scores: np.ndarray, *, output_file: str = "3d_occlusion_analysis.html"):
    """Save an interactive 3D surface plot."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise ImportError("plotly is required for 3D plotting. Install with `pip install plotly`.") from e

    y = np.arange(occlusion_scores.shape[0])
    x = np.arange(occlusion_scores.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=occlusion_scores, colorscale="Viridis")])
    fig.update_layout(
        title="3D Occlusion Impact Surface",
        scene=dict(
            xaxis_title="X (grid)",
            yaxis_title="Y (grid)",
            zaxis_title="ΔP",
        ),
        width=1000,
        height=750,
        margin=dict(l=40, r=40, b=40, t=80),
    )
    fig.write_html(output_file)
    print(f"🎯 3D security analysis saved as: {output_file}")


def run_occlusion_analysis_command(args) -> int:
    """CLI command handler used by `python -m neurinspectre.cli occlusion-analysis ...`"""
    device = _select_device()
    print("🛡️ Starting NeurInSpectre Occlusion Analysis...")
    print(f"🔧 Using device: {device}")

    AutoModelForImageClassification, Processor = _require_transformers()

    model_name = getattr(args, "model", None) or "google/vit-base-patch16-224"
    print(f"📥 Loading model: {model_name}")

    try:
        model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
        processor = Processor.from_pretrained(model_name)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Load image
    try:
        if getattr(args, "image_path", None):
            image = load_image(image_path=args.image_path)
            print(f"📷 Loaded image from: {args.image_path}")
        elif getattr(args, "image_url", None):
            image = load_image(url=args.image_url)
            print(f"📷 Loaded image from URL: {args.image_url}")
        else:
            image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = load_image(url=image_url)
            print("📷 Using default image from COCO dataset")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return 1

    # Run analysis + plots
    try:
        occlusion_scores, original_class = run_occlusion_analysis(
            model,
            processor,
            image,
            patch_size=int(getattr(args, "patch_size", 32)),
            stride=int(getattr(args, "stride", 16)),
            device=device,
        )

        np.save("occlusion_scores.npy", occlusion_scores)
        print("💾 Occlusion scores saved as: occlusion_scores.npy")

        output_2d = getattr(args, "output_2d", None) or "occlusion_analysis_results.png"
        output_3d = getattr(args, "output_3d", None) or "3d_occlusion_analysis.html"

        plot_results(image, occlusion_scores, model, original_class, output_file=output_2d)
        plot_3d_results(occlusion_scores, output_file=output_3d)

        print("\n✅ Analysis complete! Use the following commands to view results:")
        print(f"   📊 2D visualization: open {output_2d}")
        print(f"   🎯 3D visualization: open {output_3d}")
        return 0
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeurInSpectre Occlusion Analysis - Adversarial vulnerability assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--image-path", "-i", help="Path to input image")
    parser.add_argument("--image-url", "-u", help="URL to input image")
    parser.add_argument("--patch-size", type=int, default=32, help="Size of occlusion patch")
    parser.add_argument("--stride", type=int, default=16, help="Stride for patch movement")
    parser.add_argument("--output-2d", "-o2", help="Output file for 2D visualization")
    parser.add_argument("--output-3d", "-o3", help="Output file for 3D visualization")

    _args = parser.parse_args()
    raise SystemExit(run_occlusion_analysis_command(_args))


