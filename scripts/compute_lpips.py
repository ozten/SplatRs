#!/usr/bin/env python3
"""
Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.

This script uses the lpips library to compute perceptual similarity between images.
Lower LPIPS values indicate more perceptually similar images.

Usage:
    python compute_lpips.py <image1.png> <image2.png>

Output:
    Prints LPIPS score to stdout as a single floating-point number.
"""

import sys
import lpips
import torch
from PIL import Image
import numpy as np


def load_image(path):
    """Load image and convert to tensor in range [-1, 1]."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    img_np = img_np * 2.0 - 1.0  # [-1, 1]

    # Convert to tensor: (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_lpips.py <image1.png> <image2.png>", file=sys.stderr)
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    # Redirect lpips informational output to stderr
    import os
    # Save original stdout
    original_stdout = sys.stdout
    # Redirect stdout to stderr temporarily (so lpips messages don't pollute output)
    sys.stdout = sys.stderr

    # Load LPIPS model (AlexNet backbone)
    loss_fn = lpips.LPIPS(net='alex', verbose=False)

    # Restore stdout
    sys.stdout = original_stdout

    # Load images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Compute LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(img1, img2).item()

    # Output just the number
    print(f"{lpips_value:.6f}")


if __name__ == "__main__":
    main()
