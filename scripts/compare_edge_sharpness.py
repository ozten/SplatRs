#!/usr/bin/env python3
"""
Compare edge sharpness between rendered views and ground truth.

Usage:
    python compare_edge_sharpness.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --num-views 10

Example:
    python compare_edge_sharpness.py \
        --model runs/edge_test_bicycle/model_final.gs \
        --dataset-root datasets/bicycle \
        --output runs/edge_test_bicycle/edge_comparison \
        --num-views 10

This script:
1. Selects validation views from the dataset
2. Renders those views using the trained model
3. Applies edge detection (Sobel, Canny) to both rendered and ground truth
4. Computes edge quality metrics
5. Generates visualizations for manual inspection
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Tuple

try:
    import cv2
    from PIL import Image
except ImportError:
    print("Error: Required packages not found.", file=sys.stderr)
    print("Install with: pip install opencv-python pillow", file=sys.stderr)
    sys.exit(1)

# Import COLMAP utilities
try:
    from colmap_utils import get_image_to_camera_mapping, read_colmap_cameras, read_colmap_images
except ImportError:
    print("Error: colmap_utils.py not found.", file=sys.stderr)
    print("Make sure colmap_utils.py is in the same directory as this script.", file=sys.stderr)
    sys.exit(1)


def load_test_split(dataset_root: Path) -> List[str]:
    """Load test/validation split from dataset."""
    # Mip-NeRF 360 datasets typically use every 8th image for testing
    # Try to load split file first, otherwise use convention

    split_file = dataset_root / "test.txt"
    if split_file.exists():
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]

    # Fallback: use every 8th image
    images_dir = dataset_root / "images"
    if not images_dir.exists():
        return []

    all_images = sorted([p.name for p in images_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']])
    return [img for i, img in enumerate(all_images) if i % 8 == 0]


def load_colmap_data(dataset_root: Path) -> Tuple[Dict[str, int], Dict[int, Dict]]:
    """
    Load COLMAP image-to-camera mapping and camera parameters.

    Returns:
        Tuple of (image_to_camera, cameras):
        - image_to_camera: Dict mapping image_name to camera_id
        - cameras: Dict mapping camera_id to camera parameters
    """
    sparse_dir = dataset_root / "sparse" / "0"

    if not sparse_dir.exists():
        print(f"Warning: COLMAP sparse dir not found: {sparse_dir}", file=sys.stderr)
        return {}, {}

    images_path = sparse_dir / "images.bin"
    cameras_path = sparse_dir / "cameras.bin"

    if not images_path.exists() or not cameras_path.exists():
        print(f"Warning: COLMAP files not found in {sparse_dir}", file=sys.stderr)
        return {}, {}

    try:
        image_to_camera = get_image_to_camera_mapping(images_path)
        cameras = read_colmap_cameras(cameras_path)
        return image_to_camera, cameras
    except Exception as e:
        print(f"Error loading COLMAP data: {e}", file=sys.stderr)
        return {}, {}


def compute_edge_map(image: np.ndarray, method: str = 'sobel') -> np.ndarray:
    """Compute edge map using specified method."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if method == 'sobel':
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to [0, 255]
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return magnitude

    elif method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        return edges

    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def compute_edge_metrics(rendered: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute edge quality metrics."""
    # Compute edge maps
    edge_rendered = compute_edge_map(rendered, method='sobel')
    edge_gt = compute_edge_map(ground_truth, method='sobel')

    # Metric 1: Mean gradient magnitude
    mean_grad_rendered = np.mean(edge_rendered)
    mean_grad_gt = np.mean(edge_gt)
    gradient_ratio = mean_grad_rendered / (mean_grad_gt + 1e-6)

    # Metric 2: Edge preservation ratio (using threshold)
    threshold = 30  # Gradient magnitude threshold for "edge"
    edges_rendered = (edge_rendered > threshold).sum()
    edges_gt = (edge_gt > threshold).sum()
    preservation_ratio = edges_rendered / (edges_gt + 1e-6)

    # Metric 3: Edge localization accuracy (mean absolute error in edge maps)
    edge_mae = np.mean(np.abs(edge_rendered.astype(float) - edge_gt.astype(float)))

    # Metric 4: Structural similarity of edge maps
    # Simple correlation coefficient
    edge_rendered_flat = edge_rendered.flatten().astype(float)
    edge_gt_flat = edge_gt.flatten().astype(float)

    edge_rendered_centered = edge_rendered_flat - np.mean(edge_rendered_flat)
    edge_gt_centered = edge_gt_flat - np.mean(edge_gt_flat)

    correlation = np.sum(edge_rendered_centered * edge_gt_centered) / (
        np.sqrt(np.sum(edge_rendered_centered**2)) *
        np.sqrt(np.sum(edge_gt_centered**2)) + 1e-6
    )

    return {
        'mean_gradient_rendered': float(mean_grad_rendered),
        'mean_gradient_gt': float(mean_grad_gt),
        'gradient_ratio': float(gradient_ratio),
        'preservation_ratio': float(preservation_ratio),
        'edge_mae': float(edge_mae),
        'edge_correlation': float(correlation),
    }


def create_comparison_visualization(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    view_name: str
):
    """Create side-by-side visualization with edge maps."""
    # Compute edge maps
    edge_rendered_sobel = compute_edge_map(rendered, method='sobel')
    edge_gt_sobel = compute_edge_map(ground_truth, method='sobel')

    edge_rendered_canny = compute_edge_map(rendered, method='canny')
    edge_gt_canny = compute_edge_map(ground_truth, method='canny')

    # Compute difference
    edge_diff = np.abs(edge_rendered_sobel.astype(float) - edge_gt_sobel.astype(float))
    edge_diff = np.clip(edge_diff, 0, 255).astype(np.uint8)

    # Convert edge maps to RGB for visualization
    edge_rendered_rgb = cv2.cvtColor(edge_rendered_sobel, cv2.COLOR_GRAY2RGB)
    edge_gt_rgb = cv2.cvtColor(edge_gt_sobel, cv2.COLOR_GRAY2RGB)
    edge_diff_rgb = cv2.cvtColor(edge_diff, cv2.COLOR_GRAY2RGB)

    # Apply color map to difference (hot colormap)
    edge_diff_colored = cv2.applyColorMap(edge_diff, cv2.COLORMAP_HOT)

    # Create grid: [rendered, ground_truth] on top, [edge_rendered, edge_gt, edge_diff] on bottom
    h, w = rendered.shape[:2]

    # Top row: original images
    top_row = np.hstack([rendered, ground_truth])

    # Bottom row: edge maps
    bottom_row = np.hstack([edge_rendered_rgb, edge_gt_rgb, edge_diff_colored])

    # Resize bottom row to match top row width
    bottom_h = int(h * bottom_row.shape[1] / top_row.shape[1])
    bottom_row_resized = cv2.resize(bottom_row, (top_row.shape[1], bottom_h))

    # Combine rows
    combined = np.vstack([top_row, bottom_row_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Top row labels
    cv2.rectangle(combined, (10, 10), (200, 40), bg_color, -1)
    cv2.putText(combined, "Rendered", (15, 32), font, font_scale, text_color, font_thickness)

    cv2.rectangle(combined, (w + 10, 10), (w + 200, 40), bg_color, -1)
    cv2.putText(combined, "Ground Truth", (w + 15, 32), font, font_scale, text_color, font_thickness)

    # Bottom row labels
    bottom_y = h
    col_width = combined.shape[1] // 3

    cv2.rectangle(combined, (10, bottom_y + 10), (220, bottom_y + 40), bg_color, -1)
    cv2.putText(combined, "Edges (Rendered)", (15, bottom_y + 32), font, font_scale, text_color, font_thickness)

    cv2.rectangle(combined, (col_width + 10, bottom_y + 10), (col_width + 200, bottom_y + 40), bg_color, -1)
    cv2.putText(combined, "Edges (GT)", (col_width + 15, bottom_y + 32), font, font_scale, text_color, font_thickness)

    cv2.rectangle(combined, (2*col_width + 10, bottom_y + 10), (2*col_width + 200, bottom_y + 40), bg_color, -1)
    cv2.putText(combined, "Edge Diff", (2*col_width + 15, bottom_y + 32), font, font_scale, text_color, font_thickness)

    # Save visualization
    output_file = output_path / f"comparison_{view_name}.png"
    cv2.imwrite(str(output_file), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"  Saved visualization: {output_file}")


def render_view(model_path: Path, dataset_root: Path, view_name: str, output_path: Path) -> bool:
    """Render a specific view using sugar-render."""
    # Note: This assumes sugar-render supports rendering specific views by name
    # Adjust based on actual sugar-render API

    print(f"  TODO: Implement rendering for view {view_name}")
    print(f"    Model: {model_path}")
    print(f"    Output: {output_path}")

    # Placeholder command (adjust based on actual sugar-render API)
    # cmd = [
    #     "./target/release/sugar-render",
    #     "--model", str(model_path),
    #     "--dataset-root", str(dataset_root),
    #     "--view", view_name,
    #     "--output", str(output_path),
    # ]
    # result = subprocess.run(cmd, capture_output=True)
    # return result.returncode == 0

    return False  # Placeholder


def main():
    parser = argparse.ArgumentParser(description="Compare edge sharpness between renders and ground truth")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.gs file)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory for comparison")
    parser.add_argument("--num-views", type=int, default=10, help="Number of views to evaluate (default: 10)")

    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)

    # Validate inputs
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Number of views: {args.num_views}\n")

    # Load COLMAP data
    print("Loading COLMAP data...")
    image_to_camera, cameras = load_colmap_data(dataset_root)

    if image_to_camera and cameras:
        print(f"  Loaded {len(image_to_camera)} image-to-camera mappings")
        print(f"  Loaded {len(cameras)} cameras")
        # Show sample mappings
        for i, (img_name, cam_id) in enumerate(list(image_to_camera.items())[:3]):
            cam = cameras.get(cam_id, {})
            print(f"    {img_name} -> camera {cam_id} "
                  f"({cam.get('width', '?')}x{cam.get('height', '?')})")
    else:
        print("  ⚠ COLMAP data not available - camera info will not be used")

    # Load test split
    print("\nLoading test split...")
    test_views = load_test_split(dataset_root)

    if not test_views:
        print("Error: No test views found in dataset", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(test_views)} test views")

    # Select subset of views to evaluate
    num_views = min(args.num_views, len(test_views))
    step = max(1, len(test_views) // num_views)
    selected_views = test_views[::step][:num_views]

    print(f"Selected {len(selected_views)} views for evaluation\n")

    # Process each view
    all_metrics = []

    for i, view_name in enumerate(selected_views):
        print(f"Processing view {i+1}/{len(selected_views)}: {view_name}")

        # Ground truth image path
        gt_path = dataset_root / "images" / view_name
        if not gt_path.exists():
            print(f"  Warning: Ground truth not found: {gt_path}")
            continue

        # Get camera info for this image
        camera_id = image_to_camera.get(view_name)
        if camera_id is not None:
            camera = cameras.get(camera_id, {})
            print(f"  Camera {camera_id}: {camera.get('width', '?')}x{camera.get('height', '?')}, "
                  f"fx={camera.get('fx', '?'):.1f}, fy={camera.get('fy', '?'):.1f}")
        else:
            print(f"  ⚠ No camera mapping found for {view_name}")

        # Render view
        rendered_path = output_dir / f"rendered_{i:03d}.png"

        print("  ⚠ Rendering not yet implemented (placeholder)")
        print(f"    Would render: {view_name} -> {rendered_path}")

        # For now, skip rendering and just document the workflow
        # In actual implementation:
        # - render_view(model_path, dataset_root, view_name, rendered_path)
        # - Load rendered and ground truth images
        # - Compute metrics
        # - Create visualizations

    print("\n⚠ NOTE: This script currently contains placeholder logic.")
    print("   The actual rendering and comparison needs to be implemented.")
    print("   Key requirements:")
    print("   1. sugar-render must support rendering specific views by name")
    print("   2. OpenCV and PIL must be installed: pip install opencv-python pillow")
    print("   3. Test split loading must match dataset format")

    print("\nNext steps after implementation:")
    print("1. Review comparison visualizations in output directory")
    print("2. Check edge quality metrics in metrics.json")
    print("3. Assess if edges are comparable to ground truth")
    print("4. Pass/Fail criteria:")
    print("   - Pass: gradient_ratio > 0.8, preservation_ratio > 0.8")
    print("   - Marginal: gradient_ratio > 0.7, preservation_ratio > 0.7")
    print("   - Fail: gradient_ratio < 0.7 or preservation_ratio < 0.7")


if __name__ == "__main__":
    main()
