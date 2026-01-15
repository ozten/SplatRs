//! TC-E2E-011: Edge Sharpness
//!
//! This test verifies that sharp edges in the scene are reconstructed without excessive blur
//! or over-smoothing. Edge quality is critical for photo-realistic rendering.
//!
//! Method:
//! - Render test views with known sharp edges (e.g., building corners, object boundaries)
//! - Compare edge quality against ground truth images
//! - Compute edge detection metrics (gradient magnitude, edge preservation)
//! - Manual visual inspection of edge sharpness
//!
//! Pass Criteria:
//! - Edges visually comparable to ground truth
//! - No systematic over-smoothing
//! - Edge gradient magnitudes within acceptable range of ground truth
//!
//! Severity: Medium
//!
//! ## How to Run This Test
//!
//! This is a MANUAL verification test that requires visual inspection of rendered output
//! and comparison against ground truth images.
//!
//! ### Quick Verification (Single Scene)
//!
//! For development/testing, use a small scene from Mip-NeRF 360:
//!
//! ```bash
//! # Step 1: Train a model (or use existing trained model)
//! cargo build --release
//! ./target/release/sugar-train \
//!   --preset full \
//!   --dataset-root datasets/bicycle \
//!   --out-dir runs/edge_test_bicycle
//!
//! # Step 2: Run edge comparison script generation test
//! cargo test --release tc_e2e_011_edge_generate_comparison_script -- --nocapture --ignored
//!
//! # Step 3: Render test views and compare edges
//! python scripts/compare_edge_sharpness.py \
//!   --model runs/edge_test_bicycle/model_final.gs \
//!   --dataset-root datasets/bicycle \
//!   --output runs/edge_test_bicycle/edge_comparison
//!
//! # Step 4: Manual inspection
//! # Review the output images:
//! # - rendered_*.png: Rendered views
//! # - ground_truth_*.png: Ground truth images
//! # - edge_rendered_*.png: Edge maps from rendered views
//! # - edge_gt_*.png: Edge maps from ground truth
//! # - edge_diff_*.png: Difference between edge maps
//! ```
//!
//! ### Full Verification (All Scenes)
//!
//! For comprehensive testing, evaluate edge sharpness across all Mip-NeRF 360 scenes:
//!
//! ```bash
//! # Generate edge comparison reports for all trained models
//! for scene in bicycle garden stump room counter kitchen bonsai; do
//!   echo "Evaluating edge sharpness for $scene..."
//!   python scripts/compare_edge_sharpness.py \
//!     --model runs/e2e_001_${scene}/model_final.gs \
//!     --dataset-root datasets/$scene \
//!     --output runs/e2e_001_${scene}/edge_comparison
//! done
//! ```
//!
//! ### Edge Quality Metrics
//!
//! The comparison script computes several edge quality metrics:
//!
//! 1. **Edge Gradient Magnitude**
//!    - Sobel filter applied to rendered and ground truth images
//!    - Compare mean gradient magnitudes
//!    - Strong edges should have similar magnitudes
//!
//! 2. **Edge Preservation Ratio**
//!    - Ratio of detected edges in rendered vs. ground truth
//!    - Values close to 1.0 indicate good edge preservation
//!    - Values < 0.8 suggest over-smoothing
//!
//! 3. **Edge Localization Accuracy**
//!    - Distance between detected edges in rendered and ground truth
//!    - Small distances indicate precise edge localization
//!    - Large distances suggest geometric inaccuracies
//!
//! ### Manual Inspection Checklist
//!
//! When reviewing edge comparisons, check for:
//!
//! 1. **Sharpness**
//!    - Crisp boundaries between objects and background
//!    - Well-defined edges on geometric features (corners, contours)
//!    - No excessive blur or feathering at edges
//!
//! 2. **Edge Preservation**
//!    - All major edges present in ground truth are visible in renders
//!    - No missing edges due to over-smoothing
//!    - Fine details (thin structures, small features) are preserved
//!
//! 3. **Systematic Issues**
//!    - Check if certain edge types are consistently blurred (e.g., high-contrast edges)
//!    - Look for direction-dependent blur (e.g., horizontal vs. vertical edges)
//!    - Check for depth-dependent blur (near vs. far edges)
//!
//! 4. **Severity Assessment**
//!    - **Pass**: Edges comparable to ground truth, minor differences acceptable
//!    - **Marginal**: Noticeable softening but overall structure preserved
//!    - **Fail**: Significant blur, missing edges, or systematic over-smoothing
//!
//! ## Implementation Notes
//!
//! This test provides infrastructure for edge sharpness evaluation but requires manual
//! visual inspection because:
//! - Edge quality is subjective and context-dependent
//! - Different scenes have different edge characteristics (indoor vs. outdoor, textures, etc.)
//! - Quantitative metrics don't always correlate with perceptual quality
//! - Manual inspection is standard practice in novel view synthesis evaluation
//!
//! The test generates helper scripts for:
//! - Rendering test views from the validation set
//! - Edge detection and comparison (Sobel, Canny edge detectors)
//! - Quantitative edge quality metrics
//! - Side-by-side visualization for manual inspection
//!
//! ## Current Status
//!
//! This test provides:
//! 1. Dataset verification (checks that test datasets exist)
//! 2. Script generation for edge comparison and evaluation
//! 3. Documentation for manual inspection workflow
//! 4. Quantitative metrics for edge quality assessment

use std::path::PathBuf;

/// Expected Mip-NeRF 360 scenes for testing
const TEST_SCENES: &[(&str, &str)] = &[
    ("bicycle", "outdoor"),
    ("garden", "outdoor"),
    ("stump", "outdoor"),
    ("room", "indoor"),
    ("counter", "indoor"),
    ("kitchen", "indoor"),
    ("bonsai", "indoor"),
];

/// TC-E2E-011: Edge Sharpness - Dataset Verification
///
/// This test verifies that test datasets are available for edge sharpness testing.
///
/// Pass Criteria:
/// - At least one Mip-NeRF 360 scene exists in datasets/
/// - Scene has required COLMAP structure and images
#[test]
fn tc_e2e_011_edge_dataset_verification() {
    println!("\n=== TC-E2E-011: Edge Sharpness - Dataset Verification ===\n");

    let mut available_scenes = Vec::new();

    for (scene, scene_type) in TEST_SCENES {
        let scene_path = PathBuf::from("datasets").join(scene);
        if !scene_path.exists() {
            continue;
        }

        let sparse_path = scene_path.join("sparse/0");
        let images_path = scene_path.join("images");

        if !sparse_path.exists() || !images_path.exists() {
            continue;
        }

        let cameras_bin = sparse_path.join("cameras.bin");
        let images_bin = sparse_path.join("images.bin");

        if cameras_bin.exists() && images_bin.exists() {
            available_scenes.push((*scene, *scene_type));
            println!("✓ {} ({}) - dataset available", scene, scene_type);
        }
    }

    if available_scenes.is_empty() {
        println!("\n⚠ No test datasets found.");
        println!("Download Mip-NeRF 360 from: http://storage.googleapis.com/gresearch/refraw360/360_v2.zip");
        panic!("At least one test dataset is required for edge sharpness testing.");
    }

    println!("\n✓ Found {} available scenes for edge sharpness testing", available_scenes.len());
    println!("\nNext steps:");
    println!("1. Train models using: cargo test tc_e2e_001_mipnerf360_quick_verify -- --ignored");
    println!("2. Generate edge comparison script: cargo test tc_e2e_011_edge_generate_comparison_script -- --ignored");
    println!("3. Run edge comparison and manually inspect results");
}

/// TC-E2E-011: Generate Edge Sharpness Comparison Script
///
/// This test generates a Python script for comparing edge sharpness between rendered views
/// and ground truth images. The script renders validation views, applies edge detection,
/// and computes edge quality metrics.
///
/// Run with: cargo test --release tc_e2e_011_edge_generate_comparison_script -- --nocapture --ignored
#[test]
#[ignore]
fn tc_e2e_011_edge_generate_comparison_script() {
    println!("\n=== TC-E2E-011: Generating Edge Sharpness Comparison Script ===\n");

    let script_content = r#"#!/usr/bin/env python3
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

    # Load test split
    print("Loading test split...")
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
"#;

    let script_dir = PathBuf::from("scripts");
    std::fs::create_dir_all(&script_dir).expect("Failed to create scripts directory");

    let script_path = script_dir.join("compare_edge_sharpness.py");
    std::fs::write(&script_path, script_content).expect("Failed to write edge comparison script");

    // Make script executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script_path)
            .expect("Failed to get file metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("Failed to set permissions");
    }

    println!("✓ Generated edge sharpness comparison script: {}", script_path.display());
    println!("\nUsage:");
    println!("  python scripts/compare_edge_sharpness.py \\");
    println!("    --model runs/edge_test_bicycle/model_final.gs \\");
    println!("    --dataset-root datasets/bicycle \\");
    println!("    --output runs/edge_test_bicycle/edge_comparison \\");
    println!("    --num-views 10");
    println!("\nNote: The script contains placeholder rendering logic.");
    println!("      Implementation depends on sugar-render supporting view-specific rendering.");
    println!("\nAfter running:");
    println!("  1. Review comparison visualizations: comparison_*.png");
    println!("  2. Check edge quality metrics: metrics.json");
    println!("  3. Assess if edges are comparable to ground truth");
}
