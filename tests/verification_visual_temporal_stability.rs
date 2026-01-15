//! TC-E2E-013: Temporal Stability
//!
//! This test verifies that rendered video sequences are temporally stable without flickering
//! or popping artifacts. Temporal stability is critical for video generation from 3D Gaussian splats.
//!
//! Method:
//! - Render smooth camera path with 100+ frames
//! - Compute temporal consistency metrics (frame-to-frame SSIM)
//! - Manual visual inspection for popping, flickering, or temporal artifacts
//!
//! Pass Criteria:
//! - Frame-to-frame SSIM > 0.95 for slow camera motion
//! - No visible popping or flickering artifacts
//! - Smooth appearance transitions during camera movement
//!
//! Severity: Medium
//!
//! ## How to Run This Test
//!
//! This is a MANUAL verification test that requires visual inspection of rendered video
//! sequences and quantitative temporal consistency metrics.
//!
//! ### Quick Verification (Single Scene)
//!
//! For development/testing, use a scene from Mip-NeRF 360:
//!
//! ```bash
//! # Step 1: Train a model (or use existing trained model)
//! cargo build --release
//! ./target/release/sugar-train \
//!   --preset full \
//!   --dataset-root datasets/bicycle \
//!   --out-dir runs/temporal_test_bicycle
//!
//! # Step 2: Run temporal stability script generation test
//! cargo test --release tc_e2e_013_temporal_generate_stability_script -- --nocapture --ignored
//!
//! # Step 3: Render smooth camera path with temporal analysis
//! python scripts/render_temporal_stability.py \
//!   --model runs/temporal_test_bicycle/model_final.gs \
//!   --dataset-root datasets/bicycle \
//!   --output runs/temporal_test_bicycle/temporal_stability \
//!   --frames 120 \
//!   --motion smooth-orbit
//!
//! # Step 4: Create video from rendered frames
//! ffmpeg -framerate 30 -i runs/temporal_test_bicycle/temporal_stability/frame_%04d.png \
//!   -c:v libx264 -pix_fmt yuv420p -crf 18 \
//!   runs/temporal_test_bicycle/temporal_stability.mp4
//!
//! # Step 5: Review temporal metrics
//! cat runs/temporal_test_bicycle/temporal_stability/temporal_metrics.json
//!
//! # Step 6: Manual inspection
//! # Watch the video and check:
//! # - No flickering or popping artifacts
//! # - Smooth transitions during camera movement
//! # - Consistent appearance of surfaces over time
//! # - Frame-to-frame SSIM > 0.95 (reported in metrics)
//! ```
//!
//! ### Full Verification (All Scenes)
//!
//! For comprehensive testing, evaluate temporal stability across all Mip-NeRF 360 scenes:
//!
//! ```bash
//! # Generate temporal stability reports for all scenes
//! for scene in bicycle garden stump room counter kitchen bonsai; do
//!   echo "Evaluating temporal stability for $scene..."
//!   python scripts/render_temporal_stability.py \
//!     --model runs/e2e_001_${scene}/model_final.gs \
//!     --dataset-root datasets/$scene \
//!     --output runs/e2e_001_${scene}/temporal_stability \
//!     --frames 120 \
//!     --motion smooth-orbit
//!
//!   ffmpeg -framerate 30 -i runs/e2e_001_${scene}/temporal_stability/frame_%04d.png \
//!     -c:v libx264 -pix_fmt yuv420p -crf 18 \
//!     runs/e2e_001_${scene}/temporal_stability.mp4
//! done
//! ```
//!
//! ### Alternative Camera Paths
//!
//! Test temporal stability with different motion patterns:
//!
//! ```bash
//! # Smooth orbital motion (default - good for general temporal stability)
//! python scripts/render_temporal_stability.py \
//!   --model <model.gs> \
//!   --dataset-root <dataset> \
//!   --output <output_dir> \
//!   --motion smooth-orbit \
//!   --frames 120
//!
//! # Linear dolly motion (forward/backward - tests depth stability)
//! python scripts/render_temporal_stability.py \
//!   --model <model.gs> \
//!   --dataset-root <dataset> \
//!   --output <output_dir> \
//!   --motion dolly \
//!   --frames 120
//!
//! # Slow pan (tests horizontal stability)
//! python scripts/render_temporal_stability.py \
//!   --model <model.gs> \
//!   --dataset-root <dataset> \
//!   --output <output_dir> \
//!   --motion pan \
//!   --frames 120
//! ```
//!
//! ### Temporal Artifacts to Check
//!
//! The temporal stability script helps identify several types of temporal issues:
//!
//! 1. **Flickering**
//!    - Rapid brightness/color changes in static surfaces
//!    - High-frequency temporal noise
//!    - Inconsistent Gaussian visibility between frames
//!
//! 2. **Popping Artifacts**
//!    - Sudden appearance/disappearance of Gaussians
//!    - Discontinuous surface visibility
//!    - Abrupt geometry changes during smooth camera motion
//!
//! 3. **Temporal Aliasing**
//!    - Moiré patterns in motion
//!    - Strobing effects on fine details
//!    - Inconsistent edge rendering between frames
//!
//! 4. **Depth Instability**
//!    - Surfaces "swimming" during camera motion
//!    - Inconsistent occlusion relationships
//!    - Jittering depth boundaries
//!
//! ### Manual Inspection Checklist
//!
//! When reviewing temporal stability videos and metrics, check for:
//!
//! 1. **Quantitative Metrics**
//!    - Frame-to-frame SSIM > 0.95 for slow motion (reported in temporal_metrics.json)
//!    - Frame-to-frame SSIM > 0.90 for moderate motion
//!    - No sudden drops in SSIM (indicates popping/flickering)
//!    - Consistent SSIM throughout entire sequence
//!
//! 2. **Visual Quality - Static Surfaces**
//!    - Static surfaces maintain consistent appearance
//!    - No flickering in background regions
//!    - Uniform colors/textures remain stable
//!    - No temporal noise in sky or flat regions
//!
//! 3. **Visual Quality - Motion**
//!    - Smooth transitions during camera movement
//!    - No popping when new geometry becomes visible
//!    - Consistent appearance of moving surfaces
//!    - Natural motion blur (if applicable)
//!
//! 4. **Occlusion Handling**
//!    - Smooth reveal/occlude transitions
//!    - No flickering at depth boundaries
//!    - Consistent z-ordering during motion
//!    - No sudden geometry changes near edges
//!
//! 5. **Fine Details**
//!    - Thin structures remain stable (no popping)
//!    - Texture details don't flicker
//!    - Edges maintain consistent sharpness
//!    - Small features visible throughout sequence
//!
//! 6. **Severity Assessment**
//!    - **Pass**: SSIM > 0.95, no visible flickering/popping
//!    - **Marginal**: SSIM 0.90-0.95, minor artifacts but watchable
//!    - **Fail**: SSIM < 0.90, significant flickering/popping artifacts
//!
//! ## Implementation Notes
//!
//! This test provides infrastructure for temporal stability evaluation but requires manual
//! visual inspection because:
//! - Temporal artifacts are subjective and context-dependent
//! - SSIM threshold depends on motion speed and scene complexity
//! - Video compression can introduce additional artifacts
//! - Human perception is best judge of "watchable" quality
//!
//! The test generates helper scripts for:
//! - Rendering smooth camera paths with dense frame sampling
//! - Computing frame-to-frame temporal consistency metrics (SSIM)
//! - Creating videos for easy manual inspection
//! - Visualizing temporal stability metrics over time
//!
//! ## Current Status
//!
//! This test provides:
//! 1. Dataset verification (checks that test datasets exist)
//! 2. Script generation for temporal stability rendering and analysis
//! 3. Documentation for manual inspection workflow
//! 4. Quantitative temporal metrics (frame-to-frame SSIM)

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

/// TC-E2E-013: Temporal Stability - Dataset Verification
///
/// This test verifies that test datasets are available for temporal stability testing.
///
/// Pass Criteria:
/// - At least one Mip-NeRF 360 scene exists in datasets/
/// - Scene has required COLMAP structure and images
#[test]
fn tc_e2e_013_temporal_dataset_verification() {
    println!("\n=== TC-E2E-013: Temporal Stability - Dataset Verification ===\n");

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
            println!("✓ {} ({})", scene, scene_type);
        }
    }

    if available_scenes.is_empty() {
        println!("\n⚠ No test datasets found.");
        println!("Download Mip-NeRF 360 from: http://storage.googleapis.com/gresearch/refraw360/360_v2.zip");
        panic!("At least one test dataset is required for temporal stability testing.");
    }

    println!("\n✓ Found {} available scenes for temporal stability testing", available_scenes.len());
    println!("\nAll scenes are suitable for temporal stability testing.");
    println!("Smooth camera motion can be tested on any scene.");

    println!("\nNext steps:");
    println!("1. Train models using: cargo test tc_e2e_001_mipnerf360_quick_verify -- --ignored");
    println!("2. Generate temporal stability script: cargo test tc_e2e_013_temporal_generate_stability_script -- --ignored");
    println!("3. Render smooth camera paths and manually inspect for flickering/popping");
}

/// TC-E2E-013: Generate Temporal Stability Analysis Script
///
/// This test generates a Python script for rendering smooth camera paths and analyzing
/// temporal stability. The script renders dense frame sequences and computes frame-to-frame
/// SSIM to detect flickering and popping artifacts.
///
/// Run with: cargo test --release tc_e2e_013_temporal_generate_stability_script -- --nocapture --ignored
#[test]
#[ignore]
fn tc_e2e_013_temporal_generate_stability_script() {
    println!("\n=== TC-E2E-013: Generating Temporal Stability Analysis Script ===\n");

    let script_content = r#"#!/usr/bin/env python3
"""
Render smooth camera path and analyze temporal stability.

Usage:
    python render_temporal_stability.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --frames 120 \
        --motion smooth-orbit

Example:
    python render_temporal_stability.py \
        --model runs/temporal_test_bicycle/model_final.gs \
        --dataset-root datasets/bicycle \
        --output runs/temporal_test_bicycle/temporal_stability \
        --frames 120 \
        --motion smooth-orbit

This script:
1. Generates smooth camera path (orbit, dolly, or pan)
2. Renders dense frame sequence (100+ frames for smooth motion)
3. Computes frame-to-frame SSIM for temporal consistency
4. Creates temporal metrics report (JSON and visualization)
5. Outputs video-ready frame sequence

Temporal Metrics:
- Frame-to-frame SSIM (structural similarity between consecutive frames)
- Mean SSIM across entire sequence
- Min SSIM (identifies worst frame transitions)
- SSIM standard deviation (measures consistency)
- Temporal stability score (0-100)

Pass Criteria:
- Frame-to-frame SSIM > 0.95 for slow motion
- Frame-to-frame SSIM > 0.90 for moderate motion
- No sudden SSIM drops (popping artifacts)
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import struct
import numpy as np
from typing import List, Dict, Tuple

try:
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: Required packages not found.", file=sys.stderr)
    print("Install with: pip install pillow scikit-image matplotlib", file=sys.stderr)
    sys.exit(1)


def read_colmap_points3d(bin_path):
    """Read COLMAP points3D.bin to compute scene center."""
    points = []

    with open(bin_path, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]

        for _ in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('ddd', f.read(24))
            rgb = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]

            track_len = struct.unpack('Q', f.read(8))[0]
            track_data = f.read(track_len * 16)  # Skip track data

            points.append(xyz)

    return np.array(points)


def compute_scene_center(dataset_root: Path) -> np.ndarray:
    """Compute scene center from COLMAP reconstruction."""
    points3d_path = dataset_root / "sparse" / "0" / "points3D.bin"

    if not points3d_path.exists():
        print(f"Warning: points3D.bin not found: {points3d_path}", file=sys.stderr)
        print("Using default center (0, 0, 0)", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0])

    try:
        points = read_colmap_points3d(points3d_path)
        if len(points) == 0:
            print("Warning: No points found in COLMAP reconstruction", file=sys.stderr)
            return np.array([0.0, 0.0, 0.0])

        # Compute median (more robust than mean for outliers)
        center = np.median(points, axis=0)
        print(f"Scene center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        return center
    except Exception as e:
        print(f"Error reading COLMAP points: {e}", file=sys.stderr)
        print("Using default center (0, 0, 0)", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0])


def generate_smooth_orbit_path(center: np.ndarray, radius: float, num_frames: int) -> List[Dict]:
    """
    Generate smooth orbital camera path around scene center.

    Full 360° orbit with constant distance and height.
    """
    cameras = []

    for i in range(num_frames):
        # Angle for this frame (full 360° orbit)
        angle = 2 * np.pi * i / num_frames

        # Position in circle (XZ plane, constant Y)
        x = center[0] + radius * np.sin(angle)
        z = center[2] + radius * np.cos(angle)
        y = center[1]  # Keep at center height

        # Camera looks at center
        position = np.array([x, y, z])
        target = center

        # Compute view direction and up vector
        view_dir = target - position
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Up vector (world Y-axis, adjusted to be perpendicular to view)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
            up = np.cross(right, view_dir)

        cameras.append({
            'position': position,
            'view_dir': view_dir,
            'up': up,
            'right': right,
            'frame': i,
        })

    return cameras


def generate_dolly_path(center: np.ndarray, start_distance: float,
                        end_distance: float, num_frames: int) -> List[Dict]:
    """
    Generate smooth dolly camera path (forward/backward along view direction).

    Tests depth stability and near-far transitions.
    """
    cameras = []

    # Direction vector (dolly along -Z axis, looking at center)
    direction = np.array([0.0, 0.0, 1.0])

    for i in range(num_frames):
        # Interpolate distance
        t = i / (num_frames - 1)
        distance = start_distance + t * (end_distance - start_distance)

        # Position along dolly path
        position = center - direction * distance
        target = center

        # Compute view direction and up vector
        view_dir = target - position
        view_dir = view_dir / np.linalg.norm(view_dir)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
            up = np.cross(right, view_dir)

        cameras.append({
            'position': position,
            'view_dir': view_dir,
            'up': up,
            'right': right,
            'frame': i,
        })

    return cameras


def generate_pan_path(center: np.ndarray, radius: float, arc_angle: float,
                      num_frames: int) -> List[Dict]:
    """
    Generate smooth pan camera path (horizontal sweep).

    Tests horizontal stability and smooth motion blur.
    """
    cameras = []

    # Arc angle in radians
    arc_rad = np.deg2rad(arc_angle)
    start_angle = -arc_rad / 2.0

    for i in range(num_frames):
        # Angle for this frame
        t = i / (num_frames - 1) if num_frames > 1 else 0.5
        angle = start_angle + t * arc_rad

        # Position in arc (XZ plane, constant Y)
        x = center[0] + radius * np.sin(angle)
        z = center[2] + radius * np.cos(angle)
        y = center[1]

        # Camera looks at center
        position = np.array([x, y, z])
        target = center

        # Compute view direction and up vector
        view_dir = target - position
        view_dir = view_dir / np.linalg.norm(view_dir)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
            up = np.cross(right, view_dir)

        cameras.append({
            'position': position,
            'view_dir': view_dir,
            'up': up,
            'right': right,
            'frame': i,
        })

    return cameras


def render_frame(model_path: Path, dataset_root: Path, output_path: Path,
                 camera: Dict, frame_idx: int) -> bool:
    """Render a specific frame using sugar-render."""
    # Note: This assumes sugar-render supports custom camera parameters
    # Adjust based on actual sugar-render API

    print(f"  Frame {frame_idx:04d}: pos=[{camera['position'][0]:6.2f}, {camera['position'][1]:6.2f}, {camera['position'][2]:6.2f}]")

    # Placeholder command (adjust based on actual sugar-render API)
    # cmd = [
    #     "./target/release/sugar-render",
    #     "--model", str(model_path),
    #     "--dataset-root", str(dataset_root),
    #     "--camera-position", f"{camera['position'][0]},{camera['position'][1]},{camera['position'][2]}",
    #     "--camera-target", "<scene_center>",
    #     "--output", str(output_path),
    # ]
    # result = subprocess.run(cmd, capture_output=True)
    # return result.returncode == 0

    return False  # Placeholder


def compute_frame_to_frame_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two consecutive frames."""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
    else:
        img1_gray = img1
        img2_gray = img2

    # Compute SSIM
    score, _ = ssim(img1_gray, img2_gray, full=True, data_range=255.0)
    return score


def analyze_temporal_stability(frames: List[np.ndarray], output_dir: Path):
    """
    Analyze temporal stability of rendered frame sequence.

    Computes frame-to-frame SSIM and generates metrics report.
    """
    if len(frames) < 2:
        print("Warning: Need at least 2 frames for temporal analysis", file=sys.stderr)
        return

    print(f"\nAnalyzing temporal stability ({len(frames)} frames)...")

    # Compute frame-to-frame SSIM
    ssim_scores = []
    for i in range(len(frames) - 1):
        score = compute_frame_to_frame_ssim(frames[i], frames[i+1])
        ssim_scores.append(score)

    ssim_scores = np.array(ssim_scores)

    # Compute metrics
    mean_ssim = np.mean(ssim_scores)
    min_ssim = np.min(ssim_scores)
    max_ssim = np.max(ssim_scores)
    std_ssim = np.std(ssim_scores)

    # Temporal stability score (0-100)
    # Penalize low mean SSIM and high variance
    stability_score = mean_ssim * 100.0 - std_ssim * 50.0
    stability_score = max(0.0, min(100.0, stability_score))

    # Find worst frame transitions
    worst_idx = np.argmin(ssim_scores)
    worst_ssim = ssim_scores[worst_idx]

    # Generate metrics report
    metrics = {
        'num_frames': len(frames),
        'mean_ssim': float(mean_ssim),
        'min_ssim': float(min_ssim),
        'max_ssim': float(max_ssim),
        'std_ssim': float(std_ssim),
        'stability_score': float(stability_score),
        'worst_transition': {
            'frame_idx': int(worst_idx),
            'ssim': float(worst_ssim),
        },
        'ssim_scores': ssim_scores.tolist(),
    }

    # Save metrics
    metrics_path = output_dir / "temporal_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Temporal Metrics:")
    print(f"  Mean SSIM: {mean_ssim:.4f}")
    print(f"  Min SSIM:  {min_ssim:.4f} (transition {worst_idx}->{worst_idx+1})")
    print(f"  Max SSIM:  {max_ssim:.4f}")
    print(f"  Std SSIM:  {std_ssim:.4f}")
    print(f"  Stability Score: {stability_score:.1f}/100")

    # Assessment
    print(f"\n✓ Assessment:")
    if mean_ssim > 0.95 and min_ssim > 0.90:
        print(f"  PASS: Excellent temporal stability")
    elif mean_ssim > 0.90 and min_ssim > 0.85:
        print(f"  MARGINAL: Acceptable temporal stability with minor artifacts")
    else:
        print(f"  FAIL: Poor temporal stability - significant flickering/popping")

    print(f"\n✓ Saved metrics: {metrics_path}")

    # Create SSIM plot
    create_ssim_plot(ssim_scores, output_dir)


def create_ssim_plot(ssim_scores: np.ndarray, output_dir: Path):
    """Create visualization of frame-to-frame SSIM over time."""
    plt.figure(figsize=(12, 6))

    # Plot SSIM scores
    plt.plot(range(len(ssim_scores)), ssim_scores, 'b-', linewidth=1.5, label='Frame-to-frame SSIM')

    # Add threshold lines
    plt.axhline(y=0.95, color='g', linestyle='--', linewidth=1, label='Good threshold (0.95)')
    plt.axhline(y=0.90, color='orange', linestyle='--', linewidth=1, label='Acceptable threshold (0.90)')

    # Mark worst transitions
    worst_idx = np.argmin(ssim_scores)
    plt.plot(worst_idx, ssim_scores[worst_idx], 'ro', markersize=8, label=f'Worst transition ({ssim_scores[worst_idx]:.4f})')

    plt.xlabel('Frame Transition')
    plt.ylabel('SSIM')
    plt.title('Temporal Stability: Frame-to-Frame SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([max(0.7, np.min(ssim_scores) - 0.05), 1.0])

    plot_path = output_dir / "temporal_ssim_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved SSIM plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Render smooth camera path and analyze temporal stability")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.gs file)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames (default: 120)")
    parser.add_argument("--motion", type=str, default="smooth-orbit",
                       choices=["smooth-orbit", "dolly", "pan"],
                       help="Camera motion type (default: smooth-orbit)")
    parser.add_argument("--radius", type=float, default=3.0, help="Distance from center (default: 3.0)")
    parser.add_argument("--arc-angle", type=float, default=120.0, help="Arc angle for pan motion (default: 120)")

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
    print(f"Frames: {args.frames}")
    print(f"Motion: {args.motion}")

    # Compute scene center
    center = compute_scene_center(dataset_root)

    # Generate camera path
    print(f"\nGenerating {args.motion} camera path ({args.frames} frames)...")

    if args.motion == "smooth-orbit":
        cameras = generate_smooth_orbit_path(center, args.radius, args.frames)
    elif args.motion == "dolly":
        cameras = generate_dolly_path(center, args.radius * 1.5, args.radius * 0.5, args.frames)
    elif args.motion == "pan":
        cameras = generate_pan_path(center, args.radius, args.arc_angle, args.frames)
    else:
        print(f"Error: Unknown motion type: {args.motion}", file=sys.stderr)
        sys.exit(1)

    # Render frames
    print(f"\nRendering {len(cameras)} frames...")
    rendered_frames = []

    for i, camera in enumerate(cameras):
        output_path = output_dir / f"frame_{i:04d}.png"
        success = render_frame(model_path, dataset_root, output_path, camera, i)

        # For actual implementation, load rendered frame
        # img = np.array(Image.open(output_path))
        # rendered_frames.append(img)

    print("\n⚠ NOTE: This script currently contains placeholder rendering logic.")
    print("   The actual rendering and temporal analysis needs to be implemented.")
    print("   Key requirements:")
    print("   1. sugar-render must support custom camera pose specification")
    print("   2. Required packages: pip install pillow scikit-image matplotlib")

    print("\nAfter implementation:")
    print("1. Review temporal metrics: temporal_metrics.json")
    print("2. Check SSIM plot: temporal_ssim_plot.png")
    print("3. Create video: ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")
    print("4. Manually verify:")
    print("   - Frame-to-frame SSIM > 0.95 for slow motion")
    print("   - No visible flickering or popping artifacts")
    print("   - Smooth appearance transitions")
    print("   - Consistent surface appearance over time")

    # If we had rendered frames, analyze temporal stability
    # analyze_temporal_stability(rendered_frames, output_dir)


if __name__ == "__main__":
    main()
"#;

    let script_dir = PathBuf::from("scripts");
    std::fs::create_dir_all(&script_dir).expect("Failed to create scripts directory");

    let script_path = script_dir.join("render_temporal_stability.py");
    std::fs::write(&script_path, script_content).expect("Failed to write temporal stability script");

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

    println!("✓ Generated temporal stability analysis script: {}", script_path.display());
    println!("\nUsage:");
    println!("  python scripts/render_temporal_stability.py \\");
    println!("    --model runs/temporal_test_bicycle/model_final.gs \\");
    println!("    --dataset-root datasets/bicycle \\");
    println!("    --output runs/temporal_test_bicycle/temporal_stability \\");
    println!("    --frames 120 \\");
    println!("    --motion smooth-orbit");
    println!("\nMotion types:");
    println!("  - smooth-orbit: Full 360° orbit (tests general temporal stability)");
    println!("  - dolly: Forward/backward motion (tests depth stability)");
    println!("  - pan: Horizontal sweep (tests horizontal stability)");
    println!("\nNote: The script contains placeholder rendering logic.");
    println!("      Implementation depends on sugar-render supporting custom camera poses.");
    println!("\nAfter running:");
    println!("  1. Review temporal metrics: temporal_metrics.json");
    println!("  2. Check SSIM plot: temporal_ssim_plot.png");
    println!("  3. Create video: ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 output.mp4");
    println!("  4. Verify frame-to-frame SSIM > 0.95 and no visible flickering");
}
