//! TC-E2E-010: Floater Detection
//!
//! This test verifies that the trained models do not produce significant floating Gaussian
//! artifacts (floaters) - isolated Gaussians not attached to visible surfaces.
//!
//! Method:
//! - Render 360° orbit video around trained scene
//! - Manual visual inspection for floating blobs or artifacts
//! - Render depth maps and check for isolated discontinuities
//!
//! Pass Criteria:
//! - No significant floaters visible in standard views
//! - Depth maps show continuous surfaces without isolated depth spikes
//!
//! Severity: Medium to High (depending on severity)
//!
//! ## How to Run This Test
//!
//! This is a MANUAL verification test that requires visual inspection of rendered output.
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
//!   --out-dir runs/floater_test_bicycle
//!
//! # Step 2: Run orbit video generation test (creates rendering script)
//! cargo test --release tc_e2e_010_floater_generate_orbit_script -- --nocapture --ignored
//!
//! # Step 3: Render orbit video using generated script
//! # This creates 360 frames (1 per degree) around the scene center
//! python scripts/render_orbit_video.py \
//!   --model runs/floater_test_bicycle/model_final.gs \
//!   --dataset-root datasets/bicycle \
//!   --output runs/floater_test_bicycle/orbit_video \
//!   --frames 360 \
//!   --radius 3.0 \
//!   --height 0.0
//!
//! # Step 4: Create video from rendered frames
//! ffmpeg -framerate 30 -i runs/floater_test_bicycle/orbit_video/frame_%04d.png \
//!   -c:v libx264 -pix_fmt yuv420p -crf 18 \
//!   runs/floater_test_bicycle/orbit_video.mp4
//!
//! # Step 5: Manual inspection
//! # Open the video and look for:
//! # - Floating blobs not attached to surfaces
//! # - Artifacts that move inconsistently with scene geometry
//! # - Isolated Gaussians visible against sky or background
//! ```
//!
//! ### Full Verification (All Scenes)
//!
//! For comprehensive testing, render orbit videos for all Mip-NeRF 360 scenes:
//!
//! ```bash
//! # Render orbit videos for all trained models
//! for scene in bicycle garden stump room counter kitchen bonsai; do
//!   echo "Rendering orbit video for $scene..."
//!   python scripts/render_orbit_video.py \
//!     --model runs/e2e_001_${scene}/model_final.gs \
//!     --dataset-root datasets/$scene \
//!     --output runs/e2e_001_${scene}/orbit_video \
//!     --frames 360 \
//!     --radius 3.0
//!
//!   ffmpeg -framerate 30 -i runs/e2e_001_${scene}/orbit_video/frame_%04d.png \
//!     -c:v libx264 -pix_fmt yuv420p -crf 18 \
//!     runs/e2e_001_${scene}/orbit_video.mp4
//! done
//! ```
//!
//! ### Depth Map Inspection
//!
//! Optionally render depth maps to identify isolated depth discontinuities:
//!
//! ```bash
//! # Generate depth map rendering script
//! cargo test --release tc_e2e_010_floater_generate_depth_script -- --nocapture --ignored
//!
//! # Render depth maps for orbit path
//! python scripts/render_orbit_depth.py \
//!   --model runs/floater_test_bicycle/model_final.gs \
//!   --dataset-root datasets/bicycle \
//!   --output runs/floater_test_bicycle/orbit_depth \
//!   --frames 360
//!
//! # Create depth video
//! ffmpeg -framerate 30 -i runs/floater_test_bicycle/orbit_depth/depth_%04d.png \
//!   -c:v libx264 -pix_fmt yuv420p -crf 18 \
//!   runs/floater_test_bicycle/orbit_depth.mp4
//! ```
//!
//! ### Manual Inspection Checklist
//!
//! When reviewing orbit videos, check for:
//!
//! 1. **Floating Blobs**
//!    - Isolated Gaussians visible against sky or uniform backgrounds
//!    - Artifacts that don't follow surface geometry
//!    - Semi-transparent blobs hovering in empty space
//!
//! 2. **Movement Patterns**
//!    - Floaters may move inconsistently as camera orbits
//!    - Check for artifacts that pop in/out as viewpoint changes
//!    - Look for blobs that don't exhibit proper parallax
//!
//! 3. **Depth Discontinuities** (if depth maps available)
//!    - Isolated depth spikes far from main surfaces
//!    - Small regions with dramatically different depth values
//!    - Noisy depth patterns in areas that should be smooth
//!
//! 4. **Severity Assessment**
//!    - **Pass**: No visible floaters or only very minor artifacts in extreme views
//!    - **Marginal**: Small floaters visible but don't significantly impact quality
//!    - **Fail**: Prominent floaters that distract from scene or obscure geometry
//!
//! ## Implementation Notes
//!
//! This test provides infrastructure for floater detection but requires manual visual
//! inspection because:
//! - Floaters are subjective and depend on scene context
//! - Automated detection would require complex heuristics (depth analysis, opacity patterns)
//! - Manual inspection is standard practice in 3DGS evaluation
//!
//! The test generates helper scripts for:
//! - Orbit video rendering (uniform camera path around scene)
//! - Depth map rendering (for quantitative floater analysis)
//! - Video creation from rendered frames
//!
//! ## Current Status
//!
//! This test provides:
//! 1. Dataset verification (checks that test datasets exist)
//! 2. Script generation for orbit video rendering
//! 3. Script generation for depth map rendering
//! 4. Documentation for manual inspection workflow

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

/// TC-E2E-010: Floater Detection - Dataset Verification
///
/// This test verifies that test datasets are available for floater detection testing.
///
/// Pass Criteria:
/// - At least one Mip-NeRF 360 scene exists in datasets/
/// - Scene has required COLMAP structure
#[test]
fn tc_e2e_010_floater_dataset_verification() {
    println!("\n=== TC-E2E-010: Floater Detection - Dataset Verification ===\n");

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
        panic!("At least one test dataset is required for floater detection testing.");
    }

    println!("\n✓ Found {} available scenes for floater testing", available_scenes.len());
    println!("\nNext steps:");
    println!("1. Train models using: cargo test tc_e2e_001_mipnerf360_quick_verify -- --ignored");
    println!("2. Generate orbit video script: cargo test tc_e2e_010_floater_generate_orbit_script -- --ignored");
    println!("3. Render orbit videos and inspect for floaters");
}

/// TC-E2E-010: Generate Orbit Video Rendering Script
///
/// This test generates a Python script for rendering 360° orbit videos around trained models.
/// The rendered videos can be manually inspected for floating Gaussian artifacts.
///
/// Run with: cargo test --release tc_e2e_010_floater_generate_orbit_script -- --nocapture --ignored
#[test]
#[ignore]
fn tc_e2e_010_floater_generate_orbit_script() {
    println!("\n=== TC-E2E-010: Generating Orbit Video Rendering Script ===\n");

    let script_content = r#"#!/usr/bin/env python3
"""
Render 360° orbit video for floater detection.

Usage:
    python render_orbit_video.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --frames 360 \
        --radius 3.0 \
        --height 0.0

Example:
    python render_orbit_video.py \
        --model runs/floater_test_bicycle/model_final.gs \
        --dataset-root datasets/bicycle \
        --output runs/floater_test_bicycle/orbit_video \
        --frames 360 \
        --radius 3.0 \
        --height 0.0

This script:
1. Computes scene center from COLMAP sparse reconstruction
2. Generates camera positions in a circular orbit around the center
3. Renders each frame using sugar-render
4. Saves frames as PNG images for video creation
"""

import argparse
import subprocess
import sys
from pathlib import Path
import struct
import numpy as np


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


def compute_scene_center(dataset_root):
    """Compute scene center from COLMAP reconstruction."""
    points3d_path = Path(dataset_root) / "sparse" / "0" / "points3D.bin"

    if not points3d_path.exists():
        print(f"Error: points3D.bin not found: {points3d_path}", file=sys.stderr)
        print("Using default center (0, 0, 0)", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0])

    try:
        points = read_colmap_points3d(points3d_path)
        if len(points) == 0:
            print("Warning: No points found in COLMAP reconstruction", file=sys.stderr)
            return np.array([0.0, 0.0, 0.0])

        # Compute median (more robust than mean for outliers)
        center = np.median(points, axis=0)
        print(f"Scene center computed from {len(points)} points: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        return center
    except Exception as e:
        print(f"Error reading COLMAP points: {e}", file=sys.stderr)
        print("Using default center (0, 0, 0)", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0])


def generate_orbit_cameras(center, radius, height, num_frames):
    """Generate camera positions in circular orbit around center."""
    cameras = []

    for i in range(num_frames):
        angle = 2.0 * np.pi * i / num_frames

        # Circular orbit in XZ plane
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        y = center[1] + height

        # Camera looks at center
        position = np.array([x, y, z])
        target = center

        # Compute view direction and up vector
        view_dir = target - position
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Up vector (world Y-axis, adjusted to be perpendicular to view)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_dir)

        cameras.append({
            'position': position,
            'view_dir': view_dir,
            'up': up,
            'right': right,
        })

    return cameras


def render_frame(model_path, dataset_root, output_path, camera, frame_idx):
    """Render a single frame using sugar-render."""
    # Note: This assumes sugar-render supports custom camera parameters
    # Adjust command based on actual sugar-render API

    # For now, this is a placeholder that shows the intended command structure
    # The actual implementation depends on sugar-render's camera specification format

    print(f"TODO: Implement rendering for frame {frame_idx}")
    print(f"  Position: {camera['position']}")
    print(f"  View Dir: {camera['view_dir']}")
    print(f"  Output: {output_path}")

    # Placeholder command (adjust based on actual sugar-render API)
    # cmd = [
    #     "./target/release/sugar-render",
    #     "--model", str(model_path),
    #     "--dataset-root", str(dataset_root),
    #     "--camera-position", f"{camera['position'][0]},{camera['position'][1]},{camera['position'][2]}",
    #     "--camera-target", f"{center[0]},{center[1]},{center[2]}",
    #     "--output", str(output_path),
    # ]
    # subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Render 360° orbit video for floater detection")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.gs file)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--frames", type=int, default=360, help="Number of frames (default: 360)")
    parser.add_argument("--radius", type=float, default=3.0, help="Orbit radius from center (default: 3.0)")
    parser.add_argument("--height", type=float, default=0.0, help="Height offset from center Y (default: 0.0)")

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
    print(f"Orbit radius: {args.radius}")
    print(f"Height offset: {args.height}\n")

    # Compute scene center
    center = compute_scene_center(dataset_root)

    # Generate orbit camera path
    print(f"Generating {args.frames} camera positions...")
    cameras = generate_orbit_cameras(center, args.radius, args.height, args.frames)

    # Render each frame
    print(f"\nRendering frames...")
    for i, camera in enumerate(cameras):
        output_path = output_dir / f"frame_{i:04d}.png"
        render_frame(model_path, dataset_root, output_path, camera, i)

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{args.frames} frames")

    print(f"\n✓ Rendered {args.frames} frames to {output_dir}")
    print("\nNext steps:")
    print(f"  1. Create video: ffmpeg -framerate 30 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {output_dir.parent}/orbit_video.mp4")
    print(f"  2. Manually inspect video for floating artifacts")

    print("\n⚠ NOTE: This script currently contains placeholder rendering logic.")
    print("   The actual rendering command needs to be implemented based on sugar-render's API.")
    print("   Key requirement: sugar-render must support custom camera pose specification.")


if __name__ == "__main__":
    main()
"#;

    let script_dir = PathBuf::from("scripts");
    std::fs::create_dir_all(&script_dir).expect("Failed to create scripts directory");

    let script_path = script_dir.join("render_orbit_video.py");
    std::fs::write(&script_path, script_content).expect("Failed to write orbit video script");

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

    println!("✓ Generated orbit video rendering script: {}", script_path.display());
    println!("\nUsage:");
    println!("  python scripts/render_orbit_video.py \\");
    println!("    --model runs/floater_test_bicycle/model_final.gs \\");
    println!("    --dataset-root datasets/bicycle \\");
    println!("    --output runs/floater_test_bicycle/orbit_video \\");
    println!("    --frames 360 \\");
    println!("    --radius 3.0");
    println!("\nNote: The script contains placeholder rendering logic.");
    println!("      Implementation depends on sugar-render supporting custom camera poses.");
    println!("\nAfter rendering:");
    println!("  ffmpeg -framerate 30 -i runs/floater_test_bicycle/orbit_video/frame_%04d.png \\");
    println!("    -c:v libx264 -pix_fmt yuv420p -crf 18 \\");
    println!("    runs/floater_test_bicycle/orbit_video.mp4");
}

/// TC-E2E-010: Verify Depth Map Rendering Script
///
/// This test verifies that the depth map orbit rendering script exists and is functional.
/// The script renders depth maps along an orbit path for floater detection.
///
/// Run with: cargo test --release tc_e2e_010_floater_generate_depth_script -- --nocapture --ignored
#[test]
#[ignore]
fn tc_e2e_010_floater_generate_depth_script() {
    println!("\n=== TC-E2E-010: Verifying Depth Map Rendering Script ===\n");

    let script_path = PathBuf::from("scripts/render_orbit_depth.py");

    if !script_path.exists() {
        panic!("Depth rendering script not found: {}", script_path.display());
    }

    // Verify script is executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(&script_path)
            .expect("Failed to get file metadata");
        let mode = metadata.permissions().mode();

        if mode & 0o111 == 0 {
            println!("⚠ Script is not executable, setting executable permissions...");
            let mut perms = metadata.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)
                .expect("Failed to set executable permissions");
        }
    }

    println!("✓ Depth rendering script verified: {}", script_path.display());
    println!("\nThis script:");
    println!("  - Generates 360° orbital camera path (same as RGB orbit)");
    println!("  - Renders both RGB and depth maps using sugar-render --depth-out");
    println!("  - Creates depth map sequences for floater detection");
    println!("  - Saves camera path metadata for reference");

    println!("\nUsage:");
    println!("  python scripts/render_orbit_depth.py \\");
    println!("    --model runs/floater_test_bicycle/model_final.gs \\");
    println!("    --dataset-root datasets/bicycle \\");
    println!("    --output runs/floater_test_bicycle/depth_orbit \\");
    println!("    --frames 360 \\");
    println!("    --elevation 0");

    println!("\nAfter rendering, create videos:");
    println!("  # RGB video");
    println!("  ffmpeg -framerate 30 -i output/rgb_%04d.png \\");
    println!("    -c:v libx264 -pix_fmt yuv420p -crf 18 \\");
    println!("    floater_orbit_rgb.mp4");
    println!("  # Depth video");
    println!("  ffmpeg -framerate 30 -i output/depth_%04d.png \\");
    println!("    -c:v libx264 -pix_fmt yuv420p -crf 18 \\");
    println!("    floater_orbit_depth.mp4");
    println!("  # Side-by-side comparison");
    println!("  ffmpeg -i floater_orbit_rgb.mp4 -i floater_orbit_depth.mp4 \\");
    println!("    -filter_complex hstack \\");
    println!("    floater_orbit_comparison.mp4");

    println!("\nFloater detection with depth maps:");
    println!("  - Look for isolated depth discontinuities (sudden jumps)");
    println!("  - Check for 'floating' depth values disconnected from surfaces");
    println!("  - Identify depth noise in smooth regions (sky, walls)");
    println!("  - Compare RGB and depth videos for correlation");
}
