#!/usr/bin/env python3
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
