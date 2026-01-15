#!/usr/bin/env python3
"""
Render 360° orbit path for floater detection.

Usage:
    python render_orbit_floater_detection.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --frames 360 \
        --elevation 0

Example:
    python render_orbit_floater_detection.py \
        --model runs/floater_test_bicycle/model_final.gs \
        --dataset-root datasets/bicycle \
        --output runs/floater_test_bicycle/floater_detection \
        --frames 360 \
        --elevation 0

This script:
1. Generates 360° orbital camera path around scene center
2. Supports multiple elevation angles (0°, 30°, 90°)
3. Renders dense frame sequence (360 frames = 1 frame per degree)
4. Creates orbit video for visual inspection of floaters
5. (Future) Renders depth maps for discontinuity detection

Floater Detection:
- Watch orbit video for floating blobs or disconnected structures
- Check for semi-transparent artifacts in empty space
- Look for "string-like" Gaussians connecting unrelated surfaces
- Verify background regions are clean (no fog/clouds)
- Document severity: Low (minor), Medium (noticeable), High (significant)

Pass Criteria:
- No significant floaters visible in standard views
- Floaters documented if present with severity rating
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
except ImportError:
    print("Error: Required packages not found.", file=sys.stderr)
    print("Install with: pip install pillow", file=sys.stderr)
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


def compute_scene_center_and_radius(dataset_root: Path) -> Tuple[np.ndarray, float]:
    """Compute scene center and orbit radius from COLMAP reconstruction."""
    points3d_path = dataset_root / "sparse" / "0" / "points3D.bin"

    if not points3d_path.exists():
        print(f"Warning: points3D.bin not found: {points3d_path}", file=sys.stderr)
        print("Using default center (0, 0, 0) and radius 3.0", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0]), 3.0

    try:
        points = read_colmap_points3d(points3d_path)
        if len(points) == 0:
            print("Warning: No points found in COLMAP reconstruction", file=sys.stderr)
            return np.array([0.0, 0.0, 0.0]), 3.0

        # Compute median center (more robust than mean for outliers)
        center = np.median(points, axis=0)

        # Compute orbit radius as 1.5x the 90th percentile distance from center
        # This ensures we're outside the scene but not too far
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.percentile(distances, 90) * 1.5

        print(f"Scene center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"Orbit radius: {radius:.3f}")
        return center, radius
    except Exception as e:
        print(f"Error reading COLMAP points: {e}", file=sys.stderr)
        print("Using default center (0, 0, 0) and radius 3.0", file=sys.stderr)
        return np.array([0.0, 0.0, 0.0]), 3.0


def generate_orbit_path(center: np.ndarray, radius: float, elevation_deg: float,
                       num_frames: int) -> List[Dict]:
    """
    Generate 360° orbital camera path around scene center.

    Args:
        center: Scene center point (x, y, z)
        radius: Orbit radius (distance from center)
        elevation_deg: Camera elevation angle in degrees (0 = ground level, 90 = top-down)
        num_frames: Number of frames (360 = 1 frame per degree)

    Returns:
        List of camera dictionaries with position, view_dir, up, right
    """
    cameras = []
    elevation_rad = np.deg2rad(elevation_deg)

    for i in range(num_frames):
        # Angle for this frame (full 360° orbit)
        angle = 2 * np.pi * i / num_frames

        # Compute position on orbit path
        # XZ plane rotation with elevation
        horizontal_radius = radius * np.cos(elevation_rad)
        vertical_offset = radius * np.sin(elevation_rad)

        x = center[0] + horizontal_radius * np.sin(angle)
        z = center[2] + horizontal_radius * np.cos(angle)
        y = center[1] + vertical_offset

        # Camera looks at center
        position = np.array([x, y, z])
        target = center

        # Compute view direction
        view_dir = target - position
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Up vector (world Y-axis, adjusted to be perpendicular to view)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view_dir, up)
        if np.linalg.norm(right) > 1e-6:
            right = right / np.linalg.norm(right)
            up = np.cross(right, view_dir)
        else:
            # Special case for top-down view
            right = np.array([1.0, 0.0, 0.0])
            up = np.cross(right, view_dir)
            up = up / np.linalg.norm(up)

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


def main():
    parser = argparse.ArgumentParser(description="Render 360° orbit for floater detection")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.gs file)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--frames", type=int, default=360, help="Number of frames (default: 360)")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Camera elevation angle in degrees (0=ground level, 90=top-down, default: 0)")
    parser.add_argument("--radius", type=float, default=None,
                       help="Orbit radius (default: auto-computed from scene)")

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
    print(f"Elevation: {args.elevation}°")

    # Compute scene center and radius
    center, auto_radius = compute_scene_center_and_radius(dataset_root)
    radius = args.radius if args.radius is not None else auto_radius

    # Generate orbit camera path
    print(f"\nGenerating 360° orbit path ({args.frames} frames, radius={radius:.2f})...")
    cameras = generate_orbit_path(center, radius, args.elevation, args.frames)

    # Save camera path for reference
    camera_path_data = {
        'center': center.tolist(),
        'radius': float(radius),
        'elevation_deg': float(args.elevation),
        'num_frames': len(cameras),
        'cameras': [
            {
                'frame': cam['frame'],
                'position': cam['position'].tolist(),
                'view_dir': cam['view_dir'].tolist(),
            }
            for cam in cameras
        ]
    }

    camera_path_file = output_dir / "camera_path.json"
    with open(camera_path_file, 'w') as f:
        json.dump(camera_path_data, f, indent=2)

    print(f"✓ Saved camera path: {camera_path_file}")

    # Render frames
    print(f"\nRendering {len(cameras)} frames...")
    for i, camera in enumerate(cameras):
        output_path = output_dir / f"frame_{i:04d}.png"
        success = render_frame(model_path, dataset_root, output_path, camera, i)

    print("\n⚠ NOTE: This script currently contains placeholder rendering logic.")
    print("   The actual rendering needs to be implemented.")
    print("   Key requirements:")
    print("   1. sugar-render must support custom camera pose specification")
    print("   2. Required packages: pip install pillow")

    print("\nAfter implementation:")
    print("1. Create orbit video:")
    print(f"   ffmpeg -framerate 30 -i {output_dir}/frame_%04d.png \\")
    print(f"     -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"     {output_dir.parent}/floater_orbit.mp4")
    print("\n2. Manually inspect video:")
    print("   - Watch for floating blobs or disconnected structures")
    print("   - Check for semi-transparent artifacts in empty space")
    print("   - Look for 'string-like' Gaussians connecting surfaces")
    print("   - Verify background regions are clean")
    print("\n3. Severity assessment:")
    print("   - Low: Minor floaters, barely visible")
    print("   - Medium: Noticeable floaters, affects quality")
    print("   - High: Significant floaters, severely impacts quality")
    print("\n4. Document findings:")
    print("   - Count approximate number of floaters")
    print("   - Note which views show floaters most clearly")
    print("   - Describe characteristics (size, opacity, location)")


if __name__ == "__main__":
    main()
