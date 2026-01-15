#!/usr/bin/env python3
"""
Render 360° orbit depth maps for floater detection.

Usage:
    python render_orbit_depth.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --frames 360 \
        --elevation 0

Example:
    python render_orbit_depth.py \
        --model runs/floater_test_bicycle/model_final.gs \
        --dataset-root datasets/bicycle \
        --output runs/floater_test_bicycle/depth_orbit \
        --frames 360 \
        --elevation 0

This script:
1. Generates 360° orbital camera path around scene center (same as RGB orbit)
2. Renders depth maps for each camera position using sugar-render --depth-out
3. Creates depth map sequence for floater detection via discontinuities
4. Saves camera path metadata for reference

Floater Detection with Depth Maps:
- Look for isolated depth discontinuities (sudden depth jumps)
- Check for "floating" depth values disconnected from main surfaces
- Identify depth noise in regions that should be smooth (sky, walls)
- Compare RGB and depth videos side-by-side for correlation

Pass Criteria:
- Depth maps show continuous surfaces without isolated spikes
- No significant depth discontinuities in empty space
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
    print("Install with: pip install pillow numpy", file=sys.stderr)
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


def read_colmap_camera(dataset_root: Path) -> Dict:
    """Read camera intrinsics from COLMAP cameras.bin."""
    cameras_path = dataset_root / "sparse" / "0" / "cameras.bin"

    if not cameras_path.exists():
        print(f"Warning: cameras.bin not found: {cameras_path}", file=sys.stderr)
        print("Using default camera parameters (640x480, fx=fy=525)", file=sys.stderr)
        return {
            'width': 640,
            'height': 480,
            'fx': 525.0,
            'fy': 525.0,
            'cx': 320.0,
            'cy': 240.0,
        }

    try:
        with open(cameras_path, 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]

            if num_cameras == 0:
                print("Warning: No cameras found in COLMAP reconstruction", file=sys.stderr)
                return {
                    'width': 640,
                    'height': 480,
                    'fx': 525.0,
                    'fy': 525.0,
                    'cx': 320.0,
                    'cy': 240.0,
                }

            # Read first camera
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]

            # Model 1 = SIMPLE_PINHOLE (fx, cx, cy)
            # Model 2 = PINHOLE (fx, fy, cx, cy)
            # Model 3 = SIMPLE_RADIAL (fx, cx, cy, k)
            # Model 4 = RADIAL (fx, cx, cy, k1, k2)

            if model_id == 1:  # SIMPLE_PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                return {
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fx,
                    'cx': cx,
                    'cy': cy,
                }
            elif model_id == 2:  # PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                fy = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                return {
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                }
            else:
                print(f"Warning: Unsupported camera model ID {model_id}", file=sys.stderr)
                return {
                    'width': int(width),
                    'height': int(height),
                    'fx': 525.0,
                    'fy': 525.0,
                    'cx': float(width) / 2.0,
                    'cy': float(height) / 2.0,
                }

    except Exception as e:
        print(f"Error reading COLMAP cameras: {e}", file=sys.stderr)
        print("Using default camera parameters", file=sys.stderr)
        return {
            'width': 640,
            'height': 480,
            'fx': 525.0,
            'fy': 525.0,
            'cx': 320.0,
            'cy': 240.0,
        }


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


def create_camera_json(camera: Dict, camera_intrinsics: Dict, output_path: Path) -> Path:
    """Create camera JSON file for sugar-render --camera-json."""
    # Convert view_dir, up, right to rotation matrix
    # Rotation matrix columns are [right, -up, view_dir]
    # OpenCV/COLMAP convention: +X right, +Y down, +Z forward
    rotation = [
        camera['right'].tolist(),
        (-camera['up']).tolist(),  # Flip up to get down
        camera['view_dir'].tolist(),
    ]

    camera_json = {
        'width': camera_intrinsics['width'],
        'height': camera_intrinsics['height'],
        'fx': camera_intrinsics['fx'],
        'fy': camera_intrinsics['fy'],
        'cx': camera_intrinsics['cx'],
        'cy': camera_intrinsics['cy'],
        'position': camera['position'].tolist(),
        'rotation': rotation,
    }

    json_path = output_path.parent / f"camera_{camera['frame']:04d}.json"
    with open(json_path, 'w') as f:
        json.dump(camera_json, f, indent=2)

    return json_path


def render_depth_frame(model_path: Path, camera_json_path: Path,
                       rgb_output_path: Path, depth_output_path: Path) -> bool:
    """Render RGB and depth map using sugar-render."""
    cmd = [
        "./target/release/sugar-render",
        "--model", str(model_path),
        "--camera-json", str(camera_json_path),
        "--out", str(rgb_output_path),
        "--depth-out", str(depth_output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Error rendering frame: {result.stderr}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout rendering frame", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Exception rendering frame: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Render 360° orbit depth maps for floater detection")
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

    # Read camera intrinsics from COLMAP
    print("\nReading camera intrinsics from COLMAP...")
    camera_intrinsics = read_colmap_camera(dataset_root)
    print(f"Camera: {camera_intrinsics['width']}x{camera_intrinsics['height']}, "
          f"fx={camera_intrinsics['fx']:.1f}, fy={camera_intrinsics['fy']:.1f}")

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
        'camera_intrinsics': camera_intrinsics,
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
    print(f"\nRendering {len(cameras)} frames (RGB + depth)...")
    success_count = 0
    fail_count = 0

    for i, camera in enumerate(cameras):
        # Create camera JSON file
        camera_json_path = create_camera_json(camera, camera_intrinsics, output_dir / f"rgb_{i:04d}.png")

        # Render RGB and depth
        rgb_output = output_dir / f"rgb_{i:04d}.png"
        depth_output = output_dir / f"depth_{i:04d}.png"

        success = render_depth_frame(model_path, camera_json_path, rgb_output, depth_output)

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Clean up temporary camera JSON
        camera_json_path.unlink(missing_ok=True)

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{len(cameras)} frames ({success_count} success, {fail_count} failed)")

    print(f"\n✓ Rendered {success_count}/{len(cameras)} frames successfully")

    if fail_count > 0:
        print(f"⚠ {fail_count} frames failed to render")

    print("\nNext steps:")
    print("1. Create RGB video:")
    print(f"   ffmpeg -framerate 30 -i {output_dir}/rgb_%04d.png \\")
    print(f"     -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"     {output_dir.parent}/floater_orbit_rgb.mp4")
    print("\n2. Create depth video:")
    print(f"   ffmpeg -framerate 30 -i {output_dir}/depth_%04d.png \\")
    print(f"     -c:v libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"     {output_dir.parent}/floater_orbit_depth.mp4")
    print("\n3. Create side-by-side comparison:")
    print(f"   ffmpeg -i {output_dir.parent}/floater_orbit_rgb.mp4 \\")
    print(f"     -i {output_dir.parent}/floater_orbit_depth.mp4 \\")
    print(f"     -filter_complex hstack \\")
    print(f"     {output_dir.parent}/floater_orbit_comparison.mp4")
    print("\n4. Manually inspect videos:")
    print("   RGB video:")
    print("     - Watch for floating blobs or disconnected structures")
    print("     - Check for semi-transparent artifacts in empty space")
    print("   Depth video:")
    print("     - Look for isolated depth discontinuities (sudden jumps)")
    print("     - Check for 'floating' depth values disconnected from surfaces")
    print("     - Identify depth noise in smooth regions (sky, walls)")
    print("   Comparison video:")
    print("     - Correlate RGB floaters with depth discontinuities")
    print("     - Verify depth structure matches visible geometry")
    print("\n5. Severity assessment:")
    print("   - Low: Minor floaters/discontinuities, barely visible")
    print("   - Medium: Noticeable floaters, affects quality")
    print("   - High: Significant floaters, severely impacts quality")


if __name__ == "__main__":
    main()
