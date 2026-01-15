#!/usr/bin/env python3
"""
Compare view-dependent effects across multiple viewpoints.

Usage:
    python compare_view_dependent.py \
        --model <path_to_model.gs> \
        --dataset-root <path_to_dataset> \
        --output <output_directory> \
        --num-viewpoints 8

Example:
    python compare_view_dependent.py \
        --model runs/view_dep_test_counter/model_final.gs \
        --dataset-root datasets/counter \
        --output runs/view_dep_test_counter/view_dependent \
        --num-viewpoints 8

This script:
1. Selects a reference point in the scene (scene center or user-specified)
2. Generates camera positions in an arc or circle around the reference point
3. Renders views from each position
4. Creates comparison visualizations to show highlight movement
5. Generates side-by-side grid for manual inspection
"""

import argparse
import subprocess
import sys
from pathlib import Path
import struct
import numpy as np
from typing import List, Dict, Tuple

try:
    from PIL import Image
    import cv2
except ImportError:
    print("Error: Required packages not found.", file=sys.stderr)
    print("Install with: pip install pillow opencv-python", file=sys.stderr)
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


def generate_arc_cameras(center: np.ndarray, radius: float, num_viewpoints: int,
                         arc_angle: float = 180.0) -> List[Dict]:
    """
    Generate camera positions in an arc around the scene center.

    Arc sweeps horizontally around the center at constant height and distance.
    This is ideal for observing specular highlight movement.
    """
    cameras = []

    # Arc angle in radians
    arc_rad = np.deg2rad(arc_angle)
    start_angle = -arc_rad / 2.0

    for i in range(num_viewpoints):
        # Angle for this viewpoint
        t = i / (num_viewpoints - 1) if num_viewpoints > 1 else 0.5
        angle = start_angle + t * arc_rad

        # Position in arc (XZ plane, constant Y)
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
            'angle_deg': np.rad2deg(angle),
        })

    return cameras


def render_viewpoint(model_path: Path, dataset_root: Path, output_path: Path,
                     camera: Dict, viewpoint_idx: int) -> bool:
    """Render a specific viewpoint using sugar-render."""
    # Note: This assumes sugar-render supports custom camera parameters
    # Adjust based on actual sugar-render API

    print(f"  TODO: Implement rendering for viewpoint {viewpoint_idx}")
    print(f"    Position: {camera['position']}")
    print(f"    Angle: {camera['angle_deg']:.1f}°")
    print(f"    Output: {output_path}")

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


def create_comparison_grid(images: List[np.ndarray], output_path: Path,
                          angles: List[float]):
    """Create a grid visualization of all viewpoints."""
    if not images:
        return

    # Determine grid layout (prefer horizontal layout for arc comparison)
    num_images = len(images)

    # For 8 or fewer images, use 2 rows
    # For more, use more rows as needed
    if num_images <= 4:
        rows, cols = 1, num_images
    elif num_images <= 8:
        rows, cols = 2, (num_images + 1) // 2
    else:
        rows = (num_images + 3) // 4
        cols = 4

    # Get dimensions
    h, w = images[0].shape[:2]

    # Create grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Add image to grid
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

        # Add label with angle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text = f"Angle: {angles[idx]:.1f}"

        text_x = col * w + 10
        text_y = row * h + 30

        # Add background rectangle
        cv2.rectangle(grid, (text_x - 5, text_y - 25), (text_x + 180, text_y + 5), (0, 0, 0), -1)
        cv2.putText(grid, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save grid
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved comparison grid: {output_path}")


def detect_highlights(image: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """
    Detect bright highlights in image (potential specular reflections).

    Returns a binary mask where highlights are detected.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Normalize to [0, 1]
    gray_norm = gray.astype(float) / 255.0

    # Threshold for bright highlights
    highlights = (gray_norm > threshold).astype(np.uint8) * 255

    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    highlights = cv2.morphologyEx(highlights, cv2.MORPH_OPEN, kernel)
    highlights = cv2.morphologyEx(highlights, cv2.MORPH_CLOSE, kernel)

    return highlights


def create_highlight_tracking(images: List[np.ndarray], output_path: Path,
                              angles: List[float]):
    """
    Create visualization showing highlight movement across viewpoints.

    Overlays detected highlights on each image with different colors.
    """
    if not images:
        return

    # Create composite visualization
    num_images = len(images)
    h, w = images[0].shape[:2]

    # Colors for each viewpoint (use HSV color wheel)
    colors = []
    for i in range(num_images):
        hue = int(180 * i / num_images)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(map(int, color)))

    # Create visualization for each image with its highlights overlaid
    highlight_images = []

    for idx, img in enumerate(images):
        # Detect highlights
        highlights = detect_highlights(img, threshold=0.85)

        # Create overlay
        overlay = img.copy()
        mask = highlights > 0
        overlay[mask] = overlay[mask] * 0.5 + np.array(colors[idx]) * 0.5

        # Add border with color
        border_thickness = 5
        cv2.rectangle(overlay, (0, 0), (w-1, h-1), colors[idx], border_thickness)

        # Add angle label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{angles[idx]:.1f}"
        cv2.rectangle(overlay, (10, 10), (150, 45), (0, 0, 0), -1)
        cv2.putText(overlay, label, (15, 35), font, 0.7, colors[idx], 2)

        highlight_images.append(overlay)

    # Create grid
    if num_images <= 4:
        rows, cols = 1, num_images
    elif num_images <= 8:
        rows, cols = 2, (num_images + 1) // 2
    else:
        rows = (num_images + 3) // 4
        cols = 4

    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, overlay in enumerate(highlight_images):
        row = idx // cols
        col = idx % cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = overlay

    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved highlight tracking: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare view-dependent effects across viewpoints")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.gs file)")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-viewpoints", type=int, default=8, help="Number of viewpoints (default: 8)")
    parser.add_argument("--radius", type=float, default=3.0, help="Distance from center (default: 3.0)")
    parser.add_argument("--arc-angle", type=float, default=180.0, help="Arc sweep angle in degrees (default: 180)")

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
    print(f"Viewpoints: {args.num_viewpoints}")
    print(f"Radius: {args.radius}")
    print(f"Arc angle: {args.arc_angle}°\n")

    # Compute scene center
    center = compute_scene_center(dataset_root)

    # Generate viewpoint cameras
    print(f"Generating {args.num_viewpoints} camera positions in {args.arc_angle}° arc...")
    cameras = generate_arc_cameras(center, args.radius, args.num_viewpoints, args.arc_angle)

    print("\nCamera positions:")
    for i, cam in enumerate(cameras):
        print(f"  {i+1}: angle={cam['angle_deg']:6.1f}°, pos=[{cam['position'][0]:6.2f}, {cam['position'][1]:6.2f}, {cam['position'][2]:6.2f}]")

    # Render each viewpoint
    print(f"\nRendering viewpoints...")
    rendered_images = []
    angles = []

    for i, camera in enumerate(cameras):
        output_path = output_dir / f"viewpoint_{i:02d}.png"
        success = render_viewpoint(model_path, dataset_root, output_path, camera, i)

        # For actual implementation, load rendered image
        # rendered_images.append(cv2.cvtColor(cv2.imread(str(output_path)), cv2.COLOR_BGR2RGB))
        # angles.append(camera['angle_deg'])

    print("\n⚠ NOTE: This script currently contains placeholder rendering logic.")
    print("   The actual rendering and comparison needs to be implemented.")
    print("   Key requirements:")
    print("   1. sugar-render must support custom camera pose specification")
    print("   2. OpenCV and PIL must be installed: pip install opencv-python pillow")

    print("\nAfter implementation:")
    print("1. Review individual viewpoint renders: viewpoint_*.png")
    print("2. Check comparison grid: comparison_grid.png")
    print("3. Examine highlight tracking: highlight_tracking.png")
    print("4. Manually verify:")
    print("   - Specular highlights move with viewpoint changes")
    print("   - Reflections show correct content from each angle")
    print("   - No baked-in highlights that stay fixed")
    print("   - View-dependent color/intensity changes are plausible")

    # If we had rendered images, create visualizations
    # create_comparison_grid(rendered_images, output_dir / "comparison_grid.png", angles)
    # create_highlight_tracking(rendered_images, output_dir / "highlight_tracking.png", angles)


if __name__ == "__main__":
    main()
