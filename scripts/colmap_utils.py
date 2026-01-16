#!/usr/bin/env python3
"""
Utility functions for reading COLMAP binary files.

This module provides functions to read:
- cameras.bin: Camera intrinsic parameters
- images.bin: Camera poses (extrinsics) and image-to-camera-ID mapping
- points3D.bin: 3D point cloud

Based on the COLMAP binary format specification and SplatRs implementation.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def read_colmap_cameras(bin_path: Path) -> Dict[int, Dict]:
    """
    Read COLMAP cameras.bin file.

    Binary format (little-endian):
    - num_cameras: u64
    - For each camera:
      - camera_id: u32
      - model_id: u32
      - width: u64
      - height: u64
      - params: array of f64 (length depends on model_id)

    Camera models:
    - Model 1 = SIMPLE_PINHOLE (fx, cx, cy)
    - Model 2 = PINHOLE (fx, fy, cx, cy)
    - Model 3 = SIMPLE_RADIAL (fx, cx, cy, k)
    - Model 4 = RADIAL (fx, cx, cy, k1, k2)

    Returns:
        Dict mapping camera_id to camera parameters dict with keys:
        - camera_id: int
        - model_id: int
        - width: int
        - height: int
        - fx, fy, cx, cy: float (intrinsics)
    """
    cameras = {}

    with open(bin_path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]

        for _ in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]

            # Read parameters based on model
            if model_id == 1:  # SIMPLE_PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                cameras[camera_id] = {
                    'camera_id': camera_id,
                    'model_id': model_id,
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fx,  # Same as fx for simple pinhole
                    'cx': cx,
                    'cy': cy,
                }
            elif model_id == 2:  # PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                fy = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                cameras[camera_id] = {
                    'camera_id': camera_id,
                    'model_id': model_id,
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                }
            elif model_id == 3:  # SIMPLE_RADIAL
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                k = struct.unpack('d', f.read(8))[0]
                cameras[camera_id] = {
                    'camera_id': camera_id,
                    'model_id': model_id,
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fx,
                    'cx': cx,
                    'cy': cy,
                    'k1': k,
                }
            elif model_id == 4:  # RADIAL
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                k1 = struct.unpack('d', f.read(8))[0]
                k2 = struct.unpack('d', f.read(8))[0]
                cameras[camera_id] = {
                    'camera_id': camera_id,
                    'model_id': model_id,
                    'width': int(width),
                    'height': int(height),
                    'fx': fx,
                    'fy': fx,
                    'cx': cx,
                    'cy': cy,
                    'k1': k1,
                    'k2': k2,
                }
            else:
                # Unsupported model - read remaining params as array
                # Number of params varies by model
                print(f"Warning: Unsupported camera model {model_id}")
                cameras[camera_id] = {
                    'camera_id': camera_id,
                    'model_id': model_id,
                    'width': int(width),
                    'height': int(height),
                    'fx': 525.0,  # Default
                    'fy': 525.0,
                    'cx': float(width) / 2.0,
                    'cy': float(height) / 2.0,
                }

    return cameras


def read_colmap_images(bin_path: Path) -> Dict[str, Dict]:
    """
    Read COLMAP images.bin file.

    Binary format (little-endian):
    - num_images: u64
    - For each image:
      - image_id: u32
      - qw, qx, qy, qz: f64 (rotation quaternion)
      - tx, ty, tz: f64 (translation)
      - camera_id: u32
      - name: null-terminated string
      - num_points2d: u64
      - points2d: array of (x: f64, y: f64, point3d_id: u64)

    Returns:
        Dict mapping image_name to image info dict with keys:
        - image_id: int
        - camera_id: int
        - name: str
        - qvec: np.ndarray (4,) - quaternion (qw, qx, qy, qz)
        - tvec: np.ndarray (3,) - translation (tx, ty, tz)
    """
    images = {}

    with open(bin_path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack('I', f.read(4))[0]

            # Read quaternion (qw, qx, qy, qz)
            qw = struct.unpack('d', f.read(8))[0]
            qx = struct.unpack('d', f.read(8))[0]
            qy = struct.unpack('d', f.read(8))[0]
            qz = struct.unpack('d', f.read(8))[0]

            # Read translation (tx, ty, tz)
            tx = struct.unpack('d', f.read(8))[0]
            ty = struct.unpack('d', f.read(8))[0]
            tz = struct.unpack('d', f.read(8))[0]

            camera_id = struct.unpack('I', f.read(4))[0]

            # Read null-terminated image name
            name_bytes = []
            while True:
                byte = f.read(1)[0]
                if byte == 0:
                    break
                name_bytes.append(byte)
            name = bytes(name_bytes).decode('utf-8')

            # Read 2D points (we skip these but need to read to advance file pointer)
            num_points2d = struct.unpack('Q', f.read(8))[0]
            # Each 2D point is: x (f64), y (f64), point3d_id (u64) = 24 bytes
            f.read(num_points2d * 24)

            images[name] = {
                'image_id': image_id,
                'camera_id': camera_id,
                'name': name,
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
            }

    return images


def get_image_to_camera_mapping(images_bin_path: Path) -> Dict[str, int]:
    """
    Get mapping from image name to camera ID.

    This is a convenience function that extracts just the image-to-camera-ID
    mapping without loading all the pose information.

    Args:
        images_bin_path: Path to COLMAP images.bin file

    Returns:
        Dict mapping image_name (str) to camera_id (int)
    """
    images = read_colmap_images(images_bin_path)
    return {name: info['camera_id'] for name, info in images.items()}


def read_colmap_points3d(bin_path: Path) -> np.ndarray:
    """
    Read COLMAP points3D.bin file.

    Binary format (little-endian):
    - num_points: u64
    - For each point:
      - point_id: u64
      - x, y, z: f64 (position)
      - r, g, b: u8 (color)
      - error: f64 (reprojection error)
      - track_len: u64
      - track: array of (image_id: u32, point2d_idx: u32)

    Returns:
        np.ndarray of shape (N, 3) containing 3D point positions
    """
    points = []

    with open(bin_path, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]

        for _ in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]

            # Read position (x, y, z)
            x = struct.unpack('d', f.read(8))[0]
            y = struct.unpack('d', f.read(8))[0]
            z = struct.unpack('d', f.read(8))[0]

            # Read color (r, g, b) - we don't need this but must read it
            r = struct.unpack('B', f.read(1))[0]
            g = struct.unpack('B', f.read(1))[0]
            b = struct.unpack('B', f.read(1))[0]

            # Read error
            error = struct.unpack('d', f.read(8))[0]

            # Read track data (we skip this but need to read to advance file pointer)
            track_len = struct.unpack('Q', f.read(8))[0]
            # Each track element is: image_id (u32), point2d_idx (u32) = 8 bytes
            f.read(track_len * 8)

            points.append([x, y, z])

    return np.array(points)


def load_colmap_reconstruction(sparse_dir: Path) -> Tuple[Dict[int, Dict], Dict[str, Dict], Optional[np.ndarray]]:
    """
    Load complete COLMAP reconstruction.

    Args:
        sparse_dir: Path to COLMAP sparse reconstruction directory (e.g., datasets/bicycle/sparse/0)

    Returns:
        Tuple of (cameras, images, points3d):
        - cameras: Dict mapping camera_id to camera parameters
        - images: Dict mapping image_name to image info (with pose and camera_id)
        - points3d: np.ndarray of 3D points (N, 3), or None if not available
    """
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    points3d_path = sparse_dir / "points3D.bin"

    cameras = {}
    images = {}
    points3d = None

    if cameras_path.exists():
        cameras = read_colmap_cameras(cameras_path)
    else:
        print(f"Warning: cameras.bin not found at {cameras_path}")

    if images_path.exists():
        images = read_colmap_images(images_path)
    else:
        print(f"Warning: images.bin not found at {images_path}")

    if points3d_path.exists():
        points3d = read_colmap_points3d(points3d_path)
    else:
        print(f"Warning: points3D.bin not found at {points3d_path}")

    return cameras, images, points3d


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python colmap_utils.py <dataset_root>")
        print("Example: python colmap_utils.py datasets/bicycle")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    sparse_dir = dataset_root / "sparse" / "0"

    if not sparse_dir.exists():
        print(f"Error: Sparse reconstruction not found at {sparse_dir}")
        sys.exit(1)

    print(f"Loading COLMAP reconstruction from {sparse_dir}")
    cameras, images, points3d = load_colmap_reconstruction(sparse_dir)

    print(f"\nCameras: {len(cameras)}")
    for camera_id, cam in cameras.items():
        print(f"  Camera {camera_id}: {cam['width']}x{cam['height']}, "
              f"fx={cam['fx']:.2f}, fy={cam['fy']:.2f}")

    print(f"\nImages: {len(images)}")
    print("  Sample images:")
    for i, (name, img) in enumerate(list(images.items())[:5]):
        print(f"    {name}: camera_id={img['camera_id']}, "
              f"pos=({img['tvec'][0]:.2f}, {img['tvec'][1]:.2f}, {img['tvec'][2]:.2f})")

    if points3d is not None:
        print(f"\n3D Points: {len(points3d)}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{points3d[:, 0].min():.2f}, {points3d[:, 0].max():.2f}]")
        print(f"    Y: [{points3d[:, 1].min():.2f}, {points3d[:, 1].max():.2f}]")
        print(f"    Z: [{points3d[:, 2].min():.2f}, {points3d[:, 2].max():.2f}]")

    print("\nImage-to-Camera mapping:")
    mapping = get_image_to_camera_mapping(sparse_dir / "images.bin")
    for name, camera_id in list(mapping.items())[:10]:
        print(f"  {name} -> camera {camera_id}")
