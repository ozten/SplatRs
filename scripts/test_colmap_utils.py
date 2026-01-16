#!/usr/bin/env python3
"""
Test script for colmap_utils.py

Verifies that COLMAP binary file parsing works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from colmap_utils import (
    read_colmap_cameras,
    read_colmap_images,
    get_image_to_camera_mapping,
    read_colmap_points3d,
    load_colmap_reconstruction
)


def test_bicycle_dataset():
    """Test with bicycle dataset if available."""
    dataset_root = Path("datasets/bicycle")
    sparse_dir = dataset_root / "sparse" / "0"

    if not sparse_dir.exists():
        print("⚠ Bicycle dataset not available, skipping test")
        return True

    print("Testing with bicycle dataset...")

    # Test reading cameras
    cameras_path = sparse_dir / "cameras.bin"
    if cameras_path.exists():
        cameras = read_colmap_cameras(cameras_path)
        assert len(cameras) > 0, "Should have at least one camera"
        print(f"✓ Read {len(cameras)} cameras")

        # Verify camera structure
        camera_id = list(cameras.keys())[0]
        cam = cameras[camera_id]
        assert 'width' in cam
        assert 'height' in cam
        assert 'fx' in cam
        assert 'fy' in cam
        assert 'cx' in cam
        assert 'cy' in cam
        print(f"✓ Camera structure valid: {cam['width']}x{cam['height']}")

    # Test reading images
    images_path = sparse_dir / "images.bin"
    if images_path.exists():
        images = read_colmap_images(images_path)
        assert len(images) > 0, "Should have at least one image"
        print(f"✓ Read {len(images)} images")

        # Verify image structure
        image_name = list(images.keys())[0]
        img = images[image_name]
        assert 'image_id' in img
        assert 'camera_id' in img
        assert 'name' in img
        assert 'qvec' in img
        assert 'tvec' in img
        assert len(img['qvec']) == 4, "Quaternion should have 4 elements"
        assert len(img['tvec']) == 3, "Translation should have 3 elements"
        print(f"✓ Image structure valid: {img['name']} -> camera {img['camera_id']}")

    # Test image-to-camera mapping
    if images_path.exists():
        mapping = get_image_to_camera_mapping(images_path)
        assert len(mapping) > 0, "Should have at least one mapping"
        print(f"✓ Image-to-camera mapping: {len(mapping)} entries")

        # Verify mapping format
        image_name = list(mapping.keys())[0]
        camera_id = mapping[image_name]
        assert isinstance(image_name, str), "Image name should be string"
        assert isinstance(camera_id, int), "Camera ID should be int"
        print(f"✓ Mapping format valid: '{image_name}' -> {camera_id}")

    # Test reading points3d
    points3d_path = sparse_dir / "points3D.bin"
    if points3d_path.exists():
        points = read_colmap_points3d(points3d_path)
        assert len(points) > 0, "Should have at least one point"
        assert points.shape[1] == 3, "Points should be 3D"
        print(f"✓ Read {len(points)} 3D points")

    # Test loading complete reconstruction
    cameras, images, points3d = load_colmap_reconstruction(sparse_dir)
    assert len(cameras) > 0, "Should have cameras"
    assert len(images) > 0, "Should have images"
    if points3d is not None:
        assert len(points3d) > 0, "Should have points"
    print(f"✓ Complete reconstruction loaded: {len(cameras)} cameras, {len(images)} images")

    return True


def test_image_camera_mapping_consistency():
    """Test that image-to-camera mapping is consistent."""
    dataset_root = Path("datasets/bicycle")
    sparse_dir = dataset_root / "sparse" / "0"

    if not sparse_dir.exists():
        print("⚠ Bicycle dataset not available, skipping consistency test")
        return True

    print("\nTesting mapping consistency...")

    images_path = sparse_dir / "images.bin"
    cameras_path = sparse_dir / "cameras.bin"

    if not (images_path.exists() and cameras_path.exists()):
        print("⚠ Required files not available")
        return True

    # Get full images
    images = read_colmap_images(images_path)
    cameras = read_colmap_cameras(cameras_path)

    # Get mapping
    mapping = get_image_to_camera_mapping(images_path)

    # Verify consistency
    for image_name, camera_id in mapping.items():
        # Check that camera_id matches full image info
        assert images[image_name]['camera_id'] == camera_id, \
            f"Mapping inconsistent for {image_name}"

        # Check that camera_id exists in cameras
        assert camera_id in cameras, \
            f"Camera {camera_id} referenced by {image_name} does not exist"

    print(f"✓ All {len(mapping)} mappings are consistent")
    print(f"✓ All camera IDs reference valid cameras")

    return True


if __name__ == "__main__":
    print("=== Testing colmap_utils.py ===\n")

    try:
        success = True
        success = test_bicycle_dataset() and success
        success = test_image_camera_mapping_consistency() and success

        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
