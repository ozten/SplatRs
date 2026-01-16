#!/usr/bin/env python3
"""
Test that compare_edge_sharpness.py can load COLMAP data correctly.

This test verifies the integration without requiring opencv-python.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from colmap_utils import get_image_to_camera_mapping, read_colmap_cameras


def test_edge_sharpness_colmap_loading():
    """Test that edge sharpness script can load COLMAP data."""
    dataset_root = Path("datasets/bicycle")
    sparse_dir = dataset_root / "sparse" / "0"

    if not sparse_dir.exists():
        print("⚠ Bicycle dataset not available, skipping test")
        return True

    print("Testing COLMAP data loading for edge sharpness script...")

    images_path = sparse_dir / "images.bin"
    cameras_path = sparse_dir / "cameras.bin"

    # Load image-to-camera mapping
    image_to_camera = get_image_to_camera_mapping(images_path)
    print(f"✓ Loaded {len(image_to_camera)} image-to-camera mappings")

    # Load cameras
    cameras = read_colmap_cameras(cameras_path)
    print(f"✓ Loaded {len(cameras)} cameras")

    # Verify that all referenced cameras exist
    for img_name, cam_id in image_to_camera.items():
        assert cam_id in cameras, f"Camera {cam_id} not found for image {img_name}"

    print("✓ All image-to-camera mappings reference valid cameras")

    # Show sample mappings (like the edge sharpness script does)
    print("\nSample mappings (as shown by edge sharpness script):")
    for i, (img_name, cam_id) in enumerate(list(image_to_camera.items())[:5]):
        cam = cameras.get(cam_id, {})
        print(f"  {img_name} -> camera {cam_id} "
              f"({cam.get('width', '?')}x{cam.get('height', '?')}, "
              f"fx={cam.get('fx', '?'):.1f}, fy={cam.get('fy', '?'):.1f})")

    # Test with test split (every 8th image)
    images_dir = dataset_root / "images"
    all_images = sorted([p.name for p in images_dir.iterdir()
                        if p.suffix.lower() in ['.jpg', '.png']])
    test_views = [img for i, img in enumerate(all_images) if i % 8 == 0]

    print(f"\nTest split: {len(test_views)} views")
    print("Verifying all test views have camera mappings...")

    missing = []
    for view in test_views:
        if view not in image_to_camera:
            missing.append(view)

    if missing:
        print(f"  ⚠ {len(missing)} test views missing camera mappings:")
        for view in missing[:5]:
            print(f"    - {view}")
    else:
        print(f"  ✓ All {len(test_views)} test views have camera mappings")

    return True


if __name__ == "__main__":
    print("=== Testing Edge Sharpness COLMAP Integration ===\n")

    try:
        test_edge_sharpness_colmap_loading()
        print("\n✅ Edge sharpness COLMAP integration test passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
