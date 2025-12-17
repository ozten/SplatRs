//! PLY I/O tests
//!
//! Tests for PLY file format export and import.
//! - Tests for implemented functions (save_colmap_points_ply)
//! - Tests that unimplemented functions panic as expected

use sugar_rs::GaussianCloud;
use sugar_rs::io::{save_ply, load_ply, save_colmap_points_ply, Point3D};
use nalgebra::Vector3;
use std::path::PathBuf;
use std::fs;

#[test]
fn test_save_colmap_points_ply_creates_valid_file() {
    // Create temp directory for test output
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("test_colmap_points.ply");

    // Create test point cloud data
    let points = vec![
        Point3D {
            id: 1,
            position: Vector3::new(1.0, 2.0, 3.0),
            color: [255, 128, 64],
            error: 0.5,
        },
        Point3D {
            id: 2,
            position: Vector3::new(-1.0, -2.0, -3.0),
            color: [0, 255, 0],
            error: 0.3,
        },
    ];

    // Save to PLY
    save_colmap_points_ply(&points, &test_file).expect("Failed to save PLY");

    // Verify file exists
    assert!(test_file.exists(), "PLY file should be created");

    // Read and verify contents
    let contents = fs::read_to_string(&test_file).expect("Failed to read PLY file");

    // Verify header
    assert!(contents.contains("ply"), "Should contain PLY magic");
    assert!(contents.contains("format ascii 1.0"), "Should specify ASCII format");
    assert!(contents.contains("element vertex 2"), "Should specify 2 vertices");
    assert!(contents.contains("property float x"), "Should have x property");
    assert!(contents.contains("property float y"), "Should have y property");
    assert!(contents.contains("property float z"), "Should have z property");
    assert!(contents.contains("property uchar red"), "Should have red property");
    assert!(contents.contains("property uchar green"), "Should have green property");
    assert!(contents.contains("property uchar blue"), "Should have blue property");
    assert!(contents.contains("end_header"), "Should have end_header marker");

    // Verify vertex data (point 1)
    assert!(contents.contains("1 2 3 255 128 64"), "Should contain first point data");

    // Verify vertex data (point 2)
    assert!(contents.contains("-1 -2 -3 0 255 0"), "Should contain second point data");

    // Clean up
    fs::remove_file(&test_file).ok();
}

#[test]
fn test_save_colmap_points_ply_empty_cloud() {
    // Test with zero points
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("test_colmap_empty.ply");

    let points: Vec<Point3D> = vec![];

    // Save to PLY
    save_colmap_points_ply(&points, &test_file).expect("Failed to save empty PLY");

    // Verify file exists
    assert!(test_file.exists(), "PLY file should be created even for empty cloud");

    // Read and verify contents
    let contents = fs::read_to_string(&test_file).expect("Failed to read PLY file");

    // Should have valid header with 0 vertices
    assert!(contents.contains("element vertex 0"), "Should specify 0 vertices");
    assert!(contents.contains("end_header"), "Should have end_header marker");

    // Clean up
    fs::remove_file(&test_file).ok();
}

#[test]
fn test_save_colmap_points_ply_single_point() {
    // Test with single point
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("test_colmap_single.ply");

    let points = vec![Point3D {
        id: 42,
        position: Vector3::new(10.5, -20.3, 30.7),
        color: [100, 200, 50],
        error: 1.0,
    }];

    // Save to PLY
    save_colmap_points_ply(&points, &test_file).expect("Failed to save single point PLY");

    // Read and verify contents
    let contents = fs::read_to_string(&test_file).expect("Failed to read PLY file");

    assert!(contents.contains("element vertex 1"), "Should specify 1 vertex");
    assert!(
        contents.contains("10.5 -20.3 30.7 100 200 50"),
        "Should contain point data with correct formatting"
    );

    // Clean up
    fs::remove_file(&test_file).ok();
}

#[test]
#[should_panic(expected = "not implemented")]
fn test_save_ply_unimplemented() {
    // save_ply should panic with unimplemented! until M10
    let cloud = GaussianCloud::new();
    let path = PathBuf::from("/tmp/test.ply");

    // This should panic
    let _ = save_ply(&cloud, &path);
}

#[test]
#[should_panic(expected = "not implemented")]
fn test_load_ply_unimplemented() {
    // load_ply should panic with unimplemented! until M10
    let path = PathBuf::from("/tmp/test.ply");

    // This should panic
    let _ = load_ply(&path);
}
