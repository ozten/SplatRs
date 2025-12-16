//! M1 Test: Load COLMAP scene and export to PLY
//!
//! This test verifies that we can:
//! 1. Parse COLMAP binary files (cameras.bin, images.bin, points3D.bin)
//! 2. Export 3D points to PLY format for visualization

use std::path::PathBuf;
use sugar_rs::io::{load_colmap_scene, save_colmap_points_ply};

#[test]
#[ignore] // E2E test - requires external dataset (use `cargo test -- --ignored`)
fn test_load_calipers_colmap() {
    // Path to the COLMAP sparse reconstruction
    let colmap_path = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );

    // Skip test if path doesn't exist (for CI/other machines)
    if !colmap_path.exists() {
        println!("Skipping test - COLMAP data not found at {:?}", colmap_path);
        return;
    }

    // Load COLMAP scene
    let scene = load_colmap_scene(&colmap_path).expect("Failed to load COLMAP scene");

    // Print statistics
    println!("Loaded COLMAP scene:");
    println!("  Cameras: {}", scene.cameras.len());
    println!("  Images: {}", scene.images.len());
    println!("  3D Points: {}", scene.points.len());

    // Verify we got some data
    assert!(scene.cameras.len() > 0, "No cameras loaded");
    assert!(scene.images.len() > 0, "No images loaded");
    assert!(scene.points.len() > 0, "No 3D points loaded");

    // Print first camera details
    if let Some(cam) = scene.cameras.values().next() {
        println!("\nFirst camera:");
        println!("  Resolution: {}x{}", cam.width, cam.height);
        println!("  Focal length: fx={:.2}, fy={:.2}", cam.fx, cam.fy);
        println!("  Principal point: cx={:.2}, cy={:.2}", cam.cx, cam.cy);
    }

    // Print some point statistics
    if !scene.points.is_empty() {
        let avg_x =
            scene.points.iter().map(|p| p.position.x).sum::<f32>() / scene.points.len() as f32;
        let avg_y =
            scene.points.iter().map(|p| p.position.y).sum::<f32>() / scene.points.len() as f32;
        let avg_z =
            scene.points.iter().map(|p| p.position.z).sum::<f32>() / scene.points.len() as f32;
        println!(
            "\nPoint cloud centroid: ({:.3}, {:.3}, {:.3})",
            avg_x, avg_y, avg_z
        );
    }

    // Export to PLY
    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let ply_path = output_dir.join("calipers_points.ply");
    save_colmap_points_ply(&scene.points, &ply_path).expect("Failed to save PLY file");

    println!("\n✅ PLY exported to: {:?}", ply_path);
    println!("   Open this file in MeshLab or Blender to visualize!");

    // Verify the file exists and has content
    assert!(ply_path.exists(), "PLY file was not created");
    let metadata = std::fs::metadata(&ply_path).expect("Failed to read PLY file metadata");
    assert!(metadata.len() > 0, "PLY file is empty");

    println!("   File size: {} bytes", metadata.len());
}

#[test]
#[ignore] // E2E test - requires external dataset (use `cargo test -- --ignored`)
fn test_colmap_camera_details() {
    let colmap_path = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );

    if !colmap_path.exists() {
        println!("Skipping test - COLMAP data not found");
        return;
    }

    let scene = load_colmap_scene(&colmap_path).expect("Failed to load scene");

    // Print detailed camera information
    for (camera_id, camera) in scene.cameras.iter() {
        println!("Camera ID {}:", camera_id);
        println!("  Resolution: {}x{}", camera.width, camera.height);
        println!(
            "  Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
            camera.fx, camera.fy, camera.cx, camera.cy
        );
    }

    // Print first few image poses
    println!("\nFirst 3 image poses:");
    for image in scene.images.iter().take(3) {
        println!("  {}: camera_id={}", image.name, image.camera_id);
        println!("    Rotation: {:?}", image.rotation);
        println!("    Translation: {:?}", image.translation);
    }
}
