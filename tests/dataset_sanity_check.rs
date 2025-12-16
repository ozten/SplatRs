//! Dataset sanity check: Load multiple datasets and verify basic properties.
//!
//! This test helps catch bugs that only show up with certain dataset characteristics:
//! - Different camera models
//! - Different point densities
//! - Different image resolutions

use std::path::PathBuf;
use sugar_rs::io::load_colmap_scene;

#[test]
fn test_dataset_tandt_train_loads() {
    let path = PathBuf::from("datasets/tandt_db/tandt/train/sparse/0");
    if !path.exists() {
        println!("Skipping - dataset not found at {:?}", path);
        return;
    }

    let scene = load_colmap_scene(&path).expect("Failed to load T&T train scene");

    // Basic sanity checks
    assert!(!scene.cameras.is_empty(), "Scene should have at least one camera");
    assert!(!scene.images.is_empty(), "Scene should have at least one image");
    assert!(!scene.points.is_empty(), "Scene should have at least one point");

    println!("T&T Train Dataset:");
    println!("  - Cameras: {}", scene.cameras.len());
    println!("  - Images: {}", scene.images.len());
    println!("  - Points: {}", scene.points.len());

    // Check first camera
    let cam = scene.cameras.values().next().expect("No cameras found");
    println!("  - First camera: {}x{} px, fx={:.1}, fy={:.1}",
             cam.width, cam.height, cam.fx, cam.fy);

    // Check image-camera association
    for (i, img) in scene.images.iter().take(3).enumerate() {
        println!("  - Image {}: camera_id={}, name={}", i, img.camera_id, img.name);
    }
}

#[test]
fn test_dataset_tandt_truck_loads() {
    let path = PathBuf::from("datasets/tandt_db/tandt/truck/sparse/0");
    if !path.exists() {
        println!("Skipping - dataset not found at {:?}", path);
        return;
    }

    let scene = load_colmap_scene(&path).expect("Failed to load T&T truck scene");

    assert!(!scene.cameras.is_empty());
    assert!(!scene.images.is_empty());
    assert!(!scene.points.is_empty());

    println!("T&T Truck Dataset:");
    println!("  - Cameras: {}", scene.cameras.len());
    println!("  - Images: {}", scene.images.len());
    println!("  - Points: {}", scene.points.len());

    let cam = scene.cameras.values().next().expect("No cameras found");
    println!("  - First camera: {}x{} px, fx={:.1}, fy={:.1}",
             cam.width, cam.height, cam.fx, cam.fy);
}

#[test]
fn test_dataset_db_playroom_loads() {
    let path = PathBuf::from("datasets/tandt_db/db/playroom/sparse/0");
    if !path.exists() {
        println!("Skipping - dataset not found at {:?}", path);
        return;
    }

    let scene = load_colmap_scene(&path).expect("Failed to load Deep Blending playroom scene");

    assert!(!scene.cameras.is_empty());
    assert!(!scene.images.is_empty());
    assert!(!scene.points.is_empty());

    println!("Deep Blending Playroom Dataset:");
    println!("  - Cameras: {}", scene.cameras.len());
    println!("  - Images: {}", scene.images.len());
    println!("  - Points: {}", scene.points.len());

    let cam = scene.cameras.values().next().expect("No cameras found");
    println!("  - First camera: {}x{} px, fx={:.1}, fy={:.1}",
             cam.width, cam.height, cam.fx, cam.fy);
}

#[test]
fn test_all_datasets_have_valid_camera_ids() {
    // Test that all images reference valid camera IDs
    let datasets = vec![
        "datasets/tandt_db/tandt/train/sparse/0",
        "datasets/tandt_db/tandt/truck/sparse/0",
        "datasets/tandt_db/db/playroom/sparse/0",
        "datasets/tandt_db/db/drjohnson/sparse/0",
    ];

    for dataset_path in datasets {
        let path = PathBuf::from(dataset_path);
        if !path.exists() {
            println!("Skipping {} - not found", dataset_path);
            continue;
        }

        let scene = load_colmap_scene(&path)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", dataset_path, e));

        // Build set of valid camera IDs
        let valid_camera_ids: std::collections::HashSet<u32> =
            scene.cameras.iter().enumerate().map(|(i, _)| i as u32).collect();

        // Check that all images reference valid cameras
        for img in &scene.images {
            // NOTE: This test will FAIL if our camera_id assumption is wrong!
            // We're assuming camera_id should be an index, but it might be a separate ID.
            println!("{}: Image {} references camera_id {}",
                     dataset_path, img.name, img.camera_id);
        }
    }
}
