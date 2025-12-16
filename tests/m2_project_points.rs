//! M2 Test: Project 3D points to images and verify alignment
//!
//! This test verifies that:
//! 1. Camera intrinsics/extrinsics are parsed correctly
//! 2. 3D→2D projection math works
//! 3. Projected points align with actual scene features

use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::{Matrix3, Vector3};
use std::path::PathBuf;
use sugar_rs::core::Camera;
use sugar_rs::io::{load_colmap_scene, ImageInfo};

/// Create a Camera by combining intrinsics and extrinsics.
///
/// Camera intrinsics come from cameras.bin.
/// Camera extrinsics (pose) come from images.bin.
fn create_camera_with_pose(base_camera: &Camera, image_info: &ImageInfo) -> Camera {
    // Convert quaternion to rotation matrix
    let rotation = image_info.rotation.to_rotation_matrix().into_inner();

    Camera::new(
        base_camera.fx,
        base_camera.fy,
        base_camera.cx,
        base_camera.cy,
        base_camera.width,
        base_camera.height,
        rotation,
        image_info.translation,
    )
}

#[test]
#[ignore] // E2E test - requires external dataset and loads images (use `cargo test -- --ignored`)
fn test_project_points_to_images() {
    // Paths
    let colmap_path = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );
    let images_dir =
        PathBuf::from("/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/input");

    // Skip test if paths don't exist
    if !colmap_path.exists() || !images_dir.exists() {
        println!("Skipping test - data not found");
        return;
    }

    // Load COLMAP scene
    let scene = load_colmap_scene(&colmap_path).expect("Failed to load COLMAP scene");

    println!(
        "Loaded scene: {} cameras, {} images, {} points",
        scene.cameras.len(),
        scene.images.len(),
        scene.points.len()
    );

    // Create output directory
    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Test with first 4 images (or fewer if not available)
    let num_test_images = 4.min(scene.images.len());

    for i in 0..num_test_images {
        let image_info = &scene.images[i];

        // Use the correct camera for this image
        let base_camera = scene
            .cameras
            .get(&image_info.camera_id)
            .expect("Camera not found");

        // Create camera with pose
        let camera = create_camera_with_pose(base_camera, image_info);

        // Load the input image
        let image_path = images_dir.join(&image_info.name);
        println!(
            "\nProcessing image {}/{}: {}",
            i + 1,
            num_test_images,
            image_info.name
        );

        let mut img = if image_path.exists() {
            image::open(&image_path)
                .expect("Failed to load image")
                .to_rgb8()
        } else {
            println!("  Image file not found, creating blank canvas");
            RgbImage::new(camera.width, camera.height)
        };

        // Project all 3D points and draw them
        let mut visible_count = 0;
        let mut in_frame_count = 0;

        for point in &scene.points {
            // Project point to 2D
            if let Some(pixel) = camera.world_to_pixel(&point.position) {
                // Check if in frame
                let x = pixel.x as u32;
                let y = pixel.y as u32;

                if x < camera.width && y < camera.height {
                    in_frame_count += 1;

                    // Draw a small cross at the projected location
                    // Use the point's color
                    let color = Rgb([point.color[0], point.color[1], point.color[2]]);

                    // Draw 3x3 cross
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            let px = (x as i32 + dx).clamp(0, camera.width as i32 - 1) as u32;
                            let py = (y as i32 + dy).clamp(0, camera.height as i32 - 1) as u32;
                            img.put_pixel(px, py, color);
                        }
                    }

                    visible_count += 1;
                }
            }
        }

        println!(
            "  Points in frame: {}/{}",
            in_frame_count,
            scene.points.len()
        );
        println!("  Points drawn: {}", visible_count);

        // Save overlay image
        let output_path = output_dir.join(format!("m2_overlay_{:02}.png", i));
        img.save(&output_path).expect("Failed to save image");
        println!("  Saved to: {:?}", output_path);
    }

    println!("\n✅ M2 projection test complete!");
    println!("Check test_output/m2_overlay_*.png to verify points align with scene features");
}

#[test]
fn test_projection_math_simple() {
    // Simple test with known values
    let camera = Camera::new(
        100.0,
        100.0, // fx, fy
        50.0,
        50.0, // cx, cy
        100,
        100, // width, height
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // Point at (1, 0, 2) should project to (100*1/2 + 50, 100*0/2 + 50) = (100, 50)
    let world_point = Vector3::new(1.0, 0.0, 2.0);
    let pixel = camera.world_to_pixel(&world_point).unwrap();

    println!(
        "Test projection: ({}, {}, {}) -> ({}, {})",
        world_point.x, world_point.y, world_point.z, pixel.x, pixel.y
    );

    approx::assert_relative_eq!(pixel.x, 100.0, epsilon = 1e-4);
    approx::assert_relative_eq!(pixel.y, 50.0, epsilon = 1e-4);
}
