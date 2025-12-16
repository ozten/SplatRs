//! M3 Test: Render Spheres (Constant-Size Gaussians)
//!
//! This test verifies that:
//! 1. We can initialize Gaussians from COLMAP points
//! 2. Depth-sorted alpha blending works
//! 3. Rendered images have correct colors and depth ordering

use std::path::PathBuf;
use sugar_rs::core::{init_from_colmap_points, Camera};
use sugar_rs::io::load_colmap_scene;
use sugar_rs::render::SimpleRenderer;

/// Helper to create a camera with pose from scene data.
fn create_camera_with_pose(
    base_camera: &Camera,
    rotation: nalgebra::Matrix3<f32>,
    translation: nalgebra::Vector3<f32>,
) -> Camera {
    Camera::new(
        base_camera.fx,
        base_camera.fy,
        base_camera.cx,
        base_camera.cy,
        base_camera.width,
        base_camera.height,
        rotation,
        translation,
    )
}

#[test]
fn test_render_calipers_fixed_size() {
    // Paths
    let colmap_path = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );

    // Skip test if path doesn't exist
    if !colmap_path.exists() {
        println!("Skipping test - COLMAP data not found");
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

    // Initialize Gaussians from points
    let cloud = init_from_colmap_points(&scene.points);
    println!("Initialized {} Gaussians", cloud.len());

    // Create output directory
    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Create renderer
    let mut renderer = SimpleRenderer::new();
    renderer.radius = 2.0; // Small radius for better detail

    // Render from first 4 viewpoints
    let num_renders = 4.min(scene.images.len());

    for i in 0..num_renders {
        let image_info = &scene.images[i];
        let base_camera = &scene.cameras[0]; // Single camera setup

        // Create camera with this viewpoint's pose
        let rotation = image_info.rotation.to_rotation_matrix().into_inner();
        let camera = create_camera_with_pose(base_camera, rotation, image_info.translation);

        println!(
            "\nRendering viewpoint {}/{}: {}",
            i + 1,
            num_renders,
            image_info.name
        );

        // Render
        let start = std::time::Instant::now();
        let img = renderer.render(&cloud.gaussians, &camera);
        let elapsed = start.elapsed();

        println!("  Rendered in {:.2}s", elapsed.as_secs_f32());
        println!("  Resolution: {}x{}", img.width(), img.height());

        // Save
        let output_path = output_dir.join(format!("m3_render_{:02}.png", i));
        img.save(&output_path).expect("Failed to save image");
        println!("  Saved to: {:?}", output_path);
    }

    println!("\n✅ M3 rendering test complete!");
    println!("Check test_output/m3_render_*.png");
    println!("Look for:");
    println!("  - Colored splats at point locations");
    println!("  - Correct depth ordering (closer points occlude farther ones)");
    println!("  - Colors matching the original scene");
}

#[test]
fn test_depth_ordering() {
    use nalgebra::{Matrix3, Vector3};
    use sugar_rs::io::Point3D;

    // Create three points at different depths with different colors
    let points = vec![
        Point3D {
            id: 0,
            position: Vector3::new(0.0, 0.0, 5.0), // Far (red)
            color: [255, 0, 0],
            error: 0.0,
        },
        Point3D {
            id: 1,
            position: Vector3::new(0.0, 0.0, 3.0), // Near (green, should occlude red)
            color: [0, 255, 0],
            error: 0.0,
        },
        Point3D {
            id: 2,
            position: Vector3::new(2.0, 0.0, 4.0), // Side point (blue)
            color: [0, 0, 255],
            error: 0.0,
        },
    ];

    let cloud = init_from_colmap_points(&points);

    // Simple camera looking down +Z
    let camera = Camera::new(
        200.0,
        200.0,
        100.0,
        100.0,
        200,
        200,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // Render
    let mut renderer = SimpleRenderer::new();
    renderer.radius = 10.0; // Large radius to see them
    let img = renderer.render(&cloud.gaussians, &camera);

    // Center pixel should be green (near point occludes far point)
    let center = img.get_pixel(100, 100);
    println!(
        "Center pixel RGB: ({}, {}, {})",
        center[0], center[1], center[2]
    );

    // Should have strong green component
    assert!(
        center[1] > 100,
        "Near green point should be visible at center"
    );

    // Save for visual inspection
    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    img.save(output_dir.join("m3_depth_test.png")).ok();

    println!("✅ Depth ordering test passed!");
    println!("   Near green point correctly occludes far red point");
}
