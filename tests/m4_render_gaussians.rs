//! M4 Visual Test: Render Full (Projected) Gaussians
//!
//! Produces images in `test_output/m4_render_*.png` for visual inspection.
//!
//! This test exercises the core forward-rendering math for M4:
//! - Reconstruct 3D covariance Σ from (scale, rotation)
//! - Transform Σ into camera space
//! - Project Σ to screen space via the perspective Jacobian
//! - Evaluate elliptical splats and alpha composite
//!
//! Notes:
//! - This test is intentionally a "visual test" and will write files.
//! - It will be skipped if the local calipers COLMAP path is missing.
//! - For speed, it downsamples the camera resolution and caps Gaussian count.

use std::path::PathBuf;

use nalgebra::{Matrix3, Vector3};
use sugar_rs::core::{init_from_colmap_points, Camera};
use sugar_rs::io::load_colmap_scene;
use sugar_rs::render::FullRenderer;

const CALIPERS_COLMAP_PATH: &str =
    "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0";

fn camera_with_pose(
    base_camera: &Camera,
    rotation: Matrix3<f32>,
    translation: Vector3<f32>,
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

fn downsample_camera(camera: &Camera, factor: f32) -> Camera {
    let width = ((camera.width as f32) * factor).round().max(1.0) as u32;
    let height = ((camera.height as f32) * factor).round().max(1.0) as u32;

    Camera::new(
        camera.fx * factor,
        camera.fy * factor,
        camera.cx * factor,
        camera.cy * factor,
        width,
        height,
        camera.rotation,
        camera.translation,
    )
}

#[test]
fn test_m4_render_calipers_projected_covariance() {
    let colmap_path = PathBuf::from(CALIPERS_COLMAP_PATH);
    if !colmap_path.exists() {
        println!("Skipping M4 visual test - COLMAP data not found at {CALIPERS_COLMAP_PATH}");
        return;
    }

    let scene = load_colmap_scene(&colmap_path).expect("Failed to load COLMAP scene");
    println!(
        "Loaded scene: {} cameras, {} images, {} points",
        scene.cameras.len(),
        scene.images.len(),
        scene.points.len()
    );

    let cloud = init_from_colmap_points(&scene.points);
    println!("Initialized {} Gaussians", cloud.len());

    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // For speed: use a subset of Gaussians and render at a lower resolution.
    let max_gaussians = 20_000.min(cloud.gaussians.len());
    let gaussians = &cloud.gaussians[..max_gaussians];
    println!("Rendering with {max_gaussians} Gaussians");

    // Render a couple of viewpoints.
    let num_renders = 2.min(scene.images.len());
    let mut renderer = FullRenderer::new();

    for i in 0..num_renders {
        let image_info = &scene.images[i];
        let base_camera = &scene.cameras[0]; // Single-camera dataset setup

        let rotation = image_info.rotation.to_rotation_matrix().into_inner();
        let camera_full = camera_with_pose(base_camera, rotation, image_info.translation);
        let camera = downsample_camera(&camera_full, 0.25);

        println!(
            "\nM4 render viewpoint {}/{}: {}",
            i + 1,
            num_renders,
            image_info.name
        );
        println!("  Resolution: {}x{}", camera.width, camera.height);

        let start = std::time::Instant::now();
        let img = renderer.render(gaussians, &camera);
        let elapsed = start.elapsed();
        println!("  Rendered in {:.2}s", elapsed.as_secs_f32());

        let output_path = output_dir.join(format!("m4_render_{:02}.png", i));
        img.save(&output_path)
            .expect("Failed to save M4 render image");
        println!("  Saved to: {:?}", output_path);
    }

    println!("\n✅ M4 visual render complete!");
    println!("Check test_output/m4_render_*.png");
    println!("Look for:");
    println!("  - Splats that vary in apparent size with depth");
    println!("  - Coherent scene structure (should resemble the calipers)");
}
