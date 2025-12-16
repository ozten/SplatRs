//! Test to verify colored points render correctly

use nalgebra::{Matrix3, Vector3};
use std::path::PathBuf;
use sugar_rs::core::{init_from_colmap_points, Camera};
use sugar_rs::io::{load_colmap_scene, Point3D};
use sugar_rs::render::SimpleRenderer;

#[test]
#[ignore] // E2E test - requires external dataset (use `cargo test -- --ignored`)
fn test_render_only_colorful_points() {
    let colmap_path = PathBuf::from(
        "/Users/ozten/Projects/GuassianPlay/digital_calipers2_project/colmap_workspace/sparse/0",
    );

    if !colmap_path.exists() {
        println!("Skipping test");
        return;
    }

    let scene = load_colmap_scene(&colmap_path).expect("Failed to load scene");

    // Filter to ONLY highly saturated points
    let colorful_points: Vec<Point3D> = scene
        .points
        .iter()
        .filter(|p| {
            let r = p.color[0] as i32;
            let g = p.color[1] as i32;
            let b = p.color[2] as i32;
            let max_diff = (r - g).abs().max((r - b).abs()).max((g - b).abs());
            max_diff > 40 // Somewhat saturated
        })
        .cloned()
        .collect();

    println!(
        "Rendering {} colorful points out of {} total",
        colorful_points.len(),
        scene.points.len()
    );

    let cloud = init_from_colmap_points(&colorful_points);

    // Use first viewpoint
    let image_info = &scene.images[0];
    let base_camera = scene.cameras.values().next().expect("No cameras found");

    let rotation = image_info.rotation.to_rotation_matrix().into_inner();
    let camera = Camera::new(
        base_camera.fx,
        base_camera.fy,
        base_camera.cx,
        base_camera.cy,
        base_camera.width,
        base_camera.height,
        rotation,
        image_info.translation,
    );

    // Render with larger radius to make them visible
    let mut renderer = SimpleRenderer::new();
    renderer.radius = 5.0; // Bigger splats

    let img = renderer.render(&cloud.gaussians, &camera);

    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let output_path = output_dir.join("colorful_points_only.png");
    img.save(&output_path).expect("Failed to save");

    println!("Saved to: {:?}", output_path);
    println!("This should show colors if the renderer is working!");
}
