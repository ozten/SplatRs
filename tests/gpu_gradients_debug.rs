//! Debug test to inspect intermediate buffers

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_inspect_intermediates() {
    use sugar_rs::gpu::{ContributionGPU, GpuRenderer, MAX_CONTRIBUTIONS_PER_PIXEL};

    println!("\n=== Inspecting GPU Intermediate Buffers ===\n");

    // Simple scene: one Gaussian in front of camera
    let gaussians = vec![Gaussian::new(
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0,
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [0.8, 0.2, 0.1];
            sh
        },
    )];

    // Small camera for easy inspection
    let camera = Camera::new(
        100.0, 100.0,
        4.0, 4.0,
        8, 8, // Very small resolution
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.1, 0.1, 0.15);
    let num_pixels = (camera.width * camera.height) as usize;

    println!("Scene:");
    println!("  Resolution: {}×{} = {} pixels", camera.width, camera.height, num_pixels);
    println!("  Gaussians: {}", gaussians.len());

    // Create fake upstream gradients
    let d_pixels: Vec<Vector3<f32>> = vec![Vector3::new(1.0, 1.0, 1.0); num_pixels];

    // Render with gradients
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let (_pixels, _grads) =
        gpu_renderer.render_with_gradients(&gaussians, &camera, &background, &d_pixels);

    // Note: We can't easily inspect GPU buffers from here without modifying the renderer
    // But we can check the final gradients

    println!("\n✅ Test completed - would need to modify renderer to expose intermediate buffers for inspection");
    println!("   Suggestion: Add debug output directly in render_with_gradients()");
}
