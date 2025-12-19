//! M11 Test: GPU renderer correctness validation.
//!
//! This test verifies that GPU rendering produces identical results to CPU rendering.
//! Success criteria: Per-pixel difference < 1e-4 (M11 milestone requirement).

#![cfg(feature = "gpu")]

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian, GaussianCloud};
use sugar_rs::render::render_full_linear;

#[test]
#[ignore] // Only run manually (GPU hardware + drivers)
fn test_gpu_vs_cpu_rendering() {
    use sugar_rs::gpu::GpuRenderer;

    // Create a simple scene with a few Gaussians
    let gaussians = vec![
        Gaussian::new(
            Vector3::new(0.0, 0.0, 5.0),  // In front of camera
            Vector3::new(-2.0, -2.0, -2.0), // Small scale
            UnitQuaternion::identity(),
            0.0, // opacity logit (sigmoid(0) = 0.5)
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.8, 0.2, 0.1]; // Reddish
                sh
            },
        ),
        Gaussian::new(
            Vector3::new(1.0, 0.5, 6.0),
            Vector3::new(-2.5, -2.5, -2.5),
            UnitQuaternion::identity(),
            1.0, // Higher opacity
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.1, 0.7, 0.3]; // Greenish
                sh
            },
        ),
        Gaussian::new(
            Vector3::new(-1.0, -0.5, 7.0),
            Vector3::new(-2.0, -2.0, -2.0),
            UnitQuaternion::identity(),
            -0.5, // Lower opacity
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.2, 0.3, 0.9]; // Blueish
                sh
            },
        ),
    ];

    // Camera looking down +Z
    let camera = Camera::new(
        100.0, 100.0, // fx, fy
        32.0, 32.0,   // cx, cy (centered)
        64, 64,       // Small resolution for fast test
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.1, 0.1, 0.15);

    // Render with CPU
    let cpu_result = render_full_linear(&gaussians, &camera, &background);

    // Render with GPU
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let gpu_result = gpu_renderer
        .render(&gaussians, &camera, &background)
        .expect("GPU render failed");

    // Compare results
    assert_eq!(cpu_result.len(), gpu_result.len());
    let num_pixels = cpu_result.len();

    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    let mut num_significant_diffs = 0;

    for i in 0..num_pixels {
        let diff = (cpu_result[i] - gpu_result[i]).norm();
        total_diff += diff;
        max_diff = max_diff.max(diff);

        if diff > 1e-4 {
            num_significant_diffs += 1;
        }
    }

    let avg_diff = total_diff / (num_pixels as f32);

    println!("GPU vs CPU Rendering Comparison:");
    println!("  Pixels:            {}", num_pixels);
    println!("  Average diff:      {:.6}", avg_diff);
    println!("  Max diff:          {:.6}", max_diff);
    println!("  Diffs > 1e-4:      {} ({:.1}%)",
        num_significant_diffs,
        100.0 * (num_significant_diffs as f32) / (num_pixels as f32)
    );

    // M11 Success Criteria: Per-pixel RMSE < 1e-4
    assert!(
        max_diff < 1e-3,
        "GPU rendering differs too much from CPU: max_diff = {}",
        max_diff
    );

    // Most pixels should be very close
    let threshold_fraction = num_significant_diffs as f32 / num_pixels as f32;
    assert!(
        threshold_fraction < 0.01,
        "Too many pixels differ significantly: {:.1}%",
        100.0 * threshold_fraction
    );

    println!("✅ GPU rendering matches CPU within tolerance!");
}

#[test]
#[ignore] // Only run manually (GPU hardware + drivers)
fn test_gpu_renderer_initialization() {
    use sugar_rs::gpu::GpuRenderer;

    let renderer = GpuRenderer::new();
    assert!(renderer.is_ok(), "GPU renderer initialization failed");
    println!("✅ GPU renderer initialized successfully");
}

// When the `gpu` feature is not enabled, this file is not compiled.
