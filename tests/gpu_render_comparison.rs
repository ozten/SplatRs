//! Compare GPU render() vs render_with_gradients() to isolate forward pass issues

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_linear;

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_render_vs_render_with_gradients() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== Comparing render() vs render_with_gradients() ===\n");

    // Same scene as full backward test
    let gaussians = vec![
        Gaussian::new(
            Vector3::new(0.0, 0.0, 5.0),
            Vector3::new(-2.0, -2.0, -2.0),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            0.0,
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.8, 0.2, 0.1];
                sh
            },
        ),
        Gaussian::new(
            Vector3::new(1.0, 0.5, 6.0),
            Vector3::new(-2.5, -2.5, -2.5),
            UnitQuaternion::from_euler_angles(-0.1, 0.15, -0.2),
            1.0,
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.1, 0.7, 0.3];
                sh
            },
        ),
        Gaussian::new(
            Vector3::new(-1.0, -0.5, 7.0),
            Vector3::new(-2.0, -2.0, -2.0),
            UnitQuaternion::from_euler_angles(0.05, -0.1, 0.25),
            -0.5,
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.2, 0.3, 0.9];
                sh
            },
        ),
    ];

    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.1, 0.1, 0.15);

    // CPU rendering
    println!("CPU rendering...");
    let cpu_pixels = render_full_linear(&gaussians, &camera, &background);

    // GPU rendering via render()
    println!("GPU render()...");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let gpu_pixels_render = gpu_renderer
        .render(&gaussians, &camera, &background)
        .expect("GPU render failed");

    // GPU rendering via render_with_gradients()
    println!("GPU render_with_gradients()...");
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels: Vec<Vector3<f32>> = vec![Vector3::new(1.0, 1.0, 1.0); num_pixels];
    let (gpu_pixels_with_grad, _grads) = gpu_renderer
        .render_with_gradients(&gaussians, &camera, &background, &d_pixels)
        .expect("GPU render_with_gradients failed");

    // Compare
    let mut max_diff_render = 0.0f32;
    let mut max_diff_with_grad = 0.0f32;

    for i in 0..num_pixels {
        let cpu_px = cpu_pixels[i];
        let gpu_render_px = gpu_pixels_render[i];
        let gpu_with_grad_px = gpu_pixels_with_grad[i];

        let diff_r = (cpu_px - gpu_render_px).norm();
        let diff_wg = (cpu_px - gpu_with_grad_px).norm();

        max_diff_render = max_diff_render.max(diff_r);
        max_diff_with_grad = max_diff_with_grad.max(diff_wg);
    }

    println!("\n--- Results ---");
    println!("Max diff render():                {:.6}", max_diff_render);
    println!("Max diff render_with_gradients(): {:.6}", max_diff_with_grad);

    if max_diff_render > 0.01 {
        println!("❌ render() produces incorrect output");
    } else {
        println!("✅ render() matches CPU");
    }

    if max_diff_with_grad > 0.01 {
        println!("❌ render_with_gradients() produces incorrect output");
    } else {
        println!("✅ render_with_gradients() matches CPU");
    }

    assert!(max_diff_render <= 0.01, "render() max diff {:.6} > 0.01", max_diff_render);
    assert!(max_diff_with_grad <= 0.01, "render_with_gradients() max diff {:.6} > 0.01", max_diff_with_grad);
}
