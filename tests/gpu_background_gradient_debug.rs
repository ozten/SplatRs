//! Test ONLY the background gradient computation

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_background_gradient_only() {
    use sugar_rs::gpu::GpuRenderer;
    use sugar_rs::render::render_full_color_grads;

    println!("\n=== Background Gradient Debug Test ===\n");

    // Simple scene: single Gaussian far from center
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0,
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [1.0, 0.0, 0.0];  // Red
            sh
        },
    );

    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.5, 0.5, 0.5);  // Gray background

    // Create uniform upstream gradients (all pixels get same gradient)
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels: Vec<Vector3<f32>> = vec![Vector3::new(1.0, 1.0, 1.0); num_pixels];

    // CPU rendering + gradients
    println!("CPU backward pass...");
    let (
        _cpu_img,
        _cpu_d_colors,
        _cpu_d_opacity_logits,
        _cpu_d_positions,
        _cpu_d_log_scales,
        _cpu_d_rot_vecs,
        cpu_d_bg,
    ) = render_full_color_grads(&[gaussian.clone()], &camera, &d_pixels, &background);

    println!("CPU d_background: {:?}", cpu_d_bg);

    // GPU rendering + gradients
    println!("\nGPU backward pass...");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let (_gpu_pixels, gpu_grads_2d) = gpu_renderer
        .render_with_gradients(&[gaussian], &camera, &background, &d_pixels)
        .expect("GPU render_with_gradients failed");

    println!("GPU d_background: {:?}", gpu_grads_2d.d_background);

    // Compare
    let diff = (cpu_d_bg - gpu_grads_2d.d_background).norm();
    println!("\nDifference: {:.6} (CPU norm: {:.6}, GPU norm: {:.6})",
        diff, cpu_d_bg.norm(), gpu_grads_2d.d_background.norm());

    // Print component-wise comparison
    println!("\nComponent-wise:");
    println!("  R: CPU={:.6}, GPU={:.6}, diff={:.6}", cpu_d_bg.x, gpu_grads_2d.d_background.x, (cpu_d_bg.x - gpu_grads_2d.d_background.x).abs());
    println!("  G: CPU={:.6}, GPU={:.6}, diff={:.6}", cpu_d_bg.y, gpu_grads_2d.d_background.y, (cpu_d_bg.y - gpu_grads_2d.d_background.y).abs());
    println!("  B: CPU={:.6}, GPU={:.6}, diff={:.6}", cpu_d_bg.z, gpu_grads_2d.d_background.z, (cpu_d_bg.z - gpu_grads_2d.d_background.z).abs());

    if diff > 0.1 {
        println!("\n❌ Background gradient FAILS - diff {:.6} > 0.1", diff);
        panic!("GPU background gradient is incorrect");
    } else {
        println!("\n✅ Background gradient matches within tolerance!");
    }
}
