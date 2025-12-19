//! Test ONLY the forward pass (rendering) to isolate bugs

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_linear;

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_forward_rendering_only() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== GPU Forward Rendering Test ===\n");

    // Single Gaussian with non-trivial rotation
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::from_euler_angles(0.3, 0.2, 0.1),
        0.5,  // opacity logit
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [1.0, 0.0, 0.0];  // Red
            sh
        },
    );

    println!("Gaussian rotation: i={:.6}, j={:.6}, k={:.6}, w={:.6}",
        gaussian.rotation.i, gaussian.rotation.j, gaussian.rotation.k, gaussian.rotation.w);

    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.0, 0.0, 0.0);

    // CPU rendering
    println!("\nCPU Rendering...");
    let cpu_pixels = render_full_linear(&[gaussian.clone()], &camera, &background);

    // GPU rendering
    println!("GPU Rendering...");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let gpu_pixels = gpu_renderer
        .render(&[gaussian], &camera, &background)
        .expect("GPU render failed");

    // Compare pixel-by-pixel
    let mut differences = Vec::new();
    for idx in 0..(camera.width * camera.height) as usize {
        let cpu_px = cpu_pixels[idx];
        let gpu_px = gpu_pixels[idx];

        let diff_r = (cpu_px.x - gpu_px.x).abs();
        let diff_g = (cpu_px.y - gpu_px.y).abs();
        let diff_b = (cpu_px.z - gpu_px.z).abs();
        let max_diff = diff_r.max(diff_g).max(diff_b);

        differences.push(max_diff);
    }

    differences.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_diff = differences.last().copied().unwrap_or(0.0);
    let median_diff = differences[differences.len() / 2];
    let p90_diff = differences[(differences.len() as f32 * 0.9) as usize];

    println!("\n--- Results ---");
    println!("Pixel differences:");
    println!("  Max:    {:.6}", max_diff);
    println!("  P90:    {:.6}", p90_diff);
    println!("  Median: {:.6}", median_diff);

    // Sample a few pixels to see values
    println!("\nSample pixels (center region):");
    for dy in -2..=2 {
        for dx in -2..=2 {
            let x = (camera.width / 2) as i32 + dx;
            let y = (camera.height / 2) as i32 + dy;
            if x >= 0 && x < camera.width as i32 && y >= 0 && y < camera.height as i32 {
                let idx = (y * camera.width as i32 + x) as usize;
                let cpu_px = cpu_pixels[idx];
                let gpu_px = gpu_pixels[idx];

                println!("  ({:2},{:2}): CPU=({:.3},{:.3},{:.3}) GPU=({:.3},{:.3},{:.3})",
                    x, y,
                    cpu_px.x, cpu_px.y, cpu_px.z,
                    gpu_px.x, gpu_px.y, gpu_px.z);
            }
        }
    }

    if max_diff > 0.01 {
        println!("\n❌ GPU rendering FAILS - max diff {:.6} > 0.01", max_diff);
        panic!("GPU forward rendering produces incorrect output");
    } else {
        println!("\n✅ GPU rendering matches CPU within tolerance!");
    }
}
