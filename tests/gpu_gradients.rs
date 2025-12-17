//! GPU Gradients Test: Validate GPU backward pass against CPU.
//!
//! This test verifies that GPU gradient computation produces identical results
//! to CPU gradient computation.
//!
//! Success criteria:
//! - Gradient difference < 1e-3 (allowing for numerical differences)
//! - GPU backward pass is significantly faster than CPU

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_color_grads;

#[test]
#[ignore] // Only run with --features gpu and --ignored
#[cfg(feature = "gpu")]
fn test_gpu_vs_cpu_gradients() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== GPU Gradients Validation Test ===\n");

    // Create a simple scene with a few Gaussians
    let gaussians = vec![
        Gaussian::new(
            Vector3::new(0.0, 0.0, 5.0), // In front of camera
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

    // Create synthetic upstream gradients (dL/d(pixel))
    // For testing, use simple pattern: gradient increases from left to right
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels: Vec<Vector3<f32>> = (0..num_pixels)
        .map(|i| {
            let x = (i % camera.width as usize) as f32 / camera.width as f32;
            Vector3::new(x, 0.5, 1.0 - x)
        })
        .collect();

    // CPU gradients
    println!("Computing CPU gradients...");
    let t_cpu_start = std::time::Instant::now();
    let (
        _cpu_img,
        cpu_d_colors,
        cpu_d_opacity_logits,
        _cpu_d_positions,
        _cpu_d_log_scales,
        _cpu_d_rot_vecs,
        _cpu_d_bg,
    ) = render_full_color_grads(&gaussians, &camera, &d_pixels, &background);
    let cpu_time = t_cpu_start.elapsed();
    println!("  CPU time: {:?}", cpu_time);

    // GPU gradients
    println!("\nComputing GPU gradients...");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");

    let t_gpu_start = std::time::Instant::now();
    let (_gpu_pixels, gpu_grads) =
        gpu_renderer.render_with_gradients(&gaussians, &camera, &background, &d_pixels);
    let gpu_time = t_gpu_start.elapsed();
    println!("  GPU time: {:?}", gpu_time);

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("\n📊 Performance:");
    println!("  CPU: {:?}", cpu_time);
    println!("  GPU: {:?}", gpu_time);
    println!("  Speedup: {:.1}x", speedup);

    // Compare gradients
    println!("\n🔍 Gradient Comparison:");

    // Debug: check if ANY GPU gradients are non-zero
    let gpu_nonzero_count = gpu_grads.d_colors.iter().filter(|c| c.norm() > 0.0).count();
    println!("  GPU gradients with non-zero d_color: {} / {}", gpu_nonzero_count, gaussians.len());

    // Print first few gradients with ratios
    for i in 0..gaussians.len().min(3) {
        let cpu_norm = cpu_d_colors[i].norm();
        let gpu_norm = gpu_grads.d_colors[i].norm();
        let ratio = if gpu_norm > 0.0 { cpu_norm / gpu_norm } else { 0.0 };

        println!("    Gaussian {}:", i);
        println!("      CPU d_color:  [{:.6}, {:.6}, {:.6}] (norm: {:.6})",
            cpu_d_colors[i].x, cpu_d_colors[i].y, cpu_d_colors[i].z, cpu_norm);
        println!("      GPU d_color:  [{:.6}, {:.6}, {:.6}] (norm: {:.6})",
            gpu_grads.d_colors[i].x, gpu_grads.d_colors[i].y, gpu_grads.d_colors[i].z, gpu_norm);
        println!("      Ratio (CPU/GPU): {:.2}x", ratio);
        println!("      CPU d_opacity_logit: {:.6}", cpu_d_opacity_logits[i]);
        println!("      GPU d_opacity_logit: {:.6}", gpu_grads.d_opacity_logits[i]);
    }

    // Check if ratio is consistent (might indicate systematic scaling issue)
    let ratios: Vec<f32> = gaussians.iter().enumerate()
        .filter(|(i, _)| gpu_grads.d_colors[*i].norm() > 1e-6)
        .map(|(i, _)| cpu_d_colors[i].norm() / gpu_grads.d_colors[i].norm())
        .collect();
    if !ratios.is_empty() {
        let avg_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
        println!("  Average CPU/GPU ratio: {:.2}x (suggests {}x scaling issue)",
            avg_ratio, avg_ratio);
    }

    // Compare d_colors
    let mut max_color_diff = 0.0f32;
    let mut total_color_diff = 0.0f32;
    for i in 0..gaussians.len() {
        let diff = (cpu_d_colors[i] - gpu_grads.d_colors[i]).norm();
        total_color_diff += diff;
        max_color_diff = max_color_diff.max(diff);
    }
    let avg_color_diff = total_color_diff / gaussians.len() as f32;
    println!("  d_colors:");
    println!("    Avg diff: {:.6}", avg_color_diff);
    println!("    Max diff: {:.6}", max_color_diff);

    // Compare d_opacity_logits
    let mut max_opacity_diff = 0.0f32;
    let mut total_opacity_diff = 0.0f32;
    for i in 0..gaussians.len() {
        let diff = (cpu_d_opacity_logits[i] - gpu_grads.d_opacity_logits[i]).abs();
        total_opacity_diff += diff;
        max_opacity_diff = max_opacity_diff.max(diff);
    }
    let avg_opacity_diff = total_opacity_diff / gaussians.len() as f32;
    println!("  d_opacity_logits:");
    println!("    Avg diff: {:.6}", avg_opacity_diff);
    println!("    Max diff: {:.6}", max_opacity_diff);

    // Validation thresholds
    // Allow for some numerical differences between CPU and GPU
    let tolerance = 1e-3;

    assert!(
        max_color_diff < tolerance,
        "Color gradient difference too large: {} (tolerance: {})",
        max_color_diff,
        tolerance
    );

    assert!(
        max_opacity_diff < tolerance,
        "Opacity gradient difference too large: {} (tolerance: {})",
        max_opacity_diff,
        tolerance
    );

    // Check that we got meaningful speedup
    // Note: First run might be slower due to shader compilation
    // But we should still see some speedup
    if speedup > 1.0 {
        println!("\n✅ GPU gradients are faster than CPU ({:.1}x speedup)!", speedup);
    } else {
        println!(
            "\n⚠️  Warning: GPU not faster than CPU ({:.1}x). This might be first run or very small scene.",
            speedup
        );
    }

    println!("\n✅ GPU gradients match CPU within tolerance!");
    println!("   Color gradient max diff:   {:.6} (< {:.6})", max_color_diff, tolerance);
    println!("   Opacity gradient max diff: {:.6} (< {:.6})", max_opacity_diff, tolerance);
}

#[test]
#[ignore] // Run with --ignored for detailed benchmark
#[cfg(feature = "gpu")]
fn test_gpu_gradients_benchmark() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== GPU Gradients Performance Benchmark ===\n");

    // Create a more realistic scene with more Gaussians
    let num_gaussians = 500;
    let mut gaussians = Vec::new();

    for i in 0..num_gaussians {
        let angle = (i as f32 / num_gaussians as f32) * std::f32::consts::PI * 2.0;
        let radius = 2.0 + (i as f32 / num_gaussians as f32) * 3.0;

        gaussians.push(Gaussian::new(
            Vector3::new(radius * angle.cos(), radius * angle.sin(), 5.0 + (i as f32) * 0.01),
            Vector3::new(-2.5, -2.5, -2.5),
            UnitQuaternion::identity(),
            (i as f32 / num_gaussians as f32) - 0.5,
            {
                let mut sh = [[0.0f32; 3]; 16];
                let hue = i as f32 / num_gaussians as f32;
                sh[0] = [hue, 1.0 - hue, 0.5];
                sh
            },
        ));
    }

    // Larger image for more realistic workload
    let camera = Camera::new(
        200.0, 200.0,
        100.0, 100.0,
        200, 200, // Larger resolution
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.1, 0.1, 0.15);

    // Synthetic upstream gradients
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels: Vec<Vector3<f32>> = (0..num_pixels)
        .map(|i| {
            let x = (i % camera.width as usize) as f32 / camera.width as f32;
            let y = (i / camera.width as usize) as f32 / camera.height as f32;
            Vector3::new(x, y, 1.0 - x)
        })
        .collect();

    println!("Scene setup:");
    println!("  Gaussians: {}", num_gaussians);
    println!("  Resolution: {}×{}", camera.width, camera.height);
    println!("  Pixels: {}", num_pixels);

    // Warm up GPU (shader compilation happens here)
    println!("\nWarming up GPU...");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let _ = gpu_renderer.render_with_gradients(&gaussians, &camera, &background, &d_pixels);
    println!("  GPU warmed up");

    // CPU benchmark
    println!("\nCPU Backward Pass:");
    let cpu_iterations = 3;
    let mut cpu_times = Vec::new();

    for i in 0..cpu_iterations {
        let t = std::time::Instant::now();
        let _ = render_full_color_grads(&gaussians, &camera, &d_pixels, &background);
        let elapsed = t.elapsed();
        cpu_times.push(elapsed);
        println!("  Iteration {}: {:?}", i + 1, elapsed);
    }

    let cpu_avg = cpu_times.iter().sum::<std::time::Duration>() / cpu_iterations as u32;
    let cpu_min = cpu_times.iter().min().unwrap();
    println!("  Average: {:?}", cpu_avg);
    println!("  Best:    {:?}", cpu_min);

    // GPU benchmark
    println!("\nGPU Backward Pass:");
    let gpu_iterations = 10; // More iterations since GPU is faster

    let mut gpu_times = Vec::new();
    for i in 0..gpu_iterations {
        let t = std::time::Instant::now();
        let _ = gpu_renderer.render_with_gradients(&gaussians, &camera, &background, &d_pixels);
        let elapsed = t.elapsed();
        gpu_times.push(elapsed);
        println!("  Iteration {}: {:?}", i + 1, elapsed);
    }

    let gpu_avg = gpu_times.iter().sum::<std::time::Duration>() / gpu_iterations as u32;
    let gpu_min = gpu_times.iter().min().unwrap();
    println!("  Average: {:?}", gpu_avg);
    println!("  Best:    {:?}", gpu_min);

    // Calculate speedup
    let avg_speedup = cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64();
    let best_speedup = cpu_min.as_secs_f64() / gpu_min.as_secs_f64();

    println!("\n📊 Performance Summary:");
    println!("  CPU (avg):  {:?}", cpu_avg);
    println!("  GPU (avg):  {:?}", gpu_avg);
    println!("  Speedup (avg): {:.1}x", avg_speedup);
    println!("  Speedup (best): {:.1}x", best_speedup);

    if avg_speedup > 5.0 {
        println!("\n✅ GPU is significantly faster! ({:.1}x average speedup)", avg_speedup);
    } else if avg_speedup > 2.0 {
        println!("\n✅ GPU is faster ({:.1}x average speedup)", avg_speedup);
    } else {
        println!(
            "\n⚠️  Warning: GPU speedup is lower than expected ({:.1}x)",
            avg_speedup
        );
        println!("    This might indicate GPU overhead is dominating for this scene size.");
    }
}
