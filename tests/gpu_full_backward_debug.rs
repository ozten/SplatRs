//! Full GPU backward pass debugging test
//!
//! This test compares the COMPLETE backward pass (rasterization + projection)
//! between GPU and CPU to find remaining bugs.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_color_grads;

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_full_backward_pass_gpu_vs_cpu() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== Full GPU Backward Pass Debugging ===\n");

    // Create a realistic scene with multiple Gaussians
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

    // Camera with identity rotation for now
    let camera = Camera::new(
        100.0, 100.0,
        32.0, 32.0,
        64, 64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let background = Vector3::new(0.1, 0.1, 0.15);

    println!("Scene:");
    println!("  Gaussians: {}", gaussians.len());
    println!("  Camera: {}×{}", camera.width, camera.height);
    println!("  Background: {:?}", background);

    // Debug: Print quaternion components
    println!("\nQuaternion debug:");
    for (i, g) in gaussians.iter().enumerate() {
        println!("  Gaussian {}: quat=(i={:.6}, j={:.6}, k={:.6}, w={:.6})",
            i, g.rotation.i, g.rotation.j, g.rotation.k, g.rotation.w);
    }

    // Create synthetic upstream gradients (dL/d(pixel))
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels: Vec<Vector3<f32>> = (0..num_pixels)
        .map(|i| {
            let x = (i % camera.width as usize) as f32 / camera.width as f32;
            Vector3::new(x, 0.5, 1.0 - x)
        })
        .collect();

    println!("\n--- CPU Full Backward Pass ---");
    let (
        cpu_img,
        cpu_d_colors,
        cpu_d_opacity_logits,
        cpu_d_positions,
        cpu_d_log_scales,
        cpu_d_rot_vecs,
        cpu_d_bg,
    ) = render_full_color_grads(&gaussians, &camera, &d_pixels, &background);

    println!("CPU Results:");
    for i in 0..gaussians.len() {
        println!("  Gaussian {}:", i);
        println!("    d_color:    {:?}", cpu_d_colors[i]);
        println!("    d_opacity:  {:.6}", cpu_d_opacity_logits[i]);
        println!("    d_position: {:?}", cpu_d_positions[i]);
        println!("    d_scale:    {:?}", cpu_d_log_scales[i]);
        println!("    d_rotation: {:?}", cpu_d_rot_vecs[i]);
    }
    println!("  d_background: {:?}", cpu_d_bg);

    // Check for non-zero gradients
    let cpu_has_nonzero_pos = cpu_d_positions.iter().any(|g| g.norm() > 1e-6);
    let cpu_has_nonzero_scale = cpu_d_log_scales.iter().any(|g| g.norm() > 1e-6);
    let cpu_has_nonzero_rot = cpu_d_rot_vecs.iter().any(|g| g.norm() > 1e-6);
    println!("\nCPU non-zero gradients:");
    println!("  Position: {}", cpu_has_nonzero_pos);
    println!("  Scale:    {}", cpu_has_nonzero_scale);
    println!("  Rotation: {}", cpu_has_nonzero_rot);

    println!("\n--- GPU Full Backward Pass ---");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let (gpu_pixels, gpu_grads_2d) = gpu_renderer
        .render_with_gradients(&gaussians, &camera, &background, &d_pixels)
        .expect("GPU render_with_gradients failed");

    // GPU projection backward
    let (gpu_d_positions, gpu_d_log_scales, gpu_d_rot_vecs, _gpu_d_sh) =
        gpu_renderer.project_gradients_backward(&gaussians, &camera, &gpu_grads_2d);

    println!("GPU Results:");
    for i in 0..gaussians.len() {
        println!("  Gaussian {}:", i);
        println!("    d_color:    {:?}", gpu_grads_2d.d_colors[i]);
        println!("    d_opacity:  {:.6}", gpu_grads_2d.d_opacity_logits[i]);
        println!("    d_position: {:?}", gpu_d_positions[i]);
        println!("    d_scale:    {:?}", gpu_d_log_scales[i]);
        println!("    d_rotation: {:?}", gpu_d_rot_vecs[i]);
    }
    println!("  d_background: {:?}", gpu_grads_2d.d_background);

    // Check for non-zero gradients
    let gpu_has_nonzero_pos = gpu_d_positions.iter().any(|g| g.norm() > 1e-6);
    let gpu_has_nonzero_scale = gpu_d_log_scales.iter().any(|g| g.norm() > 1e-6);
    let gpu_has_nonzero_rot = gpu_d_rot_vecs.iter().any(|g| g.norm() > 1e-6);
    println!("\nGPU non-zero gradients:");
    println!("  Position: {}", gpu_has_nonzero_pos);
    println!("  Scale:    {}", gpu_has_nonzero_scale);
    println!("  Rotation: {}", gpu_has_nonzero_rot);

    println!("\n--- Comparison ---");

    // Compare rendered images
    let mut pixel_diffs = Vec::new();
    for y in 0..camera.height {
        for x in 0..camera.width {
            let idx = (y * camera.width + x) as usize;
            let cpu_px = cpu_img.get_pixel(x, y);
            let gpu_px = gpu_pixels[idx];

            let cpu_r = cpu_px[0] as f32 / 255.0;
            let cpu_g = cpu_px[1] as f32 / 255.0;
            let cpu_b = cpu_px[2] as f32 / 255.0;

            let diff_r = (cpu_r - gpu_px.x).abs();
            let diff_g = (cpu_g - gpu_px.y).abs();
            let diff_b = (cpu_b - gpu_px.z).abs();

            pixel_diffs.push(diff_r.max(diff_g).max(diff_b));
        }
    }

    pixel_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_pixel_diff = pixel_diffs.last().copied().unwrap_or(0.0);
    let median_pixel_diff = pixel_diffs[pixel_diffs.len() / 2];

    println!("Forward pass (rendered images):");
    println!("  Max pixel diff:    {:.6}", max_pixel_diff);
    println!("  Median pixel diff: {:.6}", median_pixel_diff);

    if max_pixel_diff > 0.01 {
        println!("  ❌ Forward pass mismatch! GPU rendering differs from CPU.");
    } else {
        println!("  ✅ Forward pass matches!");
    }

    // Compare gradients
    println!("\nGradient comparison:");

    for i in 0..gaussians.len() {
        println!("\n  Gaussian {}:", i);

        // Color gradients
        let color_diff = (cpu_d_colors[i] - gpu_grads_2d.d_colors[i]).norm();
        println!("    d_color diff:    {:.6} (CPU: {:.6}, GPU: {:.6})",
            color_diff, cpu_d_colors[i].norm(), gpu_grads_2d.d_colors[i].norm());

        // Opacity gradients
        let opacity_diff = (cpu_d_opacity_logits[i] - gpu_grads_2d.d_opacity_logits[i]).abs();
        println!("    d_opacity diff:  {:.6} (CPU: {:.6}, GPU: {:.6})",
            opacity_diff, cpu_d_opacity_logits[i].abs(), gpu_grads_2d.d_opacity_logits[i].abs());

        // Position gradients
        let pos_diff = (cpu_d_positions[i] - gpu_d_positions[i]).norm();
        println!("    d_position diff: {:.6} (CPU: {:.6}, GPU: {:.6})",
            pos_diff, cpu_d_positions[i].norm(), gpu_d_positions[i].norm());

        // Scale gradients
        let scale_diff = (cpu_d_log_scales[i] - gpu_d_log_scales[i]).norm();
        println!("    d_scale diff:    {:.6} (CPU: {:.6}, GPU: {:.6})",
            scale_diff, cpu_d_log_scales[i].norm(), gpu_d_log_scales[i].norm());

        // Rotation gradients
        let rot_diff = (cpu_d_rot_vecs[i] - gpu_d_rot_vecs[i]).norm();
        println!("    d_rotation diff: {:.6} (CPU: {:.6}, GPU: {:.6})",
            rot_diff, cpu_d_rot_vecs[i].norm(), gpu_d_rot_vecs[i].norm());
    }

    // Background gradient
    let bg_diff = (cpu_d_bg - gpu_grads_2d.d_background).norm();
    println!("\n  d_background diff: {:.6} (CPU: {:.6}, GPU: {:.6})",
        bg_diff, cpu_d_bg.norm(), gpu_grads_2d.d_background.norm());

    // Summary diagnostics
    println!("\n--- Diagnostics ---");

    if max_pixel_diff > 0.01 {
        println!("❌ FORWARD PASS BUG: GPU rendering differs from CPU");
        println!("   This means the bug is in the forward pass, not backward pass");
    }

    // Check each gradient type
    let tolerance = 0.01;
    let mut has_bugs = false;

    for i in 0..gaussians.len() {
        let color_diff = (cpu_d_colors[i] - gpu_grads_2d.d_colors[i]).norm();
        let pos_diff = (cpu_d_positions[i] - gpu_d_positions[i]).norm();
        let scale_diff = (cpu_d_log_scales[i] - gpu_d_log_scales[i]).norm();
        let rot_diff = (cpu_d_rot_vecs[i] - gpu_d_rot_vecs[i]).norm();

        if color_diff > tolerance {
            println!("❌ Gaussian {}: Color gradient mismatch (diff={:.6})", i, color_diff);
            has_bugs = true;
        }
        if pos_diff > tolerance {
            println!("❌ Gaussian {}: Position gradient mismatch (diff={:.6})", i, pos_diff);
            has_bugs = true;
        }
        if scale_diff > tolerance {
            println!("❌ Gaussian {}: Scale gradient mismatch (diff={:.6})", i, scale_diff);
            has_bugs = true;
        }
        if rot_diff > tolerance {
            println!("❌ Gaussian {}: Rotation gradient mismatch (diff={:.6})", i, rot_diff);
            has_bugs = true;
        }
    }

    if !has_bugs {
        println!("✅ All gradients match within tolerance!");
    } else {
        panic!("GPU backward pass produces incorrect gradients");
    }
}
