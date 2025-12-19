//! Debug test for GPU projection backward pass
//!
//! This test compares GPU vs CPU projection backward implementation
//! to identify where gradients are becoming zero.

use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};
use sugar_rs::core::{Camera, Gaussian};

#[cfg(feature = "gpu")]
use sugar_rs::gpu::{chain_2d_to_3d_gradients_cpu, GaussianGradients2D};

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_projection_backward_gpu_vs_cpu() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== GPU Projection Backward Debugging ===\n");

    // Create simple scene: single Gaussian
    let position = Vector3::new(0.0, 0.0, 5.0);
    let log_scale = Vector3::new(-2.0, -2.0, -2.0);
    let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);

    let gaussians = vec![Gaussian::new(
        position,
        log_scale,
        rotation,
        0.0,
        {
            let mut sh = [[0.0f32; 3]; 16];
            sh[0] = [0.8, 0.2, 0.1];
            sh
        },
    )];

    // Camera
    let camera = Camera::new(
        100.0, 100.0, // fx, fy
        32.0, 32.0,   // cx, cy
        64, 64,       // resolution
        Matrix3::identity(),
        Vector3::zeros(),
    );

    println!("Test scene:");
    println!("  Gaussian position: {:?}", position);
    println!("  Gaussian log_scale: {:?}", log_scale);
    println!("  Gaussian rotation (euler): (0.1, 0.2, 0.3)");
    println!("  Camera fx={}, fy={}", camera.fx, camera.fy);

    // Create non-trivial 2D gradients
    let mut grads_2d = GaussianGradients2D::zeros(1);

    // Gradient w.r.t. 2D mean (pixel position)
    grads_2d.d_mean_px[0] = Vector2::new(1.0, 0.5);

    // Gradient w.r.t. 2D covariance (xx, xy, yy)
    grads_2d.d_cov_2d[0] = Vector3::new(0.8, 0.3, 0.6);

    // Color gradient (for SH)
    grads_2d.d_colors[0] = Vector3::new(0.2, 0.3, 0.4);

    println!("\nInput 2D gradients:");
    println!("  d_mean_px:  {:?}", grads_2d.d_mean_px[0]);
    println!("  d_cov_2d:   {:?}", grads_2d.d_cov_2d[0]);
    println!("  d_color:    {:?}", grads_2d.d_colors[0]);

    // CPU projection backward (reference)
    println!("\n--- CPU Projection Backward ---");
    let (cpu_d_pos, cpu_d_scale, cpu_d_rot, cpu_d_bg) =
        chain_2d_to_3d_gradients_cpu(&grads_2d, &gaussians, &camera);

    println!("CPU Results:");
    println!("  d_position:  {:?}", cpu_d_pos[0]);
    println!("  d_log_scale: {:?}", cpu_d_scale[0]);
    println!("  d_rot_vec:   {:?}", cpu_d_rot[0]);
    println!("  d_background: {:?}", cpu_d_bg);

    // GPU projection backward
    println!("\n--- GPU Projection Backward ---");
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let (gpu_d_pos, gpu_d_scale, gpu_d_rot, gpu_d_sh) =
        gpu_renderer.project_gradients_backward(&gaussians, &camera, &grads_2d);

    println!("GPU Results:");
    println!("  d_position:  {:?}", gpu_d_pos[0]);
    println!("  d_log_scale: {:?}", gpu_d_scale[0]);
    println!("  d_rot_vec:   {:?}", gpu_d_rot[0]);
    println!("  d_sh[0]:     {:?}", gpu_d_sh[0][0]);

    // Comparison
    println!("\n--- Comparison ---");

    let pos_diff = (cpu_d_pos[0] - gpu_d_pos[0]).norm();
    let scale_diff = (cpu_d_scale[0] - gpu_d_scale[0]).norm();
    let rot_diff = (cpu_d_rot[0] - gpu_d_rot[0]).norm();

    println!("Differences (L2 norm):");
    println!("  d_position:  {:.6} (CPU norm: {:.6}, GPU norm: {:.6})",
        pos_diff, cpu_d_pos[0].norm(), gpu_d_pos[0].norm());
    println!("  d_log_scale: {:.6} (CPU norm: {:.6}, GPU norm: {:.6})",
        scale_diff, cpu_d_scale[0].norm(), gpu_d_scale[0].norm());
    println!("  d_rot_vec:   {:.6} (CPU norm: {:.6}, GPU norm: {:.6})",
        rot_diff, cpu_d_rot[0].norm(), gpu_d_rot[0].norm());

    // Component-wise comparison
    println!("\nComponent-wise comparison:");
    println!("  d_position.x:  CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_pos[0].x, gpu_d_pos[0].x, (cpu_d_pos[0].x - gpu_d_pos[0].x).abs());
    println!("  d_position.y:  CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_pos[0].y, gpu_d_pos[0].y, (cpu_d_pos[0].y - gpu_d_pos[0].y).abs());
    println!("  d_position.z:  CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_pos[0].z, gpu_d_pos[0].z, (cpu_d_pos[0].z - gpu_d_pos[0].z).abs());

    println!("\n  d_log_scale.x: CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_scale[0].x, gpu_d_scale[0].x, (cpu_d_scale[0].x - gpu_d_scale[0].x).abs());
    println!("  d_log_scale.y: CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_scale[0].y, gpu_d_scale[0].y, (cpu_d_scale[0].y - gpu_d_scale[0].y).abs());
    println!("  d_log_scale.z: CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_scale[0].z, gpu_d_scale[0].z, (cpu_d_scale[0].z - gpu_d_scale[0].z).abs());

    println!("\n  d_rot_vec.x:   CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_rot[0].x, gpu_d_rot[0].x, (cpu_d_rot[0].x - gpu_d_rot[0].x).abs());
    println!("  d_rot_vec.y:   CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_rot[0].y, gpu_d_rot[0].y, (cpu_d_rot[0].y - gpu_d_rot[0].y).abs());
    println!("  d_rot_vec.z:   CPU={:.6}  GPU={:.6}  diff={:.6}",
        cpu_d_rot[0].z, gpu_d_rot[0].z, (cpu_d_rot[0].z - gpu_d_rot[0].z).abs());

    // Diagnostics
    println!("\n--- Diagnostics ---");

    if gpu_d_pos[0].norm() < 1e-6 && cpu_d_pos[0].norm() > 1e-3 {
        println!("❌ GPU position gradient is zero while CPU is non-zero!");
        println!("   This indicates a bug in the shader's position gradient computation.");
    }

    if gpu_d_scale[0].norm() < 1e-6 && cpu_d_scale[0].norm() > 1e-3 {
        println!("❌ GPU scale gradient is zero while CPU is non-zero!");
        println!("   This indicates a bug in the shader's scale gradient computation.");
    }

    if gpu_d_rot[0].norm() < 1e-6 && cpu_d_rot[0].norm() > 1e-3 {
        println!("❌ GPU rotation gradient is zero while CPU is non-zero!");
        println!("   This indicates a bug in the shader's rotation gradient computation.");
    }

    let tolerance = 1e-3;
    if pos_diff > tolerance {
        println!("❌ Position gradient mismatch exceeds tolerance!");
    }
    if scale_diff > tolerance {
        println!("❌ Scale gradient mismatch exceeds tolerance!");
    }
    if rot_diff > tolerance {
        println!("❌ Rotation gradient mismatch exceeds tolerance!");
    }

    // Assertions
    assert!(
        pos_diff < tolerance,
        "Position gradient mismatch: diff={:.6} (tolerance={:.6})",
        pos_diff,
        tolerance
    );
    assert!(
        scale_diff < tolerance,
        "Scale gradient mismatch: diff={:.6} (tolerance={:.6})",
        scale_diff,
        tolerance
    );
    assert!(
        rot_diff < tolerance,
        "Rotation gradient mismatch: diff={:.6} (tolerance={:.6})",
        rot_diff,
        tolerance
    );

    println!("\n✅ GPU projection backward matches CPU within tolerance!");
}

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_projection_backward_zero_check() {
    use sugar_rs::gpu::GpuRenderer;

    println!("\n=== Zero Gradient Bug Reproduction ===\n");

    // Create multiple Gaussians with varying parameters
    let num_gaussians = 5;
    let mut gaussians = Vec::new();

    for i in 0..num_gaussians {
        let angle = (i as f32 / num_gaussians as f32) * std::f32::consts::PI * 2.0;
        gaussians.push(Gaussian::new(
            Vector3::new(angle.cos() * 2.0, angle.sin() * 2.0, 5.0 + i as f32 * 0.5),
            Vector3::new(-2.0 + i as f32 * 0.1, -2.0, -2.0),
            UnitQuaternion::from_euler_angles(0.1 * i as f32, 0.05, 0.0),
            0.0,
            {
                let mut sh = [[0.0f32; 3]; 16];
                sh[0] = [0.5 + i as f32 * 0.1, 0.3, 0.2];
                sh
            },
        ));
    }

    let camera = Camera::new(
        200.0, 200.0,
        64.0, 64.0,
        128, 128,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // Create non-zero 2D gradients for all Gaussians
    let mut grads_2d = GaussianGradients2D::zeros(num_gaussians);
    for i in 0..num_gaussians {
        grads_2d.d_mean_px[i] = Vector2::new(0.5 + i as f32 * 0.1, 0.3);
        grads_2d.d_cov_2d[i] = Vector3::new(0.4, 0.2, 0.5);
        grads_2d.d_colors[i] = Vector3::new(0.1, 0.2, 0.3);
    }

    println!("Testing {} Gaussians...", num_gaussians);

    // CPU reference
    let (cpu_d_pos, cpu_d_scale, cpu_d_rot, _) =
        chain_2d_to_3d_gradients_cpu(&grads_2d, &gaussians, &camera);

    // GPU test
    let gpu_renderer = GpuRenderer::new().expect("Failed to initialize GPU");
    let (gpu_d_pos, gpu_d_scale, gpu_d_rot, _) =
        gpu_renderer.project_gradients_backward(&gaussians, &camera, &grads_2d);

    println!("\nResults:");
    for i in 0..num_gaussians {
        let cpu_pos_norm = cpu_d_pos[i].norm();
        let gpu_pos_norm = gpu_d_pos[i].norm();
        let cpu_scale_norm = cpu_d_scale[i].norm();
        let gpu_scale_norm = gpu_d_scale[i].norm();
        let cpu_rot_norm = cpu_d_rot[i].norm();
        let gpu_rot_norm = gpu_d_rot[i].norm();

        println!("\nGaussian {}:", i);
        println!("  Position gradient: CPU={:.6}  GPU={:.6}  ratio={:.2}",
            cpu_pos_norm, gpu_pos_norm,
            if gpu_pos_norm > 1e-9 { cpu_pos_norm / gpu_pos_norm } else { f32::INFINITY });
        println!("  Scale gradient:    CPU={:.6}  GPU={:.6}  ratio={:.2}",
            cpu_scale_norm, gpu_scale_norm,
            if gpu_scale_norm > 1e-9 { cpu_scale_norm / gpu_scale_norm } else { f32::INFINITY });
        println!("  Rotation gradient: CPU={:.6}  GPU={:.6}  ratio={:.2}",
            cpu_rot_norm, gpu_rot_norm,
            if gpu_rot_norm > 1e-9 { cpu_rot_norm / gpu_rot_norm } else { f32::INFINITY });

        if gpu_pos_norm < 1e-9 && cpu_pos_norm > 1e-6 {
            println!("  ❌ GPU position gradient is ZERO!");
        }
        if gpu_scale_norm < 1e-9 && cpu_scale_norm > 1e-6 {
            println!("  ❌ GPU scale gradient is ZERO!");
        }
        if gpu_rot_norm < 1e-9 && cpu_rot_norm > 1e-6 {
            println!("  ❌ GPU rotation gradient is ZERO!");
        }
    }

    // Count failures
    let mut zero_pos_count = 0;
    let mut zero_scale_count = 0;
    let mut zero_rot_count = 0;

    for i in 0..num_gaussians {
        if gpu_d_pos[i].norm() < 1e-9 && cpu_d_pos[i].norm() > 1e-6 {
            zero_pos_count += 1;
        }
        if gpu_d_scale[i].norm() < 1e-9 && cpu_d_scale[i].norm() > 1e-6 {
            zero_scale_count += 1;
        }
        if gpu_d_rot[i].norm() < 1e-9 && cpu_d_rot[i].norm() > 1e-6 {
            zero_rot_count += 1;
        }
    }

    println!("\n--- Summary ---");
    println!("Gaussians with zero gradients (GPU vs non-zero CPU):");
    println!("  Position: {} / {}", zero_pos_count, num_gaussians);
    println!("  Scale:    {} / {}", zero_scale_count, num_gaussians);
    println!("  Rotation: {} / {}", zero_rot_count, num_gaussians);

    if zero_pos_count > 0 || zero_scale_count > 0 || zero_rot_count > 0 {
        println!("\n❌ GPU shader has zero gradient bug!");
        panic!("GPU projection backward produces zero gradients");
    } else {
        println!("\n✅ All gradients are non-zero!");
    }
}
