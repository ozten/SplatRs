//! GPU vs CPU gradient parity on a DEEP stack of overlapping Gaussians.
//!
//! Regression test for the per-pixel contribution cap removed 2026-07-08: the old GPU
//! backward recorded only the 16 front-most contributors per pixel, so every Gaussian
//! at depth rank > 16 received exactly zero gradient (measured on a converged 60k model:
//! 88% of the population was gradient-starved). The rewritten backward re-walks the
//! sorted list per pixel, so ALL contributors must receive gradients matching the
//! (uncapped) CPU backward.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_color_grads;

#[cfg(feature = "gpu")]
use sugar_rs::gpu::GpuRenderer;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_gradients_reach_beyond_16_contributors() {
    const N: usize = 40; // 2.5x the old 16-slot cap

    let camera = Camera::new(
        4.0,
        4.0,
        3.5,
        3.5,
        8,
        8,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    // N Gaussians stacked along +z on the optical axis, sigma_px 1..2 (3-sigma radius
    // stays under the projection screen-size cull on both CPU and GPU). Opacity 0.008
    // keeps the alpha >= 1e-4 footprint (2.96 sigma) INSIDE the CPU's 3-sigma bbox, so
    // the CPU and GPU contributor sets coincide exactly and gradients are directly
    // comparable. No early termination: T_final ~ 0.75 after all 40 layers, so every
    // layer keeps contributing at the central pixels.
    let gaussians: Vec<Gaussian> = (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            Gaussian::new(
                Vector3::new(0.0, 0.0, 2.0 + 0.05 * i as f32),
                Vector3::new(0.0, 0.0, 0.0), // sigma_world = 1.0
                UnitQuaternion::identity(),
                -4.82, // sigmoid ~= 0.008
                sh_constant_color(Vector3::new(0.2 + 0.6 * t, 0.8 - 0.6 * t, 0.3)),
            )
        })
        .collect();

    let bg = Vector3::new(0.02, 0.03, 0.04);
    let num_pixels = (camera.width * camera.height) as usize;
    let d_pixels = vec![Vector3::new(1.0, -0.5, 0.25); num_pixels];

    let (_cpu_img, cpu_d_colors, cpu_d_opacity, _cpu_d_pos, _cpu_d_scale, _cpu_d_rot, cpu_d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_pixels, &bg, false);

    let (gpu_img, gpu_grads) = GpuRenderer::new()
        .expect("Failed to initialize GPU")
        .render_with_gradients(&gaussians, &camera, &bg, &d_pixels)
        .expect("GPU render_with_gradients failed");

    // Forward parity gate: if the images already diverge, the gradient comparison is
    // meaningless (projection difference, not a backward bug).
    let cpu_linear =
        sugar_rs::render::render_full_linear(&gaussians, &camera, &bg, false);
    let mut max_px_diff = 0.0f32;
    for (c, g) in cpu_linear.iter().zip(gpu_img.iter()) {
        max_px_diff = max_px_diff.max((c - g).abs().max());
    }
    println!("forward max pixel diff: {max_px_diff:.6}");
    assert!(
        max_px_diff < 1e-3,
        "forward CPU/GPU images diverge ({max_px_diff}) — contributor sets don't match; \
         gradient comparison would be meaningless"
    );

    // (1) EVERY Gaussian must receive a nonzero opacity gradient -- under the old
    // 16-slot cap, indices 16..40 got exactly zero.
    let mut starved = Vec::new();
    for i in 0..N {
        if gpu_grads.d_opacity_logits[i].abs() < 1e-7 {
            starved.push(i);
        }
    }
    assert!(
        starved.is_empty(),
        "Gaussians with zero opacity gradient (depth-starved): {:?}",
        starved
    );

    // (2) GPU gradients must match the uncapped CPU backward for ALL depth ranks.
    // Tolerance: 5% relative + absolute floor. The residual difference is contributor-set
    // noise at the hard alpha >= 1e-4 cutoff — pixels on the cutoff circle (~2πr of them)
    // flip inclusion between CPU and GPU from float-level covariance differences, each
    // worth ~T·1e-4 of gradient. Genuine backward bugs (starvation, wrong T chain) show
    // up as zero or order-of-magnitude errors, not few-percent ones.
    let rel_ok = |cpu: f32, gpu: f32| -> bool {
        let diff = (cpu - gpu).abs();
        diff <= 0.05 * cpu.abs().max(gpu.abs()) + 2e-3
    };
    for i in 0..N {
        for c in 0..3 {
            assert!(
                rel_ok(cpu_d_colors[i][c], gpu_grads.d_colors[i][c]),
                "d_color mismatch at gaussian {} ch {}: cpu={} gpu={}",
                i,
                c,
                cpu_d_colors[i][c],
                gpu_grads.d_colors[i][c]
            );
        }
        assert!(
            rel_ok(cpu_d_opacity[i], gpu_grads.d_opacity_logits[i]),
            "d_opacity mismatch at gaussian {}: cpu={} gpu={}",
            i,
            cpu_d_opacity[i],
            gpu_grads.d_opacity_logits[i]
        );
    }

    // (3) Background gradient must use the TRUE final transmittance (the old backward
    // recomputed it from the <=16 recorded contributions, overestimating it badly on
    // deep pixels).
    for c in 0..3 {
        assert!(
            rel_ok(cpu_d_bg[c], gpu_grads.d_background[c]),
            "d_background mismatch ch {}: cpu={} gpu={}",
            c,
            cpu_d_bg[c],
            gpu_grads.d_background[c]
        );
    }
}
