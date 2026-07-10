//! GPU depth-sort correctness at realistic population sizes.
//!
//! The old `unit_gpu_depth_ordering` test used N=2, which the bitonic network sorts
//! correctly even with a wrong direction bit; this test stacks hundreds of translucent
//! Gaussians (non-power-of-two count, shuffled input order) on one ray so the composited
//! color is dominated by ordering, then requires the GPU forward to match the CPU forward
//! (which sorts with std::sort). Catches both historical sort bugs: the wrong
//! direction bit (bit `stage` instead of `stage+1`) and the skipped compare-exchanges
//! for non-power-of-two counts.

#![cfg(feature = "gpu")]

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::gpu::GpuRenderer;
use sugar_rs::render::full_diff::render_full_linear;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

/// Render an n-Gaussian shuffled depth stack through CPU and GPU, return mean abs diff.
/// Colors are keyed to depth rank so any mis-ordering shifts the composited color.
fn stack_divergence(n: usize) -> f32 {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    order.shuffle(&mut rng);

    let gaussians: Vec<Gaussian> = order
        .iter()
        .map(|&k| {
            let t = k as f32 / (n - 1) as f32;
            let z = 2.0 + 48.0 * t;
            let color = Vector3::new(t, 1.0 - t, 0.5);
            Gaussian::new(
                Vector3::new(0.0, 0.0, z),
                Vector3::new(-3.0, -3.0, -3.0),
                UnitQuaternion::identity(),
                0.0,
                sh_constant_color(color),
            )
        })
        .collect();

    let camera = Camera::new(
        100.0,
        100.0,
        8.0,
        8.0,
        16,
        16,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let bg = Vector3::new(0.1, 0.2, 0.3);

    let cpu = render_full_linear(&gaussians, &camera, &bg, false);
    let gpu = GpuRenderer::new()
        .expect("GPU init")
        .render(&gaussians, &camera, &bg)
        .expect("GPU render");

    cpu.iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs().sum())
        .sum::<f32>()
        / (3.0 * cpu.len() as f32)
}

/// The 2026-07-10 high-count regression: GPU rendering diverged at counts that never
/// occurred under the old 60k cap. Bisection on the watchdog-abort model showed clean
/// output at 60k/100k but corruption at count == padded (65,536) and at every count
/// padding to 262,144 (>= 131,073). Guard all three regimes.
#[test]
fn test_gpu_sort_high_count_regimes() {
    for (n, label) in [
        (100_000usize, "non-power count below 2^17 (control)"),
        (65_536, "count exactly a power of two"),
        (160_000, "count padding to 2^18"),
    ] {
        let mean_abs = stack_divergence(n);
        assert!(
            mean_abs < 2e-3,
            "GPU diverges from CPU at n={n} ({label}): mean abs diff {mean_abs}"
        );
    }
}

#[test]
fn test_gpu_sort_matches_cpu_on_deep_shuffled_stack() {
    // 300 (not a power of two) translucent Gaussians stacked along the optical axis.
    let n = 300usize;
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    order.shuffle(&mut rng);

    let gaussians: Vec<Gaussian> = order
        .iter()
        .map(|&k| {
            let t = k as f32 / (n - 1) as f32;
            let z = 2.0 + 48.0 * t; // depths 2..50, distinct
            // Color keyed to depth rank so any mis-ordering shifts the blend visibly.
            let color = Vector3::new(t, 1.0 - t, 0.5);
            Gaussian::new(
                Vector3::new(0.0, 0.0, z),
                Vector3::new(-3.0, -3.0, -3.0), // sigma ~0.05 world units
                UnitQuaternion::identity(),
                0.0, // sigmoid(0) = 0.5 opacity
                sh_constant_color(color),
            )
        })
        .collect();

    let camera = Camera::new(
        100.0,
        100.0,
        16.0,
        16.0,
        32,
        32,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let bg = Vector3::new(0.1, 0.2, 0.3);

    let cpu = render_full_linear(&gaussians, &camera, &bg, false);
    let gpu = GpuRenderer::new()
        .expect("GPU init")
        .render(&gaussians, &camera, &bg)
        .expect("GPU render");

    let center = (16 * 32 + 16) as usize;
    let d_center = (cpu[center] - gpu[center]).abs();
    let mean_abs: f32 = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs().sum())
        .sum::<f32>()
        / (3.0 * cpu.len() as f32);

    assert!(
        d_center.max() < 2e-3,
        "center pixel diverges: cpu={:?} gpu={:?} (depth ordering broken?)",
        cpu[center],
        gpu[center]
    );
    assert!(
        mean_abs < 2e-3,
        "mean abs CPU/GPU divergence too high: {mean_abs}"
    );
}
