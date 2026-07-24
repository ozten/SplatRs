//! Shared scene fixtures for the golden parity/regression harness (docs/TILE_RASTER_PLAN.md
//! Part A). Procedurally generated (seeded `StdRng`) so the harness works from a fresh
//! clone without depending on the gitignored `datasets/` directory.
//!
//! Pulled into `tests/golden_*.rs` test binaries via `#[path]` mod — the standard
//! `tests/common` idiom (files under a `tests/` subdirectory are not auto-discovered by
//! Cargo as their own test targets, only compiled when explicitly `mod`-included).
//!
//! Idioms (seeding, SH-constant-color helper, `Camera::new`) follow
//! tests/unit_gpu_sort_order.rs and tests/synthetic_scene_tests.rs.

#![allow(dead_code)]

use nalgebra::{Matrix3, Unit, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sugar_rs::core::{Camera, Gaussian};

/// SH DC coefficient normalization constant: Y_0^0 = 1/(2*sqrt(pi)).
pub const SH_C0: f32 = 0.282_094_791_773_878_14;

/// Create SH coefficients for a constant (view-independent) color.
pub fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

/// Small smoke scene: ~2k gaussians @ 128x96. Cheap enough to run repeatedly, including
/// inside GPU-gated tests, without materially slowing the suite.
///
/// Returns (gaussians, camera, background).
pub fn smoke_scene() -> (Vec<Gaussian>, Camera, Vector3<f32>) {
    let n = 2_000usize;
    let width = 128u32;
    let height = 96u32;
    let mut rng = StdRng::seed_from_u64(20260723);

    let gaussians: Vec<Gaussian> = (0..n)
        .map(|_| {
            let position = Vector3::new(
                rng.gen_range(-3.0..3.0),
                rng.gen_range(-2.2..2.2),
                rng.gen_range(2.0..40.0),
            );
            let log_scale = Vector3::new(
                rng.gen_range(-4.5..-2.5),
                rng.gen_range(-4.5..-2.5),
                rng.gen_range(-4.5..-2.5),
            );
            let opacity_logit: f32 = rng.gen_range(-1.0..3.0);
            let color = Vector3::new(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            );
            Gaussian::new(
                position,
                log_scale,
                UnitQuaternion::identity(),
                opacity_logit,
                sh_constant_color(color),
            )
        })
        .collect();

    let camera = Camera::new(
        150.0,
        150.0,
        (width as f32) / 2.0,
        (height as f32) / 2.0,
        width,
        height,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let bg = Vector3::new(0.1, 0.1, 0.15);
    (gaussians, camera, bg)
}

/// Regression scene: ~20k gaussians with mixed anisotropy/opacity scattered in x/y/depth
/// @ 490x273 (canonical half-res per docs/TILE_RASTER_PLAN.md). One third of the
/// gaussians are near-isotropic (small, roughly round splats); the rest are strongly
/// anisotropic on a random axis (needle-like splats), exercising the covariance/rotation
/// path. Opacity spans near-transparent to near-opaque.
///
/// Returns (gaussians, camera, background).
pub fn regression_scene() -> (Vec<Gaussian>, Camera, Vector3<f32>) {
    let n = 20_000usize;
    let width = 490u32;
    let height = 273u32;
    let mut rng = StdRng::seed_from_u64(20260723);

    let gaussians: Vec<Gaussian> = (0..n)
        .map(|i| {
            let position = Vector3::new(
                rng.gen_range(-6.0..6.0),
                rng.gen_range(-3.4..3.4),
                rng.gen_range(2.0..60.0),
            );

            // Mix isotropic (1/3) and strongly anisotropic (2/3) splats.
            let log_scale = if i % 3 == 0 {
                Vector3::new(
                    rng.gen_range(-5.0..-3.5),
                    rng.gen_range(-5.0..-3.5),
                    rng.gen_range(-5.0..-3.5),
                )
            } else {
                Vector3::new(
                    rng.gen_range(-3.0..-1.5),
                    rng.gen_range(-5.0..-4.0),
                    rng.gen_range(-5.0..-4.0),
                )
            };

            let axis_raw = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let axis = if axis_raw.norm() > 1e-6 {
                Unit::new_normalize(axis_raw)
            } else {
                Vector3::z_axis()
            };
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::PI);
            let rotation = UnitQuaternion::from_axis_angle(&axis, angle);

            // Spans near-transparent (sigmoid(-3) ~= 0.05) to near-opaque (sigmoid(4) ~= 0.98).
            let opacity_logit: f32 = rng.gen_range(-3.0..4.0);
            let color = Vector3::new(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            );

            Gaussian::new(
                position,
                log_scale,
                rotation,
                opacity_logit,
                sh_constant_color(color),
            )
        })
        .collect();

    let camera = Camera::new(
        400.0,
        400.0,
        (width as f32) / 2.0,
        (height as f32) / 2.0,
        width,
        height,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let bg = Vector3::new(0.12, 0.14, 0.18);
    (gaussians, camera, bg)
}

/// Deep stack of N=40 overlapping Gaussians on the optical axis (factored out of
/// tests/unit_gpu_deep_blend_gradients.rs so tests/unit_gpu_tile_backward_gradients.rs's
/// Stage 5b naive-vs-tiled backward parity gate can reuse it — this is the regression
/// fixture for the per-pixel contribution cap removed 2026-07-08: 2.5x the old 16-slot
/// cap, so it stresses multi-batch back-to-front walks and rank > 16 contributors.
///
/// N Gaussians stacked along +z on the optical axis, sigma_px 1..2 (3-sigma radius stays
/// under the projection screen-size cull on both CPU and GPU). Opacity 0.008 keeps the
/// alpha >= 1e-4 footprint (2.96 sigma) INSIDE the CPU's 3-sigma bbox, so the CPU and GPU
/// contributor sets coincide exactly and gradients are directly comparable. No early
/// termination: T_final ~ 0.75 after all 40 layers, so every layer keeps contributing at
/// the central pixels.
///
/// Returns (gaussians, camera, background).
pub fn deep_stack_scene() -> (Vec<Gaussian>, Camera, Vector3<f32>) {
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
    (gaussians, camera, bg)
}
