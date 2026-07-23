//! Determinism harness (docs/TILE_RASTER_PLAN.md Part A, step 1 — run first; the whole
//! CPU/GPU tolerance scheme in golden_cpu_gpu_parity.rs and golden_psnr_floor.rs depends
//! on the outcome here).
//!
//! CPU forward (`render_full_linear`, src/render/full_diff.rs:509) is sequential with a
//! stable sort — bit-exact repeatable expected and asserted with `==` below.
//!
//! GPU forward has no atomics (only the backward pass uses fixed-point atomicAdd) — also
//! expected bit-exact repeatable. Asserted with `==` first; if driver/hardware ever
//! breaks that, fall back to the 1e-6 tolerance already proven safe on this hardware by
//! tests/gpu_deterministic_rendering.rs (`test_deterministic_rendering`), and update the
//! doc comment on that test with the measured drift.

#[path = "golden/fixtures.rs"]
mod fixtures;

#[cfg(feature = "gpu")]
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};
use sugar_rs::render::full_diff::render_full_linear;

#[test]
fn test_cpu_render_bit_exact_repeatable_smoke() {
    let (gaussians, camera, bg) = smoke_scene();
    let a = render_full_linear(&gaussians, &camera, &bg, false);
    let b = render_full_linear(&gaussians, &camera, &bg, false);
    assert_eq!(a, b, "CPU render is not bit-exact repeatable on smoke_scene");
}

#[test]
fn test_cpu_render_bit_exact_repeatable_regression() {
    let (gaussians, camera, bg) = regression_scene();
    let a = render_full_linear(&gaussians, &camera, &bg, false);
    let b = render_full_linear(&gaussians, &camera, &bg, false);
    assert_eq!(
        a, b,
        "CPU render is not bit-exact repeatable on regression_scene"
    );
}

/// GPU render twice on the same (small) scene. Kept to smoke_scene only — this is the
/// determinism *gate*, not a full parity sweep, and GPU test executions are being kept
/// to a minimum while a training run shares the device.
///
/// MEASURED 2026-07-23 on Apple M2 Max (Metal): max abs diff across 10 repeat renders
/// was exactly 0.0 — bit-exact, confirming the "no atomics in forward" prediction. So
/// this asserts exact `==` rather than the 1e-6 fallback. If a different driver/adapter
/// ever breaks that, switch the assertion to `max_diff <= 1e-6` (same tolerance already
/// proven safe by tests/gpu_deterministic_rendering.rs::test_deterministic_rendering)
/// and record the measured drift + hardware here.
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_render_bit_exact_repeatable_smoke() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, bg) = smoke_scene();
    let a = gpu.render(&gaussians, &camera, &bg).expect("GPU render 1");
    let b = gpu.render(&gaussians, &camera, &bg).expect("GPU render 2");
    assert_eq!(
        a, b,
        "GPU render is not bit-exact repeatable on smoke_scene (see doc comment for the documented 1e-6 fallback)"
    );
}
