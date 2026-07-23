//! CPU vs GPU render parity (docs/TILE_RASTER_PLAN.md Part A, step 3).
//!
//! Both backends are bit-exact repeatable (tests/golden_determinism.rs), so any drift
//! measured here reflects genuine CPU/GPU numerical differences (fp32 accumulation
//! order, etc.), not nondeterminism.
//!
//! Tolerances are reused, not invented (per docs/TILE_RASTER_PLAN.md "Ground truth"):
//! mean < 2e-3 matches tests/unit_gpu_sort_order.rs; max < 1e-2 matches
//! tests/unit_gpu_render_smoke.rs.
//!
//! Whole-file gpu gate follows the tests/unit_gpu_sort_order.rs convention: every test
//! here is inherently a GPU-vs-CPU comparison, so there is no CPU-only variant.
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/golden_io.rs"]
mod golden_io;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};
use golden_io::compare;
use sugar_rs::render::full_diff::render_full_linear;

const MEAN_TOL: f32 = 2e-3;
const MAX_TOL: f32 = 1e-2;

#[test]
fn test_cpu_gpu_parity_smoke() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, bg) = smoke_scene();
    let cpu = render_full_linear(&gaussians, &camera, &bg, false);
    let gpu_out = gpu.render(&gaussians, &camera, &bg).expect("GPU render");
    assert_eq!(cpu.len(), gpu_out.len());

    let (mean_abs, max_abs) = compare(&cpu, &gpu_out);
    eprintln!("smoke_scene CPU-vs-GPU: mean_abs={mean_abs:.6} max_abs={max_abs:.6}");
    assert!(
        mean_abs < MEAN_TOL,
        "smoke_scene CPU-vs-GPU mean abs diff {mean_abs} >= {MEAN_TOL}"
    );
    assert!(
        max_abs < MAX_TOL,
        "smoke_scene CPU-vs-GPU max abs diff {max_abs} >= {MAX_TOL}"
    );
}

#[test]
fn test_cpu_gpu_parity_regression() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, bg) = regression_scene();
    let cpu = render_full_linear(&gaussians, &camera, &bg, false);
    let gpu_out = gpu.render(&gaussians, &camera, &bg).expect("GPU render");
    assert_eq!(cpu.len(), gpu_out.len());

    let (mean_abs, max_abs) = compare(&cpu, &gpu_out);
    eprintln!("regression_scene CPU-vs-GPU: mean_abs={mean_abs:.6} max_abs={max_abs:.6}");
    assert!(
        mean_abs < MEAN_TOL,
        "regression_scene CPU-vs-GPU mean abs diff {mean_abs} >= {MEAN_TOL}"
    );
    assert!(
        max_abs < MAX_TOL,
        "regression_scene CPU-vs-GPU max abs diff {max_abs} >= {MAX_TOL}"
    );
}
