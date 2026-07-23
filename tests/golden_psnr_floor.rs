//! PSNR floor regression test against a committed golden image
//! (docs/TILE_RASTER_PLAN.md Part A, step 4).
//!
//! The golden is `tests/golden/goldens/regression_scene_cpu.png` — a 16-bit PNG storing
//! raw linear [0,1] pixels (see tests/golden/golden_io.rs doc comment) from
//! `render_full_linear` on `regression_scene()`. CPU render is bit-exact repeatable
//! (tests/golden_determinism.rs), so CPU-vs-golden drift measured here means the
//! renderer itself changed, not golden-generation instability.
//!
//! Regenerate ONLY via:
//!   `SUGAR_REGENERATE_GOLDEN=1 cargo test --features gpu --test golden_psnr_floor -- --test-threads=1`
//! (`--test-threads=1` avoids a race between the CPU test writing the golden and the GPU
//! test reading it — only relevant during regeneration; normal verification runs don't
//! write and are safe in parallel.)
//!
//! `compute_psnr` is promoted `pub` in src/optim/trainer.rs for this file's use — the one
//! non-test source change made for Part A.
//!
//! MEASURED 2026-07-23 on Apple M2 Max (Metal), regression_scene (20k gaussians, 490x273):
//!   CPU vs golden: mean_abs=0.000004 max_abs=0.000008 PSNR=100.00 dB
//!   GPU vs golden: PSNR=100.00 dB
//! `compute_psnr` caps at 100.0 dB whenever MSE < 1e-10 (src/optim/trainer.rs), so both
//! numbers above are display-capped, not literal — the naive renderer round-trips through
//! the 16-bit-linear golden essentially losslessly on both backends. A "measured minus
//! ~5 dB" floor would be meaningless against a capped value, so instead the floor is set
//! to a conservative 40 dB (the plan's own illustrative number): comfortably below
//! anything seen here, but tight enough to catch a genuinely broken renderer (published
//! Gaussian-splatting PSNRs run ~25-35 dB for *good* reconstructions; a regression this
//! test should catch — e.g. a bad Part-B tile-rasterizer substitution — would fall well
//! below that). Also generous enough to tolerate the plan's documented 1e-4 tile-oracle
//! parity fallback (abs diff 1e-4 -> MSE ~1e-8 -> ~80 dB) without flaking.

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/golden_io.rs"]
mod golden_io;
#[cfg(feature = "gpu")]
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::regression_scene;
use golden_io::{compare, load_golden, save_golden};
use std::path::PathBuf;
use sugar_rs::optim::trainer::compute_psnr;
use sugar_rs::render::full_diff::render_full_linear;

/// floor = conservative fixed value, well below the measured (display-capped) 100 dB —
/// see the MEASURED note above for why "measured minus margin" doesn't apply literally.
const PSNR_FLOOR_DB: f32 = 40.0;

/// CPU-vs-golden absolute-diff tolerance: absorbs 16-bit quantization only
/// (~1/65535 ~= 1.5e-5 per channel).
const CPU_MEAN_TOL: f32 = 5e-4;
const CPU_MAX_TOL: f32 = 2e-3;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/goldens/regression_scene_cpu.png")
}

#[test]
fn test_cpu_psnr_floor_regression() {
    let (gaussians, camera, bg) = regression_scene();
    let cpu = render_full_linear(&gaussians, &camera, &bg, false);
    let path = golden_path();

    if std::env::var("SUGAR_REGENERATE_GOLDEN").is_ok() {
        save_golden(&path, &cpu, camera.width, camera.height);
        eprintln!("Regenerated golden at {}", path.display());
    }

    let (golden, gw, gh) = load_golden(&path);
    assert_eq!(
        (gw, gh),
        (camera.width, camera.height),
        "golden dimensions mismatch — regenerate with SUGAR_REGENERATE_GOLDEN=1"
    );

    let (mean_abs, max_abs) = compare(&cpu, &golden);
    let psnr = compute_psnr(&cpu, &golden);
    eprintln!("CPU-vs-golden: mean_abs={mean_abs:.6} max_abs={max_abs:.6} psnr={psnr:.2}dB");

    assert!(
        mean_abs < CPU_MEAN_TOL,
        "CPU-vs-golden mean abs diff {mean_abs} >= {CPU_MEAN_TOL}"
    );
    assert!(
        max_abs < CPU_MAX_TOL,
        "CPU-vs-golden max abs diff {max_abs} >= {CPU_MAX_TOL}"
    );
    assert!(
        psnr >= PSNR_FLOOR_DB,
        "CPU-vs-golden PSNR {psnr:.2}dB below floor {PSNR_FLOOR_DB}dB"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_psnr_floor_regression() {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, bg) = regression_scene();
    let gpu_out = gpu.render(&gaussians, &camera, &bg).expect("GPU render");
    let path = golden_path();
    let (golden, gw, gh) = load_golden(&path);
    assert_eq!(
        (gw, gh),
        (camera.width, camera.height),
        "golden dimensions mismatch — regenerate with SUGAR_REGENERATE_GOLDEN=1"
    );

    let psnr = compute_psnr(&gpu_out, &golden);
    eprintln!("GPU-vs-golden PSNR: {psnr:.2}dB");
    assert!(
        psnr >= PSNR_FLOOR_DB,
        "GPU-vs-golden PSNR {psnr:.2}dB below floor {PSNR_FLOOR_DB}dB"
    );
}
