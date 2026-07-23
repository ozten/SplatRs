//! Tile-binning Stage 3 gate (docs/TILE_RASTER_PLAN.md Part B): the tile-binned
//! rasterizer must match the naive oracle on the SAME device with the SAME projected
//! data. Both paths run identical WGSL math per pixel; the only difference is iteration
//! set/order (this tile's depth-sorted range vs the global depth-sorted array), so the
//! gate is far tighter than CPU-vs-GPU parity: max <= 1e-5, mean <= 1e-6.
//! (Documented fallback per the plan, only if measured drift justifies it: 1e-4 / 1e-5.)
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/golden_io.rs"]
mod golden_io;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};
use golden_io::compare;
use sugar_rs::gpu::RenderOptions;

const MEAN_TOL: f32 = 1e-6;
const MAX_TOL: f32 = 1e-5;

fn check_parity(
    name: &str,
    scene: (
        Vec<sugar_rs::core::Gaussian>,
        sugar_rs::core::Camera,
        nalgebra::Vector3<f32>,
    ),
) {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, bg) = scene;
    let oracle = gpu.render(&gaussians, &camera, &bg).expect("oracle render");
    let tiled = gpu
        .render_with_options(
            &gaussians,
            &camera,
            &bg,
            RenderOptions {
                tile_rasterizer: true,
                ..Default::default()
            },
        )
        .expect("tiled render");
    assert_eq!(oracle.len(), tiled.len());
    let (mean_abs, max_abs) = compare(&oracle, &tiled);
    eprintln!("[{name}] tiled-vs-oracle: mean_abs={mean_abs:.9} max_abs={max_abs:.9}");
    assert!(
        mean_abs < MEAN_TOL && max_abs < MAX_TOL,
        "[{name}] tiled rasterizer diverges from oracle: mean {mean_abs:.9} (tol {MEAN_TOL}), max {max_abs:.9} (tol {MAX_TOL})"
    );
}

#[test]
fn tile_raster_parity_smoke() {
    check_parity("smoke", smoke_scene());
}

#[test]
fn tile_raster_parity_regression() {
    check_parity("regression", regression_scene());
}
