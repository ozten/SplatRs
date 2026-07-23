//! Tile-binning Stage 1 gate (docs/TILE_RASTER_PLAN.md Part B): the GPU tile-touch
//! counting kernel must agree EXACTLY (integer-for-integer) with the CPU oracle
//! `render::tile_math`, evaluated on the very projected values the GPU consumed
//! (read back from the projection buffer, so both sides see identical f32 bits).
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};
use sugar_rs::render::tile_math::{tile_grid_dims, tile_touch_count};

fn assert_counts_match(name: &str, scene: (Vec<sugar_rs::core::Gaussian>, sugar_rs::core::Camera, nalgebra::Vector3<f32>)) {
    let Some(gpu) = gpu_skip::try_gpu_renderer() else {
        return;
    };
    let (gaussians, camera, _bg) = scene;
    let (tiles_x, tiles_y) = tile_grid_dims(camera.width, camera.height);
    let (projected, counts) = gpu
        .debug_tile_touch_counts(&gaussians, &camera)
        .expect("debug_tile_touch_counts");
    assert_eq!(projected.len(), gaussians.len());
    assert_eq!(counts.len(), gaussians.len());

    let mut mismatches = 0usize;
    let mut live = 0usize;
    let mut multi_tile = 0usize;
    for (i, (p, &got)) in projected.iter().zip(counts.iter()).enumerate() {
        let radius = p.cov[3];
        let expected = tile_touch_count(p.mean[0], p.mean[1], radius, tiles_x, tiles_y);
        if got != expected {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "[{name}] MISMATCH i={i}: mean=({},{}) r={} gpu={got} cpu={expected}",
                    p.mean[0], p.mean[1], radius
                );
            }
        }
        if expected > 0 {
            live += 1;
        }
        if expected > 1 {
            multi_tile += 1;
        }
    }
    assert_eq!(mismatches, 0, "[{name}] GPU/CPU tile-count mismatches");
    // The gate is only meaningful if the scene actually exercises the interesting cases.
    assert!(live > 0, "[{name}] no live gaussians — fixture broken");
    assert!(
        multi_tile > 0,
        "[{name}] no gaussian touches >1 tile — fixture too easy for the boundary math"
    );
    eprintln!(
        "[{name}] {} gaussians: {} live, {} multi-tile, 0 mismatches (grid {}x{})",
        gaussians.len(),
        live,
        multi_tile,
        tiles_x,
        tiles_y
    );
}

#[test]
fn gpu_tile_counts_match_cpu_oracle_smoke() {
    assert_counts_match("smoke", smoke_scene());
}

#[test]
fn gpu_tile_counts_match_cpu_oracle_regression() {
    assert_counts_match("regression", regression_scene());
}
