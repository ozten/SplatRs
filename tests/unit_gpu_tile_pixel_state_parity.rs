//! Tile-binning Stage 5a gate (docs/TILE_RASTER_PLAN.md Part B): the tiled forward pass's
//! per-pixel lifecycle state (final transmittance + identity of the last blended
//! contributor) must resolve to the SAME transmittance and the SAME ORIGINAL Gaussian
//! identity as the naive oracle's forward pass — even though the two paths track that
//! identity through different sorted artifacts:
//!   - naive: `pixel_state.y` is a SORTED index into the globally depth-sorted
//!     `Gaussian2DGPU` buffer; resolve via `sorted[y].gaussian_idx_pad[0]`.
//!   - tiled: `pixel_state.y` is an ABSOLUTE index into the global (tile, depth)-sorted
//!     `pairs` buffer; resolve via `pairs[y].gaussian_idx` (the tile path never reorders
//!     `projected`, so `pairs[y].gaussian_idx` is already an original Gaussian index).
//!
//! Both `0xFFFFFFFF` sentinels must agree (no contributor blended in either path), and
//! where both are live, the resolved original indices must match exactly (integer
//! comparison — no tolerance). Transmittance is compared with a tight float tolerance
//! since it's the same accumulation math, just walked in tile-batched order.
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};

const T_TOL: f32 = 1e-5;
const SENTINEL: u32 = 0xFFFFFFFF;

fn check_pixel_state(
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

    let (naive_pixels, naive_state, sorted) = gpu
        .debug_render_naive_pixel_state(&gaussians, &camera, &bg)
        .expect("naive pixel state");
    let (tiled_pixels, tiled_state, pairs) = gpu
        .debug_render_tiled_pixel_state(&gaussians, &camera, &bg)
        .expect("tiled pixel state");

    assert_eq!(naive_state.len(), tiled_state.len(), "[{name}] pixel_state length mismatch");
    assert_eq!(naive_pixels.len(), tiled_pixels.len(), "[{name}] pixel length mismatch");

    let mut checked = 0usize;
    let mut both_live = 0usize;
    let mut max_t_diff = 0.0f32;
    let mut sentinel_mismatches = 0usize;
    let mut identity_mismatches = 0usize;

    for i in 0..naive_state.len() {
        let n = naive_state[i];
        let t = tiled_state[i];

        // (a) transmittance must match within a tight float tolerance.
        let t_n = f32::from_bits(n[0]);
        let t_t = f32::from_bits(t[0]);
        let d = (t_n - t_t).abs();
        if d > max_t_diff {
            max_t_diff = d;
        }
        assert!(
            d < T_TOL,
            "[{name}] pixel {i}: transmittance mismatch naive={t_n} tiled={t_t} diff={d} (tol {T_TOL})"
        );

        // (b) sentinel iff sentinel.
        let n_sentinel = n[1] == SENTINEL;
        let t_sentinel = t[1] == SENTINEL;
        if n_sentinel != t_sentinel {
            sentinel_mismatches += 1;
            if sentinel_mismatches <= 5 {
                eprintln!(
                    "[{name}] pixel {i}: sentinel mismatch naive_sentinel={n_sentinel} (y={}) tiled_sentinel={t_sentinel} (y={})",
                    n[1], t[1]
                );
            }
        }

        // (c) when both non-sentinel: resolved ORIGINAL Gaussian identity must match
        // exactly. naive.y is a sorted-buffer index; tiled.y is a pairs-buffer index.
        if !n_sentinel && !t_sentinel {
            both_live += 1;
            let naive_gidx = sorted[n[1] as usize].gaussian_idx_pad[0];
            let tiled_gidx = pairs[t[1] as usize].gaussian_idx;
            if naive_gidx != tiled_gidx {
                identity_mismatches += 1;
                if identity_mismatches <= 5 {
                    eprintln!(
                        "[{name}] pixel {i}: identity mismatch naive_gidx={naive_gidx} tiled_gidx={tiled_gidx} (naive.y={}, tiled.y={})",
                        n[1], t[1]
                    );
                }
            }
        }
        checked += 1;
    }

    assert_eq!(
        sentinel_mismatches, 0,
        "[{name}] {sentinel_mismatches} pixel(s) disagree on whether any contributor blended"
    );
    assert_eq!(
        identity_mismatches, 0,
        "[{name}] {identity_mismatches} pixel(s) resolve to a different original Gaussian"
    );
    assert!(
        both_live > 0,
        "[{name}] no pixel had a live (non-sentinel) contributor in both paths — fixture too sparse to gate anything"
    );

    // Diagnostic only (the tight forward-pixel parity gate lives in
    // unit_gpu_tile_raster_parity.rs): both paths' resolved colors should agree too.
    let mut color_max_diff = 0.0f32;
    for (a, b) in naive_pixels.iter().zip(tiled_pixels.iter()) {
        let d = (a - b).abs();
        color_max_diff = color_max_diff.max(d.x.max(d.y).max(d.z));
    }

    eprintln!(
        "[{name}] pixel-state parity: {checked} pixels checked, {both_live} with a live contributor in both paths, max T diff={max_t_diff:.9}, max color diff={color_max_diff:.9}"
    );
}

#[test]
fn tile_pixel_state_parity_smoke() {
    check_pixel_state("smoke", smoke_scene());
}

#[test]
fn tile_pixel_state_parity_regression() {
    check_pixel_state("regression", regression_scene());
}
