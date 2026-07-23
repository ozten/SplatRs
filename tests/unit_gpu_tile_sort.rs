//! Tile-binning Stage 2 gate (docs/TILE_RASTER_PLAN.md Part B): pair emission +
//! (tile, depth) bitonic sort + tile-range boundary detection. All assertions are exact
//! (integers/orderings), no float tolerances:
//!   (a) within every tile's [start, end) range, key_tile matches and depth keys are
//!       non-decreasing (front-to-back compositing order — the property that matters);
//!   (b) pair conservation: every gaussian appears exactly counts[i] times, every pair's
//!       tile lies inside that gaussian's CPU-oracle tile rect, no sentinel leaks into
//!       the real region (the 836f4d0 padding-regression class);
//!   (c) ranges partition the pair array: disjoint, sorted, covering all total_pairs.
#![cfg(feature = "gpu")]

#[path = "golden/fixtures.rs"]
mod fixtures;
#[path = "golden/gpu_skip.rs"]
mod gpu_skip;

use fixtures::{regression_scene, smoke_scene};
use sugar_rs::render::tile_math::{tile_grid_dims, tile_touch_rect};

fn check_scene(
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
    let (gaussians, camera, _bg) = scene;
    let (tiles_x, tiles_y) = tile_grid_dims(camera.width, camera.height);
    let num_tiles = (tiles_x * tiles_y) as usize;

    let (projected, counts, pairs, ranges) = gpu
        .debug_tile_binning(&gaussians, &camera)
        .expect("debug_tile_binning");
    let total_pairs: u32 = counts.iter().sum();
    assert_eq!(pairs.len(), total_pairs as usize, "[{name}] pair count");
    assert_eq!(ranges.len(), num_tiles, "[{name}] ranges length");
    assert!(total_pairs > 0, "[{name}] fixture produced no pairs");
    assert!(
        !total_pairs.is_power_of_two(),
        "[{name}] fixture pair count is a power of two — padding path untested, adjust fixture"
    );

    // (0) global lexicographic sortedness — the property the bitonic network must deliver.
    let mut violations = 0usize;
    for j in 1..pairs.len() {
        let (a, b) = (&pairs[j - 1], &pairs[j]);
        if (a.key_tile, a.key_depth) > (b.key_tile, b.key_depth) {
            violations += 1;
            if violations <= 5 {
                eprintln!(
                    "[{name}] sort violation at {j}: ({},{}) > ({},{})",
                    a.key_tile, a.key_depth, b.key_tile, b.key_depth
                );
            }
        }
    }
    assert_eq!(violations, 0, "[{name}] pair array not sorted");

    // (b) conservation + membership + no sentinel leak.
    let mut per_gaussian = vec![0u32; gaussians.len()];
    for (j, p) in pairs.iter().enumerate() {
        assert!(
            (p.key_tile as usize) < num_tiles,
            "[{name}] pair {j}: sentinel/out-of-range tile {} in real region",
            p.key_tile
        );
        let gi = p.gaussian_idx as usize;
        assert!(gi < gaussians.len(), "[{name}] pair {j}: bad gaussian_idx");
        per_gaussian[gi] += 1;
        let pr = &projected[gi];
        let rect = tile_touch_rect(pr.mean[0], pr.mean[1], pr.cov[3], tiles_x, tiles_y)
            .unwrap_or_else(|| panic!("[{name}] pair {j} from culled gaussian {gi}"));
        let (x0, x1, y0, y1) = rect;
        let (tx, ty) = (p.key_tile % tiles_x, p.key_tile / tiles_x);
        assert!(
            tx >= x0 && tx <= x1 && ty >= y0 && ty <= y1,
            "[{name}] pair {j}: tile ({tx},{ty}) outside gaussian {gi} rect ({x0}..{x1},{y0}..{y1})"
        );
        // Depth key must be the bitcast of the projected depth.
        assert_eq!(
            p.key_depth,
            pr.mean[2].to_bits(),
            "[{name}] pair {j}: depth key mismatch"
        );
    }
    assert_eq!(per_gaussian, counts, "[{name}] per-gaussian pair conservation");

    // Diagnostics: nonzero ranges and the pair array's tile-run structure.
    for (t, r) in ranges.iter().enumerate() {
        if r[0] != 0 || r[1] != 0 {
            eprintln!("[{name}] ranges[{t}] = ({}, {})", r[0], r[1]);
        }
    }
    let mut run_start = 0usize;
    for j in 1..=pairs.len() {
        if j == pairs.len() || pairs[j].key_tile != pairs[run_start].key_tile {
            eprintln!(
                "[{name}] pairs run: tile {} at [{run_start}, {j})",
                pairs[run_start].key_tile
            );
            run_start = j;
        }
    }

    // (a) per-tile ordering + (c) partition.
    let mut covered: u32 = 0;
    let mut prev_end: u32 = 0;
    let mut tiles_with_pairs = 0usize;
    for (t, r) in ranges.iter().enumerate() {
        let (start, end) = (r[0], r[1]);
        if start == 0 && end == 0 {
            continue; // empty tile
        }
        assert!(end > start, "[{name}] tile {t}: inverted range {start}..{end}");
        assert!(
            start >= prev_end,
            "[{name}] tile {t}: range {start}..{end} overlaps previous end {prev_end}"
        );
        prev_end = end;
        tiles_with_pairs += 1;
        covered += end - start;
        let mut last_depth = 0u32;
        for j in start..end {
            let p = &pairs[j as usize];
            assert_eq!(
                p.key_tile as usize, t,
                "[{name}] tile {t}: pair {j} has tile {}",
                p.key_tile
            );
            assert!(
                p.key_depth >= last_depth,
                "[{name}] tile {t}: depth order violated at pair {j}"
            );
            last_depth = p.key_depth;
        }
    }
    assert_eq!(
        covered, total_pairs,
        "[{name}] ranges do not partition the pair array"
    );
    assert!(
        tiles_with_pairs > 1,
        "[{name}] only one tile populated — fixture too easy"
    );
    eprintln!(
        "[{name}] {} pairs over {}/{} tiles, ordering+conservation+partition all exact",
        total_pairs, tiles_with_pairs, num_tiles
    );
}

#[test]
fn tile_sort_properties_smoke() {
    check_scene("smoke", smoke_scene());
}

#[test]
fn tile_sort_properties_regression() {
    check_scene("regression", regression_scene());
}
