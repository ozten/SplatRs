//! CPU reference math for tile-binned rasterization (docs/TILE_RASTER_PLAN.md Part B).
//!
//! Pure Rust and feature-independent on purpose: the Stage-1 GPU counting kernel
//! (`tile_bin.wgsl`) must agree EXACTLY (integer-for-integer) with these functions — they
//! are the validation oracle for tile binning, and `tests/unit_gpu_tile_counting.rs`
//! asserts the match. Any change here must be mirrored in the WGSL and vice versa.

/// Tile edge length in pixels. Must match the workgroup size of the tile raster kernel.
pub const TILE_SIZE: u32 = 16;

/// Number of tiles along each axis for an image of `width`×`height` pixels.
pub fn tile_grid_dims(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(TILE_SIZE), height.div_ceil(TILE_SIZE))
}

/// Inclusive tile-index rectangle `(x0, x1, y0, y1)` touched by a splat whose screen-space
/// center is `(mx, my)` (pixels) with a square 3σ AABB half-extent `radius` (pixels),
/// clipped to the tile grid. `None` when the AABB misses the image entirely or
/// `radius <= 0` (culled / degenerate).
///
/// Convention (the WGSL kernel mirrors this exactly): the AABB is the closed interval
/// `[m - radius, m + radius]`; a tile `t` spans pixel interval `[16t, 16t + 15]` and is
/// touched when `floor((m - radius)/16) <= t <= floor((m + radius)/16)` after clipping.
pub fn tile_touch_rect(
    mx: f32,
    my: f32,
    radius: f32,
    tiles_x: u32,
    tiles_y: u32,
) -> Option<(u32, u32, u32, u32)> {
    if !(radius > 0.0) || tiles_x == 0 || tiles_y == 0 {
        return None;
    }
    let t = TILE_SIZE as f32;
    let x0 = ((mx - radius) / t).floor() as i64;
    let x1 = ((mx + radius) / t).floor() as i64;
    let y0 = ((my - radius) / t).floor() as i64;
    let y1 = ((my + radius) / t).floor() as i64;
    let max_x = tiles_x as i64 - 1;
    let max_y = tiles_y as i64 - 1;
    if x1 < 0 || y1 < 0 || x0 > max_x || y0 > max_y {
        return None;
    }
    Some((
        x0.clamp(0, max_x) as u32,
        x1.clamp(0, max_x) as u32,
        y0.clamp(0, max_y) as u32,
        y1.clamp(0, max_y) as u32,
    ))
}

/// Number of tiles touched (0 when off-screen/culled).
pub fn tile_touch_count(mx: f32, my: f32, radius: f32, tiles_x: u32, tiles_y: u32) -> u32 {
    match tile_touch_rect(mx, my, radius, tiles_x, tiles_y) {
        Some((x0, x1, y0, y1)) => (x1 - x0 + 1) * (y1 - y0 + 1),
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_dims_round_up() {
        assert_eq!(tile_grid_dims(490, 273), (31, 18));
        assert_eq!(tile_grid_dims(980, 545), (62, 35));
        assert_eq!(tile_grid_dims(16, 16), (1, 1));
        assert_eq!(tile_grid_dims(17, 1), (2, 1));
    }

    #[test]
    fn fully_inside_one_tile() {
        // Center of tile (1,1): pixels 16..31; radius small enough to stay inside.
        assert_eq!(tile_touch_rect(24.0, 24.0, 3.0, 31, 18), Some((1, 1, 1, 1)));
        assert_eq!(tile_touch_count(24.0, 24.0, 3.0, 31, 18), 1);
    }

    #[test]
    fn spans_tile_boundary() {
        // AABB [15, 17] crosses the x=16 boundary: tiles 0 and 1 in x, tile 0 in y.
        assert_eq!(tile_touch_rect(16.0, 8.0, 1.0, 31, 18), Some((0, 1, 0, 0)));
        assert_eq!(tile_touch_count(16.0, 8.0, 1.0, 31, 18), 2);
    }

    #[test]
    fn clipped_at_image_edge() {
        // Center off the left edge; AABB pokes into tile 0 only.
        assert_eq!(tile_touch_rect(-4.0, 8.0, 6.0, 31, 18), Some((0, 0, 0, 0)));
    }

    #[test]
    fn clipped_at_bottom_right() {
        // Explicit arithmetic: x in [449, 529] -> tiles 28..33 -> clamp to 28..30.
        assert_eq!(
            tile_touch_rect(489.0, 272.0, 40.0, 31, 18),
            Some((28, 30, 14, 17))
        );
    }

    #[test]
    fn off_screen_and_degenerate_are_none() {
        assert_eq!(tile_touch_rect(-100.0, 8.0, 5.0, 31, 18), None); // fully left
        assert_eq!(tile_touch_rect(8.0, 10_000.0, 5.0, 31, 18), None); // fully below
        assert_eq!(tile_touch_rect(8.0, 8.0, 0.0, 31, 18), None); // zero radius
        assert_eq!(tile_touch_rect(8.0, 8.0, f32::NAN, 31, 18), None); // NaN radius
    }

    #[test]
    fn screen_filling_splat_touches_every_tile() {
        // The pathological case TILE_RASTER_PLAN.md risk #1 worries about.
        assert_eq!(
            tile_touch_rect(245.0, 136.0, 10_000.0, 31, 18),
            Some((0, 30, 0, 17))
        );
        assert_eq!(tile_touch_count(245.0, 136.0, 10_000.0, 31, 18), 31 * 18);
    }
}
