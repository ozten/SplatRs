// Tile binning for tile-based rasterization (docs/TILE_RASTER_PLAN.md Part B).
//
// Stage 1: per-gaussian tile-touch counting. MUST agree integer-for-integer with the CPU
// oracle `render::tile_math::tile_touch_count` — tests/unit_gpu_tile_counting.rs asserts
// exact equality on the projected values this kernel actually reads, so any change to the
// AABB/clip convention must land in BOTH places.

// Gaussian 2D structure (output of project_gaussians; layout matches rasterize.wgsl).
struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad); culled => depth = -1
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy, 3-sigma bound radius); culled => 0
    color: vec4<f32>,
    opacity_pad: vec4<f32>,
    gaussian_idx_pad: vec4<u32>,
}

struct TileBinParams {
    tiles_x: u32,
    tiles_y: u32,
    num_gaussians: u32,
    pad: u32,
}

const TILE_SIZE: f32 = 16.0;

@group(0) @binding(0) var<uniform> params: TileBinParams;
@group(0) @binding(1) var<storage, read> projected: array<Gaussian2D>;
@group(0) @binding(2) var<storage, read_write> touch_counts: array<u32>;

@compute @workgroup_size(256)
fn count_tile_touches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_gaussians) {
        return;
    }
    let g = projected[i];
    let radius = g.cov.w;
    // Mirrors tile_touch_rect's `!(radius > 0.0)` guard: culled gaussians write cov = 0,
    // and a NaN radius must also land here (NaN <= 0.0 is false, so don't use `<=`).
    if (!(radius > 0.0)) {
        touch_counts[i] = 0u;
        return;
    }
    let x0 = floor((g.mean.x - radius) / TILE_SIZE);
    let x1 = floor((g.mean.x + radius) / TILE_SIZE);
    let y0 = floor((g.mean.y - radius) / TILE_SIZE);
    let y1 = floor((g.mean.y + radius) / TILE_SIZE);
    let max_x = f32(params.tiles_x - 1u);
    let max_y = f32(params.tiles_y - 1u);
    if (x1 < 0.0 || y1 < 0.0 || x0 > max_x || y0 > max_y) {
        touch_counts[i] = 0u;
        return;
    }
    let cx0 = clamp(x0, 0.0, max_x);
    let cx1 = clamp(x1, 0.0, max_x);
    let cy0 = clamp(y0, 0.0, max_y);
    let cy1 = clamp(y1, 0.0, max_y);
    touch_counts[i] = u32(cx1 - cx0 + 1.0) * u32(cy1 - cy0 + 1.0);
}
