// Tile-binned forward rasterizer (docs/TILE_RASTER_PLAN.md Part B Stage 3).
//
// One 16x16 workgroup per tile. The workgroup cooperatively loads batches of 256
// (tile, gaussian) pairs — already depth-sorted within the tile by Stage 2 — into
// workgroup memory, then every thread blends its own pixel against the batch.
//
// PARITY CONTRACT: the per-pixel math below replicates rasterize.wgsl (the oracle)
// expression-for-expression: same bbox floor/ceil integer-pixel test, same
// eval_gaussian_2d, same alpha = min(opacity * w, 0.99), same alpha < 1e-4 skip, same
// front-to-back blend and T < 1e-4 early stop. The only intended difference is WHICH
// gaussians are iterated (this tile's sorted range instead of the global sorted array) —
// the binning rect is a superset of the bbox test's pixel reach, so no contributor is
// missed. tests/unit_gpu_tile_raster_parity.rs enforces max <= 1e-5 vs the oracle.
//
// CONSTRAINT (from the Stage-2 bug): no two threads may write different components of a
// shared storage vector — every store below is a whole-vec4 store to a thread-owned pixel.

struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad)
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy, 3-sigma bound radius)
    color: vec4<f32>,         // Linear RGB
    opacity_pad: vec4<f32>,   // Opacity [0,1]
    gaussian_idx_pad: vec4<u32>,
}

struct TileGaussianPair {
    key_tile: u32,
    key_depth: u32,
    gaussian_idx: u32,
    pad: u32,
}

struct TileRasterParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    save_intermediates: u32,  // 1 = save per-pixel state for the backward pass (Stage 5b), 0 = don't
    pad0: u32,
    pad1: u32,
    pad2: u32,
    background: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: TileRasterParams;
@group(0) @binding(1) var<storage, read> projected: array<Gaussian2D>;
@group(0) @binding(2) var<storage, read> pairs: array<TileGaussianPair>;
// Flat [start0, end0, start1, end1, ...] per tile (see tile_bin.wgsl).
@group(0) @binding(3) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;
// Per-pixel forward state for the backward pass (Stage 5b), same contract as
// rasterize.wgsl's pixel_state:
//   x = final transmittance after the last blended contributor (bitcast f32)
//   y = ABSOLUTE index into the global `pairs` buffer of the last blended contributor
//       (0xFFFFFFFF if none). Unlike the naive path (whose sorted index refers into the
//       globally depth-sorted Gaussian2D array), the tiled path's "resolved identity" for
//       a stored index is `pairs[y].gaussian_idx` — the pair buffer is the only
//       sorted-order artifact this kernel walks.
// Only written when save_intermediates != 0. This is a SAFE whole-vec2 store: every
// pixel_idx is written by exactly one thread (the pixel it owns), unlike tile_ranges in
// tile_bin.wgsl where two DIFFERENT threads write different components of one tile's
// entry and a component store to a storage vector silently races (the Stage-2 bug) —
// that constraint doesn't apply here because there's no cross-thread sharing of one vector.
@group(0) @binding(5) var<storage, read_write> pixel_state: array<vec2<u32>>;

const BATCH: u32 = 256u;
// (mean.x, mean.y, cov_xx, cov_xy)
var<workgroup> batch_a: array<vec4<f32>, 256>;
// (cov_yy, color.r, color.g, color.b)
var<workgroup> batch_b: array<vec4<f32>, 256>;
// (opacity, bound_radius)
var<workgroup> batch_c: array<vec2<f32>, 256>;

fn eval_gaussian_2d(mean_x: f32, mean_y: f32, cov_xx: f32, cov_xy: f32, cov_yy: f32,
                     pixel_x: f32, pixel_y: f32) -> f32 {
    let dx = pixel_x - mean_x;
    let dy = pixel_y - mean_y;
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return 0.0;
    }
    let inv_det = 1.0 / det;
    let inv_xx = cov_yy * inv_det;
    let inv_xy = -cov_xy * inv_det;
    let inv_yy = cov_xx * inv_det;
    let quad_form = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
    return exp(-0.5 * quad_form);
}

@compute @workgroup_size(16, 16)
fn rasterize_tiled(@builtin(workgroup_id) wg_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tile_id = wg_id.y * params.tiles_x + wg_id.x;
    let px = wg_id.x * 16u + local_id.x;
    let py = wg_id.y * 16u + local_id.y;
    let local_idx = local_id.y * 16u + local_id.x;
    let in_image = px < params.width && py < params.height;
    let pixel_idx = py * params.width + px;

    let range_start = tile_ranges[2u * tile_id];
    let range_end = tile_ranges[2u * tile_id + 1u];

    let pixel_x = f32(px) + 0.5;
    let pixel_y = f32(py) + 0.5;

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance = 1.0;
    var last_pair_idx = 0xFFFFFFFFu;
    // done only suppresses further blending for THIS thread; every thread stays in the
    // batch loop so workgroupBarrier() control flow remains workgroup-uniform.
    var done = !in_image;

    var batch_start = range_start;
    loop {
        if (batch_start >= range_end) { break; }
        let batch_len = min(BATCH, range_end - batch_start);
        if (local_idx < batch_len) {
            let g = projected[pairs[batch_start + local_idx].gaussian_idx];
            batch_a[local_idx] = vec4<f32>(g.mean.x, g.mean.y, g.cov.x, g.cov.y);
            batch_b[local_idx] = vec4<f32>(g.cov.z, g.color.x, g.color.y, g.color.z);
            batch_c[local_idx] = vec2<f32>(g.opacity_pad.x, g.cov.w);
        }
        workgroupBarrier();
        if (!done) {
            for (var k = 0u; k < batch_len; k++) {
                let a = batch_a[k];
                let c = batch_c[k];
                let radius = c.y;
                // Identical per-pixel bbox test to the oracle (integer-pixel semantics).
                if (f32(px) < floor(a.x - radius) || f32(px) > ceil(a.x + radius) ||
                    f32(py) < floor(a.y - radius) || f32(py) > ceil(a.y + radius)) {
                    continue;
                }
                let b = batch_b[k];
                let weight = eval_gaussian_2d(a.x, a.y, a.z, a.w, b.x, pixel_x, pixel_y);
                let alpha = min(c.x * weight, 0.99);
                if (alpha < 1e-4) {
                    continue;
                }
                // Track the last blended contributor as an ABSOLUTE pairs-buffer index
                // (batch_start + local batch slot k) for the backward pass (Stage 5b).
                last_pair_idx = batch_start + k;
                color = color + transmittance * alpha * b.yzw;
                transmittance = transmittance * (1.0 - alpha);
                if (transmittance < 1e-4) {
                    done = true;
                    break;
                }
            }
        }
        workgroupBarrier();
        batch_start += BATCH;
    }

    if (in_image) {
        // Save per-pixel forward state for the backward pass BEFORE compositing the
        // background (matches rasterize.wgsl's ordering; the background add doesn't
        // change `transmittance` itself, so this is not order-sensitive for correctness,
        // just for parity with the naive kernel's structure).
        if (params.save_intermediates != 0u) {
            pixel_state[pixel_idx] = vec2<u32>(bitcast<u32>(transmittance), last_pair_idx);
        }
        color += transmittance * params.background.xyz;
        output[pixel_idx] = vec4<f32>(color, 1.0);
    }
}
