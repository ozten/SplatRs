// Tiled backward pass (docs/TILE_RASTER_PLAN.md Part B Stage 5b).
//
// One 16x16 workgroup per tile — SAME wg_id -> tile_id/px/py mapping as
// rasterize_tiled.wgsl. Walks the FULL tile range BACK-TO-FRONT in 256-pair batches
// (cooperative load into workgroup memory, workgroupBarrier, per-thread process,
// workgroupBarrier), reconstructing T_i exactly as backward.wgsl's naive re-walk does
// (t_i = t_after / (1 - alpha)), and accumulating gradients into the SAME global
// fixed-point atomic buffer via the SAME atomic helpers/scales. gaussian_idx needs NO
// indirection: the tile path's `pairs` buffer already carries ORIGINAL Gaussian indices
// (projection never reorders `projected` on the tile path — see rasterize_tiled.wgsl).
//
// PARITY CONTRACT:
//  - Math (eval_gaussian_2d, gaussian2d_grad_mean, gaussian2d_grad_cov, atomic_add_f32,
//    atomic_add_f32_position, FIXED_POINT_SCALE(+POSITION), GRADIENT_STRIDE) is copied
//    VERBATIM from backward.wgsl.
//  - The per-pixel contributor TEST (bbox + alpha) matches rasterize_tiled.wgsl's forward
//    kernel exactly, NOT backward.wgsl's extra `g.mean.z < 0.0` check — that check only
//    exists in the naive backward because it re-walks the full GLOBALLY sorted array
//    (which contains culled/padding sentinels sorted to the front). The tile path's
//    `pairs` buffer never contains a culled gaussian in the first place (tile_bin.wgsl's
//    counting/emit kernels only touch tiles for gaussians that pass the forward cull), so
//    there is nothing to filter here.
//  - `pixel_state.y` (last_pair_idx) is an ABSOLUTE index into the GLOBAL `pairs` buffer,
//    always inside [range_start, range_end) of the tile owning that pixel (or the
//    0xFFFFFFFF sentinel) — see rasterize_tiled.wgsl's pixel_state contract. The gate
//    `pair_idx > last_pair_idx -> continue` is therefore sufficient to reproduce exactly
//    the set of contributors the forward pass blended: forward walks this same range
//    ascending and blends every pair that passes bbox+alpha until (and including) the one
//    that pushes T below the early-stop threshold, so every pair_idx <= last_pair_idx that
//    ALSO passes bbox+alpha was blended, and none past it were reached.

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

struct TileBackwardParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
    background: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: TileBackwardParams;
@group(0) @binding(1) var<storage, read> projected: array<Gaussian2D>;
@group(0) @binding(2) var<storage, read> pairs: array<TileGaussianPair>;
// Flat [start0, end0, start1, end1, ...] per tile (see tile_bin.wgsl / rasterize_tiled.wgsl).
@group(0) @binding(3) var<storage, read> tile_ranges: array<u32>;
// Per-pixel forward state (matches rasterize_tiled.wgsl's pixel_state contract):
//   x = final transmittance (bitcast f32)
//   y = ABSOLUTE index into `pairs` of the last blended contributor (sentinel 0xFFFFFFFF)
@group(0) @binding(4) var<storage, read> pixel_state: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> d_pixels: array<vec4<f32>>; // Upstream gradients
@group(0) @binding(6) var<storage, read_write> gradient_atomic: array<atomic<i32>>; // Per-Gaussian fixed-point gradients
@group(0) @binding(7) var<storage, read_write> d_background_pixels: array<vec4<f32>>; // Per-pixel background gradient (summed on CPU)

const BATCH: u32 = 256u;
// (mean.x, mean.y, cov_xx, cov_xy) — same packing as rasterize_tiled.wgsl's batch_a.
var<workgroup> batch_a: array<vec4<f32>, 256>;
// (cov_yy, color.r, color.g, color.b) — same packing as rasterize_tiled.wgsl's batch_b.
var<workgroup> batch_b: array<vec4<f32>, 256>;
// (opacity, bound_radius) — same packing as rasterize_tiled.wgsl's batch_c.
var<workgroup> batch_c: array<vec2<f32>, 256>;
// ORIGINAL gaussian index (pairs[i].gaussian_idx) — NEW vs the forward kernel, needed so
// the atomic gradient accumulation doesn't have to re-touch `pairs` per contributor.
var<workgroup> batch_idx: array<u32, 256>;

// ---- Verbatim from backward.wgsl ----

// Evaluate 2D Gaussian at a pixel (same as in rasterize.wgsl / rasterize_tiled.wgsl)
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

// Fixed-point scales for atomic gradient accumulation (see backward.wgsl for the full
// derivation/comment — reproduced here verbatim since this shader must use the SAME
// scales as the naive backward, since both accumulate into the same gradient_atomic
// buffer contract read back by GpuRenderer).
const FIXED_POINT_SCALE: f32 = 1e7;
const FIXED_POINT_SCALE_POSITION: f32 = 1e9;

// Atomic add for color/opacity gradients (scale 10^7)
fn atomic_add_f32(index: u32, value: f32) {
    if (abs(value) < 1e-12) {
        return;
    }
    let scaled = value * FIXED_POINT_SCALE;
    let clamped = clamp(scaled, -2147483647.0, 2147483647.0);
    let fixed = i32(clamped);
    if (fixed != 0) {
        atomicAdd(&gradient_atomic[index], fixed);
    }
}

// Atomic add for position/covariance gradients (scale 10^9)
fn atomic_add_f32_position(index: u32, value: f32) {
    if (abs(value) < 1e-14) {
        return;
    }
    let scaled = value * FIXED_POINT_SCALE_POSITION;
    let clamped = clamp(scaled, -2147483647.0, 2147483647.0);
    let fixed = i32(clamped);
    if (fixed != 0) {
        atomicAdd(&gradient_atomic[index], fixed);
    }
}

// Gradient buffer layout (16 u32s per Gaussian, each u32 is bitcast f32):
// [0-3]: d_color (vec4<f32> as bitcast u32)
// [4-7]: d_opacity_logit_pad (vec4<f32> as bitcast u32)
// [8-11]: d_mean_px (vec4<f32> as bitcast u32)
// [12-15]: d_cov_2d (vec4<f32> as bitcast u32)
const GRADIENT_STRIDE: u32 = 16u;  // 16 u32s = 64 bytes per Gaussian

// Compute gradient of Gaussian 2D evaluation w.r.t. mean
fn gaussian2d_grad_mean(
    mean_x: f32, mean_y: f32,
    cov_xx: f32, cov_xy: f32, cov_yy: f32,
    pixel_x: f32, pixel_y: f32,
    weight: f32
) -> vec2<f32> {
    let dx = pixel_x - mean_x;
    let dy = pixel_y - mean_y;

    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    let inv_det = 1.0 / det;
    let inv_xx = cov_yy * inv_det;
    let inv_xy = -cov_xy * inv_det;
    let inv_yy = cov_xx * inv_det;

    let d_mean_x = weight * (inv_xx * dx + inv_xy * dy);
    let d_mean_y = weight * (inv_xy * dx + inv_yy * dy);

    return vec2<f32>(d_mean_x, d_mean_y);
}

// Compute gradient of Gaussian 2D evaluation w.r.t. covariance
fn gaussian2d_grad_cov(
    mean_x: f32, mean_y: f32,
    cov_xx: f32, cov_xy: f32, cov_yy: f32,
    pixel_x: f32, pixel_y: f32,
    weight: f32
) -> vec3<f32> {
    let dx = pixel_x - mean_x;
    let dy = pixel_y - mean_y;

    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let inv_det = 1.0 / det;
    let inv_xx = cov_yy * inv_det;
    let inv_xy = -cov_xy * inv_det;
    let inv_yy = cov_xx * inv_det;

    let inv_d_x = inv_xx * dx + inv_xy * dy;
    let inv_d_y = inv_xy * dx + inv_yy * dy;

    let d_cov_xx = 0.5 * weight * inv_d_x * inv_d_x;
    let d_cov_xy = 0.5 * weight * inv_d_x * inv_d_y * 2.0; // Factor of 2 for symmetry
    let d_cov_yy = 0.5 * weight * inv_d_y * inv_d_y;

    return vec3<f32>(d_cov_xx, d_cov_xy, d_cov_yy);
}

// ---- End verbatim section ----

@compute @workgroup_size(16, 16)
fn backward_pass_tiled(@builtin(workgroup_id) wg_id: vec3<u32>,
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

    // Per-pixel forward state. Only read/written for in-image pixels — px/py can run past
    // width/height for the partial tiles at the image edges (16 doesn't divide width or
    // height in general), and pixel_idx computed from an out-of-range px/py is not a valid
    // buffer index to read (same convention as rasterize_tiled.wgsl's output/pixel_state
    // writes, which are also gated by `in_image`).
    var d_out = vec3<f32>(0.0, 0.0, 0.0);
    var t_final = 1.0;
    var last_pair_idx = 0xFFFFFFFFu;
    if (in_image) {
        d_out = d_pixels[pixel_idx].xyz;
        let state = pixel_state[pixel_idx];
        t_final = bitcast<f32>(state.x);
        last_pair_idx = state.y;

        // Background gradient: dL/d(bg) = d_out * T_final, using the forward pass's TRUE
        // final transmittance (same as backward.wgsl). Written for every in-image pixel
        // regardless of whether it had any contributor.
        d_background_pixels[pixel_idx] = vec4<f32>(d_out * t_final, 0.0);
    }

    // has_contrib: this pixel both exists in the image AND had at least one blended
    // contributor. Threads that fail this stay in the workgroup-uniform batch loop below
    // (for barrier uniformity) but never do any per-contributor work.
    let has_contrib = in_image && (last_pair_idx != 0xFFFFFFFFu);

    // Reverse-mode transmittance-gradient accumulation (see backward.wgsl): g_t_next
    // starts from the background term's contribution to dL/dT_N, and t_after tracks
    // T_{i+1} for the contributor currently being processed as we walk back-to-front.
    var g_t_next = dot(d_out, params.background.xyz);
    var t_after = t_final;

    // Walk the FULL tile range back-to-front in <=256-pair batches. batch_end descends
    // from range_end to range_start; every thread in the workgroup takes the same number
    // of outer-loop iterations (range_start/range_end depend only on wg_id, not
    // local_id), so workgroupBarrier() below stays in workgroup-uniform control flow.
    var batch_end = range_end;
    loop {
        if (batch_end <= range_start) {
            break;
        }
        var batch_begin = range_start;
        if (batch_end - range_start > BATCH) {
            batch_begin = batch_end - BATCH;
        }
        let batch_len = batch_end - batch_begin;

        // Cooperative load: thread local_idx loads pair (batch_begin + local_idx).
        if (local_idx < batch_len) {
            let pair = pairs[batch_begin + local_idx];
            let g = projected[pair.gaussian_idx];
            batch_a[local_idx] = vec4<f32>(g.mean.x, g.mean.y, g.cov.x, g.cov.y);
            batch_b[local_idx] = vec4<f32>(g.cov.z, g.color.x, g.color.y, g.color.z);
            batch_c[local_idx] = vec2<f32>(g.opacity_pad.x, g.cov.w);
            batch_idx[local_idx] = pair.gaussian_idx;
        }
        workgroupBarrier();

        if (has_contrib) {
            // Inner loop: k = batch_len down to 1 (kk = k-1 is the 0-based shared-memory
            // slot), so pair_idx walks batch_end-1 down to batch_begin — back-to-front
            // within this batch, continuing the back-to-front walk across batches.
            for (var k = batch_len; k > 0u; k--) {
                let kk = k - 1u;
                let pair_idx = batch_begin + kk;
                if (pair_idx > last_pair_idx) {
                    // Forward never reached this pair for this pixel (it's past the
                    // point where the forward pass stopped blending).
                    continue;
                }

                let a = batch_a[kk];
                let c = batch_c[kk];
                let radius = c.y;
                // Identical per-pixel bbox test to rasterize_tiled.wgsl / the oracle.
                if (f32(px) < floor(a.x - radius) || f32(px) > ceil(a.x + radius) ||
                    f32(py) < floor(a.y - radius) || f32(py) > ceil(a.y + radius)) {
                    continue;
                }

                let b = batch_b[kk];
                let mean_x = a.x;
                let mean_y = a.y;
                let cov_xx = a.z;
                let cov_xy = a.w;
                let cov_yy = b.x;
                let color = b.yzw;
                let opacity = c.x;

                let fwd_weight = eval_gaussian_2d(
                    mean_x, mean_y,
                    cov_xx, cov_xy, cov_yy,
                    pixel_x, pixel_y
                );
                let fwd_alpha_raw = opacity * fwd_weight;
                let alpha = min(fwd_alpha_raw, 0.99);
                if (alpha < 1e-4) {
                    continue;
                }

                // Reconstruct T before this Gaussian from T after it (same recurrence as
                // backward.wgsl: alpha is capped at 0.99, so the divisor is >= 0.01).
                let t_i = t_after / (1.0 - alpha);
                t_after = t_i;

                let gaussian_idx = batch_idx[kk];

                // dL/dc_i = d_out * (T_i * a_i)
                let d_color = d_out * (t_i * alpha);

                // dL/da_i: direct term from the output + indirect term via T_{i+1}.
                let direct = dot(d_out, color * t_i);
                let indirect = g_t_next * (-t_i);
                let d_alpha = direct + indirect;

                // Accumulate dL/dT_i and carry it as g_t_next for the next (earlier)
                // contributor.
                let g_t_i_from_out = dot(d_out, color * alpha);
                let g_t_i_from_next = g_t_next * (1.0 - alpha);
                let g_t_i = g_t_i_from_out + g_t_i_from_next;
                g_t_next = g_t_i;

                // Chain rule: d_alpha -> d_opacity_logit, d_weight.
                let weight = fwd_weight;
                let was_clamped = fwd_alpha_raw >= 0.99;
                let d_opacity = select(d_alpha * weight, 0.0, was_clamped);
                let d_weight = select(d_alpha * opacity, 0.0, was_clamped);
                let d_opacity_logit = d_opacity * opacity * (1.0 - opacity);

                // Chain rule: d_weight -> d_mean_px, d_cov_2d.
                let d_mean = gaussian2d_grad_mean(
                    mean_x, mean_y,
                    cov_xx, cov_xy, cov_yy,
                    pixel_x, pixel_y,
                    weight
                );
                let d_cov = gaussian2d_grad_cov(
                    mean_x, mean_y,
                    cov_xx, cov_xy, cov_yy,
                    pixel_x, pixel_y,
                    weight
                );

                // Accumulate to per-Gaussian gradients using atomic operations. No
                // indirection needed for gaussian_idx: pairs carry ORIGINAL indices.
                let base_idx = gaussian_idx * GRADIENT_STRIDE;

                atomic_add_f32(base_idx + 0u, d_color.x);
                atomic_add_f32(base_idx + 1u, d_color.y);
                atomic_add_f32(base_idx + 2u, d_color.z);

                atomic_add_f32(base_idx + 4u, d_opacity_logit);

                atomic_add_f32_position(base_idx + 8u, d_mean.x * d_weight);
                atomic_add_f32_position(base_idx + 9u, d_mean.y * d_weight);

                atomic_add_f32_position(base_idx + 12u, d_cov.x * d_weight);
                atomic_add_f32_position(base_idx + 13u, d_cov.y * d_weight);
                atomic_add_f32_position(base_idx + 14u, d_cov.z * d_weight);
            }
        }
        workgroupBarrier();
        batch_end = batch_begin;
    }
}
