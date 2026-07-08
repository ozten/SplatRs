// WGSL shader for backward pass (gradient computation).
//
// Reference-3DGS-style backward: the forward pass stores only the per-pixel final
// transmittance and the sorted index of the last blended contributor. This shader
// re-walks the sorted Gaussian list BACK-TO-FRONT from that index, re-applying the
// forward pass's exact tests and recomputing each alpha, reconstructing T_i
// incrementally via T_i = T_{i+1} / (1 - a_i). Every contributor down to the
// forward's termination point receives gradients — there is NO per-pixel
// contribution cap (the old 16-slot intermediates scheme gave exactly zero gradient
// to all Gaussians at depth rank > 16, starving ~90% of a converged population).

// Gradient structure for a single Gaussian (must match GradientGPU in Rust!)
struct Gradient {
    d_color: vec4<f32>,              // dL/d(color) + padding
    d_opacity_logit_pad: vec4<f32>,  // dL/d(opacity_logit) + padding
    d_mean_px: vec4<f32>,            // dL/d(mean_px) + padding
    d_cov_2d: vec4<f32>,             // dL/d(cov_2d) + padding
}

// Gaussian 2D structure (same as rasterize.wgsl)
struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad)
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy,pad)
    color: vec4<f32>,         // Linear RGB
    opacity_pad: vec4<f32>,   // Opacity [0,1]
    gaussian_idx_pad: vec4<u32>, // Source index
}

// Uniforms for backward pass
struct BackwardParams {
    width: u32,          // Full image width (for pixel_state indexing)
    height: u32,         // Full image height
    num_gaussians: u32,
    tile_start_x: u32,   // Tile offset in global coordinates
    tile_start_y: u32,   // Tile offset in global coordinates
    tile_width: u32,     // Tile dimensions (for boundary checks)
    tile_height: u32,    // Tile dimensions
    pad: u32,            // Padding for alignment
    background: vec4<f32>,    // Background color
}

@group(0) @binding(0) var<uniform> params: BackwardParams;
// Per-pixel forward state (matches rasterize.wgsl):
//   x = final transmittance (bitcast f32), y = sorted index of last blended contributor
@group(0) @binding(1) var<storage, read> pixel_state: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read> d_pixels: array<vec4<f32>>;  // Upstream gradients
@group(0) @binding(4) var<storage, read_write> gradient_atomic: array<atomic<i32>>;  // Per-Gaussian gradients as fixed-point i32
@group(0) @binding(5) var<storage, read_write> d_background_pixels: array<vec4<f32>>;  // Per-pixel background gradient contribution (f32, summed on CPU)


// Evaluate 2D Gaussian at a pixel (same as in rasterize.wgsl)
fn eval_gaussian_2d(mean_x: f32, mean_y: f32, cov_xx: f32, cov_xy: f32, cov_yy: f32,
                     pixel_x: f32, pixel_y: f32) -> f32 {
    let dx = pixel_x - mean_x;
    let dy = pixel_y - mean_y;

    // Compute inverse covariance
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return 0.0;
    }

    let inv_det = 1.0 / det;
    let inv_xx = cov_yy * inv_det;
    let inv_xy = -cov_xy * inv_det;
    let inv_yy = cov_xx * inv_det;

    // Quadratic form: (x - μ)^T Σ^-1 (x - μ)
    let quad_form = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;

    // Gaussian weight
    return exp(-0.5 * quad_form);
}

// Zero gradient initializer
fn zero_gradient() -> Gradient {
    return Gradient(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0)
    );
}

// Fixed-point scales for atomic gradient accumulation.
// Different scales for color/opacity vs position/covariance because position
// gradients have more multiplicative factors and are ~100× smaller.
//
// Color/opacity scale (10^7):
// - Color gradients: d_out × alpha × transmittance ≈ 10^-6 × 10^7 = 10 per pixel
// - With 1000 pixels: 10,000 per Gaussian → safe
//
// Position/covariance scale (10^9):
// - Position gradients: d_mean × d_weight ≈ 10^-8 × 10^9 = 10 per pixel
// - With 1000 pixels: 10,000 per Gaussian → safe
// - Higher scale needed because position grads are ~100× smaller than color
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
// Higher precision needed because these gradients are ~100× smaller
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

// Note: Background gradient is now stored per-pixel (not atomic) to avoid i32 overflow
// when summing across thousands of pixels. The CPU will sum the per-pixel contributions.

// Gradient buffer layout (16 u32s per Gaussian, each u32 is bitcast f32):
// [0-3]: d_color (vec4<f32> as bitcast u32)
// [4-7]: d_opacity_logit_pad (vec4<f32> as bitcast u32)
// [8-11]: d_mean_px (vec4<f32> as bitcast u32)
// [12-15]: d_cov_2d (vec4<f32> as bitcast u32)
const GRADIENT_STRIDE: u32 = 16u;  // 16 u32s = 64 bytes per Gaussian


// Compute gradient of Gaussian 2D evaluation w.r.t. mean
//
// Given weight = exp(-0.5 * quadratic_form), where
// quadratic_form = (x-μ)^T Σ^{-1} (x-μ)
//
// d(weight)/d(μ) = weight * Σ^{-1} (x - μ)
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

    // dw/dμ = weight * Σ^{-1} (x - μ)
    let d_mean_x = weight * (inv_xx * dx + inv_xy * dy);
    let d_mean_y = weight * (inv_xy * dx + inv_yy * dy);

    return vec2<f32>(d_mean_x, d_mean_y);
}

// Compute gradient of Gaussian 2D evaluation w.r.t. covariance
//
// d(weight)/d(Σ) = -0.5 * weight * d(quadratic_form)/d(Σ)
//
// Using: quadratic_form = (x-μ)^T Σ^{-1} (x-μ)
// and d(Σ^{-1})/d(Σ) = -Σ^{-1} * dΣ * Σ^{-1}
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

    // Outer product: (Σ^{-1} d) ⊗ (Σ^{-1} d)
    let inv_d_x = inv_xx * dx + inv_xy * dy;
    let inv_d_y = inv_xy * dx + inv_yy * dy;

    // d(quadratic_form)/d(Σ^{-1}) = (x-μ) ⊗ (x-μ)
    // d(Σ^{-1})/d(Σ_xx) = -Σ^{-1}_{·,0} Σ^{-1}_{0,·}
    // This gives d(w)/d(Σ) = 0.5 * w * Σ^{-1} (x-μ) ⊗ Σ^{-1} (x-μ)

    let d_cov_xx = 0.5 * weight * inv_d_x * inv_d_x;
    let d_cov_xy = 0.5 * weight * inv_d_x * inv_d_y * 2.0; // Factor of 2 for symmetry
    let d_cov_yy = 0.5 * weight * inv_d_y * inv_d_y;

    return vec3<f32>(d_cov_xx, d_cov_xy, d_cov_yy);
}

@compute @workgroup_size(16, 16)
fn backward_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Tile-local coordinates (within this tile)
    let tile_px = global_id.x;
    let tile_py = global_id.y;

    // Check tile bounds
    if (tile_px >= params.tile_width || tile_py >= params.tile_height) {
        return;
    }

    // Compute global pixel coordinates (in full image)
    let global_px = params.tile_start_x + tile_px;
    let global_py = params.tile_start_y + tile_py;

    // Global pixel index (for intermediates lookup in full image)
    let pixel_idx = global_py * params.width + global_px;

    // Tile-local pixel index (for gradient buffer writes)
    let tile_pixel_idx = tile_py * params.tile_width + tile_px;

    // Pixel center in global coordinates (for Gaussian evaluation)
    let pixel_x = f32(global_px) + 0.5;
    let pixel_y = f32(global_py) + 0.5;

    // Get upstream gradient for this pixel
    let d_out = d_pixels[pixel_idx].xyz;

    // Per-pixel forward state: TRUE final transmittance + last blended sorted index
    let state = pixel_state[pixel_idx];
    let t_final = bitcast<f32>(state.x);
    let last_idx = state.y;

    // Background gradient: dL/d(bg) = d_out * T_final, using the forward pass's true
    // final transmittance (the old scheme recomputed it from <=16 recorded contributions,
    // overestimating d_bg ~10x on dense pixels and dragging the background to black).
    d_background_pixels[pixel_idx] = vec4<f32>(d_out * t_final, 0.0);

    if (last_idx == 0xFFFFFFFFu) {
        // No contributors touched this pixel
        return;
    }

    // Blend backward pass (same logic as CPU blend_backward_with_bg)
    //
    // We backpropagate through:
    //   out = sum_i T_i * a_i * c_i + T_final * bg
    //   T_{i+1} = T_i * (1 - a_i)
    //
    // Using reverse-mode accumulation of transmittance gradients, walking the sorted
    // list back-to-front from the last blended contributor and re-applying the forward
    // pass's exact tests. T_i is reconstructed incrementally: T_i = T_{i+1} / (1 - a_i)
    // (safe: alpha is capped at 0.99, so the divisor is >= 0.01).

    // Initialize g_T_N from background term: out includes T_N * bg
    // dL/dT_N = d_out · bg (because changing T_N changes out by T_N * bg)
    var g_t_next = dot(d_out, params.background.xyz); // dL/d(T_{i+1}) as we go backwards
    var t_after = t_final;                            // T_{i+1} for the current contributor

    // Process contributors in reverse sorted order
    for (var j = 0u; j <= last_idx; j++) {
        let sorted_idx = last_idx - j;
        let g = gaussians[sorted_idx];

        // Same tests as the forward pass, so the contributor set matches exactly
        if (g.mean.z < 0.0) {
            continue;
        }

        // 3-sigma bounding box, identical to the forward rasterize pass (radius in cov.w)
        let radius = g.cov.w;
        if (f32(global_px) < floor(g.mean.x - radius) || f32(global_px) > ceil(g.mean.x + radius) ||
            f32(global_py) < floor(g.mean.y - radius) || f32(global_py) > ceil(g.mean.y + radius)) {
            continue;
        }

        let color = g.color.xyz;
        let opacity = g.opacity_pad.x;
        let mean_x = g.mean.x;
        let mean_y = g.mean.y;
        let cov_xx = g.cov.x;
        let cov_xy = g.cov.y;
        let cov_yy = g.cov.z;

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

        // Reconstruct T before this Gaussian from T after it
        let t_i = t_after / (1.0 - alpha);
        t_after = t_i;

        // Get original Gaussian index for gradient accumulation
        let gaussian_idx = g.gaussian_idx_pad.x;

        // Gradient of output w.r.t. color: dL/dc_i = d_out * (T_i * a_i)
        let d_color = d_out * (t_i * alpha);

        // Gradient w.r.t. alpha has two parts:
        // (1) Direct from output term: d_out · (T_i * c_i)
        // (2) Indirect via T_{i+1} = T_i * (1 - a_i)
        let direct = dot(d_out, color * t_i);
        let indirect = g_t_next * (-t_i);
        let d_alpha = direct + indirect;

        // Accumulate gradient for transmittance at position i:
        // dL/dT_i from output term: d_out · (a_i * c_i)
        // dL/dT_i from future transmittance: g_T_{i+1} * (1 - a_i)
        let g_t_i_from_out = dot(d_out, color * alpha);
        let g_t_i_from_next = g_t_next * (1.0 - alpha);
        let g_t_i = g_t_i_from_out + g_t_i_from_next;
        g_t_next = g_t_i;

        // Chain rule: d_alpha -> d_opacity_logit, d_weight
        //
        // alpha = min(opacity * weight, 0.99)
        // opacity = sigmoid(opacity_logit)
        let weight = fwd_weight;

        // Check if alpha was clamped
        let was_clamped = fwd_alpha_raw >= 0.99;

        // d(alpha)/d(opacity) = weight (if not clamped), 0 (if clamped)
        // d(alpha)/d(weight) = opacity (if not clamped), 0 (if clamped)
        let d_opacity = select(d_alpha * weight, 0.0, was_clamped);
        let d_weight = select(d_alpha * opacity, 0.0, was_clamped);

        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        let d_opacity_logit = d_opacity * opacity * (1.0 - opacity);

        // Chain rule: d_weight -> d_mean_px, d_cov_2d
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

        // Accumulate to per-Gaussian gradients using atomic operations
        // Each Gaussian has 16 u32s in the gradient_atomic buffer
        let base_idx = gaussian_idx * GRADIENT_STRIDE;

        // d_color (offsets 0-2 for RGB, 3 is padding)
        atomic_add_f32(base_idx + 0u, d_color.x);
        atomic_add_f32(base_idx + 1u, d_color.y);
        atomic_add_f32(base_idx + 2u, d_color.z);

        // d_opacity_logit_pad (offset 4 for opacity, 5-7 are padding)
        atomic_add_f32(base_idx + 4u, d_opacity_logit);

        // d_mean_px (offsets 8-9 for x,y; 10-11 are padding)
        // Use higher precision (10^9) for position gradients
        atomic_add_f32_position(base_idx + 8u, d_mean.x * d_weight);
        atomic_add_f32_position(base_idx + 9u, d_mean.y * d_weight);

        // d_cov_2d (offsets 12-14 for xx,xy,yy; 15 is padding)
        // Use higher precision (10^9) for covariance gradients
        atomic_add_f32_position(base_idx + 12u, d_cov.x * d_weight);
        atomic_add_f32_position(base_idx + 13u, d_cov.y * d_weight);
        atomic_add_f32_position(base_idx + 14u, d_cov.z * d_weight);
    }
}
