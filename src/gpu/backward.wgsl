// WGSL shader for backward pass (gradient computation).
//
// This shader reads intermediate values saved during the forward pass
// (transmittances, alphas, Gaussian indices) and computes gradients
// w.r.t. Gaussian parameters using the chain rule.
//
// Phase 2: Per-pixel gradient computation with per-workgroup accumulation.

// Contribution structure (matches forward pass)
struct Contribution {
    transmittance: f32,       // T before this Gaussian
    alpha: f32,               // α of this Gaussian
    gaussian_idx: u32,        // Source Gaussian index
    pad: u32,                 // Alignment
}

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
    width: u32,          // Full image width (for intermediates indexing)
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
@group(0) @binding(1) var<storage, read> intermediates: array<Contribution>;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read> d_pixels: array<vec4<f32>>;  // Upstream gradients
@group(0) @binding(4) var<storage, read_write> gradient_atomic: array<atomic<i32>>;  // Per-Gaussian gradients as atomic i32 (16 i32s per Gaussian)
@group(0) @binding(5) var<storage, read_write> d_background_pixels: array<vec4<f32>>;  // Per-pixel background gradient contribution (f32, summed on CPU)

// Maximum contributions per pixel (must match Rust constant)
const MAX_CONTRIBUTIONS_PER_PIXEL: u32 = 16u;

// Fixed-point scale for gradient accumulation
// Gradients are multiplied by this before converting to i32
// This preserves precision while using integer atomics (Metal-compatible)
const FIXED_POINT_SCALE: f32 = 10000000.0;  // 7 decimal places of precision

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

// Atomic add for f32 using fixed-point integer atomics
// Metal doesn't support atomicCompareExchange, so we use fixed-point arithmetic
// Convert f32 to scaled i32, use atomicAdd, then convert back to f32 on CPU
fn atomic_add_f32(index: u32, value: f32) {
    let scaled = i32(value * FIXED_POINT_SCALE);
    atomicAdd(&gradient_atomic[index], scaled);
}

// Note: Background gradient is now stored per-pixel (not atomic) to avoid i32 overflow
// when summing across thousands of pixels. The CPU will sum the per-pixel contributions.

// Gradient buffer layout (16 i32s per Gaussian):
// [0-3]: d_color (vec4<f32> as fixed-point i32)
// [4-7]: d_opacity_logit_pad (vec4<f32> as fixed-point i32)
// [8-11]: d_mean_px (vec4<f32> as fixed-point i32)
// [12-15]: d_cov_2d (vec4<f32> as fixed-point i32)
const GRADIENT_STRIDE: u32 = 16u;  // 16 i32s = 64 bytes per Gaussian


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

    // Read contributions for this pixel
    let base_contrib_idx = pixel_idx * MAX_CONTRIBUTIONS_PER_PIXEL;

    // Count valid contributions
    var num_contribs = 0u;
    for (var i = 0u; i < MAX_CONTRIBUTIONS_PER_PIXEL; i++) {
        let contrib = intermediates[base_contrib_idx + i];
        if (contrib.gaussian_idx == 0xFFFFFFFFu) {
            break;
        }
        num_contribs += 1u;
    }

    // Blend backward pass (same logic as CPU blend_backward_with_bg)
    //
    // We backpropagate through:
    //   out = sum_i T_i * a_i * c_i + T_final * bg
    //   T_{i+1} = T_i * (1 - a_i)
    //
    // Using reverse-mode accumulation of transmittance gradients.

    var g_t_next = 0.0; // dL/d(T_{i+1}) as we go backwards

    // Process contributions in reverse order
    for (var i = 0u; i < num_contribs; i++) {
        let k = num_contribs - 1u - i;  // Reverse index
        let contrib = intermediates[base_contrib_idx + k];
        let sorted_idx = contrib.gaussian_idx;  // This is the sorted index
        let alpha = contrib.alpha;
        let t_i = contrib.transmittance;

        // Read Gaussian from sorted array using sorted index
        let g = gaussians[sorted_idx];

        // Get original Gaussian index for gradient accumulation
        let gaussian_idx = g.gaussian_idx_pad.x;

        let color = g.color.xyz;
        let opacity = g.opacity_pad.x;
        let mean_x = g.mean.x;
        let mean_y = g.mean.y;
        let cov_xx = g.cov.x;
        let cov_xy = g.cov.y;
        let cov_yy = g.cov.z;

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
        //
        // We need to recompute weight from the Gaussian evaluation
        let weight = eval_gaussian_2d(
            mean_x, mean_y,
            cov_xx, cov_xy, cov_yy,
            pixel_x, pixel_y
        );

        // Check if alpha was clamped
        let alpha_raw = opacity * weight;
        let was_clamped = alpha_raw >= 0.99;

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
        atomic_add_f32(base_idx + 8u, d_mean.x * d_weight);
        atomic_add_f32(base_idx + 9u, d_mean.y * d_weight);

        // d_cov_2d (offsets 12-14 for xx,xy,yy; 15 is padding)
        atomic_add_f32(base_idx + 12u, d_cov.x * d_weight);
        atomic_add_f32(base_idx + 13u, d_cov.y * d_weight);
        atomic_add_f32(base_idx + 14u, d_cov.z * d_weight);
    }

    // Background gradient contribution
    // out = ... + T_final * bg
    // dL/d(bg) = d_out * T_final
    //
    // T_final = T_0 * prod(1 - a_i) for all contributions
    // We need to compute this by going forward through contributions
    var t_final = 1.0;  // T_0 = 1.0
    for (var i = 0u; i < num_contribs; i++) {
        let contrib = intermediates[base_contrib_idx + i];
        let alpha = contrib.alpha;
        t_final *= (1.0 - alpha);
    }

    // dL/d(bg) = d_out * T_final
    let d_background = d_out * t_final;

    // Store per-pixel background gradient (CPU will sum all pixels)
    // Use pixel_idx (global pixel index) for storage
    d_background_pixels[pixel_idx] = vec4<f32>(d_background, 0.0);
}
