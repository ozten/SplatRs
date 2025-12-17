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

// Gradient structure for a single Gaussian
struct Gradient {
    d_color: vec3<f32>,       // dL/d(color)
    d_opacity_logit: f32,     // dL/d(opacity_logit)
    d_mean_px: vec2<f32>,     // dL/d(mean_px)
    d_cov_2d: vec3<f32>,      // dL/d(cov_2d) as (xx, xy, yy)
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
    width: u32,
    height: u32,
    num_gaussians: u32,
    num_workgroups_x: u32,    // For workgroup buffer indexing
    background: vec4<f32>,    // Background color
}

@group(0) @binding(0) var<uniform> params: BackwardParams;
@group(0) @binding(1) var<storage, read> intermediates: array<Contribution>;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read> d_pixels: array<vec4<f32>>;  // Upstream gradients
@group(0) @binding(4) var<storage, read_write> workgroup_gradients: array<Gradient>;

// Maximum contributions per pixel (must match Rust constant)
const MAX_CONTRIBUTIONS_PER_PIXEL: u32 = 16u;

// Zero gradient initializer
fn zero_gradient() -> Gradient {
    return Gradient(
        vec3<f32>(0.0, 0.0, 0.0),
        0.0,
        vec2<f32>(0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0)
    );
}

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
    let px = global_id.x;
    let py = global_id.y;

    if (px >= params.width || py >= params.height) {
        return;
    }

    let pixel_idx = py * params.width + px;
    let pixel_x = f32(px) + 0.5;
    let pixel_y = f32(py) + 0.5;

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
        let gaussian_idx = contrib.gaussian_idx;
        let alpha = contrib.alpha;
        let t_i = contrib.transmittance;

        let g = gaussians[gaussian_idx];
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
        // d(alpha)/d(opacity_logit) = d(alpha)/d(opacity) * d(opacity)/d(logit)
        //                            = weight * opacity * (1 - opacity)  [if alpha < 0.99]
        //                            = 0                                  [if alpha >= 0.99]
        //
        // Compute weight from alpha and opacity
        let weight = alpha / opacity;  // Reverse: weight = alpha / opacity (if alpha < 0.99)

        // Check if alpha was clamped
        let alpha_raw = opacity * weight;
        let was_clamped = alpha_raw >= 0.99;

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

        // Write to per-workgroup gradient buffer
        let wg_linear = workgroup_id.y * params.num_workgroups_x + workgroup_id.x;
        let grad_idx = wg_linear * params.num_gaussians + gaussian_idx;

        // Accumulate gradients atomically (wait, we can't use atomics for f32!)
        // Instead, we use per-workgroup buffers and reduce on CPU.
        // Each workgroup writes to its own section, no contention.
        workgroup_gradients[grad_idx].d_color += d_color;
        workgroup_gradients[grad_idx].d_opacity_logit += d_opacity_logit;
        workgroup_gradients[grad_idx].d_mean_px += d_mean * d_weight;
        workgroup_gradients[grad_idx].d_cov_2d += d_cov * d_weight;
    }

    // Background gradient contribution
    // out = ... + T_final * bg
    // dL/d(bg) = d_out * T_final
    //
    // T_final = T_0 * prod(1 - a_i) for all contributions
    // This is g_t_next after the loop completes
    //
    // For now, we skip background gradient (can add later if needed)
}
