// WGSL shader for rasterizing 2D Gaussians to pixels.
//
// This is a naive per-pixel implementation. Each thread handles one pixel
// and loops through all Gaussians. This is slow but correct - we'll optimize
// with tile-based rasterization later (M12).

// Gaussian 2D structure (input from projection)
struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad)
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy,pad)
    color: vec4<f32>,         // Linear RGB
    opacity_pad: vec4<f32>,   // Opacity [0,1]
    gaussian_idx_pad: vec4<u32>, // Source index
}

// Uniforms
struct RenderParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    pad: u32,
    background: vec4<f32>,    // Background color (r,g,b,pad)
}

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian2D>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// Evaluate 2D Gaussian at a pixel
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

// Alpha blend a single Gaussian
fn blend_gaussian(color_accum: vec3<f32>, transmittance: f32,
                  gauss_color: vec3<f32>, gauss_alpha: f32) -> vec4<f32> {
    let contrib = transmittance * gauss_alpha;
    let new_color = color_accum + contrib * gauss_color;
    let new_transmittance = transmittance * (1.0 - gauss_alpha);
    return vec4<f32>(new_color, new_transmittance);
}

@compute @workgroup_size(16, 16)
fn rasterize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;

    if (px >= params.width || py >= params.height) {
        return;
    }

    let pixel_x = f32(px) + 0.5;
    let pixel_y = f32(py) + 0.5;

    // Alpha blending loop
    var color = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance = 1.0;

    // Loop through all Gaussians (depth-sorted on CPU for now)
    for (var i = 0u; i < params.num_gaussians; i++) {
        let g = gaussians[i];

        // Skip culled Gaussians (marked with negative depth)
        if (g.mean.z < 0.0) {
            continue;
        }

        // Evaluate Gaussian weight at this pixel
        let weight = eval_gaussian_2d(
            g.mean.x, g.mean.y,
            g.cov.x, g.cov.y, g.cov.z,
            pixel_x, pixel_y
        );

        // Compute alpha
        let alpha_raw = g.opacity_pad.x * weight;
        let alpha = min(alpha_raw, 0.99);

        // Skip if negligible contribution
        if (alpha < 1e-4) {
            continue;
        }

        // Blend
        let result = blend_gaussian(color, transmittance, g.color.xyz, alpha);
        color = result.xyz;
        transmittance = result.w;

        // Early termination
        if (transmittance < 1e-4) {
            break;
        }
    }

    // Add background
    color += transmittance * params.background.xyz;

    // Write output (linear RGB)
    let pixel_idx = py * params.width + px;
    output[pixel_idx] = vec4<f32>(color, 1.0);
}
