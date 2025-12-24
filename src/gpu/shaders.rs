//! WGSL shader modules.
//!
//! This module contains compute shaders for:
//! - Gaussian projection (3D → 2D)
//! - Tile assignment
//! - Rasterization

use wgpu::{Device, ShaderModule};

/// WGSL shader for projecting 3D Gaussians to 2D.
///
/// Input: Array of GaussianGPU structs
/// Output: Array of Gaussian2DGPU structs
pub const PROJECT_SHADER: &str = r#"
// Gaussian 3D structure (matches GaussianGPU)
struct Gaussian3D {
    position: vec4<f32>,      // World space (x,y,z,pad)
    scale: vec4<f32>,         // Log-space (x,y,z,pad)
    rotation: vec4<f32>,      // Quaternion uploaded as (w,i,j,k)
    opacity_pad: vec4<f32>,   // Logit-space opacity
    sh_coeffs: array<vec4<f32>, 16>, // RGB SH coefficients
}

// Gaussian 2D structure (output)
struct Gaussian2D {
    mean: vec4<f32>,          // Pixel space (x,y,depth,pad)
    cov: vec4<f32>,           // 2D covariance (xx,xy,yy,pad)
    color: vec4<f32>,         // Linear RGB
    opacity_pad: vec4<f32>,   // Opacity [0,1]
    gaussian_idx_pad: vec4<u32>, // Source index
}

// Camera parameters
struct Camera {
    focal: vec4<f32>,         // (fx, fy, cx, cy)
    dims: vec4<u32>,          // (width, height, pad, pad)
    rotation: mat3x4<f32>,    // Rotation matrix (row-major)
    translation: vec4<f32>,   // (x, y, z, pad)
}

// Uniforms
@group(0) @binding(0) var<uniform> camera: Camera;

// Input/Output buffers
@group(0) @binding(1) var<storage, read> gaussians_in: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read_write> gaussians_out: array<Gaussian2D>;

// Sigmoid function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Convert quaternion to rotation matrix
// NOTE: Quaternion uploaded as (w, i, j, k) at positions (q.x, q.y, q.z, q.w)
fn quat_to_matrix(q_raw: vec4<f32>) -> mat3x3<f32> {
    // Normalize the quaternion
    let n = length(q_raw);
    let q = q_raw / n;

    // Extract components: uploaded as (w, i, j, k)
    let w = q.x;
    let i = q.y;
    let j = q.z;
    let k = q.w;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(j*j + k*k), 2.0*(i*j - k*w), 2.0*(i*k + j*w)),
        vec3<f32>(2.0*(i*j + k*w), 1.0 - 2.0*(i*i + k*k), 2.0*(j*k - i*w)),
        vec3<f32>(2.0*(i*k - j*w), 2.0*(j*k + i*w), 1.0 - 2.0*(i*i + j*j))
    );
}

// Evaluate full spherical harmonics (degree 0-3, 16 coefficients)
fn eval_sh(sh_coeffs: array<vec4<f32>, 16>, dir: vec3<f32>) -> vec3<f32> {
    // Normalize direction (should already be normalized, but be safe)
    let d = normalize(dir);
    let x = d.x;
    let y = d.y;
    let z = d.z;

    // SH basis constants (matching CPU implementation in src/core/sh.rs)
    let C0 = 0.28209479177387814;
    let C1 = 0.48860251190291992;
    let C2_0 = 1.0925484305920792;
    let C2_1 = 0.31539156525252005;
    let C2_2 = 0.54627421529603959;
    let C3_0 = 0.5900435899266435;
    let C3_1 = 2.890611442640554;
    let C3_2 = 0.4570457994644658;
    let C3_3 = 0.3731763325901154;
    let C3_4 = 1.445305721320277;
    let C3_5 = 0.5900435899266435;

    // Precompute monomials
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    // Compute SH basis functions (16 total)
    var basis: array<f32, 16>;

    // Degree 0
    basis[0] = C0;

    // Degree 1
    basis[1] = C1 * y;
    basis[2] = C1 * z;
    basis[3] = C1 * x;

    // Degree 2
    basis[4] = C2_0 * xy;
    basis[5] = C2_0 * yz;
    basis[6] = C2_1 * (3.0 * z2 - 1.0);
    basis[7] = C2_0 * xz;
    basis[8] = C2_2 * (x2 - y2);

    // Degree 3
    basis[9] = C3_0 * y * (3.0 * x2 - y2);
    basis[10] = C3_1 * xy * z;
    basis[11] = C3_2 * y * (5.0 * z2 - 1.0);
    basis[12] = C3_3 * z * (5.0 * z2 - 3.0);
    basis[13] = C3_2 * x * (5.0 * z2 - 1.0);
    basis[14] = C3_4 * z * (x2 - y2);
    basis[15] = C3_5 * x * (x2 - 3.0 * y2);

    // Accumulate color (dot product of basis with coefficients)
    // Note: Manual unroll required because WGSL requires constant array indices
    var color = vec3<f32>(0.0, 0.0, 0.0);

    color += basis[0] * vec3<f32>(sh_coeffs[0].x, sh_coeffs[0].y, sh_coeffs[0].z);
    color += basis[1] * vec3<f32>(sh_coeffs[1].x, sh_coeffs[1].y, sh_coeffs[1].z);
    color += basis[2] * vec3<f32>(sh_coeffs[2].x, sh_coeffs[2].y, sh_coeffs[2].z);
    color += basis[3] * vec3<f32>(sh_coeffs[3].x, sh_coeffs[3].y, sh_coeffs[3].z);
    color += basis[4] * vec3<f32>(sh_coeffs[4].x, sh_coeffs[4].y, sh_coeffs[4].z);
    color += basis[5] * vec3<f32>(sh_coeffs[5].x, sh_coeffs[5].y, sh_coeffs[5].z);
    color += basis[6] * vec3<f32>(sh_coeffs[6].x, sh_coeffs[6].y, sh_coeffs[6].z);
    color += basis[7] * vec3<f32>(sh_coeffs[7].x, sh_coeffs[7].y, sh_coeffs[7].z);
    color += basis[8] * vec3<f32>(sh_coeffs[8].x, sh_coeffs[8].y, sh_coeffs[8].z);
    color += basis[9] * vec3<f32>(sh_coeffs[9].x, sh_coeffs[9].y, sh_coeffs[9].z);
    color += basis[10] * vec3<f32>(sh_coeffs[10].x, sh_coeffs[10].y, sh_coeffs[10].z);
    color += basis[11] * vec3<f32>(sh_coeffs[11].x, sh_coeffs[11].y, sh_coeffs[11].z);
    color += basis[12] * vec3<f32>(sh_coeffs[12].x, sh_coeffs[12].y, sh_coeffs[12].z);
    color += basis[13] * vec3<f32>(sh_coeffs[13].x, sh_coeffs[13].y, sh_coeffs[13].z);
    color += basis[14] * vec3<f32>(sh_coeffs[14].x, sh_coeffs[14].y, sh_coeffs[14].z);
    color += basis[15] * vec3<f32>(sh_coeffs[15].x, sh_coeffs[15].y, sh_coeffs[15].z);

    // Return unclamped color (clamping happens later in render pass)
    return color;
}

// Main projection kernel
@compute @workgroup_size(256)
fn project_gaussians(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_gaussians = arrayLength(&gaussians_in);

    if (idx >= num_gaussians) {
        return;
    }

    let g = gaussians_in[idx];

    // 1. Transform position to camera space
    // Note: camera.rotation is uploaded as column-major (see types.rs)
    // WGSL mat3x3 constructor takes columns, so no transpose needed
    let rot_mat = mat3x3<f32>(
        camera.rotation[0].xyz,
        camera.rotation[1].xyz,
        camera.rotation[2].xyz
    );
    let pos_world = g.position.xyz;
    let pos_cam = rot_mat * pos_world + camera.translation.xyz;

    // Cull if behind camera OR too close to near plane
    // Near-plane threshold prevents huge splats from divide-by-near-zero in Jacobian
    let NEAR_PLANE: f32 = 0.01;
    if (pos_cam.z <= NEAR_PLANE) {
        // Write sentinel values for all fields to ensure buffer is fully initialized
        gaussians_out[idx].mean = vec4<f32>(0.0, 0.0, -1.0, 0.0); // Mark as culled with z=-1
        gaussians_out[idx].cov = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].opacity_pad = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].gaussian_idx_pad = vec4<u32>(idx, 0u, 0u, 0u);
        return;
    }

    // 2. Project to pixel coordinates
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let cx = camera.focal.z;
    let cy = camera.focal.w;

    let x_px = fx * pos_cam.x / pos_cam.z + cx;
    let y_px = fy * pos_cam.y / pos_cam.z + cy;

    // 3. Compute 3D covariance matrix
    let scale_exp = exp(g.scale.xyz);
    let R = quat_to_matrix(g.rotation);

    // Σ_world = R * S * S^T * R^T
    // For diagonal S, S*S^T = diag(sx^2, sy^2, sz^2)
    let S_sq = scale_exp * scale_exp;
    let sigma_world = R * mat3x3<f32>(
        vec3<f32>(S_sq.x, 0.0, 0.0),
        vec3<f32>(0.0, S_sq.y, 0.0),
        vec3<f32>(0.0, 0.0, S_sq.z)
    ) * transpose(R);

    // 4. Transform covariance to camera space
    // Σ_cam = R * Σ_world * R^T
    let sigma_cam = rot_mat * sigma_world * transpose(rot_mat);

    // 5. Project covariance using Jacobian
    // J = [[fx/z, 0, -fx*x/z^2],
    //      [0, fy/z, -fy*y/z^2]]
    let z = pos_cam.z;
    let x = pos_cam.x;
    let y = pos_cam.y;
    let z2 = z * z;

    let j00 = fx / z;
    let j02 = -fx * x / z2;
    let j11 = fy / z;
    let j12 = -fy * y / z2;

    // Σ_2d = J * Σ_cam * J^T
    // Manual 2x3 * 3x3 * 3x2 multiplication
    let s = sigma_cam;
    let cov_xx = j00*j00*s[0][0] + 2.0*j00*j02*s[0][2] + j02*j02*s[2][2];
    let cov_xy = j00*j11*s[0][1] + j00*j12*s[0][2] + j02*j11*s[1][2] + j02*j12*s[2][2];
    let cov_yy = j11*j11*s[1][1] + 2.0*j11*j12*s[1][2] + j12*j12*s[2][2];

    // Add small epsilon for stability
    let eps = 1e-6;

    // Cull if 2D covariance is degenerate or too large
    // Note: WGSL has no isnan/isinf - use bounds check instead
    let max_cov = max(cov_xx, cov_yy);
    if (max_cov <= 0.0 || max_cov > 1e10) {
        // Degenerate covariance - mark as culled
        gaussians_out[idx].mean = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        gaussians_out[idx].cov = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].opacity_pad = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].gaussian_idx_pad = vec4<u32>(idx, 0u, 0u, 0u);
        return;
    }

    // Cull if radius would exceed screen dimensions (prevents screen-filling splats)
    let radius_sq = 9.0 * max_cov; // 3-sigma radius squared
    let max_screen_dim = max(f32(camera.dims.x), f32(camera.dims.y));
    if (radius_sq > max_screen_dim * max_screen_dim) {
        // Too large - mark as culled
        gaussians_out[idx].mean = vec4<f32>(0.0, 0.0, -1.0, 0.0);
        gaussians_out[idx].cov = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].opacity_pad = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        gaussians_out[idx].gaussian_idx_pad = vec4<u32>(idx, 0u, 0u, 0u);
        return;
    }

    // 6. Evaluate view-dependent color with full SH
    // Compute view direction: FROM Gaussian TO camera (in world space)
    // Camera center in world space: C = -R^T * t
    let rot_mat_transpose = transpose(rot_mat);
    let camera_center_world = -(rot_mat_transpose * camera.translation.xyz);
    let view_dir = normalize(camera_center_world - pos_world);
    let color = eval_sh(g.sh_coeffs, view_dir);

    // 7. Convert opacity from logit to [0,1]
    let opacity = sigmoid(g.opacity_pad.x);

    // Write output
    gaussians_out[idx].mean = vec4<f32>(x_px, y_px, pos_cam.z, 0.0);
    gaussians_out[idx].cov = vec4<f32>(cov_xx + eps, cov_xy, cov_yy + eps, 0.0);
    gaussians_out[idx].color = vec4<f32>(color, 0.0);
    gaussians_out[idx].opacity_pad = vec4<f32>(opacity, 0.0, 0.0, 0.0);
    gaussians_out[idx].gaussian_idx_pad = vec4<u32>(idx, 0u, 0u, 0u);
}
"#;

pub fn create_project_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Project Shader"),
        source: wgpu::ShaderSource::Wgsl(PROJECT_SHADER.into()),
    })
}

/// WGSL shader for rasterizing 2D Gaussians to pixels.
pub const RASTERIZE_SHADER: &str = include_str!("rasterize.wgsl");

pub fn create_rasterize_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Rasterize Shader"),
        source: wgpu::ShaderSource::Wgsl(RASTERIZE_SHADER.into()),
    })
}

/// WGSL shader for backward pass (gradient computation).
pub const BACKWARD_SHADER: &str = include_str!("backward.wgsl");

pub fn create_backward_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Backward Shader"),
        source: wgpu::ShaderSource::Wgsl(BACKWARD_SHADER.into()),
    })
}

/// WGSL shader for projection backward pass (2D→3D gradient computation).
pub const PROJECT_BACKWARD_SHADER: &str = include_str!("project_backward.wgsl");

pub fn create_project_backward_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Project Backward Shader"),
        source: wgpu::ShaderSource::Wgsl(PROJECT_BACKWARD_SHADER.into()),
    })
}

