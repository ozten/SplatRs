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
    rotation: vec4<f32>,      // Quaternion (i,j,k,w)
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
fn quat_to_matrix(q: vec4<f32>) -> mat3x3<f32> {
    let i = q.x;
    let j = q.y;
    let k = q.z;
    let w = q.w;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(j*j + k*k), 2.0*(i*j - k*w), 2.0*(i*k + j*w)),
        vec3<f32>(2.0*(i*j + k*w), 1.0 - 2.0*(i*i + k*k), 2.0*(j*k - i*w)),
        vec3<f32>(2.0*(i*k - j*w), 2.0*(j*k + i*w), 1.0 - 2.0*(i*i + j*j))
    );
}

// Evaluate SH (simplified: just DC term for now)
fn eval_sh_dc(sh_coeffs: array<vec4<f32>, 16>) -> vec3<f32> {
    let C0 = 0.28209479177387814; // sqrt(1/(4*pi))
    return vec3<f32>(sh_coeffs[0].x, sh_coeffs[0].y, sh_coeffs[0].z) * C0;
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
    // Note: camera.rotation is stored row-major, but mat3x3<f32>() expects columns
    // so we need to transpose to get the correct matrix
    let rot_mat_T = mat3x3<f32>(
        camera.rotation[0].xyz,
        camera.rotation[1].xyz,
        camera.rotation[2].xyz
    );
    let rot_mat = transpose(rot_mat_T);
    let pos_world = g.position.xyz;
    let pos_cam = rot_mat * pos_world + camera.translation.xyz;

    // Cull if behind camera
    if (pos_cam.z <= 0.0) {
        gaussians_out[idx].mean.z = -1.0; // Mark as culled
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

    // 6. Evaluate color (just DC term for now)
    let color = eval_sh_dc(g.sh_coeffs);

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

