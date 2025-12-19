//! GPU-friendly data types for Gaussian splatting.
//!
//! These types are designed to be uploaded directly to GPU buffers:
//! - Flat memory layout (no pointers)
//! - Proper alignment (16-byte for vec3/vec4)
//! - bytemuck Pod + Zeroable traits

use crate::core::Gaussian;

/// GPU representation of a Gaussian.
///
/// Memory layout matches what GPU shaders expect:
/// - Aligned to 16 bytes per vec3/vec4
/// - Total size: ~256 bytes per Gaussian
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianGPU {
    /// Position in world space (x, y, z, padding)
    pub position: [f32; 4],

    /// Log-space scale (x, y, z, padding)
    pub scale: [f32; 4],

    /// Rotation quaternion (x, y, z, w)
    pub rotation: [f32; 4],

    /// Opacity (logit-space) and padding
    pub opacity_pad: [f32; 4],

    /// SH coefficients: 16 RGB triplets = 48 floats
    /// Laid out as: [r0,g0,b0,pad, r1,g1,b1,pad, ...]
    pub sh_coeffs: [[f32; 4]; 16],
}

impl GaussianGPU {
    /// Convert from CPU Gaussian to GPU format.
    pub fn from_gaussian(g: &Gaussian) -> Self {
        let mut sh_coeffs = [[0.0f32; 4]; 16];
        for i in 0..16 {
            sh_coeffs[i][0] = g.sh_coeffs[i][0]; // R
            sh_coeffs[i][1] = g.sh_coeffs[i][1]; // G
            sh_coeffs[i][2] = g.sh_coeffs[i][2]; // B
            sh_coeffs[i][3] = 0.0; // Padding
        }

        Self {
            position: [g.position.x, g.position.y, g.position.z, 0.0],
            scale: [g.scale.x, g.scale.y, g.scale.z, 0.0],
            // CRITICAL: Shader expects (w, x, y, z) at (q.x, q.y, q.z, q.w)
            rotation: [
                g.rotation.w,  // w at q.x
                g.rotation.i,  // x (i) at q.y
                g.rotation.j,  // y (j) at q.z
                g.rotation.k,  // z (k) at q.w
            ],
            opacity_pad: [g.opacity, 0.0, 0.0, 0.0],
            sh_coeffs,
        }
    }
}

/// GPU representation of a projected 2D Gaussian.
///
/// This is the output of the projection shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gaussian2DGPU {
    /// 2D mean in pixel space (x, y, depth, padding)
    pub mean: [f32; 4],

    /// 2D covariance (xx, xy, yy, padding)
    pub cov: [f32; 4],

    /// Color in linear RGB (r, g, b, padding)
    pub color: [f32; 4],

    /// Opacity [0,1] and padding
    pub opacity_pad: [f32; 4],

    /// Source Gaussian index (for debugging) and padding
    pub gaussian_idx_pad: [u32; 4],
}

/// Camera parameters for GPU shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraGPU {
    /// Focal lengths (fx, fy, cx, cy)
    pub focal: [f32; 4],

    /// Image dimensions (width, height, padding, padding)
    pub dims: [u32; 4],

    /// Rotation matrix (row-major, 3×3 with padding)
    pub rotation: [[f32; 4]; 3],

    /// Translation (x, y, z, padding)
    pub translation: [f32; 4],
}

impl CameraGPU {
    /// Convert from CPU Camera to GPU format.
    pub fn from_camera(camera: &crate::core::Camera) -> Self {
        let r = camera.rotation;
        Self {
            focal: [camera.fx, camera.fy, camera.cx, camera.cy],
            dims: [camera.width, camera.height, 0, 0],
            // CRITICAL: WGSL mat3x3 is column-major, so we upload columns (transpose of row-major)
            // Each array below represents one column of the matrix
            rotation: [
                [r[(0, 0)], r[(1, 0)], r[(2, 0)], 0.0],  // Column 0
                [r[(0, 1)], r[(1, 1)], r[(2, 1)], 0.0],  // Column 1
                [r[(0, 2)], r[(1, 2)], r[(2, 2)], 0.0],  // Column 2
            ],
            translation: [camera.translation.x, camera.translation.y, camera.translation.z, 0.0],
        }
    }
}

/// GPU representation of a contribution for backward pass.
///
/// Stores the intermediate values needed for gradient computation:
/// - transmittance (T) at this depth layer
/// - alpha (α) of this Gaussian
/// - index of source Gaussian
///
/// For Phase 1: We use fixed-size allocation per pixel for simplicity.
/// Each pixel reserves space for MAX_CONTRIBUTIONS_PER_PIXEL.
/// Unused slots have gaussian_idx = 0xFFFFFFFF.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContributionGPU {
    /// Transmittance before this Gaussian (for chain rule)
    pub transmittance: f32,

    /// Alpha value of this Gaussian at this pixel
    pub alpha: f32,

    /// Index of source Gaussian (0xFFFFFFFF if unused slot)
    pub gaussian_idx: u32,

    /// Padding to 16-byte alignment
    pub pad: u32,
}

impl ContributionGPU {
    /// Create an empty (unused) contribution slot.
    pub const fn empty() -> Self {
        Self {
            transmittance: 0.0,
            alpha: 0.0,
            gaussian_idx: 0xFFFFFFFF,
            pad: 0,
        }
    }

    /// Check if this slot is used.
    pub fn is_used(&self) -> bool {
        self.gaussian_idx != 0xFFFFFFFF
    }
}

/// Configuration for intermediate storage.
///
/// This determines how much memory to allocate for forward pass intermediates.
/// Conservative default: 16 contributions per pixel is enough for most scenes.
pub const MAX_CONTRIBUTIONS_PER_PIXEL: u32 = 16;

/// GPU representation of gradients for a single Gaussian.
///
/// This matches the CPU gradient structure but in GPU-friendly layout.
/// Used for per-workgroup gradient accumulation during backward pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GradientGPU {
    /// Gradient w.r.t. color (r, g, b, padding)
    pub d_color: [f32; 4],

    /// Gradient w.r.t. opacity logit and padding
    pub d_opacity_logit_pad: [f32; 4],

    /// Gradient w.r.t. 2D mean in pixel space (x, y, padding, padding)
    pub d_mean_px: [f32; 4],

    /// Gradient w.r.t. 2D covariance (xx, xy, yy, padding)
    pub d_cov_2d: [f32; 4],
}

impl GradientGPU {
    /// Create a zero gradient (for initialization).
    pub const fn zero() -> Self {
        Self {
            d_color: [0.0; 4],
            d_opacity_logit_pad: [0.0; 4],
            d_mean_px: [0.0; 4],
            d_cov_2d: [0.0; 4],
        }
    }
}

/// GPU representation of 3D gradients for a single Gaussian.
///
/// This is the output of the projection backward shader.
/// Gradients w.r.t. 3D Gaussian parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gradient3DGPU {
    /// Gradient w.r.t. 3D position (x, y, z, padding)
    pub d_position: [f32; 4],

    /// Gradient w.r.t. log-scale (x, y, z, padding)
    pub d_log_scale: [f32; 4],

    /// Gradient w.r.t. rotation (SO(3) vector or quaternion, w,x,y,z)
    pub d_rotation: [f32; 4],

    /// Gradient w.r.t. SH coefficients (16 RGB triplets = 64 floats)
    pub d_sh: [[f32; 4]; 16],
}

impl Gradient3DGPU {
    /// Create a zero gradient (for initialization).
    pub const fn zero() -> Self {
        Self {
            d_position: [0.0; 4],
            d_log_scale: [0.0; 4],
            d_rotation: [0.0; 4],
            d_sh: [[0.0; 4]; 16],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_gaussian_gpu_size() {
        // Verify alignment and size
        assert_eq!(std::mem::size_of::<GaussianGPU>() % 16, 0);
        println!("GaussianGPU size: {} bytes", std::mem::size_of::<GaussianGPU>());
    }

    #[test]
    fn test_gaussian_gpu_conversion() {
        let g = Gaussian::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            UnitQuaternion::identity(),
            -1.5,
            [[0.5; 3]; 16],
        );

        let gpu = GaussianGPU::from_gaussian(&g);
        assert_eq!(gpu.position[0], 1.0);
        assert_eq!(gpu.scale[1], 0.2);
        assert_eq!(gpu.opacity_pad[0], -1.5);
        assert_eq!(gpu.sh_coeffs[0][0], 0.5);
    }
}
