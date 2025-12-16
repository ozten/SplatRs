//! Gaussian representation and cloud data structure.
//!
//! A Gaussian is parameterized by:
//! - Position (mean μ)
//! - Scale (log-space: exp(scale) gives actual scale)
//! - Rotation (quaternion)
//! - Opacity (logit-space: sigmoid(opacity) gives actual opacity)
//! - Spherical harmonics coefficients (view-dependent color)

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// A 3D Gaussian primitive.
///
/// Covariance is stored factorized as scale + rotation for numerical stability:
/// Σ = R · S · S^T · R^T where S = diag(exp(scale))
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gaussian {
    /// Position (mean μ)
    pub position: Vector3<f32>,

    /// Log-space scale (actual scale = exp(scale))
    /// Stored in log-space for unbounded optimization
    pub scale: Vector3<f32>,

    /// Rotation as unit quaternion
    /// Quaternions are more stable than Euler angles or matrices for optimization
    pub rotation: UnitQuaternion<f32>,

    /// Opacity in logit-space (actual opacity = sigmoid(opacity))
    /// Logit-space ensures opacity stays in (0, 1) during optimization
    pub opacity: f32,

    /// Spherical harmonics coefficients for view-dependent color
    /// [RGB × 16 coefficients] for degree-3 SH
    /// Index 0 is DC component (view-independent color)
    pub sh_coeffs: [[f32; 3]; 16],
}

impl Gaussian {
    /// Create a new Gaussian with given parameters.
    pub fn new(
        position: Vector3<f32>,
        scale: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
        opacity: f32,
        sh_coeffs: [[f32; 3]; 16],
    ) -> Self {
        Self {
            position,
            scale,
            rotation,
            opacity,
            sh_coeffs,
        }
    }

    /// Compute the 3D covariance matrix Σ = R · S · S^T · R^T
    ///
    /// The covariance is stored factorized for numerical stability:
    /// - Scale in log-space: actual_scale = exp(self.scale)
    /// - Rotation as unit quaternion
    ///
    /// Reconstruction:
    /// 1. R = quaternion_to_matrix(rotation)
    /// 2. S = diag(exp(scale.x), exp(scale.y), exp(scale.z))
    /// 3. Σ = R · S · S^T · R^T
    pub fn covariance_matrix(&self) -> Matrix3<f32> {
        use crate::core::quaternion_to_matrix;

        // Get rotation matrix from quaternion
        let rotation_matrix = quaternion_to_matrix(&self.rotation);

        // Get scale matrix S (diagonal)
        // Scale is stored in log-space, so exp() to get actual values
        let sx = self.scale.x.exp();
        let sy = self.scale.y.exp();
        let sz = self.scale.z.exp();

        // S · S^T for diagonal matrix is just diag(sx², sy², sz²)
        let s_squared = Matrix3::from_diagonal(&nalgebra::Vector3::new(sx * sx, sy * sy, sz * sz));

        // Σ = R · S · S^T · R^T = R · S² · R^T
        rotation_matrix * s_squared * rotation_matrix.transpose()
    }

    /// Get the actual opacity value (sigmoid of stored logit value)
    pub fn actual_opacity(&self) -> f32 {
        // TODO: Implement sigmoid
        crate::core::sigmoid(self.opacity)
    }

    /// Get the actual scale values (exp of stored log values)
    pub fn actual_scale(&self) -> Vector3<f32> {
        Vector3::new(self.scale.x.exp(), self.scale.y.exp(), self.scale.z.exp())
    }
}

/// A 2D Gaussian after projection to screen space.
///
/// This is the intermediate representation used during rasterization.
#[derive(Clone, Debug)]
pub struct Gaussian2D {
    /// 2D mean position in pixel coordinates
    pub mean: Vector3<f32>, // (x, y, depth) - depth for sorting

    /// 2D covariance matrix (2×2 symmetric)
    /// Stored as 3 values: [cov_xx, cov_xy, cov_yy]
    pub cov: Vector3<f32>,

    /// Color at this Gaussian (after SH evaluation)
    pub color: Vector3<f32>,

    /// Opacity
    pub opacity: f32,

    /// Index of the original 3D Gaussian (for gradient backprop)
    pub gaussian_idx: usize,
}

impl Gaussian2D {
    /// Get the inverse covariance matrix for fast evaluation.
    ///
    /// For a 2×2 symmetric matrix:
    /// inv = 1/det * [cov_yy, -cov_xy]
    ///               [-cov_xy, cov_xx]
    pub fn inverse_covariance(&self) -> (f32, f32, f32) {
        let mut cov_xx = self.cov.x;
        let cov_xy = self.cov.y;
        let mut cov_yy = self.cov.z;

        // det([a b; b c]) = a*c - b*b
        let mut det = cov_xx * cov_yy - cov_xy * cov_xy;

        // Add a small diagonal term if the covariance is near-singular.
        // This keeps the renderer numerically stable and matches common practice
        // in splatting implementations (tiny blur in degenerate cases).
        if det.abs() < 1e-12 {
            let eps = 1e-6;
            cov_xx += eps;
            cov_yy += eps;
            det = cov_xx * cov_yy - cov_xy * cov_xy;
        }

        // inv = 1/det * [ c  -b]
        //               [-b   a]
        let inv_det = 1.0 / det;
        let inv_xx = cov_yy * inv_det;
        let inv_xy = -cov_xy * inv_det;
        let inv_yy = cov_xx * inv_det;

        (inv_xx, inv_xy, inv_yy)
    }

    /// Evaluate the 2D Gaussian at a given pixel position.
    ///
    /// Returns exp(-0.5 * (p - μ)^T Σ^{-1} (p - μ))
    pub fn evaluate_at(&self, pixel: Vector3<f32>) -> f32 {
        let dx = pixel.x - self.mean.x;
        let dy = pixel.y - self.mean.y;

        let (inv_xx, inv_xy, inv_yy) = self.inverse_covariance();

        // For a symmetric 2×2 inverse covariance:
        // q = [dx dy] * [inv_xx inv_xy; inv_xy inv_yy] * [dx; dy]
        //   = inv_xx*dx² + 2*inv_xy*dx*dy + inv_yy*dy²
        let quad_form = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
        (-0.5 * quad_form).exp()
    }
}

/// A collection of Gaussians.
///
/// Currently using Array-of-Structs (AoS) layout for simplicity.
/// Can migrate to Struct-of-Arrays (SoA) later if needed for SIMD/GPU.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian>,
}

impl GaussianCloud {
    /// Create a new empty Gaussian cloud.
    pub fn new() -> Self {
        Self {
            gaussians: Vec::new(),
        }
    }

    /// Create a cloud from a vector of Gaussians.
    pub fn from_gaussians(gaussians: Vec<Gaussian>) -> Self {
        Self { gaussians }
    }

    /// Number of Gaussians in the cloud.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Check if the cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Add a Gaussian to the cloud.
    pub fn push(&mut self, gaussian: Gaussian) {
        self.gaussians.push(gaussian);
    }

    /// Get a reference to the Gaussians.
    pub fn as_slice(&self) -> &[Gaussian] {
        &self.gaussians
    }

    /// Get a mutable reference to the Gaussians.
    pub fn as_mut_slice(&mut self) -> &mut [Gaussian] {
        &mut self.gaussians
    }
}

impl Default for GaussianCloud {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian2d_inverse_covariance_identity() {
        let g = Gaussian2D {
            mean: Vector3::new(0.0, 0.0, 1.0),
            cov: Vector3::new(1.0, 0.0, 1.0),
            color: Vector3::new(1.0, 1.0, 1.0),
            opacity: 1.0,
            gaussian_idx: 0,
        };

        let (inv_xx, inv_xy, inv_yy) = g.inverse_covariance();
        assert_relative_eq!(inv_xx, 1.0, epsilon = 1e-6);
        assert_relative_eq!(inv_xy, 0.0, epsilon = 1e-6);
        assert_relative_eq!(inv_yy, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian2d_evaluate_at_mean_is_one() {
        let g = Gaussian2D {
            mean: Vector3::new(3.0, -2.0, 1.0),
            cov: Vector3::new(2.0, 0.0, 2.0),
            color: Vector3::new(0.0, 0.0, 0.0),
            opacity: 1.0,
            gaussian_idx: 0,
        };

        let v = g.evaluate_at(Vector3::new(3.0, -2.0, 0.0));
        assert_relative_eq!(v, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian2d_evaluate_at_one_sigma() {
        // With Σ = I, moving by 1 in x gives exp(-0.5).
        let g = Gaussian2D {
            mean: Vector3::new(0.0, 0.0, 1.0),
            cov: Vector3::new(1.0, 0.0, 1.0),
            color: Vector3::new(0.0, 0.0, 0.0),
            opacity: 1.0,
            gaussian_idx: 0,
        };

        let v = g.evaluate_at(Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(v, (-0.5f32).exp(), epsilon = 1e-6);
    }
}
