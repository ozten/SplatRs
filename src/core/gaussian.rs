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
    pub fn covariance_matrix(&self) -> Matrix3<f32> {
        // TODO: Implement for M3
        // 1. Convert quaternion to rotation matrix R
        // 2. Compute S = diag(exp(scale))
        // 3. Return R · S · S^T · R^T
        unimplemented!("See M3 - covariance reconstruction")
    }

    /// Get the actual opacity value (sigmoid of stored logit value)
    pub fn actual_opacity(&self) -> f32 {
        // TODO: Implement sigmoid
        crate::core::sigmoid(self.opacity)
    }

    /// Get the actual scale values (exp of stored log values)
    pub fn actual_scale(&self) -> Vector3<f32> {
        Vector3::new(
            self.scale.x.exp(),
            self.scale.y.exp(),
            self.scale.z.exp(),
        )
    }
}

/// A 2D Gaussian after projection to screen space.
///
/// This is the intermediate representation used during rasterization.
#[derive(Clone, Debug)]
pub struct Gaussian2D {
    /// 2D mean position in pixel coordinates
    pub mean: Vector3<f32>,  // (x, y, depth) - depth for sorting

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
        // TODO: Implement for M4
        // Returns (inv_xx, inv_xy, inv_yy)
        unimplemented!("See M4 - 2D covariance inverse")
    }

    /// Evaluate the 2D Gaussian at a given pixel position.
    ///
    /// Returns exp(-0.5 * (p - μ)^T Σ^{-1} (p - μ))
    pub fn evaluate_at(&self, pixel: Vector3<f32>) -> f32 {
        // TODO: Implement for M4
        unimplemented!("See M4 - 2D Gaussian evaluation")
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
