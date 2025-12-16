//! Gradients for covariance reconstruction and projection.
//!
//! This focuses on the "critical math path" needed for M4/M6:
//!   scale (log) + rotation -> Σ (3×3) -> Σ_cam -> Σ₂d
//!
//! We start with gradients w.r.t. the log-scale parameters only (holding
//! rotation and the perspective Jacobian fixed). This is enough to validate
//! the core matrix calculus and sets up later extensions for rotation and
//! Jacobian/mean gradients.

use nalgebra::{Matrix2, Matrix2x3, Matrix3, Vector3};

/// Project a 3D covariance (from log-scales + rotation) into a 2D covariance.
///
/// Σ = R diag(exp(2s)) Rᵀ
/// Σ_cam = W Σ Wᵀ
/// Σ₂d = J Σ_cam Jᵀ
pub fn project_covariance_2d(
    camera_rotation: &Matrix3<f32>,
    jacobian: &Matrix2x3<f32>,
    gaussian_rotation: &Matrix3<f32>,
    log_scale: &Vector3<f32>,
) -> Matrix2<f32> {
    let v = Vector3::new(
        (2.0 * log_scale.x).exp(),
        (2.0 * log_scale.y).exp(),
        (2.0 * log_scale.z).exp(),
    );
    let d = Matrix3::from_diagonal(&v);
    let sigma = gaussian_rotation * d * gaussian_rotation.transpose();
    let sigma_cam = camera_rotation * sigma * camera_rotation.transpose();
    jacobian * sigma_cam * jacobian.transpose()
}

/// Gradient of `project_covariance_2d` w.r.t. the log-scales.
///
/// Inputs:
/// - `d_sigma2d`: upstream gradient dL/dΣ₂d (2×2)
///
/// Returns:
/// - dL/d(log_scale) as a Vector3 (x,y,z)
pub fn project_covariance_2d_grad_log_scale(
    camera_rotation: &Matrix3<f32>,
    jacobian: &Matrix2x3<f32>,
    gaussian_rotation: &Matrix3<f32>,
    log_scale: &Vector3<f32>,
    d_sigma2d: &Matrix2<f32>,
) -> Vector3<f32> {
    // Backprop:
    // Σ₂d = J Σ_cam Jᵀ
    // Σ_cam = W Σ Wᵀ
    // Σ = R D Rᵀ, with D = diag(v) and v_i = exp(2 s_i)
    //
    // Let G2 = dL/dΣ₂d. Then:
    // dL/dΣ_cam = Jᵀ G2 J
    // dL/dΣ = Wᵀ (dL/dΣ_cam) W
    // Let M = Rᵀ (dL/dΣ) R. Since D is diagonal:
    // dL/dv_i = M_ii
    // dL/ds_i = dL/dv_i * dv_i/ds_i = M_ii * (2 * exp(2 s_i))

    // dΣ_cam
    let d_sigma_cam: Matrix3<f32> = jacobian.transpose() * d_sigma2d * jacobian;

    // dΣ
    let d_sigma = camera_rotation.transpose() * d_sigma_cam * camera_rotation;

    // M = Rᵀ dΣ R
    let m = gaussian_rotation.transpose() * d_sigma * gaussian_rotation;

    let v = Vector3::new(
        (2.0 * log_scale.x).exp(),
        (2.0 * log_scale.y).exp(),
        (2.0 * log_scale.z).exp(),
    );

    Vector3::new(
        m[(0, 0)] * 2.0 * v.x,
        m[(1, 1)] * 2.0 * v.y,
        m[(2, 2)] * 2.0 * v.z,
    )
}

