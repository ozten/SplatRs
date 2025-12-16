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

fn covariance_2d_grad_j(
    j: &Matrix2x3<f32>,
    a: &Matrix3<f32>,
    d_sigma2d: &Matrix2<f32>,
) -> Matrix2x3<f32> {
    // Σ₂d = J A Jᵀ
    // Let G = dL/dΣ₂d. Then:
    // dL/dJ = (G + Gᵀ) J A   (when A is symmetric; if not, this still matches using A + Aᵀ)
    let sym_g = d_sigma2d + d_sigma2d.transpose();
    sym_g * j * a
}

fn perspective_jacobian_grad_point(
    point_cam: &Vector3<f32>,
    fx: f32,
    fy: f32,
    d_j: &Matrix2x3<f32>,
) -> Vector3<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;

    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;
    let z_inv3 = z_inv2 * z_inv;

    // J00 = fx / z
    // J02 = -fx * x / z^2
    // J11 = fy / z
    // J12 = -fy * y / z^2
    let d_j00 = d_j[(0, 0)];
    let d_j02 = d_j[(0, 2)];
    let d_j11 = d_j[(1, 1)];
    let d_j12 = d_j[(1, 2)];

    let d_x = d_j02 * (-fx * z_inv2);
    let d_y = d_j12 * (-fy * z_inv2);
    let d_z = d_j00 * (-fx * z_inv2)
        + d_j02 * (2.0 * fx * x * z_inv3)
        + d_j11 * (-fy * z_inv2)
        + d_j12 * (2.0 * fy * y * z_inv3);

    Vector3::new(d_x, d_y, d_z)
}

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

/// Gradient of the projected 2D covariance w.r.t. the camera-space point via the perspective Jacobian.
///
/// This treats `camera_rotation` (W) and `gaussian_rotation` (R) and `log_scale` (s) as constants.
/// Only the Jacobian J(point_cam) changes with the point.
pub fn project_covariance_2d_grad_point_cam(
    point_cam: &Vector3<f32>,
    fx: f32,
    fy: f32,
    camera_rotation: &Matrix3<f32>,
    gaussian_rotation: &Matrix3<f32>,
    log_scale: &Vector3<f32>,
    d_sigma2d: &Matrix2<f32>,
) -> Vector3<f32> {
    let j = crate::core::perspective_jacobian(point_cam, fx, fy);

    let v = Vector3::new(
        (2.0 * log_scale.x).exp(),
        (2.0 * log_scale.y).exp(),
        (2.0 * log_scale.z).exp(),
    );
    let d = Matrix3::from_diagonal(&v);
    let sigma = gaussian_rotation * d * gaussian_rotation.transpose();
    let sigma_cam = camera_rotation * sigma * camera_rotation.transpose();

    let d_j = covariance_2d_grad_j(&j, &sigma_cam, d_sigma2d);
    perspective_jacobian_grad_point(point_cam, fx, fy, &d_j)
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

/// Gradient of the projected 2D covariance w.r.t. a *local* SO(3) rotation vector ω
/// applied on the left: `R(ω) = exp([ω]×) R0`.
///
/// This returns dL/dω at ω = 0 (i.e. around the provided `gaussian_rotation_r0`).
///
/// This is useful for gradient checking the rotation→covariance path without
/// dealing with quaternion normalization constraints yet.
pub fn project_covariance_2d_grad_rotation_vector_at_r0(
    camera_rotation: &Matrix3<f32>,
    jacobian: &Matrix2x3<f32>,
    gaussian_rotation_r0: &Matrix3<f32>,
    log_scale: &Vector3<f32>,
    d_sigma2d: &Matrix2<f32>,
) -> Vector3<f32> {
    let v = Vector3::new(
        (2.0 * log_scale.x).exp(),
        (2.0 * log_scale.y).exp(),
        (2.0 * log_scale.z).exp(),
    );
    let d = Matrix3::from_diagonal(&v);

    // dΣ_cam and dΣ as in the log-scale gradient.
    let d_sigma_cam: Matrix3<f32> = jacobian.transpose() * d_sigma2d * jacobian;
    let d_sigma = camera_rotation.transpose() * d_sigma_cam * camera_rotation;

    // For Σ = R D Rᵀ and L = <G, Σ>, the gradient w.r.t. R is:
    //   dL/dR = (G + Gᵀ) R D
    // (For symmetric G, this is 2 G R D.)
    let g = d_sigma;
    let g_r = (g + g.transpose()) * gaussian_rotation_r0 * d;

    // Basis skew matrices for ω = (ωx, ωy, ωz).
    let kx = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    let ky = Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
    let kz = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    let d_r_x = kx * gaussian_rotation_r0;
    let d_r_y = ky * gaussian_rotation_r0;
    let d_r_z = kz * gaussian_rotation_r0;

    let grad_x = (g_r.component_mul(&d_r_x)).sum();
    let grad_y = (g_r.component_mul(&d_r_y)).sum();
    let grad_z = (g_r.component_mul(&d_r_z)).sum();

    Vector3::new(grad_x, grad_y, grad_z)
}
