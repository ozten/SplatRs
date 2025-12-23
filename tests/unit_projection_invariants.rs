//! Invariants for projecting 3D covariance to 2D.
//!
//! These catch sign/order mistakes in the projection math.

use nalgebra::{Matrix2, Matrix3, SymmetricEigen, Vector3};
use sugar_rs::core::perspective_jacobian;
use sugar_rs::diff::covariance_grad::project_covariance_2d;

#[test]
fn test_projected_covariance_is_symmetric_and_psd() {
    let camera_rotation = Matrix3::identity();
    let gaussian_rotation = Matrix3::identity();
    let log_scale = Vector3::new(-0.2, 0.1, -0.4);

    let point_cam = Vector3::new(0.2, -0.1, 3.0);
    let j = perspective_jacobian(&point_cam, 200.0, 150.0);
    let sigma_2d: Matrix2<f32> =
        project_covariance_2d(&camera_rotation, &j, &gaussian_rotation, &log_scale);

    let symmetry_error = (sigma_2d - sigma_2d.transpose()).abs().max();
    assert!(symmetry_error < 1e-6, "Σ₂d not symmetric");

    let sigma_sym = 0.5 * (sigma_2d + sigma_2d.transpose());
    let eig = SymmetricEigen::new(sigma_sym);
    for v in eig.eigenvalues.iter() {
        assert!(*v >= -1e-6, "Σ₂d not PSD, eigenvalue {}", v);
    }
}
