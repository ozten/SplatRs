//! Unit tests for core math invariants and small, deterministic examples.
//!
//! These tests are designed to be educational: each one checks a property
//! the renderer/optimizer relies on, with simple numbers you can verify by hand.

use approx::assert_relative_eq;
use nalgebra::{Matrix2, Matrix3, SymmetricEigen, UnitQuaternion, Vector3};
use sugar_rs::core::{math::perspective_jacobian, Camera, Gaussian, Gaussian2D};

#[test]
fn test_camera_world_to_camera_identity_rotation() {
    let camera = Camera::new(
        100.0,
        100.0,
        50.0,
        60.0,
        640,
        480,
        Matrix3::identity(),
        Vector3::new(1.0, 2.0, 3.0),
    );
    let point_world = Vector3::new(4.0, 5.0, 6.0);
    let point_camera = camera.world_to_camera(&point_world);

    // With identity rotation, world_to_camera should be point + translation.
    assert_relative_eq!(point_camera.x, 5.0, epsilon = 1e-6);
    assert_relative_eq!(point_camera.y, 7.0, epsilon = 1e-6);
    assert_relative_eq!(point_camera.z, 9.0, epsilon = 1e-6);
}

#[test]
fn test_camera_project_pinhole_example() {
    let camera = Camera::new(
        100.0,
        100.0,
        50.0,
        60.0,
        640,
        480,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let point_camera = Vector3::new(1.0, 2.0, 4.0);

    // u = fx * x / z + cx = 100 * 1 / 4 + 50 = 75
    // v = fy * y / z + cy = 100 * 2 / 4 + 60 = 110
    let pixel = camera.project(&point_camera).expect("point should be in front of camera");
    assert_relative_eq!(pixel.x, 75.0, epsilon = 1e-6);
    assert_relative_eq!(pixel.y, 110.0, epsilon = 1e-6);
}

#[test]
fn test_camera_project_rejects_points_behind_camera() {
    let camera = Camera::new(
        100.0,
        100.0,
        50.0,
        60.0,
        640,
        480,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let point_camera = Vector3::new(0.1, 0.1, -0.5);
    assert!(camera.project(&point_camera).is_none());
}

#[test]
fn test_perspective_jacobian_matches_finite_difference() {
    let fx = 320.0;
    let fy = 240.0;
    let cx = 50.0;
    let cy = 60.0;
    let p = Vector3::new(0.3, -1.2, 2.5);
    let eps = 1e-4;

    let j = perspective_jacobian(&p, fx, fy);

    let project_uv = |pt: Vector3<f32>| -> (f32, f32) {
        let u = fx * pt.x / pt.z + cx;
        let v = fy * pt.y / pt.z + cy;
        (u, v)
    };

    for axis in 0..3 {
        let mut p_plus = p;
        let mut p_minus = p;
        if axis == 0 {
            p_plus.x += eps;
            p_minus.x -= eps;
        } else if axis == 1 {
            p_plus.y += eps;
            p_minus.y -= eps;
        } else {
            p_plus.z += eps;
            p_minus.z -= eps;
        }

        let (u_plus, v_plus) = project_uv(p_plus);
        let (u_minus, v_minus) = project_uv(p_minus);
        let du = (u_plus - u_minus) / (2.0 * eps);
        let dv = (v_plus - v_minus) / (2.0 * eps);

        assert_relative_eq!(du, j[(0, axis)], epsilon = 1e-3);
        assert_relative_eq!(dv, j[(1, axis)], epsilon = 1e-3);
    }
}

#[test]
fn test_gaussian_covariance_symmetry_and_positive_definite() {
    let g = Gaussian::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(-1.0, -2.0, -3.0), // log-scales -> positive values
        UnitQuaternion::from_euler_angles(0.2, -0.4, 0.1),
        0.0,
        [[0.0f32; 3]; 16],
    );
    let cov = g.covariance_matrix();

    // Covariance should be symmetric.
    let symmetry_error = (cov - cov.transpose()).abs().max();
    assert!(symmetry_error < 1e-5, "covariance matrix not symmetric");

    // Covariance should be positive definite (all eigenvalues > 0).
    let cov_sym = 0.5 * (cov + cov.transpose());
    let eig = SymmetricEigen::new(cov_sym);
    for v in eig.eigenvalues.iter() {
        assert!(*v > 0.0, "eigenvalue not positive: {}", v);
    }
}

#[test]
fn test_gaussian2d_inverse_covariance_identity() {
    let g2d = Gaussian2D {
        mean: Vector3::new(10.0, 20.0, 5.0),
        cov: Vector3::new(2.0, 0.3, 1.5), // [a, b, c]
        color: Vector3::zeros(),
        opacity: 1.0,
        gaussian_idx: 0,
    };

    let (inv_xx, inv_xy, inv_yy) = g2d.inverse_covariance();
    let cov = Matrix2::new(g2d.cov.x, g2d.cov.y, g2d.cov.y, g2d.cov.z);
    let inv = Matrix2::new(inv_xx, inv_xy, inv_xy, inv_yy);
    let ident = cov * inv;

    assert_relative_eq!(ident[(0, 0)], 1.0, epsilon = 1e-5);
    assert_relative_eq!(ident[(1, 1)], 1.0, epsilon = 1e-5);
    assert_relative_eq!(ident[(0, 1)], 0.0, epsilon = 1e-5);
    assert_relative_eq!(ident[(1, 0)], 0.0, epsilon = 1e-5);
}

#[test]
fn test_gaussian2d_evaluate_at_mean_is_one() {
    let g2d = Gaussian2D {
        mean: Vector3::new(3.0, -2.0, 1.0),
        cov: Vector3::new(1.0, 0.0, 1.0),
        color: Vector3::zeros(),
        opacity: 1.0,
        gaussian_idx: 0,
    };
    let value = g2d.evaluate_at(g2d.mean);
    assert_relative_eq!(value, 1.0, epsilon = 1e-6);
}
