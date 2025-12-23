//! Unit tests for projection math and gradients.
//!
//! These are tiny finite-difference checks to validate the math in isolation.

use approx::assert_relative_eq;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use sugar_rs::core::perspective_jacobian;
use sugar_rs::diff::{covariance_grad, project_grad};

fn scalar_from_sigma2d(sigma2d: &Matrix2<f32>, weights: &Matrix2<f32>) -> f32 {
    (sigma2d.component_mul(weights)).sum()
}

fn rotation_from_omega(omega: &Vector3<f32>) -> Matrix3<f32> {
    nalgebra::UnitQuaternion::from_scaled_axis(*omega)
        .to_rotation_matrix()
        .into_inner()
}

#[test]
fn test_project_covariance_2d_identity_rotation_matches_jjt() {
    let camera_rotation = Matrix3::identity();
    let gaussian_rotation = Matrix3::identity();
    let log_scale = Vector3::new(0.0, 0.0, 0.0); // exp(2s) = 1

    let fx = 200.0;
    let fy = 150.0;
    let point_cam = Vector3::new(0.2, -0.1, 2.0);
    let j = perspective_jacobian(&point_cam, fx, fy);

    let sigma2d = covariance_grad::project_covariance_2d(
        &camera_rotation,
        &j,
        &gaussian_rotation,
        &log_scale,
    );

    // With identity rotations and unit scale, Σ₂d = J * I * Jᵀ = J * Jᵀ.
    let expected = j * j.transpose();
    for r in 0..2 {
        for c in 0..2 {
            assert_relative_eq!(sigma2d[(r, c)], expected[(r, c)], epsilon = 1e-5);
        }
    }
}

#[test]
fn test_project_point_grad_point_cam_finite_difference() {
    let fx = 300.0;
    let fy = 200.0;
    let cx = 10.0;
    let cy = -5.0;
    let point_cam = Vector3::new(0.3, -0.4, 1.7);
    let d_uv = Vector2::new(0.7, -1.1);

    let analytic = project_grad::project_point_grad_point_cam(&point_cam, fx, fy, &d_uv);

    let eps = 2e-3;
    for axis in 0..3 {
        let mut p_plus = point_cam;
        let mut p_minus = point_cam;
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

        let uv_plus = project_grad::project_point(&p_plus, fx, fy, cx, cy);
        let uv_minus = project_grad::project_point(&p_minus, fx, fy, cx, cy);
        let loss_plus = d_uv.dot(&uv_plus);
        let loss_minus = d_uv.dot(&uv_minus);
        let numeric = (loss_plus - loss_minus) / (2.0 * eps);

        let analytic_component = if axis == 0 {
            analytic.x
        } else if axis == 1 {
            analytic.y
        } else {
            analytic.z
        };

        assert_relative_eq!(
            numeric,
            analytic_component,
            epsilon = 2e-2,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_project_covariance_2d_grad_log_scale_finite_difference() {
    let camera_rotation = Matrix3::identity();
    let gaussian_rotation = Matrix3::identity();
    let log_scale = Vector3::new(-0.2, 0.1, 0.05);

    let fx = 250.0;
    let fy = 180.0;
    let point_cam = Vector3::new(-0.4, 0.6, 2.2);
    let j = perspective_jacobian(&point_cam, fx, fy);

    let weights = Matrix2::new(0.6, -0.2, -0.2, 0.4);

    let analytic = covariance_grad::project_covariance_2d_grad_log_scale(
        &camera_rotation,
        &j,
        &gaussian_rotation,
        &log_scale,
        &weights,
    );

    let eps = 2e-3;
    for axis in 0..3 {
        let mut s_plus = log_scale;
        let mut s_minus = log_scale;
        if axis == 0 {
            s_plus.x += eps;
            s_minus.x -= eps;
        } else if axis == 1 {
            s_plus.y += eps;
            s_minus.y -= eps;
        } else {
            s_plus.z += eps;
            s_minus.z -= eps;
        }

        let sigma_plus =
            covariance_grad::project_covariance_2d(&camera_rotation, &j, &gaussian_rotation, &s_plus);
        let sigma_minus = covariance_grad::project_covariance_2d(
            &camera_rotation,
            &j,
            &gaussian_rotation,
            &s_minus,
        );
        let loss_plus = scalar_from_sigma2d(&sigma_plus, &weights);
        let loss_minus = scalar_from_sigma2d(&sigma_minus, &weights);
        let numeric = (loss_plus - loss_minus) / (2.0 * eps);

        let analytic_component = if axis == 0 {
            analytic.x
        } else if axis == 1 {
            analytic.y
        } else {
            analytic.z
        };

        assert_relative_eq!(
            numeric,
            analytic_component,
            epsilon = 2e-2,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_project_covariance_2d_grad_point_cam_finite_difference() {
    let camera_rotation = Matrix3::identity();
    let gaussian_rotation = Matrix3::identity();
    let log_scale = Vector3::new(-0.3, -0.1, 0.2);

    let fx = 280.0;
    let fy = 190.0;
    let point_cam = Vector3::new(0.2, -0.3, 1.8);
    let weights = Matrix2::new(0.5, 0.1, 0.1, -0.2);

    let analytic = covariance_grad::project_covariance_2d_grad_point_cam(
        &point_cam,
        fx,
        fy,
        &camera_rotation,
        &gaussian_rotation,
        &log_scale,
        &weights,
    );

    let eps = 2e-3;
    for axis in 0..3 {
        let mut p_plus = point_cam;
        let mut p_minus = point_cam;
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

        let j_plus = perspective_jacobian(&p_plus, fx, fy);
        let j_minus = perspective_jacobian(&p_minus, fx, fy);
        let sigma_plus =
            covariance_grad::project_covariance_2d(&camera_rotation, &j_plus, &gaussian_rotation, &log_scale);
        let sigma_minus = covariance_grad::project_covariance_2d(
            &camera_rotation,
            &j_minus,
            &gaussian_rotation,
            &log_scale,
        );
        let loss_plus = scalar_from_sigma2d(&sigma_plus, &weights);
        let loss_minus = scalar_from_sigma2d(&sigma_minus, &weights);
        let numeric = (loss_plus - loss_minus) / (2.0 * eps);

        let analytic_component = if axis == 0 {
            analytic.x
        } else if axis == 1 {
            analytic.y
        } else {
            analytic.z
        };

        assert_relative_eq!(
            numeric,
            analytic_component,
            epsilon = 2e-2,
            max_relative = 1e-3
        );
    }
}

#[test]
fn test_project_covariance_2d_grad_rotation_vector_finite_difference() {
    let camera_rotation = Matrix3::identity();
    let gaussian_rotation_r0 = Matrix3::identity();
    let log_scale = Vector3::new(-0.25, 0.15, -0.05);

    let fx = 240.0;
    let fy = 210.0;
    let point_cam = Vector3::new(0.1, -0.2, 2.3);
    let j = perspective_jacobian(&point_cam, fx, fy);

    let weights = Matrix2::new(0.4, -0.1, -0.1, 0.2);

    let analytic = covariance_grad::project_covariance_2d_grad_rotation_vector_at_r0(
        &camera_rotation,
        &j,
        &gaussian_rotation_r0,
        &log_scale,
        &weights,
    );

    let eps = 2e-3;
    for axis in 0..3 {
        let mut w_plus = Vector3::zeros();
        let mut w_minus = Vector3::zeros();
        if axis == 0 {
            w_plus.x = eps;
            w_minus.x = -eps;
        } else if axis == 1 {
            w_plus.y = eps;
            w_minus.y = -eps;
        } else {
            w_plus.z = eps;
            w_minus.z = -eps;
        }

        let r_plus = rotation_from_omega(&w_plus) * gaussian_rotation_r0;
        let r_minus = rotation_from_omega(&w_minus) * gaussian_rotation_r0;

        let sigma_plus =
            covariance_grad::project_covariance_2d(&camera_rotation, &j, &r_plus, &log_scale);
        let sigma_minus =
            covariance_grad::project_covariance_2d(&camera_rotation, &j, &r_minus, &log_scale);
        let loss_plus = scalar_from_sigma2d(&sigma_plus, &weights);
        let loss_minus = scalar_from_sigma2d(&sigma_minus, &weights);
        let numeric = (loss_plus - loss_minus) / (2.0 * eps);

        let analytic_component = if axis == 0 {
            analytic.x
        } else if axis == 1 {
            analytic.y
        } else {
            analytic.z
        };

        assert_relative_eq!(
            numeric,
            analytic_component,
            epsilon = 2e-2,
            max_relative = 1e-3
        );
    }
}
