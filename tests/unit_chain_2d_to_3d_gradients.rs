//! Finite-difference checks for chaining 2D grads to 3D parameters.
#![cfg(feature = "gpu")]

use approx::assert_relative_eq;
use nalgebra::{Matrix2, Matrix3, UnitQuaternion, Vector2, Vector3};
use sugar_rs::core::{perspective_jacobian, Camera, Gaussian};
use sugar_rs::diff::covariance_grad::project_covariance_2d;
use sugar_rs::gpu::{chain_2d_to_3d_gradients_cpu, GaussianGradients2D};

fn rotation_from_omega(omega: &Vector3<f32>) -> Matrix3<f32> {
    UnitQuaternion::from_scaled_axis(*omega)
        .to_rotation_matrix()
        .into_inner()
}

fn mean_and_cov(
    camera: &Camera,
    position: &Vector3<f32>,
    log_scale: &Vector3<f32>,
    rotation: &Matrix3<f32>,
) -> (Vector2<f32>, Matrix2<f32>) {
    let mean_cam = camera.world_to_camera(position);
    let mean_px = camera.project(&mean_cam).expect("point should be in front");
    let j = perspective_jacobian(&mean_cam, camera.fx, camera.fy);
    let cov2d = project_covariance_2d(&camera.rotation, &j, rotation, log_scale);
    (mean_px, cov2d)
}

fn loss_from_mean_cov(
    d_mean: &Vector2<f32>,
    d_cov: &Vector3<f32>,
    mean_px: &Vector2<f32>,
    cov2d: &Matrix2<f32>,
) -> f32 {
    let cov_mat = Matrix2::new(d_cov.x, d_cov.y, d_cov.y, d_cov.z);
    d_mean.dot(mean_px) + (cov_mat.component_mul(cov2d)).sum()
}

#[test]
fn test_chain_2d_to_3d_gradients_matches_finite_difference() {
    let camera = Camera::new(
        200.0,
        150.0,
        50.0,
        60.0,
        640,
        480,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let gaussian = Gaussian::new(
        Vector3::new(0.2, -0.1, 2.5),
        Vector3::new(-0.3, 0.2, -0.1),
        UnitQuaternion::identity(),
        0.0,
        [[0.0f32; 3]; 16],
    );

    let mut grads_2d = GaussianGradients2D::zeros(1);
    grads_2d.d_mean_px[0] = Vector2::new(0.7, -0.4);
    grads_2d.d_cov_2d[0] = Vector3::new(0.3, -0.1, 0.2);

    let (d_pos, d_log_scale, d_rot_vecs, _d_bg) =
        chain_2d_to_3d_gradients_cpu(&grads_2d, &[gaussian.clone()], &camera);

    let eps = 2e-3;

    // Position finite differences.
    for axis in 0..3 {
        let mut p_plus = gaussian.position;
        let mut p_minus = gaussian.position;
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

        let (mean_plus, cov_plus) = mean_and_cov(&camera, &p_plus, &gaussian.scale, &Matrix3::identity());
        let (mean_minus, cov_minus) = mean_and_cov(&camera, &p_minus, &gaussian.scale, &Matrix3::identity());
        let l_plus = loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_plus, &cov_plus);
        let l_minus =
            loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_minus, &cov_minus);
        let numeric = (l_plus - l_minus) / (2.0 * eps);

        let analytic = if axis == 0 {
            d_pos[0].x
        } else if axis == 1 {
            d_pos[0].y
        } else {
            d_pos[0].z
        };

        assert_relative_eq!(numeric, analytic, epsilon = 3e-2, max_relative = 2e-3);
    }

    // Log-scale finite differences.
    for axis in 0..3 {
        let mut s_plus = gaussian.scale;
        let mut s_minus = gaussian.scale;
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

        let (mean_plus, cov_plus) = mean_and_cov(&camera, &gaussian.position, &s_plus, &Matrix3::identity());
        let (mean_minus, cov_minus) = mean_and_cov(&camera, &gaussian.position, &s_minus, &Matrix3::identity());
        let l_plus = loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_plus, &cov_plus);
        let l_minus =
            loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_minus, &cov_minus);
        let numeric = (l_plus - l_minus) / (2.0 * eps);

        let analytic = if axis == 0 {
            d_log_scale[0].x
        } else if axis == 1 {
            d_log_scale[0].y
        } else {
            d_log_scale[0].z
        };

        assert_relative_eq!(numeric, analytic, epsilon = 3e-2, max_relative = 2e-3);
    }

    // Rotation vector finite differences (local SO(3) at R0).
    let r0 = Matrix3::identity();
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

        let r_plus = rotation_from_omega(&w_plus) * r0;
        let r_minus = rotation_from_omega(&w_minus) * r0;

        let (mean_plus, cov_plus) = mean_and_cov(&camera, &gaussian.position, &gaussian.scale, &r_plus);
        let (mean_minus, cov_minus) = mean_and_cov(&camera, &gaussian.position, &gaussian.scale, &r_minus);
        let l_plus = loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_plus, &cov_plus);
        let l_minus =
            loss_from_mean_cov(&grads_2d.d_mean_px[0], &grads_2d.d_cov_2d[0], &mean_minus, &cov_minus);
        let numeric = (l_plus - l_minus) / (2.0 * eps);

        let analytic = if axis == 0 {
            d_rot_vecs[0].x
        } else if axis == 1 {
            d_rot_vecs[0].y
        } else {
            d_rot_vecs[0].z
        };

        assert_relative_eq!(numeric, analytic, epsilon = 3e-2, max_relative = 2e-3);
    }
}
