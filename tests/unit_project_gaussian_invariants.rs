//! Invariants for the full 3D->2D projection path via rendering.
//!
//! This uses a single Gaussian and compares a predicted off-center pixel value
//! against the renderer output. It exercises:
//! - world->camera mean
//! - 3D covariance reconstruction
//! - Σ₂d projection via Jacobian
//! - 2D Gaussian evaluation

use approx::assert_relative_eq;
use nalgebra::{Matrix2, Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{perspective_jacobian, sigmoid, Camera, Gaussian};
use sugar_rs::diff::covariance_grad::project_covariance_2d;
use sugar_rs::render::render_full_linear;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

fn gaussian_weight_at_pixel(mean_px: Vector3<f32>, cov_2d: Matrix2<f32>, px: i32, py: i32) -> f32 {
    let eps = 1e-6;
    let cov_xx = cov_2d[(0, 0)] + eps;
    let cov_xy = cov_2d[(0, 1)];
    let cov_yy = cov_2d[(1, 1)] + eps;

    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    let inv_xx = cov_yy / det;
    let inv_xy = -cov_xy / det;
    let inv_yy = cov_xx / det;

    let pixel_x = px as f32 + 0.5;
    let pixel_y = py as f32 + 0.5;
    let dx = pixel_x - mean_px.x;
    let dy = pixel_y - mean_px.y;

    let quad_form = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
    (-0.5 * quad_form).exp()
}

#[test]
fn test_project_gaussian_matches_expected_off_center_value() {
    let camera = Camera::new(
        4.0,
        4.0,
        3.5,
        3.5,
        8,
        8,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let log_scale = Vector3::new(-1.5, -2.0, -1.0);
    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        log_scale,
        UnitQuaternion::identity(),
        -1.0, // sigmoid ~ 0.2689
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    let bg = Vector3::zeros();
    let rendered = render_full_linear(&[gaussian.clone()], &camera, &bg);

    // Compute expected mean and Σ₂d.
    let mean_cam = camera.world_to_camera(&gaussian.position);
    let mean_px = camera.project(&mean_cam).expect("mean should be in front of camera");
    let mean_px = Vector3::new(mean_px.x, mean_px.y, mean_cam.z);

    let j = perspective_jacobian(&mean_cam, camera.fx, camera.fy);
    let sigma_2d = project_covariance_2d(
        &camera.rotation,
        &j,
        &Matrix3::identity(),
        &log_scale,
    );

    // Use a pixel one step to the right of the mean.
    let px = 4;
    let py = 3;
    let weight = gaussian_weight_at_pixel(mean_px, sigma_2d, px, py);
    let alpha = sigmoid(gaussian.opacity) * weight;

    let out_px = rendered[(py as u32 * camera.width + px as u32) as usize];
    assert_relative_eq!(out_px.x, alpha, epsilon = 1e-3);
    assert_relative_eq!(out_px.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(out_px.z, 0.0, epsilon = 1e-6);
}
