//! M8 Unit Test: smoke-test rotation-vector gradients from the differentiable renderer.
//!
//! This checks that the renderer produces finite gradients for the Gaussian rotation
//! (SO(3) tangent vector at the current rotation).

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::full_diff::{render_full_color_grads, render_full_linear};

#[test]
fn test_rotation_grads_are_finite_for_simple_scene() {
    let camera = Camera::new(
        100.0,
        100.0,
        32.0,
        32.0,
        64,
        64,
        Matrix3::identity(),
        Vector3::zeros(),
    );

    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [0.5, 0.2, 0.1];
    let g = Gaussian::new(
        Vector3::new(0.05, -0.02, 2.0),
        Vector3::new(0.15f32.ln(), 0.10f32.ln(), 0.12f32.ln()),
        UnitQuaternion::from_euler_angles(0.1, -0.05, 0.02),
        0.0,
        sh,
    );
    let gaussians = vec![g];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    let rendered = render_full_linear(&gaussians, &camera, &bg);
    let target: Vec<Vector3<f32>> = rendered.iter().map(|p| *p * 0.9).collect();
    let d_image: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| (*r - *t) * 2.0)
        .collect();

    let (_img, _d_color, _d_opacity, _d_pos, _d_log_scales, d_rot, _d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_image, &bg);

    assert_eq!(d_rot.len(), gaussians.len());
    let dr = d_rot[0];
    assert!(dr.x.is_finite() && dr.y.is_finite() && dr.z.is_finite());
}

