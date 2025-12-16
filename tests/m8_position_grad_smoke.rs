//! M8 Unit Test: smoke-test position gradients from the differentiable renderer.
//!
//! This does not assert "training improves PSNR" (too slow / too dataset-dependent).
//! It asserts that enabling `learn_position` produces finite (non-NaN) position
//! gradients for at least one Gaussian on a small scene.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::full_diff::{render_full_color_grads, render_full_linear};

#[test]
fn test_position_grads_are_finite_for_simple_scene() {
    // Simple camera looking down +Z.
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

    // One gaussian in front of the camera.
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [0.5, 0.2, 0.1];
    let g = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(0.1f32.ln(), 0.1f32.ln(), 0.1f32.ln()),
        UnitQuaternion::identity(),
        0.0, // sigmoid(0)=0.5 opacity
        sh,
    );
    let gaussians = vec![g];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    let rendered = render_full_linear(&gaussians, &camera, &bg);
    // Target is a slightly darker version: encourages non-zero gradients.
    let target: Vec<Vector3<f32>> = rendered.iter().map(|p| *p * 0.9).collect();
    let d_image: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(r, t)| (*r - *t) * 2.0)
        .collect();

    let (_img, _d_color, _d_opacity, d_pos, _d_log_scales, _d_bg) =
        render_full_color_grads(&gaussians, &camera, &d_image, &bg);

    assert_eq!(d_pos.len(), gaussians.len());
    let dp = d_pos[0];
    assert!(dp.x.is_finite() && dp.y.is_finite() && dp.z.is_finite());
}
