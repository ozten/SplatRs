//! Single-step gradient descent should reduce loss on a toy scene.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::Camera;
use sugar_rs::core::Gaussian;
use sugar_rs::render::{render_full_color_grads, render_full_linear};

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

fn l2_loss(rendered: &[Vector3<f32>], target: &[Vector3<f32>]) -> f32 {
    rendered
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).norm_squared())
        .sum::<f32>()
}

#[test]
fn test_single_step_reduces_loss() {
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

    let mut gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0,
        sh_constant_color(Vector3::new(0.6, 0.2, 0.1)),
    );

    let target_gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0,
        sh_constant_color(Vector3::new(0.2, 0.7, 0.1)),
    );

    let bg = Vector3::zeros();
    let target = render_full_linear(&[target_gaussian], &camera, &bg, false);

    let rendered = render_full_linear(&[gaussian.clone()], &camera, &bg, false);
    let loss_before = l2_loss(&rendered, &target);

    let d_pixels: Vec<Vector3<f32>> = rendered
        .iter()
        .zip(target.iter())
        .map(|(a, b)| *a - *b)
        .collect();

    let (_img, d_colors, _d_opacity, _d_pos, _d_scale, _d_rot, _d_bg) =
        render_full_color_grads(&[gaussian.clone()], &camera, &d_pixels, &bg, false);

    let lr = 0.5;
    let d_color = d_colors[0];
    gaussian.sh_coeffs[0][0] -= lr * d_color.x * SH_C0;
    gaussian.sh_coeffs[0][1] -= lr * d_color.y * SH_C0;
    gaussian.sh_coeffs[0][2] -= lr * d_color.z * SH_C0;

    let rendered_after = render_full_linear(&[gaussian], &camera, &bg, false);
    let loss_after = l2_loss(&rendered_after, &target);

    assert!(loss_after < loss_before, "loss did not decrease");
}
