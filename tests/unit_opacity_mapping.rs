//! Opacity/logit mapping invariants in rendering.

use approx::assert_relative_eq;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{sigmoid, Camera, Gaussian};
use sugar_rs::render::render_full_linear;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_white() -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [1.0 / SH_C0, 1.0 / SH_C0, 1.0 / SH_C0];
    sh
}

#[test]
fn test_opacity_logit_maps_to_center_pixel_intensity() {
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

    let g = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0, // sigmoid(0) = 0.5
        sh_constant_white(),
    );

    let out = render_full_linear(&[g], &camera, &Vector3::zeros());
    let center = out[(3 * camera.width + 3) as usize];

    let expected = sigmoid(0.0);
    assert_relative_eq!(center.x, expected, epsilon = 1e-3);
    assert_relative_eq!(center.y, expected, epsilon = 1e-3);
    assert_relative_eq!(center.z, expected, epsilon = 1e-3);
}

#[test]
fn test_opacity_is_clamped_below_one() {
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

    let g = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        50.0, // sigmoid(50) ~ 1.0, but renderer clamps alpha to 0.99
        sh_constant_white(),
    );

    let out = render_full_linear(&[g], &camera, &Vector3::zeros());
    let center = out[(3 * camera.width + 3) as usize];

    assert!(center.x < 0.999, "alpha clamp expected (<1.0)");
    assert!(center.x > 0.98, "alpha clamp expected (~0.99)");
}
