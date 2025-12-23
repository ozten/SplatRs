//! Depth ordering invariant for full renderer (front-to-back compositing).

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_linear;

const SH_C0: f32 = 0.282_094_791_773_878_14;

fn sh_constant_color(rgb: Vector3<f32>) -> [[f32; 3]; 16] {
    let mut sh = [[0.0f32; 3]; 16];
    sh[0] = [rgb.x / SH_C0, rgb.y / SH_C0, rgb.z / SH_C0];
    sh
}

#[test]
fn test_near_gaussian_dominates_center_pixel() {
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

    let near = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0, // high opacity
        sh_constant_color(Vector3::new(0.0, 1.0, 0.0)),
    );

    let far = Gaussian::new(
        Vector3::new(0.0, 0.0, 4.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        2.0,
        sh_constant_color(Vector3::new(1.0, 0.0, 0.0)),
    );

    let out = render_full_linear(&[far, near], &camera, &Vector3::zeros());
    let center = out[(3 * camera.width + 3) as usize];

    assert!(center.y > center.x, "near green should dominate red");
}
