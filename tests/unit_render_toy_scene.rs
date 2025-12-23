//! Toy scene render tests with small, deterministic invariants.
//!
//! These checks avoid pixel-perfect comparisons and instead verify:
//! - Peak intensity at the projected mean
//! - Symmetry around the mean
//! - Radial falloff away from the mean

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::render_full_linear;

fn pixel(out: &[Vector3<f32>], width: u32, x: u32, y: u32) -> Vector3<f32> {
    out[(y * width + x) as usize]
}

#[test]
fn test_toy_scene_single_gaussian_invariants() {
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

    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0][0] = 1.0; // DC red component (scaled by Y_0^0 in SH)

    let gaussian = Gaussian::new(
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(-2.0, -2.0, -2.0),
        UnitQuaternion::identity(),
        0.0, // sigmoid(0) = 0.5 opacity
        sh_coeffs,
    );

    let bg = Vector3::zeros();
    let out = render_full_linear(&[gaussian], &camera, &bg);

    let center = pixel(&out, camera.width, 3, 3).x;
    let right = pixel(&out, camera.width, 4, 3).x;
    let left = pixel(&out, camera.width, 2, 3).x;
    let up = pixel(&out, camera.width, 3, 2).x;
    let down = pixel(&out, camera.width, 3, 4).x;
    let diag = pixel(&out, camera.width, 2, 2).x;

    // Peak at the mean.
    let mut max_val = -1.0;
    let mut max_xy = (0u32, 0u32);
    for y in 0..camera.height {
        for x in 0..camera.width {
            let v = pixel(&out, camera.width, x, y).x;
            if v > max_val {
                max_val = v;
                max_xy = (x, y);
            }
        }
    }
    assert_eq!(max_xy, (3, 3));

    // Symmetry around the mean.
    approx::assert_relative_eq!(left, right, epsilon = 1e-6);
    approx::assert_relative_eq!(up, down, epsilon = 1e-6);

    // Radial falloff away from the mean.
    assert!(center > right);
    assert!(right > diag);
}
