//! Unit tests for spherical harmonics evaluation invariants.
//!
//! These are small, deterministic checks that teach expected properties:
//! normalization behavior, linearity (unclamped), and clamping behavior.

use approx::assert_relative_eq;
use nalgebra::Vector3;
use sugar_rs::core::{evaluate_sh, evaluate_sh_unclamped, sh_basis};

#[test]
fn test_evaluate_sh_unclamped_is_linear_in_coeffs() {
    let dir = Vector3::new(0.3, -0.4, 0.866_025_4);

    let mut a = [[0.0f32; 3]; 16];
    a[0][0] = 0.2;
    a[3][1] = -0.1;

    let mut b = [[0.0f32; 3]; 16];
    b[6][0] = 0.4;
    b[1][2] = 0.3;

    let fa = evaluate_sh_unclamped(&a, &dir);
    let fb = evaluate_sh_unclamped(&b, &dir);

    let mut c = a;
    for i in 0..16 {
        c[i][0] += b[i][0];
        c[i][1] += b[i][1];
        c[i][2] += b[i][2];
    }
    let fc = evaluate_sh_unclamped(&c, &dir);

    assert_relative_eq!(fc.x, fa.x + fb.x, epsilon = 1e-6);
    assert_relative_eq!(fc.y, fa.y + fb.y, epsilon = 1e-6);
    assert_relative_eq!(fc.z, fa.z + fb.z, epsilon = 1e-6);
}

#[test]
fn test_evaluate_sh_normalizes_direction() {
    let dir_unit = Vector3::new(1.0, 0.0, 0.0);
    let dir_scaled = Vector3::new(2.0, 0.0, 0.0);

    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0] = [0.8, 0.2, 0.1];
    sh_coeffs[3][0] = 0.5; // X-axis term

    let color_unit = evaluate_sh_unclamped(&sh_coeffs, &dir_unit);
    let color_scaled = evaluate_sh_unclamped(&sh_coeffs, &dir_scaled);

    assert_relative_eq!(color_unit, color_scaled, epsilon = 1e-6);
}

#[test]
fn test_evaluate_sh_clamps_output() {
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let basis = sh_basis(&dir);

    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0] = [5.0, -5.0, 0.5];

    let unclamped = evaluate_sh_unclamped(&sh_coeffs, &dir);
    let clamped = evaluate_sh(&sh_coeffs, &dir);

    assert!(unclamped.x > 1.0);
    assert!(unclamped.y < 0.0);
    assert_relative_eq!(unclamped.z, basis[0] * 0.5, epsilon = 1e-6);

    assert_relative_eq!(clamped.x, 1.0, epsilon = 1e-6);
    assert_relative_eq!(clamped.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(clamped.z, unclamped.z, epsilon = 1e-6);
}
