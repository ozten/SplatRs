//! Rasterization invariants for 2D Gaussian evaluation.
//!
//! These focus on shape properties rather than pixel-perfect images.

use approx::assert_relative_eq;
use nalgebra::Vector3;
use sugar_rs::core::Gaussian2D;

#[test]
fn test_gaussian2d_symmetry_about_mean() {
    let g2d = Gaussian2D {
        mean: Vector3::new(10.0, -5.0, 1.0),
        cov: Vector3::new(1.5, 0.0, 1.5),
        color: Vector3::zeros(),
        opacity: 1.0,
        gaussian_idx: 0,
    };

    let p_left = Vector3::new(9.0, -5.0, 0.0);
    let p_right = Vector3::new(11.0, -5.0, 0.0);
    let p_up = Vector3::new(10.0, -6.0, 0.0);
    let p_down = Vector3::new(10.0, -4.0, 0.0);

    assert_relative_eq!(g2d.evaluate_at(p_left), g2d.evaluate_at(p_right), epsilon = 1e-6);
    assert_relative_eq!(g2d.evaluate_at(p_up), g2d.evaluate_at(p_down), epsilon = 1e-6);
}

#[test]
fn test_gaussian2d_radial_falloff_is_monotonic() {
    let g2d = Gaussian2D {
        mean: Vector3::new(0.0, 0.0, 1.0),
        cov: Vector3::new(1.0, 0.0, 1.0), // isotropic
        color: Vector3::zeros(),
        opacity: 1.0,
        gaussian_idx: 0,
    };

    let p0 = Vector3::new(0.0, 0.0, 0.0);
    let p1 = Vector3::new(1.0, 0.0, 0.0);
    let p2 = Vector3::new(2.0, 0.0, 0.0);

    let v0 = g2d.evaluate_at(p0);
    let v1 = g2d.evaluate_at(p1);
    let v2 = g2d.evaluate_at(p2);

    assert!(v0 > v1);
    assert!(v1 > v2);
}

#[test]
fn test_gaussian2d_anisotropy_affects_falloff() {
    let g2d = Gaussian2D {
        mean: Vector3::new(0.0, 0.0, 1.0),
        cov: Vector3::new(4.0, 0.0, 1.0), // wider in x than y
        color: Vector3::zeros(),
        opacity: 1.0,
        gaussian_idx: 0,
    };

    let px = Vector3::new(2.0, 0.0, 0.0);
    let py = Vector3::new(0.0, 2.0, 0.0);

    // With larger variance in x, decay along x should be slower.
    let vx = g2d.evaluate_at(px);
    let vy = g2d.evaluate_at(py);
    assert!(vx > vy);
}
