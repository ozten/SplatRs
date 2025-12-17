//! Renderer edge case tests
//!
//! Tests for edge cases in the rendering pipeline, particularly:
//! - NaN/Inf handling in projected Gaussians
//! - Empty Gaussian lists
//! - Very small/large images

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::full_diff::render_full_color_grads;

fn create_test_camera(width: u32, height: u32) -> Camera {
    Camera::new(
        100.0,                // fx
        100.0,                // fy
        (width / 2) as f32,   // cx
        (height / 2) as f32,  // cy
        width,
        height,
        Matrix3::identity(),  // rotation
        Vector3::zeros(),     // translation
    )
}

fn create_valid_gaussian(position: Vector3<f32>) -> Gaussian {
    Gaussian {
        position,
        scale: Vector3::new(-2.0, -2.0, -2.0), // Log-space, exp(-2) ≈ 0.135
        rotation: UnitQuaternion::identity(),
        opacity: 2.0, // Logit-space, sigmoid(2) ≈ 0.88
        sh_coeffs: [[0.5; 3]; 16], // Neutral gray color
    }
}

#[test]
fn test_render_with_nan_position_gaussian() {
    let camera = create_test_camera(50, 50);

    // Create a Gaussian with NaN position
    let mut gaussian = create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0));
    gaussian.position.x = f32::NAN;

    let gaussians = vec![gaussian];
    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic - NaN Gaussians should be filtered out
    let (img, _d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    // Should return a valid image
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);
}

#[test]
fn test_render_with_inf_depth_gaussian() {
    let camera = create_test_camera(50, 50);

    // Create a Gaussian at infinity depth
    let mut gaussian = create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0));
    gaussian.position.z = f32::INFINITY;

    let gaussians = vec![gaussian];
    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic - Inf Gaussians should be filtered out
    let (img, _d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    // Should return a valid image
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);
}

#[test]
fn test_render_all_gaussians_with_invalid_depth() {
    let camera = create_test_camera(50, 50);

    // Create multiple Gaussians, all with invalid depth
    let gaussians = vec![
        {
            let mut g = create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0));
            g.position.z = f32::NAN;
            g
        },
        {
            let mut g = create_valid_gaussian(Vector3::new(1.0, 0.0, 5.0));
            g.position.z = f32::INFINITY;
            g
        },
        {
            let mut g = create_valid_gaussian(Vector3::new(-1.0, 0.0, 5.0));
            g.position.z = -f32::INFINITY;
            g
        },
    ];

    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic even when all Gaussians are filtered
    let (img, _d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    // Should return a valid image (likely all background color)
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);
}

#[test]
fn test_render_mixed_valid_invalid_gaussians() {
    let camera = create_test_camera(50, 50);

    // Mix of valid and invalid Gaussians
    let gaussians = vec![
        create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0)), // Valid
        {
            let mut g = create_valid_gaussian(Vector3::new(1.0, 0.0, 5.0));
            g.position.z = f32::NAN; // Invalid - will be filtered
            g
        },
        create_valid_gaussian(Vector3::new(0.0, 1.0, 5.0)), // Valid
        {
            let mut g = create_valid_gaussian(Vector3::new(-1.0, 0.0, 5.0));
            g.position.x = f32::INFINITY; // Invalid - will be filtered
            g
        },
        create_valid_gaussian(Vector3::new(0.0, -1.0, 5.0)), // Valid
    ];

    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should render only the valid Gaussians
    let (img, d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    // Should return valid results
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);

    // Should have gradients for all Gaussians (even if some were filtered)
    assert_eq!(d_pos.len(), 5);
}

#[test]
fn test_render_empty_gaussian_list() {
    let camera = create_test_camera(50, 50);
    let gaussians: Vec<Gaussian> = vec![];
    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic with empty Gaussian list
    let (img, d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    // Should return valid results
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);

    // No gradients for empty list
    assert_eq!(d_pos.len(), 0);
}

#[test]
fn test_render_very_small_image() {
    // Test with minimal 1x1 image
    let camera = create_test_camera(1, 1);
    let gaussian = create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0));
    let gaussians = vec![gaussian];
    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 1];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic with tiny image
    let (img, _d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
}

#[test]
fn test_render_with_extreme_scale_gaussian() {
    let camera = create_test_camera(50, 50);

    // Create a Gaussian with very large scale (but not NaN/Inf)
    let mut gaussian = create_valid_gaussian(Vector3::new(0.0, 0.0, 5.0));
    gaussian.scale = Vector3::new(10.0, 10.0, 10.0); // exp(10) = ~22000 - very large!

    let gaussians = vec![gaussian];
    let d_image = vec![Vector3::new(0.0, 0.0, 0.0); 50 * 50];
    let bg = Vector3::new(0.0, 0.0, 0.0);

    // Should not panic - extreme but valid scale
    let (img, _d_pos, _d_scale, _d_rot, _d_opa, _d_sh, _d_bg) = render_full_color_grads(
        &gaussians,
        &camera,
        &d_image,
        &bg,
    );

    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 50);
}
