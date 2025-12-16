//! M5 Visual Test: Spherical Harmonics View-Dependent Color
//!
//! Produces images in `test_output/m5_sh_*.png`.
//!
//! This test renders the same single Gaussian from two camera centers placed symmetrically
//! about the scene. We set a non-DC SH coefficient so that color depends on view direction.

use std::path::PathBuf;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use sugar_rs::core::{Camera, Gaussian};
use sugar_rs::render::FullRenderer;

#[test]
fn test_m5_sh_view_dependent_color_changes() {
    let width = 96;
    let height = 96;
    let fx = 120.0;
    let fy = 120.0;
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;

    // Camera centers at (-1,0,0) and (+1,0,0) with identity rotation.
    // With p_cam = R p_world + t and C = -R^T t, we set t = -C.
    let cam_left = Camera::new(
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        Matrix3::identity(),
        Vector3::new(1.0, 0.0, 0.0),
    );
    let cam_right = Camera::new(
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        Matrix3::identity(),
        Vector3::new(-1.0, 0.0, 0.0),
    );

    // One Gaussian in front of the cameras.
    let position = Vector3::new(0.0, 0.0, 4.0);
    let scale = Vector3::new((0.15f32).ln(), (0.15f32).ln(), (0.15f32).ln());
    let rotation = UnitQuaternion::identity();
    let opacity = 2.2;

    // SH coefficients:
    // - DC sets a neutral gray baseline
    // - Add a strong Y_1^1 (index 3) term to the red channel so red varies with view-direction x.
    let mut sh_coeffs = [[0.0f32; 3]; 16];
    let y00 = 0.282_094_791_773_878_14_f32;
    sh_coeffs[0] = [0.5 / y00, 0.5 / y00, 0.5 / y00];
    sh_coeffs[3][0] = 3.0;

    let gaussian = Gaussian::new(position, scale, rotation, opacity, sh_coeffs);

    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let mut renderer = FullRenderer::new();
    let img_left = renderer.render(&[gaussian.clone()], &cam_left);
    let img_right = renderer.render(&[gaussian], &cam_right);

    let left_path = output_dir.join("m5_sh_left.png");
    let right_path = output_dir.join("m5_sh_right.png");
    img_left.save(&left_path).expect("Failed to save m5_sh_left.png");
    img_right.save(&right_path).expect("Failed to save m5_sh_right.png");

    // Assert the red channel near the projected mean differs meaningfully.
    // (The mean shifts slightly because the cameras are translated in X.)
    let mean_left = cam_left.world_to_camera(&position);
    let mean_right = cam_right.world_to_camera(&position);
    let px_left = cam_left.project(&mean_left).expect("Left camera projection failed");
    let px_right = cam_right.project(&mean_right).expect("Right camera projection failed");

    let sample_x_left = (px_left.x.round() as i32).clamp(0, (width - 1) as i32) as u32;
    let sample_y_left = (px_left.y.round() as i32).clamp(0, (height - 1) as i32) as u32;
    let sample_x_right = (px_right.x.round() as i32).clamp(0, (width - 1) as i32) as u32;
    let sample_y_right = (px_right.y.round() as i32).clamp(0, (height - 1) as i32) as u32;

    let p_left = img_left.get_pixel(sample_x_left, sample_y_left);
    let p_right = img_right.get_pixel(sample_x_right, sample_y_right);
    let red_diff = (p_left[0] as i32 - p_right[0] as i32).abs();
    assert!(
        red_diff > 10,
        "Expected view-dependent red shift, got red_diff={red_diff} (left={}, right={}) at left=({},{}), right=({}, {})",
        p_left[0],
        p_right[0],
        sample_x_left,
        sample_y_left,
        sample_x_right,
        sample_y_right
    );
}
