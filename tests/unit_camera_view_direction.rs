//! View direction invariants for camera math.

use approx::assert_relative_eq;
use nalgebra::{Matrix3, Vector3};
use sugar_rs::core::Camera;

#[test]
fn test_view_direction_identity_camera() {
    let camera = Camera::new(
        100.0,
        100.0,
        50.0,
        50.0,
        640,
        480,
        Matrix3::identity(),
        Vector3::zeros(),
    );
    let point_world = Vector3::new(0.0, 0.0, 1.0);
    let dir = camera.view_direction(&point_world);
    assert_relative_eq!(dir, Vector3::new(0.0, 0.0, -1.0), epsilon = 1e-6);
}
