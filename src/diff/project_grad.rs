//! Gradients for camera projection and Gaussian projection helpers.
//!
//! This module focuses on the parts of projection that depend on the 3D mean:
//! - Pinhole projection `u = fx * x/z + cx`, `v = fy * y/z + cy`
//! - The perspective Jacobian dependency used in covariance projection (handled in `covariance_grad`)
//!
//! We keep these functions small so they can be composed in tests.

use nalgebra::{Vector2, Vector3};

/// Project a camera-space point to pixel coordinates.
///
/// Assumes `z > 0`.
pub fn project_point(point_cam: &Vector3<f32>, fx: f32, fy: f32, cx: f32, cy: f32) -> Vector2<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;
    Vector2::new(fx * x / z + cx, fy * y / z + cy)
}

/// Gradient of `project_point` w.r.t. `point_cam`, given upstream `d_uv`.
///
/// Returns dL/d[x,y,z].
pub fn project_point_grad_point_cam(point_cam: &Vector3<f32>, fx: f32, fy: f32, d_uv: &Vector2<f32>) -> Vector3<f32> {
    let x = point_cam.x;
    let y = point_cam.y;
    let z = point_cam.z;

    let z_inv = 1.0 / z;
    let z_inv2 = z_inv * z_inv;

    // u = fx * x / z + cx
    // v = fy * y / z + cy
    let du_dx = fx * z_inv;
    let du_dy = 0.0;
    let du_dz = -fx * x * z_inv2;

    let dv_dx = 0.0;
    let dv_dy = fy * z_inv;
    let dv_dz = -fy * y * z_inv2;

    let d_x = d_uv.x * du_dx + d_uv.y * dv_dx;
    let d_y = d_uv.x * du_dy + d_uv.y * dv_dy;
    let d_z = d_uv.x * du_dz + d_uv.y * dv_dz;

    Vector3::new(d_x, d_y, d_z)
}

