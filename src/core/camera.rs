//! Camera model (pinhole camera with intrinsics and extrinsics).
//!
//! Cameras are used to:
//! - Project 3D points to 2D image coordinates
//! - Transform Gaussians from world space to camera space
//! - Compute viewing directions for SH evaluation

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// A pinhole camera with intrinsic and extrinsic parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    // Intrinsic parameters (camera internals)
    /// Focal length in X (pixels)
    pub fx: f32,

    /// Focal length in Y (pixels)
    pub fy: f32,

    /// Principal point X (pixels)
    pub cx: f32,

    /// Principal point Y (pixels)
    pub cy: f32,

    /// Image width (pixels)
    pub width: u32,

    /// Image height (pixels)
    pub height: u32,

    // Extrinsic parameters (camera pose in world)
    /// Rotation from world to camera coordinates
    pub rotation: Matrix3<f32>,

    /// Translation from world to camera coordinates
    pub translation: Vector3<f32>,
}

impl Camera {
    /// Create a new camera with given parameters.
    pub fn new(
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: u32,
        height: u32,
        rotation: Matrix3<f32>,
        translation: Vector3<f32>,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            rotation,
            translation,
        }
    }

    /// Transform a point from world coordinates to camera coordinates.
    ///
    /// p_camera = R * p_world + t
    pub fn world_to_camera(&self, point_world: &Vector3<f32>) -> Vector3<f32> {
        // TODO: Implement for M2
        self.rotation * point_world + self.translation
    }

    /// Project a point in camera coordinates to pixel coordinates.
    ///
    /// Returns None if the point is behind the camera (z <= 0).
    ///
    /// Projection: [u, v] = [fx * x/z + cx, fy * y/z + cy]
    pub fn project(&self, point_camera: &Vector3<f32>) -> Option<Vector2<f32>> {
        // TODO: Implement for M2
        if point_camera.z <= 0.0 {
            return None;
        }

        let x = point_camera.x / point_camera.z;
        let y = point_camera.y / point_camera.z;

        let u = self.fx * x + self.cx;
        let v = self.fy * y + self.cy;

        Some(Vector2::new(u, v))
    }

    /// Project a point from world coordinates directly to pixel coordinates.
    ///
    /// Convenience method combining world_to_camera and project.
    pub fn world_to_pixel(&self, point_world: &Vector3<f32>) -> Option<Vector2<f32>> {
        let point_camera = self.world_to_camera(point_world);
        self.project(&point_camera)
    }

    /// Get the view matrix (world to camera transform as 4×4 matrix).
    ///
    /// Used for certain operations where homogeneous coordinates are convenient.
    pub fn view_matrix(&self) -> Matrix4<f32> {
        // TODO: Implement for M4
        // Construct 4×4 matrix: [R | t]
        //                       [0 | 1]
        unimplemented!("See M4 - view matrix construction")
    }

    /// Get the projection matrix (camera to pixel transform as 4×4 matrix).
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        // TODO: Implement for M4
        // Standard pinhole projection matrix
        unimplemented!("See M4 - projection matrix construction")
    }

    /// Compute the Jacobian of perspective projection at a given point.
    ///
    /// This is needed for projecting 3D covariance to 2D.
    ///
    /// J = ∂[u,v]/∂[x,y,z] evaluated at point_camera
    ///
    /// Returns a 2×3 matrix (but we only need certain elements).
    pub fn projection_jacobian(&self, point_camera: &Vector3<f32>) -> Matrix3<f32> {
        // TODO: Implement for M4 (critical for covariance projection!)
        // This is where the magic happens for splatting
        unimplemented!("See M4 - perspective projection Jacobian")
    }

    /// Get the camera center in world coordinates.
    ///
    /// The camera looks from this point.
    pub fn camera_center(&self) -> Vector3<f32> {
        // Camera center in world: C = -R^T * t
        -self.rotation.transpose() * self.translation
    }

    /// Get the viewing direction for a point in world space.
    ///
    /// Used for spherical harmonics evaluation (view-dependent color).
    pub fn view_direction(&self, point_world: &Vector3<f32>) -> Vector3<f32> {
        let dir = point_world - self.camera_center();
        dir.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_projection() {
        // Simple test camera: identity rotation, zero translation
        let cam = Camera::new(
            100.0, // fx
            100.0, // fy
            50.0,  // cx
            50.0,  // cy
            100,   // width
            100,   // height
            Matrix3::identity(),
            Vector3::zeros(),
        );

        // Point at (1, 0, 2) should project to (100*1/2 + 50, 100*0/2 + 50) = (100, 50)
        let world_point = Vector3::new(1.0, 0.0, 2.0);
        let pixel = cam.world_to_pixel(&world_point).unwrap();

        approx::assert_relative_eq!(pixel.x, 100.0, epsilon = 1e-5);
        approx::assert_relative_eq!(pixel.y, 50.0, epsilon = 1e-5);
    }

    #[test]
    fn test_point_behind_camera() {
        let cam = Camera::new(
            100.0, 100.0, 50.0, 50.0, 100, 100,
            Matrix3::identity(),
            Vector3::zeros(),
        );

        // Point with negative z (behind camera)
        let world_point = Vector3::new(0.0, 0.0, -1.0);
        assert!(cam.world_to_pixel(&world_point).is_none());
    }
}
