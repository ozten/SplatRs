//! Initialization utilities for Gaussian clouds.
//!
//! Functions to initialize Gaussians from various sources:
//! - COLMAP point clouds
//! - Random initialization
//! - Custom point clouds

use crate::core::{Gaussian, GaussianCloud};
use crate::io::Point3D;
use nalgebra::{UnitQuaternion, Vector3};

/// Initialize Gaussians from COLMAP 3D points.
///
/// Creates one Gaussian per point with:
/// - Position from point location
/// - Color from point RGB (stored in DC SH coefficient)
/// - Identity rotation
/// - Small uniform scale
/// - Full opacity
pub fn init_from_colmap_points(points: &[Point3D]) -> GaussianCloud {
    let gaussians: Vec<Gaussian> = points
        .iter()
        .map(|point| {
            // Position
            let position = point.position;

            // Small uniform scale (log-space, so exp(scale) = actual size)
            // Start with scale that gives ~0.01 unit radius
            let scale = Vector3::new(-4.6, -4.6, -4.6); // exp(-4.6) ≈ 0.01

            // Identity rotation
            let rotation = UnitQuaternion::identity();

            // Full opacity in logit space: inverse_sigmoid(0.9) ≈ 2.2
            let opacity = 2.2;

            // Convert RGB color (0-255) to SH DC coefficient (0-1)
            // For spherical harmonics, the DC coefficient is color / Y_0^0
            // where Y_0^0 = 0.28209479
            let mut sh_coeffs = [[0.0f32; 3]; 16];
            sh_coeffs[0] = [
                (point.color[0] as f32 / 255.0) / 0.28209479,
                (point.color[1] as f32 / 255.0) / 0.28209479,
                (point.color[2] as f32 / 255.0) / 0.28209479,
            ];

            Gaussian::new(position, scale, rotation, opacity, sh_coeffs)
        })
        .collect();

    GaussianCloud::from_gaussians(gaussians)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_from_points() {
        let points = vec![
            Point3D {
                id: 0,
                position: Vector3::new(1.0, 2.0, 3.0),
                color: [255, 128, 64],
                error: 0.1,
            },
            Point3D {
                id: 1,
                position: Vector3::new(4.0, 5.0, 6.0),
                color: [100, 200, 50],
                error: 0.2,
            },
        ];

        let cloud = init_from_colmap_points(&points);

        assert_eq!(cloud.len(), 2);

        // Check first Gaussian
        let g0 = &cloud.gaussians[0];
        assert_eq!(g0.position, Vector3::new(1.0, 2.0, 3.0));
        assert!(g0.scale.x < 0.0); // Should be in log-space

        // Check SH DC component is set
        assert!(g0.sh_coeffs[0][0] > 0.0);
    }
}
