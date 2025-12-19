//! Simple renderer for M3: Fixed-size Gaussians.
//!
//! This is a minimal renderer that draws each Gaussian as a fixed-size circle.
//! No covariance projection yet - just validates depth sorting and alpha blending.

use crate::core::color::linear_f32_to_srgb_u8;
use crate::core::{evaluate_sh, sigmoid, Camera, Gaussian};
use image::{Rgb, RgbImage};
use nalgebra::Vector3;

/// Simple rendered Gaussian (2D circle with fixed radius).
#[derive(Clone, Debug)]
struct SimpleGaussian2D {
    /// 2D position in pixels
    x: f32,
    y: f32,

    /// Depth (for sorting)
    depth: f32,

    /// RGB color
    color: [u8; 3],

    /// Opacity (0-1)
    opacity: f32,

    /// Fixed radius in pixels
    radius: f32,
}

/// Simple CPU renderer for M3.
///
/// Renders Gaussians as fixed-size colored circles with depth sorting.
pub struct SimpleRenderer {
    /// Fixed radius for all Gaussians (in pixels)
    pub radius: f32,
}

impl SimpleRenderer {
    /// Create a new simple renderer.
    pub fn new() -> Self {
        Self { radius: 3.0 }
    }

    /// Render Gaussians to an image.
    pub fn render(&mut self, gaussians: &[Gaussian], camera: &Camera) -> RgbImage {
        // Create output image
        let mut img = RgbImage::new(camera.width, camera.height);

        // Project all Gaussians to 2D
        let mut projected: Vec<SimpleGaussian2D> = gaussians
            .iter()
            .filter_map(|g| {
                // Transform to camera space
                let pos_cam = camera.world_to_camera(&g.position);

                // Check if in front of camera
                if pos_cam.z <= 0.0 {
                    return None;
                }

                // Project to pixels
                let pixel = camera.project(&pos_cam)?;

                // Evaluate color (just DC component for now)
                let view_dir = camera.view_direction(&g.position);
                let color_vec = evaluate_sh(&g.sh_coeffs, &view_dir);

                // Convert linear RGB to sRGB for display
                let color = [
                    linear_f32_to_srgb_u8(color_vec.x),
                    linear_f32_to_srgb_u8(color_vec.y),
                    linear_f32_to_srgb_u8(color_vec.z),
                ];

                // Get opacity
                let opacity = sigmoid(g.opacity);

                Some(SimpleGaussian2D {
                    x: pixel.x,
                    y: pixel.y,
                    depth: pos_cam.z,
                    color,
                    opacity,
                    radius: self.radius,
                })
            })
            .collect();

        // Filter out NaN depths to prevent sorting panic
        let original_count = projected.len();
        projected.retain(|g| g.depth.is_finite());
        let filtered_count = original_count - projected.len();
        if filtered_count > 0 {
            eprintln!(
                "[SIMPLE RENDER WARNING] Filtered {} Gaussians with invalid depth values",
                filtered_count
            );
        }

        // Sort by depth (front to back for early termination)
        projected.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

        // Render each pixel
        for py in 0..camera.height {
            for px in 0..camera.width {
                let pixel_x = px as f32 + 0.5;
                let pixel_y = py as f32 + 0.5;

                // Accumulate color via alpha blending
                let mut color = Vector3::<f32>::zeros();
                let mut transmittance = 1.0;

                for g in &projected {
                    // Distance from pixel center to Gaussian center
                    let dx = pixel_x - g.x;
                    let dy = pixel_y - g.y;
                    let dist_sq = dx * dx + dy * dy;

                    // Simple circular falloff (Gaussian-like)
                    // exp(-dist^2 / (2 * sigma^2))
                    // For fixed radius, use sigma = radius / 2
                    let sigma = g.radius / 2.0;
                    let alpha = g.opacity * (-dist_sq / (2.0 * sigma * sigma)).exp();

                    // Skip if contribution is negligible
                    if alpha < 0.001 {
                        continue;
                    }

                    // Accumulate color
                    let g_color = Vector3::new(
                        g.color[0] as f32 / 255.0,
                        g.color[1] as f32 / 255.0,
                        g.color[2] as f32 / 255.0,
                    );

                    color += transmittance * alpha * g_color;
                    transmittance *= 1.0 - alpha;

                    // Early termination if fully opaque
                    if transmittance < 0.001 {
                        break;
                    }
                }

                // Convert linear RGB to sRGB for display
                let r = linear_f32_to_srgb_u8(color.x);
                let g = linear_f32_to_srgb_u8(color.y);
                let b = linear_f32_to_srgb_u8(color.z);

                img.put_pixel(px, py, Rgb([r, g, b]));
            }
        }

        img
    }
}

impl Default for SimpleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::init::init_from_colmap_points;
    use crate::io::Point3D;
    use nalgebra::Matrix3;

    #[test]
    fn test_simple_render() {
        // Create a simple scene with 3 points
        let points = vec![
            Point3D {
                id: 0,
                position: Vector3::new(0.0, 0.0, 5.0),
                color: [255, 0, 0],
                error: 0.0,
            },
            Point3D {
                id: 1,
                position: Vector3::new(1.0, 0.0, 5.0),
                color: [0, 255, 0],
                error: 0.0,
            },
            Point3D {
                id: 2,
                position: Vector3::new(0.0, 1.0, 5.0),
                color: [0, 0, 255],
                error: 0.0,
            },
        ];

        let cloud = init_from_colmap_points(&points);

        // Simple camera
        let camera = Camera::new(
            100.0,
            100.0,
            50.0,
            50.0,
            100,
            100,
            Matrix3::identity(),
            Vector3::zeros(),
        );

        // Render
        let mut renderer = SimpleRenderer::new();
        let img = renderer.render(&cloud.gaussians, &camera);

        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);

        // Center pixel should have some color (red from first point)
        let center_pixel = img.get_pixel(50, 50);
        assert!(center_pixel[0] > 0, "Should have red component");
    }
}
