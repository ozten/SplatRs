//! Full renderer for M4: Elliptical Gaussians (no SH beyond DC).
//!
//! This extends the M3 "fixed-size circle" renderer by:
//! - Reconstructing the 3D covariance Σ from (scale, rotation)
//! - Projecting Σ to screen space as a 2D covariance Σ₂d
//! - Evaluating the resulting elliptical Gaussian per pixel
//!
//! For now this is a simple (CPU) implementation that prioritizes clarity over speed.

use crate::core::{Camera, Gaussian, Gaussian2D};
use image::{Rgb, RgbImage};
use nalgebra::{Matrix2, Vector3};

fn project_gaussian(
    gaussian: &Gaussian,
    camera: &Camera,
    gaussian_idx: usize,
) -> Option<Gaussian2D> {
    // 1) Transform mean to camera space.
    let mean_cam = camera.world_to_camera(&gaussian.position);
    if mean_cam.z <= 0.0 {
        return None;
    }

    // 2) Project mean to pixel coordinates.
    let mean_px = camera.project(&mean_cam)?;

    // 3) Reconstruct 3D covariance in world space.
    let sigma_world = gaussian.covariance_matrix();

    // 4) Rotate covariance into camera space (translation has no effect on covariance).
    let sigma_cam = camera.rotation * sigma_world * camera.rotation.transpose();

    // 5) Project covariance to 2D via the perspective Jacobian at the mean.
    //
    // Σ₂d = J Σ_cam Jᵀ
    let j = camera.projection_jacobian(&mean_cam);
    let sigma_2d: Matrix2<f32> = j * sigma_cam * j.transpose();

    // 6) Pack symmetric Σ₂d into (xx, xy, yy). Add a tiny diagonal regularizer for stability.
    let eps = 1e-6;
    let cov_xx = sigma_2d[(0, 0)] + eps;
    let cov_xy = sigma_2d[(0, 1)];
    let cov_yy = sigma_2d[(1, 1)] + eps;

    Some(Gaussian2D {
        mean: Vector3::new(mean_px.x, mean_px.y, mean_cam.z),
        cov: Vector3::new(cov_xx, cov_xy, cov_yy),
        color: crate::core::evaluate_sh(
            &gaussian.sh_coeffs,
            &camera.view_direction(&gaussian.position),
        ),
        opacity: crate::core::sigmoid(gaussian.opacity),
        gaussian_idx,
    })
}

/// Simple CPU renderer for M4.
///
/// Renders projected ellipses with front-to-back alpha compositing.
pub struct FullRenderer;

impl FullRenderer {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&mut self, gaussians: &[Gaussian], camera: &Camera) -> RgbImage {
        let mut img = RgbImage::new(camera.width, camera.height);

        // Project all Gaussians to 2D.
        let mut projected: Vec<Gaussian2D> = gaussians
            .iter()
            .enumerate()
            .filter_map(|(i, g)| project_gaussian(g, camera, i))
            .collect();

        // Front-to-back (small z first) for early termination.
        projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());

        // Precompute inverse covariances and a conservative bounding circle radius
        // from the largest eigenvalue of Σ₂d.
        struct Prepared {
            mean_x: f32,
            mean_y: f32,
            inv_xx: f32,
            inv_xy: f32,
            inv_yy: f32,
            opacity: f32,
            color: Vector3<f32>,
            min_x: i32,
            max_x: i32,
            min_y: i32,
            max_y: i32,
        }

        let prepared: Vec<Prepared> = projected
            .iter()
            .map(|g| {
                let (inv_xx, inv_xy, inv_yy) = g.inverse_covariance();

                let a = g.cov.x;
                let b = g.cov.y;
                let c = g.cov.z;
                let trace = a + c;
                let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
                let lambda_max = 0.5 * (trace + disc).max(0.0);

                // 3-sigma conservative bounding circle in pixels.
                let radius = 3.0 * lambda_max.sqrt();
                let min_x = (g.mean.x - radius).floor() as i32;
                let max_x = (g.mean.x + radius).ceil() as i32;
                let min_y = (g.mean.y - radius).floor() as i32;
                let max_y = (g.mean.y + radius).ceil() as i32;

                Prepared {
                    mean_x: g.mean.x,
                    mean_y: g.mean.y,
                    inv_xx,
                    inv_xy,
                    inv_yy,
                    opacity: g.opacity,
                    color: g.color,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                }
            })
            .collect();

        // Rasterize: iterate pixels, consider only gaussians whose bounds include the pixel.
        for py in 0..camera.height as i32 {
            for px in 0..camera.width as i32 {
                let pixel_x = px as f32 + 0.5;
                let pixel_y = py as f32 + 0.5;

                let mut color = Vector3::<f32>::zeros();
                let mut transmittance = 1.0f32;

                for g in &prepared {
                    if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                        continue;
                    }

                    let dx = pixel_x - g.mean_x;
                    let dy = pixel_y - g.mean_y;
                    let quad_form =
                        g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                    let weight = (-0.5 * quad_form).exp();

                    // Alpha contribution at this pixel (clamp for stability, like the paper implementations).
                    let alpha = (g.opacity * weight).min(0.99);
                    if alpha < 1e-4 {
                        continue;
                    }

                    color += transmittance * alpha * g.color;
                    transmittance *= 1.0 - alpha;

                    if transmittance < 1e-3 {
                        break;
                    }
                }

                let r = (color.x * 255.0).clamp(0.0, 255.0) as u8;
                let g = (color.y * 255.0).clamp(0.0, 255.0) as u8;
                let b = (color.z * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(px as u32, py as u32, Rgb([r, g, b]));
            }
        }

        img
    }
}

impl Default for FullRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, UnitQuaternion, Vector3 as Vec3};

    #[test]
    fn test_full_renderer_smoke_single_gaussian() {
        let camera = Camera::new(
            100.0,
            100.0,
            10.0,
            10.0,
            20,
            20,
            Matrix3::identity(),
            Vec3::zeros(),
        );

        // One gaussian at the center, with a moderate size in pixel space.
        let mut sh_coeffs = [[0.0f32; 3]; 16];
        sh_coeffs[0] = [1.0 / 0.28209479, 0.0, 0.0];

        let g = Gaussian::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new((0.1f32).ln(), (0.1f32).ln(), (0.1f32).ln()),
            UnitQuaternion::identity(),
            2.2,
            sh_coeffs,
        );

        let mut renderer = FullRenderer::new();
        let img = renderer.render(&[g], &camera);

        // Center pixel should have some red.
        let p = img.get_pixel(10, 10);
        assert!(p[0] > 0);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
    }
}
