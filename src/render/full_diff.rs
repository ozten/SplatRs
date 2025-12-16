//! Differentiable (CPU) renderer pieces for M7.
//!
//! This file intentionally implements a *minimal* backward pass that only
//! computes gradients w.r.t. the Gaussian color (SH coefficients), keeping all
//! geometry fixed.
//!
//! Why start here?
//! - It validates the full loss -> pixel gradients -> blend -> per-Gaussian color
//!   gradients pipeline end-to-end.
//! - It is computationally simpler and more stable than immediately optimizing
//!   position/scale/rotation.
//!
//! Next steps (not yet implemented here):
//! - Backprop into alpha (opacity + 2D Gaussian weight)
//! - Backprop into 2D covariance / mean (projection math)
//! - Full training loop (M7+)

use crate::core::{Camera, Gaussian, Gaussian2D};
use crate::diff::blend_grad::{blend_backward_with_bg, blend_forward_with_bg};
use image::RgbImage;
use nalgebra::{Matrix2, Vector3};

fn project_gaussian(gaussian: &Gaussian, camera: &Camera, gaussian_idx: usize) -> Option<Gaussian2D> {
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
        color: crate::core::evaluate_sh(&gaussian.sh_coeffs, &camera.view_direction(&gaussian.position)),
        opacity: crate::core::sigmoid(gaussian.opacity),
        gaussian_idx,
    })
}

struct Prepared {
    mean_x: f32,
    mean_y: f32,
    inv_xx: f32,
    inv_xy: f32,
    inv_yy: f32,
    opacity: f32,
    color: Vector3<f32>,
    gaussian_idx: usize,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

fn prepare(projected: &[Gaussian2D]) -> Vec<Prepared> {
    projected
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
                gaussian_idx: g.gaussian_idx,
                min_x,
                max_x,
                min_y,
                max_y,
            }
        })
        .collect()
}

/// Render the image and compute gradients w.r.t. Gaussian RGB color.
///
/// Inputs:
/// - `d_image`: per-pixel upstream gradient dL/d(pixel_rgb) in [0,1] linear space
///
/// Returns:
/// - rendered image (RGB8)
/// - per-Gaussian color gradients (accumulated over pixels)
pub fn render_full_color_grads(
    gaussians: &[Gaussian],
    camera: &Camera,
    d_image: &[Vector3<f32>],
    bg: &Vector3<f32>,
) -> (RgbImage, Vec<Vector3<f32>>, Vector3<f32>) {
    let width = camera.width as i32;
    let height = camera.height as i32;
    assert_eq!(d_image.len(), (width * height) as usize);

    // Project and sort front-to-back.
    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();
    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());

    let prepared = prepare(&projected);

    let mut img = RgbImage::new(camera.width, camera.height);
    let mut d_colors = vec![Vector3::<f32>::zeros(); gaussians.len()];
    let mut d_bg = Vector3::<f32>::zeros();

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            // Gather contributing gaussians in depth order for this pixel.
            let mut alphas: Vec<f32> = Vec::new();
            let mut colors: Vec<Vector3<f32>> = Vec::new();
            let mut indices: Vec<usize> = Vec::new();

            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form =
                    g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();

                let alpha = (g.opacity * weight).min(0.99);
                if alpha < 1e-4 {
                    continue;
                }

                alphas.push(alpha);
                colors.push(g.color);
                indices.push(g.gaussian_idx);
            }

            let forward = blend_forward_with_bg(&alphas, &colors, bg);
            let out = forward.out;

            // Write output pixel.
            let r = (out.x * 255.0).clamp(0.0, 255.0) as u8;
            let g = (out.y * 255.0).clamp(0.0, 255.0) as u8;
            let b = (out.z * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(px as u32, py as u32, image::Rgb([r, g, b]));

            // Backward for this pixel: accumulate dL/d(color_i) only.
            let upstream = &d_image[(py * width + px) as usize];
            let grads = blend_backward_with_bg(&alphas, &colors, &forward, bg, upstream);

            for (k, &gi) in indices.iter().enumerate() {
                d_colors[gi] += grads.d_colors[k];
            }
            d_bg += grads.d_bg;
        }
    }

    (img, d_colors, d_bg)
}

/// Forward render that returns linear RGB pixels in [0,1] (no quantization).
pub fn render_full_linear(gaussians: &[Gaussian], camera: &Camera, bg: &Vector3<f32>) -> Vec<Vector3<f32>> {
    let width = camera.width as i32;
    let height = camera.height as i32;
    let mut out = vec![Vector3::<f32>::zeros(); (width * height) as usize];

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();
    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected);

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut alphas: Vec<f32> = Vec::new();
            let mut colors: Vec<Vector3<f32>> = Vec::new();

            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form =
                    g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha < 1e-4 {
                    continue;
                }

                alphas.push(alpha);
                colors.push(g.color);
            }

            out[(py * width + px) as usize] = blend_forward_with_bg(&alphas, &colors, bg).out;
        }
    }

    out
}

/// Helper: compute L2 loss gradient w.r.t. rendered pixel colors.
///
/// `rendered` and `target` are linear RGB in [0,1].
pub fn l2_image_grad(rendered: &[Vector3<f32>], target: &[Vector3<f32>]) -> (f32, Vec<Vector3<f32>>) {
    assert_eq!(rendered.len(), target.len());
    let n = rendered.len() as f32;
    let mut loss = 0.0f32;
    let mut d = vec![Vector3::<f32>::zeros(); rendered.len()];

    for i in 0..rendered.len() {
        let diff = rendered[i] - target[i];
        loss += diff.dot(&diff);
        d[i] = diff * (2.0 / n);
    }
    (loss / n, d)
}

/// Convert an `RgbImage` to linear [0,1] `Vector3` pixels.
pub fn rgb8_to_linear_vec(img: &RgbImage) -> Vec<Vector3<f32>> {
    let mut out = Vec::with_capacity((img.width() * img.height()) as usize);
    for p in img.pixels() {
        out.push(Vector3::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        ));
    }
    out
}

/// Downsample an image to match a camera resolution (nearest neighbor, for simplicity).
pub fn downsample_rgb_nearest(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    let mut out = RgbImage::new(width, height);
    let sx = img.width() as f32 / width as f32;
    let sy = img.height() as f32 / height as f32;
    for y in 0..height {
        for x in 0..width {
            let src_x = (x as f32 * sx).floor().clamp(0.0, (img.width() - 1) as f32) as u32;
            let src_y = (y as f32 * sy).floor().clamp(0.0, (img.height() - 1) as f32) as u32;
            let p = *img.get_pixel(src_x, src_y);
            out.put_pixel(x, y, p);
        }
    }
    out
}

/// Debug: draw projected Gaussian means as colored dots on top of a target image.
///
/// Uses the same projection and Gaussian subset as the renderer.
pub fn debug_overlay_means(target: &RgbImage, gaussians: &[Gaussian], camera: &Camera, radius_px: i32) -> RgbImage {
    let width = camera.width as i32;
    let height = camera.height as i32;
    assert_eq!(target.width() as i32, width);
    assert_eq!(target.height() as i32, height);

    // Project and sort (order doesn't matter for dots, but reuse existing path).
    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();
    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());

    let mut out = target.clone();
    for g in &projected {
        let cx = g.mean.x.round() as i32;
        let cy = g.mean.y.round() as i32;
        let color = [
            (g.color.x * 255.0).clamp(0.0, 255.0) as u8,
            (g.color.y * 255.0).clamp(0.0, 255.0) as u8,
            (g.color.z * 255.0).clamp(0.0, 255.0) as u8,
        ];

        for dy in -radius_px..=radius_px {
            for dx in -radius_px..=radius_px {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || x >= width || y < 0 || y >= height {
                    continue;
                }
                if dx * dx + dy * dy > radius_px * radius_px {
                    continue;
                }
                out.put_pixel(x as u32, y as u32, image::Rgb(color));
            }
        }
    }
    out
}

/// Debug: compute a per-pixel coverage mask (whether any Gaussian contributes).
///
/// The output is a grayscale image where 0=uncovered, 255=covered.
pub fn debug_coverage_mask(gaussians: &[Gaussian], camera: &Camera) -> RgbImage {
    let width = camera.width as i32;
    let height = camera.height as i32;

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();
    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected);

    let mut img = RgbImage::new(camera.width, camera.height);

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut covered = false;
            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form =
                    g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha >= 1e-4 {
                    covered = true;
                    break;
                }
            }

            let v = if covered { 255u8 } else { 0u8 };
            img.put_pixel(px as u32, py as u32, image::Rgb([v, v, v]));
        }
    }

    img
}
