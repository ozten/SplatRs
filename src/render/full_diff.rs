//! Differentiable (CPU) renderer pieces for M7/M8.
//!
//! This file intentionally implements a *minimal* backward pass that only
//! computes gradients w.r.t. Gaussian color (SH coefficients) and opacity,
//! keeping all geometry fixed.
//!
//! Why start here?
//! - It validates the full loss -> pixel gradients -> blend -> per-Gaussian color
//!   gradients pipeline end-to-end.
//! - It is computationally simpler and more stable than immediately optimizing
//!   position/scale/rotation.
//!
//! Next steps (not yet implemented here):
//! - Backprop into 2D covariance / mean (projection math)
//! - Full training loop (M7+)

use crate::core::{Camera, Gaussian, Gaussian2D};
use crate::diff::blend_grad::{blend_backward_with_bg, blend_forward_with_bg};
use crate::diff::covariance_grad::{
    project_covariance_2d_grad_log_scale, project_covariance_2d_grad_point_cam,
};
use crate::diff::gaussian2d_grad::gaussian2d_evaluate_with_grads;
use crate::diff::project_grad::project_point_grad_point_cam;
use image::RgbImage;
use nalgebra::{Matrix2, Vector2, Vector3};

fn srgb_u8_to_linear_f32(u: u8) -> f32 {
    let cs = (u as f32) / 255.0;
    if cs <= 0.04045 {
        cs / 12.92
    } else {
        ((cs + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_f32_to_srgb_u8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let cs = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (cs * 255.0).round().clamp(0.0, 255.0) as u8
}

fn alpha_from_opacity_logit(opacity_logit: f32, weight: f32) -> (f32, f32) {
    let opacity = crate::core::sigmoid(opacity_logit);
    let alpha_raw = opacity * weight;
    let alpha = alpha_raw.min(0.99);
    let d_alpha_d_logit = if alpha_raw < 0.99 {
        weight * opacity * (1.0 - opacity)
    } else {
        0.0
    };
    (alpha, d_alpha_d_logit)
}

fn project_gaussian(
    gaussian: &Gaussian,
    camera: &Camera,
    gaussian_idx: usize,
) -> Option<Gaussian2D> {
    // 1) Transform mean to camera space.
    let mean_cam = camera.world_to_camera(&gaussian.position);

    // Cull if behind camera OR too close to near plane
    // Near-plane threshold prevents huge splats from divide-by-near-zero in Jacobian
    const NEAR_PLANE: f32 = 0.01;
    if mean_cam.z <= NEAR_PLANE {
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

    // 7) Cull if 2D covariance is degenerate or too large
    // Compute approximate radius: r ≈ sqrt(max(cov_xx, cov_yy))
    // This is a conservative approximation (actual max eigenvalue could be larger)
    let max_cov = cov_xx.max(cov_yy);
    if !max_cov.is_finite() || max_cov <= 0.0 {
        return None; // Degenerate covariance
    }

    // Cull if radius would exceed screen dimensions (prevents screen-filling splats)
    let radius_sq = 3.0 * 3.0 * max_cov; // 3-sigma radius squared
    let max_screen_dim = camera.width.max(camera.height) as f32;
    if radius_sq > (max_screen_dim * max_screen_dim) {
        return None; // Too large - would fill screen
    }

    Some(Gaussian2D {
        mean: Vector3::new(mean_px.x, mean_px.y, mean_cam.z),
        cov: Vector3::new(cov_xx, cov_xy, cov_yy),
        color: crate::core::evaluate_sh_unclamped(
            &gaussian.sh_coeffs,
            &camera.view_direction(&gaussian.position),
        ),
        opacity: crate::core::sigmoid(gaussian.opacity),
        gaussian_idx,
    })
}

struct Prepared {
    mean_x: f32,
    mean_y: f32,
    cov_xx: f32,
    cov_xy: f32,
    cov_yy: f32,
    inv_xx: f32,
    inv_xy: f32,
    inv_yy: f32,
    opacity: f32,
    color: Vector3<f32>,
    gaussian_idx: usize,
    point_cam: Vector3<f32>,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

fn prepare(projected: &[Gaussian2D], gaussians: &[Gaussian], camera: &Camera) -> Vec<Prepared> {
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
                cov_xx: g.cov.x,
                cov_xy: g.cov.y,
                cov_yy: g.cov.z,
                inv_xx,
                inv_xy,
                inv_yy,
                opacity: g.opacity,
                color: g.color,
                gaussian_idx: g.gaussian_idx,
                point_cam: camera.world_to_camera(&gaussians[g.gaussian_idx].position),
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
) -> (
    RgbImage,
    Vec<Vector3<f32>>,
    Vec<f32>,
    Vec<Vector3<f32>>,
    Vec<Vector3<f32>>,
    Vec<Vector3<f32>>,
    Vector3<f32>,
) {
    let enable_timing = std::env::var("SUGAR_BACKWARD_TIMING").is_ok();
    let t_total = if enable_timing { Some(std::time::Instant::now()) } else { None };

    let width = camera.width as i32;
    let height = camera.height as i32;
    assert_eq!(d_image.len(), (width * height) as usize);

    // Project and sort front-to-back.
    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();

    // Filter out invalid Gaussians (NaN or inf depth) before sorting
    let original_count = projected.len();
    projected.retain(|g| g.mean.z.is_finite());
    let filtered_count = original_count - projected.len();
    if filtered_count > 0 {
        eprintln!("[CPU WARNING] Filtered {} Gaussians with invalid depth values", filtered_count);
    }

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());

    let prepared = prepare(&projected, gaussians, camera);

    let mut img = RgbImage::new(camera.width, camera.height);

    // Use parallel iteration for gradient computation with thread-local accumulation
    // to eliminate mutex contention (major speedup!).
    use rayon::prelude::*;

    // Thread-local gradient accumulation: each thread gets its own gradient buffers
    // and we reduce at the end (much faster than mutex locking on every pixel!).
    struct ThreadLocalGrads {
        d_colors: Vec<Vector3<f32>>,
        d_opacity_logits: Vec<f32>,
        d_mean_px: Vec<Vector2<f32>>,
        d_cov_2d: Vec<Vector3<f32>>,
        d_bg: Vector3<f32>,
    }

    impl ThreadLocalGrads {
        fn new(num_gaussians: usize) -> Self {
            Self {
                d_colors: vec![Vector3::<f32>::zeros(); num_gaussians],
                d_opacity_logits: vec![0.0f32; num_gaussians],
                d_mean_px: vec![Vector2::<f32>::zeros(); num_gaussians],
                d_cov_2d: vec![Vector3::<f32>::zeros(); num_gaussians],
                d_bg: Vector3::<f32>::zeros(),
            }
        }
    }

    // Parallel iteration over pixels with thread-local gradient accumulation
    let pixels: Vec<(i32, i32)> = (0..height)
        .flat_map(|py| (0..width).map(move |px| (px, py)))
        .collect();

    let thread_grads: Vec<ThreadLocalGrads> = pixels
        .par_iter()
        .fold(
            || ThreadLocalGrads::new(gaussians.len()),
            |mut local_grads, &(px, py)| {
                let pixel_x = px as f32 + 0.5;
                let pixel_y = py as f32 + 0.5;

                // Gather contributing gaussians in depth order for this pixel.
                let mut alphas: Vec<f32> = Vec::new();
                let mut d_alpha_d_logits: Vec<f32> = Vec::new();
                let mut d_alpha_d_weights: Vec<f32> = Vec::new();
                let mut d_weight_d_means: Vec<Vector2<f32>> = Vec::new();
                let mut d_weight_d_covs: Vec<Vector3<f32>> = Vec::new();
                let mut colors: Vec<Vector3<f32>> = Vec::new();
                let mut indices: Vec<usize> = Vec::new();

                for g in &prepared {
                    if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                        continue;
                    }

                    let det = g.cov_xx * g.cov_yy - g.cov_xy * g.cov_xy;
                    if det <= 1e-10 {
                        continue;
                    }

                    let w_grads = gaussian2d_evaluate_with_grads(
                        Vector2::new(g.mean_x, g.mean_y),
                        g.cov_xx,
                        g.cov_xy,
                        g.cov_yy,
                        Vector2::new(pixel_x, pixel_y),
                    );
                    let weight = w_grads.value;

                    let (alpha, d_alpha_d_logit) =
                        alpha_from_opacity_logit(gaussians[g.gaussian_idx].opacity, weight);
                    let opacity = crate::core::sigmoid(gaussians[g.gaussian_idx].opacity);
                    let alpha_raw = opacity * weight;
                    let d_alpha_d_weight = if alpha_raw < 0.99 { opacity } else { 0.0 };
                    if alpha < 1e-4 {
                        continue;
                    }

                    alphas.push(alpha);
                    d_alpha_d_logits.push(d_alpha_d_logit);
                    d_alpha_d_weights.push(d_alpha_d_weight);
                    d_weight_d_means.push(w_grads.d_mean);
                    d_weight_d_covs.push(Vector3::new(
                        w_grads.d_cov_xx,
                        w_grads.d_cov_xy,
                        w_grads.d_cov_yy,
                    ));
                    colors.push(g.color);
                    indices.push(g.gaussian_idx);
                }

                let forward = blend_forward_with_bg(&alphas, &colors, bg);

                // Backward for this pixel
                let upstream = &d_image[(py * width + px) as usize];
                let grads = blend_backward_with_bg(&alphas, &colors, &forward, bg, upstream);

                // Accumulate into thread-local gradients (no locking!)
                for (k, &gi) in indices.iter().enumerate() {
                    local_grads.d_colors[gi] += grads.d_colors[k];
                    local_grads.d_opacity_logits[gi] += grads.d_alphas[k] * d_alpha_d_logits[k];
                    let d_weight = grads.d_alphas[k] * d_alpha_d_weights[k];
                    local_grads.d_mean_px[gi] += d_weight_d_means[k] * d_weight;
                    local_grads.d_cov_2d[gi] += d_weight_d_covs[k] * d_weight;
                }
                local_grads.d_bg += grads.d_bg;

                local_grads
            },
        )
        .collect();

    // Final reduction: combine all thread-local gradients
    let mut d_colors = vec![Vector3::<f32>::zeros(); gaussians.len()];
    let mut d_opacity_logits = vec![0.0f32; gaussians.len()];
    let mut d_mean_px = vec![Vector2::<f32>::zeros(); gaussians.len()];
    let mut d_cov_2d = vec![Vector3::<f32>::zeros(); gaussians.len()];
    let mut d_bg = Vector3::<f32>::zeros();

    for tg in thread_grads {
        for i in 0..gaussians.len() {
            d_colors[i] += tg.d_colors[i];
            d_opacity_logits[i] += tg.d_opacity_logits[i];
            d_mean_px[i] += tg.d_mean_px[i];
            d_cov_2d[i] += tg.d_cov_2d[i];
        }
        d_bg += tg.d_bg;
    }

    // Create output image (fast sequential render)
    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut alphas_img: Vec<f32> = Vec::new();
            let mut colors_img: Vec<Vector3<f32>> = Vec::new();

            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }
                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form = g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha < 1e-4 {
                    continue;
                }
                alphas_img.push(alpha);
                colors_img.push(g.color);
            }

            let out = blend_forward_with_bg(&alphas_img, &colors_img, bg).out;
            img.put_pixel(
                px as u32,
                py as u32,
                image::Rgb([
                    linear_f32_to_srgb_u8(out.x),
                    linear_f32_to_srgb_u8(out.y),
                    linear_f32_to_srgb_u8(out.z),
                ]),
            );
        }
    }

    let mut d_positions = vec![Vector3::<f32>::zeros(); gaussians.len()];
    let mut d_log_scales = vec![Vector3::<f32>::zeros(); gaussians.len()];
    let mut d_rot_vecs = vec![Vector3::<f32>::zeros(); gaussians.len()];
    for g in &prepared {
        let gi = g.gaussian_idx;
        let mut d_point_cam_total = Vector3::<f32>::zeros();

        let d_uv = d_mean_px[gi];
        if d_uv != Vector2::zeros() {
            d_point_cam_total +=
                project_point_grad_point_cam(&g.point_cam, camera.fx, camera.fy, &d_uv);
        }

        let d_cov = d_cov_2d[gi];
        if d_cov != Vector3::zeros() {
            // Backprop through Σ₂d = J(point_cam) Σ_cam Jᵀ into:
            // - point_cam (via Jacobian dependence on depth)
            // - log_scale (via Σ reconstruction)
            // - rotation (via Σ reconstruction)
            let d_sigma2d = Matrix2::new(d_cov.x, d_cov.y, d_cov.y, d_cov.z);

            let gaussian_r = crate::core::quaternion_to_matrix(&gaussians[gi].rotation);
            let log_scale = gaussians[gi].scale;

            d_point_cam_total += project_covariance_2d_grad_point_cam(
                &g.point_cam,
                camera.fx,
                camera.fy,
                &camera.rotation,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );

            let j = camera.projection_jacobian(&g.point_cam);
            d_log_scales[gi] = project_covariance_2d_grad_log_scale(
                &camera.rotation,
                &j,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );

            d_rot_vecs[gi] = crate::diff::covariance_grad::project_covariance_2d_grad_rotation_vector_at_r0(
                &camera.rotation,
                &j,
                &gaussian_r,
                &log_scale,
                &d_sigma2d,
            );
        }

        // point_cam = R * p_world + t  =>  dL/dp_world = R^T * dL/dpoint_cam
        d_positions[gi] = camera.rotation.transpose() * d_point_cam_total;
    }

    (
        img,
        d_colors,
        d_opacity_logits,
        d_positions,
        d_log_scales,
        d_rot_vecs,
        d_bg,
    )
}

/// Forward render that returns linear RGB pixels in [0,1] (no quantization).
pub fn render_full_linear(
    gaussians: &[Gaussian],
    camera: &Camera,
    bg: &Vector3<f32>,
) -> Vec<Vector3<f32>> {
    let width = camera.width as i32;
    let height = camera.height as i32;
    let mut out = vec![Vector3::<f32>::zeros(); (width * height) as usize];

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();

    // Be robust to upstream numerical issues (e.g., NaN depth after training).
    // If we try to sort with NaNs present, `partial_cmp` returns `None` and panics on unwrap.
    let original_count = projected.len();
    projected.retain(|g| g.mean.z.is_finite());
    let filtered_count = original_count - projected.len();
    if filtered_count > 0 {
        eprintln!(
            "[CPU WARNING] Filtered {} Gaussians with invalid depth values",
            filtered_count
        );
    }

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected, gaussians, camera);

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
                let quad_form = g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
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
pub fn l2_image_grad(
    rendered: &[Vector3<f32>],
    target: &[Vector3<f32>],
) -> (f32, Vec<Vector3<f32>>) {
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
            srgb_u8_to_linear_f32(p[0]),
            srgb_u8_to_linear_f32(p[1]),
            srgb_u8_to_linear_f32(p[2]),
        ));
    }
    out
}

/// Convert linear [0,1] RGB pixels to an `RgbImage` (sRGB encoded).
pub fn linear_vec_to_rgb8_img(linear: &[Vector3<f32>], width: u32, height: u32) -> RgbImage {
    assert_eq!(linear.len(), (width * height) as usize);
    let mut img = RgbImage::new(width, height);
    for (i, p) in linear.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(
            x,
            y,
            image::Rgb([
                linear_f32_to_srgb_u8(p.x),
                linear_f32_to_srgb_u8(p.y),
                linear_f32_to_srgb_u8(p.z),
            ]),
        );
    }
    img
}

/// Downsample an image using box filter (averaging blocks of pixels).
///
/// This is used for power-of-2 downsampling (divisor = 2, 4, 8, etc.) and provides
/// perfect anti-aliasing with no artifacts.
///
/// IMPORTANT: Averages in linear color space for correct color blending.
///
/// # Arguments
/// * `img` - Source image
/// * `divisor` - Power-of-2 divisor (2, 4, 8, 16, etc.)
///
/// # Returns
/// Downsampled image with dimensions `(img.width() / divisor, img.height() / divisor)`
pub fn downsample_rgb_box(img: &RgbImage, divisor: u32) -> RgbImage {
    let out_width = (img.width() + divisor - 1) / divisor;
    let out_height = (img.height() + divisor - 1) / divisor;
    let mut out = RgbImage::new(out_width, out_height);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let src_x_start = out_x * divisor;
            let src_y_start = out_y * divisor;
            let src_x_end = (src_x_start + divisor).min(img.width());
            let src_y_end = (src_y_start + divisor).min(img.height());

            // Average all pixels in the block IN LINEAR SPACE
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut count = 0u32;

            for src_y in src_y_start..src_y_end {
                for src_x in src_x_start..src_x_end {
                    let pixel = img.get_pixel(src_x, src_y);
                    // Convert sRGB to linear before averaging
                    sum_r += srgb_u8_to_linear_f32(pixel[0]);
                    sum_g += srgb_u8_to_linear_f32(pixel[1]);
                    sum_b += srgb_u8_to_linear_f32(pixel[2]);
                    count += 1;
                }
            }

            let count_f = count as f32;
            let avg_r_linear = sum_r / count_f;
            let avg_g_linear = sum_g / count_f;
            let avg_b_linear = sum_b / count_f;

            // Convert back to sRGB
            out.put_pixel(
                out_x,
                out_y,
                image::Rgb([
                    linear_f32_to_srgb_u8(avg_r_linear),
                    linear_f32_to_srgb_u8(avg_g_linear),
                    linear_f32_to_srgb_u8(avg_b_linear),
                ]),
            );
        }
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
            let src_y = (y as f32 * sy)
                .floor()
                .clamp(0.0, (img.height() - 1) as f32) as u32;
            let p = *img.get_pixel(src_x, src_y);
            out.put_pixel(x, y, p);
        }
    }
    out
}

/// Downsample an image using bilinear interpolation in linear color space.
///
/// This produces higher quality results than nearest-neighbor for fractional
/// scaling factors, avoiding aliasing artifacts.
pub fn downsample_rgb_bilinear(img: &RgbImage, target_width: u32, target_height: u32) -> RgbImage {
    use crate::core::color::{linear_f32_to_srgb_u8, srgb_u8_to_linear_f32};

    let mut out = RgbImage::new(target_width, target_height);

    let scale_x = img.width() as f32 / target_width as f32;
    let scale_y = img.height() as f32 / target_height as f32;

    for out_y in 0..target_height {
        for out_x in 0..target_width {
            // Map output pixel center to input space
            let src_x = (out_x as f32 + 0.5) * scale_x - 0.5;
            let src_y = (out_y as f32 + 0.5) * scale_y - 0.5;

            // Clamp to valid range
            let src_x = src_x.max(0.0).min(img.width() as f32 - 1.0);
            let src_y = src_y.max(0.0).min(img.height() as f32 - 1.0);

            // Get integer and fractional parts
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(img.width() - 1);
            let y1 = (y0 + 1).min(img.height() - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            // Sample 4 neighbors
            let p00 = img.get_pixel(x0, y0);
            let p10 = img.get_pixel(x1, y0);
            let p01 = img.get_pixel(x0, y1);
            let p11 = img.get_pixel(x1, y1);

            // Bilinear interpolation per channel IN LINEAR SPACE
            let mut result = [0u8; 3];
            for c in 0..3 {
                // Convert sRGB to linear
                let linear_00 = srgb_u8_to_linear_f32(p00[c]);
                let linear_10 = srgb_u8_to_linear_f32(p10[c]);
                let linear_01 = srgb_u8_to_linear_f32(p01[c]);
                let linear_11 = srgb_u8_to_linear_f32(p11[c]);

                // Bilinear interpolation
                let top = linear_00 * (1.0 - fx) + linear_10 * fx;
                let bottom = linear_01 * (1.0 - fx) + linear_11 * fx;
                let interpolated = top * (1.0 - fy) + bottom * fy;

                // Convert back to sRGB
                result[c] = linear_f32_to_srgb_u8(interpolated);
            }

            out.put_pixel(out_x, out_y, image::Rgb(result));
        }
    }

    out
}

/// Debug: draw projected Gaussian means as colored dots on top of a target image.
///
/// Uses the same projection and Gaussian subset as the renderer.
pub fn debug_overlay_means(
    target: &RgbImage,
    gaussians: &[Gaussian],
    camera: &Camera,
    radius_px: i32,
) -> RgbImage {
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

    // Filter out invalid Gaussians (NaN or inf depth) before sorting
    projected.retain(|g| g.mean.z.is_finite());

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());

    let mut out = target.clone();
    for g in &projected {
        let cx = g.mean.x.round() as i32;
        let cy = g.mean.y.round() as i32;
        let color = [
            linear_f32_to_srgb_u8(g.color.x),
            linear_f32_to_srgb_u8(g.color.y),
            linear_f32_to_srgb_u8(g.color.z),
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_alpha_from_opacity_logit_matches_finite_difference() {
        let opacity_logit = 0.3f32;
        let weight = 0.7f32;

        let (_alpha, d_alpha_d_logit) = alpha_from_opacity_logit(opacity_logit, weight);
        assert!(d_alpha_d_logit.is_finite());

        let eps = 1e-4f32;
        let (alpha_p, _) = alpha_from_opacity_logit(opacity_logit + eps, weight);
        let (alpha_m, _) = alpha_from_opacity_logit(opacity_logit - eps, weight);
        let numerical = (alpha_p - alpha_m) / (2.0 * eps);

        let diff = (numerical - d_alpha_d_logit).abs();
        assert!(
            diff < 1e-3,
            "finite diff mismatch: numerical={numerical} analytical={d_alpha_d_logit} diff={diff}"
        );
    }

    #[test]
    fn test_srgb_linear_endpoints() {
        assert_relative_eq!(srgb_u8_to_linear_f32(0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(srgb_u8_to_linear_f32(255), 1.0, epsilon = 1e-6);
        assert_eq!(linear_f32_to_srgb_u8(0.0), 0);
        assert_eq!(linear_f32_to_srgb_u8(1.0), 255);
    }

    #[test]
    fn test_srgb_midpoint_sanity() {
        // 128/255 ≈ 0.50196 sRGB corresponds to ≈ 0.21586 linear.
        let lin = srgb_u8_to_linear_f32(128);
        assert_relative_eq!(lin, 0.21586, epsilon = 5e-3);
        let back = linear_f32_to_srgb_u8(lin);
        assert!((back as i32 - 128).abs() <= 1);
    }
}

/// Debug: compute a per-pixel coverage mask (whether any Gaussian contributes).
///
/// The output is a grayscale image where 0=uncovered, 255=covered.
pub fn debug_coverage_mask(gaussians: &[Gaussian], camera: &Camera) -> RgbImage {
    let width = camera.width as i32;
    let height = camera.height as i32;

    let covered = coverage_mask_bool(gaussians, camera);
    let mut img = RgbImage::new(camera.width, camera.height);

    for py in 0..height {
        for px in 0..width {
            let v = if covered[(py * width + px) as usize] {
                255u8
            } else {
                0u8
            };
            img.put_pixel(px as u32, py as u32, image::Rgb([v, v, v]));
        }
    }

    img
}

/// Compute a per-pixel coverage mask (whether any Gaussian contributes).
///
/// Returns a Vec<bool> with length width*height, row-major.
pub fn coverage_mask_bool(gaussians: &[Gaussian], camera: &Camera) -> Vec<bool> {
    let width = camera.width as i32;
    let height = camera.height as i32;

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();

    // Filter out invalid Gaussians (NaN or inf depth) before sorting
    let original_count = projected.len();
    projected.retain(|g| g.mean.z.is_finite());
    let filtered_count = original_count - projected.len();
    if filtered_count > 0 {
        eprintln!("[CPU WARNING] coverage_mask_bool filtered {} Gaussians with invalid depth values", filtered_count);
    }

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected, gaussians, camera);

    let mut covered = vec![false; (width * height) as usize];

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut is_covered = false;
            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form = g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha >= 1e-4 {
                    is_covered = true;
                    break;
                }
            }

            covered[(py * width + px) as usize] = is_covered;
        }
    }

    covered
}

/// Compute coverage weights at downsampled resolution and upsample to full resolution.
///
/// This provides a fast approximation of coverage weighting for GPU training,
/// where computing full-resolution coverage on CPU would be too slow.
///
/// Returns a Vec<f32> with length width*height, row-major.
/// Covered pixels get weight 1.0, uncovered pixels get weight 0.1.
pub fn coverage_weights_downsampled(
    gaussians: &[Gaussian],
    camera: &Camera,
    downsample: u32,
) -> Vec<f32> {
    let full_width = camera.width;
    let full_height = camera.height;

    // Create downsampled camera
    let ds_width = (full_width + downsample - 1) / downsample;
    let ds_height = (full_height + downsample - 1) / downsample;
    let scale = 1.0 / downsample as f32;

    let ds_camera = Camera::new(
        camera.fx * scale,
        camera.fy * scale,
        camera.cx * scale,
        camera.cy * scale,
        ds_width,
        ds_height,
        camera.rotation,
        camera.translation,
    );

    // Compute coverage at low resolution
    let ds_coverage = coverage_mask_bool(gaussians, &ds_camera);

    // Upsample to full resolution with nearest-neighbor
    let mut weights = vec![0.1f32; (full_width * full_height) as usize];

    for py in 0..full_height {
        for px in 0..full_width {
            let ds_x = px / downsample;
            let ds_y = py / downsample;
            let ds_idx = (ds_y * ds_width + ds_x) as usize;

            if ds_idx < ds_coverage.len() && ds_coverage[ds_idx] {
                weights[(py * full_width + px) as usize] = 1.0;
            }
        }
    }

    weights
}

/// Debug: visualize the final transmittance `T_N` per pixel (white=transparent, black=opaque).
pub fn debug_final_transmittance(gaussians: &[Gaussian], camera: &Camera) -> RgbImage {
    let width = camera.width as i32;
    let height = camera.height as i32;

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();

    // Filter out invalid Gaussians (NaN or inf depth) before sorting
    projected.retain(|g| g.mean.z.is_finite());

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected, gaussians, camera);

    let mut img = RgbImage::new(camera.width, camera.height);

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut t = 1.0f32;
            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form = g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha < 1e-4 {
                    continue;
                }
                t *= 1.0 - alpha;
            }

            let v = linear_f32_to_srgb_u8(t);
            img.put_pixel(px as u32, py as u32, image::Rgb([v, v, v]));
        }
    }

    img
}

/// Debug: visualize number of contributing Gaussians per pixel (brighter = more contributors).
pub fn debug_contrib_count(gaussians: &[Gaussian], camera: &Camera, clamp_max: u32) -> RgbImage {
    let width = camera.width as i32;
    let height = camera.height as i32;
    let clamp_max = clamp_max.max(1) as f32;

    let mut projected: Vec<Gaussian2D> = gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project_gaussian(g, camera, i))
        .collect();

    // Filter out invalid Gaussians (NaN or inf depth) before sorting
    projected.retain(|g| g.mean.z.is_finite());

    projected.sort_by(|a, b| a.mean.z.partial_cmp(&b.mean.z).unwrap());
    let prepared = prepare(&projected, gaussians, camera);

    let mut img = RgbImage::new(camera.width, camera.height);

    for py in 0..height {
        for px in 0..width {
            let pixel_x = px as f32 + 0.5;
            let pixel_y = py as f32 + 0.5;

            let mut count = 0u32;
            for g in &prepared {
                if px < g.min_x || px > g.max_x || py < g.min_y || py > g.max_y {
                    continue;
                }

                let dx = pixel_x - g.mean_x;
                let dy = pixel_y - g.mean_y;
                let quad_form = g.inv_xx * dx * dx + 2.0 * g.inv_xy * dx * dy + g.inv_yy * dy * dy;
                let weight = (-0.5 * quad_form).exp();
                let alpha = (g.opacity * weight).min(0.99);
                if alpha >= 1e-4 {
                    count += 1;
                }
            }

            let t = (count as f32 / clamp_max).clamp(0.0, 1.0);
            let v = linear_f32_to_srgb_u8(t);
            img.put_pixel(px as u32, py as u32, image::Rgb([v, v, v]));
        }
    }

    img
}
