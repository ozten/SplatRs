//! Loss functions for training (M7+).

use nalgebra::Vector3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    L2,
    L1Dssim,
}

/// Mean squared error over an image, returning (loss, d_rendered).
///
/// `rendered` and `target` are linear RGB in [0,1].
pub fn l2_image_loss_and_grad(
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

/// Weighted mean squared error over an image, returning (loss, d_rendered).
///
/// `weights` is per-pixel scalar weight (same length as pixels). The loss is
/// normalized by `sum(weights)`, not by pixel count.
pub fn l2_image_loss_and_grad_weighted(
    rendered: &[Vector3<f32>],
    target: &[Vector3<f32>],
    weights: &[f32],
) -> (f32, Vec<Vector3<f32>>) {
    assert_eq!(rendered.len(), target.len());
    assert_eq!(rendered.len(), weights.len());

    let mut weight_sum = 0.0f32;
    for &w in weights {
        weight_sum += w;
    }
    let denom = weight_sum.max(1e-6);

    let mut loss = 0.0f32;
    let mut d = vec![Vector3::<f32>::zeros(); rendered.len()];

    for i in 0..rendered.len() {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let diff = rendered[i] - target[i];
        loss += w * diff.dot(&diff);
        d[i] = diff * (2.0 * w / denom);
    }

    (loss / denom, d)
}

fn luminance(rgb: Vector3<f32>) -> f32 {
    0.299 * rgb.x + 0.587 * rgb.y + 0.114 * rgb.z
}

fn d_luminance_to_rgb(dy: f32) -> Vector3<f32> {
    Vector3::new(0.299 * dy, 0.587 * dy, 0.114 * dy)
}

fn gaussian_kernel_offsets(radius: i32, sigma: f32) -> Vec<(i32, i32, f32)> {
    let mut out = Vec::new();
    let denom = 2.0 * sigma * sigma;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let r2 = (dx * dx + dy * dy) as f32;
            let w = (-r2 / denom).exp();
            out.push((dx, dy, w));
        }
    }
    out
}

/// Paper-style image loss: `0.8 * L1 + 0.2 * (1 - SSIM)`.
///
/// This is a deliberately direct, educational implementation:
/// - SSIM is computed on luminance (linear RGB → Y).
/// - Uses an 11×11 Gaussian window with sigma=1.5.
/// - Returns per-pixel gradient dL/d(rendered_rgb) in linear RGB.
///
/// The loss is normalized by `sum(weights)` (like the weighted L2 loss).
pub fn l1_dssim_image_loss_and_grad_weighted(
    rendered: &[Vector3<f32>],
    target: &[Vector3<f32>],
    weights: &[f32],
    width: u32,
    height: u32,
) -> (f32, Vec<Vector3<f32>>) {
    assert_eq!(rendered.len(), target.len());
    assert_eq!(rendered.len(), weights.len());
    assert_eq!(rendered.len(), (width * height) as usize);

    let mut weight_sum = 0.0f32;
    for &w in weights {
        weight_sum += w;
    }
    let denom = weight_sum.max(1e-6);

    // L1 loss (RGB, mean over channels).
    let mut l1 = 0.0f32;
    let mut d_l1 = vec![Vector3::<f32>::zeros(); rendered.len()];
    for i in 0..rendered.len() {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let diff = rendered[i] - target[i];
        l1 += w * (diff.x.abs() + diff.y.abs() + diff.z.abs()) / 3.0;
        let sign = Vector3::new(diff.x.signum(), diff.y.signum(), diff.z.signum());
        d_l1[i] = sign * (w / (denom * 3.0));
    }
    l1 /= denom;

    // SSIM on luminance.
    let rendered_y: Vec<f32> = rendered.iter().copied().map(luminance).collect();
    let target_y: Vec<f32> = target.iter().copied().map(luminance).collect();
    let mut d_dssim_y = vec![0.0f32; rendered.len()];

    let radius = 5i32; // 11x11
    let sigma = 1.5f32;
    let kernel = gaussian_kernel_offsets(radius, sigma);

    let c1 = 0.01f32 * 0.01f32;
    let c2 = 0.03f32 * 0.03f32;

    let w_i = width as i32;
    let h_i = height as i32;

    let mut dssim = 0.0f32;
    for py in 0..h_i {
        for px in 0..w_i {
            let center = (py * w_i + px) as usize;
            let w_center = weights[center];
            if w_center == 0.0 {
                continue;
            }

            // Normalize kernel weights for valid pixels at the boundary.
            let mut wsum_local = 0.0f32;
            for &(dx, dy, w) in &kernel {
                let x = px + dx;
                let y = py + dy;
                if x < 0 || x >= w_i || y < 0 || y >= h_i {
                    continue;
                }
                wsum_local += w;
            }
            if wsum_local <= 0.0 {
                continue;
            }

            let mut mu_x = 0.0f32;
            let mut mu_y = 0.0f32;
            for &(dx, dy, w) in &kernel {
                let x = px + dx;
                let y = py + dy;
                if x < 0 || x >= w_i || y < 0 || y >= h_i {
                    continue;
                }
                let wi = w / wsum_local;
                let idx = (y * w_i + x) as usize;
                mu_x += wi * rendered_y[idx];
                mu_y += wi * target_y[idx];
            }

            let mut var_x = 0.0f32;
            let mut var_y = 0.0f32;
            let mut cov = 0.0f32;
            for &(dx, dy, w) in &kernel {
                let x = px + dx;
                let y = py + dy;
                if x < 0 || x >= w_i || y < 0 || y >= h_i {
                    continue;
                }
                let wi = w / wsum_local;
                let idx = (y * w_i + x) as usize;
                let dxv = rendered_y[idx] - mu_x;
                let dyv = target_y[idx] - mu_y;
                var_x += wi * dxv * dxv;
                var_y += wi * dyv * dyv;
                cov += wi * dxv * dyv;
            }

            let a = 2.0 * mu_x * mu_y + c1;
            let b = 2.0 * cov + c2;
            let c = mu_x * mu_x + mu_y * mu_y + c1;
            let d = var_x + var_y + c2;
            let inv_cd = 1.0 / (c * d).max(1e-6);
            let ssim = (a * b) * inv_cd;

            dssim += w_center * (1.0 - ssim);

            // dssim/d(rendered_y[q]) for all q in the window.
            let d_ssim_da = b * inv_cd;
            let d_ssim_db = a * inv_cd;
            let d_ssim_dc = -(ssim / c.max(1e-6));
            let d_ssim_dd = -(ssim / d.max(1e-6));

            let d_ssim_d_mu_x = d_ssim_da * (2.0 * mu_y) + d_ssim_dc * (2.0 * mu_x);
            let d_ssim_d_var_x = d_ssim_dd;
            let d_ssim_d_cov = d_ssim_db * 2.0;

            let coeff = w_center / denom;
            for &(dx, dy, w) in &kernel {
                let x = px + dx;
                let y = py + dy;
                if x < 0 || x >= w_i || y < 0 || y >= h_i {
                    continue;
                }
                let wi = w / wsum_local;
                let idx = (y * w_i + x) as usize;
                let dxv = rendered_y[idx] - mu_x;
                let dyv = target_y[idx] - mu_y;

                // d(ssim)/d(x_q)
                let d_ssim_dxq = d_ssim_d_mu_x * wi
                    + d_ssim_d_var_x * (2.0 * wi * dxv)
                    + d_ssim_d_cov * (wi * dyv);

                // d(1-ssim)/d(x_q) = -d(ssim)/d(x_q)
                d_dssim_y[idx] += coeff * (-d_ssim_dxq);
            }
        }
    }
    dssim /= denom;

    let dssim_weight = 0.2f32;
    let l1_weight = 0.8f32;

    let mut loss = l1_weight * l1 + dssim_weight * dssim;
    if !loss.is_finite() {
        loss = 0.0;
    }

    let mut d = vec![Vector3::<f32>::zeros(); rendered.len()];
    for i in 0..rendered.len() {
        d[i] = d_l1[i] * l1_weight + d_luminance_to_rgb(d_dssim_y[i] * dssim_weight);
    }

    (loss, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_l1_dssim_loss_gradient_smoke() {
        let width = 8u32;
        let height = 8u32;
        let n = (width * height) as usize;
        let rendered = vec![Vector3::new(0.2, 0.4, 0.6); n];
        let target = vec![Vector3::new(0.25, 0.35, 0.55); n];
        let weights = vec![1.0f32; n];
        let (loss, grad) =
            l1_dssim_image_loss_and_grad_weighted(&rendered, &target, &weights, width, height);
        assert!(loss.is_finite());
        assert_eq!(grad.len(), n);
        for g in grad {
            assert!(g.x.is_finite() && g.y.is_finite() && g.z.is_finite());
        }
    }

    #[test]
    fn test_l1_loss_component_matches_expected_on_single_pixel() {
        let width = 1u32;
        let height = 1u32;
        let rendered = vec![Vector3::new(0.0, 1.0, 0.0)];
        let target = vec![Vector3::new(1.0, 0.0, 0.0)];
        let weights = vec![1.0f32];

        let (loss, grad) =
            l1_dssim_image_loss_and_grad_weighted(&rendered, &target, &weights, width, height);
        // SSIM on a 1x1 image is ill-defined for our windowed stats, but we clamp denominators.
        assert!(loss.is_finite());
        assert_relative_eq!(grad[0].x.signum(), -1.0, epsilon = 0.0);
        assert_relative_eq!(grad[0].y.signum(), 1.0, epsilon = 0.0);
    }

    #[test]
    fn test_l1_dssim_gradient_matches_finite_difference_single_pixel() {
        let width = 8u32;
        let height = 8u32;
        let n = (width * height) as usize;

        let mut rendered = vec![Vector3::new(0.2, 0.2, 0.2); n];
        let target = vec![Vector3::new(0.25, 0.15, 0.3); n];
        let weights = vec![1.0f32; n];

        let (loss, grad) =
            l1_dssim_image_loss_and_grad_weighted(&rendered, &target, &weights, width, height);
        assert!(loss.is_finite());

        let idx = (3 * (width as usize) + 4) as usize;
        let eps = 1e-3f32;

        let base = rendered[idx];
        rendered[idx] = Vector3::new(base.x + eps, base.y, base.z);
        let (loss_p, _) =
            l1_dssim_image_loss_and_grad_weighted(&rendered, &target, &weights, width, height);
        rendered[idx] = Vector3::new(base.x - eps, base.y, base.z);
        let (loss_m, _) =
            l1_dssim_image_loss_and_grad_weighted(&rendered, &target, &weights, width, height);
        rendered[idx] = base;

        let numerical = (loss_p - loss_m) / (2.0 * eps);
        let analytical = grad[idx].x;

        let diff = (numerical - analytical).abs();
        assert!(
            diff < 5e-2,
            "finite diff mismatch: numerical={numerical} analytical={analytical} diff={diff}"
        );
    }
}
