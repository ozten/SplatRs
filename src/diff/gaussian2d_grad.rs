//! Gradients for 2D Gaussian evaluation.
//!
//! Forward (see `core::Gaussian2D::evaluate_at`):
//!   w = exp(-0.5 * q)
//!   q = d^T Σ^{-1} d
//!   d = [dx, dy] = [px - mx, py - my]
//!
//! This module provides gradients w.r.t:
//! - mean (mx, my)
//! - covariance entries (a, b, c) of Σ = [[a, b], [b, c]]
//!
//! Notes:
//! - These derivatives assume Σ is well-conditioned (det not ~0) so the forward
//!   inverse does not take any "near-singular" stabilization branch.

use nalgebra::Vector2;

#[derive(Clone, Copy, Debug)]
pub struct Gaussian2DEvalGrads {
    pub value: f32,
    pub d_mean: Vector2<f32>,
    pub d_cov_xx: f32,
    pub d_cov_xy: f32,
    pub d_cov_yy: f32,
}

/// Evaluate a 2D Gaussian and compute gradients w.r.t. mean and covariance entries.
///
/// `mean`: (mx, my)
/// `cov`: (a, b, c) storing Σ = [[a, b], [b, c]]
/// `pixel`: (px, py)
pub fn gaussian2d_evaluate_with_grads(
    mean: Vector2<f32>,
    cov_xx: f32,
    cov_xy: f32,
    cov_yy: f32,
    pixel: Vector2<f32>,
) -> Gaussian2DEvalGrads {
    let dx = pixel.x - mean.x;
    let dy = pixel.y - mean.y;

    // Σ^{-1} for symmetric 2x2: inv = 1/det * [[c, -b], [-b, a]]
    let a = cov_xx;
    let b = cov_xy;
    let c = cov_yy;
    let det = a * c - b * b;
    let inv_det = 1.0 / det;

    let inv_xx = c * inv_det;
    let inv_xy = -b * inv_det;
    let inv_yy = a * inv_det;

    // q = inv_xx*dx^2 + 2*inv_xy*dx*dy + inv_yy*dy^2
    let q = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy;
    let value = (-0.5 * q).exp();

    // d(value)/d(q) = -0.5 * value
    let d_value_d_q = -0.5 * value;

    // d(q)/d(dx,dy) = 2 * Σ^{-1} * d
    // d(value)/d(dx,dy) = d(value)/d(q) * d(q)/d(dx,dy) = -value * Σ^{-1} * d
    let ad_x = inv_xx * dx + inv_xy * dy;
    let ad_y = inv_xy * dx + inv_yy * dy;
    let d_value_d_dx = -value * ad_x;
    let d_value_d_dy = -value * ad_y;

    // d(dx)/d(mx) = -1, so d(value)/d(mx) = -d(value)/d(dx)
    // similarly for y.
    let d_mean = Vector2::new(-d_value_d_dx, -d_value_d_dy);

    // Gradients w.r.t inverse covariance elements:
    // q = inv_xx*dx^2 + 2*inv_xy*dx*dy + inv_yy*dy^2
    let d_q_d_inv_xx = dx * dx;
    let d_q_d_inv_xy = 2.0 * dx * dy;
    let d_q_d_inv_yy = dy * dy;

    let d_value_d_inv_xx = d_value_d_q * d_q_d_inv_xx;
    let d_value_d_inv_xy = d_value_d_q * d_q_d_inv_xy;
    let d_value_d_inv_yy = d_value_d_q * d_q_d_inv_yy;

    // Chain to covariance entries via inverse formula:
    //
    // inv_xx = c / det
    // inv_xy = -b / det
    // inv_yy = a / det
    //
    // det = a*c - b^2
    let det2 = det * det;

    // Partial derivatives of inverse entries w.r.t (a,b,c).
    let d_inv_xx_d_a = -(c * c) / det2;
    let d_inv_xx_d_b = (2.0 * b * c) / det2;
    let d_inv_xx_d_c = -(b * b) / det2;

    let d_inv_xy_d_a = (b * c) / det2;
    let d_inv_xy_d_b = -1.0 / det - (2.0 * b * b) / det2;
    let d_inv_xy_d_c = (a * b) / det2;

    let d_inv_yy_d_a = -(b * b) / det2;
    let d_inv_yy_d_b = (2.0 * a * b) / det2;
    let d_inv_yy_d_c = -(a * a) / det2;

    let d_cov_xx = d_value_d_inv_xx * d_inv_xx_d_a
        + d_value_d_inv_xy * d_inv_xy_d_a
        + d_value_d_inv_yy * d_inv_yy_d_a;
    let d_cov_xy = d_value_d_inv_xx * d_inv_xx_d_b
        + d_value_d_inv_xy * d_inv_xy_d_b
        + d_value_d_inv_yy * d_inv_yy_d_b;
    let d_cov_yy = d_value_d_inv_xx * d_inv_xx_d_c
        + d_value_d_inv_xy * d_inv_xy_d_c
        + d_value_d_inv_yy * d_inv_yy_d_c;

    Gaussian2DEvalGrads {
        value,
        d_mean,
        d_cov_xx,
        d_cov_xy,
        d_cov_yy,
    }
}

