//! Gradients for spherical harmonics evaluation.
//!
//! Forward op (in `core/sh.rs`):
//! `color = sum_i basis[i] * sh_coeffs[i]` (per-channel).
//!
//! For gradient checking (M6), we start with the simplest and most important part:
//! gradients w.r.t. the SH coefficients. This is a pure linear map, so the
//! derivative is exact and easy to verify.
//!
//! Note: the forward `evaluate_sh` clamps the final color to [0, 1]. For gradient
//! checks to be meaningful, choose coefficients that keep the output strictly
//! inside (0, 1), so the clamp is inactive.

use nalgebra::Vector3;

/// Gradient of `evaluate_sh` w.r.t. the SH coefficients.
///
/// Given:
/// - `basis`: `[f32; 16]` SH basis evaluated at the view direction
/// - `d_color`: upstream gradient dL/dcolor (RGB)
///
/// Returns:
/// - `d_sh_coeffs`: dL/d(sh_coeffs) with shape `[16][3]` (RGB × 16)
pub fn evaluate_sh_grad_coeffs(basis: &[f32; 16], d_color: &Vector3<f32>) -> [[f32; 3]; 16] {
    let mut d_sh = [[0.0f32; 3]; 16];
    for i in 0..16 {
        let b = basis[i];
        d_sh[i][0] = d_color.x * b;
        d_sh[i][1] = d_color.y * b;
        d_sh[i][2] = d_color.z * b;
    }
    d_sh
}

