//! Micro gradient checks for SH evaluation.
//!
//! These validate the per-coefficient gradient used by the optimizer,
//! using a finite-difference scalar loss on a tiny example.

use approx::assert_relative_eq;
use nalgebra::Vector3;
use sugar_rs::core::{evaluate_sh_unclamped, sh_basis};
use sugar_rs::diff::sh_grad::evaluate_sh_grad_coeffs;

#[test]
fn test_evaluate_sh_grad_coeffs_matches_finite_difference() {
    let dir = Vector3::new(0.2, -0.4, 0.894_427_2); // already normalized
    let basis = sh_basis(&dir);
    let d_color = Vector3::new(0.3, -0.2, 0.1);

    let mut sh_coeffs = [[0.0f32; 3]; 16];
    sh_coeffs[0] = [0.2, 0.1, 0.05];
    sh_coeffs[3] = [0.02, 0.01, 0.03];
    sh_coeffs[6] = [0.05, 0.02, 0.01];

    let analytic = evaluate_sh_grad_coeffs(&basis, &d_color);

    let eps = 1e-4;
    for i in 0..16 {
        for c in 0..3 {
            let mut plus = sh_coeffs;
            let mut minus = sh_coeffs;
            plus[i][c] += eps;
            minus[i][c] -= eps;

            let color_plus = evaluate_sh_unclamped(&plus, &dir);
            let color_minus = evaluate_sh_unclamped(&minus, &dir);
            let loss_plus = d_color.dot(&color_plus);
            let loss_minus = d_color.dot(&color_minus);
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);

            let analytic_component = analytic[i][c];
            assert_relative_eq!(
                numeric,
                analytic_component,
                epsilon = 1e-4,
                max_relative = 1e-4
            );
        }
    }
}
