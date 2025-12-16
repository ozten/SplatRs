//! Gradient checking tests - THE MOST IMPORTANT TESTS (M6)
//!
//! These tests verify that analytical gradients match numerical gradients
//! computed via finite differences. This is critical for correct training.
//!
//! For every differentiable operation, we test:
//! - Numerical: (f(x+ε) - f(x-ε)) / 2ε
//! - Analytical: backward pass implementation
//! - Assert relative error < 1e-4

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use sugar_rs::core::{evaluate_sh, sh_basis};
    use sugar_rs::diff::sh_grad::evaluate_sh_grad_coeffs;

    // These tests are NON-NEGOTIABLE - bugs in gradients cause silent failures.
    fn rel_err(a: f32, b: f32) -> f32 {
        let denom = a.abs().max(b.abs()).max(1e-6);
        (a - b).abs() / denom
    }

    #[test]
    #[ignore] // Remove when implementing
    fn test_sigmoid_gradient() {
        // TODO: Test sigmoid and inverse_sigmoid gradients
    }

    #[test]
    #[ignore]
    fn test_quaternion_to_matrix_gradient() {
        // TODO: Test quaternion → rotation matrix gradients
    }

    #[test]
    #[ignore]
    fn test_covariance_projection_gradient() {
        // TODO: Test 3D → 2D covariance projection gradients
    }

    #[test]
    #[ignore]
    fn test_gaussian_2d_evaluation_gradient() {
        // TODO: Test 2D Gaussian evaluation gradients
    }

    #[test]
    #[ignore]
    fn test_alpha_blending_gradient() {
        // TODO: Test alpha blending gradients
    }

    #[test]
    fn test_sh_evaluation_gradient() {
        // Gradient check for SH evaluation w.r.t. coefficients.
        //
        // Forward:
        //   color = sum_i basis[i] * sh_coeffs[i]   (per channel)
        //
        // Loss:
        //   L = w · color
        //
        // Then dL/d(sh_coeffs[i][c]) = w[c] * basis[i]
        //
        // We choose small coefficients so `evaluate_sh`'s output clamp is inactive.
        let mut rng = StdRng::seed_from_u64(0x5EED_5A5A_u64);
        // Use an epsilon that adapts to the basis magnitude so the induced color delta
        // is "big enough" for f32, but still far from the clamp region.
        let target_delta = 0.05f32;
        let tol = 1e-4f32;

        for _ in 0..50 {
            // Random direction (do NOT pre-normalize; `evaluate_sh` normalizes internally).
            let mut dir_raw = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            if dir_raw.norm_squared() < 1e-6 {
                dir_raw = Vector3::new(0.0, 0.0, 1.0);
            }
            let dir_norm = dir_raw.normalize();
            let basis = sh_basis(&dir_norm);

            // Small coefficients centered around a mid-gray via DC.
            // We explicitly set the DC coefficient so the resulting base color is ~0.5
            // (far from the clamp at 0/1), and keep higher-order terms tiny so the
            // clamp remains inactive.
            let base_r = 0.5;
            let base_g = 0.5;
            let base_b = 0.5;

            // Retry until the clamp is comfortably inactive.
            let sh_coeffs = loop {
                let mut sh_coeffs = [[0.0f32; 3]; 16];
                sh_coeffs[0] = [base_r / basis[0], base_g / basis[0], base_b / basis[0]];
                for i in 1..16 {
                    for c in 0..3 {
                        sh_coeffs[i][c] = rng.gen_range(-0.001..0.001);
                    }
                }

                let color = evaluate_sh(&sh_coeffs, &dir_raw);
                if (0.2..0.8).contains(&color.x) && (0.2..0.8).contains(&color.y) && (0.2..0.8).contains(&color.z) {
                    break sh_coeffs;
                }
            };

            // Loss weight vector (upstream gradient for color).
            let w = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );

            // Analytical gradient.
            let d_sh = evaluate_sh_grad_coeffs(&basis, &w);

            // Numerical gradient for a random handful of coefficients.
            let mut checked = 0;
            for _ in 0..200 {
                let i = rng.gen_range(0..16);
                let c = rng.gen_range(0..3);

                let mut plus = sh_coeffs;
                let mut minus = sh_coeffs;
                let eps = (target_delta / basis[i].abs().max(1e-3)).clamp(1e-3, 0.2);
                plus[i][c] += eps;
                minus[i][c] -= eps;

                let f = |coeffs: &[[f32; 3]; 16]| -> f32 {
                    let color = evaluate_sh(coeffs, &dir_raw);
                    w.dot(&color)
                };

                // Skip cases where the clamp might activate for either perturbation.
                // This keeps the function locally linear for finite differences.
                let color_plus = evaluate_sh(&plus, &dir_raw);
                let color_minus = evaluate_sh(&minus, &dir_raw);
                let clamp_safe = |v: f32| (0.05..0.95).contains(&v);
                if !clamp_safe(color_plus.x)
                    || !clamp_safe(color_plus.y)
                    || !clamp_safe(color_plus.z)
                    || !clamp_safe(color_minus.x)
                    || !clamp_safe(color_minus.y)
                    || !clamp_safe(color_minus.z)
                {
                    continue;
                }

                // Do the subtraction in f64 to reduce cancellation/rounding error.
                let f_plus = f(&plus) as f64;
                let f_minus = f(&minus) as f64;
                let num = ((f_plus - f_minus) / (2.0 * (eps as f64))) as f32;
                let ana = d_sh[i][c];

                let abs_err = (num - ana).abs();
                let ok = rel_err(num, ana) < tol || abs_err < 1e-6;
                assert!(
                    ok,
                    "SH grad mismatch at i={i} c={c}: num={num} ana={ana} abs_err={abs_err} rel_err={} basis_i={} w={:?}",
                    rel_err(num, ana),
                    basis[i],
                    w
                );

                checked += 1;
                if checked >= 20 {
                    break;
                }
            }
            assert!(checked >= 20, "Expected to check >= 20 coefficient grads, got {checked}");
        }
    }
}
