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

    use sugar_rs::core::{evaluate_sh, inverse_sigmoid, perspective_jacobian, sh_basis, sigmoid};
    use sugar_rs::diff::math_grad::{inverse_sigmoid_grad, sigmoid_grad_from_sigmoid};
    use sugar_rs::diff::gaussian2d_grad::gaussian2d_evaluate_with_grads;
    use sugar_rs::diff::sh_grad::evaluate_sh_grad_coeffs;
    use sugar_rs::diff::blend_grad::{blend_backward, blend_forward};
    use sugar_rs::diff::covariance_grad::{
        project_covariance_2d_grad_log_scale, project_covariance_2d_grad_rotation_vector_at_r0,
    };
    use sugar_rs::diff::covariance_grad::project_covariance_2d_grad_point_cam;
    use sugar_rs::diff::quaternion_grad::{quaternion_raw_to_matrix, quaternion_raw_to_matrix_grad};
    use sugar_rs::diff::project_grad::project_point_grad_point_cam;

    // These tests are NON-NEGOTIABLE - bugs in gradients cause silent failures.
    fn rel_err(a: f32, b: f32) -> f32 {
        let denom = a.abs().max(b.abs()).max(1e-6);
        (a - b).abs() / denom
    }

    #[test]
    fn test_sigmoid_gradient() {
        // Gradient check for:
        // - sigmoid(x) w.r.t x
        // - inverse_sigmoid(p) w.r.t p  (away from clamp)
        let mut rng = StdRng::seed_from_u64(0x51C1_01D_u64);

        // Sigmoid: choose x in a moderate range to avoid saturation.
        for _ in 0..200 {
            let x = rng.gen_range(-4.0..4.0);
            let eps = 1e-2f32;

            let f = |v: f32| sigmoid(v);
            let f_plus = f(x + eps) as f64;
            let f_minus = f(x - eps) as f64;
            let num = ((f_plus - f_minus) / (2.0 * eps as f64)) as f32;

            let s = sigmoid(x);
            let ana = sigmoid_grad_from_sigmoid(s);

            let abs_err = (num - ana).abs();
            assert!(
                rel_err(num, ana) < 1e-4 || abs_err < 1e-5,
                "sigmoid grad mismatch: x={x} num={num} ana={ana} abs_err={abs_err} rel_err={}",
                rel_err(num, ana)
            );
        }

        // inverse_sigmoid (logit): choose p safely away from clamp and 0/1.
        // Near 0/1 the derivative explodes, which makes finite differences noisy in f32.
        for _ in 0..200 {
            let p = rng.gen_range(0.1..0.9);
            let eps = 1e-3f32;

            let f = |v: f32| inverse_sigmoid(v);
            let num = ((f(p + eps) as f64 - f(p - eps) as f64) / (2.0 * eps as f64)) as f32;
            let ana = inverse_sigmoid_grad(p);

            let abs_err = (num - ana).abs();
            assert!(
                rel_err(num, ana) < 1e-4 || abs_err < 1e-5,
                "inverse_sigmoid grad mismatch: p={p} num={num} ana={ana} abs_err={abs_err} rel_err={}",
                rel_err(num, ana)
            );
        }
    }

    #[test]
    fn test_quaternion_to_matrix_gradient() {
        // Gradient check for raw quaternion (w,x,y,z) -> rotation matrix conversion.
        //
        // We parameterize with a raw 4-vector and normalize inside the function.
        let mut rng = StdRng::seed_from_u64(0x0A76_A4D1_u64);
        let tol = 5e-4f32;

        for _ in 0..200 {
            // Random non-zero quaternion raw parameters.
            let mut q_raw = nalgebra::Vector4::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            if q_raw.norm() < 1e-3 {
                q_raw.x = 1.0;
            }

            // Upstream gradient dL/dR.
            let d_r = nalgebra::Matrix3::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );

            let ana = quaternion_raw_to_matrix_grad(&q_raw, &d_r);

            // Scalar loss L = <d_r, R(q_raw)>
            let loss = |q: &nalgebra::Vector4<f32>| -> f64 {
                let r = quaternion_raw_to_matrix(q);
                (r[(0, 0)] as f64) * (d_r[(0, 0)] as f64)
                    + (r[(0, 1)] as f64) * (d_r[(0, 1)] as f64)
                    + (r[(0, 2)] as f64) * (d_r[(0, 2)] as f64)
                    + (r[(1, 0)] as f64) * (d_r[(1, 0)] as f64)
                    + (r[(1, 1)] as f64) * (d_r[(1, 1)] as f64)
                    + (r[(1, 2)] as f64) * (d_r[(1, 2)] as f64)
                    + (r[(2, 0)] as f64) * (d_r[(2, 0)] as f64)
                    + (r[(2, 1)] as f64) * (d_r[(2, 1)] as f64)
                    + (r[(2, 2)] as f64) * (d_r[(2, 2)] as f64)
            };

            let eps = 1e-3f32;
            for k in 0..4 {
                let mut plus = q_raw;
                let mut minus = q_raw;
                plus[k] += eps;
                minus[k] -= eps;

                let num = ((loss(&plus) - loss(&minus)) / (2.0 * eps as f64)) as f32;
                let got = ana[k];
                let abs_err = (got - num).abs();
                assert!(
                    rel_err(got, num) < tol || abs_err < 5e-4,
                    "quat->mat grad mismatch k={k}: num={num} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, num)
                );
            }
        }
    }

    #[test]
    fn test_rotation_to_covariance_gradient() {
        // Gradient check for local rotation-vector parameterization ω around a base rotation R0.
        // This checks the rotation -> covariance -> projection chain without quaternions.
        let mut rng = StdRng::seed_from_u64(0xA11C_E501_u64);
        let tol_f32 = 5e-4f32;

        for _ in 0..100 {
            let cam_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let w = cam_q.to_rotation_matrix().into_inner();

            let g_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let r0 = g_q.to_rotation_matrix().into_inner();

            let point_cam = nalgebra::Vector3::new(
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(1.0f32..4.0f32),
            );
            let fx = rng.gen_range(100.0f32..400.0f32);
            let fy = rng.gen_range(100.0f32..400.0f32);
            let j = perspective_jacobian(&point_cam, fx, fy);

            let log_scale = nalgebra::Vector3::new(
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
            );

            let g00 = rng.gen_range(-1.0f32..1.0f32);
            let g01 = rng.gen_range(-1.0f32..1.0f32);
            let g11 = rng.gen_range(-1.0f32..1.0f32);
            let d_sigma2d = nalgebra::Matrix2::new(g00, g01, g01, g11);

            let grad_f32 =
                project_covariance_2d_grad_rotation_vector_at_r0(&w, &j, &r0, &log_scale, &d_sigma2d);

            // f64 forward for numeric gradient.
            let w64 = w.map(|x| x as f64);
            let j64 = j.map(|x| x as f64);
            let d64 = d_sigma2d.map(|x| x as f64);
            let log_scale64 = log_scale.map(|x| x as f64);
            let r0_64 = r0.map(|x| x as f64);

            let d_vec = nalgebra::Vector3::new(
                (2.0 * log_scale64.x).exp(),
                (2.0 * log_scale64.y).exp(),
                (2.0 * log_scale64.z).exp(),
            );
            let d_mat = nalgebra::Matrix3::from_diagonal(&d_vec);

            let loss64 = |r: nalgebra::Matrix3<f64>| -> f64 {
                let sigma = r * d_mat * r.transpose();
                let sigma_cam = w64 * sigma * w64.transpose();
                let sigma2d = j64 * sigma_cam * j64.transpose();
                (sigma2d.component_mul(&d64)).sum()
            };

            let rot_x = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
            };
            let rot_y = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
            };
            let rot_z = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
            };

            let eps = 1e-4f64;
            let num_x = (loss64(rot_x(eps) * r0_64) - loss64(rot_x(-eps) * r0_64)) / (2.0 * eps);
            let num_y = (loss64(rot_y(eps) * r0_64) - loss64(rot_y(-eps) * r0_64)) / (2.0 * eps);
            let num_z = (loss64(rot_z(eps) * r0_64) - loss64(rot_z(-eps) * r0_64)) / (2.0 * eps);

            let num = nalgebra::Vector3::new(num_x as f32, num_y as f32, num_z as f32);

            for axis in 0..3 {
                let got = grad_f32[axis];
                let reference = num[axis];
                let abs_err = (got - reference).abs();
                assert!(
                    rel_err(got, reference) < tol_f32 || abs_err < 5e-4,
                    "rot->cov grad mismatch axis={axis}: num={reference} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, reference)
                );
            }
        }
    }

    #[test]
    fn test_covariance_projection_gradient() {
        // Gradient check for 3D -> 2D covariance projection w.r.t. log-scales (scale-only).
        //
        // This holds rotation and the perspective Jacobian fixed. We'll extend to rotation/J later.
        let mut rng = StdRng::seed_from_u64(0xC0A2_D5CA_u64);
        let tol = 5e-4f32;

        for _ in 0..200 {
            // Random camera rotation (orthonormal via nalgebra quaternion).
            let cam_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let w = cam_q.to_rotation_matrix().into_inner();

            // Random gaussian rotation.
            let g_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let r = g_q.to_rotation_matrix().into_inner();

            // Point in camera space for Jacobian.
            let point_cam = nalgebra::Vector3::new(
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(1.0f32..4.0f32),
            );
            // Keep intrinsics modest to avoid huge loss magnitudes (which make finite differences noisy in f32).
            let fx = rng.gen_range(100.0f32..400.0f32);
            let fy = rng.gen_range(100.0f32..400.0f32);
            let j = perspective_jacobian(&point_cam, fx, fy);

            // Log-scales in a modest range.
            let log_scale = nalgebra::Vector3::new(
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
            );

            // Upstream gradient for Σ₂d: pick a symmetric matrix.
            let g00 = rng.gen_range(-1.0f32..1.0f32);
            let g01 = rng.gen_range(-1.0f32..1.0f32);
            let g11 = rng.gen_range(-1.0f32..1.0f32);
            let d_sigma2d = nalgebra::Matrix2::new(g00, g01, g01, g11);

            // Compute production (f32) analytic gradient.
            let d_log_scale_f32 =
                project_covariance_2d_grad_log_scale(&w, &j, &r, &log_scale, &d_sigma2d);

            // For a stable gradient check, compute forward + analytic gradient in f64.
            let w64 = w.map(|x| x as f64);
            let r64 = r.map(|x| x as f64);
            let j64 = j.map(|x| x as f64);
            let g64 = d_sigma2d.map(|x| x as f64);
            let log_scale64 = log_scale.map(|x| x as f64);

            let loss64 = |s: nalgebra::Vector3<f64>| -> f64 {
                let sigma2d = {
                    let v = nalgebra::Vector3::new((2.0 * s.x).exp(), (2.0 * s.y).exp(), (2.0 * s.z).exp());
                    let d = nalgebra::Matrix3::from_diagonal(&v);
                    let sigma = r64 * d * r64.transpose();
                    let sigma_cam = w64 * sigma * w64.transpose();
                    j64 * sigma_cam * j64.transpose()
                };
                (sigma2d.component_mul(&g64)).sum()
            };

            let grad64 = |s: nalgebra::Vector3<f64>| -> nalgebra::Vector3<f64> {
                let v = nalgebra::Vector3::new((2.0 * s.x).exp(), (2.0 * s.y).exp(), (2.0 * s.z).exp());
                let d_sigma_cam: nalgebra::Matrix3<f64> = j64.transpose() * g64 * j64;
                let d_sigma = w64.transpose() * d_sigma_cam * w64;
                let m = r64.transpose() * d_sigma * r64;
                nalgebra::Vector3::new(m[(0, 0)] * 2.0 * v.x, m[(1, 1)] * 2.0 * v.y, m[(2, 2)] * 2.0 * v.z)
            };

            let ana64 = grad64(log_scale64);

            // Finite differences for each scale component (f64).
            for axis in 0..3 {
                let eps = 1e-4f64;
                let mut plus = log_scale64;
                let mut minus = log_scale64;
                plus[axis] += eps;
                minus[axis] -= eps;

                let num = (loss64(plus) - loss64(minus)) / (2.0 * eps);
                let ana = ana64[axis];
                let denom = num.abs().max(ana.abs()).max(1e-9);
                let rel = (num - ana).abs() / denom;
                assert!(rel < 1e-6, "cov proj grad mismatch axis={axis}: num={num} ana={ana} rel={rel}");

                // Also ensure our f32 implementation matches the f64 analytic result.
                let ana_f32_ref = ana as f32;
                let got = d_log_scale_f32[axis];
                let abs_err = (got - ana_f32_ref).abs();
                assert!(
                    rel_err(got, ana_f32_ref) < tol || abs_err < 5e-4,
                    "cov proj f32 mismatch axis={axis}: got={got} ref={ana_f32_ref} abs_err={abs_err} rel_err={}",
                    rel_err(got, ana_f32_ref)
                );
            }
        }
    }

    #[test]
    fn test_gaussian_2d_evaluation_gradient() {
        // Gradient check for 2D Gaussian evaluation w.r.t mean and covariance entries.
        //
        // This test uses a well-conditioned covariance to avoid any stabilization branches.
        let mut rng = StdRng::seed_from_u64(0xD00D_2D_u64);
        let tol = 5e-4f32;

        for _ in 0..200 {
            // Random mean and pixel within a modest range.
            let mean = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                0.0,
            );
            let pixel = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                0.0,
            );

            // SPD covariance with controlled off-diagonal.
            let cov_xx = rng.gen_range(0.5f32..2.0f32);
            let cov_yy = rng.gen_range(0.5f32..2.0f32);
            let max_b = 0.25 * (cov_xx * cov_yy).sqrt();
            let cov_xy = rng.gen_range(-max_b..max_b);

            let det = cov_xx * cov_yy - cov_xy * cov_xy;
            assert!(det > 0.1, "det too small for stable test: {det}");

            let grads = gaussian2d_evaluate_with_grads(
                nalgebra::Vector2::new(mean.x, mean.y),
                cov_xx,
                cov_xy,
                cov_yy,
                nalgebra::Vector2::new(pixel.x, pixel.y),
            );

            // Mean gradients (finite difference on mx, my).
            for (which, ana) in [("mx", grads.d_mean.x), ("my", grads.d_mean.y)] {
                let eps = 1e-3f32;
                let f = |mx: f32, my: f32| -> f32 {
                    gaussian2d_evaluate_with_grads(
                        nalgebra::Vector2::new(mx, my),
                        cov_xx,
                        cov_xy,
                        cov_yy,
                        nalgebra::Vector2::new(pixel.x, pixel.y),
                    )
                    .value
                };

                let (plus, minus) = match which {
                    "mx" => (f(mean.x + eps, mean.y), f(mean.x - eps, mean.y)),
                    "my" => (f(mean.x, mean.y + eps), f(mean.x, mean.y - eps)),
                    _ => unreachable!(),
                };
                let num = ((plus as f64 - minus as f64) / (2.0 * eps as f64)) as f32;

                let abs_err = (num - ana).abs();
                assert!(
                    rel_err(num, ana) < tol || abs_err < 2e-4,
                    "2D eval mean grad mismatch {which}: num={num} ana={ana} abs_err={abs_err} rel_err={}",
                    rel_err(num, ana)
                );
            }

            // Covariance gradients (finite difference on a,b,c).
            let eps = 1e-2f32;
            let f = |a: f32, b: f32, c: f32| -> f32 {
                gaussian2d_evaluate_with_grads(
                    nalgebra::Vector2::new(mean.x, mean.y),
                    a,
                    b,
                    c,
                    nalgebra::Vector2::new(pixel.x, pixel.y),
                )
                .value
            };

            let num_a = ((f(cov_xx + eps, cov_xy, cov_yy) as f64 - f(cov_xx - eps, cov_xy, cov_yy) as f64)
                / (2.0 * eps as f64)) as f32;
            let num_b = ((f(cov_xx, cov_xy + eps, cov_yy) as f64 - f(cov_xx, cov_xy - eps, cov_yy) as f64)
                / (2.0 * eps as f64)) as f32;
            let num_c = ((f(cov_xx, cov_xy, cov_yy + eps) as f64 - f(cov_xx, cov_xy, cov_yy - eps) as f64)
                / (2.0 * eps as f64)) as f32;

            for (name, num, ana) in [
                ("cov_xx", num_a, grads.d_cov_xx),
                ("cov_xy", num_b, grads.d_cov_xy),
                ("cov_yy", num_c, grads.d_cov_yy),
            ] {
                let abs_err = (num - ana).abs();
                assert!(
                    rel_err(num, ana) < tol || abs_err < 2e-4,
                    "2D eval cov grad mismatch {name}: num={num} ana={ana} abs_err={abs_err} rel_err={}",
                    rel_err(num, ana)
                );
            }
        }
    }

    #[test]
    fn test_alpha_blending_gradient() {
        // Gradient check for front-to-back alpha compositing w.r.t.:
        // - alphas a_i
        // - colors c_i (RGB)
        let mut rng = StdRng::seed_from_u64(0xB1ED_0BAD_u64);
        let tol = 5e-4f32;

        for _ in 0..200 {
            let n = rng.gen_range(1..6);
            let mut alphas = Vec::with_capacity(n);
            let mut colors = Vec::with_capacity(n);

            // Keep alpha away from 0/1 to avoid extremely small transmittance and numerical issues.
            for _ in 0..n {
                alphas.push(rng.gen_range(0.05f32..0.5f32));
                colors.push(Vector3::new(
                    rng.gen_range(0.0f32..1.0f32),
                    rng.gen_range(0.0f32..1.0f32),
                    rng.gen_range(0.0f32..1.0f32),
                ));
            }

            let forward = blend_forward(&alphas, &colors);

            // Loss: L = w · out
            let w = Vector3::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );

            let grads = blend_backward(&alphas, &colors, &forward, &w);

            let eps = 1e-3f32;
            let f = |a: &[f32], c: &[Vector3<f32>]| -> f32 {
                let out = blend_forward(a, c).out;
                w.dot(&out)
            };

            // Check alpha gradients.
            for i in 0..n {
                let mut a_plus = alphas.clone();
                let mut a_minus = alphas.clone();
                a_plus[i] += eps;
                a_minus[i] -= eps;

                let num = ((f(&a_plus, &colors) as f64 - f(&a_minus, &colors) as f64) / (2.0 * eps as f64)) as f32;
                let ana = grads.d_alphas[i];
                let abs_err = (num - ana).abs();
                assert!(
                    rel_err(num, ana) < tol || abs_err < 2e-4,
                    "blend alpha grad mismatch i={i}: num={num} ana={ana} abs_err={abs_err} rel_err={}",
                    rel_err(num, ana)
                );
            }

            // Check a few random color components.
            for _ in 0..5 {
                let i = rng.gen_range(0..n);
                let channel = rng.gen_range(0..3);

                let mut c_plus = colors.clone();
                let mut c_minus = colors.clone();
                match channel {
                    0 => {
                        c_plus[i].x += eps;
                        c_minus[i].x -= eps;
                    }
                    1 => {
                        c_plus[i].y += eps;
                        c_minus[i].y -= eps;
                    }
                    2 => {
                        c_plus[i].z += eps;
                        c_minus[i].z -= eps;
                    }
                    _ => unreachable!(),
                }

                let num = ((f(&alphas, &c_plus) as f64 - f(&alphas, &c_minus) as f64) / (2.0 * eps as f64)) as f32;
                let ana = match channel {
                    0 => grads.d_colors[i].x,
                    1 => grads.d_colors[i].y,
                    2 => grads.d_colors[i].z,
                    _ => unreachable!(),
                };

                let abs_err = (num - ana).abs();
                assert!(
                    rel_err(num, ana) < tol || abs_err < 2e-4,
                    "blend color grad mismatch i={i} ch={channel}: num={num} ana={ana} abs_err={abs_err} rel_err={}",
                    rel_err(num, ana)
                );
            }
        }
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

    #[test]
    fn test_covariance_projection_point_gradient() {
        // Gradient check for dependence on point_cam through the perspective Jacobian J(point_cam).
        let mut rng = StdRng::seed_from_u64(0xB01A_7CA1_u64);

        for _ in 0..200 {
            let cam_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let w = cam_q.to_rotation_matrix().into_inner();

            let g_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let r = g_q.to_rotation_matrix().into_inner();

            let log_scale = nalgebra::Vector3::new(
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
            );

            let point_cam = nalgebra::Vector3::new(
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(1.0f32..4.0f32),
            );

            let fx = rng.gen_range(100.0f32..400.0f32);
            let fy = rng.gen_range(100.0f32..400.0f32);

            let g00 = rng.gen_range(-1.0f32..1.0f32);
            let g01 = rng.gen_range(-1.0f32..1.0f32);
            let g11 = rng.gen_range(-1.0f32..1.0f32);
            let d_sigma2d = nalgebra::Matrix2::new(g00, g01, g01, g11);

            let ana = project_covariance_2d_grad_point_cam(&point_cam, fx, fy, &w, &r, &log_scale, &d_sigma2d);

            // f64 numeric derivative of L = <G, J Σ_cam Jᵀ>
            let w64 = w.map(|x| x as f64);
            let r64 = r.map(|x| x as f64);
            let g64 = d_sigma2d.map(|x| x as f64);
            let s64 = log_scale.map(|x| x as f64);

            let v = nalgebra::Vector3::new((2.0 * s64.x).exp(), (2.0 * s64.y).exp(), (2.0 * s64.z).exp());
            let d = nalgebra::Matrix3::from_diagonal(&v);
            let sigma = r64 * d * r64.transpose();
            let sigma_cam = w64 * sigma * w64.transpose();

            let loss64 = |p: nalgebra::Vector3<f64>| -> f64 {
                let j = {
                    let x = p.x;
                    let y = p.y;
                    let z = p.z;
                    let z_inv = 1.0 / z;
                    let z_inv2 = z_inv * z_inv;
                    nalgebra::Matrix2x3::new(
                        (fx as f64) * z_inv, 0.0, -(fx as f64) * x * z_inv2,
                        0.0, (fy as f64) * z_inv, -(fy as f64) * y * z_inv2,
                    )
                };
                let sigma2d = j * sigma_cam * j.transpose();
                (sigma2d.component_mul(&g64)).sum()
            };

            let eps = 1e-4f64;
            for axis in 0..3 {
                let mut plus = point_cam.map(|x| x as f64);
                let mut minus = point_cam.map(|x| x as f64);
                plus[axis] += eps;
                minus[axis] -= eps;
                let num = (loss64(plus) - loss64(minus)) / (2.0 * eps);
                let num_f32 = num as f32;

                let got = ana[axis];
                let abs_err = (got - num_f32).abs();
                assert!(
                    rel_err(got, num_f32) < 5e-4 || abs_err < 5e-4,
                    "cov proj point grad mismatch axis={axis}: num={num_f32} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, num_f32)
                );
            }
        }
    }

    #[test]
    fn test_full_projection_combined_gradients() {
        // Combined gradient check for:
        // - mean projection (u,v) w.r.t point_cam
        // - covariance projection Σ₂d w.r.t (point_cam, log_scale, rotation-vector)
        //
        // This mirrors the forward math used in the renderer at a single projected Gaussian.
        let mut rng = StdRng::seed_from_u64(0xF001_C0DE_u64);

        for _ in 0..100 {
            let cam_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let w = cam_q.to_rotation_matrix().into_inner();

            let g_q = nalgebra::UnitQuaternion::from_euler_angles(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let r0 = g_q.to_rotation_matrix().into_inner();

            let log_scale = nalgebra::Vector3::new(
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
                rng.gen_range(-4.0f32..-0.5f32),
            );

            let point_cam = nalgebra::Vector3::new(
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(-0.5f32..0.5f32),
                rng.gen_range(1.0f32..4.0f32),
            );

            let fx = rng.gen_range(100.0f32..400.0f32);
            let fy = rng.gen_range(100.0f32..400.0f32);
            let cx = 0.0f32;
            let cy = 0.0f32;

            // Loss weights:
            // - mean term: L_mean = w_uv · uv
            // - cov term:  L_cov = <G, Σ₂d>
            let w_uv = nalgebra::Vector2::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            );
            let g00 = rng.gen_range(-1.0f32..1.0f32);
            let g01 = rng.gen_range(-1.0f32..1.0f32);
            let g11 = rng.gen_range(-1.0f32..1.0f32);
            let g_cov = nalgebra::Matrix2::new(g00, g01, g01, g11);

            // Analytic gradients.
            let d_point_from_mean = project_point_grad_point_cam(&point_cam, fx, fy, &w_uv);
            let d_point_from_cov = project_covariance_2d_grad_point_cam(&point_cam, fx, fy, &w, &r0, &log_scale, &g_cov);
            let d_point = d_point_from_mean + d_point_from_cov;

            let j = perspective_jacobian(&point_cam, fx, fy);
            let d_log_scale = project_covariance_2d_grad_log_scale(&w, &j, &r0, &log_scale, &g_cov);
            let d_rot = project_covariance_2d_grad_rotation_vector_at_r0(&w, &j, &r0, &log_scale, &g_cov);

            // Numeric gradients in f64.
            let w64 = w.map(|x| x as f64);
            let r0_64 = r0.map(|x| x as f64);
            let log_scale64 = log_scale.map(|x| x as f64);
            let g_cov64 = g_cov.map(|x| x as f64);

            let d_vec = nalgebra::Vector3::new(
                (2.0 * log_scale64.x).exp(),
                (2.0 * log_scale64.y).exp(),
                (2.0 * log_scale64.z).exp(),
            );
            let d_mat = nalgebra::Matrix3::from_diagonal(&d_vec);

            let sigma = r0_64 * d_mat * r0_64.transpose();
            let _sigma_cam = w64 * sigma * w64.transpose();

            let loss64 = |p: nalgebra::Vector3<f64>, s: nalgebra::Vector3<f64>, r: nalgebra::Matrix3<f64>| -> f64 {
                let uv = {
                    let x = p.x;
                    let y = p.y;
                    let z = p.z;
                    nalgebra::Vector2::new((fx as f64) * x / z + (cx as f64), (fy as f64) * y / z + (cy as f64))
                };

                let d_vec = nalgebra::Vector3::new((2.0 * s.x).exp(), (2.0 * s.y).exp(), (2.0 * s.z).exp());
                let d_mat = nalgebra::Matrix3::from_diagonal(&d_vec);
                let sigma = r * d_mat * r.transpose();
                let sigma_cam = w64 * sigma * w64.transpose();

                let j = {
                    let x = p.x;
                    let y = p.y;
                    let z = p.z;
                    let z_inv = 1.0 / z;
                    let z_inv2 = z_inv * z_inv;
                    nalgebra::Matrix2x3::new(
                        (fx as f64) * z_inv, 0.0, -(fx as f64) * x * z_inv2,
                        0.0, (fy as f64) * z_inv, -(fy as f64) * y * z_inv2,
                    )
                };
                let sigma2d = j * sigma_cam * j.transpose();
                let l_cov = (sigma2d.component_mul(&g_cov64)).sum();
                let l_mean = (uv.x * (w_uv.x as f64)) + (uv.y * (w_uv.y as f64));
                l_mean + l_cov
            };

            // Point_cam numeric.
            let eps_p = 1e-4f64;
            let p0 = point_cam.map(|x| x as f64);
            for axis in 0..3 {
                let mut plus = p0;
                let mut minus = p0;
                plus[axis] += eps_p;
                minus[axis] -= eps_p;
                let num = (loss64(plus, log_scale64, r0_64) - loss64(minus, log_scale64, r0_64)) / (2.0 * eps_p);
                let num_f32 = num as f32;
                let got = d_point[axis];
                let abs_err = (got - num_f32).abs();
                assert!(
                    rel_err(got, num_f32) < 5e-4 || abs_err < 5e-4,
                    "combined d_point mismatch axis={axis}: num={num_f32} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, num_f32)
                );
            }

            // log_scale numeric.
            let eps_s = 1e-4f64;
            let s0 = log_scale64;
            for axis in 0..3 {
                let mut plus = s0;
                let mut minus = s0;
                plus[axis] += eps_s;
                minus[axis] -= eps_s;
                let num = (loss64(p0, plus, r0_64) - loss64(p0, minus, r0_64)) / (2.0 * eps_s);
                let num_f32 = num as f32;
                let got = d_log_scale[axis];
                let abs_err = (got - num_f32).abs();
                assert!(
                    rel_err(got, num_f32) < 5e-4 || abs_err < 5e-4,
                    "combined d_log_scale mismatch axis={axis}: num={num_f32} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, num_f32)
                );
            }

            // rotation-vector numeric around r0 (left-multiply small axis rotations).
            let rot_x = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
            };
            let rot_y = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
            };
            let rot_z = |theta: f64| -> nalgebra::Matrix3<f64> {
                let (s, c) = theta.sin_cos();
                nalgebra::Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
            };

            let eps_r = 1e-4f64;
            let num_rx = (loss64(p0, s0, rot_x(eps_r) * r0_64) - loss64(p0, s0, rot_x(-eps_r) * r0_64)) / (2.0 * eps_r);
            let num_ry = (loss64(p0, s0, rot_y(eps_r) * r0_64) - loss64(p0, s0, rot_y(-eps_r) * r0_64)) / (2.0 * eps_r);
            let num_rz = (loss64(p0, s0, rot_z(eps_r) * r0_64) - loss64(p0, s0, rot_z(-eps_r) * r0_64)) / (2.0 * eps_r);

            let num_r = nalgebra::Vector3::new(num_rx as f32, num_ry as f32, num_rz as f32);
            for axis in 0..3 {
                let got = d_rot[axis];
                let reference = num_r[axis];
                let abs_err = (got - reference).abs();
                assert!(
                    rel_err(got, reference) < 5e-4 || abs_err < 5e-4,
                    "combined d_rot mismatch axis={axis}: num={reference} ana={got} abs_err={abs_err} rel_err={}",
                    rel_err(got, reference)
                );
            }
        }
    }
}
