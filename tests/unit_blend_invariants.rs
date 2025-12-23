//! Alpha blending invariants for front-to-back compositing.

use approx::assert_relative_eq;
use nalgebra::Vector3;
use sugar_rs::diff::blend_grad::blend_forward_with_bg;

#[test]
fn test_blend_forward_with_bg_matches_manual_two_layer() {
    let alphas = vec![0.2f32, 0.5f32];
    let colors = vec![Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)];
    let bg = Vector3::new(0.1, 0.2, 0.3);

    let out = blend_forward_with_bg(&alphas, &colors, &bg).out;

    // Manual:
    // T0 = 1
    // out = T0*a0*c0 + T1*a1*c1 + T2*bg
    // T1 = 1 - a0
    // T2 = T1 * (1 - a1)
    let t0 = 1.0;
    let t1 = 1.0 - alphas[0];
    let t2 = t1 * (1.0 - alphas[1]);

    let expected = t0 * alphas[0] * colors[0]
        + t1 * alphas[1] * colors[1]
        + t2 * bg;

    assert_relative_eq!(out, expected, epsilon = 1e-6);
}

#[test]
fn test_blend_order_matters_front_to_back() {
    let alphas = vec![0.8f32, 0.8f32];
    let red = Vector3::new(1.0, 0.0, 0.0);
    let green = Vector3::new(0.0, 1.0, 0.0);
    let bg = Vector3::zeros();

    let out_rg = blend_forward_with_bg(&alphas, &[red, green], &bg).out;
    let out_gr = blend_forward_with_bg(&alphas, &[green, red], &bg).out;

    assert_ne!(out_rg, out_gr);
    assert!(out_rg.x > out_rg.y, "red-first should skew red");
    assert!(out_gr.y > out_gr.x, "green-first should skew green");
}
