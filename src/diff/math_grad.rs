//! Gradients for small scalar math utilities (sigmoid, inverse_sigmoid, etc.).
//!
//! These are foundational for M6 gradient checks.

/// Derivative of sigmoid σ(x) = 1 / (1 + e^{-x}) with respect to x.
///
/// dσ/dx = σ(x) * (1 - σ(x))
pub fn sigmoid_grad_from_sigmoid(sigmoid_x: f32) -> f32 {
    sigmoid_x * (1.0 - sigmoid_x)
}

/// Derivative of logit(p) = log(p / (1-p)) with respect to p.
///
/// d/dp logit(p) = 1 / (p * (1 - p))
///
/// Note: `core::inverse_sigmoid` clamps `p` into [1e-6, 1-1e-6] to avoid infinities.
/// For gradient checking, keep `p` away from those clamps so the derivative matches.
pub fn inverse_sigmoid_grad(p: f32) -> f32 {
    1.0 / (p * (1.0 - p))
}
