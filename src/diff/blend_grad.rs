//! Gradients for alpha blending (front-to-back compositing).
//!
//! Forward (scalar, per-pixel, per-channel identical):
//!   T_0 = 1
//!   for i in 0..N:
//!     out += T_i * a_i * c_i
//!     T_{i+1} = T_i * (1 - a_i)
//!
//! where:
//! - `a_i` is the per-pixel alpha contribution of gaussian i (already includes opacity * weight, and any clamp)
//! - `c_i` is RGB color
//! - `T_i` is transmittance before applying gaussian i
//!
//! We provide gradients for the common case where `a_i` and `c_i` are the differentiable inputs.

use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub struct BlendForward {
    pub out: Vector3<f32>,
    /// T_i for i=0..=N (length N+1)
    pub transmittance: Vec<f32>,
}

/// Forward alpha compositing, returning the output color and saved transmittances.
pub fn blend_forward(alphas: &[f32], colors: &[Vector3<f32>]) -> BlendForward {
    assert_eq!(alphas.len(), colors.len());

    let mut out = Vector3::<f32>::zeros();
    let mut transmittance = Vec::with_capacity(alphas.len() + 1);
    let mut t = 1.0f32;
    transmittance.push(t);

    for (a, c) in alphas.iter().copied().zip(colors.iter().copied()) {
        out += t * a * c;
        t *= 1.0 - a;
        transmittance.push(t);
    }

    BlendForward { out, transmittance }
}

#[derive(Clone, Debug)]
pub struct BlendGrads {
    pub d_alphas: Vec<f32>,
    pub d_colors: Vec<Vector3<f32>>,
}

/// Backward pass for alpha compositing.
///
/// Inputs:
/// - `alphas`, `colors`: same as forward
/// - `forward`: output of `blend_forward` (contains all T_i)
/// - `d_out`: upstream gradient dL/d(out)
///
/// Returns:
/// - gradients w.r.t. alphas and colors
pub fn blend_backward(
    alphas: &[f32],
    colors: &[Vector3<f32>],
    forward: &BlendForward,
    d_out: &Vector3<f32>,
) -> BlendGrads {
    assert_eq!(alphas.len(), colors.len());
    assert_eq!(forward.transmittance.len(), alphas.len() + 1);

    let n = alphas.len();
    let mut d_alphas = vec![0.0f32; n];
    let mut d_colors = vec![Vector3::<f32>::zeros(); n];

    // We backprop through:
    // out = sum_i T_i * a_i * c_i
    // T_{i+1} = T_i * (1 - a_i)
    //
    // We'll treat the transmittance recurrence explicitly with a reverse scan.
    //
    // Let g_Ti be dL/d(T_i). We have:
    // - out depends on T_i directly via term_i = T_i * a_i * c_i
    // - T_{i+1} depends on T_i
    //
    // Initialize g_T_{N} = 0 (out does not depend on final transmittance directly).
    let mut g_t_next = 0.0f32; // g_T_{i+1} as we go backwards

    for i in (0..n).rev() {
        let a_i = alphas[i];
        let c_i = colors[i];
        let t_i = forward.transmittance[i];

        // out contribution: out += T_i * a_i * c_i
        // dL/dc_i = dL/dout * d(out)/d(c_i) = d_out * (T_i * a_i)
        d_colors[i] = *d_out * (t_i * a_i);

        // dL/da_i has two parts:
        // (1) direct from out term: d_out · (T_i * c_i)
        // (2) indirect via T_{i+1} = T_i * (1 - a_i) affecting future terms through g_T_{i+1}
        //     dT_{i+1}/da_i = -T_i
        let direct = d_out.dot(&(c_i * t_i));
        let indirect = g_t_next * (-t_i);
        d_alphas[i] = direct + indirect;

        // Now accumulate g_T_i:
        // out term contributes: d_out · (a_i * c_i)
        // future transmittance contributes: g_T_{i+1} * dT_{i+1}/dT_i = g_T_{i+1} * (1 - a_i)
        let g_t_i_from_out = d_out.dot(&(c_i * a_i));
        let g_t_i_from_next = g_t_next * (1.0 - a_i);
        let g_t_i = g_t_i_from_out + g_t_i_from_next;

        g_t_next = g_t_i;
    }

    BlendGrads { d_alphas, d_colors }
}

