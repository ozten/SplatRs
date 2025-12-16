//! Loss functions for training (M7+).

use nalgebra::Vector3;

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
