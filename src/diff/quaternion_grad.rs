//! Gradients for quaternion -> rotation matrix conversion.
//!
//! We treat the optimization parameters as an unconstrained 4-vector `q_raw` and
//! normalize internally:
//!   q = q_raw / ||q_raw||
//!
//! The rotation matrix uses the standard unit-quaternion formula with component
//! order (w, x, y, z).
//!
//! This matches the common 3DGS parameterization and avoids the "unit length"
//! constraint leaking into the optimizer state.

use nalgebra::{Matrix3, Vector4};

/// Convert a raw quaternion (w,x,y,z) to a rotation matrix by normalizing it first.
pub fn quaternion_raw_to_matrix(q_raw: &Vector4<f32>) -> Matrix3<f32> {
    let n = q_raw.norm();
    let q = q_raw / n;

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    Matrix3::new(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
}

/// Gradient of `quaternion_raw_to_matrix` w.r.t. `q_raw`, given upstream `d_r`.
///
/// `d_r` is dL/dR (3×3). Returns dL/dq_raw as (w,x,y,z).
pub fn quaternion_raw_to_matrix_grad(q_raw: &Vector4<f32>, d_r: &Matrix3<f32>) -> Vector4<f32> {
    let n = q_raw.norm();
    let q = q_raw / n;

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // Let L = sum_ij d_r[i,j] * R[i,j].
    // Compute dL/dq (unit quaternion) by differentiating the closed form.
    let g00 = d_r[(0, 0)];
    let g01 = d_r[(0, 1)];
    let g02 = d_r[(0, 2)];
    let g10 = d_r[(1, 0)];
    let g11 = d_r[(1, 1)];
    let g12 = d_r[(1, 2)];
    let g20 = d_r[(2, 0)];
    let g21 = d_r[(2, 1)];
    let g22 = d_r[(2, 2)];

    // dL/dw
    let dw = g01 * (-2.0 * z)
        + g02 * (2.0 * y)
        + g10 * (2.0 * z)
        + g12 * (-2.0 * x)
        + g20 * (-2.0 * y)
        + g21 * (2.0 * x);

    // dL/dx
    let dx = g01 * (2.0 * y)
        + g02 * (2.0 * z)
        + g10 * (2.0 * y)
        + g11 * (-4.0 * x)
        + g12 * (-2.0 * w)
        + g20 * (2.0 * z)
        + g21 * (2.0 * w)
        + g22 * (-4.0 * x);

    // dL/dy
    let dy = g00 * (-4.0 * y)
        + g01 * (2.0 * x)
        + g02 * (2.0 * w)
        + g10 * (2.0 * x)
        + g12 * (2.0 * z)
        + g20 * (-2.0 * w)
        + g21 * (2.0 * z)
        + g22 * (-4.0 * y);

    // dL/dz
    let dz = g00 * (-4.0 * z)
        + g01 * (-2.0 * w)
        + g02 * (2.0 * x)
        + g10 * (2.0 * w)
        + g11 * (-4.0 * z)
        + g12 * (2.0 * y)
        + g20 * (2.0 * x)
        + g21 * (2.0 * y);

    let grad_unit = Vector4::new(dw, dx, dy, dz);

    // Backprop through normalization: q = q_raw / ||q_raw||.
    // dL/dq_raw = (I - q q^T) / ||q_raw|| * dL/dq
    let dot = q.dot(&grad_unit);
    (grad_unit - q * dot) / n
}
