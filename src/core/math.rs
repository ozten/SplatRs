//! Mathematical utilities (quaternions, activation functions, etc.).

use nalgebra::{Matrix3, UnitQuaternion};

/// Convert a unit quaternion to a 3×3 rotation matrix.
///
/// Formula (from quaternion q = w + xi + yj + zk):
/// R = | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)  |
///     | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)  |
///     | 2(xz-wy)     2(yz+wx)    1-2(x²+y²)|
pub fn quaternion_to_matrix(q: &UnitQuaternion<f32>) -> Matrix3<f32> {
    // nalgebra's UnitQuaternion already has to_rotation_matrix()
    // But we implement it explicitly for educational purposes

    // TODO: Implement for M3
    // For now, use nalgebra's built-in
    q.to_rotation_matrix().into_inner()
}

/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
///
/// Maps R → (0, 1)
/// Used for opacity (converts unbounded optimization to valid probability)
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse sigmoid (logit): logit(p) = log(p / (1-p))
///
/// Maps (0, 1) → R
/// Used to convert initial opacity values to optimization space
pub fn inverse_sigmoid(p: f32) -> f32 {
    // Clamp to avoid log(0) or division by zero
    let p_clamped = p.clamp(1e-6, 1.0 - 1e-6);
    (p_clamped / (1.0 - p_clamped)).ln()
}

/// Compute the Jacobian of perspective projection.
///
/// For a point p_cam = [x, y, z] in camera space,
/// the projected point is [u, v] = [fx*x/z + cx, fy*y/z + cy]
///
/// The Jacobian ∂[u,v]/∂[x,y,z] is:
/// J = | fx/z    0      -fx*x/z² |
///     |  0     fy/z    -fy*y/z² |
///
/// This is critical for projecting 3D covariance to 2D.
pub fn perspective_jacobian(
    point_camera: &nalgebra::Vector3<f32>,
    fx: f32,
    fy: f32,
) -> nalgebra::Matrix2x3<f32> {
    let x = point_camera.x;
    let y = point_camera.y;
    let z = point_camera.z;

    let z_inv = 1.0 / z;
    let z_inv_sq = z_inv * z_inv;

    // J = | fx/z    0      -fx*x/z² |
    //     |  0     fy/z    -fy*y/z² |
    nalgebra::Matrix2x3::new(
        fx * z_inv,
        0.0,
        -fx * x * z_inv_sq,
        0.0,
        fy * z_inv,
        -fy * y * z_inv_sq,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_sigmoid_inverse_roundtrip() {
        let p = 0.7;
        let x = inverse_sigmoid(p);
        let p_recovered = sigmoid(x);
        assert_relative_eq!(p, p_recovered, epsilon = 1e-6);
    }

    #[test]
    fn test_quaternion_to_matrix_identity() {
        let q = UnitQuaternion::identity();
        let r = quaternion_to_matrix(&q);
        assert_relative_eq!(r, Matrix3::identity(), epsilon = 1e-6);
    }

    #[test]
    fn test_quaternion_to_matrix_orthogonal() {
        // Any rotation matrix should be orthogonal: R * R^T = I
        let q = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let r = quaternion_to_matrix(&q);
        let product = r * r.transpose();
        assert_relative_eq!(product, Matrix3::identity(), epsilon = 1e-5);
    }

    #[test]
    fn test_quaternion_to_matrix_determinant() {
        // Rotation matrices have determinant +1
        let q = UnitQuaternion::from_euler_angles(0.5, -0.3, 1.2);
        let r = quaternion_to_matrix(&q);
        assert_relative_eq!(r.determinant(), 1.0, epsilon = 1e-5);
    }
}
