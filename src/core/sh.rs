//! Spherical harmonics evaluation for view-dependent color.
//!
//! Gaussians store color as spherical harmonics coefficients rather than
//! a single RGB value. This allows view-dependent effects (specular highlights, etc.)
//!
//! We use degree-3 SH, which requires 16 coefficients per color channel.

use nalgebra::Vector3;

/// Evaluate spherical harmonics basis functions up to degree 3 (real SH).
///
/// `direction` must be normalized.
///
/// Ordering matches the docs and common 3DGS implementations:
/// - Degree 0:  Y_0^0
/// - Degree 1:  Y_1^{-1}, Y_1^0, Y_1^1
/// - Degree 2:  Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
/// - Degree 3:  Y_3^{-3}, Y_3^{-2}, Y_3^{-1}, Y_3^0, Y_3^1, Y_3^2, Y_3^3
pub fn sh_basis(direction: &Vector3<f32>) -> [f32; 16] {
    let x = direction.x;
    let y = direction.y;
    let z = direction.z;

    // Constants for real SH basis (degree 0..3)
    // Commonly used in 3DGS / nerfstudio / instant-ngp implementations.
    const C0: f32 = 0.282_094_791_773_878_14;
    const C1: f32 = 0.488_602_511_902_919_9;
    const C2_0: f32 = 1.092_548_430_592_079_2;
    const C2_1: f32 = 0.315_391_565_252_520_05;
    const C2_2: f32 = 0.546_274_215_296_039_6;
    const C3_0: f32 = 0.590_043_589_926_643_5;
    const C3_1: f32 = 2.890_611_442_640_554;
    const C3_2: f32 = 0.457_045_799_464_465_8;
    const C3_3: f32 = 0.373_176_332_590_115_4;
    const C3_4: f32 = 1.445_305_721_320_277;
    const C3_5: f32 = 0.590_043_589_926_643_5;

    let mut basis = [0.0f32; 16];

    // l = 0
    basis[0] = C0;

    // l = 1
    basis[1] = C1 * y; // Y_1^{-1}
    basis[2] = C1 * z; // Y_1^{0}
    basis[3] = C1 * x; // Y_1^{1}

    // Precompute monomials
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    // l = 2
    basis[4] = C2_0 * xy; // Y_2^{-2}
    basis[5] = C2_0 * yz; // Y_2^{-1}
    basis[6] = C2_1 * (3.0 * z2 - 1.0); // Y_2^{0}
    basis[7] = C2_0 * xz; // Y_2^{1}
    basis[8] = C2_2 * (x2 - y2); // Y_2^{2}

    // l = 3
    basis[9] = C3_0 * y * (3.0 * x2 - y2); // Y_3^{-3}
    basis[10] = C3_1 * xy * z; // Y_3^{-2}
    basis[11] = C3_2 * y * (5.0 * z2 - 1.0); // Y_3^{-1}
    basis[12] = C3_3 * z * (5.0 * z2 - 3.0); // Y_3^{0}
    basis[13] = C3_2 * x * (5.0 * z2 - 1.0); // Y_3^{1}
    basis[14] = C3_4 * z * (x2 - y2); // Y_3^{2}
    basis[15] = C3_5 * x * (x2 - 3.0 * y2); // Y_3^{3}

    basis
}

/// Evaluate view-dependent color from SH coefficients.
///
/// Given:
/// - SH coefficients: [[f32; 3]; 16] (RGB × 16 basis functions)
/// - View direction: normalized vector
///
/// Returns: RGB color at this viewing angle
pub fn evaluate_sh(sh_coeffs: &[[f32; 3]; 16], direction: &Vector3<f32>) -> Vector3<f32> {
    // Normalize direction (should already be normalized, but be safe)
    let dir = direction.normalize();

    // Get basis function values
    let basis = sh_basis(&dir);

    // Compute color as dot product of basis with coefficients
    let mut color = Vector3::<f32>::zeros();
    for i in 0..16 {
        color.x += basis[i] * sh_coeffs[i][0]; // R channel
        color.y += basis[i] * sh_coeffs[i][1]; // G channel
        color.z += basis[i] * sh_coeffs[i][2]; // B channel
    }

    // Clamp to valid color range [0, 1]
    // (SH can produce negative values or values > 1)
    Vector3::new(
        color.x.clamp(0.0, 1.0),
        color.y.clamp(0.0, 1.0),
        color.z.clamp(0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_basis_dc_component() {
        // DC component should be constant regardless of direction
        let dir1 = Vector3::new(1.0, 0.0, 0.0);
        let dir2 = Vector3::new(0.0, 1.0, 0.0);

        let basis1 = sh_basis(&dir1.normalize());
        let basis2 = sh_basis(&dir2.normalize());

        // Y_0^0 should be the same
        approx::assert_relative_eq!(basis1[0], basis2[0], epsilon = 1e-6);
        approx::assert_relative_eq!(basis1[0], 0.282_094_8, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_sh_dc_only() {
        // With only DC coefficients set, color should be view-independent
        let mut sh_coeffs = [[0.0f32; 3]; 16];
        sh_coeffs[0] = [1.0, 0.5, 0.2]; // DC component only

        let dir1 = Vector3::new(1.0, 0.0, 0.0);
        let dir2 = Vector3::new(0.0, 0.0, 1.0);

        let color1 = evaluate_sh(&sh_coeffs, &dir1);
        let color2 = evaluate_sh(&sh_coeffs, &dir2);

        // Should be the same from all directions
        approx::assert_relative_eq!(color1, color2, epsilon = 1e-5);
    }

    #[test]
    fn test_sh_basis_x_axis_values() {
        // Direction along +X.
        let dir = Vector3::new(1.0, 0.0, 0.0);
        let basis = sh_basis(&dir);

        // l=1: only Y_1^1 (x) is non-zero.
        approx::assert_relative_eq!(basis[1], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(basis[2], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(basis[3], 0.488_602_52, epsilon = 1e-6);

        // l=2: Y_2^0 = C2_1*(3z^2-1) with z=0 => -C2_1, and Y_2^2 = C2_2*(x^2-y^2) => C2_2
        approx::assert_relative_eq!(basis[6], -0.315_391_57, epsilon = 1e-6);
        approx::assert_relative_eq!(basis[8], 0.546_274_24, epsilon = 1e-6);

        // l=3: Y_3^1 = C3_2*x*(5z^2-1) => -C3_2, Y_3^3 = C3_5*x*(x^2-3y^2) => C3_5
        approx::assert_relative_eq!(basis[13], -0.457_045_8, epsilon = 1e-6);
        approx::assert_relative_eq!(basis[15], 0.590_043_6, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_sh_single_basis_matches_basis_value() {
        // If only one coefficient is set to 1, the resulting color channel should equal that basis value.
        let dir = Vector3::new(0.3, -0.4, 0.866_025_4).normalize();
        let basis = sh_basis(&dir);

        let mut sh_coeffs = [[0.0f32; 3]; 16];
        sh_coeffs[7][0] = 1.0; // Only affect red channel with Y_2^1

        let color = evaluate_sh(&sh_coeffs, &dir);
        approx::assert_relative_eq!(color.x, basis[7].clamp(0.0, 1.0), epsilon = 1e-6);
        approx::assert_relative_eq!(color.y, 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(color.z, 0.0, epsilon = 1e-6);
    }
}
