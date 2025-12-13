//! Spherical harmonics evaluation for view-dependent color.
//!
//! Gaussians store color as spherical harmonics coefficients rather than
//! a single RGB value. This allows view-dependent effects (specular highlights, etc.)
//!
//! We use degree-3 SH, which requires 16 coefficients per color channel.

use nalgebra::Vector3;

/// Evaluate spherical harmonics basis functions up to degree 3.
///
/// Given a normalized direction vector, returns the 16 SH basis function values.
///
/// The basis functions are ordered as:
/// - Degree 0 (1 function): Y_0^0
/// - Degree 1 (3 functions): Y_1^{-1}, Y_1^0, Y_1^1
/// - Degree 2 (5 functions): Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
/// - Degree 3 (7 functions): Y_3^{-3}, Y_3^{-2}, Y_3^{-1}, Y_3^0, Y_3^1, Y_3^2, Y_3^3
pub fn sh_basis(direction: &Vector3<f32>) -> [f32; 16] {
    // TODO: Implement for M5
    // This is standard SH math - can reference Wikipedia or the 3DGS paper appendix
    //
    // For now, return just the DC component (constant illumination)
    // This is enough to get started - view-independent color

    let mut basis = [0.0f32; 16];

    // Y_0^0 = 0.28209479177387814 (constant)
    basis[0] = 0.28209479;

    // TODO: Implement degrees 1-3 for view-dependent color
    // See: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    // Or 3DGS paper supplementary material

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
        color.x += basis[i] * sh_coeffs[i][0];  // R channel
        color.y += basis[i] * sh_coeffs[i][1];  // G channel
        color.z += basis[i] * sh_coeffs[i][2];  // B channel
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

        let basis1 = sh_basis(&dir1);
        let basis2 = sh_basis(&dir2);

        // Y_0^0 should be the same
        approx::assert_relative_eq!(basis1[0], basis2[0], epsilon = 1e-6);
        approx::assert_relative_eq!(basis1[0], 0.28209479, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_sh_dc_only() {
        // With only DC coefficients set, color should be view-independent
        let mut sh_coeffs = [[0.0f32; 3]; 16];
        sh_coeffs[0] = [1.0, 0.5, 0.2];  // DC component only

        let dir1 = Vector3::new(1.0, 0.0, 0.0);
        let dir2 = Vector3::new(0.0, 0.0, 1.0);

        let color1 = evaluate_sh(&sh_coeffs, &dir1);
        let color2 = evaluate_sh(&sh_coeffs, &dir2);

        // Should be the same from all directions
        approx::assert_relative_eq!(color1, color2, epsilon = 1e-5);
    }
}
