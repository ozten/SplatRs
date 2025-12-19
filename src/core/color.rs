//! Color space conversion utilities
//!
//! Single source of truth for all sRGB↔linear conversions.
//! Uses official sRGB transfer function (not gamma 2.2 approximation).
//!
//! ## Color Pipeline
//!
//! All training and rendering happens in **linear RGB**. Only convert to/from sRGB
//! at system boundaries (input/output).
//!
//! ### Standard sRGB Transfer Function
//!
//! **sRGB to Linear**:
//! - if sRGB <= 0.04045: linear = sRGB / 12.92
//! - if sRGB > 0.04045: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
//!
//! **Linear to sRGB**:
//! - if linear <= 0.0031308: sRGB = 12.92 * linear
//! - if linear > 0.0031308: sRGB = 1.055 * linear ^ (1/2.4) - 0.055
//!
//! This is NOT the same as simple gamma 2.2 (`x^(1/2.2)`).

/// Convert sRGB u8 (0-255) to linear f32 (0.0-1.0).
///
/// Uses the official sRGB transfer function with breakpoint at 0.04045.
///
/// # Arguments
/// * `u` - sRGB color value (0-255)
///
/// # Returns
/// * Linear color value (0.0-1.0)
///
/// # Example
/// ```
/// use sugar_rs::core::color::srgb_u8_to_linear_f32;
///
/// // Middle gray in sRGB (128) is about 0.21 in linear space
/// let linear = srgb_u8_to_linear_f32(128);
/// assert!((linear - 0.2126).abs() < 0.01);
/// ```
pub fn srgb_u8_to_linear_f32(u: u8) -> f32 {
    let cs = (u as f32) / 255.0;
    if cs <= 0.04045 {
        cs / 12.92
    } else {
        ((cs + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear f32 (0.0-1.0) to sRGB u8 (0-255).
///
/// Uses the official sRGB inverse transfer function with breakpoint at 0.0031308.
///
/// # Arguments
/// * `x` - Linear color value (0.0-1.0)
///
/// # Returns
/// * sRGB color value (0-255)
///
/// # Example
/// ```
/// use sugar_rs::core::color::linear_f32_to_srgb_u8;
///
/// // Linear 0.5 is about sRGB 188
/// let srgb = linear_f32_to_srgb_u8(0.5);
/// assert_eq!(srgb, 188);
/// ```
pub fn linear_f32_to_srgb_u8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let cs = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (cs * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear_roundtrip() {
        for val in [0u8, 32, 64, 128, 192, 255] {
            let linear = srgb_u8_to_linear_f32(val);
            let back = linear_f32_to_srgb_u8(linear);
            assert_eq!(back, val, "Roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_linear_to_srgb_roundtrip() {
        for val in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let srgb = linear_f32_to_srgb_u8(val);
            let back = srgb_u8_to_linear_f32(srgb);
            // Allow small error due to quantization
            assert!((back - val).abs() < 0.005,
                "Roundtrip failed for {}: got {}", val, back);
        }
    }

    #[test]
    fn test_black_and_white() {
        assert_eq!(srgb_u8_to_linear_f32(0), 0.0);
        assert_eq!(srgb_u8_to_linear_f32(255), 1.0);
        assert_eq!(linear_f32_to_srgb_u8(0.0), 0);
        assert_eq!(linear_f32_to_srgb_u8(1.0), 255);
    }

    #[test]
    fn test_middle_gray() {
        // sRGB 128 (middle gray) should be about 0.21 linear (not 0.5!)
        let linear = srgb_u8_to_linear_f32(128);
        assert!((linear - 0.2126).abs() < 0.01);

        // Linear 0.5 should be about sRGB 188 (not 128!)
        let srgb = linear_f32_to_srgb_u8(0.5);
        assert_eq!(srgb, 188);
    }

    #[test]
    fn test_not_simple_gamma_2_2() {
        // Verify we're NOT using simple gamma 2.2
        // If it were x^(1/2.2), then:
        let val = 128u8;
        let linear_actual = srgb_u8_to_linear_f32(val);
        let linear_gamma_2_2 = ((val as f32 / 255.0).powf(2.2));

        // These should be different (sRGB has a linear segment at low values)
        assert!((linear_actual - linear_gamma_2_2).abs() > 0.001,
            "Using incorrect gamma 2.2 instead of sRGB curve!");
    }
}
