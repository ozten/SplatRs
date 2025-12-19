//! Color space management for image loading
//!
//! This module handles color profile conversion to ensure all images
//! are in sRGB color space for consistent training.

use image::RgbImage;
use std::path::Path;

/// Load an image and convert to sRGB color space if needed.
///
/// This function:
/// 1. Loads the image using the `image` crate
/// 2. Extracts embedded color profile (if any) using ImageMagick or exiftool
/// 3. Converts from source profile to sRGB using lcms2 if needed
/// 4. Returns an sRGB RgbImage
///
/// # Arguments
/// * `path` - Path to the image file
///
/// # Returns
/// * `Ok(RgbImage)` - Image in sRGB color space
/// * `Err(String)` - Error message if loading/conversion fails
pub fn load_image_to_srgb(path: &Path) -> Result<RgbImage, String> {
    // Load image using image crate (which ignores color profiles)
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image: {}", e))?
        .to_rgb8();

    // Try to extract embedded color profile
    let profile_data = extract_color_profile(path)?;

    if let Some(profile_bytes) = profile_data {
        // Image has an embedded profile - convert to sRGB
        convert_to_srgb(&img, &profile_bytes)
    } else {
        // No profile found - assume already sRGB
        Ok(img)
    }
}

/// Extract embedded ICC color profile from an image file.
///
/// Uses `exiftool` to extract the ICC profile binary data.
/// Returns None if no profile is embedded or exiftool is not available.
fn extract_color_profile(path: &Path) -> Result<Option<Vec<u8>>, String> {
    use std::process::Command;

    // Try to extract ICC profile using exiftool
    let output = Command::new("exiftool")
        .arg("-icc_profile")
        .arg("-b")  // Binary output
        .arg(path)
        .output();

    match output {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            Ok(Some(out.stdout))
        }
        Ok(_) => {
            // exiftool succeeded but no profile found
            Ok(None)
        }
        Err(_) => {
            // exiftool not available - fall back to checking file metadata
            // For now, just return None and assume sRGB
            Ok(None)
        }
    }
}

/// Convert an RGB image from its source color profile to sRGB.
///
/// # Arguments
/// * `img` - Source image (pixel values in source color space)
/// * `source_profile_data` - ICC profile binary data for source color space
///
/// # Returns
/// * `Ok(RgbImage)` - Converted image in sRGB color space
/// * `Err(String)` - Conversion error
fn convert_to_srgb(img: &RgbImage, source_profile_data: &[u8]) -> Result<RgbImage, String> {
    use lcms2::*;

    // Create source profile from embedded ICC data
    let source_profile = Profile::new_icc(source_profile_data)
        .map_err(|e| format!("Failed to parse source ICC profile: {:?}", e))?;

    // Create sRGB destination profile
    let srgb_profile = Profile::new_srgb();

    // Create color transform
    let transform = Transform::new(
        &source_profile,
        PixelFormat::RGB_8,
        &srgb_profile,
        PixelFormat::RGB_8,
        Intent::Perceptual,
    )
    .map_err(|e| format!("Failed to create color transform: {:?}", e))?;

    // Convert pixels
    let mut output = RgbImage::new(img.width(), img.height());
    let input_pixels = img.as_raw();
    let output_pixels = output.as_mut();

    transform.transform_pixels(input_pixels, output_pixels);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires test image with Display P3 profile
    fn test_p3_to_srgb_conversion() {
        // Test with IMG_0910.jpg which has Display P3 profile
        let test_image = Path::new("datasets/dollhouse_sm/input/IMG_0910.jpg");

        if !test_image.exists() {
            println!("Skipping test - image not found");
            return;
        }

        let srgb_img = load_image_to_srgb(test_image).unwrap();

        println!("Converted image: {}×{}", srgb_img.width(), srgb_img.height());

        // Save for visual inspection
        srgb_img.save("/tmp/test_p3_converted_to_srgb.png").unwrap();
        println!("Saved converted image to /tmp/test_p3_converted_to_srgb.png");
    }
}
