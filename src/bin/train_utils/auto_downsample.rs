//! Automatic downsample factor calculation based on GPU buffer constraints.
//!
//! This module calculates the optimal downsample factor to ensure training images
//! fit within GPU memory limits (specifically the max_storage_buffer_binding_size).

use image::GenericImageView;
use std::path::Path;

/// Calculate required downsample factor to fit image within GPU buffer limits.
///
/// Prefers power-of-2 divisors (1/2, 1/4, 1/8, 1/16, etc.) for clean box filtering.
/// Falls back to fractional downsample only if no power-of-2 fits.
///
/// # Arguments
/// * `image_width` - Original image width in pixels
/// * `image_height` - Original image height in pixels
/// * `max_buffer_size` - GPU's max storage buffer binding size in bytes
///
/// # Returns
/// Downsample factor between 0.01 and 1.0. Returns 1.0 if no downsampling needed.
pub fn calculate_downsample_factor(
    image_width: u32,
    image_height: u32,
    max_buffer_size: u64,
) -> f32 {
    // Intermediates buffer calculation (from renderer.rs:675-677):
    // intermediates_bytes = num_pixels × MAX_CONTRIBUTIONS_PER_PIXEL × sizeof(ContributionGPU)
    //                     = (width × height) × 16 × 16 bytes
    //                     = (width × height) × 256 bytes
    const MAX_CONTRIBUTIONS_PER_PIXEL: u64 = 16;
    const CONTRIBUTION_SIZE_BYTES: u64 = 16; // 4 f32s
    const BYTES_PER_PIXEL: u64 = MAX_CONTRIBUTIONS_PER_PIXEL * CONTRIBUTION_SIZE_BYTES;

    let num_pixels = (image_width as u64) * (image_height as u64);
    let buffer_needed = num_pixels * BYTES_PER_PIXEL;

    if buffer_needed <= max_buffer_size {
        // No downsampling needed
        return 1.0;
    }

    // Try power-of-2 divisors first (2, 4, 8, 16, 32, 64)
    // These enable clean box filtering with perfect anti-aliasing
    for divisor in [2u32, 4, 8, 16, 32, 64] {
        let downsampled_width = (image_width + divisor - 1) / divisor; // Round up
        let downsampled_height = (image_height + divisor - 1) / divisor;
        let downsampled_pixels = (downsampled_width as u64) * (downsampled_height as u64);
        let downsampled_buffer = downsampled_pixels * BYTES_PER_PIXEL;

        if downsampled_buffer <= max_buffer_size {
            // Found a power-of-2 that fits!
            return 1.0 / (divisor as f32);
        }
    }

    // No power-of-2 fits - fall back to fractional downsample
    // Calculate required downsample factor
    // buffer_needed * downsample^2 <= max_buffer_size
    // downsample^2 <= max_buffer_size / buffer_needed
    // downsample <= sqrt(max_buffer_size / buffer_needed)
    let downsample_squared = (max_buffer_size as f64) / (buffer_needed as f64);
    let downsample = downsample_squared.sqrt() as f32;

    // Round down slightly to add safety margin (5%)
    (downsample * 0.95).max(0.01) // Ensure we don't go below 1%
}

/// Check if a downsample factor is a clean power-of-2 (1/2, 1/4, 1/8, etc.)
pub fn is_power_of_2_downsample(factor: f32) -> Option<u32> {
    const EPSILON: f32 = 0.001;

    for divisor in [2u32, 4, 8, 16, 32, 64] {
        let expected = 1.0 / (divisor as f32);
        if (factor - expected).abs() < EPSILON {
            return Some(divisor);
        }
    }
    None
}

/// Get GPU max storage buffer binding size by initializing a minimal GPU context.
///
/// Falls back to 128 MB (Apple Silicon Metal limit) if GPU init fails.
#[cfg(feature = "gpu")]
pub fn get_gpu_max_buffer_size() -> u64 {
    match sugar_rs::gpu::GpuContext::new_blocking() {
        Ok(ctx) => ctx.device.limits().max_storage_buffer_binding_size as u64,
        Err(_) => {
            eprintln!("Warning: Failed to initialize GPU, assuming 128 MB buffer limit");
            128 * 1024 * 1024
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub fn get_gpu_max_buffer_size() -> u64 {
    // Conservative default (Apple Silicon Metal limit)
    128 * 1024 * 1024
}

/// Automatically determine optimal downsample factor for a dataset.
///
/// Reads the first image from the dataset and calculates the downsample factor
/// needed to fit within GPU buffer limits.
///
/// # Arguments
/// * `first_image_path` - Path to the first image in the dataset
/// * `max_buffer_size` - GPU's max storage buffer binding size in bytes
///
/// # Returns
/// * `Ok((downsample_factor, width, height))` - Calculated downsample and original dimensions
/// * `Err(message)` - Error message if image cannot be loaded
pub fn determine_auto_downsample(
    first_image_path: &Path,
    max_buffer_size: u64,
) -> Result<(f32, u32, u32), String> {
    let img = image::open(first_image_path)
        .map_err(|e| format!("Failed to open image: {}", e))?;

    let (width, height) = img.dimensions();
    let downsample = calculate_downsample_factor(width, height, max_buffer_size);

    Ok((downsample, width, height))
}

/// Print detailed warning about auto-downsampling configuration.
///
/// # Arguments
/// * `original_width` - Original image width
/// * `original_height` - Original image height
/// * `max_buffer_mb` - GPU buffer limit in MB
/// * `downsample_factor` - Calculated downsample factor
pub fn print_auto_downsample_warning(
    original_width: u32,
    original_height: u32,
    max_buffer_mb: u64,
    downsample_factor: f32,
) {
    let downsampled_width = (original_width as f32 * downsample_factor) as u32;
    let downsampled_height = (original_height as f32 * downsample_factor) as u32;

    eprintln!("⚠️  WARNING: Auto-downsampling enabled");
    eprintln!("   Original resolution: {}×{} ({:.1} MP)",
        original_width, original_height,
        (original_width * original_height) as f64 / 1_000_000.0);
    eprintln!("   GPU buffer limit: {} MB", max_buffer_mb);

    if let Some(divisor) = is_power_of_2_downsample(downsample_factor) {
        eprintln!("   Using power-of-2 divisor: 1/{} (box filter, no aliasing)", divisor);
    } else {
        eprintln!("   Using fractional downsample: {:.3} (nearest-neighbor)", downsample_factor);
    }

    eprintln!("   Training resolution: {}×{} ({:.1} MP)",
        downsampled_width, downsampled_height,
        (downsampled_width * downsampled_height) as f64 / 1_000_000.0);
    eprintln!("   To disable auto-downsampling, explicitly set --downsample <factor>");
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_downsampling_needed() {
        let max_buffer = 128 * 1024 * 1024; // 128 MB
        let downsample = calculate_downsample_factor(800, 600, max_buffer);
        assert_eq!(downsample, 1.0);
    }

    #[test]
    fn test_large_image_requires_downsampling() {
        let max_buffer = 128 * 1024 * 1024; // 128 MB
        let downsample = calculate_downsample_factor(5712, 4284, max_buffer);

        // Should select 1/8 (0.125) as the power-of-2 divisor
        assert_eq!(downsample, 0.125);

        // Verify it's recognized as power-of-2
        assert_eq!(is_power_of_2_downsample(downsample), Some(8));

        // Verify result fits in buffer
        let new_w = (5712 + 8 - 1) / 8; // 714
        let new_h = (4284 + 8 - 1) / 8; // 536
        let buffer_needed = (new_w as u64) * (new_h as u64) * 256;
        assert!(buffer_needed <= max_buffer);
    }

    #[test]
    fn test_power_of_2_preference() {
        // Test that we prefer power-of-2 divisors over fractional
        let max_buffer = 128 * 1024 * 1024;

        // Image that needs downsampling but fits with 1/2
        let downsample = calculate_downsample_factor(1024, 768, max_buffer);
        assert_eq!(downsample, 0.5); // Should be exactly 1/2
        assert_eq!(is_power_of_2_downsample(downsample), Some(2));

        // Image that needs 1/4
        let downsample = calculate_downsample_factor(2048, 1536, max_buffer);
        assert_eq!(downsample, 0.25); // Should be exactly 1/4
        assert_eq!(is_power_of_2_downsample(downsample), Some(4));

        // Image that needs 1/8
        let downsample = calculate_downsample_factor(4096, 3072, max_buffer);
        assert_eq!(downsample, 0.125); // Should be exactly 1/8
        assert_eq!(is_power_of_2_downsample(downsample), Some(8));
    }
}
