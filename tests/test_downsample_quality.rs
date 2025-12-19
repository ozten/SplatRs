//! Image downsampling quality tests
//!
//! These tests ensure our downsampling code preserves image quality
//! and doesn't introduce artifacts.

use image::RgbImage;
use nalgebra::Vector3;
use sugar_rs::render::full_diff::{downsample_rgb_box, downsample_rgb_nearest, rgb8_to_linear_vec};

/// Compute PSNR between two linear RGB images (higher is better, >40 dB is excellent)
fn compute_psnr(img1: &[Vector3<f32>], img2: &[Vector3<f32>]) -> f32 {
    if img1.len() != img2.len() || img1.is_empty() {
        return 0.0;
    }

    let mse: f32 = img1
        .iter()
        .zip(img2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff.norm_squared()
        })
        .sum::<f32>()
        / (img1.len() as f32);

    if mse < 1e-10 {
        100.0 // Near-identical
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// Convert RgbImage to linear Vec for PSNR comparison
fn image_to_linear(img: &RgbImage) -> Vec<Vector3<f32>> {
    rgb8_to_linear_vec(img)
}

#[test]
fn test_no_downsampling_preserves_quality() {
    // Load the reference image (already at 714×535)
    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dollhouse_clean.jpg");
    let original = image::open(img_path)
        .expect("Failed to load test image")
        .to_rgb8();

    println!(
        "Original image: {}×{} ({} bytes)",
        original.width(),
        original.height(),
        original.len()
    );

    // Test 1: No downsampling (divisor=1) should return identical image
    let processed = downsample_rgb_box(&original, 1);

    assert_eq!(processed.width(), original.width());
    assert_eq!(processed.height(), original.height());

    let original_linear = image_to_linear(&original);
    let processed_linear = image_to_linear(&processed);
    let psnr = compute_psnr(&original_linear, &processed_linear);

    println!("PSNR with divisor=1 (no downsampling): {:.2} dB", psnr);

    // Should be identical or near-identical (PSNR > 50 dB)
    assert!(
        psnr > 50.0,
        "No downsampling should preserve image quality. Got PSNR={:.2} dB, expected >50 dB",
        psnr
    );
}

#[test]
fn test_power_of_2_downsampling_quality() {
    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dollhouse_clean.jpg");
    let original = image::open(img_path)
        .expect("Failed to load test image")
        .to_rgb8();

    println!("Testing power-of-2 downsampling with linear color space averaging:");

    // Test divisors: 2, 4, 8
    for divisor in [2u32, 4, 8] {
        let downsampled = downsample_rgb_box(&original, divisor);

        let expected_width = (original.width() + divisor - 1) / divisor;
        let expected_height = (original.height() + divisor - 1) / divisor;

        assert_eq!(downsampled.width(), expected_width);
        assert_eq!(downsampled.height(), expected_height);

        println!(
            "  Divisor 1/{}: {}×{} → {}×{}",
            divisor,
            original.width(),
            original.height(),
            downsampled.width(),
            downsampled.height()
        );

        // Upsample back to original size for comparison
        let upsampled = upsample_nearest(&downsampled, original.width(), original.height());

        let original_linear = image_to_linear(&original);
        let upsampled_linear = image_to_linear(&upsampled);
        let psnr = compute_psnr(&original_linear, &upsampled_linear);

        println!("    PSNR after downsample+upsample: {:.2} dB", psnr);

        // Quality thresholds (realistic expectations after downsample+upsample)
        // When you throw away pixels, you lose information - that's expected
        let min_psnr = match divisor {
            2 => 13.0,  // 1/2 loses 75% of pixels
            4 => 11.0,  // 1/4 loses 93.75% of pixels
            8 => 9.0,   // 1/8 loses 98.4% of pixels
            _ => 8.0,
        };

        assert!(
            psnr > min_psnr,
            "Downsample by 1/{} quality too low. Got PSNR={:.2} dB, expected >{:.1} dB",
            divisor,
            psnr,
            min_psnr
        );

        // More importantly: box filter should not introduce artifacts
        // No banding, no strange patterns - just smooth blurring
        // This is qualitative but the PSNR should be consistent across divisors
    }
}

#[test]
fn test_box_filter_vs_nearest_neighbor() {
    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dollhouse_clean.jpg");
    let original = image::open(img_path)
        .expect("Failed to load test image")
        .to_rgb8();

    println!("Comparing box filter vs nearest-neighbor:");

    let divisor = 4u32;

    // Box filter (our implementation with linear averaging)
    let box_filtered = downsample_rgb_box(&original, divisor);

    // Nearest neighbor
    let target_width = (original.width() + divisor - 1) / divisor;
    let target_height = (original.height() + divisor - 1) / divisor;
    let nearest = downsample_rgb_nearest(&original, target_width, target_height);

    // Upsample both back to original size for comparison
    let box_upsampled = upsample_nearest(&box_filtered, original.width(), original.height());
    let nearest_upsampled = upsample_nearest(&nearest, original.width(), original.height());

    let original_linear = image_to_linear(&original);
    let box_linear = image_to_linear(&box_upsampled);
    let nearest_linear = image_to_linear(&nearest_upsampled);

    let psnr_box = compute_psnr(&original_linear, &box_linear);
    let psnr_nearest = compute_psnr(&original_linear, &nearest_linear);

    println!("  Box filter PSNR: {:.2} dB", psnr_box);
    println!("  Nearest neighbor PSNR: {:.2} dB", psnr_nearest);
    println!("  Improvement: {:.2} dB", psnr_box - psnr_nearest);

    // Box filter should be significantly better than nearest-neighbor
    assert!(
        psnr_box > psnr_nearest,
        "Box filter should produce better quality than nearest-neighbor. \
         Box={:.2} dB, Nearest={:.2} dB",
        psnr_box,
        psnr_nearest
    );

    // Box filter should be at least 2 dB better (noticeable difference)
    assert!(
        psnr_box - psnr_nearest > 2.0,
        "Box filter improvement should be significant (>2 dB). Got {:.2} dB improvement",
        psnr_box - psnr_nearest
    );
}

#[test]
fn test_linear_vs_srgb_averaging() {
    // This test verifies that linear color space averaging produces better results
    // than sRGB averaging for bright highlights (like the chandelier lights)

    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dollhouse_clean.jpg");
    let original = image::open(img_path)
        .expect("Failed to load test image")
        .to_rgb8();

    println!("Testing linear vs sRGB color space averaging:");

    let divisor = 8u32;

    // Our box filter (uses linear averaging)
    let linear_result = downsample_rgb_box(&original, divisor);

    // sRGB averaging (incorrect but for comparison)
    let srgb_result = downsample_rgb_box_srgb(&original, divisor);

    // Upsample both
    let linear_upsampled = upsample_nearest(&linear_result, original.width(), original.height());
    let srgb_upsampled = upsample_nearest(&srgb_result, original.width(), original.height());

    let original_linear = image_to_linear(&original);
    let linear_avg_linear = image_to_linear(&linear_upsampled);
    let srgb_avg_linear = image_to_linear(&srgb_upsampled);

    let psnr_linear = compute_psnr(&original_linear, &linear_avg_linear);
    let psnr_srgb = compute_psnr(&original_linear, &srgb_avg_linear);

    println!("  Linear averaging PSNR: {:.2} dB", psnr_linear);
    println!("  sRGB averaging PSNR: {:.2} dB", psnr_srgb);
    println!("  Improvement: {:.2} dB", psnr_linear - psnr_srgb);

    // Linear averaging should be better
    assert!(
        psnr_linear > psnr_srgb,
        "Linear color space averaging should produce better quality. \
         Linear={:.2} dB, sRGB={:.2} dB",
        psnr_linear,
        psnr_srgb
    );
}

// Helper: Upsample using nearest-neighbor (for comparison purposes)
fn upsample_nearest(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    let mut out = RgbImage::new(width, height);
    let sx = img.width() as f32 / width as f32;
    let sy = img.height() as f32 / height as f32;

    for y in 0..height {
        for x in 0..width {
            let src_x = (x as f32 * sx).floor().clamp(0.0, (img.width() - 1) as f32) as u32;
            let src_y = (y as f32 * sy).floor().clamp(0.0, (img.height() - 1) as f32) as u32;
            let pixel = *img.get_pixel(src_x, src_y);
            out.put_pixel(x, y, pixel);
        }
    }
    out
}

// Helper: Box filter with sRGB averaging (incorrect, for comparison)
fn downsample_rgb_box_srgb(img: &RgbImage, divisor: u32) -> RgbImage {
    let out_width = (img.width() + divisor - 1) / divisor;
    let out_height = (img.height() + divisor - 1) / divisor;
    let mut out = RgbImage::new(out_width, out_height);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let src_x_start = out_x * divisor;
            let src_y_start = out_y * divisor;
            let src_x_end = (src_x_start + divisor).min(img.width());
            let src_y_end = (src_y_start + divisor).min(img.height());

            // Average in sRGB space (incorrect)
            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;
            let mut count = 0u32;

            for src_y in src_y_start..src_y_end {
                for src_x in src_x_start..src_x_end {
                    let pixel = img.get_pixel(src_x, src_y);
                    sum_r += pixel[0] as u32;
                    sum_g += pixel[1] as u32;
                    sum_b += pixel[2] as u32;
                    count += 1;
                }
            }

            let avg_r = ((sum_r + count / 2) / count) as u8;
            let avg_g = ((sum_g + count / 2) / count) as u8;
            let avg_b = ((sum_b + count / 2) / count) as u8;

            out.put_pixel(out_x, out_y, image::Rgb([avg_r, avg_g, avg_b]));
        }
    }

    out
}
