//! Test to verify image loading doesn't introduce artifacts

use image::RgbImage;

#[test]
#[ignore] // Run manually: cargo test --test test_image_loading -- --ignored --nocapture
fn test_jpg_to_png_roundtrip() {
    // This test checks if the image loading/saving pipeline introduces artifacts

    // Load one of the source images
    let src_path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/dollhouse_sm/input/IMG_0910.jpg");
    let src = image::open(src_path)
        .expect("Failed to load source image")
        .to_rgb8();

    println!("Loaded source: {}×{}", src.width(), src.height());

    // Clone it (what downsample_image_smart does with factor=1.0)
    let cloned = src.clone();

    // Save as PNG (what the trainer does)
    let temp_path = "/tmp/test_roundtrip.png";
    cloned.save(temp_path).expect("Failed to save");

    // Reload the PNG
    let reloaded = image::open(temp_path)
        .expect("Failed to reload")
        .to_rgb8();

    println!("Reloaded: {}×{}", reloaded.width(), reloaded.height());

    // Compare pixel-by-pixel
    assert_eq!(src.width(), reloaded.width());
    assert_eq!(src.height(), reloaded.height());

    let src_pixels = src.as_raw();
    let reloaded_pixels = reloaded.as_raw();

    if src_pixels == reloaded_pixels {
        println!("✅ JPG->RgbImage->clone()->PNG->RgbImage roundtrip is IDENTICAL");
    } else {
        let mut diff_count = 0;
        let mut max_diff = 0u8;
        let mut first_diffs = Vec::new();

        for (i, (a, b)) in src_pixels.iter().zip(reloaded_pixels.iter()).enumerate() {
            if a != b {
                diff_count += 1;
                max_diff = max_diff.max(a.abs_diff(*b));
                if first_diffs.len() < 10 {
                    first_diffs.push((i, *a, *b, a.abs_diff(*b)));
                }
            }
        }

        println!("❌ Images differ!");
        println!("Total differences: {} / {} bytes ({:.2}%)",
            diff_count, src_pixels.len(),
            100.0 * diff_count as f64 / src_pixels.len() as f64);
        println!("Max pixel difference: {}", max_diff);
        println!("\nFirst 10 differences:");
        for (i, a, b, diff) in &first_diffs {
            println!("  Byte {}: {} vs {} (diff={})", i, a, b, diff);
        }

        panic!("Image roundtrip introduced {} differences!", diff_count);
    }
}

#[test]
#[ignore]
fn compare_saved_target_with_source() {
    // Compare the saved target from actual training run with source

    let src_path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/dollhouse_sm/input/IMG_0910.jpg");
    let saved_path = concat!(env!("CARGO_MANIFEST_DIR"), "/runs/20251219_1939_micro/m8_test_view_target.png");

    let src = match image::open(src_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            println!("Skipping test - source image not found: {}", e);
            return;
        }
    };

    let saved = match image::open(saved_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            println!("Skipping test - saved image not found: {}", e);
            return;
        }
    };

    println!("Source: {}×{}", src.width(), src.height());
    println!("Saved:  {}×{}", saved.width(), saved.height());

    let src_pixels = src.as_raw();
    let saved_pixels = saved.as_raw();

    if src_pixels == saved_pixels {
        println!("✅ Saved target is IDENTICAL to source");
    } else {
        let mut diff_count = 0;
        let mut max_diff = 0u8;

        for (a, b) in src_pixels.iter().zip(saved_pixels.iter()) {
            if a != b {
                diff_count += 1;
                max_diff = max_diff.max(a.abs_diff(*b));
            }
        }

        println!("❌ Saved target differs from source!");
        println!("Differences: {} / {} bytes ({:.2}%)",
            diff_count, src_pixels.len(),
            100.0 * diff_count as f64 / src_pixels.len() as f64);
        println!("Max pixel difference: {}", max_diff);
    }
}
