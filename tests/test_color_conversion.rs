//! Test color profile conversion

#[test]
#[ignore]
fn test_p3_conversion_changes_pixels() {
    use sugar_rs::io::load_image_to_srgb;

    // Load with color conversion
    let converted = load_image_to_srgb(
        std::path::Path::new("datasets/dollhouse_sm/input/IMG_0910.jpg")
    ).unwrap();

    // Load without color conversion (what image crate does)
    let direct = image::open("datasets/dollhouse_sm/input/IMG_0910.jpg")
        .unwrap()
        .to_rgb8();

    // Compare pixels
    let mut diff_count = 0;
    let mut max_diff = 0u8;
    let mut total_diff = 0u64;

    for (i, (c, d)) in converted.as_raw().iter().zip(direct.as_raw().iter()).enumerate() {
        if c != d {
            diff_count += 1;
            let diff = c.abs_diff(*d);
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
        }
    }

    let num_pixels = converted.as_raw().len();
    let diff_pct = 100.0 * diff_count as f64 / num_pixels as f64;
    let avg_diff = if diff_count > 0 { total_diff as f64 / diff_count as f64 } else { 0.0 };

    println!("Pixels changed: {} / {} ({:.2}%)", diff_count, num_pixels, diff_pct);
    println!("Average difference: {:.2}", avg_diff);
    println!("Max difference: {}", max_diff);

    // Color conversion should change pixels
    assert!(diff_count > 0, "Color conversion should change some pixels!");

    // Save both for visual comparison
    converted.save("/tmp/converted_p3_to_srgb.png").unwrap();
    direct.save("/tmp/direct_no_conversion.png").unwrap();

    println!("\nSaved:");
    println!("  /tmp/converted_p3_to_srgb.png (with color conversion)");
    println!("  /tmp/direct_no_conversion.png (without color conversion)");
}
