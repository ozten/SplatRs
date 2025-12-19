//! Test different PNG encoding methods to find one without artifacts

use image::ImageEncoder;

#[test]
#[ignore] // Run manually: cargo test --test test_png_encoding -- --ignored --nocapture
fn test_png_encoding_methods() {
    use std::io::BufWriter;
    use std::fs::File;

    let src_path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/dollhouse_sm/input/IMG_0910.jpg");
    let src = image::open(src_path)
        .expect("Failed to load source")
        .to_rgb8();

    println!("Source: {}×{}", src.width(), src.height());

    // Method 1: Default save (what trainer currently does)
    let default_path = "/tmp/sugar_test_default.png";
    src.save(default_path).unwrap();
    println!("Saved {} (default .save())", default_path);

    // Method 2: Explicit PNG encoder
    let explicit_path = "/tmp/sugar_test_explicit.png";
    let file = File::create(explicit_path).unwrap();
    let writer = BufWriter::new(file);
    let encoder = image::codecs::png::PngEncoder::new(writer);
    encoder.write_image(
        src.as_raw(),
        src.width(),
        src.height(),
        image::ExtendedColorType::Rgb8,
    ).unwrap();
    println!("Saved {} (explicit PNG encoder)", explicit_path);

    // Method 3: With compression level
    use image::codecs::png::{PngEncoder, CompressionType, FilterType};
    let fast_path = "/tmp/sugar_test_fast.png";
    let file = File::create(fast_path).unwrap();
    let writer = BufWriter::new(file);
    let mut encoder = PngEncoder::new_with_quality(
        writer,
        CompressionType::Fast,
        FilterType::NoFilter,
    );
    encoder.write_image(
        src.as_raw(),
        src.width(),
        src.height(),
        image::ExtendedColorType::Rgb8,
    ).unwrap();
    println!("Saved {} (fast compression, no filter)", fast_path);

    println!("\nNow visually compare:");
    println!("  Source JPG:  {}", src_path);
    println!("  Default PNG: {}", default_path);
    println!("  Explicit PNG: {}", explicit_path);
    println!("  Fast PNG:    {}", fast_path);
}
