#!/usr/bin/env rust-script
//! Test if loading and saving introduces artifacts
use image::GenericImageView;

fn main() {
    // Load one of the dataset images
    let img = image::open("datasets/dollhouse_sm/input/IMG_0909.jpg")
        .expect("Failed to load")
        .to_rgb8();

    println!("Loaded: {}×{}", img.width(), img.height());

    // Save as PNG (what the trainer does)
    img.save("test_roundtrip.png").expect("Failed to save");
    println!("Saved to test_roundtrip.png");

    // Now load the PNG and compare
    let reloaded = image::open("test_roundtrip.png")
        .expect("Failed to reload")
        .to_rgb8();

    // Check if identical
    let identical = img.as_raw() == reloaded.as_raw();
    println!("Images identical after round-trip: {}", identical);

    if !identical {
        println!("WARNING: Round-trip introduced differences!");

        // Find first difference
        for (i, (a, b)) in img.as_raw().iter().zip(reloaded.as_raw().iter()).enumerate() {
            if a != b {
                println!("First difference at byte {}: {} -> {}", i, a, b);
                break;
            }
        }
    }
}
